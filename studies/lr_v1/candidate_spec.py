"""The two approved settings and study-only output boundaries."""
from pathlib import Path

NAMES = ('lr-positive-weight-1', 'lr-reg-0p3')


def validate_approval(approval):
    if tuple(c['name'] for c in approval['candidates']) != NAMES:
        raise ValueError('Only the two approved candidates may run, in order')
    base = approval['baseline_settings']
    expected_changes = ('positive_class_weight', 'classifier_settings.regParam')
    for index, candidate in enumerate(approval['candidates']):
        settings = candidate['settings']
        differences = []
        if set(settings) != set(base) or set(settings['classifier_settings']) != set(base['classifier_settings']):
            raise ValueError('Estimator setting keys changed')
        for key in base:
            if isinstance(base[key], dict):
                differences.extend(key + '.' + k for k in base[key] if settings[key][k] != base[key][k])
            elif settings[key] != base[key]:
                differences.append(key)
        if differences != [expected_changes[index]]:
            raise ValueError('A candidate must change exactly its approved factor')
    a, b = approval['candidates']
    if (a['settings']['positive_class_weight'] != 1.0 or a['settings']['classifier_settings']['regParam'] != 0.1
            or b['settings']['positive_class_weight'] != 2.0 or b['settings']['classifier_settings']['regParam'] != 0.3):
        raise ValueError('Candidate values differ from the user approval')
    if approval['automatic_retries'] or approval['maximum_seconds_per_candidate_including_evaluation'] != 3600:
        raise ValueError('Retry/time policy differs from approval')
    return approval['candidates']


def exclusive_candidate_path(study_root, name):
    if name not in NAMES:
        raise ValueError('Unapproved candidate')
    parent = Path(study_root).resolve() / 'candidates'
    path = parent / name
    if parent.is_symlink() or path.is_symlink() or path.resolve().parent != parent:
        raise ValueError('Candidate outputs may not escape the study directory')
    if path.exists():
        raise FileExistsError('An existing attempt must be preserved; retries are not approved')
    return path


def candidate_time_limit(remaining):
    # Reserve ten seconds inside the 60-minute limit for process termination.
    hard_limit = min(3600, int(remaining))
    if hard_limit < 30:
        raise ValueError('Insufficient remaining budget to start a candidate')
    return hard_limit, hard_limit - 10
