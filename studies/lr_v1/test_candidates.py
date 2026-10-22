import copy
import json
from pathlib import Path

import pytest

from studies.lr_v1.candidate_spec import (validate_approval, exclusive_candidate_path,
                                          candidate_time_limit, NAMES)


def approval():
    return json.loads(Path(__file__).with_name('approved-candidates.json').read_text())


def test_exactly_the_two_approved_one_factor_candidates():
    assert tuple(c['name'] for c in validate_approval(approval())) == NAMES


@pytest.mark.parametrize('damage', ['combine', 'extra', 'reverse', 'other_setting', 'retry'])
def test_reject_unapproved_variations(damage):
    value = approval()
    if damage == 'combine':
        value['candidates'][0]['settings']['classifier_settings']['regParam'] = 0.3
    elif damage == 'extra':
        value['candidates'].append(copy.deepcopy(value['candidates'][0]))
    elif damage == 'reverse':
        value['candidates'].reverse()
    elif damage == 'other_setting':
        value['candidates'][1]['settings']['classifier_settings']['maxIter'] = 10
    else:
        value['automatic_retries'] = True
    with pytest.raises(ValueError):
        validate_approval(value)


def test_output_is_exclusive_and_cannot_follow_a_symlink(tmp_path):
    path = exclusive_candidate_path(tmp_path, NAMES[0])
    assert path.parent == tmp_path / 'candidates'
    path.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        exclusive_candidate_path(tmp_path, NAMES[0])
    other = tmp_path / 'outside'
    other.mkdir()
    (tmp_path / 'candidates' / NAMES[1]).symlink_to(other)
    with pytest.raises(ValueError):
        exclusive_candidate_path(tmp_path, NAMES[1])
    with pytest.raises(ValueError):
        exclusive_candidate_path(tmp_path, '../models')


def test_caps_never_expand_remaining_budget():
    assert candidate_time_limit(13906.805741877004) == (3600, 3590)
    assert candidate_time_limit(55.8) == (55, 45)
    with pytest.raises(ValueError):
        candidate_time_limit(29)
