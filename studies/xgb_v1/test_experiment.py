"""The gate must include even total swap, without mistaking heap for host RAM."""
import json
import pytest
from studies.xgb_v1.experiment import BASELINE, memory_bound
from studies.xgb_v1.run import verify_distributed_fit
from studies.lr_v1.diagnose import budget_metrics  # Verify legacy utility imports in the mounted layout.


def test_full_population_cannot_fit_even_with_host_swap():
    result = memory_bound(8946370, 784, 12542668800, 4294967296)
    assert result['dense_bytes'] == 28055816320
    assert result['dense_bytes_per_worker'] == 14027908160
    assert result['blocked']
    assert not memory_bound(256, 784, 12542668800, 0)['blocked']


def test_baseline_uses_full_evidence_with_integer_confusion_counts():
    months = json.loads(BASELINE.read_text())['validation_months']
    assert set(months) == {'2024-05', '2024-06', '2024-07', '2024-08'}
    for m in months.values():
        assert sum(m['confusion_matrix'].values()) == m['rows']
        assert all(type(n) is int for n in m['confusion_matrix'].values())
    assert sum(m['pr_auc'] for m in months.values()) / 4 == 0.4280983226531411


def test_distributed_fit_does_not_require_executor_stderr_in_driver_log():
    verify_distributed_fit('Running xgboost-3.0.5 on 2 workers\nFinished xgboost training!',
                           {'workers': [{'host': 'a'}, {'host': 'b'}]},
                           {'paramMap': {'num_workers': 2}})


@pytest.mark.parametrize('log,hosts,workers', [
    ('Running xgboost-3.0.5 on 2 workers', ['a', 'b'], 2),
    ('Running xgboost-3.0.5 on 2 workers\nFinished xgboost training!', ['a', 'a'], 2),
    ('Running xgboost-3.0.5 on 2 workers\nFinished xgboost training!', ['a', 'b'], 1),
])
def test_distributed_fit_rejects_incomplete_evidence(log, hosts, workers):
    with pytest.raises(ValueError, match='two-worker'):
        verify_distributed_fit(log, {'workers': [{'host': h} for h in hosts]},
                               {'paramMap': {'num_workers': workers}})
