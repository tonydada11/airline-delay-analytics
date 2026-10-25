# Model comparison research

This branch preserves the original LR, GBT and XGBoost studies independently of
production. Results below are **saved historical results**, not fresh full-data
fits from reconstruction. Production source, defaults and root README are unchanged.
The reconstructed Git dates organize the narrative; original execution timestamps
remain in the evidence files.

| Candidate | Equal-month validation PR-AUC | Decision |
|---|---:|---|
| Accepted LR reference | 0.4280983226531411 | Retain reference |
| LR positive class weight 1 | 0.4277249363114941 | Not qualified |
| LR regularization 0.3 | 0.4237132163107035 | Not qualified |
| GBT, 60 iterations | 0.4274314742703018 | Not qualified |
| Spark XGBoost | No full-data result | Memory feasibility blocked |

These means give each of May–August 2024 equal weight. They differ from the
pooled validation PR-AUC reported by the production evaluation. The completed
60-iteration GBT has a -0.0006668483828393 change against the LR reference;
completion is not qualification. No model from this branch is promoted.

## Sources and evidence

- [LR candidate implementation and results](lr_v1/CANDIDATE_RESULTS.md): two fixed
  changes using accepted train-only preprocessing, with preserved monthly metrics.
- [GBT 60-iteration result](gbt_v1/RESULTS_60ITER.md): the same bounded runner also
  preserves the 10- and 30-iteration comparisons and the no-fit prefix diagnostic.
- [XGBoost compatibility study](xgb_feasibility_v1/REPORT.md): historical bounded
  generated-fixture fits, sparse/dense semantics, runtime and memory measurements.
- [Latest XGBoost feasibility decision](xgb_v1/RESULTS.md): zero full-data fits;
  the dense float32 lower bound alone exceeds available host memory.

## Checks executed during reconstruction

The pinned Python 3.10 / Spark 3.5.9 image executed 97 model-comparison tests:
94 passed in the combined invocation; its one missing-input failure was rerun
successfully after an independent copy of the saved 30-tree model was supplied.
Two additional offline XGBoost supervisor regression checks passed.
The initial collection failure exposed a legacy import-time path assumption.
Moving `diagnose.py`'s three path initializations inside `main()` fixed imports
without changing metric calculations. Full logs, commands and exit codes are in
the external reconstruction report. These tests are fixtures and reload checks,
not new full-population candidate evaluations or XGBoost distributed fits.

From the pinned runtime, with repository root and `src` on `PYTHONPATH`:

```bash
python3 -m pytest -q studies/lr_v1/test_metrics.py studies/lr_v1/test_candidates.py \
  studies/gbt_v1/test_candidate.py studies/gbt_v1/test_prefix_diagnostic.py \
  studies/xgb_feasibility_v1/test_supervisor.py studies/xgb_v1/test_experiment.py
```

The complete 97-test suite was also rerun successfully from the final branch layout.

The prefix reload test requires the ignored saved model at
`data/studies/gbt-improvement-v1/attempt-002/gbt-all-30iter/classifier`.
Model binaries are deliberately excluded from Git; a clone alone lacks them.

## Replaying historical experiments

Read each study's plan before running its controller. The original full-fit
controllers deliberately bind the accepted historical bundle
`d0098a9572834c53bf76b04d732a94a6`, dataset and source hashes, and in some cases
nested `data/studies/.../project` layouts. A fresh production pipeline generates
new bundle identities and is not automatically a valid replacement. Restore the
specified frozen inputs and isolated runtime before replay; do not remove identity
checks or relabel new fits as the original study. The independent reconstruction
contains no original Git objects. Source references in saved evidence name the
original repository, not commits in this new history.

Transient logs, redundant delivery archives and superseded failure snapshots are
excluded. Compact results, provenance, bounded controllers and focused tests are
retained. Production dependencies remain pinned; XGBoost has its own isolated
requirements and runtime. Neither research branch is merged.
