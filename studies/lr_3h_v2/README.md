# Controlled full-data LR + six qualified 3h features, v2

Reference `88b632f`; prior blocked attempt preserved in `studies/lr_3h_v1`.
The v2 study uses a new branch and exclusive output directory. It corrects only
the helper mount depth and adds actual-runtime focused preflight tests. The
existing host repository, Docker daemon, immutable Spark image and data are used.
No environment installation, rebuild or resource changes are performed.

Inputs are the 80 SHA-256 pinned saved feature files and frozen accepted models,
selection data and baseline evidence. No DWD or test-period dataset is mounted.
The accepted classifier is loaded solely to verify its contract; never scored.
All twenty months must pass exact original-column/label multiset equality,
feature-domain/schema checks and the required split totals before fitting.

Run focused tests with the pinned compose.preflight.yml cluster, using:
`/opt/spark/bin/spark-submit --driver-memory 768m /study/preflight.py`.
These test imports, frozen preprocessing on sixteen saved training rows, forced
null-rate imputation, original 784-coordinate prefix, fixed six-coordinate order,
preprocessing save/reload, and distributed duplicate budget handling, with zero
classifier fits. The implementation must be committed after passing these tests.

Execute exactly once: `python3 studies/lr_3h_v2/run.py`.
The exact supervised command is:

```bash
docker compose -f studies/lr_3h_v2/compose.yml exec -T driver \
  timeout --signal=TERM --kill-after=15 14100 \
  /opt/spark/bin/spark-submit --driver-memory 768m /study/experiment.py
```

The supervisor stops work by 14,220 seconds and reserves cleanup within 14,400.
The exclusive runtime directory prevents retries. Immutable image:
`sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`.
Spark 3.5.9/Python 3.10.12; two one-core workers; 640 MiB executor heaps;
768 MiB driver; unchanged 2 GiB worker/driver and 512 MiB master container limits;
four shuffle partitions, 64 MiB file partitions, no task/stage retries.

Only train estimates the two new means. Frozen preprocessing is never refit.
All prepared rows verify exact 784-prefix and 790 total width. The original LR
settings and 2:1 weights are retained; Spark LR exposes no random seed parameter.
The frozen contract's reproducibility seed remains 42, with no randomized step
introduced. Candidate model save/reload must preserve all validation scores.
One pooled May–August threshold is selected from the existing 101-step grid using
exact F1 and lower-step tie break. Each month's unweighted exact PR-AUC and exact
10%/20% budget metrics reuse existing helpers, including fractional duplicate ties.
Compare only to frozen evidence: mean paired PR-AUC gain >=0.005 and mean recall
changes >=-0.01 at both budgets. Complete artifacts and successful shutdown are
also required. Production LR remains at 0.39; no promotion or merge occurs.

Resource summaries run offline: `python3 studies/lr_3h_v2/summarize.py`.
Model/prediction Parquet and full Spark events remain in ignored runtime/output.
Stop after this result, whether qualified, failed or blocked.

Completed result: [PASS report](REPORT.md). The single fit and all validation
checks completed; do not rerun or start another experiment under this authorization.
