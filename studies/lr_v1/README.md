# LR improvement study

Start with the [completed candidate results](CANDIDATE_RESULTS.md). Both approved
one-factor experiments completed; neither met the preregistered improvement
criterion. The accepted production LR at threshold 0.39 remains unchanged.

- [Full-precision candidate comparison](candidate-results-summary.json)
- [User-approved candidate settings and limits](approved-candidates.json)
- [Original measurement protocol](protocol.json)
- [Earlier diagnostic report](REPORT.md), written before candidate approval
- [Original candidate proposal](proposed-candidates.json), retained as history
- [Baseline diagnostic chart](diagnostics.png)
- [Frozen baseline diagnostics](evidence/diagnostics/results.json)
- [Candidate execution results](evidence/candidate-results.json)
- [Compute and command ledger](evidence/budget.json)
- [Preservation before](evidence/candidate-preservation-before-candidates.json)
  and [after](evidence/candidate-preservation-after-candidates.json)

The measurement code uses the four May-August 2024 validation months equally.
September-December 2024 was not used to select candidates. Findings remain
retrospective comparisons on development data; no later untouched evaluation
period is available.

## Review without raw artifacts

From the repository root, check the frozen-input hashes and recompute the paired
comparisons using Python's standard library:

```bash
python3 -m studies.lr_v1.review_results
```

The existing synthetic tests require pytest and do not fit models or start Spark:

```bash
PYTHONPATH=.:src python3 -m pytest -q studies/lr_v1/test_metrics.py studies/lr_v1/test_candidates.py
```

The originally executed Python source is preserved byte-for-byte. Only
`review_results.py` is new. The original study ran in an isolated worktree at
`data/studies/lr-improvement-v1/project/`, with generated outputs alongside it.
The fitting, diagnostics, plotting, preservation, reporting, and full-ZIP
packaging tools retain that original layout and its strict historical identity
checks; they are historical execution source, not fresh-clone entry points at
this exported location. Their original commands and commits are in the ledger.
Do not rerun completed candidate attempts or reset their output guards.

The evidence directory contains aggregate JSON, parameter/qualification evidence,
preservation records, and tiny synthetic-test summaries. Fitted model binaries,
prediction Parquet, runtime logs, raw data, and delivery ZIPs are excluded.
The review manifest (historical original-repository file `../../results/review-manifest.json`; omitted from this compact reconstruction) records original paths
and source/export hashes. The review guide (historical original-repository file `../../docs/REVIEW_GUIDE.md`; omitted from this compact reconstruction) covers
production reproduction and the distinction between original execution records
and this publication.
