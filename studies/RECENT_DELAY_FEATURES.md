# Recent-delay feature research

This independent branch preserves the original three-hour airport histories,
LR study, availability audit and offline scoring delivery. It does not alter
production source, default model, thresholds, splits or root README. Git dates
are a disclosed retrospective reconstruction; evidence retains original dates.

## Historical result and qualification

The saved research comparison reports equal-month May–August 2024 PR-AUC of
**0.46373486139457926**, against **0.4280983226531411** for the accepted LR,
an absolute improvement of **0.035636538741438184**. The selected threshold
is **0.39**. These are historical measurements, not newly fitted results from
reconstruction. Equal-month PR-AUC is distinct from pooled production PR-AUC.

The candidate met its historical research comparison, but operational availability
remains **BLOCKED**. Reconstructed actual arrival with zero reporting latency does
not establish when an outcome was received. Schedule versions and correction
availability are also unproven. Neither the research gain nor offline score parity
resolves these missing as-of guarantees. No production promotion or merge occurs.

## Implementation and evidence

- [Three-hour histories](track_a_3h_v1/REPORT.md): origin and destination rates,
  counts and missing-history flags. Eligible history must fall before each
  scheduled-departure cutoff; current, future and in-flight outcomes are excluded.
- [LR experiment](lr_3h_v2/REPORT.md): frozen 784-coordinate preprocessing prefix
  plus six appended coordinates, train-only imputation and one classifier fit.
- [Availability audit](lr_3h_availability_audit_v1/REPORT.md) and
  [promotion review](lr_3h_promotion_review_v1/REPORT.md): performance and deployment
  readiness are separate decisions.
- [Offline batch scoring](OFFLINE_LR3H.md): precomputed compatible rows only,
  immutable hash-checked bundle, no feature construction or model fitting.

## Checks executed during reconstruction

Nine focused unittest checks passed for time-window boundaries, comparison
protocol and absence of fitting in the offline scorer. The existing Spark
`checks.generated` fixture also passed exact-boundary, duplicate, null,
future/current/in-flight exclusion, forbidden-label perturbation and ordering
checks in the pinned runtime. These small fixtures do not repeat the historical
full-population feature build or candidate training.

```bash
python3 -m unittest studies.ml_three_track_v1.test_track_a \
  studies.lr_3h_v2.test_protocol studies.lr_3h_offline_v1.test_no_fits -v
```

A new [16-row offline smoke](lr_3h_offline_finalize_v1/evidence/reconstruction-smoke.json)
also passed from the reconstructed branch in an isolated two-worker cluster.
It reloaded the immutable bundle, verified 790 coordinates and threshold 0.39,
scored unlabeled precomputed rows, matched saved scores/predictions exactly,
and performed zero classifier fits or preprocessing refits. Cleanup succeeded.
This small parity check is not new full-data performance evidence.

Full command records and logs live in the external reconstruction report.
The offline bundle is independently copied locally under ignored
`artifacts/offline/offline-lr3h-ca23fe0-v1`; binary models and precomputed rows
are not committed. A clone alone cannot score without those artifacts.

## Portability and replay boundaries

The shared `lr_v1/diagnose.py` path initialization now occurs inside `main()` so
imports work in the reconstructed layout; metric mathematics is unchanged.
The feature and LR controllers inventory current `HEAD` rather than two original
short Git IDs unavailable in this independent repository. Saved evidence remains
unaltered and continues to refer to the original source commits.

Historical full-data controllers still require their documented frozen artifact
identities and study layouts. They are not silently redirected at the new
production model. Restore those prerequisites to replay. The historical source
`ml_three_track_v1/track_a.py` is retained solely for the executable reference
used by the feature checks; this does not create a third research branch.

Offline scoring preserves the 790-coordinate contract and applies probability
>=0.39 explicitly. It accepts precomputed research rows and optional labels;
it does not establish live receipt-aware correctness or an online release path.
