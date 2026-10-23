# Offline LR-3h bundle and batch scoring

For finalized offline use, see [the stable bundle and opt-in command](../OFFLINE_LR3H.md).
The bundle now also lives at `artifacts/offline/offline-lr3h-ca23fe0-v1/`
with unchanged contents and hashes. The full verification commands below are
historical evidence; finalization requires only the documented small smoke test.

Opt-in research-only packaging of candidate `ca23fe0`. Schema-3 production is
unchanged. No receipt-aware or live prediction claim is made: input histories
retain the original scheduled-departure UTC / reconstructed arrival / zero
reporting latency assumption. Inputs must already contain the approved original
50 predictors, six 3h columns and six business keys. No features are constructed.
Target label and split are optional; they are carried through when supplied.

The self-contained bundle contains byte-copied fitted classifier/preprocessing,
exact saved means, offline code, contract, source inventories, frozen results and
threshold evidence. Loader uses only stdlib/PySpark; no schema-3 validator or
study path is needed. Every file is SHA-256 checked. Fixed means/order and model
width/settings are checked; explicit probability >=0.39 overrides the saved
classifier's internal default 0.5 decision. No fitting APIs are called.

Focused tests: `python3 -m unittest studies.lr_3h_offline_v1.test_no_fits -v` plus
`docker compose -f studies/lr_3h_offline_v1/compose.focused.yml exec -T driver
/opt/spark/bin/spark-submit --driver-memory 768m /study/verify.py --mode focused`
(on its isolated pinned cluster, followed by `down --timeout 15`).

After committing the tested implementation, execute the exclusive verification:

```bash
python3 studies/lr_3h_offline_v1/run.py
```

This sequentially packages/stages and checks all validation vectors/scores;
starts a fresh isolated cluster mounting only the copied bundle and precomputed
validation inputs; runs the documented standalone CLI; then verifies the fresh
output against the saved candidate and publishes the immutable final bundle.
The completion manifest is written only after staged artifact/load/full saved
score verification; final bundle directory is published only after fresh-process
score parity succeeds. New outputs never overwrite existing paths. No model is
trained or threshold selected. The controller retains existing two-core Spark
settings and container limits, does not overlap clusters, caps work/cleanup at
four hours, and records memory, exact commands, cleanup and preservation.

Standalone command (inside an identical pinned cluster with `/bundle` and
precomputed rows mounted read-only, a separate writable `/output`, and no study
or production mounts):

```bash
/opt/spark/bin/spark-submit --driver-memory 768m /bundle/offline.py \
  --bundle /bundle \
  --input /input/2024-05 /input/2024-06 /input/2024-07 /input/2024-08 \
  --output /output/scored-offline-v1
```

See `compose.fresh.yml` for the precise verification mounts. After publication,
mount `runtime/output/bundle` at `/bundle` (the verification used its staged
name). Input/output paths are explicit and the output must not exist. Outputs
include original business keys, optional flight_id/label/split, probability,
explicit prediction, model_id and feature_policy; completion binds model/input
and output hashes. The contract intentionally supports precomputed research
rows only. It does not fetch outcomes or validate live as-of receipt.

Expected local artifacts: `runtime/output/bundle/`,
`runtime/output/scored-offline-v1/`; large binaries remain ignored. The bundle's
relative inventory survives relocation. Transfer the entire directory and verify
its manifest hash against the report. Preserve the accepted production bundle
`d0098a9572834c53bf76b04d732a94a6` and its default 0.39 route.

Stop after the result. No production switch, promotion, fitting or feature build
is authorized by this offline verification.

Completed verification, exact commands, identities, resources and retained correction history: [REPORT.md](REPORT.md).
