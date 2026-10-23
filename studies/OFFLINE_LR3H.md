# Offline LR-3h batch scoring

This is explicitly opt-in research scoring of **precomputed** flight features.
Production schema-3 LR, its defaults and its 784-coordinate bundle remain
unchanged. No features are built, and no models or preprocessing are fitted.

The immutable local bundle is
`artifacts/offline/offline-lr3h-ca23fe0-v1/`. Its model ID is
`offline-lr3h-ca23fe0-v1`; its manifest SHA-256 is
`9157123a82926767a0ac69b6c1f9ab04899d0ad86ec32c95b42ffe46809a564f`.
This is a byte-preserved copy of the bundle verified at `93f0c58`. The bundle
contains its loader/scorer, frozen classifier/preprocessing, exact means,
feature contract and experiment evidence; it needs no original study mounts.

Binary artifacts are local and ignored by Git. A clone alone does not contain
the model. Transfer the **entire directory**, including hidden CRC files,
with file modes preserved; compare the manifest hash and complete inventory in
[finalization evidence](lr_3h_offline_finalize_v1/evidence/artifact.json)
before use. Preserve the original directory and prior scored outputs. Do not
edit or regenerate files inside this model identity. The loader checks all
bundle file hashes on every load.

Run the following from the repository root on the existing pinned Docker host,
with other Spark workloads stopped. Set `OFFLINE_FEATURES` to an absolute
Parquet directory containing the approved original predictors, six 3h columns
and business keys. Set `OFFLINE_OUTPUT` to a **new** versioned run directory.
The example input is the existing qualified May 2024 feature partition; choose
another compatible precomputed directory as needed. The output path must be
new; the scorer refuses to overwrite `scores/`.

```bash
export OFFLINE_FEATURES="$PWD/studies/track_a_3h_v1/runtime/output/features/validation-2024-05"
export OFFLINE_OUTPUT="$PWD/artifacts/offline-runs/lr3h-user-run-001"
mkdir -p "$(dirname "$OFFLINE_OUTPUT")"
mkdir "$OFFLINE_OUTPUT"
(
  set -eu
  trap 'docker compose -f conf/docker-compose.offline-lr3h.yml down --timeout 15' EXIT
  docker compose -f conf/docker-compose.offline-lr3h.yml up -d
  docker compose -f conf/docker-compose.offline-lr3h.yml exec -T driver \
    timeout --signal=TERM --kill-after=15 14100 \
    /opt/spark/bin/spark-submit --driver-memory 768m /bundle/offline.py \
    --bundle /bundle --input /input --output /output/scores
)
```

The configuration mounts the stable bundle and input read-only, and only the
new run directory writable. It retains the verified image, Spark 3.5.9 /
Python 3.10.12, two one-core workers and existing memory/shuffle limits. It
changes no production configuration or routing.

The existing scorer preserves the fitted 784-coordinate prefix, imputes only
`origin_delay_rate_3h` with `0.08079999358628864` and
`destination_delay_rate_3h` with `0.0796816299126744`, then appends, in order:

1. `origin_delay_rate_3h`
2. `origin_flight_count_3h`
3. `origin_history_missing_3h`
4. `destination_delay_rate_3h`
5. `destination_flight_count_3h`
6. `destination_history_missing_3h`

Final width is 790. The scorer uses positive-class probability and explicitly
sets `prediction = probability >= 0.39`, discarding the classifier's internal
0.5 prediction. Labels and split are optional. Output preserves business keys,
optional flight_id/label/split and multiplicity, and includes probability,
prediction, model ID and feature policy. `scores/_completion.json` records
input/output hashes, model manifest hash, row count and threshold.

The final-location smoke test uses 16 original feature rows and the unchanged
scorer's assembly/scoring functions, checks saved score/prediction parity and
unlabeled input, and does not repeat full validation. See the
[finalization report](lr_3h_offline_finalize_v1/REPORT.md) for its exact
command and results; prior full parity is frozen in the
[original report](lr_3h_offline_v1/REPORT.md).

This retains the research zero-reporting-latency policy. Reconstructed arrival
is not measured outcome receipt; schedule versions and correction availability
are unproven. This is neither live availability qualification nor independent
performance evidence. No production activation or main merge is authorized.
