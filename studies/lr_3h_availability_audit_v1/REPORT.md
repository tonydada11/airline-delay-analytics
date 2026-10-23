# Phase 1 — operational data availability: BLOCKED

References: candidate **`ca23fe0`**, promotion review **`c196317`**. **Available local evidence does not establish strict operational as-of correctness.** The LR-3h research validation PASS remains valid under its declared assumption; it is not a proven live availability policy. **Mandatory mid-review stop: Phases 2–5 have not started.**

## Findings

| Required evidence | Finding | Decision |
|---|---|---|
| Actual outcome receipt/ingestion before prediction | `_ingested_at` exists, but the ODS builder assigns `F.current_timestamp()` during bulk processing. It is not provider publication, first receipt or first usable outcome time. No per-outcome receipt/version log was found in the inspected configured sources. | BLOCKED |
| Schedule known at prediction time | Raw `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, airports and carrier are retrospective snapshot values. No observed-at timestamp, predeparture schedule snapshots or effective-version intervals were found. | BLOCKED |
| Corrections and record versions | One pinned monthly file per month; ODS/DWD overwrite datasets. No event sequence, supersession chain or bitemporal record history establishes when corrected labels, statuses or schedules became usable. | BLOCKED |
| 7/30/90d and origin/destination 3h coverage | Monthly historical files exist, but no receipt-qualified coverage/completeness evidence exists at prediction cutoffs. The source starts January 2023, leaving initial lookback warm-up truncated, and ends December 2024 rather than providing current operational history. | BLOCKED |
| Actual prediction cutoff | The research contract uses scheduled-departure UTC. No inspected request/decision log establishes actual issue time or links a prediction to schedule/outcome versions known then. | BLOCKED |

No missing timestamp was interpreted as availability. No lag constant, zero-latency SLA, correction ordering or operational coverage percentage was inferred.

## Evidence inspected

**Source inventory and schema.** `conf/project.yaml` points to local BTS CSV → ODS → DWD snapshots. The configured download entry point is [`scripts/scripts_download_bts.sh`](../../scripts/scripts_download_bts.sh), using monthly BTS PREZIP files. Local raw storage has **24 CSVs and 24 ZIPs**, January 2023–December 2024; the DWD directory has the corresponding 24 year/month partitions. All 24 CSV headers agree: **109 named fields plus one empty trailing CSV artifact column**. They contain flight event clocks and schedule/outcome values, but no per-record receipt, publication, observed-at or revision timestamps. `DOT_ID_Reporting_Airline` identifies a carrier; it is not a reporting timestamp. Only headers were read from raw CSVs, not outcome rows.

**Stored ingestion metadata.** Read-only Parquet inspection found **97 ODS files**, one **111-field schema**, and **13,926,960 rows from footer counts**. The additional fields are `_source_file` and `_ingested_at`. A single source row, selected only from a row group whose date bounds lie within train/validation, contains:

- `FlightDate`: **2023-01-02**.
- `_source_file`: `On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2023_1.csv`.
- `_ingested_at`: decoded as **2026-10-20 17:18:40.619454**; this representation has no asserted timezone.

This proves a stored bulk-processing timestamp is not evidence of receipt before that flight. It does **not** measure a provider latency distribution. All 97 inspected row groups lack `_ingested_at` footer statistics; this one metadata-row sample is not an all-row timestamp census. The January 2023 DWD footer has **114 physical columns**, plus year/month partition keys; `_source_file` and `_ingested_at` are retained, with no additional receipt/version fields. See [Parquet evidence](evidence/parquet-metadata.json).

Code semantics independently establish the limitation: [`ods/build_flights.py:add_lineage_columns`](../../src/airline/ods/build_flights.py) creates the timestamp at query execution; `write_ods_flights` and [`dwd/build_flights.py:write_dwd_flights`](../../src/airline/dwd/build_flights.py) overwrite snapshots. DWD derives the target from observed `arr_delay_minutes` and eventual cancellation/diversion flags without recording their first availability. [`test_ods_flights.py`](../../tests/integration/test_ods_flights.py) deliberately excludes ingestion time from stable rebuild comparisons. Metadata non-nullness tests do not validate historical receipt semantics.

**Download and snapshot provenance.** The download script does not retain per-event receipt logs, provider response-header histories or timestamped successive downloads. It reuses an existing ZIP and extracts CSVs with overwrite. Inspected ZIP member timestamps are **2023-03-28 08:33:34** for January 2023 CSV and **2024-09-27 11:26:26** for July 2024 CSV. ZIP metadata is not proof of first publication/receipt, has no demonstrated event-level or timezone contract, and cannot justify an assumed reporting lag. See [ZIP evidence](evidence/zip-metadata.json).

The raw manifest has only `year,month,filename,sha256`; [`raw_snapshot.py`](../../src/airline/common/raw_snapshot.py) verifies bytes and coverage, not as-of availability. Four discovered manifest copies under configuration/local study/archive evidence locations have the same digest **`fef0d967bbc36d2beb2898701428ca55aa5b0f0a7c9bb0f5ede2ed8ac7852efb`**, matching the accepted bundle's raw fingerprint. Copies of one inventory are not four event-version histories. [Manifest-copy evidence](evidence/raw-manifest-copies.json).

**History/schedule archives.** `data/history/baseline/checkpoint/` holds selection/test feature and model/evaluation recovery snapshots, as described in `docs/CHECKPOINT.md` (historical original-repository file `../../docs/CHECKPOINT.md`; omitted from this compact reconstruction). Pipeline code revisions, checkpoint timestamps, model versions, filesystem mtimes and dated delivery archives establish artifact provenance; they do not establish when an individual flight outcome or schedule revision was known. The inspected source/configuration contains no configured operational outcome stream or schedule change-history store. Cached TAF research reports do not supply flight-outcome or schedule availability. This is a finding about the inspected local sources, not a claim that no external provider could supply the missing evidence; no external operational account/feed was supplied or queried.

## As-of and coverage implications

The current [`flight_time.py`](../../src/airline/dws/flight_time.py) explicitly declares **zero reporting latency** and reconstructs availability as scheduled departure UTC + signed departure delay + actual elapsed time. The feature builders admit historical outcomes whose reconstructed arrival is strictly before scheduled target departure. They do not test actual receipt; `_ingested_at` is not used as historical availability. Source event completion, provider publication, local receipt and model-ready availability are distinct clocks.

For a proven operational policy, define the actual decision cutoff **p**, retain schedule versions observed before p, and choose only source versions whose required outcome fields/statuses were usable strictly before p. A conservative source check also requires completed arrival and source departure before p. Keep the declared departure-window anchor explicit: 3h for origin/destination and 7/30/90d for all legacy entities. If p precedes scheduled departure t, information received in `(p,t)` cannot be used merely because the research cutoff was t. Neither actual issue time nor these version links are established here.

Corrections must be replayed as known then, not applied retrospectively. The current `flight_id` includes scheduled departure time, so changing a schedule can change the business key; it is not by itself a stable cross-revision identity. Unknown availability must remain unknown/excluded. The eventual eligibility filter is suitable for labeled research rows, not proof of which future target schedules existed at p.

Physical month coverage and prior feature-build row counts do not prove live completeness by carrier, airport or route. No receipt-qualified 3h/7d/30d/90d coverage, outage detection, field-finalization timing or lateness distribution can be established from these files. The absence of pre-2023 source files also prevents a full initial 90-day warm-up (and shorter boundary windows). Empty histories at the snapshot boundary must not be described as proof of no historical flights. No configured post-2024 source establishes current recent histories or an untouched prospective as-of-valid assessment period. The previously inspected September–December 2024 test period remains non-pristine; it was not evaluated here.

A receipt-aware or earlier-cutoff policy needs its own identity and separately authorized qualification. **Do not transfer the research candidate's +0.0356365387 mean PR-AUC gain to changed availability semantics.**

## Evidence needed to reopen the gate

Obtain immutable, timestamped outcome and schedule versions with stable cross-revision identifiers; documented event/publication/receipt/model-ready clock semantics and UTC synchronization; append-only correction/cancellation lineage; prediction issue logs bound to input versions; and completeness/retention evidence across every legacy entity and both 3h roles, including 90-day warm-up and outages. Demonstrate strict cutoffs with actual as-of replays and a genuinely untouched/prospective period under a protocol frozen before labels are examined. These are requirements, not results of this audit.

After that evidence is audited, **explicit authorization is still required before Phase 2**, even if a subsequent Phase 1 finding is GO. A research-only package could be proposed separately under the unchanged assumption, but has not been implemented and is not live-production authorization.

## Reproduction, scope and mandatory stop

Only the new audit directory contains writes. [Source/header inventory and hashes](evidence/source-audit.json) bind the inspected code, configuration, reference reports and accepted manifest. The [audit reader](audit_metadata.py) uses PyArrow without starting Spark, reading Parquet footers plus one bounded train/validation ingestion-metadata row. It reads no flight outcome values or test-period rows; whole-ODS footer counts are structural metadata, not test evaluation.

The production base image lacks PyArrow; that initial read-only attempt failed with `ModuleNotFoundError`. Nothing was installed or rebuilt. The successful audit used PyArrow already present in the earlier study image below; no model library was imported. This was a disposable network-disabled metadata tool, not a replacement or resize of the production runtime. Exact successful command:

```bash
docker run --rm --network none --read-only --cpus 1 --memory 2g \
  --entrypoint python3 -e PYTHONPATH=/opt/xgb-study \
  -v /home/server/workspace/industry/Airline/studies/lr_3h_availability_audit_v1/audit_metadata.py:/audit_metadata.py:ro \
  -v /home/server/workspace/industry/Airline/data/warehouse/ods/flights:/ods:ro \
  -v /home/server/workspace/industry/Airline/data/warehouse/dwd/flights/year=2023/month=1:/dwd-selection-month:ro \
  sha256:c254b81baee4c85f919de3e1fd9ae6be383f0948c9e64750ec227572fe0c326e \
  -B /audit_metadata.py \
  > studies/lr_3h_availability_audit_v1/evidence/parquet-metadata.json
```

Other checks used Python standard-library CSV header/ZIP directory parsing, manifest SHA-256 comparisons and bounded source/path searches. No external download, dependency installation, Spark session, feature rebuild, model loading/scoring/fitting, test evaluation, packaging, routing change, production modification or main-branch merge occurred. Existing model identity remains **`d0098a9572834c53bf76b04d732a94a6`**, schema-3 LR, threshold **0.39**. The audit container exited successfully and was removed with `--rm`.

**STOP at the Phase 1 approval gate. Operational availability is BLOCKED. Await explicit authorization; do not enter Phases 2–5.**
