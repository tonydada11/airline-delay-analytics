# Data dictionary

Counts below are observed for the January 2023–December 2024 BTS snapshot used by this repository. They validate that snapshot; they do not make Spark part-file names or `_ingested_at` timestamp values part of the contract.

## Raw, ODS, and DWD

| Logical dataset | Path | Grain / key | Partitioning | Observed rows | Contract and important columns |
|---|---|---|---|---:|---|
| Raw BTS CSV | `data/raw/csv` | one source flight record | 24 monthly CSV files | 13,926,960 | BTS On-Time Reporting Carrier source; 2023-01-01 through 2024-12-31 |
| ODS flights | `data/warehouse/ods/flights` | one source flight record | unpartitioned Parquet dataset | 13,926,960 | 109 typed BTS fields plus `_source_file` and `_ingested_at`; source content and basename are stable, ingestion timestamp is not |
| DWD flights | `data/warehouse/dwd/flights` | source flight record; candidate key `(flight_date, reporting_airline, flight_number_reporting_airline, origin_airport_id, dest_airport_id, crs_dep_time)` | `year`, `month` | 13,926,960 | snake_case source fields; `is_cancelled`, `is_diverted`, schedule hour/minute fields, and nullable `is_arrival_delayed_15m` |

The DWD target is defined only for non-cancelled, non-diverted flights with known arrival delay minutes. It equals 1 when `arr_delay_minutes >= 15`, 0 otherwise, and null when ineligible. The observed DWD contains 2,836,665 delayed, 10,871,985 not-delayed, and 218,310 null-target rows.

## Dimensions

| Dataset | Path | Grain / keys | Partitioning | Observed rows | Important columns |
|---|---|---|---|---:|---|
| Date | `data/warehouse/dim/date` | one `flight_date`; key `date_key` | none | 731 | `year`, `month`, `day_of_month`, Monday-based `day_of_week`, `day_name` |
| Carrier | `data/warehouse/dim/carrier` | one `reporting_airline`; key `carrier_key` | none | 15 | DOT and IATA reporting-carrier identifiers; latest deterministic attributes |
| Airport | `data/warehouse/dim/airport` | one `airport_code`; key `airport_key` | none | 355 | airport/sequence/city-market IDs, city, state, WAC; union of origins and destinations |
| Route | `data/warehouse/dim/route` | directional `(origin, destination)`; key `origin->destination` | none | 7,093 | `origin_airport_key`, `destination_airport_key` |

All surrogate keys and natural grains are unique and non-null, and every DWD date, carrier, airport, and route resolves to its dimension.

## DWS and ADS analytics

Daily DWS and presentation ADS tables share these metrics:

- `scheduled_flights`, `completed_flights`, `cancelled_flights`, `diverted_flights`
- `eligible_delay_flights`, `delayed_flights`
- `delay_rate = delayed_flights / eligible_delay_flights` when the denominator is positive
- `avg_arrival_delay` over completed flights with known arrival delay
- `cancellation_rate = cancelled_flights / scheduled_flights`

| Dataset | Path | Grain | Partitioning | Observed rows |
|---|---|---|---|---:|
| Carrier daily | `data/warehouse/dws/carrier_daily` | `(flight_date, reporting_airline)` | none | 10,965 |
| Origin-airport daily | `data/warehouse/dws/origin_airport_daily` | `(flight_date, origin)` | none | 239,591 |
| Directional-route daily | `data/warehouse/dws/route_daily` | `(flight_date, origin, destination)` | none | 3,554,199 |
| Monthly operations | `data/warehouse/ads/monthly_operations` | `(year, month)` | none | 24 |
| Carrier scorecard | `data/warehouse/ads/carrier_scorecard` | `reporting_airline` | none | 15 |
| Airport scorecard | `data/warehouse/ads/airport_scorecard` | `origin` | none | 355 |
| Route ranking | `data/warehouse/ads/route_ranking` | `(origin, destination)` | none | 7,093 |

ADS count metrics are sums of the matching daily DWS values. `avg_arrival_delay` is recomputed from DWD so the result is correctly weighted rather than an average of daily averages.

## Production ML features and predictions

The feature contract is `m7-flighttime-v2`, defined in `src/airline/dws/ml_feature_contract.py`. Separate datasets at `data/warehouse/dws/ml_flighttime_v3_selection` and `data/warehouse/dws/ml_flighttime_v3_test` preserve one row per eligible flight across the locked splits. Snapshot-specific acceptance counts live in `tests/data_quality/test_ml_feature_quality.py`.

| Columns | Meaning |
|---|---|
| `reporting_airline`, `origin`, `destination`, `calendar_month`, `calendar_weekday` | Five categorical predictors |
| `distance`, `crs_elapsed_time` | Scheduled route length and duration |
| `departure_sin`, `departure_cos`, `arrival_sin`, `arrival_cos` | Scheduled local minute-of-day cyclic encodings |
| `departure_missing`, `arrival_missing`, `duration_missing` | Three schedule missing indicators |
| `{carrier,origin,destination,route}_flight_count_{7,30,90}d` | Prior eligible-flight count; empty history is 0 |
| `{carrier,origin,destination,route}_delay_rate_{7,30,90}d` | Prior delayed / eligible count; empty history is null |
| `{carrier,origin,destination,route}_history_missing_{7,30,90}d` | 1 for empty history, otherwise 0 |
| `is_arrival_delayed_15m` | Binary target; excluded from predictors |
| `flight_date`, `split` | Date and train/validation/test controls; excluded from predictors |

Output order is lineage metadata; categorical predictors; schedule numerics; history rates; history counts; history missing indicators; target; split. All actual-time/delay fields remain excluded. Mean imputation is learned from training only for schedule numerics and history rates.

For target scheduled departure UTC P, a historical source must depart in `[P−N*24h, P)` and its reconstructed actual arrival must be strictly before P. Arrival = scheduled departure UTC + signed departure delay + actual elapsed time, assuming zero reporting latency. Publication/ingestion timestamps are unobserved. Same-day completed flights may contribute; in-flight overnight and future flights cannot. Selection sources exclude test rows; final-test histories can use causally available earlier test outcomes. Unknown target times yield empty histories; invalid source outcomes are excluded. Pinned airport timezones and locked pytz handle local times and DST; `2400` rolls to the following date.

Split ranges (inclusive) are 2023-01-01–2024-04-30, 2024-05-01–2024-08-31, and 2024-09-01–2024-12-31. **The last range is a previously inspected historical test period, not a pristine external holdout.**

The feature dataset is range-distributed on split/date using `max(4, ceil(rows/250000))`, then written into split directories. Execution partition count and file count need not equal the three split directories.

Models reside under `artifacts/flighttime-v3/models/{preprocessing,classifier,manifest.json}`. Evaluation outputs are `artifacts/flighttime-v3/metrics/{validation,test}.json` and `artifacts/flighttime-v3/predictions/{validation,test}/`. Predictions contain `flight_id`, all six business-key fields, `split`, `is_arrival_delayed_15m`, scalar `probability`, and binary `prediction` (probability >= the validation-selected integer threshold step / 100). Metrics include rows, prevalence, confusion matrix, precision, recall, F1, predicted-positive rate, ROC-AUC, PR-AUC, model identity and historical-test status. Ranking metrics are null with an explicit reason for single-class splits. Evaluation reports additionally contain versioned completion evidence: content fingerprints and relative file inventories for the evaluated model/dataset and saved predictions, validated row count, and report digest. Invalid or legacy completion evidence triggers scoring again with the frozen model and threshold, without fitting or tuning.

## ML schema version 3

The feature-definition version is m7-flighttime-v2; schema 1/2 artifacts require rebuilding and reselection. Feature output order is flight_id (string SHA-256), flight_date (date), flight_number_reporting_airline, origin_airport_id, dest_airport_id, crs_dep_time (integer types), the existing 50 feature columns, target, split. reporting_airline remains an existing categorical predictor and is also part of the business key. The added metadata never enters vectors. Prediction output is flight_id, the six ordered business-key fields, split, target, probability (double), prediction (int). See [canonical serialization and duplicate rules](pipeline-rules.md). `_dataset_manifest.json` records exact ordered output types, snapshot provenance, duplicate counts, phase/split inventory, time-policy/reference version, unknown-time counts, and completion. Test manifests identify their frozen selection bundle/run. Production manifests embed exact validation selection evidence and selected classifier/threshold; no fixed model or threshold contract remains.
