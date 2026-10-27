# Implementation Planning

## From architecture to milestones

**User:**

We've picked the architecture. Can we turn it into something I can implement a piece at a time? I need the files, their responsibilities, and the order to build them.

**ChatGPT:**

I'd use `src/airline/` as the source package, with small runnable modules for each persisted output. Each job should have a transformation we can test separately from reading and writing.

Here's the order I'd use:

| Milestone | Deliverable |
|---|---|
| M0 | Shared configuration, paths, schema, and Spark runtime |
| M1 | ODS ingestion |
| M2 | Trusted DWD flight facts |
| M3 | Date, carrier, airport, and route dimensions |
| M4 | Daily analytical DWS tables |
| M5 | Flight-level ML features |
| M6 | ADS reports |
| M7 | Training, selection, and evaluation |
| M8 | Full pipeline runner and acceptance |
| M9 | Spark execution experiments |
| M10 | Documentation and reproduction |

M5 and M6 are separate branches from the warehouse, so neither needs the other to finish first.

For each module, I'd separate three concerns: transformation semantics, input/output handling, and execution configuration. That lets a small integration fixture exercise the transformation without requiring every upstream production directory to exist.

The milestone order is also a dependency plan, not just a checklist. DWD must exist before dimensions or features can consume it, while the report and feature branches can evolve independently. Acceptance should be attached to each output contract so we can stop at a verified intermediate state rather than needing the entire project to work before learning whether ingestion was correct.

**User:**

Why is the runner so late? How do I run M1 before M8 exists?

**ChatGPT:**

Each job gets a `main()` entry point. You can run ODS through `python3 -m airline.ods.build_flights` inside the configured environment.

M8 connects those already-runnable jobs and their quality gates. You won't need notebook cells to execute the early stages.

The entry point should resolve configuration, create the Spark session, invoke the stage builder, and propagate failures. It shouldn't contain a second copy of the business transformation. That makes the module callable both from a script and from a focused integration test.

When M8 arrives, it should invoke those same entry points. This avoids a common problem where notebooks use one sequence of transformations while the production runner uses another. Development remains incremental, but the execution surface is already compatible with the eventual end-to-end pipeline.

## M0: get the foundation working

**User:**

Okay. What belongs in common? This is where I tend to overbuild things.

**ChatGPT:**

Start with five modules:

- `config.py`: load project settings into `ProjectConfig`.
- `paths.py`: resolve repository-relative input and output paths.
- `schemas.py`: declare the explicit BTS schema.
- `spark_session.py`: create or reuse the Spark session.
- `quality.py`: a few shared assertions, such as required columns and row counts.

Keep the Spark wrapper small. The runtime should supply the master, so the application shouldn't force `.master("local[*]")`.

I'd keep configuration loading on the driver and pass resolved settings into jobs. Functions that transform DataFrames shouldn't independently search for configuration files or assume the current working directory is the repository root.

The common package should provide concrete shared behavior, not a hierarchy of generic job classes. For example, a schema constant and a path resolver have clear consumers. An abstract ETL framework would create decisions about lifecycle hooks and inheritance before we have evidence that the jobs need them. Start with explicit functions and extract reuse only where it simplifies the actual stages.

**User:**

How do we check that both workers are actually there?

**ChatGPT:**

Add a cluster-health entry point that checks registered workers and runs a real multi-partition action. Printing the configured master address isn't enough; we need evidence that the Standalone cluster executed work.

Keep startup separate from that check. A running container isn't necessarily a registered, usable worker.

There are several distinct checks: containers are up, workers are registered, executors can start, and tasks can read or compute successfully. Passing an earlier check doesn't imply the later ones work. A worker may register while a mounted path or Python environment still prevents useful execution.

The smoke action should have enough partitions to make distributed execution observable, and its evidence should identify the runtime master and executor/task placement. Worker count is a preflight requirement; actual task execution establishes that the configured cluster can perform work. Keep both in the report rather than treating either as a substitute for the other.

**User:**

The inputs are fixed too: 24 monthly CSVs, 13,926,960 records. Can we catch a missing or replaced file before starting?

**ChatGPT:**

Yes. Put the pinned inventory check in `common/raw_snapshot.py`, with a manifest reader and verifier. Run it before ingestion. The check should establish that we're reading the expected snapshot, rather than whatever files happen to match a glob.

For M0, stop once paths, configuration, schema, snapshot verification, and the distributed smoke check work. ODS can wait until that foundation is usable.

A row count is useful reconciliation, but it isn't a complete source identity. The manifest should capture the expected file inventory and whatever identity evidence the project pins, so replacing a file with a different release cannot pass merely because its row count is similar.

The verifier should fail before downstream writes if the snapshot is incompatible. That avoids mixing outputs from different inputs under the same run label. This is a narrow reproducibility contract for the fixed dataset; we don't need to generalize it into a discovery service for arbitrary future datasets.

## M1–M2: ingest and standardize flights

**User:**

Let's map out ODS next. The CSVs stay unchanged, and Spark reads them directly.

**ChatGPT:**

Use `ods/build_flights.py` with `read_raw_flights()`, `add_lineage_columns()`, `write_ods_flights()`, and a `build_ods_flights()` function connecting them. `main()` handles configuration and the Spark session.

The reader uses the explicit 109-field schema. Lineage retains the source file and ingestion timestamp. The output is typed Parquet, with rows reconciled against the pinned input.

The read function should return a typed frame without embedding downstream business decisions. Lineage enrichment should add provenance without filtering the population. The write function owns the intended output path and write behavior, while the builder coordinates validation around those steps.

For the first fixture, use distinctive values in fields across the beginning, middle, and end of the record. A successful read with the expected number of columns can still conceal positional misalignment. The full-data gate then checks month coverage and row reconciliation after the reader's behavior has been established on something small enough to inspect directly.

**User:**

There's also that empty trailing CSV field. Handle it.

**ChatGPT:**

That belongs in parsing. We should test that it doesn't shift the meaningful fields, while leaving the source files untouched.

An empty terminal token is a serialization detail, so the reader and schema need to agree on how it is handled. We should verify that the last meaningful source field retains its correct value and type, and that no placeholder field enters downstream feature lists.

The fixture should reflect the actual CSV shape rather than relying on a hand-created row that omits the problematic delimiter. That makes the test protect against the parsing issue we observed while keeping the business schema and the original files intact.

**User:**

What happens when I run ODS twice?

**ChatGPT:**

Overwrite the intended snapshot. Then check that the second build has the same business rows and source lineage. The ingestion timestamp may change, so comparing every output byte would be the wrong test.

Logical equality is the appropriate acceptance criterion. Parquet filenames, partition ordering, and ingestion timestamps can change even when the business content is identical. Conversely, matching total counts doesn't prove equality if one record was lost and another duplicated.

Compare the stable projected rows as a multiset, retain source lineage in that projection, and reconcile counts. Keep fixture output directories isolated so the test can exercise overwrite behavior without touching the pinned raw input or a completed production warehouse. The same logical contract can then be applied to the full-snapshot rerun.

**User:**

For DWD, I'd like more than renamed columns. What goes in the transformation?

**ChatGPT:**

`dwd/build_flights.py` should standardize names and types, parse schedule components, derive status flags, and create the nullable arrival-delay label. Keep input validation, transformation, and writing separate.

Partition the output by year and month. Preserve one output record per source occurrence, including cancellations and diversions.

For schedule parsing, distinguish valid conventions from malformed values. A value such as `2400` needs a documented midnight interpretation, while impossible hour/minute combinations should be identified by validation rather than quietly converted to a plausible hour.

For labels, derive the binary target only where the outcome is assessable. Null should remain a meaningful state for ineligible records. Separate those decisions from column renaming so tests can assert both population preservation and semantic values. The writer can then materialize the validated frame with the intended date partition columns.

**User:**

And don't deduplicate on `flight_id` automatically. Two rows having the same business key doesn't tell us which one to delete.

**ChatGPT:**

Agreed. Preserve multiplicity and report the duplicate profile. If the fixed snapshot has an expected uniqueness property, check it explicitly rather than forcing it with `dropDuplicates()`.

The ML eligibility filter belongs downstream. A cancelled flight should still contribute to scheduled volume even though its delay label is null.

The distinction is between a key value and an occurrence. A deterministic hash of business fields can identify the same value on rerun, but it doesn't necessarily distinguish two physical rows sharing those fields. Joining on that key later also requires care, because multiple occurrences on both sides can multiply results.

Acceptance should therefore examine both row counts and join cardinality. If we ever choose to deduplicate, it needs an explicit source-backed rule and an accounting of removed occurrences. It shouldn't emerge incidentally from a convenience operation used to satisfy a uniqueness assertion.

**User:**

What would make M2 done?

**ChatGPT:**

Row reconciliation, correct types and status semantics, nullable labels, valid schedule handling, and the expected Parquet layout. Also verify a second build and a date-filtered scan, so we're checking rerun behavior and partition pruning as well as the transform.

I'd organize M2 acceptance around several different failure modes:

- Lost or duplicated occurrences: multiset and count reconciliation.
- Incorrect business meaning: status and nullable-label fixtures.
- Bad schedule interpretation: valid, invalid, and midnight boundary cases.
- Ineffective storage layout: partition directories and a filtered execution plan.
- Unstable rebuilds: equality of stable business content after a second write.

These checks complement one another. A schema-valid output may have the wrong population, and a logically correct output may still be written in a layout that prevents the intended partition pruning.

## M3–M4: dimensions and daily metrics

**User:**

For dimensions, can we just have four straightforward builders?

**ChatGPT:**

Yes: `build_date.py`, `build_carrier.py`, `build_airport.py`, and `build_route.py` under `dim/`. Each can follow the same read, validate, transform, write pattern without introducing a framework.

Use BTS natural identifiers where they work. Build the airport dimension from both endpoint roles and derive directed route keys from ordered origin/destination pairs.

The pattern can be shared conceptually without requiring a configurable framework. Date generation, carrier attributes, airport union, and ordered route pairs are different enough that explicit transforms will be easier to inspect.

For attributes that appear in multiple source records, define deterministic selection rather than taking an arbitrary first row. Then validate the dimension key's uniqueness before publishing. A dimension that accidentally contains duplicate keys can multiply fact rows on enrichment even though every fact key appears somewhere in the dimension, so uniqueness and reference coverage are separate acceptance checks.

**User:**

So an airport that only appears as a destination still needs to be there.

**ChatGPT:**

Exactly. Check key uniqueness and use anti-joins to find any DWD references that don't resolve. Include reverse routes in the fixture so an accidentally unordered route key fails visibly.

For coverage, project the fact's relevant natural keys, compare them with the dimension keys using a left-anti join, and require no unresolved references under the intended contract. Do this separately for airport origin and destination roles.

Also validate the dimension itself before interpreting the coverage result. A zero-row anti-join says the keys exist, but it doesn't say each key exists only once. Route fixtures should contain both A→B and B→A so a sorted endpoint pair cannot accidentally satisfy a superficial count check while erasing directionality.

**User:**

The next three daily tables share most of their metrics. Where should that logic live?

**ChatGPT:**

Put it in `dws/daily_metrics.py`. An `aggregate_daily_flights()` function can accept the grain columns; the carrier, origin-airport, and route builders supply their own grouping keys.

Calculate scheduled, completed, eligible, delayed, cancelled, and diverted counts before deriving rates. Keep the definitions in one place.

The shared aggregator should expose the components needed by later consumers. Counts are additive across compatible groups; already-computed rates usually aren't. That is why scheduled count, eligible count, delayed count, and any sum/count components for averages should remain available in the daily outputs.

Each builder still validates its output grain, such as date plus carrier. A shared formula doesn't guarantee the grouping columns are correct. Tests should cover both the arithmetic and the uniqueness of the resulting keys, then reconcile additive totals with the trusted flight population.

**User:**

I want early arrivals included in average delay. And what does delay rate mean when there are no eligible flights?

**ChatGPT:**

Use signed arrival delay, including negative minutes. For a zero eligible denominator, return a defined null rather than zero. Zero would imply we observed eligible flights and none were delayed.

We'll test mixed-status groups and reconcile the aggregate counts back to DWD.

For a simple fixture, take eligible arrival delays of −10, 0, and 40 minutes. The mean signed delay is 10 minutes, while clipping negative delays to zero would produce about 13.3. That difference makes the intended semantics visible.

A separate group containing only cancellations or otherwise ineligible outcomes should have no assessable arrival-delay rate. Null communicates “undefined for this population”; zero communicates an observed rate. Keeping that distinction lets reports decide how to display missing measures without silently changing the underlying calculation.

## M5–M6: features and reports

**User:**

Can we get the eligible rows and schedule features working before adding all the history calculations?

**ChatGPT:**

Yes. Start `dws/build_ml_flight_features.py` with eligibility, schedule features, and chronological split assignment. Keep the feature lists and split constants in `ml_feature_contract.py`, so feature generation doesn't import training code.

Then add `entity_history()` as a separately tested step. It will need a timestamp policy shared across all entity histories.

The initial acceptance should establish eligibility counts, stable row multiplicity, scheduled predictors, and exact split boundaries. That creates a simpler reference before historical features introduce additional joins and temporal conditions.

When histories are added, compare the eligible keys and split assignments before and after. Feature enrichment should not change which target occurrences exist. Keeping the contract constants separate also makes it possible for validation and manifest code to inspect expected columns without loading model-fitting machinery or starting unrelated Spark work.

**User:**

We're using carrier, origin, destination, and route histories over 7, 30, and 90 days. How do you avoid a huge join against every earlier flight?

**ChatGPT:**

An event-sweep calculation is worth implementing for those bounded windows. Represent the relevant times precisely, track when a source outcome becomes eligible, and expire it at the lookback boundary.

First verify it against a small independent calculation. For target cutoff P, the source scheduled departure must be inside the lookback interval and its reconstructed arrival must be strictly before P. Duplicate occurrences must contribute separately.

The conceptual challenge is that a source has two relevant conditions: it must still be inside the departure-based window, and its outcome must already be available. Sorting by departure alone doesn't encode the second condition. An event-based calculation can track additions and expirations, but equality ordering must implement the strict cutoff deliberately.

Before optimizing, write an independent small oracle that enumerates qualifying source rows for each target. Compare counts and sums for multiple entities, duplicate occurrences, empty histories, and exact boundaries. The optimized Spark result should match that oracle rather than merely agree with another invocation of its own helper.

**User:**

Let's keep the timezone work out of a giant feature-builder function.

**ChatGPT:**

Use `dws/flight_time.py` for converting origin-local schedules to UTC and reconstructing arrival from scheduled departure, signed departure delay, and actual elapsed time. Test overnight flights, DST, and the `2400` convention there.

The availability assumption remains zero reporting latency. These timestamps don't establish when an operational system received the result.

The helper should keep local schedule interpretation separate from UTC arithmetic. Once scheduled departure is resolved to an instant, signed departure delay and actual elapsed time can reconstruct an arrival event under the chosen policy.

DST requires explicit behavior for ambiguous or nonexistent local times; machine-local timezone defaults are not a sufficient contract. Tests should target those cases directly and verify missing inputs remain visible. Using a precise common timestamp representation also helps keep equality comparisons consistent across feature construction and the independent test oracle.

**User:**

Now ADS. Monthly operations, airline and airport scorecards, and route rankings. These can use DWS and DWD, right?

**ChatGPT:**

Yes. Give each output a builder under `ads/`, sharing the rate and aggregation rules where appropriate. They shouldn't reread raw CSVs.

For monthly rates, add numerators and denominators before dividing. Check a small result directly against DWD as well as DWS, so a shared formula error can't pass merely because two layers agree.

The report builders should make their final grains explicit: monthly system metrics, carrier scorecards, airport scorecards, and directed-route rankings are different outputs. Shared arithmetic can live in a helper, but each builder remains responsible for grouping and any ranking semantics.

For example, test a high-volume low-delay day beside a low-volume high-delay day. The correct combined rate should differ substantially from the mean daily rate. That fixture makes an aggregation bug observable without needing to scan the warehouse or compare against another output using the same helper.

## M7–M8: fit, evaluate, and run the pipeline

**User:**

I'd keep modeling small: a baseline, Logistic Regression, and one GBT. How do we separate fitting from evaluation?

**ChatGPT:**

Use `ml/preprocessing.py` for train-fitted preprocessing, `ml/experiments.py` for candidate and threshold selection, and `ml/train.py` to publish the selected fitted model. `ml/evaluate.py` only loads and scores it. Shared metric calculations belong in `ml/metrics.py`.

Fit on train, choose the classifier and threshold on validation, and save that exact winning fit. Don't refit on train plus validation before reporting test results.

Learned preprocessing belongs inside the fitted artifact boundary. Category mappings, imputation values, and the model must correspond to the same training population and feature order. Reloading only the estimator while rebuilding preprocessing could change the meaning of vector coordinates.

Selection should preserve enough metadata to identify candidate settings, threshold steps, and the chosen fit. Evaluation then becomes a narrower operation: validate input compatibility, load the bundle, produce continuous scores, apply the saved threshold, and calculate reports. That separation makes it easier to verify that no test-time fitting or implicit retuning occurs.

**User:**

Then M8 can connect everything. One shell runner, no wrapper for every job.

**ChatGPT:**

Use `scripts/run_pipeline.sh` for preflight, stage execution, and quality gates. Keep cluster startup separate. If DWD validation fails, downstream stages must not run.

The order should be ODS, DWD, dimensions, analytical DWS, selection features, ADS, model selection and validation, then frozen historical-test processing. A separate test runner can distinguish ordinary fixture checks from expensive full-data acceptance.

The shell script should report the failed stage and preserve the failing exit status. A quality gate should be part of the stage's required success path, not a best-effort diagnostic whose failure is ignored.

Use a controlled test with stub stage commands that record their invocation order and intentionally fail in the middle. That verifies actual stop behavior rather than searching the script for `set -e`. Full-data checks can be opt-in during ordinary development, but an acceptance report must say whether those gates ran before calling the complete pipeline accepted.

## M9–M10: experiments and handoff

**User:**

For M9, I want more than `explain()`. Can we compare joins, partition counts, caching, and shuffles on actual warehouse data?

**ChatGPT:**

Yes. Use a bounded real workload in `performance/run_experiments.py`. Compare logically equivalent variants, save physical plans and settings, and repeat measurements with alternating order where appropriate.

Don't assume caching or extra partitions will help. The point is to explain the observed behavior on this two-worker setup.

Separate correctness comparison from timing. First establish that variants return equivalent results, including duplicate multiplicity where relevant. Then record durations, physical operators, task/shuffle information, and resource settings under repeated comparable runs.

Caching needs special care: materializing a cache has a cost, and whether it pays off depends on reuse. Likewise, more partitions can add scheduling overhead to a bounded workload. Record the actions included in each timed region so the report makes clear whether a comparison measures initial materialization, reuse, or an entire sequence.

**User:**

I'd also like to kill a worker and see Spark recover.

**ChatGPT:**

Keep that in an isolated lab. Kill a worker while a task is active, then require evidence of executor loss, a retry on the surviving executor, and correct final output.

It will demonstrate task recovery. Both workers still share one host and filesystem, so it won't demonstrate host-level durability or driver recovery.

Synchronize failure injection with observed task activity instead of relying only on a fixed delay after application startup. Otherwise a fast run could finish before the worker is killed, producing a misleading success.

After injection, inspect event evidence linking the lost executor to a failed task attempt and subsequent successful retry. Verify the deterministic result independently. The lab should run in an isolated topology so deliberately stopping a worker doesn't interrupt unrelated warehouse jobs or leave the accepted environment in an unexplained state.

**User:**

And M10 is documentation. I want someone else to be able to run this, not just read that it worked on my machine.

**ChatGPT:**

Document the pinned inputs, runtime, commands, table grains, feature assumptions, evaluation rules, and saved results. Maintain `docs/guides/project-guide.html` directly as the authoritative project guide.

Record which checks actually ran. A skipped full-data test must remain a skipped test in the handoff.

A useful handoff connects claims to evidence. A row-count claim should identify the pinned snapshot and reconciliation; a model score should identify the population, model, threshold, and report; a speed claim should identify the compared workload and resource settings.

Documentation should also explain how to reproduce a bounded check before attempting the expensive full pipeline. Keep the maintained HTML guide aligned with the executable source, and describe notebooks as explanatory entry points to the same source modules. That reduces the chance of a reader following a polished but outdated execution path.

**User:**

Good. Let's use this as the plan and review each milestone separately. Once the accepted version works, I want to stop adding features without a reason.

**ChatGPT:**

Then start with M0 and verify it before building ODS. Keep later model or feature ideas as separate experiments with their own hypothesis and acceptance criteria.

At each milestone, keep the accepted behavior and the evidence that established it together. If a later change alters the feature schema or evaluation contract, make that change explicit and rerun the checks affected by it.

Once the agreed scope is complete, improvements should begin with a concrete question: what limitation are we addressing, what will change, and what observation would justify keeping it? That gives you room for research without letting each review silently redefine what the production pipeline is supposed to do.

---