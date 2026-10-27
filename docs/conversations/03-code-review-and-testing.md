# Codex Milestone Reviews and Test Development

## Setting up the review

**User:**

As I finish each milestone from M0 through M9, I'll ask you to write and run its tests before I implement the next one. For the completed milestone, inspect the implementation and existing tests, identify missing coverage, then create or extend the tests yourself. Include the relevant unit, integration, data-quality, orchestration, performance-harness, and worker-recovery checks. Reuse tests that already establish the required behavior.

Give a brief explanation of each milestone's approach, then do the work and run the relevant checks. Finish testing the completed milestone without waiting for me to request each test. Use small isolated fixtures for ordinary runs; implement expensive full-data gates as opt-in and report whether they ran. Don't weaken assertions or change expected behavior to get a green result. If a test exposes a real implementation bug, show the reproducer and make the smallest justified fix. I don't want a general refactor.

After each milestone, report changed files, commands actually executed, results, skipped checks, and any blockers. Explaining what a test should do is only the starting point; I want the tests implemented in the repository.

**Codex:**

When you finish a milestone, I'll inspect its source and existing assertions, implement any missing tests, and run the focused checks. I'll use the documented contracts and independent expected values so the tests can expose incorrect behavior. I'll wait for you to finish the next milestone before testing it.

I'll distinguish existing coverage, new tests, and confirmed implementation defects. Each milestone report will identify what changed and what actually ran. Full-data and worker-loss checks will remain separate from ordinary fixture runs, with unexecuted checks reported explicitly.

## M0: schema and runtime

**User:**

I've finished M0: configuration, paths, schema, snapshot verification, and Spark runtime setup. Write and run the tests for it.

**Codex:**

I'll write or extend the tests for configuration loading, repository-relative paths, the BTS schema, snapshot verification, and Spark-session creation, then run the relevant checks. A schema test that only counts columns would be too weak: it could pass even if two fields were swapped.

For example, two adjacent numeric fields could be swapped while the total field count and even their data types remain unchanged. Only an ordered field-name assertion, combined with a parsing fixture containing distinctive values, would expose the mismatch.

Configuration and paths need a different kind of test: resolve from a known project root, verify required settings, and make failures observable. Spark-session creation should inherit the supplied runtime configuration. If the test substitutes local mode, it may verify transformation semantics while providing no evidence about the intended Standalone execution environment.

**User:**

There should be 109 fields. I want names, order, and types checked. Also make sure a test isn't hiding cluster problems by forcing local mode.

**Codex:**

I'll check and extend `tests/unit/test_schemas.py` for field order, explicit types, and nullability. The existing assertions check selected positions and types, so I'll verify the complete contract rather than infer coverage from the test names. I'll use `test_paths.py`, `test_config.py`, `test_quality.py`, and `test_raw_snapshot.py` for the other shared contracts, including missing, replaced, or incompatible source files.

Runtime is a separate question. `tests/pipeline/test_runtime_contract.py` covers the pinned configuration, but cluster acceptance also needs registered workers and a real distributed action. A correct master URL alone doesn't prove tasks reached the workers.

I would treat the evidence as several layers. Schema tests establish the declared interface. Parsing tests establish how actual record shapes map into that interface. Runtime checks establish that the configured processes can execute work.

None replaces the others: a correct schema declaration doesn't prove a CSV was parsed correctly, and a passing local DataFrame fixture doesn't prove two workers are usable. When summarizing acceptance, identify which layer each command exercised. That makes it possible to report strong unit coverage without accidentally claiming a distributed run that wasn't performed.

**User:**

Do we need all the negative configuration tests? Some settings are controlled by the repository.

**Codex:**

Keep the ones protecting actual inputs or path boundaries. I wouldn't add fallback branches just to make a hypothetical configuration work. If the required cluster configuration is missing, a visible failure is useful.

A useful question is whether the input can realistically differ from the repository's expected value and what damage accepting it would cause. A caller-supplied path pointing outside an intended output area is different from an internal constant that no supported interface can change.

For realistic boundaries, fail with a specific explanation. Avoid broad exception handling that turns misconfiguration into a silent default, because it can make a test pass in the wrong environment. Tests should protect the supported behavior and failure modes rather than encourage implementation branches with no actual consumer.

## M1: CSV parsing and reruns

**User:**

I've finished M1. ODS reads all the files and writes Parquet. Write and run tests that catch parser problems rather than just prove that a directory appeared.

**Codex:**

I'll use `tests/integration/test_ods_flights.py` to test field values across the wide schema, extending existing coverage where needed. `test_read_raw_flights_applies_all_109_fields_without_shift` checks alignment errors that a row-count check misses.

The same module has `test_build_writes_parquet_with_lineage_and_is_idempotent` for writing and rerunning the fixture build.

The fixture should use values that would be visibly wrong in neighboring columns: distinctive airport identifiers, dates, schedules, flags, and numerical measures. Checking only null counts or successful conversion can miss a shift when adjacent fields happen to accept the same type.

Round-trip the output and assert those values again after reload. That covers the reader, lineage enrichment, and writer together. Keep the expected values independent of the parsing implementation; generating expectations from the same schema-indexing helper would risk reproducing the bug in both actual and expected results.

**User:**

But `_ingested_at` changes. We shouldn't expect every column to be identical on rerun.

**Codex:**

Right. Compare stable business columns and source lineage, excluding the intentionally changing ingestion timestamp. Also check that the second build doesn't append another copy of the rows.

I'd project the output onto explicitly stable columns, compare that projection as a multiset, and separately assert that the expected lineage fields are present. That avoids both overly strict byte equality and overly weak distinct-row equality.

The second run should target the same fixture output location. Writing twice to two fresh directories wouldn't demonstrate that overwrite replaces the prior snapshot correctly. After the rerun, verify content and count rather than relying solely on the configured write-mode string, since the observable result is the contract we care about.

**User:**

I don't want every unit-test run scanning fourteen million flights.

**Codex:**

Keep that separate. Small fixtures test parsing and write behavior; `tests/data_quality/test_ods_quality.py::test_full_ods_flights_quality` checks the pinned 24 files and 13,926,960-row reconciliation under the full-data flag.

Fixtures should use disposable inputs and outputs. The original CSVs remain read-only.

This separation also improves failure diagnosis. A small integration test can tell us that a quoted field or trailing delimiter is parsed incorrectly. A full-data check tells us that the real snapshot reconciles under the established behavior. Running the large check first can leave us with an expensive failure and little information about its cause.

The test report should make the selected scope explicit. If the full-data flag is absent, report that gate as not executed. A fast green fixture suite is useful evidence, but it doesn't certify the state of the production warehouse on disk.

## M2: preserve the flight population

**User:**

I've finished M2, the DWD flight builder. Write and run its tests. I especially want to catch cancelled flights disappearing because their label is null.

**Codex:**

I'll write or extend fixtures with an on-time flight, a delayed flight, cancellations, diversions, missing arrival outcomes, and repeated business keys, then check row preservation and derived values. I'll test invalid schedules separately as validation failures.

The warehouse population is scheduled flight occurrences. Only the ML feature stage narrows it to eligible labeled flights.

For each fixture row, write down the expected status, label, and whether it remains in the output. For example, an observed arrival delay below 15 minutes is a negative label, while an unassessable outcome is null; those should not become interchangeable through a blanket null fill.

Include a duplicate occurrence with otherwise identical business values. Its presence tests population preservation independently of the label formulas. Also distinguish invalid source values that should fail validation from valid but ML-ineligible flights that should remain in DWD. Those are different behaviors and need different assertions.

**User:**

And two rows sharing `flight_id` must still be two rows.

**Codex:**

Yes. A set comparison could miss that loss. Use a comparison that preserves duplicate counts and assert that repeated-key fixture rows survive.

I'll use `tests/integration/test_dwd_flights.py` for target/schedule/duplicate logic, invalid flag domains, impossible HHMM values, partition layout, and second-build idempotency. I'll keep full reconciliation in `tests/data_quality/test_dwd_quality.py` and reuse the duplicate-preservation checks in `tests/integration/test_lineage.py`.

A concrete example is an input containing two occurrences of row A and one of row B. An output containing one A and one B has the same distinct set, but it has lost one flight occurrence. A count check catches that simple example, while a multiset comparison also catches compensating errors where another row was duplicated.

The same concern applies to joins: duplicate keys can multiply matches even without an explicit deduplication call. Test the transformation's output cardinality and values together, rather than assuming a stable business-key hash provides unique physical identity.

**User:**

So those are the checks to examine, not a claim that you've found a deduplication bug.

**Codex:**

Correct. Preserving multiplicity is the requirement. We'd need a failing case or a concrete source path that violates it before reporting a defect.

A defensible review comment would say which transformation changes the multiplicity, show the smallest input that demonstrates it, and state the expected versus observed output. Without that evidence, the accurate finding is a coverage question or an unverified risk.

If a fixture already exercises the case and the implementation preserves it, there may be nothing to change. The purpose of reviewing the invariant is to establish confidence in the behavior, not to guarantee that every milestone produces a defect or a new layer of validation code.

## M3: dimensions

**User:**

I've finished M3, the four dimension builders. Write and run their tests, including key uniqueness and fact-reference coverage.

**Codex:**

I'll test the actual keys and joins, including an airport that appears only as a destination, both directions of a route, and DWD keys that must resolve to dimensions.

I'll reuse and extend `tests/integration/test_dimensions.py` for grain and directionality, airport union, left-anti reference checks, deterministic attribute tie-breaking, and reruns. Full-snapshot checks belong in `tests/data_quality/test_dim_quality.py`. Those checks are more informative than adding several versions of a nonempty assertion.

For the airport fixture, include an identifier that occurs only in the destination column and verify it appears exactly once in the dimension. For routes, include A→B and B→A and verify they have distinct ordered identities.

Reference coverage is a separate property from dimension uniqueness. A left-anti join can establish that every fact reference has a match, but duplicate dimension keys could still multiply records during an ordinary enrichment join. Both checks are needed. If attributes conflict across source rows, a deterministic tie-break test prevents output values from depending on partition order.

## M4: daily aggregates

**User:**

I've finished M4, the daily DWS builders. Write and run their tests. I want to be sure cancellations don't dilute arrival-delay rate. Early arrivals should still count toward average delay too.

**Codex:**

I'll build a mixed fixture with different statuses and signed arrival delays, assert hand-calculated counts and averages, and add a group with no eligible arrivals to check the null denominator behavior.

I'll use `tests/integration/test_dws_analytics.py` for early arrivals, zero denominators, reverse routes, internal count/rate consistency, and reconciliation with DWD, with full-data acceptance in `tests/data_quality/test_dws_quality.py`. The three daily builders share `dws/daily_metrics.py`, so I'll follow that formula through each output grain.

A useful numeric fixture would contain eligible delays of −10, 0, and 40 minutes, plus a cancelled row. The eligible mean is 10 minutes and the delayed count is one; the cancellation should affect scheduled volume and cancellation rate without becoming a negative arrival-delay label.

Then add a group with no eligible arrivals. Its delay rate and mean should follow the declared undefined-value policy. These examples test what the metrics mean, while the grain checks confirm that the same formulas are grouped by the intended date and entity keys.

## M5: historical features

**User:**

I've finished M5, the flight-level feature builder. Write and run its tests. The feature table can have the right columns and still leak future outcomes.

**Codex:**

I'll examine `build_ml_flight_features.py`, `flight_time.py`, `ml_feature_contract.py`, and `dataset_manifest.py` together, then write or extend `tests/integration/test_ml_flight_features.py`, `tests/unit/test_flight_time.py`, and `tests/integration/test_dataset_completion.py`. Schema checks cover only part of the contract.

The central question is whether each source outcome was eligible before its target cutoff. Then we need to check that joins preserve one output occurrence per eligible target row.

I would trace one target flight through the entire path: eligibility filtering, prediction-time construction, history membership, feature assembly, split assignment, and publication. That makes it easier to identify where a correct intermediate value could be altered or attached to the wrong occurrence.

The feature blacklist protects against directly passing target outcomes into the vector. It doesn't protect against an apparently safe aggregate containing future outcomes. For that, inspect the source membership condition and test the resulting values. The manifest then protects compatibility and completion, not the mathematical correctness of the history itself.

**User:**

Say a flight departed earlier but hasn't landed yet. It shouldn't contribute. What about one landing at exactly the cutoff?

**Codex:**

Exclude both. The reconstructed arrival must be strictly earlier than the target prediction time. Test just before, exactly at, and just after the boundary.

The timestamp fixtures also need overnight flights, timezones, DST, missing times, and `2400`. A calendar-date comparison won't cover those cases.

Construct a target at a fixed UTC instant and source flights whose reconstructed arrivals lie just before, exactly at, and just after it. Only the first qualifies, provided it also satisfies the lookback-window condition. Test the window's departure boundaries separately so a passing arrival test doesn't conceal an off-by-one expiration error.

Timezone fixtures should represent the same absolute ordering through different local clocks. An overnight case should cross a calendar date, and DST cases should exercise the documented conversion policy. These tests verify information ordering rather than merely that a timestamp column is non-null.

**User:**

Could we change future outcomes and show that earlier features stay the same?

**Codex:**

I'll add or extend that test: modify only outcomes unavailable at the selected target's cutoff, rebuild, and compare that target's features. A change would reveal backward information flow.

The integration suite also perturbs test-period outcomes to check selection isolation. For full-data history checks, `tests/data_quality/test_ml_feature_quality.py` independently samples calculations; its opt-in comprehensive rebuild uses `EXCEPT ALL` to compare multisets.

This is a metamorphic test: instead of requiring a hand-calculated value for every feature, it asserts that a change outside the allowed information set cannot affect the target. Keep the target population and stable identifiers fixed, modify the later outcomes, and compare the selected earlier feature rows.

It complements an independent numeric oracle rather than replacing one. A builder that always returns zero could pass future-perturbation isolation while computing useless histories. We need both correct values on known examples and invariance to information that should be unavailable.

**User:**

And none of these tests proves BTS published a result the moment the flight landed.

**Codex:**

Right. They can verify the implemented zero-reporting-latency rule. They can't supply missing publication or receipt timestamps. That remains an offline assumption, even if every boundary test passes.

Think of two separate claims. The implementation claim is that every included source has reconstructed arrival before the target cutoff. The availability claim is that the outcome was actually obtainable by the system at that time. Historical event data can support the first under the reconstruction policy without establishing the second.

A test suite can verify strict inequalities, timezone behavior, and isolation from later rows. It cannot infer a publication timestamp that isn't present. Documenting the assumption keeps the offline experiment interpretable and identifies what additional evidence a live feature pipeline would require.

## M6: reports

**User:**

I've finished M6, the ADS report builders. Write and run their tests, including a tiny example where averaging daily percentages gives the wrong answer.

**Codex:**

I'll use two days with very different flight volumes and delay rates, calculate the combined delayed count divided by the combined eligible count, and assert the ADS result against that expectation.

I'll reuse and extend `tests/integration/test_ads_analytics.py`, including `test_weighted_rates_and_arrival_averages_are_not_daily_averages`, monthly reconciliation against daily DWS and independent DWD calculations, grains, route direction, and reruns. I'll keep the full-snapshot gate in `tests/data_quality/test_ads_quality.py`.

For example, day one could have 1 delayed flight out of 2, while day two has 9 out of 98. The daily-rate average is roughly 29.6%, but the combined monthly rate is 10/100, or 10%. The difference is large enough that an incorrect mean-of-rates implementation cannot hide behind rounding.

For arrival averages, use signed delay sums and eligible counts at the relevant grain. Then compare the final ADS values with independently computed DWD expectations. That verifies the report arithmetic rather than only the existence and schema of its Parquet output.

## M7: model artifacts

**User:**

I've finished M7, training, selection, publication, and evaluation. Write and run its tests. `.fit()` completing isn't enough; check train-only fitting and frozen evaluation.

**Codex:**

I'll follow the path from preprocessing through selection, publication, and evaluation, using `tests/integration/test_ml_training.py` for train-only preprocessing, save/reload, selection isolation, direct threshold comparisons, and publication of a selected GBT as well as the production path.

I'll check and extend `tests/unit/test_experiment_selection.py` for exact F1 ordering and ties and `tests/unit/test_model_contract.py` for the frozen evaluation contract. Full-data model and prediction checks belong in `tests/data_quality/test_ml_quality.py`.

To test train-only preprocessing, include a validation-only category or an extreme validation value and verify that it doesn't alter the fitted training mappings or imputation statistics. The model bundle should preserve the fitted preprocessing used during selection.

Save/reload tests then establish that the published artifact behaves consistently. Static checks for fit calls are useful guardrails, but behavioral tests should also verify that evaluation uses the saved bundle and frozen threshold. Together they cover the difference between “the evaluator contains no obvious training call” and “the evaluated predictions actually come from the selected fit.”

**User:**

The selected threshold may not be 0.5.

**Codex:**

The evaluator applies the selected threshold explicitly to probabilities. The selector uses integer steps from 0 through 100 and exact rational F1 comparisons, with lower-step and LR tie-breaks.

I'll test those cases directly. Near ties and the default classifier prediction column are both places where a plausible-looking result could differ from the selection rule.

The useful boundary fixture includes a probability exactly equal to the saved threshold, as well as values just below and above it. That establishes the intended comparison operator and catches discrepancies between selection counts and evaluation predictions.

For F1 selection, compare integer ratios rather than rounded decimal strings. Two displayed values may look equal while their exact counts imply a winner. Genuine equality should follow the documented lower-threshold and model tie rules, making the result deterministic across reruns rather than dependent on iteration order.

**User:**

What if publication stops halfway through?

**Codex:**

I'll exercise that boundary in `tests/unit/test_artifact_publication.py::test_publication_is_complete_or_preserves_previous`, and extend the integration checks for manifest compatibility and incomplete-evaluation recovery where needed.

A model directory existing is weaker evidence than a compatible bundle successfully reloading.

The interruption test should exercise a partially written candidate while a previous complete bundle exists. After failure, the old bundle should still load and the incomplete candidate should not be advertised as current. A separate successful path should verify that the newly published bundle is complete and compatible.

Evaluation recovery has a related but different contract: predictions or a metrics file appearing on disk do not establish a completed evaluation. Completion and cache identity need to correspond to the model, dataset, and report being reused, so interrupted output cannot masquerade as a valid cached result.

**User:**

One other thing: the September–December period was already inspected earlier. We still shouldn't tune against it.

**Codex:**

Keep it out of selection and describe it as historical evaluation. Calling it an untouched external holdout would overstate what that result establishes.

There is a distinction between preventing direct selection on test and restoring the status of an untouched holdout. We can enforce the former in code, but we cannot undo prior inspection simply by moving the files or renaming the split.

The honest report should identify which choices were made using validation and describe the historical test as an evaluation of the frozen choice. If future research repeatedly reacts to that test score, it becomes less useful as independent evidence. A genuinely fresh generalization check would require an additional uninspected evaluation population.

## M8: orchestration and acceptance

**User:**

I've finished M8, the pipeline and test runners. Write and run their tests, including a controlled failure that proves downstream stages stop.

**Codex:**

I'll use controlled command stubs in `tests/pipeline/test_m8_orchestration.sh` to inject a failure and check for a nonzero exit, no downstream stage invocation, and no completion message. I'll also use `tests/pipeline/test_cluster_environment.sh` for startup behavior and `tests/unit/test_test_runner.py` for separate full-data processes and opt-in comprehensive rebuilds.

Also check the stage order, especially that model selection completes before historical-test processing.

The test can replace expensive stages with small commands that append their names to a temporary log. Make a chosen stage return a known nonzero status, then assert that the log contains only the expected prefix of the dependency sequence.

Also test a failing quality gate after a successful producer. The runner must stop in that case too; successfully writing an output isn't enough to authorize its consumers. This verifies shell control flow and error propagation without rerunning the warehouse merely to establish orchestration semantics.

**User:**

Some full-data gates are opt-in. I want the report to distinguish.

**Codex:**

Then report the executed command, exit code, and scope. Unit and integration passes don't imply that full snapshot reconciliation or a comprehensive feature rebuild ran. Skips need to stay visible.

An acceptance record should distinguish at least the command, runtime/input scope, exit status, passed checks, skipped checks, and any incomplete expensive gates. That gives a reader enough context to decide what the result supports.

For example, 212 fixture tests passing is evidence about those fixtures under that environment. It doesn't imply the full snapshot was available or that a comprehensive rebuild compared every feature occurrence. Avoid merging those facts into a single undifferentiated “all tests passed” statement when different commands exercise different levels of the system.

## M9: performance experiments and worker recovery

**User:**

I've finished M9, the performance experiment harness. Write and run its tests. I don't want a test failing because broadcast happened to be slower one day.

**Codex:**

I'll test the benchmark's correctness and recording behavior, including equivalent outputs, bounded workloads, variant order, configuration restoration, cleanup, and repeated measurements. The assertions won't require a promised speedup.

I'll use `tests/unit/test_m9_performance.py` for harness behavior and `tests/data_quality/test_m9_performance_quality.py` for results and recorded execution evidence. The latter is opt-in with `AIRLINE_RUN_FULL_M9_QUALITY=1`. Timings are observations for that workload and machine.

The harness itself can be wrong even when the reported timing looks plausible. One variant might include cache materialization while another excludes it, or a configuration change might leak into the following experiment. Tests should verify the measurement boundaries and restoration behavior.

Output equivalence comes first: faster execution is irrelevant if a variant filters different rows or loses duplicate occurrences. After that, save repeated durations and plan/operator evidence. The interpretation should say what was observed under the pinned workload and resources, including cases where an expected optimization did not help.

**User:**

I've also finished the worker-loss lab. Write and run tests for its verifier. It should prove the failure happened during work.

**Codex:**

I'll use `tests/fault_tolerance/test_worker_loss.py` to test the verifier with synthetic event records: a task must be active before the kill, followed by an executor-loss failure, a retry of the same stage/partition on the surviving executor, and the correct final output. I'll also test rejection of missing events, a retry of the wrong partition, and duplicated output. Running these verifier tests is separate from executing the Docker worker-loss experiment.

That supports task recovery on this setup. It doesn't establish recovery from losing the shared host or driver.

The event sequence matters more than an eventual successful process exit. We need to connect an active task to the lost executor, then connect its retry to the same logical work on the surviving executor. Finally, compare the output with the independent expected result.

A test of the evidence verifier should reject incomplete sequences, such as a worker disappearing after work finishes or a final result with no recorded retry. That prevents the lab report from accepting a superficially successful run that never exercised the recovery mechanism being claimed.
