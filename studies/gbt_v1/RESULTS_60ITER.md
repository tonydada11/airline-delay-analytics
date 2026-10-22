# GBT all-features, 60 iterations — results

**COMPLETE — DOES NOT QUALIFY.** One from-scratch `gbt-all-60iter` fit completed
all four validation months, all required artifact/input checks and successful
shutdown. Equal-month PR-AUC is **0.4274314742703018**, a paired mean change of
**-0.0006668483828393135 versus accepted LR**. The required +0.005 gain fails;
both monthly budget-recall guardrails pass.

GBT60 improves over saved GBT30 in every month's PR-AUC, mean
**+0.0038935770033446115**. That descriptive improvement does not qualify it
against LR. Relative to LR, June/July improve while May/August decline.
This is a complete evaluated result. No additional experiment or promotion is
selected. Production remains **accepted LR at 0.39**, bundle
`d0098a9572834c53bf76b04d732a94a6`; these results await separate review.

## Controlled change and execution

[PLAN_60ITER.md](PLAN_60ITER.md) is frozen at executed implementation commit
**`f5794673b48ff4fdc1970a98a9b8aeb63b5b4e67`**, based on clean
`1b6d6e5174f87fcef1e82a0660edded2c6147d34`, branch `study/gbt-improvement-v1`.
Only **maxIter 30→60 changes relative to GBT30**; featureSubsetStrategy remains
`all`. Relative to production `GBT_SETTINGS`, two fields differ: maxIter and
featureSubsetStrategy. Source/test changes only add a third explicit fixed
selection and its pinned saved reference; the original commands retain their
identities and occupied paths.

Depth 4, learning rate 0.1, maxBins 32, subsamplingRate 0.5, maxMemoryInMB 128,
seed 42, positive/negative fitting weights 2.0/1.0 and all other effective defaults
remain unchanged. The runner copied production settings and fitted a fresh
classifier; it did not extend GBT30 or introduce early stopping.
[inputs.json](evidence_60iter/inputs.json) records complete parameters and defaults.

Exactly this command was executed once from the clean committed repository root:

```sh
python3 -B studies/gbt_v1/launch.py --execute --candidate gbt-all-60iter
```

**Launcher exit 0; worker exit 0; full-data classifier fits started: 1; retries: 0.**
`attempt-003/` was absent and created only by the supervisor. No overwrite,
resume, replacement attempt, direct worker launch, sampling, reduced evaluation,
or budget extension occurred.

| Timing / allowance | GBT30 reference | GBT60 |
| --- | ---: | ---: |
| Fit seconds | 4790.826843599993 | 9417.098940163996 |
| Complete attempt seconds | 5280.869890715025 | 9951.583586491994 |
| Hard cap / final termination allowance | 14400 / 60 | 14400 / 60 |

GBT60 fit: **2 h 36 min 57.099 s**; total: **2 h 45 min 51.584 s**. Total time
includes preparation, staging, full fit, complete evaluation, serialization,
preservation checks, source cleanup and shutdown. Preparation handoff charged
0.19114026401075535 s; only durations crossed host/container clock boundaries.
The actual container worker process group remained bounded. These measured times
do not alter the cap or qualification.

America/Chicago start/finish: **2026-10-22T13:51:37.776555-05:00 – 2026-10-22T16:37:29.168027-05:00**
(UTC 2026-10-22T18:51:37.776555 through 21:37:29.168027).

Accepted fitted preprocessing `PipelineModel_196025fd5a4b` was loaded without
refitting. All **8,946,370** training rows, 2023-01-01–2024-04-30, and
**2,418,739** validation rows, 2024-05-01–2024-08-31, were preserved, including
duplicate multiplicity. The existing runner validated actual populations, schema, feature
order/version (`m7-flighttime-v2`, schema 3, 50 predictors, 784 prepared dimensions),
and split/date agreement. The selection manifest declares 11,365,109 rows; its
claim was reconciled by the runner. No test-period inputs or evaluation were used.

Pinned image remained
`sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`,
Spark 3.5.9 / Python 3.10.12 / x86_64, Docker context `default` and inherited
master. Resources stayed two total cores, one core per executor, 640m executor /
768m driver, shuffle partitions 4, max file partition 67,108,864 bytes, batch size
1,000, 36 prepared-training partitions and DISK_ONLY persistence. No builds,
pulls, installs, resource changes or environment switches occurred.

## Qualification and monthly ranking

Saved full-precision values were recalculated with existing `monthly_comparison`
in pinned Python 3.10.12, without Spark, baseline scoring or retraining. It exactly
reproduced supervisor qualification. LR mean is **0.4280983226531411** and its
+0.005 target is **0.4330983226531411**. Decisions use paired changes and full
precision; displayed cells round to nine decimals. No pooled PR-AUC, average
precision or extra rule was substituted.

| LR qualification criterion | Actual | Result |
| --- | ---: | --- |
| Mean monthly PR-AUC gain >= 0.005 | -0.0006668483828393135 | FAIL |
| Mean top-10% recall change >= -0.01 | -0.00009408345789172629 | PASS |
| Mean top-20% recall change >= -0.01 | +0.000906637302573024 | PASS |
| Complete execution, one fit, four months and all checks | Passed; successful shutdown | PASS |

| Month | LR PR-AUC | GBT30 PR-AUC | GBT60 PR-AUC | Change vs LR | Change vs GBT30 | GBT60 ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-05 | 0.432454907 | 0.424094552 | 0.426503513 | -0.005951394 | +0.002408961 | 0.687439396 |
| 2024-06 | 0.403536552 | 0.399533583 | 0.404524584 | +0.000988032 | +0.004991001 | 0.688208040 |
| 2024-07 | 0.470587237 | 0.470524123 | 0.475357986 | +0.004770749 | +0.004833863 | 0.697869706 |
| 2024-08 | 0.405814595 | 0.399999331 | 0.403339814 | -0.002474781 | +0.003340483 | 0.698109773 |
| Equal-month mean | 0.428098323 | 0.423537897 | 0.427431474 | -0.000666848 | +0.003893577 | 0.692906729 |

Worst PR-AUC change vs LR is **-0.005951393714163977 (May)**; worst vs GBT30
is **+0.0024089605577808926 (May)**. GBT30's saved equal-month mean is
**0.4235378972669571**. Its secondary comparison has no qualification flag.
[comparison.json](evidence_60iter/comparison.json) contains full-precision paired
monthly changes, means and worst changes against both references.

## Threshold and operating metrics

The complete-validation 101-step inclusive grid selected **36/100 = 0.36**.
Independent exact-rational F1 verification gives **224251/461258**; step 36 is
the unique optimum. The selected value happens to equal GBT30's threshold; it
was selected independently, with the existing lower-step tie rule, rather than
inherited. One threshold is used for all monthly confusion metrics.
[thresholds.json](evidence_60iter/thresholds.json) preserves every step's counts.

| Overall May–August at 0.36 | Full-precision value |
| --- | ---: |
| rows | 2418739 |
| precision | 0.3697052427843915 |
| recall | 0.7097696147162992 |
| f1 | 0.48617259754844355 |
| roc_auc | 0.6922487439191897 |
| pr_auc | 0.42690074820465346 |
| prevalence | 0.26125100723972283 |
| predicted_positive_rate | 0.5015563895070944 |

Overall confusion counts: TP **448502**, FP **764632**, FN **183396**,
TN **1022209**. Pooled overall PR-AUC is descriptive. Existing unweighted
natural-prevalence areaUnderPR/ROC evaluators retain `numBins=0`.

| Month at 0.36 | Rows | Precision | Recall | F1 | Prevalence | Alert rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-05 | 599226 | 0.389141271 | 0.634290733 | 0.482354863 | 0.264134066 | 0.430532053 |
| 2024-06 | 601272 | 0.344391164 | 0.747674147 | 0.471569543 | 0.249379648 | 0.541403890 |
| 2024-07 | 613938 | 0.396624437 | 0.766893025 | 0.522843058 | 0.296226981 | 0.572769563 |
| 2024-08 | 604303 | 0.347257997 | 0.680675824 | 0.459893660 | 0.234670356 | 0.459987788 |
| Equal-month mean | — | 0.369353717 | 0.707383432 | 0.484165281 | 0.261102763 | 0.501173324 |

## Monthly fixed alert budgets

Capacity remains `rows * percent // 100`, score descending then flight ID
ascending, with fractional mass for indistinguishable boundary groups and no
label-based tie breaking. Duplicates retain multiplicity. The candidate's saved
months report zero duplicate score/ID groups and fractional boundary groups.
Budget metrics are independent of the selected threshold.

**Top 10%**

| Month | Capacity | Precision | P change vs LR | P change vs GBT30 | Recall | R change vs LR | R change vs GBT30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-05 | 59922 | 0.510780682 | -0.005473783 | +0.001685525 | 0.193377391 | -0.002072329 | +0.000638126 |
| 2024-06 | 60127 | 0.482329070 | +0.006669217 | +0.007683736 | 0.193410917 | +0.002674314 | +0.003081130 |
| 2024-07 | 61393 | 0.563484436 | +0.003974394 | +0.005538091 | 0.190218019 | +0.001341655 | +0.001869519 |
| 2024-08 | 60430 | 0.488482542 | -0.005444316 | +0.003061393 | 0.208155868 | -0.002319973 | +0.001304544 |
| Equal-month mean | — | 0.511269182 | -0.000068622 | +0.004492186 | 0.196290549 | -0.000094083 | +0.001723330 |

**Top 20%**

| Month | Capacity | Precision | P change vs LR | P change vs GBT30 | Recall | R change vs LR | R change vs GBT30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-05 | 119845 | 0.462897910 | -0.002736868 | +0.003270892 | 0.350501655 | -0.002072329 | +0.002476686 |
| 2024-06 | 120254 | 0.437149700 | +0.002594508 | +0.003550817 | 0.350588549 | +0.002080763 | +0.002847711 |
| 2024-07 | 122787 | 0.518914869 | +0.007207603 | +0.001848730 | 0.350347785 | +0.004866247 | +0.001248179 |
| 2024-08 | 120860 | 0.436604336 | -0.001464504 | +0.003657124 | 0.372098271 | -0.001248131 | +0.003116803 |
| Equal-month mean | — | 0.463891704 | +0.001400185 | +0.003081891 | 0.355884065 | +0.000906637 | +0.002422345 |


Mean budget-recall changes vs GBT30 are **+0.0017233295490793263** (10%) and
**+0.0024223445608579502** (20%). These descriptive gains do not replace the LR
qualification reference.

| Metric | Worst monthly change vs LR | Worst monthly change vs GBT30 |
| --- | ---: | ---: |
| pr_auc | -0.005951393714163977 | 0.0024089605577808926 |
| roc_auc | 0.001014565253619204 | 0.0023104595607702505 |
| precision | -0.025728800085548786 | -0.0023171540481941033 |
| recall | -0.008143403073708533 | -0.01360913846698164 |
| f1 | 0.0007542749919642877 | -0.00046236728004594996 |
| prevalence | 0.0 | 0.0 |
| predicted_positive_rate | -0.019158286341617647 | -0.014225016938517354 |
| top-10% precision | -0.005473782584025866 | 0.0016855245152030962 |
| top-10% recall | -0.0023199729218965903 | 0.0006381258055548622 |
| top-20% precision | -0.002736868455087804 | 0.0018487299144045455 |
| top-20% recall | -0.0020723293487326377 | 0.0012481785940120504 |

Worst changes are descriptive, not additional qualification conditions. Full
monthly operating-metric changes remain in the comparison evidence.

## Tests, verification and preservation

Before committing/executing, focused tests ran in the unchanged pinned environment:

```sh
docker compose exec -T -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/tmp/gbt60-fixtures/source/project:/workspace/src:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip \
  -w /workspace spark-notebook python3 -B -m pytest -q -p no:cacheprovider \
  /tmp/gbt60-fixtures/source/project/studies/gbt_v1/test_candidate.py \
  /tmp/gbt60-fixtures/source/project/studies/lr_v1/test_metrics.py \
  /workspace/tests/unit/test_experiment_selection.py \
  /workspace/tests/unit/test_model_contract.py
```

**86 passed, 1 existing PySpark deprecation warning, 117.39 seconds, exit 0.**
Three tiny generated-data classifier fits (10/30/60 trees) verify API/reload
wiring; they are distinct from the one authorized full-data fit. Tests do not read
warehouse rows. Coverage includes immutable settings, fixed output/plan/reference
selection, training-only fitting, validation-only evaluation, exact F1 and capacity
semantics, schema/multiplicity/threshold checks, completion/failure rejection,
blocking timeout interruption and deliberately incompatible clock origins.
No full pipeline/test suite was run. See [tests.json](evidence_60iter/tests.json).

All five runner checks passed: parameters/defaults/dimensions; complete-validation
score multiset equality before/after serialization; saved prediction schema,
lineage/label population and threshold decisions; all four months; unchanged
inputs. Saved metadata confirms 60 trees, 784 features and the expected GBT class.
Independent saved-record verification passed **31 assertions** covering completion,
exit/fit count/shutdown, digests, effective-parameter changes, model/prediction
inventories, exact F1, population sums, budgets, means and comparison helpers.
No extra Spark scoring or reference evaluation was used for report verification.

[completion.json](evidence_60iter/completion.json) is authoritative and matches
final [attempt.json](evidence_60iter/attempt.json) byte-for-byte. Original
[worker results](evidence_60iter/results.json) intentionally retain
`pending_termination`; generated records were never rewritten. Results SHA-256:
`9e4ef9de9fc6deb58b1de201088a5f63d3be9a8f7e0887b325a6b68545c03870`.
All ten published generated JSON records are verbatim copies. See
[verification.json](evidence_60iter/verification.json) and
[provenance.json](evidence_60iter/provenance.json).

Targeted checks found **all 397 protected historical/artifact/configuration files
unchanged**, including accepted models, both prior attempts, previous reports and
evidence, and the prefix diagnostic. All three historical output file sets and
all **97 selection-file metadata records** remain unchanged. The runner also
verified all 47 selection Parquet parts plus `_SUCCESS` against the frozen
inventory before and after execution. Permitted study runner/launcher/test edits
are explicitly excluded from the protected inventory. No raw or whole-warehouse
hash scan occurred. [preservation.json](evidence_60iter/preservation.json) retains
the initial inventory and reconciliation.

The owned application `app-20261022185141-0001` reached **FINISHED**; no active
applications, owned worker/driver processes or staged source directories remained.
All four existing project services were initially stopped, started only for this
task and restored to stopped. Unrelated applications and the existing LR worktree
were preserved.

Original local artifacts remain under
`data/studies/gbt-improvement-v1/attempt-003/`: supervisor records at its root,
candidate JSON, `classifier/` and `validation-predictions/` under
`gbt-all-60iter/`, and `worker.log` at the attempt root. Models, prediction Parquet,
warehouse data and large logs remain ignored/local. References are saved LR
`studies/lr_v1/evidence/diagnostics/results.json` and saved GBT30
`studies/gbt_v1/evidence_30iter/results.json`; provenance retains their digests.
Accepted selection identity remains `production-5936316dd7764ed79b049ae10ee1ca18`.

`git diff --check` and staged scope checks passed. Delivery uses a separate results
commit and normal push to `github` → `tonydada11/airline-spark`, branch
`study/gbt-improvement-v1`. No promotion, main merge, force-push, history rewrite,
additional candidate or next-experiment selection occurred.
