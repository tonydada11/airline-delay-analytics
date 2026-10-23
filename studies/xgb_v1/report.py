"""Render only observed results; an incomplete candidate never qualifies."""
import json
from pathlib import Path


def generate(root):
    root = Path(root)
    study = root / 'studies/xgb_v1'
    out = root / 'artifacts/experiments/xgb-v1'
    evidence = out / 'execution'
    def read(path, default=None):
        return json.loads(path.read_text()) if path.exists() else default
    status = read(evidence / 'status.json')
    pre = read(out / 'preflight.json', {})
    env = read(out / 'environment.json', {})
    population = read(out / 'population.json', {})
    full = read(out / 'full/results.json') if status['status'] == 'complete' else None
    baseline = read(root / 'studies/lr_v1/evidence/diagnostics/results.json')['validation_months']
    manifest = read(root / 'artifacts/flighttime-v3/models/manifest.json')
    pooled_lr = manifest['selection']['validation_results']['lr']['validation']
    recommendation = ('QUALIFIES FOR REVIEW' if full['comparison']['qualifies'] else 'RETAIN BASELINE') if full else 'INCOMPLETE'
    summary = dict(status=status['status'].upper(), recommendation=recommendation,
                   qualification=full['comparison'] if full else None,
                   baseline_months={m:{k:v[k] for k in ('rows', 'pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'budgets')} for m,v in baseline.items()},
                   baseline_pooled=pooled_lr, baseline_threshold=.39,
                   xgb=full, population=population, environment=env, preflight=pre,
                   execution=status)
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    def n(value):
        return 'N/A' if value is None else f'{value:.12f}'
    def difference(x, y):
        return None if x is None else x-y
    prefix = '../../artifacts/experiments/xgb-v1'
    tests_log = evidence / 'tests.log'
    test_result = tests_log.read_text().strip().splitlines()[-1] if tests_log.exists() else 'Tests unavailable'
    smoke_action = ('The existing single smoke fit was reused; current checks reloaded it, serialized it again, and verified all probabilities/labels and dense/sparse inference.' if pre.get('smoke', {}).get('reused_existing_fit') else 'The single smoke classifier was fitted, saved, reloaded and checked in this execution.')
    lines = ['# XGB-v1 results', '', f'**Execution status: {summary["status"]}. Recommendation: {recommendation}.**', '',
             ('Full-data fit and validation evaluation completed.' if full else f'**No valid full-data performance comparison exists.** Full-data fits started: {status.get("full_classifier_fits_started", 0)}. Complete candidate metrics and selected threshold are unavailable.'), '',
             f'[Full-precision metrics and execution summary]({prefix}/metrics.json). [Reproduction instructions](README.md).', '',
             '**Environment and model.** Spark 3.5.9, Python 3.10.12; study image built offline from the unchanged production image with hash-locked wheels. Driver and both executor hosts use identical versions:', '',
             '| Dependency | Version |', '|---|---|']
    lines += [f'| {key} | {value} |' for key,value in env.get('packages', {}).items()]
    lines += ['', 'Two one-core workers, 768 MiB worker advertisements, **640 MiB executor heaps**, 768 MiB driver heap, one CPU per task, four shuffle partitions. Isolated container caps: workers/driver 2 GiB each; master 512 MiB. No resources increased. Exact runtime and image identity are recorded in the artifact directory.', '',
              'Official `SparkXGBClassifier`: 100 estimators, depth 4, learning rate 0.05, CPU `hist`, seed 42, two workers; columns `features`, `is_arrival_delayed_15m`, `class_weight`. Binary labels select `binary:logistic` automatically because the Spark API does not accept an explicit objective override. Weights: positive 2.0, negative 1.0. All other estimator defaults retained, including `missing=NaN` and disabled sparse optimization. No early stopping or tuning.', '',
              'Compatibility was tested empirically with the exact installed packages. PyArrow 19.0.1 is outside the older `<13` range documented for pandas API/Spark Connect in the [Spark 3.5.9 installation table](https://spark.apache.org/docs/3.5.9/api/python/getting_started/install.html); this report claims only the exercised Spark XGBoost paths, not broad official compatibility.', '',
              '**Dataset and preprocessing.** Accepted bundle `d0098a9572834c53bf76b04d732a94a6`; preprocessing `PipelineModel_196025fd5a4b` reused without fitting, **50 input features / 784 prepared dimensions**. Full feature validity, manifest, chronological bounds, and monthly positive counts were checked.', '',
              '| Split | Dates | Rows | Positive rows |', '|---|---|---:|---:|']
    lines += [f'| {p["split"]} | {p["start"]}–{p["end"]} | {p["rows"]:,} | {p["positives"]:,} |' for p in population.get('populations', [])]
    lines += ['', '**Validation comparison.** LR values are frozen evidence. N/A is unavailable, never zero. Display rounding is not used in qualification.', '',
              '| Month | LR ROC | XGB ROC | LR PR | XGB PR | Paired PR difference |', '|---|---:|---:|---:|---:|---:|']
    for month, b in {**baseline, 'Pooled': pooled_lr}.items():
        c = (full['pooled'] if month == 'Pooled' else full['months'][month]) if full else {}
        lines.append(f'| {month} | {n(b["roc_auc"])} | {n(c.get("roc_auc"))} | {n(b["pr_auc"])} | {n(c.get("pr_auc"))} | {n(difference(c.get("pr_auc"),b["pr_auc"]))} |')
    mean = sum(m['pr_auc'] for m in full['months'].values())/4 if full else None
    lines += [f'| Equal-month mean | — | — | **0.4280983226531411** | {n(mean)} | {n(difference(mean,.4280983226531411))} |', '',
              f'LR threshold **0.39**; XGB threshold **{n(full["threshold"] if full else None)}**. Existing inclusive 0.00–1.00 grid maximizes exact rational pooled F1, then prefers the lower threshold.', '',
              '| Month | LR precision / recall / F1 | XGB precision / recall / F1 |', '|---|---|---|']
    for month,b in {**baseline,'Pooled':pooled_lr}.items():
        c = (full['pooled'] if month == 'Pooled' else full['months'][month]) if full else {}
        lines.append('| '+month+' | '+' / '.join(n(b[k]) for k in ('precision','recall','f1'))+' | '+' / '.join(n(c.get(k)) for k in ('precision','recall','f1'))+' |')
    lines += ['', '| Month / budget | LR precision | LR recall | XGB precision | XGB recall | Δ precision | Δ recall |', '|---|---:|---:|---:|---:|---:|---:|']
    for month,b in baseline.items():
        for budget in ('10','20'):
            bb=b['budgets'][budget]
            cb=full['months'][month]['budgets'][budget] if full else {}
            values=[bb['precision'],bb['recall'],cb.get('precision'),cb.get('recall'),difference(cb.get('precision'),bb['precision']),difference(cb.get('recall'),bb['recall'])]
            lines.append(f'| {month} / {budget}% | '+' | '.join(n(v) for v in values)+' |')
    lines += ['', 'Metrics reuse unweighted Spark AUC (`numBins=0`) and the LR study budget utility: `floor(budget × monthly rows)`, score descending then flight ID ascending, fractional selection within indistinguishable boundary duplicates. Threshold choice does not imply improved ranking.', '', '**Compute and blocker.**']
    if not full and pre.get('memory', {}).get('blocked'):
        memory=pre['memory']
        lines += [f'The installed adapter materializes and retains dense float32 batches before QuantileDMatrix: **8,946,370 × 784 × 4 = {memory["dense_bytes"]:,} bytes (26.129 GiB)**, or 13.065 GiB per worker at an even split. Current host RAM is {memory["host_physical_bytes"]/2**30:.3f} GiB; total swap is {memory["host_swap_bytes"]/2**30:.3f} GiB. Input alone exceeds both combined, excluding JVM, Arrow, labels, weights and matrix overhead. Sparse optimization would reinterpret valid zeros as missing, changing the frozen contract. This is a verified lower bound, not an observed full-fit OOM.', '',
                  'Minimum prerequisite for any future full attempt: an explicitly authorized change in available memory beyond this input lower bound, plus separately assessed overhead. A sufficient total cannot be inferred from the smoke fit. That would depart from the current fixed-resource instruction, so no such change was made.', '']
    phase=read(evidence/'preflight-execution.json', {})
    lines += [f'Current supervisor elapsed: **{status["elapsed_seconds"]:.3f} s**; Spark preflight: **{phase.get("elapsed_seconds",0):.3f} s**. Original smoke training: **{pre.get("smoke",{}).get("fit_seconds",0):.3f} s**, 256 synthetic rows. {smoke_action} Its weighted constant probability is 0.4000000059604645 (analytic value 0.4). This is not flight-model performance.', '',
              f'Full-data fit time: {n(full["fit_seconds"] if full else None)}; evaluation time: {n(full["evaluation_seconds"] if full else None)}. The complete full attempt has a 7,200-second cap, including preparation, fitting, serialization and evaluation, using host supervision and `timeout --signal=TERM --kill-after=10s 7170`. No automatic fit retries. '+('The resource gate prevented entering that attempt.' if pre.get('memory',{}).get('blocked') else 'See execution status for the actual outcome.'), '',
              '| Container | Lifetime peak MiB | OOM kills |', '|---|---:|---:|']
    for name,c in read(evidence/'cgroups.json', {}).items():
        if 'memory.peak' in c:
            events=dict(line.split() for line in c['memory.events'].splitlines())
            lines.append(f'| {name} | {int(c["memory.peak"])/2**20:.1f} | {events["oom_kill"]} |')
    lines += ['', 'Peaks include cache and are not necessarily simultaneous.', '', '**Commands and verification.** `python3 studies/xgb_v1/run.py` built the image, ran targeted tests and the Spark preflight, preserved production, stopped study containers and generated this report. Exact argv and exit codes:', '']
    for name in ('build', 'tests', 'preflight-execution'):
        r=read(evidence/(name+'.json'), {})
        if r:
            lines += [f'- `{name}`: exit **{r["exit_code"]}**; [command and log record]({prefix}/execution/{name}.json).']
    preservation=read(evidence/'preservation.json', {})
    lines += ['', f'**{test_result}.** [Test log]({prefix}/execution/tests.log); [pip check]({prefix}/execution/pip-check.json); [runtime]({prefix}/environment.json); [effective fitted parameters]({prefix}/smoke-booster-config.json). Tests cover metric/tie calculations, baseline evidence, the memory bound, and distributed execution evidence. Runtime checks cover all feature/population contracts, label/weight alignment, missing-value interpretation and model serialization.', '',
              f'Preservation: {preservation.get("protected_files",0)} files checked by SHA-256 (unchanged: {preservation.get("sha256_unchanged")}); {preservation.get("artifact_metadata_files",0)} warehouse/raw/model/metric/acceptance files checked by size and mtime (unchanged: {preservation.get("artifact_metadata_unchanged")}). Metadata checks do not hash content. Test-period contents were not opened for these checks. Production containers unchanged: {preservation.get("production_containers_unchanged")}. Only selection data and accepted models were mounted read-only; no test data was scored. No production pipeline, promotion or push.', '',
              'Files changed: study `run.py`, `experiment.py`, `compose.yml`, `RESULTS.md`; added `Dockerfile`, `report.py`, `README.md`. Existing targeted tests and study lock retained. Generated outputs/logs now reside solely in `artifacts/experiments/xgb-v1/` for this execution: `execution/`, `build/`, `smoke-model/`, `smoke-reload-check/`, `metrics.json`, and environment/population records. Prior evidence and reports were preserved byte-for-byte in `prior/`; original ignored data artifacts remain untouched. '+('Full candidate artifacts are under `full/`.' if full else 'Any partial candidate artifacts are excluded from comparison.'), '',
              'Repository identity: worked in the supplied checkout; README names `tonydada11/airline-spark`, while the configured origin is a Wisconsin GitLab URL. No remote was changed. Historical setup failures and the prior supervisor logging correction remain in `prior/`; the current reuse/fit counts are recorded in the summary.', '',
              f'**Decision: {recommendation}.** Qualification requires mean monthly PR-AUC gain ≥0.005 and mean recall change ≥−0.01 at both budgets. '+('The measured comparison is in metrics.json. Production remains unchanged.' if full else '**Criteria cannot be evaluated. No full-data performance conclusion is available.** The accepted LR remains at threshold 0.39.'), '']
    (study/'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    generate(Path(__file__).resolve().parents[2])
