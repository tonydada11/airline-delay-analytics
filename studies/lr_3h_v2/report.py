"""Offline result rendering; never starts Spark or fits a model."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;E=HERE/'evidence'
def load(name,default=None):
 p=E/(name+'.json');return json.loads(p.read_text()) if p.exists() else default

def main():
 execution=load('execution');r=load('results',{});resources=load('resources',{});population=load('population',[])
 complete=r.get('status')=='complete' and execution['exit_code']==0 and execution['preserved'] and execution['study_containers_removed']
 status=('PASS' if r['comparison']['qualifies'] else 'FAIL') if complete else 'BLOCKED'
 baseline=json.loads((HERE.parents[1]/'data/studies/lr-improvement-v1/diagnostics/results.json').read_text())['validation_months']
 files=[]
 for p in sorted((HERE/'runtime/output').rglob('*')):
  if p.is_file() and 'events' not in p.parts:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
   files.append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 (E/'artifact-hashes.json').write_text(json.dumps(files,indent=2)+'\n')
 contract=load('append-contract',{})
 lines=[f'# Controlled six-feature LR experiment — {status}', '',
 'Reference `88b632f`; implementation committed before execution as `c6921d4`. Branch `study/lr-3h-v2` preserves the prior blocked-run commits. Production code/artifacts and resource limits were preserved; the accepted baseline was never refit or rescored, and no test-period rows were read.', '',
 f'**Execution:** one exclusive full-data attempt, {execution["wall_seconds_including_cleanup"]:.3f}s including cleanup; cap 14,400s. Exit {execution["exit_code"]}. Study containers/network removed: {execution["study_containers_removed"]}. Original files/containers preserved: {execution["preserved"]}.', '',
 '**Preflight:** existing Airline host Docker daemon and pinned image verified; all 80 saved feature file hashes, 64 preprocessing hashes and frozen baseline-evidence hash matched. Focused tests passed on actual Spark 3.5.9/Python 3.10.12: imports, 16-row frozen 784d preprocessing, null-rate means, exact six-coordinate order/790d width, preprocessing save/reload, distributed duplicate-budget fixture. Three local protocol tests also passed. The first fixture had an unstable unmaterialized limit; its failure evidence is retained, the fixture was cached/fixed and tested before commit. Preflight started zero classifier fits.', '',
 f'**Population:** {len(population)}/20 months passed exact bidirectional exceptAll over every original column, including business keys and labels, plus feature schema/domain checks. Required final counts: 8,946,370 train and 2,418,739 validation. Recorded final counts: {r.get("counts", "not reached")}. Full prepared rows verify exact 784-value original prefixes and total width 790 before fitting.', '',
 '**Features:** frozen accepted PipelineModel reused without fit; only the origin/destination nullable rate means use training rows. Appended order: `origin_delay_rate_3h`, `origin_flight_count_3h`, `origin_history_missing_3h`, `destination_delay_rate_3h`, `destination_flight_count_3h`, `destination_history_missing_3h`. Original types are nullable double rates, bigint counts and double 0/1 missing indicators. Qualified historical values were reused verbatim under the accepted zero-latency availability assumption; no histories were rebuilt.', '',
 f'Training-only rate means: `{contract.get("training_only_rate_means")}`. LR settings: `{contract.get("settings")}`; class weights 2:1. All defined estimator parameters were compared with the accepted LR contract. No random step was introduced; Spark LR has no seed parameter (frozen reproducibility seed 42).', '']
 if complete:
  c=r['comparison']
  lines += [f'**Qualification: {status}.** Mean monthly PR-AUC {r["mean_monthly_PR_AUC"]:.16f}, frozen baseline 0.4280983226531411; paired gain **{c["mean_monthly_PR_AUC_gain"]:+.16f}**, required ≥0.005. Mean recall changes: top-10% **{c["mean_monthly_recall_changes"]["10"]:+.16f}**, top-20% **{c["mean_monthly_recall_changes"]["20"]:+.16f}**; both required ≥−0.01.', '',
   f'Fit elapsed: {r["fit_seconds"]:.3f}s. Classifier fits started: {r["classifier_fits_started"]}. Candidate threshold: **{r["threshold"]}**, selected once on pooled May–August validation using the existing 101-step grid, exact F1 comparison and lower-step tie break. Production threshold remains **0.39**.', '',
   '| Month | Rows | PR-AUC | Baseline PR-AUC | Paired gain | Recall @10% | Δ recall @10% | Recall @20% | Δ recall @20% |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for month,m in r['months'].items():
   b=baseline[month]
   lines.append(f'| {month} | {m["rows"]:,} | {m["pr_auc"]:.12f} | {b["pr_auc"]:.12f} | {m["pr_auc"]-b["pr_auc"]:+.12f} | {m["budgets"]["10"]["recall"]:.12f} | {m["budgets"]["10"]["recall"]-b["budgets"]["10"]["recall"]:+.12f} | {m["budgets"]["20"]["recall"]:.12f} | {m["budgets"]["20"]["recall"]-b["budgets"]["20"]["recall"]:+.12f} |')
  lines += ['', 'Monthly JSON evidence contains exact confusion matrices, PR/ROC AUC, precision/recall and budget capacities. Ranking is descending score then ascending flight_id; identical score/id boundary rows retain equal fractional selection mass. No deduplication. Candidate model parameters/790d width were verified after reload; all 2,418,739 validation score rows matched exactly before/after reload, and saved predictions matched full business-key/label multiplicity.', '']
 else:lines += [f'Blocking evidence: `{execution.get("error")}`. See build.log and partial evidence; no quality conclusion is justified.', '']
 metrics=resources.get('task_metrics_by_phase',{});keys=('shuffle_read_bytes','shuffle_write_bytes','Disk Bytes Spilled','Memory Bytes Spilled')
 totals={k:sum(m.get(k,0) for m in metrics.values()) for k in keys}
 lines += ['**Resources:** unchanged pinned image `sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`; two one-core executors, 640 MiB heaps, 768 MiB driver, four shuffle partitions; 2 GiB worker/driver and 512 MiB master cgroup limits. Task/stage retries disabled.', '',
 'Task totals: '+', '.join(f'{k}: {v/2**30:.3f} GiB' for k,v in totals.items())+f'. Failed tasks: {len(resources.get("failed_tasks",[]))}. Memory-spill bytes are deserialized spill volume, not RAM usage.', '',
 '| Container | Lifetime peak MiB | OOM / kills |', '|---|---:|---:|']
 for name,v in execution['cgroups'].items():
  events=dict(x.split() for x in v['memory.events'].splitlines())
  lines.append(f'| {name} | {int(v["memory.peak"])/2**20:.2f} | {events["oom"]} / {events["oom_kill"]} |')
 lines += ['', 'Phase-tagged memory samples and shuffle/spill/task measurements: [resources.json](evidence/resources.json). Cgroup peaks include cache; per-phase sampled maxima are lower bounds. No spare-memory guarantee is inferred.', '',
 '## Commands and provenance', '', '```bash', 'python3 -m unittest studies.lr_3h_v2.test_protocol -v', 'python3 -m py_compile studies/lr_3h_v2/*.py', 'docker compose -f studies/lr_3h_v2/compose.preflight.yml up -d', 'docker compose -f studies/lr_3h_v2/compose.preflight.yml exec -T driver timeout --kill-after=15 1200 /opt/spark/bin/spark-submit --driver-memory 768m /study/preflight.py', 'docker compose -f studies/lr_3h_v2/compose.preflight.yml down --timeout 15', 'python3 studies/lr_3h_v2/run.py', '# Supervisor invokes:', 'docker compose -f studies/lr_3h_v2/compose.yml exec -T driver timeout --signal=TERM --kill-after=15 14100 /opt/spark/bin/spark-submit --driver-memory 768m /study/experiment.py', '# Always clean up:', 'docker compose -f studies/lr_3h_v2/compose.yml down --timeout 15', '# Offline summaries only:', 'python3 studies/lr_3h_v2/summarize.py', 'python3 studies/lr_3h_v2/report.py', '```', '',
 'Input identities: [input verification](evidence/input-verification.json). Saved model, score and prediction file identities: [artifact hashes](evidence/artifact-hashes.json). Exact model/validation outputs remain in `studies/lr_3h_v2/runtime/output/`, with hashes committed and binary files ignored. Complete optimizer objective history, selected threshold grid and all four monthly results are retained. Full Spark event-log hashes appear in resource evidence. [Execution/cleanup](evidence/execution.json); [preservation](evidence/preservation.json); [runtime preflight](evidence/preflight.json); [population](evidence/population.json).', '',
 '**Stop:** no additional fit, baseline scoring, test-period evaluation, model promotion or main-branch merge. Production remains LR at 0.39.']
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
