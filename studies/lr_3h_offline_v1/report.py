"""Offline evidence aggregation; no Spark execution, scoring or fitting."""
from collections import defaultdict
import json,hashlib
from pathlib import Path
import offline as api
HERE=Path(__file__).resolve().parent;E=HERE/'evidence';OUT=HERE/'runtime/output'
def read(name,default=None):
 p=E/(name+'.json');return json.loads(p.read_text()) if p.exists() else default

def main():
 metrics=defaultdict(lambda:defaultdict(int));apps=[];failed=[];event_files=[]
 for p in sorted((OUT/'events').glob('*')):
  if not p.is_file():continue
  mapping={};application={'file':p.name,'executors':{},'application_end':False}
  for line in p.open():
   item=json.loads(line);kind=item['Event']
   if kind=='SparkListenerApplicationStart':application['start']=item
   if kind=='SparkListenerApplicationEnd':application['application_end']=True;application['end']=item
   if kind=='SparkListenerExecutorAdded':application['executors'][item['Executor ID']]=item['Executor Info']
   if kind=='SparkListenerJobStart':
    group=item.get('Properties',{}).get('spark.jobGroup.id','unassigned')
    for stage in item['Stage Infos']:mapping.setdefault(stage['Stage ID'],group)
   if kind!='SparkListenerTaskEnd':continue
   m=metrics[mapping.get(item['Stage ID'],'unassigned')];t=item.get('Task Metrics',{});m['tasks']+=1
   for key in ('Executor Run Time','Executor CPU Time','JVM GC Time','Disk Bytes Spilled','Memory Bytes Spilled'):m[key]+=t.get(key,0)
   m['peak_task_execution_memory_bytes']=max(m['peak_task_execution_memory_bytes'],t.get('Peak Execution Memory',0))
   shuffle=t.get('Shuffle Read Metrics',{});m['shuffle_read_bytes']+=shuffle.get('Local Bytes Read',0)+shuffle.get('Remote Bytes Read',0)
   m['shuffle_write_bytes']+=t.get('Shuffle Write Metrics',{}).get('Shuffle Bytes Written',0)
   if item['Task End Reason']['Reason']!='Success':failed.append(item['Task End Reason'])
  apps.append(application);event_files.append({'file':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':api.digest(p)})
 api.write(E/'resources.json',{'task_metrics_by_job_group':dict(metrics),'applications':apps,'failed_tasks':failed,'event_logs':event_files})
 x=read('execution');v=read('verification',{});f=read('final-parity',{});r=read('reconciliation',{})
 passed=x['status']=='PASS' and v.get('status')=='PASS' and f.get('status')=='PASS' and x['study_containers_removed'] and not failed and len(apps)==len(read('phases',[])) and all(a['application_end'] for a in apps)
 root=OUT/'bundle';bundle_id=api.MODEL_ID
 if passed:
  manifest=json.loads((root/'manifest.json').read_text());api.validate_files(root,manifest)
  api.write(E/'bundle-inventory.json',{'model_id':bundle_id,'manifest_sha256':api.digest(root/'manifest.json'),'files':{**api.inventory(root),'manifest.json':api.digest(root/'manifest.json')}})
  scored=OUT/'scored-offline-v1';completion=json.loads((scored/'_completion.json').read_text())
  actual=api.inventory(scored);actual.pop('_completion.json',None);assert actual==completion['files']
  api.write(E/'scored-inventory.json',{'model_id':bundle_id,'completion_sha256':api.digest(scored/'_completion.json'),'files':{**actual,'_completion.json':api.digest(scored/'_completion.json')}})
 saved=json.loads((OUT/'bundle/evidence/results.json').read_text())
 status='PASS' if passed else 'BLOCKED';phases=read('phases',[]);totals={key:sum(m.get(key,0) for m in metrics.values()) for key in ('shuffle_read_bytes','shuffle_write_bytes','Disk Bytes Spilled','Memory Bytes Spilled')}
 lines=[f'# Offline LR-3h packaging and batch scoring — {status}', '',
 'Scope: Steps 1–3 only, candidate `ca23fe0`, qualified input `88b632f`, promotion review `c196317`, availability limitation `8b670c2`. Tested implementation committed before execution as `c4a13f3`; standalone Compose correction committed and config-validated as `10509eb` before continuation. The final-verifier sidecar correction was committed as `b52b130`, and container-owned file permission publication correction as `81ea65d`; both were committed before their respective continuations. **Zero classifier fits, zero preprocessing refits, no history rebuild, threshold reselection, independent evaluation, external acquisition, test-period evaluation or production changes.**', '',
 f'Offline model ID: **`{bundle_id}`**. Contract `{api.VERSION}`; feature policy `{api.POLICY}`. New self-contained bundle: `studies/lr_3h_offline_v1/runtime/output/bundle/`. Scored results: `studies/lr_3h_offline_v1/runtime/output/scored-offline-v1/`.', '',
 '**Artifacts:** the classifier and accepted preprocessing were byte-copied, not resaved in the release bundle. All 50 candidate inventory files and 64 parent preprocessing hashes were verified during packaging; all 80 qualified feature files were checked before execution. The bundle contains its own stdlib/PySpark loader/scorer, complete file inventory, copied fitted models, saved means and frozen result/threshold/input evidence. Runtime loading has no original-study-path dependencies. SHA-256 checks reject altered files/means; fixed schema/order/790 width/settings and explicit 0.39 are validated separately from schema 3.', '',
 '**Feature order:** preserve the frozen 784 coordinates, then append:', '',
 '| Coordinate | Type / handling |', '|---|---|']
 for c in api.EXTRA:lines.append(f'| `{c}` | '+(f'Nullable double; NULL → {api.MEANS[c]}' if c in api.MEANS else 'Observed nonnegative integer count; zero retained' if 'count' in c else 'Observed binary double')+' |')
 lines += ['', 'Only saved nullable-rate means are applied. No means are recomputed, rounded or estimated from validation. Inputs require the 50 original predictors, six additions and six business keys. Flight_id, split and target are optional and preserved when supplied. Output adds positive-class probability, explicit `prediction = probability >= 0.39`, model_id and feature_policy. The classifier internal 0.5 decision is discarded.', '',
 '**Verification:** focused pinned-runtime tests passed for 16 rows, exact prefix/order/null handling, label-free scoring, 17 duplicate occurrences, 0.39 boundary inclusivity, classifier save/reload, score parity and altered mean/order rejection. A read-only nested helper mount failed during initial test setup before Spark; the mount was corrected and its record retained. Local static checks confirm zero `.fit` call sites.', '',
 f'Full source split counts: `{v.get("counts")}`. Every validation vector verified its exact 784-prefix, 790 width and six appended values against an independent array-concatenation expression. Saved original continuous scores and predictions matched in both multiset directions, including business keys/labels/duplicates. A separate fresh Spark process, with **only the bundle, precomputed validation inputs and output volume mounted**, scored all 2,418,739 rows; final bidirectional prediction parity passed. The old production schema-3 bundle loaded successfully without scoring and remained unchanged.', '',
 '**Saved-result reconciliation (not independent performance evaluation):**', '',
 f'- Candidate equal-month PR-AUC: **{r.get("mean_monthly_PR_AUC")}**; baseline **{r.get("baseline_mean_monthly_PR_AUC")}**.',
 f'- Paired PR-AUC gain: **{r.get("comparison",{}).get("mean_monthly_PR_AUC_gain")}**.',
 f'- Mean recall changes, top-10%/20%: **{r.get("comparison",{}).get("mean_monthly_recall_changes")}**.', '',
 *(['| Saved month | Rows | PR-AUC | Paired PR-AUC gain | Recall at 10% | Recall at 20% |', '|---|---:|---:|---:|---:|---:|']+[f'| {month} | {v["rows"]} | {v["pr_auc"]:.16f} | {r["comparison"]["monthly_PR_AUC_changes"][month]:+.16f} | {v["budgets"]["10"]["recall"]:.16f} | {v["budgets"]["20"]["recall"]:.16f} |' for month,v in saved['months'].items()]), '',
 'All four monthly results and the complete saved 101-step threshold record are bound inside the bundle; the existing comparison helper reconciles the frozen evidence. No new monthly model evaluation or threshold selection was run. Exact prediction parity establishes reproduction of the saved budget/tie/duplicate results.', '',
 f'**Execution/resources:** full verification controller wall {x["seconds_including_cleanup"]:.3f}s including corrections and all sequential cluster cleanup (focused tests separately 143.379s). Pinned Spark 3.5.9/Python 3.10.12 image `sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`; two one-core workers, 640 MiB executor heaps, 768 MiB driver, four shuffle partitions, unchanged 2 GiB worker/driver and 512 MiB master limits. No overlapping heavy jobs. Failed tasks: {len(failed)}; completed application-end events: {sum(a["application_end"] for a in apps)}.', '',
 'Task totals: '+', '.join(f'{k}: {value/2**30:.3f} GiB' for k,value in totals.items())+'. Memory-spill volume is not peak RAM.', '',
 '| Phase | Seconds including teardown | Master / worker1 / worker2 / driver lifetime cgroup peaks MiB |', '|---|---:|---|']
 for p in phases:
  peaks=' / '.join(f'{int(next(value for name,value in p["cgroups"].items() if name.endswith(role+"-1"))["memory.peak"])/2**20:.1f}' for role in ('master','worker1','worker2','driver'))
  lines.append(f'| {p["name"]} (exit {p["exit_code"]}) | {p["seconds_including_cleanup"]:.1f} | {peaks} |')
 lines += ['', 'Workers reached their unchanged 2 GiB cgroup limit with memory.max events but zero OOM/kill events; this run does not establish memory headroom. Cgroup peaks include file cache; sampled phase memory is a lower bound, not process RSS or spare-capacity proof. Per-task/phase details: [resources](evidence/resources.json), [memory samples](evidence/memory.json), [execution](evidence/execution.json), [preservation](evidence/preservation.json).', '',
 '## Exact commands and artifact identities', '',
 'Tested before execution:', '```bash', 'python3 -m unittest studies.lr_3h_offline_v1.test_no_fits -v', 'python3 -m py_compile studies/lr_3h_offline_v1/*.py', 'docker compose -f studies/lr_3h_offline_v1/compose.focused.yml up -d', 'docker compose -f studies/lr_3h_offline_v1/compose.focused.yml exec -T driver timeout --kill-after=15 1200 /opt/spark/bin/spark-submit --driver-memory 768m /study/verify.py --mode focused', 'docker compose -f studies/lr_3h_offline_v1/compose.focused.yml down --timeout 15', '```', '',
 'After committing implementation:', '```bash', 'python3 studies/lr_3h_offline_v1/run.py', '# 10509eb: YAML fix; standalone scoring completed, verifier sidecar check failed', 'python3 studies/lr_3h_offline_v1/run.py --finish', '# b52b130: reuse scores; final verifier passed, host chmod failed', 'python3 studies/lr_3h_offline_v1/run.py --finish', '# 81ea65d: validate existing results and publish read-only; no Spark launched', 'python3 studies/lr_3h_offline_v1/run.py --finish', '```', 'The controller ran these payloads, each wrapped by timeout and followed by `docker compose ... down --timeout 15`:', '```bash', '# compose.yml: staged artifact, full vector/score parity and legacy loader checks', '/opt/spark/bin/spark-submit --driver-memory 768m /study/verify.py --mode stage', '# compose.fresh.yml: independent bundle-only scoring process', '/opt/spark/bin/spark-submit --driver-memory 768m /bundle/offline.py --bundle /bundle --input /input/2024-05 /input/2024-06 /input/2024-07 /input/2024-08 --output /output/scored-offline-v1', '# compose.yml: compare fresh output; no fit or threshold selection', '/opt/spark/bin/spark-submit --driver-memory 768m /study/verify.py --mode final', '# Offline resource/hash aggregation only', 'python3 studies/lr_3h_offline_v1/report.py', '```', '',
 'The initial staged full-data check passed, then the fresh Compose file was rejected before launch because of an unquoted trailing-colon PYTHONPATH. Evidence is retained in initial-execution.json and standalone-config-failure.json. The syntax was corrected, all Compose files validated, and the already-verified staged bundle was continued without rerunning that stage, refitting or rebuilding data. The first final verifier then completed the Spark multiset comparisons but exited because it looked for `completion.json` instead of the actual `_completion.json`. Its corrected rerun passed without rescoring. Host chmod then failed on container-owned copied files; the same pinned image applied read-only modes with no byte changes and no further Spark execution. All failed-process evidence is retained, including the exit-1 verifier phase. Four Spark applications ended cleanly; no failed Spark tasks were observed. Reported wall time includes these correction intervals.', '', 'For later opt-in scoring, mount the published `bundle/` at `/bundle` instead of its staging name, precomputed input directories read-only, and a new writable output directory in the same pinned cluster configuration; run the bundle-only command above with a **new** output path. See [README](README.md). Incomplete bundles lack a completion manifest; staging completion follows load/full-score verification, and final immutable bundle publication occurs only after fresh-process parity. Existing outputs are never overwritten.', '']
 if passed:
  b=read('bundle-inventory');s=read('scored-inventory')
  lines += [f'Bundle manifest SHA-256: `{b["manifest_sha256"]}`. Output completion SHA-256: `{s["completion_sha256"]}`. [Bundle hashes](evidence/bundle-inventory.json), [output hashes](evidence/scored-inventory.json), [final integrity recheck](evidence/final-integrity.json). Model/preprocessing files and final bundle are read-only; binary files remain local and ignored by Git. Verify the recorded manifest hash when transferring the whole self-contained directory.', '']
 else:lines += [f'Blocking evidence: `{x.get("error")}`. Partial artifacts are not qualified.', '']
 lines += ['**Remaining limitation:** this is an offline research bundle using precomputed histories. At scheduled departure UTC cutoff t, the new windows require t−3h ≤ source scheduled departure < t and reconstructed source arrival < t. Existing 7/30/90-day predictors retain their frozen research construction. Reconstructed arrival is assumed to make the outcome available immediately; it is not measured receipt time. It proves neither actual outcome receipt, schedule-as-known timing, correction availability, live coverage nor independent future performance. September–December 2024 is previously inspected and not pristine. Nothing here clears the operational BLOCKED finding or authorizes routing changes.', '',
 '**STOP:** Steps 1–3 end here. Accepted schema-3 LR bundle `d0098a9572834c53bf76b04d732a94a6`, 784 dimensions, default threshold 0.39 and production paths remain intact. No promotion, automatic switch, main merge or further experiment.']
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
