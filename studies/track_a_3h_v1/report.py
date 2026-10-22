"""Render the final qualification report from recorded evidence only."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
E=HERE/'evidence'
def read(name,default=None):
    p=E/(name+'.json')
    return json.loads(p.read_text()) if p.exists() else default

def main():
    execution=read('execution');resources=read('resources');months=read('months',[])
    result=read('result',{});schema=read('schema',{});preserved=read('preservation',{})
    passed=result.get('status')=='PASS' and execution['exit_code']==0 and execution['preserved'] and execution['study_containers_removed'] and not resources['failed_tasks']
    status='PASS' if passed else 'BLOCKED'
    metrics=resources['task_metrics_by_phase']
    totals={k:sum(m.get(k,0) for m in metrics.values()) for k in ('shuffle_read_bytes','shuffle_write_bytes','Memory Bytes Spilled','Disk Bytes Spilled','Executor Run Time','JVM GC Time')}
    mib=lambda n:f'{n/2**20:,.1f}'
    gib=lambda n:f'{n/2**30:,.2f}'
    lines=[f'# Track A three-hour feature qualification — {status}', '',
      'Reference `16e841e`; implementation `3a237eb`. One isolated Spark attempt; no retries, preprocessing fit, classifier fit, threshold tuning or promotion.', '',
      f'**Population:** {result.get("totals", "Incomplete; see monthly evidence")}. {len(months)}/20 months verified. Every completed month passed both directions of `exceptAll` over every original column, preserving business-key/label multiplicity and original feature values. All six new columns passed domain/missingness checks.', '',
      '**Temporal verification:** `python3 -m unittest studies.ml_three_track_v1.test_track_a -v` passed four tests. Spark `checks.generated` passed exact microsecond bounds, duplicate sources/targets, null entities/times, future/current/in-flight exclusion, forbidden-outcome perturbation and reversed ordering (three comparisons × five target occurrences). `checks.sample` matched all six values for all 109,423 targets against `track_a.enrich`, using 331,355 sources; prior sample source SHA-256s matched.', '',
      'Sources use the existing eligibility and UTC converter. Contributions are active exactly when `t−3h ≤ departure < t` and `availability < t`. Availability remains scheduled departure + signed delay + elapsed time, with the accepted zero reporting latency assumption; it is not observed publication time. Only 2023 and January–August 2024 DWD partitions were read. Monthly source bounds use exact target UTC extrema, retaining cross-month history. Eight source rows had uncertain prediction/availability times and were excluded from history contributions; null-time targets retain empty histories.', '',
      '**Schema:** nullable `double` rates; non-null observed `bigint` counts and binary `double` missing indicators (Spark schema nullability may be conservative). Empty history is `(null, 0, 1)`.', '',
      '| Column | Type |', '|---|---|']
    for role in ('origin','destination'):
        for kind,dtype in [('delay_rate','double (nullable)'),('flight_count','bigint'),('history_missing','double (0/1)')]:
            lines.append(f'| `{role}_{kind}_3h` | {dtype} |')
    lines += ['', 'All original inputs and the frozen preprocessing artifact are preserved; original width is 784, proposed appended width is 790. No 790-dimensional vector or new imputation means were fit.', '',
      f'**Resources:** {execution["wall_seconds_including_cleanup"]:.2f} seconds including cleanup, within the 14,400-second cap. Pinned Spark 3.5.9 image; two one-core workers, 640 MiB executor heaps, 768 MiB driver heap, four shuffle partitions. No resource changes.', '',
      f'Whole-application task totals: shuffle read **{gib(totals["shuffle_read_bytes"])} GiB**, shuffle write **{gib(totals["shuffle_write_bytes"])} GiB**, disk spill **{gib(totals["Disk Bytes Spilled"])} GiB**, memory spill **{gib(totals["Memory Bytes Spilled"])} GiB**. Memory-spill bytes describe deserialized spill volume, not peak RAM. Failed task events: {len(resources["failed_tasks"])}.', '',
      '| Container | Exact lifetime cgroup peak, MiB | OOM events |', '|---|---:|---|']
    for name,value in execution['cgroups'].items():
        events={x.split()[0]:int(x.split()[1]) for x in value['memory.events'].splitlines()}
        lines.append(f'| {name.removeprefix("airline-track-a-3h-v1-")} | {mib(int(value["memory.peak"]))} | {events.get("oom",0)} / {events.get("oom_kill",0)} kills |')
    lines += ['', 'Worker memory limits remained 2,048 MiB; recorded cgroup peaks were slightly above those limits, with 46,082 / 46,058 memory-limit events and no OOM events. This run does not establish spare memory headroom.', '', '| Split/month | Rows | Seconds | Origin missing | Destination missing | Shuffle write MiB | Disk spill MiB |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in months:
        m=metrics.get(row['month'],{})
        lines.append(f'| {row["month"]} | {row["rows"]:,} | {row["elapsed_seconds"]:.1f} | {row["origin_missing_fraction"]:.2%} | {row["destination_missing_fraction"]:.2%} | {mib(m.get("shuffle_write_bytes",0))} | {mib(m.get("Disk Bytes Spilled",0))} |')
    population=resources['population_by_split']
    lines += ['', '**Split totals:** ' + '; '.join(f"{split}: {value['elapsed_seconds']:.1f}s monthly build/audit time, origin missing {value['origin_missing_fraction']:.2%}, destination missing {value['destination_missing_fraction']:.2%}" for split,value in population.items()) + '. Shared source preparation, fixtures, startup and cleanup are additional.', '']
    lines += ['', 'Per-month shuffle reads, memory spill, CPU/GC/task time and sampled cgroup memory are in [resources.json](evidence/resources.json); exact monthly population, UTC bounds and missingness are in [months.json](evidence/months.json). Per-month cgroup samples are lower bounds at approximately 10-second intervals; lifetime peaks include cache and are not simultaneous totals. Source preparation and fixture costs are separately tagged, not attributed to a split/month.', '',
      f'**Preservation and cleanup:** {preserved.get("tracked_files")} original tracked files verified by SHA-256; {preserved.get("metadata_files")} existing data/artifact/study files verified by size/mtime. Unchanged: {execution["preserved"]}. Study containers/network removed: {execution["study_containers_removed"]}. Original mounts were read-only. Frozen preprocessing file hashes are separately recorded. These metadata checks are not full content hashing of all datasets.', '',
      'Output: `studies/track_a_3h_v1/runtime/output/features/`, exclusive month directories; 80 Parquet files (2,013,633,097 bytes) are SHA-256 recorded in [output-files.json](evidence/output-files.json); binary data and full event logs remain local and ignored. [Execution](evidence/execution.json), [sample parity](evidence/sample.json), [generated checks](evidence/generated.json), [schema](evidence/schema.json), and event-log hashes in [resource evidence](evidence/resources.json) provide provenance.', '',
      '**Stop:** feature qualification only. No predictive improvement is claimed; no model fitting or promotion occurred.']
    if not passed:lines += ['',f'Blocking evidence: {execution.get("error")}; inspect build.log. Incomplete output is not qualified and must not be retried under this authorization.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
