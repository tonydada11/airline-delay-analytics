"""Offline Spark task and cgroup evidence; does not run Spark or retry work."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
E=HERE/'evidence'

def main():
    stages={};metrics=defaultdict(lambda:defaultdict(int));failed=[];executors={};logs=[]
    for path in sorted((HERE/'runtime/output/events').glob('*')):
        if not path.is_file():continue
        logs.append({'path':str(path.relative_to(HERE)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for line in path.open():
            e=json.loads(line);kind=e['Event']
            if kind=='SparkListenerExecutorAdded':
                executors[e['Executor ID']]=e['Executor Info']
            if kind=='SparkListenerJobStart':
                group=e.get('Properties',{}).get('spark.jobGroup.id','unassigned')
                for s in e['Stage Infos']:stages.setdefault(s['Stage ID'],group)
            if kind!='SparkListenerTaskEnd':continue
            group=stages.get(e['Stage ID'],'unassigned');m=metrics[group];t=e.get('Task Metrics',{})
            m['tasks']+=1
            for key in ('Executor Run Time','Executor CPU Time','JVM GC Time','Memory Bytes Spilled','Disk Bytes Spilled'):
                m[key]+=t.get(key,0)
            m['peak_task_execution_memory_bytes']=max(m['peak_task_execution_memory_bytes'],t.get('Peak Execution Memory',0))
            read=t.get('Shuffle Read Metrics',{})
            m['shuffle_read_bytes']+=read.get('Remote Bytes Read',0)+read.get('Local Bytes Read',0)
            m['shuffle_write_bytes']+=t.get('Shuffle Write Metrics',{}).get('Shuffle Bytes Written',0)
            info=e['Task Info']
            m['first_task_epoch_ms']=min(m.get('first_task_epoch_ms',info['Launch Time']),info['Launch Time'])
            m['last_task_epoch_ms']=max(m.get('last_task_epoch_ms',0),info['Finish Time'])
            if e.get('Task End Reason',{}).get('Reason')!='Success':failed.append({'stage':e['Stage ID'],'reason':e.get('Task End Reason')})
    memory=defaultdict(dict)
    for sample in json.loads((E/'memory.json').read_text()):
        for name,value in sample['containers'].items():
            memory[sample['phase']][name]=max(memory[sample['phase']].get(name,0),int(value['memory.current']))
    split_metrics={}
    for split in ('train','validation'):
        selected=[m for phase,m in metrics.items() if phase.startswith(split+'-')]
        split_metrics[split]={k:sum(m.get(k,0) for m in selected) for k in
            ('tasks','Executor Run Time','Executor CPU Time','JVM GC Time','Memory Bytes Spilled','Disk Bytes Spilled','shuffle_read_bytes','shuffle_write_bytes')}
    months_path=E/'months.json'
    split_population={}
    if months_path.exists():
        months=json.loads(months_path.read_text())
        for split in ('train','validation'):
            selected=[m for m in months if m['month'].startswith(split+'-')]
            rows=sum(m['rows'] for m in selected)
            split_population[split]={'rows':rows,'elapsed_seconds':sum(m['elapsed_seconds'] for m in selected),
                **{role+'_missing_fraction':sum(m[role+'_missing'] for m in selected)/rows if rows else None for role in ('origin','destination')}}
    split_memory={split:{name:max((values.get(name,0) for phase,values in memory.items() if phase.startswith(split+'-')),default=0)
        for name in {name for values in memory.values() for name in values}} for split in ('train','validation')}
    result={'sampled_cgroup_memory_max_by_split':split_memory,'task_metrics_by_split':split_metrics,'population_by_split':split_population,'task_metrics_by_phase':dict(metrics),'sampled_cgroup_memory_max_by_phase':dict(memory),
            'memory_note':'10-second sampled current usage, lower bound per phase; exact whole-container lifetime peaks in execution.json. Task peak execution memory is per task, not process RSS.',
            'failed_tasks':failed,'executors':executors,'event_logs':logs}
    (E/'resources.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
