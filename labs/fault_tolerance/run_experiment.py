"""Bounded, isolated worker-loss lab; host controller and Spark workload in one file."""
import argparse
import datetime
import json
import hashlib
import socket
import os
from pathlib import Path
import subprocess
import time
import uuid


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def expected():
    return [[k, [100, 49500 + 100*k]] for k in range(10)]


def partition(values, group):
    from pyspark import TaskContext
    context=TaskContext.get()
    marker=Path('/evidence')/('active-'+socket.gethostname()+'.json')
    if group=='recovery':
        value={'Stage ID':context.stageId(),'Task Info':{'Task ID':context.taskAttemptId(),
               'Index':context.partitionId(),'Attempt':context.attemptNumber(),'Host':socket.gethostname()}}
        temporary=marker.with_suffix('.tmp')
        write(temporary,value);temporary.replace(marker)
    try:
        time.sleep(10)  # active task window before emitting any output
        for value in values:
            yield value % 10, (1, value)
    finally:
        if group=='recovery': marker.unlink(missing_ok=True)


def workload():
    from pyspark import SparkConf, SparkContext
    conf = (SparkConf().setAppName('isolated-worker-loss').setMaster('spark://master:7077')
            .set('spark.driver.host', 'driver').set('spark.driver.bindAddress', '0.0.0.0')
            .set('spark.cores.max', '2').set('spark.executor.cores', '1')
            .set('spark.executor.memory', '640m').set('spark.task.maxFailures', '4')
            .set('spark.speculation', 'false').set('spark.eventLog.enabled', 'true')
            .set('spark.eventLog.compress', 'false').set('spark.eventLog.dir', 'file:/evidence/events'))
    with SparkContext(conf=conf) as sc:
        sc.setLogLevel('WARN')
        for group in ('baseline', 'recovery'):
            sc.setJobGroup(group, group)
            result = sorted(sc.parallelize(range(1000), 8).mapPartitions(lambda values: partition(values, group))
                            .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]), 4).collect())
            normalized = json.loads(json.dumps(result))
            if normalized != expected():
                raise RuntimeError('Incorrect or duplicated distributed output')
            write(Path('/evidence') / (group+'.json'), normalized)


def events(root):
    result = []
    for path in sorted((root/'events').glob('*')):
        if path.is_file() and not path.name.startswith('.'):
            for line in path.read_text().splitlines():
                try: result.append(json.loads(line))
                except json.JSONDecodeError: pass  # only a final partial line during live polling
    return result


def active_victim(rows):
    stages = set()
    executors = set()
    running = {}
    for e in rows:
        kind = e['Event']
        if kind == 'SparkListenerExecutorAdded' and e['Executor Info']['Host'] == 'worker1':
            executors.add(e['Executor ID'])
        if kind == 'SparkListenerJobStart' and e.get('Properties',{}).get('spark.jobGroup.id') == 'recovery':
            stages.update(e['Stage IDs'])
        if kind == 'SparkListenerTaskStart': running[e['Task Info']['Task ID']] = e
        if kind == 'SparkListenerTaskEnd': running.pop(e['Task Info']['Task ID'], None)
    return next((e for e in running.values() if e['Stage ID'] in stages and e['Task Info']['Executor ID'] in executors), None)


def verify(root):
    rows = events(root)
    injection = json.loads((root/'injection.json').read_text())
    victim = injection['active_task']
    task = victim['Task Info']
    starts=[e for e in rows if e['Event']=='SparkListenerTaskStart' and e['Task Info']['Task ID']==task['Task ID']]
    if not starts or starts[0]['Stage ID']!=victim['Stage ID']:
        raise ValueError('Injected task has no matching Spark start event')
    ends = [e for e in rows if e['Event']=='SparkListenerTaskEnd']
    failures = [e for e in ends if e['Task Info']['Task ID']==task['Task ID']
                and e['Task End Reason']['Reason']=='ExecutorLostFailure']
    retries = [e for e in ends if e['Stage ID']==victim['Stage ID']
               and e['Task Info']['Index']==task['Index']
               and e['Task Info']['Attempt']>task['Attempt']
               and e['Task End Reason']['Reason']=='Success'
               and e['Task Info']['Executor ID']!=task['Executor ID']]
    removals = [e for e in rows if e['Event']=='SparkListenerExecutorRemoved' and e['Executor ID']==task['Executor ID']]
    jobs = [e for e in rows if e['Event']=='SparkListenerJobEnd']
    if not failures or not retries or not removals:
        raise ValueError('Missing active-task loss, successful retry, or executor removal evidence')
    if len(jobs)!=2 or any(e['Job Result']['Result']!='JobSucceeded' for e in jobs):
        raise ValueError('Both baseline and recovery jobs must succeed')
    if not any(e['Event']=='SparkListenerApplicationEnd' for e in rows):
        raise ValueError('Application did not end cleanly')
    for name in ('baseline','recovery'):
        if json.loads((root/(name+'.json')).read_text()) != expected():
            raise ValueError('Incorrect output: '+name)
    return dict(status='passed',rows=1000,sum=499500,groups=10,jobs=len(jobs),
                stages=len([e for e in rows if e['Event']=='SparkListenerStageCompleted']),
                task_attempts=len(ends),failed_task=failures[0],successful_retry=retries[0],
                executor_removed=removals[0],scope='one worker/executor loss; no driver recovery or external writes')


def run(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'events').mkdir()
    project = 'airline-ft-' + uuid.uuid4().hex[:12]
    env = dict(os.environ,LAB_OUTPUT=str(output),LAB_UID=str(os.getuid()),LAB_GID=str(os.getgid()))
    base = ['docker','compose','-p',project,'-f',str(Path(__file__).with_name('compose.yml'))]
    commands=[]
    source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('run_experiment.py','compose.yml')}
    def command(args, timeout=30, check=True):
        full=base+args
        result=subprocess.run(full,env=env,capture_output=True,text=True,timeout=timeout)
        commands.append(dict(argv=full,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if check and result.returncode: raise RuntimeError(result.stderr)
        return result
    before=subprocess.check_output(['docker','ps','-a','--no-trunc','--format','{{.ID}} {{.Names}} {{.Status}}'],text=True)
    process=None
    try:
        command(['up','-d'],timeout=60)
        write(output/'topology.json',[json.loads(line) for line in command(['ps','--format','json']).stdout.splitlines()])
        deadline=time.monotonic()+240
        probe=['exec','-T','driver','python3','-c',"import json,urllib.request; d=json.load(urllib.request.urlopen('http://master:8080/json/')); assert len([w for w in d['workers'] if w['state']=='ALIVE'])==2; print(json.dumps(d))"]
        while True:
            ready=command(probe,check=False)
            if ready.returncode==0: break
            if time.monotonic()>deadline: raise TimeoutError('Two-worker cluster did not become ready')
            exited=command(['ps','-a','--status','exited','-q']).stdout.strip()
            if exited:
                command(['logs','--no-color'],check=False)
                raise RuntimeError('Lab service exited during startup')
            time.sleep(1)
        (output/'master.json').write_text(ready.stdout)
        argv=base+['exec','-T','driver','python3','/lab/run_experiment.py','--workload']
        with (output/'driver.log').open('w') as log:
            process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
            injected=False
            while process.poll() is None:
                if time.monotonic()>deadline: raise TimeoutError('240-second lab deadline exceeded')
                marker=output/'active-worker1.json'
                victim=json.loads(marker.read_text()) if marker.exists() else None
                if victim:
                    executor=next((e['Executor ID'] for e in events(output) if e['Event']=='SparkListenerExecutorAdded' and e['Executor Info']['Host']=='worker1'),None)
                    if executor is None: victim=None
                    else: victim['Task Info']['Executor ID']=executor
                if victim and not injected:
                    cid=command(['ps','-q','worker1']).stdout.strip()
                    labels=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .Config.Labels}}',cid],text=True))
                    if labels.get('com.docker.compose.project')!=project or labels.get('com.docker.compose.service')!='worker1':
                        raise RuntimeError('Refusing to terminate a worker outside this experiment')
                    injection_ms=int(time.time()*1000)
                    failure=command(['kill','-s','SIGKILL','worker1'])
                    write(output/'injection.json',dict(command=commands[-1],container=cid,labels=labels,active_task=victim,injection_ms=injection_ms))
                    injected=True
                time.sleep(.2)
            commands.append(dict(argv=argv,exit_code=process.returncode))
            if process.returncode: raise RuntimeError('Spark workload failed; inspect driver.log')
            if not injected: raise RuntimeError('No active task was affected')
        write(output/'verification.json',verify(output))
    finally:
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        cleanup=command(['down','--volumes','--remove-orphans'],timeout=45,check=False)
        remaining=subprocess.check_output(['docker','ps','-a','--filter','label=com.docker.compose.project='+project,'-q'],text=True).strip()
        after=subprocess.check_output(['docker','ps','-a','--no-trunc','--format','{{.ID}} {{.Names}} {{.Status}}'],text=True)
        write(output/'execution.json',dict(project=project,source_sha256=source_hashes,commands=commands,cleanup_exit_code=cleanup.returncode,
             remaining_containers=remaining,preexisting_containers_before=before,preexisting_containers_after=after,
             actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        if cleanup.returncode or remaining: raise RuntimeError('Incomplete lab cleanup')
    print(json.dumps(verify(output),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload',action='store_true')
    parser.add_argument('--verify',type=Path)
    parser.add_argument('--output',type=Path,default=Path('artifacts/fault_tolerance')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args=parser.parse_args()
    if args.workload: workload()
    elif args.verify: print(json.dumps(verify(args.verify),indent=2))
    else: run(args.output)
