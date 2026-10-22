"""Single attempt; 4h cap includes teardown. Never restart or resize containers."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
IMAGE='sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf'
COMPOSE=['docker','compose','-f',str(HERE/'compose.yml')]
NAMES=['airline-track-a-3h-v1-'+s+'-1' for s in ('master','worker1','worker2','driver')]

def output(cmd):
    return subprocess.check_output(cmd,text=True,timeout=30).strip()

def save(name,value):
    (HERE/'evidence'/name).write_text(json.dumps(value,indent=2)+'\n')

def snapshot():
    tracked={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in output(['git','ls-tree','-r','--name-only','HEAD']).splitlines()}
    metadata={str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for folder in ('data','artifacts','studies') for p in (ROOT/folder).rglob('*') if p.is_file() and HERE not in p.parents and '__pycache__' not in p.parts}
    return {'tracked_sha256':tracked,'existing_file_metadata':metadata}

def main():
    # Exclusive directory is also the one-attempt marker, even if launch fails.
    runtime=HERE/'runtime';runtime.mkdir(exist_ok=False)
    (runtime/'output/events').mkdir(parents=True)
    before=snapshot();save('preservation-before.json',before)
    production=output(['docker','ps','-a','--format','{{.Names}}']).splitlines()
    assert all(n.startswith('airline-revision-') for n in production), production
    before_docker=json.loads(output(['docker','inspect',*production]))
    assert all(not x['State']['Running'] for x in before_docker)
    assert output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'])==IMAGE
    # Recheck every source byte used by the previously reviewed sample.
    prior=json.loads((ROOT/'studies/ml_three_track_v1/evidence/a-results.json').read_text())
    for record in prior['source_files']:
        path=ROOT/'data/warehouse/dwd/flights'/record['path'].removeprefix('/dwd/')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],path
    save('sample-provenance.json',{'source_files_sha256_match':len(prior['source_files']),'reference':'16e841e'})
    start=time.monotonic();process=None;code=None;error=None;peaks={};samples=[]
    try:
        subprocess.run(COMPOSE+['up','-d'],check=True,timeout=60)
        save('containers.json',json.loads(output(['docker','inspect',*NAMES])))
        with (HERE/'evidence/build.log').open('w') as log:
            process=subprocess.Popen(COMPOSE+['exec','-T','driver','timeout','--signal=TERM','--kill-after=15','14100',
                '/opt/spark/bin/spark-submit','--driver-memory','768m','/study/build.py'],stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if time.monotonic()-start>14220:
                    raise TimeoutError('Stop work with 180 seconds reserved for cleanup inside 4h cap')
                phase_path=runtime/'output/phase.json'
                phase=json.loads(phase_path.read_text())['name'] if phase_path.exists() else 'startup'
                measurements={}
                for name in NAMES:
                    measurements[name]=json.loads(output(['docker','exec',name,'python3','-c',
                        'from pathlib import Path; import json; print(json.dumps({n:(Path("/sys/fs/cgroup")/n).read_text().strip() for n in ["memory.current","memory.peak","memory.events"]}))']))
                samples.append({'elapsed_seconds':time.monotonic()-start,'phase':phase,'containers':measurements})
                save('memory.json',samples)
                time.sleep(10)
            code=process.wait()
        for name in NAMES:
            peaks[name]=json.loads(output(['docker','exec',name,'python3','-c',
                'from pathlib import Path; import json; print(json.dumps({n:(Path("/sys/fs/cgroup")/n).read_text().strip() for n in ["memory.peak","memory.max","memory.events"]}))']))
        if code: raise RuntimeError(f'Build exit {code}; no retry')
    except Exception as exc:
        error=repr(exc)
    finally:
        for name in NAMES:
            with (HERE/'evidence'/(name+'.log')).open('w') as log:
                subprocess.run(['docker','logs',name],stdout=log,stderr=subprocess.STDOUT,timeout=20)
        subprocess.run(COMPOSE+['down','--timeout','15'],check=True,timeout=60)
        if process is not None and process.poll() is None:process.wait(timeout=30)
        wall=time.monotonic()-start
        after=snapshot()
        after_docker=json.loads(output(['docker','inspect',*production]))
        def stable(items):
            return [{k:(sorted(v,key=lambda m:m['Destination']) if k=='Mounts' else v) for k,v in x.items() if k in ('Id','Image','State','Mounts','Config','HostConfig')} for x in items]
        preserved=before==after and stable(before_docker)==stable(after_docker)
        save('execution.json',{'exit_code':code,'error':error,'wall_seconds_including_cleanup':wall,'cap_seconds':14400,'cgroups':peaks,'preserved':preserved,'study_containers_removed':not any(n in output(['docker','ps','-a','--format','{{.Names}}']).splitlines() for n in NAMES)})
        save('preservation.json',{'unchanged':preserved,'tracked_files':len(before['tracked_sha256']),'metadata_files':len(before['existing_file_metadata'])})
        for p in (runtime/'output').glob('*.json'):
            (HERE/'evidence'/p.name).write_bytes(p.read_bytes())
        assert preserved
        assert wall<14400
    if error: raise RuntimeError(error)

if __name__=='__main__':main()
