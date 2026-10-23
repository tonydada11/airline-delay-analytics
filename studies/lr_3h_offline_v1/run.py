"""Sequential bounded verification and standalone scoring; zero fits, no production route."""
import argparse,json,subprocess,time
from pathlib import Path
import offline as api
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NAMES=['airline-lr3h-offline-v1-'+s+'-1' for s in ('master','worker1','worker2','driver')]
IMAGE='sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf'
def output(cmd):return subprocess.check_output(cmd,text=True,timeout=30).strip()
def save(name,value):api.write(HERE/'evidence'/name,value)
def snapshot():
    tracked={s:api.digest(ROOT/s) for s in output(['git','ls-tree','-r','--name-only','HEAD']).splitlines()}
    files={str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for d in ('data','artifacts','studies') for p in (ROOT/d).rglob('*') if p.is_file() and HERE not in p.parents and '__pycache__' not in p.parts}
    return {'tracked_sha256':tracked,'file_metadata':files}
def main(finish=False):
    runtime=HERE/'runtime'
    if finish:
        assert json.loads((runtime/'output/verification.json').read_text())['status']=='PASS'
        root=runtime/'output/staged-bundle'
        if not root.exists():root=runtime/'output/bundle'
        api.validate_files(root,json.loads((root/'manifest.json').read_text()))
    runtime.mkdir(exist_ok=finish);(runtime/'output/events').mkdir(parents=True,exist_ok=finish)
    start=time.monotonic()-(time.time()-runtime.stat().st_mtime) if finish else time.monotonic()
    before=snapshot();save('preservation-before.json',before)
    production=output(['docker','ps','-a','--format','{{.Names}}']).splitlines();assert all(n.startswith('airline-revision-') for n in production)
    prod_before=output(['docker','inspect',*production]);assert all(not c['State']['Running'] for c in json.loads(prod_before))
    assert output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'])==IMAGE
    files=json.loads((ROOT/'studies/track_a_3h_v1/evidence/output-files.json').read_text())['files']
    for record in files:
        p=ROOT/'studies/track_a_3h_v1'/record['path'];assert p.is_file() and p.stat().st_size==record['bytes'] and api.digest(p)==record['sha256'],p
    baseline=ROOT/'data/studies/lr-improvement-v1/diagnostics/results.json'
    assert api.digest(baseline)=='c1313197e55e9d9d90793ec3401c3c0d3d61597ca35da94eae5c1f7b8fdee828'
    save('inputs.json',{'qualified_feature_files_sha256_match':80,'baseline_sha256':api.digest(baseline),'implementation_commit':output(['git','rev-parse','HEAD']),'image':IMAGE})
    samples=json.loads((HERE/'evidence/memory.json').read_text()) if finish else []
    phases=json.loads((HERE/'evidence/phases.json').read_text()) if finish else []
    error=None
    def execute(config,label,command):
        compose=['docker','compose','-f',str(HERE/config)];process=None;began=time.monotonic();peaks={};code=None
        try:
            subprocess.run(compose+['up','-d'],check=True,timeout=60)
            save(label+'-containers.json',json.loads(output(['docker','inspect',*NAMES])))
            with (HERE/'evidence'/(label+'.log')).open('w') as log:
                left=max(1,int(14100-(time.monotonic()-start)))
                process=subprocess.Popen(compose+['exec','-T','driver','timeout','--signal=TERM','--kill-after=15',str(left),'/opt/spark/bin/spark-submit','--driver-memory','768m',*command],stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if time.monotonic()-start>14220:raise TimeoutError('Reserve cleanup within 4h wall cap')
                    measurements={name:json.loads(output(['docker','exec',name,'python3','-c','from pathlib import Path; import json; print(json.dumps({n:(Path("/sys/fs/cgroup")/n).read_text().strip() for n in ["memory.current","memory.peak","memory.events"]}))'])) for name in NAMES}
                    samples.append({'phase':label,'elapsed_seconds':time.monotonic()-start,'containers':measurements});save('memory.json',samples);time.sleep(10)
                code=process.wait()
            for name in NAMES:
                peaks[name]=json.loads(output(['docker','exec',name,'python3','-c','from pathlib import Path; import json; print(json.dumps({n:(Path("/sys/fs/cgroup")/n).read_text().strip() for n in ["memory.peak","memory.max","memory.events"]}))']))
            subprocess.run(['docker','exec',NAMES[-1],'chmod','-R','a+rX','/output/events'],check=True,timeout=15)
            assert code==0,f'{label}: exit {code}'
        finally:
            subprocess.run(compose+['down','--timeout','15'],check=True,timeout=60)
            if process is not None and process.poll() is None:process.wait(timeout=30)
            phases.append({'name':label,'exit_code':code,'seconds_including_cleanup':time.monotonic()-began,'cgroups':peaks});save('phases.json',phases)
    try:
        if not finish:execute('compose.yml','stage',['/study/verify.py','--mode','stage'])
        # No original studies, accepted model, source tree or helper mounts exist here.
        completed=runtime/'output/scored-offline-v1/_completion.json'
        if finish and completed.exists():
            expected=json.loads(completed.read_text());actual=api.inventory(completed.parent);actual.pop('_completion.json',None)
            assert actual==expected['files'] and expected['rows']==2418739 and expected['threshold']==0.39
        else:execute('compose.fresh.yml','fresh',['/bundle/offline.py','--bundle','/bundle','--input',*[f'/input/{m}' for m in ('2024-05','2024-06','2024-07','2024-08')],'--output','/output/scored-offline-v1'])
        final_check=runtime/'output/final-parity.json'
        if not (finish and final_check.exists() and json.loads(final_check.read_text())['status']=='PASS'):
            execute('compose.yml','final',['/study/verify.py','--mode','final'])
        stage=runtime/'output/staged-bundle';bundle=runtime/'output/bundle'
        if stage.exists():stage.rename(bundle)  # Final publication after all parity checks.
        api.validate_files(bundle,json.loads((bundle/'manifest.json').read_text()))
        # Files were created by the container UID; retain that UID for mode changes.
        subprocess.run(['docker','run','--rm','--network','none','--user','0','--entrypoint','python3',
            '-v',str(bundle)+':/bundle',IMAGE,'-c',
            'from pathlib import Path; root=Path("/bundle"); [p.chmod(0o555 if p.is_dir() else 0o444) for p in root.rglob("*")]; root.chmod(0o555)'],check=True,timeout=30)
        assert all(not (p.stat().st_mode & 0o222) for p in bundle.rglob('*'))
    except Exception as exc:error=repr(exc)
    finally:
        after=snapshot();prod_after=output(['docker','inspect',*production])
        def stable(text):
            return [{k:(sorted(v,key=lambda m:m['Destination']) if k=='Mounts' else v) for k,v in c.items() if k in ('Id','Image','State','Mounts','Config','HostConfig')} for c in json.loads(text)]
        preserved=before==after and stable(prod_before)==stable(prod_after)
        wall=time.monotonic()-start
        save('execution.json',{'status':'PASS' if not error and preserved else 'BLOCKED','error':error,'seconds_including_cleanup':wall,'wall_cap_seconds':14400,'preserved':preserved,'study_containers_removed':not any(n in output(['docker','ps','-a','--format','{{.Names}}']).splitlines() for n in NAMES),'classifier_fits':0,'preprocessing_refits':0})
        save('preservation.json',{'unchanged':preserved,'tracked_files':len(before['tracked_sha256']),'existing_files_metadata':len(before['file_metadata'])})
        for p in (runtime/'output').glob('*.json'):(HERE/'evidence'/p.name).write_bytes(p.read_bytes())
        assert preserved and wall<14400
    if error:raise RuntimeError(error)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--finish',action='store_true',help='Finish verified stages after a recorded launcher/verifier correction');main(parser.parse_args().finish)
