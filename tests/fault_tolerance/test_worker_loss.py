# AI-generated test code: generated with OpenAI Codex.
"""Reject false-positive recovery claims using compact synthetic event records."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('worker_loss',Path(__file__).resolve().parents[2]/'labs/fault_tolerance/run_experiment.py')
lab=importlib.util.module_from_spec(spec);spec.loader.exec_module(lab)


def fixture(tmp_path):
    task={'Task ID':1,'Executor ID':'0','Index':2,'Attempt':0}
    active={'Event':'SparkListenerTaskStart','Stage ID':2,'Task Info':task}
    rows=[{'Event':'SparkListenerExecutorAdded','Executor ID':'0','Executor Info':{'Host':'worker1'}},
          {'Event':'SparkListenerJobStart','Stage IDs':[2,3],'Properties':{'spark.jobGroup.id':'recovery'}},active,
          {'Event':'SparkListenerTaskEnd','Stage ID':2,'Task Info':task,'Task End Reason':{'Reason':'ExecutorLostFailure'}},
          {'Event':'SparkListenerExecutorRemoved','Executor ID':'0'},
          {'Event':'SparkListenerTaskEnd','Stage ID':2,'Task Info':dict(task,**{'Task ID':2,'Executor ID':'1','Attempt':1}),'Task End Reason':{'Reason':'Success'}},
          *[{'Event':'SparkListenerJobEnd','Job Result':{'Result':'JobSucceeded'}} for _ in range(2)],
          {'Event':'SparkListenerApplicationEnd'}]
    (tmp_path/'events').mkdir();lab.write(tmp_path/'injection.json',{'active_task':active})
    for name in ('baseline','recovery'): lab.write(tmp_path/(name+'.json'),lab.expected())
    return rows


def save(tmp_path,rows):
    (tmp_path/'events/app').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_missing_active_task_start_rejected(tmp_path):
    rows=fixture(tmp_path);rows.pop(2);save(tmp_path,rows)
    with pytest.raises(ValueError, match='matching Spark start'):
        lab.verify(tmp_path)


def test_correct_recovery(tmp_path):
    rows=fixture(tmp_path);save(tmp_path,rows)
    assert lab.verify(tmp_path)['status']=='passed'


@pytest.mark.parametrize('remove',[3,4,5,6,8])
def test_missing_loss_retry_or_completion_rejected(tmp_path,remove):
    rows=fixture(tmp_path);rows.pop(remove);save(tmp_path,rows)
    with pytest.raises(ValueError): lab.verify(tmp_path)


def test_success_on_wrong_partition_rejected(tmp_path):
    rows=fixture(tmp_path);rows[5]['Task Info']['Index']=3;save(tmp_path,rows)
    with pytest.raises(ValueError): lab.verify(tmp_path)


def test_duplicate_output_rejected(tmp_path):
    rows=fixture(tmp_path);save(tmp_path,rows)
    lab.write(tmp_path/'recovery.json',lab.expected()+lab.expected()[:1])
    with pytest.raises(ValueError): lab.verify(tmp_path)
