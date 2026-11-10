#!/usr/bin/env python3
"""Durable local continuation through confirmation, sensitivity and reporting.

This controller does not hold worker CPU/GPU leases. All scientific jobs run
through the budget guard. It never computes or reads partial external metrics.
"""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
PY=str(ROOT/'.venv/bin/python')
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
     'NUMEXPR_NUM_THREADS':'1','PYTHONPATH':str(ROOT/'src'),'MPLCONFIGDIR':str(ROOT/'cache/matplotlib'),
     'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}


def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)


def status(phase,**extra):
    event={'controller_pid':os.getpid(),'controller_start_time':psutil.Process().create_time(),
           'phase':phase,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),**extra}
    save(ROOT/'runs/controller_status.json',event)
    state=json.loads((ROOT/'STATE.json').read_text());state['phase']=phase;state['controller']=event
    if phase != 'complete' and state.get('status') not in ['READY_FOR_AUTHOR_REVIEW','NOT_READY']:
        state['status']='CONTINUE'
    save(ROOT/'STATE.json',state)


def existing_pipeline():
    matches=[]
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            if 'scripts/pipeline.py' in (p.info['cmdline'] or []) and Path(p.cwd())==ROOT:
                matches.append(p)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    if len(matches)>1:raise RuntimeError('Multiple coordinating pipelines found; inspect without starting another')
    return matches[0] if matches else None


def wait_process(p,phase):
    start=p.create_time()
    while True:
        try:
            if not p.is_running() or p.status()==psutil.STATUS_ZOMBIE or p.create_time()!=start:return
        except psutil.NoSuchProcess:return
        status(phase,worker_pid=p.pid)
        time.sleep(30)


def run(command,phase):
    status(phase,command=command)
    process=subprocess.Popen(command,cwd=ROOT,env=ENV)
    while process.poll() is None:
        status(phase,worker_pid=process.pid,command=command)
        time.sleep(15)
    if process.returncode:
        raise RuntimeError(f'{phase} exited {process.returncode}; inspect preserved run logs')


def guarded(name,command):
    run([PY,'scripts/run_guard.py','--run-id',name,'--stage','review','--threads','1','--',*command],name)


def main():
    lease=(ROOT/'runs/controller.lock').open('w')
    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Continuation controller already active')
    try:
        current=existing_pipeline()
        if current is not None:wait_process(current,'locked_primary_pipeline_running')
        if not (ROOT/'artifacts/PREDICTION_COMPLETENESS.json').exists():
            run([PY,'scripts/pipeline.py','--phase','all'],'locked_primary_pipeline_running')
        for kind in ['author_labels','dd_original']:
            run([PY,'scripts/sensitivity_pipeline.py','--kind',kind],f'sensitivity_{kind}')
        complete=json.loads((ROOT/'artifacts/PREDICTION_COMPLETENESS.json').read_text())
        if not complete.get('complete'):raise RuntimeError('Main predictions incomplete: do not unseal')
        guarded('review_primary_evaluation_v1',[PY,'-m','bh_transfer.evaluate',
                '--predictions',str(ROOT/'artifacts/predictions.parquet'),'--lock',str(ROOT/'FINAL_EVAL_LOCK.json'),
                '--out',str(ROOT/'reports/final')])
        for kind in ['author_labels','dd_original']:
            project=ROOT/'sensitivity'/kind
            completion=project/'artifacts/PREDICTION_COMPLETENESS.json'
            if completion.exists() and json.loads(completion.read_text()).get('complete'):
                guarded('review_'+kind+'_evaluation_v1',[PY,'-m','bh_transfer.evaluate',
                        '--predictions',str(project/'artifacts/predictions.parquet'),'--lock',str(project/'FINAL_EVAL_LOCK.json'),
                        '--out',str(project/'reports/final')])
        guarded('review_independent_report_v1',[PY,'scripts/finalize_report.py','--project',str(ROOT)])
        # The report owns scientific readiness; controller completion cannot
        # upgrade NOT_READY to READY_FOR_AUTHOR_REVIEW.
        state=json.loads((ROOT/'STATE.json').read_text())
        status('complete',scientific_status=state['status'])
        print(json.dumps({'controller':'complete','scientific_status':state['status']}),flush=True)
    except Exception as exc:
        status('continuation_needs_inspection',error=str(exc))
        raise


if __name__=='__main__':main()
