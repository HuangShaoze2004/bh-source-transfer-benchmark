#!/usr/bin/env python3
"""Resume the locked nested experiment without revealing external scores."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit

ROOT=Path(__file__).resolve().parents[1]
PY=str(ROOT/'.venv/bin/python')
OBJECTIVES={'erm':('B','B_S'),'bal':('B_bal','B_bal_S'),'dro':('B_A','B_A_S')}
TREES=['R0_RF','R0_HGB','R1_substitute']


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def job(name,stage,script,spec,gpu=False):
    spec_path=ROOT/'runs/specs'/(name+'.json')
    if spec_path.exists() and json.loads(spec_path.read_text())!=spec:
        raise ValueError(f'Existing run spec changed: {name}')
    if not spec_path.exists():dump(spec_path,spec)
    cmd=[PY,'scripts/run_guard.py','--run-id',name,'--stage',stage,'--threads','1']
    if gpu:cmd+=['--gpu']
    cmd+=['--',PY,script,'--spec',str(spec_path)]
    subprocess.run(cmd,cwd=ROOT,check=True)


def temperature(y,z,sources):
    if len(y)==0 or len(set(y))<2:return 1.
    _,inv,counts=np.unique(sources,return_inverse=True,return_counts=True)
    weights=1/counts[inv];weights/=weights.sum()
    def loss(t):
        zz=z/np.exp(t)
        return np.dot(weights,np.logaddexp(0,zz)-y*zz)
    result=minimize_scalar(loss,bounds=(-3.,3.),method='bounded')
    return float(np.exp(result.x)) if result.success else 1.


def val_temperature(frame,files,key):
    ids=[];zs=[]
    for file in files:
        with np.load(file) as f:
            ids.extend(f['row_ids'].astype(int)); z=f[key]
            zs.extend(logit(np.clip(z,1e-7,1-1e-7)) if key=='score' else z)
    rows=frame.loc[ids]
    return temperature(rows.y.to_numpy(float),np.array(zs),rows.source.to_numpy())


def run_development(lock,splits,frame):
    grid=lock['neural_grid'];base=lock['neural_base_config']
    lock_hash=hashlib.sha256((ROOT/'FINAL_EVAL_LOCK.json').read_bytes()).hexdigest()
    import importlib.util
    module_spec=importlib.util.spec_from_file_location('bh_run_guard',ROOT/'scripts/run_guard.py')
    guard=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(guard)
    search_exhausted=False
    selections={}
    for fold,part in splits['folds'].items():
        if not part['test_ids']:continue
        selection_path=ROOT/'runs/selections'/(fold+'.json')
        if selection_path.exists():
            saved=json.loads(selection_path.read_text())
            if saved.get('lock_sha256')!=lock_hash:raise ValueError('Selection cache belongs to another protocol')
            selections[fold]=saved['choices'];continue
        choices={}
        completed_grid=[]; cache={}
        # Finish a common configuration across ALL objectives and inner folds
        # before it can enter selection. A budget stop never favors a partial grid.
        for gi,hp in enumerate(grid):
            if search_exhausted:break
            this_grid={}; grid_complete=True
            for objective in OBJECTIVES:
                results=[];dirs=[]
                for ii,inner in enumerate(part['inner']):
                    name=f'dev_{fold}_{objective}_g{gi}_i{ii}'
                    out=ROOT/'runs/trajectories'/name
                    spec={'project':str(ROOT),'train_ids':inner['train_ids'],'val_ids':inner['val_ids'],
                          'config':{**base,**hp,'objective':objective,'seed':42},'output_dir':str(out)}
                    try:
                        job(name,'development','scripts/run_neural_job.py',spec,True)
                    except subprocess.CalledProcessError:
                        if guard.totals()['development_gpu_hours'] >= 50-125/3600:
                            search_exhausted=True;grid_complete=False;break
                        raise
                    results.append(json.loads((out/'result.json').read_text()));dirs.append(out)
                this_grid[objective]=(results,dirs)
                if not grid_complete:break
            if grid_complete:
                completed_grid.append(gi);cache[gi]=this_grid
            else:
                dump(ROOT/'runs/selections'/(fold+'_budget_stop.json'),
                     {'lock_sha256':lock_hash,'discarded_partial_grid':gi,'common_completed_grids':completed_grid,
                      'rule':'Only complete matched configurations used; otherwise fixed defaults; formal work continues'})
        for objective,names in OBJECTIVES.items():
            candidates=[]
            for gi in completed_grid:
                results,dirs=cache[gi][objective]
                if results:
                    scores=[next(c for c in r['candidates'] if c['name']=='ordinary_best')['val_metrics']['auroc'] for r in results]
                    candidates.append((float(np.mean(scores)),gi,results,dirs))
            if candidates:
                # Identical shared backbone configuration for ordinary and SWA;
                # selected solely by ordinary inner AUROC, deterministic grid tie.
                _,gi,results,dirs=max(candidates,key=lambda x:(x[0],-x[1]))
                fraction=float(np.mean([r['ordinary_best_step']/r['total_steps'] for r in results]))
                swa_name=max(['swa_050','swa_075'],key=lambda k:(
                    np.mean([next(c for c in r['candidates'] if c['name']==k)['val_metrics']['auroc'] for r in results]),k))
                t_plain=val_temperature(frame,[d/'ordinary_best_validation.npz' for d in dirs],'logits')
                t_swa=val_temperature(frame,[d/(swa_name+'_validation.npz') for d in dirs],'logits')
            else:
                gi=lock['default_grid_index'];fraction=1.;swa_name='swa_075';t_plain=t_swa=1.
            choices[objective]={'config':{**base,**grid[gi],'objective':objective,'ordinary_fraction':fraction},
                                'grid_index':gi,'swa_candidate':swa_name,'temperatures':{names[0]:t_plain,names[1]:t_swa},
                                'inner_valid':bool(candidates),'common_completed_grids':completed_grid}
        for model in TREES:
            files=[]
            for ii,inner in enumerate(part['inner']):
                name=f'dev_{fold}_{model}_i{ii}';out=ROOT/'runs/tabular'/name
                spec={'project':str(ROOT),'train_ids':inner['train_ids'],'val_ids':inner['val_ids'],
                      'model':model,'seed':42,'output_dir':str(out)}
                job(name,'development','scripts/run_tabular_job.py',spec)
                files.append(out/'validation.npz')
            choices[model]={'temperature':val_temperature(frame,files,'score') if files else 1.}
        dump(selection_path,{'lock_sha256':lock_hash,'choices':choices});selections[fold]=choices
        print(json.dumps({'phase':'inner_selection_complete','fold':fold}),flush=True)
    return selections


def run_formal(lock,splits,selections):
    for fold,part in splits['folds'].items():
        if not part['test_ids']:continue
        for seed in lock['seeds']:
            for objective,names in OBJECTIVES.items():
                selected=selections[fold][objective]
                name=f'formal_{fold}_{objective}_s{seed}';out=ROOT/'runs/trajectories'/name
                spec={'project':str(ROOT),'train_ids':part['train_ids'],'val_ids':[],
                      'config':{**selected['config'],'seed':seed},'output_dir':str(out)}
                job(name,'formal','scripts/run_neural_job.py',spec,True)
                for model,filename in zip(names,['best.pt',selected['swa_candidate']+'.pt']):
                    pred={'project':str(ROOT),'ids':part['test_ids'],'model':model,'seed':seed,'fold':fold,
                          'checkpoint':str(out/filename),'temperature':selected['temperatures'][model],
                          'output':str(ROOT/'artifacts/sealed_predictions'/f'{fold}_{model}_s{seed}.parquet')}
                    job('predict_'+fold+'_'+model+f'_s{seed}','formal','scripts/predict_job.py',pred,True)
            for model in TREES:
                name=f'formal_{fold}_{model}_s{seed}';out=ROOT/'runs/tabular'/name
                spec={'project':str(ROOT),'train_ids':part['train_ids'],'val_ids':[],
                      'model':model,'seed':seed,'output_dir':str(out)}
                job(name,'formal','scripts/run_tabular_job.py',spec)
                pred={'project':str(ROOT),'ids':part['test_ids'],'model':model,'seed':seed,'fold':fold,
                      'checkpoint':str(out/'model.pkl'),'temperature':selections[fold][model]['temperature'],
                      'output':str(ROOT/'artifacts/sealed_predictions'/f'{fold}_{model}_s{seed}.parquet')}
                job('predict_'+fold+'_'+model+f'_s{seed}','formal','scripts/predict_job.py',pred)
        print(json.dumps({'phase':'formal_predictions_complete','fold':fold,'metrics_released':False}),flush=True)


def combine(lock,splits,frame):
    files=[]
    for fold,part in splits['folds'].items():
        if not part['test_ids']:continue
        for model,seed in itertools.product(lock['models'],lock['seeds']):
            file=ROOT/'artifacts/sealed_predictions'/f'{fold}_{model}_s{seed}.parquet'
            if not file.exists():raise ValueError(f'Missing prediction artifact {file}')
            files.append(file)
    predictions=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    from bh_transfer.evaluate import validate_predictions
    validate_predictions(predictions,lock,ROOT/'FINAL_EVAL_LOCK.json')
    # Independent label and chemical identity check against frozen records.
    unique=predictions.drop_duplicates('row_id').set_index('row_id')
    truth=frame.loc[unique.index]
    for a,b in [('y_true','y'),('source','source'),('aryl_id','aryl_id'),('amine_id','amine_id'),('pair_id','pair_id')]:
        if not np.array_equal(unique[a].to_numpy(),truth[b].to_numpy()):raise AssertionError('Prediction metadata mismatch '+a)
    predictions.to_parquet(ROOT/'artifacts/predictions.parquet',index=False)
    dump(ROOT/'artifacts/PREDICTION_COMPLETENESS.json',{'complete':True,'rows':len(predictions),
          'files':len(files),'lock_sha256':hashlib.sha256((ROOT/'FINAL_EVAL_LOCK.json').read_bytes()).hexdigest(),
          'predictions_sha256':hashlib.sha256((ROOT/'artifacts/predictions.parquet').read_bytes()).hexdigest(),
          'truth_and_chemistry_verified':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','development','formal','combine'],default='all');a=p.parse_args()
    lock=json.loads((ROOT/'FINAL_EVAL_LOCK.json').read_text())
    if lock['status']!='FROZEN':raise ValueError('Need frozen protocol')
    for relative,expected in lock['artifact_hashes'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen artifact changed: '+relative)
    splits=json.loads((ROOT/'data/processed/splits.json').read_text())
    frame=pd.read_parquet(ROOT/'data/processed/records.parquet').set_index('row_id',drop=False)
    if a.phase in ['all','development']:selections=run_development(lock,splits,frame)
    else:
        selections={}
        for f,x in splits['folds'].items():
            if not x['test_ids']:continue
            saved=json.loads((ROOT/'runs/selections'/(f+'.json')).read_text())
            if saved.get('lock_sha256')!=hashlib.sha256((ROOT/'FINAL_EVAL_LOCK.json').read_bytes()).hexdigest():
                raise ValueError('Selection cache belongs to another protocol')
            selections[f]=saved['choices']
    if a.phase in ['all','formal']:run_formal(lock,splits,selections)
    if a.phase in ['all','combine']:combine(lock,splits,frame)


if __name__=='__main__':main()
