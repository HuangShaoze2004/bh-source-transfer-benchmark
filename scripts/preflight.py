#!/usr/bin/env python3
"""Create and run bounded fitting/throughput diagnostics on development rows."""
import argparse,json,math,os,subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PY=str(ROOT/'.venv/bin/python')


def run(name,ids,config,fit=False,inference_ids=None):
    spec={'project':str(ROOT),'train_ids':list(map(int,ids)),'val_ids':[],
          'config':config,'output_dir':str(ROOT/'runs/trajectories'/name),
          'fit_diagnostic':fit,'benchmark_inference_ids':list(map(int,inference_ids or []))}
    path=ROOT/'runs/specs'/(name+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(spec,indent=2))
    subprocess.run([PY,'scripts/run_guard.py','--run-id',name,'--stage','feasibility','--gpu','--threads','1',
                    '--',PY,'scripts/run_neural_job.py','--spec',str(path)],cwd=ROOT,check=True)
    return json.loads((ROOT/'runs/trajectories'/name/'result.json').read_text())


def main():
    frame=pd.read_parquet(ROOT/'data/processed/records.parquet').set_index('row_id',drop=False)
    splits=json.loads((ROOT/'data/processed/splits.json').read_text())
    # No outer test labels or performance are used. Choose first lexical fold;
    # diagnostics see its inner training portion only.
    fold,part=next((f,p) for f,p in sorted(splits['folds'].items()) if p['test_ids'] and p['inner'])
    pool=frame.loc[part['inner'][0]['train_ids']]
    small=[]
    for (_,label),g in pool.groupby(['source','y']):
        small.extend(g.sample(n=min(4,len(g)),random_state=42).row_id.tolist())
    base={'objective':'erm','seed':42,'lr':3e-4,'weight_decay':.01,'epochs':100,
          'max_steps':160,'batch_per_group':8,'microbatch_size':8,'device':'cuda',
          'ordinary_fraction':1.,'checkpoint_every_steps':50}
    fitting=run('feasibility_small_fit',small,base,True)
    if fitting['training_fit_diagnostic']['auroc']<.9:
        raise RuntimeError('Small-sample fitting gate failed; diagnose before experiments')
    ids=pool.row_id.tolist(); results={}
    # Same representative pool and batch scale for both cost profiles.
    for objective in ['erm','dro']:
        config={**base,'objective':objective,'epochs':1,'max_steps':120,'lr':3e-5}
        results[objective]=run('feasibility_throughput_'+objective,ids,config,
                               inference_ids=ids[:min(1024,len(ids))])
    summary={'diagnostic_fold':fold,'used_only_inner_training_rows':True,
             'small_fit':fitting,'throughput':results}
    (ROOT/'artifacts/FEASIBILITY.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'fit_gate':'passed','profiles_complete':list(results)}))


if __name__=='__main__':main()
