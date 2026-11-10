#!/usr/bin/env python3
"""Audit saved training budgets and resumable states without test scores."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def inspect_run(directory, train_ids=None, expected_config=None):
    import torch
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text())
    state = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
    required = {'optimizer','scheduler','dro','swa','sampler','rng','descriptor_transform',
                'swa_snapshot_steps','sample_exposure','implementation_sha256'}
    if not required <= set(state):
        raise ValueError('Incomplete resumable state: '+str(required-set(state)))
    cfg = state['config']; n_groups = len(state['groups'])
    expected_steps = cfg['epochs']*math.ceil(len(state['train_ids'])/(cfg['batch_per_group']*n_groups))
    if cfg.get('max_steps') is not None: expected_steps=min(expected_steps,cfg['max_steps'])
    if not (state['completed_steps']==state['total_steps']==result['total_steps']==expected_steps):
        raise ValueError('Optimizer-update budget mismatch')
    if train_ids is not None:
        if state['train_ids'] != train_ids or len(state['val_ids']):
            raise ValueError('Final fit uses wrong training rows or external validation')
    if expected_config is not None:
        for key,value in expected_config.items():
            if cfg[key] != value:raise ValueError('Selected configuration mismatch: '+key)
    if not result['encoder_grad_verified']:raise ValueError('Encoder gradient unverified')
    if set(state['rng']) != {'python','numpy','torch','cuda'}:
        raise ValueError('Incomplete GPU random-state snapshot')
    exposure=int(np.asarray(state['sample_exposure']).sum())
    if exposure != expected_steps*cfg['batch_per_group']*n_groups:
        raise ValueError('Sample exposure and optimizer budget disagree')
    snapshots={}
    for name,average in state['swa'].items():
        steps=state['swa_snapshot_steps'][name]
        if int(average['n_averaged']) != len(steps) or len(steps)<2:
            raise ValueError('SWA is missing genuine multiple-snapshot averaging')
        if steps != sorted(set(steps)) or steps[-1] != expected_steps:
            raise ValueError('SWA snapshot ordering/end mismatch')
        snapshots[name]=len(steps)
    history=pd.read_csv(directory/'group_history.csv')
    if len(history)!=expected_steps*n_groups or history.duplicated(['step','source']).any():
        raise ValueError('Group log coverage is incomplete')
    weights=history.pivot(index='step',columns='source',values='weight')
    if not np.isfinite(weights).all().all() or not np.allclose(weights.sum(axis=1),1,atol=1e-5):
        raise ValueError('Group risk weights are not a probability distribution')
    if cfg['objective'] in ('bal','dro'):
        if not (history.batch_count==cfg['batch_per_group']).all():
            raise ValueError('Balanced effective batches miss a source or have unequal counts')
    if cfg['objective']=='bal' and not np.allclose(weights,1/n_groups,atol=1e-6):
        raise ValueError('Balanced ERM changed its group weights')
    if ((history.exposure_ess<0)|(history.exposure_ess>history.unique_exposed+1e-5)).any():
        raise ValueError('Invalid sample effective-size diagnostic')
    q=state['dro']['log_q'].double().softmax(0).numpy()
    if cfg['objective']=='dro' and not np.allclose(q,weights.iloc[-1].reindex(state['groups']),atol=1e-5):
        raise ValueError('Checkpoint group weights disagree with the final recorded update')
    final=history[history.step==expected_steps]
    return {'objective':cfg['objective'],'seed':cfg['seed'],'updates':expected_steps,
            'training_rows':len(state['train_ids']),'groups':n_groups,'exposures':exposure,
            'ordinary_step':result['ordinary_best_step'],'swa_050_snapshots':snapshots['swa_050'],
            'swa_075_snapshots':snapshots['swa_075'],'max_group_weight':float(weights.max().max()),
            'final_group_weight_ess':float(final.group_weight_ess.iloc[0]),
            'minimum_final_sample_ess':float(final.exposure_ess.min())},history


def audit_project(project):
    project=Path(project)
    lock=json.loads((project/'FINAL_EVAL_LOCK.json').read_text())
    splits=json.loads((project/'data/processed/splits.json').read_text())
    out=project/'reports/final';out.mkdir(parents=True,exist_ok=True)
    rows=[]; group_rows=[]
    for fold,part in splits['folds'].items():
        if not part['test_ids']:continue
        choices=json.loads((project/'runs/selections'/f'{fold}.json').read_text())['choices']
        for seed in lock['seeds']:
            for objective in ('erm','bal','dro'):
                directory=project/'runs/trajectories'/f'formal_{fold}_{objective}_s{seed}'
                expected={**choices[objective]['config'],'seed':seed}
                row,history=inspect_run(directory,part['train_ids'],expected)
                row.update(fold=fold,lr=expected['lr'],weight_decay=expected['weight_decay'],
                           ordinary_fraction=expected['ordinary_fraction'],
                           selected_swa=choices[objective]['swa_candidate'])
                rows.append(row)
                if objective=='dro':
                    # Preserve the last diagnostics and a fixed 100-point trace.
                    selected=np.unique(np.linspace(1,row['updates'],100).round().astype(int))
                    h=history[history.step.isin(selected)].copy()
                    h['fold']=fold;h['seed']=seed;h['training_fraction']=h.step/row['updates']
                    group_rows.append(h)
    table=pd.DataFrame(rows)
    if (table.groupby(['fold','seed']).updates.nunique()!=1).any():
        raise ValueError('ERM/balanced/DRO did not receive matching optimizer budgets')
    table.to_csv(out/'training_budget_audit.csv',index=False)
    pd.concat(group_rows,ignore_index=True).to_csv(out/'groupdro_trajectories.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    traces=pd.concat(group_rows,ignore_index=True)
    folds=sorted(traces.fold.unique())
    fig,axes=plt.subplots(len(folds),1,figsize=(9,3*len(folds)),squeeze=False)
    for ax,fold in zip(axes[:,0],folds):
        p=traces[traces.fold==fold]
        for source,g in p.groupby('source'):
            # Lines are separate seeds; no chemical-sample CI is implied.
            for seed,s in g.groupby('seed'):
                ax.plot(s.training_fraction,s.weight,alpha=.65,label=f'{source} / {seed}')
        ax.set(title=fold,xlabel='Training fraction',ylabel='GroupDRO weight',ylim=(0,1))
        ax.legend(fontsize=5,ncol=3)
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(out/f'groupdro_weight_trajectories.{suffix}',dpi=180)
    plt.close(fig)
    summary={'status':'PASS','formal_trajectories':len(rows),'matched_update_budgets':True,
             'resumable_states_verified':True,'all_swa_multiple_snapshots':True,
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'training_audit.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',default='.')
    args=parser.parse_args();print(json.dumps(audit_project(args.project),indent=2))
