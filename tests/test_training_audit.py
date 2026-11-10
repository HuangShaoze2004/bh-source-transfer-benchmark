import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def fixture(tmp_path):
    spec=importlib.util.spec_from_file_location('training_auditor',Path(__file__).resolve().parents[1]/'scripts/audit_training.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    cfg={'epochs':4,'batch_per_group':2,'objective':'dro','seed':17}
    state={'config':cfg,'groups':['a','b'],'train_ids':[1,2,3,4],'val_ids':[],
           'completed_steps':4,'total_steps':4,'sample_exposure':np.array([4,4,4,4]),
           'rng':dict(python=None,numpy=None,torch=None,cuda=None),
           'dro':{'log_q':torch.tensor([.25,.75]).log()},
           'swa':{name:{'n_averaged':torch.tensor(2)} for name in ['swa_050','swa_075']},
           'swa_snapshot_steps':{name:[3,4] for name in ['swa_050','swa_075']},
           'optimizer':{},'scheduler':{},'sampler':{},'descriptor_transform':{},'implementation_sha256':{}}
    torch.save(state,tmp_path/'last.pt')
    result={'total_steps':4,'encoder_grad_verified':True,'ordinary_best_step':2}
    (tmp_path/'result.json').write_text(json.dumps(result))
    history=pd.DataFrame([{'step':s,'source':g,'weight':w,'batch_count':2,'exposure_ess':2.,
        'unique_exposed':2,'group_weight_ess':1.6} for s in range(1,5) for g,w in [('a',.25),('b',.75)]])
    history.to_csv(tmp_path/'group_history.csv',index=False)
    return audit,tmp_path,state,history


def test_audits_actual_updates_and_multiple_swa_snapshots(fixture):
    audit,path,_,_=fixture
    summary,_=audit.inspect_run(path,[1,2,3,4],{'seed':17})
    assert summary['updates']==4 and summary['swa_075_snapshots']==2


@pytest.mark.parametrize('fault',['updates','swa','rng','val_rows'])
def test_rejects_incomplete_state_or_wrong_budget(fixture,fault):
    audit,path,state,_=fixture
    if fault=='updates':state['completed_steps']=3
    if fault=='swa':state['swa']['swa_075']['n_averaged']=torch.tensor(1)
    if fault=='rng':state['rng'].pop('cuda')
    if fault=='val_rows':state['val_ids']=[99]
    torch.save(state,path/'last.pt')
    with pytest.raises(ValueError):audit.inspect_run(path,[1,2,3,4])


def test_rejects_group_weight_or_missing_group_log(fixture):
    audit,path,_,history=fixture
    history.loc[0,'weight']=.9
    history.to_csv(path/'group_history.csv',index=False)
    with pytest.raises(ValueError,match='probability'):audit.inspect_run(path)
    history.iloc[1:].to_csv(path/'group_history.csv',index=False)
    with pytest.raises(ValueError,match='coverage'):audit.inspect_run(path)
