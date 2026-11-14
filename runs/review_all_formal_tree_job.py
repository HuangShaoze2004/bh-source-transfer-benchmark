import json,pickle,datetime
from pathlib import Path
import numpy as np
root=Path.cwd();lock=json.loads((root/"FINAL_EVAL_LOCK.json").read_text());splits=json.loads((root/"data/processed/splits.json").read_text())
with np.load(root/"data/processed/features.npz") as a:features={k:a[k].copy() for k in a.files}
pos={int(x):i for i,x in enumerate(features["row_ids"])};rows=[]
for fold,part in splits["folds"].items():
 if not part["test_ids"]:continue
 ti=[pos[int(i)] for i in part["train_ids"]];n=len(ti)
 medians={}
 for feature in ["drfp","author_substr"]:
  ones=features[feature][ti].sum(axis=0,dtype=np.int64)
  medians[feature]=np.where(2*ones>n,1.,np.where(2*ones==n,.5,0.))
 expected_rdkit=np.median(features["descriptors"][ti],axis=0)
 for seed in lock["seeds"]:
  for model in ["R0_RF","R0_HGB","R1_substitute"]:
   name=f"formal_{fold}_{model}_s{seed}";directory=root/"runs/tabular"/name
   r=json.loads((directory/"result.json").read_text());status=json.loads((root/"runs"/name/"job_status.json").read_text())
   assert status["status"]=="completed" and status["exit_code"]==0
   assert r["train_ids"]==part["train_ids"] and r["val_ids"]==[] and r["label"]=="y"
   assert r["seed"]==seed and r["model"]==model
   with np.load(directory/"validation.npz") as z:assert len(z["row_ids"])==len(z["score"])==0
   with open(directory/"model.pkl","rb") as f:pipe=pickle.load(f)
   expected=np.r_[medians["author_substr"],expected_rdkit] if model=="R1_substitute" else medians["drfp"]
   assert pipe.n_features_in_==len(expected)
   np.testing.assert_allclose(pipe.steps[0][1].statistics_,expected,rtol=1e-6,atol=1e-6)
   e=pipe.steps[-1][1];assert e.random_state==seed and np.array_equal(e.classes_,[0,1])
   if model=="R0_HGB":
    assert e.max_iter==e.n_iter_==200 and e.early_stopping is False
    assert e.learning_rate==.1 and e.max_leaf_nodes==31 and e.l2_regularization==1.
   else:
    assert len(e.estimators_)==e.n_estimators==(500 if model=="R1_substitute" else 300)
    assert e.class_weight=="balanced" and e.criterion=="entropy" and e.max_features=="log2"
    assert e.min_samples_split==4 and e.min_samples_leaf==1 and e.max_depth is None and e.n_jobs==1 and e.bootstrap
   rows.append(dict(run_id=name,fold=fold,seed=seed,model=model,n_training=n,n_features=len(expected),train_imputer_statistics_verified=True))
assert len(rows)==45
out={"status":"PASS","checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":"all45 locked formal tree fits; no external predictions or metrics read","runs_verified":len(rows),"checks":["full locked outer development IDs","no final-fit validation IDs or predictions","three prescribed seeds","estimator parameters and fitted tree/iteration counts","training-only imputer statistics independently reconstructed","feature dimensions and class identities"],"runs":rows}
(root/"reports/ALL_FORMAL_TREE_RUNTIME_AUDIT.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!="runs"}))
