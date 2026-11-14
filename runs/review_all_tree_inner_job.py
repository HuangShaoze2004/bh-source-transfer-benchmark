import json,pickle,datetime
from pathlib import Path
import numpy as np
root=Path.cwd();splits=json.loads((root/"data/processed/splits.json").read_text());rows=[]
for fold,part in splits["folds"].items():
 if not part["test_ids"]:continue
 for model in ["R0_RF","R0_HGB","R1_substitute"]:
  for ii,inner in enumerate(part["inner"]):
   name=f"dev_{fold}_{model}_i{ii}";directory=root/"runs/tabular"/name
   r=json.loads((directory/"result.json").read_text())
   assert r["train_ids"]==inner["train_ids"] and r["val_ids"]==inner["val_ids"]
   assert r["seed"]==42 and r["model"]==model and r["label"]=="y"
   assert not set(r["train_ids"])&set(r["val_ids"])
   status=json.loads((root/"runs"/name/"job_status.json").read_text());assert status["status"]=="completed" and status["exit_code"]==0
   with np.load(directory/"validation.npz") as z:
    assert np.array_equal(z["row_ids"],inner["val_ids"])
    assert np.isfinite(z["score"]).all() and ((z["score"]>=0)&(z["score"]<=1)).all()
   with open(directory/"model.pkl","rb") as f:pipe=pickle.load(f)
   n_features=2069 if model=="R1_substitute" else 2048
   assert pipe.n_features_in_==n_features
   estimator=pipe.steps[-1][1];assert estimator.random_state==42
   if model=="R0_HGB":assert estimator.max_iter==200 and not estimator.early_stopping
   else:
    assert estimator.n_estimators==(500 if model=="R1_substitute" else 300)
    assert estimator.class_weight=="balanced" and estimator.criterion=="entropy"
    assert estimator.max_features=="log2" and estimator.min_samples_split==4 and estimator.n_jobs==1
   rows.append(dict(run_id=name,fold=fold,model=model,n_training=len(inner["train_ids"]),n_validation=len(inner["val_ids"]),n_features=n_features))
assert len(rows)==30
out={"status":"PASS","checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":"all locked inner tree controls; no external metrics","runs_verified":len(rows),"checks":["exact locked train and validation IDs","completed successful guard state","validation row order and finite bounded probabilities","model dimensionality and pinned estimator parameters","one CPU worker per fit"],"runs":rows}
(root/"reports/ALL_TREE_INNER_RUNTIME_AUDIT.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!="runs"}))
