"""Fixed strong tabular controls with training-only preprocessing."""
from pathlib import Path
import json
import os
import pickle
import time
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, log_loss


def load_features(project, model):
    project=Path(project)
    frame=pd.read_parquet(project/'data/processed/records.parquet').set_index('row_id',drop=False)
    with np.load(project/'data/processed/features.npz') as f:
        ids=f['row_ids'].astype(int)
        x=(np.concatenate([f['author_substr'],f['descriptors']],axis=1)
           if model=='R1_substitute' else f['drfp'])
    if len(set(ids))!=len(ids): raise ValueError('Duplicate feature row IDs')
    return frame, x.astype(np.float32), {int(v):i for i,v in enumerate(ids)}


def build(model,seed,threads=1):
    if model=='R0_HGB':
        estimator=HistGradientBoostingClassifier(learning_rate=.1,max_iter=200,max_leaf_nodes=31,
             l2_regularization=1.,early_stopping=False,random_state=seed)
    else:
        estimator=RandomForestClassifier(n_estimators=500 if model=='R1_substitute' else 300,
             max_depth=None,min_samples_split=4,min_samples_leaf=1,criterion='entropy',
             class_weight='balanced',max_features='log2',random_state=seed,n_jobs=threads)
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),estimator)


def fit(project,train_ids,val_ids,model,seed,out_dir,label='y'):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    completed=out/'result.json'
    df,x,pos=load_features(project,model)
    identity_data={'train_ids':list(map(int,train_ids)),'val_ids':list(map(int,val_ids)),
                   'model':model,'seed':int(seed),'label':label,
                   'features_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                   'records_sha256':hashlib.sha256((Path(project)/'data/processed/records.parquet').read_bytes()).hexdigest(),
                   'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    identity=hashlib.sha256(json.dumps(identity_data,sort_keys=True).encode()).hexdigest()
    if completed.exists():
        previous=json.loads(completed.read_text())
        if previous.get('identity')!=identity:raise ValueError('Existing tabular fit identity differs; use another run directory')
        return previous
    ti=np.array([pos[int(i)] for i in train_ids]); vi=np.array([pos[int(i)] for i in val_ids])
    if set(train_ids)&set(val_ids): raise ValueError('Train/validation intersection')
    y=df.loc[train_ids,label].to_numpy(int)
    if len(np.unique(y))<2: raise ValueError('One-class training set')
    estimator=build(model,seed,int(os.getenv('BH_PROJECT_THREADS','1')))
    t=time.time(); estimator.fit(x[ti],y)
    with (out/'model.pkl').open('wb') as f: pickle.dump(estimator,f,protocol=5)
    scores=estimator.predict_proba(x[vi])[:,1] if len(vi) else np.empty(0)
    np.savez_compressed(out/'validation.npz',row_ids=np.array(val_ids),score=scores)
    metrics={}
    if len(vi):
        for source,g in df.loc[val_ids].groupby('source'):
            score_map=dict(zip(map(int,val_ids),scores))
            yy=g[label].to_numpy(int); pp=np.array([score_map[int(i)] for i in g.row_id])
            metrics[source]={'auroc':float(roc_auc_score(yy,pp)) if len(set(yy))==2 else None,
                             'log_loss':float(log_loss(yy,pp,labels=[0,1]))}
    result={'model':model,'seed':seed,'train_ids':list(map(int,train_ids)),
            'val_ids':list(map(int,val_ids)),'checkpoint':str(out/'model.pkl'),
            'elapsed_seconds':time.time()-t,'val_metrics':metrics,'label':label,'identity':identity}
    completed.write_text(json.dumps(result,indent=2)); return result


def predict(project,ids,model,checkpoint):
    _,x,pos=load_features(project,model)
    with open(checkpoint,'rb') as f: estimator=pickle.load(f)
    return estimator.predict_proba(x[[pos[int(i)] for i in ids]])[:,1]
