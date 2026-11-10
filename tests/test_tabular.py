import numpy as np
import pandas as pd
import pytest
from bh_transfer.tabular import fit,predict


def dataset(tmp_path):
    p=tmp_path/'data/processed';p.mkdir(parents=True)
    ids=np.array([40,20,90,10,80,60,70,30])
    y=np.array([0,1,0,1,0,1,0,1])
    pd.DataFrame({'row_id':ids,'source':['a']*6+['b']*2,'y':y,
                  'author_label':y}).to_parquet(p/'records.parquet',index=False)
    rng=np.random.default_rng(9);x=rng.integers(0,2,(8,2048),dtype=np.uint8)
    d=rng.normal(size=(8,21)).astype(np.float32);d[0,0]=np.nan
    np.savez(p/'features.npz',row_ids=ids,drfp=x,author_substr=x,descriptors=d)
    return ids


def test_reuse_rejects_changed_identity(tmp_path):
    ids=dataset(tmp_path)
    r=fit(tmp_path,ids[:6].tolist(),ids[6:].tolist(),'R1_substitute',17,tmp_path/'fit')
    assert r['identity']
    assert np.isfinite(predict(tmp_path,ids[6:],'R1_substitute',r['checkpoint'])).all()
    assert fit(tmp_path,ids[:6].tolist(),ids[6:].tolist(),'R1_substitute',17,tmp_path/'fit')['identity']==r['identity']
    with pytest.raises(ValueError,match='identity'):
        fit(tmp_path,ids[:6].tolist(),ids[6:].tolist(),'R1_substitute',42,tmp_path/'fit')


def test_train_only_imputation(tmp_path):
    ids=dataset(tmp_path)
    r=fit(tmp_path,ids[:6].tolist(),ids[6:].tolist(),'R0_RF',17,tmp_path/'fit')
    import pickle
    with open(r['checkpoint'],'rb') as f: fitted=pickle.load(f)
    with np.load(tmp_path/'data/processed/features.npz') as f:expected=np.median(f['drfp'][:6],axis=0)
    np.testing.assert_array_equal(fitted.steps[0][1].statistics_,expected)
