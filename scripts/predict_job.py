#!/usr/bin/env python3
"""Save sealed probabilities and metadata; no test performance computation."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit


IDENTITY_ARTIFACTS = (
    'data/processed/records.parquet', 'data/processed/features.npz',
    'src/bh_transfer/neural.py', 'src/bh_transfer/tokenizer.py', 'src/bh_transfer/tabular.py',
    'weights/rxnfp/manifest.json', 'weights/rxnfp/tokenizer_extension_manifest.json',
    'weights/rxnfp/pytorch_model.bin', 'weights/rxnfp/config.json',
    'weights/rxnfp/vocab.txt', 'weights/rxnfp/extended_vocab.txt',
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_lock(root):
    root = Path(root)
    lock = json.loads((root / 'FINAL_EVAL_LOCK.json').read_text())
    if lock.get('status') != 'FROZEN':
        raise ValueError('Protocol is not frozen')
    for relative, expected in lock.get('artifact_hashes', {}).items():
        if file_sha256(root / relative) != expected:
            raise ValueError('Frozen artifact changed: ' + relative)
    return lock


def prediction_identity(root, spec):
    root = Path(root)
    # Lock validation precedes every cache reuse, including matching old specs.
    checked_lock(root)
    inputs = {'schema_version': 1,
              'spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              'artifacts': {name: file_sha256(root / name) for name in IDENTITY_ARTIFACTS},
              'evaluation_lock_sha256': file_sha256(root / 'FINAL_EVAL_LOCK.json'),
              'checkpoint_sha256': file_sha256(spec['checkpoint']),
              'job_script_sha256': file_sha256(__file__)}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(), inputs


def sidecar_path(out):
    out = Path(out)
    return out.with_name(out.name + '.identity.json')


def verify_existing_prediction(out, identity):
    out = Path(out)
    sidecar = sidecar_path(out)
    if not out.exists() and not sidecar.exists():
        return False
    if not out.is_file() or not sidecar.is_file():
        raise ValueError('Prediction artifact/identity sidecar is incomplete; preserve it and use a new output path')
    saved = json.loads(sidecar.read_text())
    if saved.get('job_identity') != identity:
        raise ValueError('Prediction identity differs or is missing; preserve it and use a new output path')
    if saved.get('prediction_sha256') != file_sha256(out):
        raise ValueError('Prediction artifact differs from its recorded hash')
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args()
    s=json.loads(Path(a.spec).read_text()); root=Path(s['project']); out=Path(s['output'])
    identity, identity_inputs = prediction_identity(root, s)
    if verify_existing_prediction(out, identity):
        print('Prediction artifact identity, frozen inputs and bytes verified; reused');return
    frame=pd.read_parquet(root/'data/processed/records.parquet').set_index('row_id',drop=False)
    ids=s['ids']; rows=frame.loc[ids]
    if s['model'].startswith('R'):
        from bh_transfer.tabular import predict
        score=predict(root,ids,s['model'],s['checkpoint'])
        logits=logit(np.clip(score,1e-7,1-1e-7))
    else:
        import torch
        torch.set_num_threads(int(os.environ.get('BH_PROJECT_THREADS','1')));torch.set_num_interop_threads(1)
        from bh_transfer.neural import predict_checkpoint
        logits=predict_checkpoint(root,ids,s['checkpoint']); score=expit(logits)
    result=rows[['row_id','source','pair_id','aryl_id','amine_id','input_key']].copy().rename(columns={'input_key':'condition_id'})
    result['y_true']=rows[s.get('label','y')].to_numpy(int)
    result['model']=s['model'];result['seed']=s['seed'];result['fold']=s['fold']
    result['score']=score;result['logit']=logits
    result['cal_score']=expit(logits/float(s.get('temperature',1.)))
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.tmp.parquet');result.to_parquet(tmp,index=False);tmp.replace(out)
    sidecar = sidecar_path(out)
    sidecar_tmp = sidecar.with_name(sidecar.name + '.tmp')
    sidecar_tmp.write_text(json.dumps({'job_identity': identity, 'identity_inputs': identity_inputs,
                                     'prediction_sha256': file_sha256(out), 'n_rows': len(result)}, indent=2))
    sidecar_tmp.replace(sidecar)
    print(json.dumps({'output':str(out),'n_rows':len(result),'finite':bool(np.isfinite(score).all())}))


if __name__=='__main__':main()
