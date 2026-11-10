#!/usr/bin/env python3
"""Run a single explicitly specified training trajectory; never score test data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from bh_transfer.neural import train_trajectory, predict_checkpoint


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


def job_identity(project, spec):
    project = Path(project)
    inputs = {'schema_version': 1,
              'spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              'artifacts': {name: file_sha256(project / name) for name in IDENTITY_ARTIFACTS},
              'job_script_sha256': file_sha256(__file__)}
    # A preflight job may precede the evaluation lock; formal jobs bind the lock.
    lock_path = project / 'FINAL_EVAL_LOCK.json'
    inputs['evaluation_lock_sha256'] = file_sha256(lock_path) if lock_path.exists() else None
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(), inputs


def verify_completed_result(out, expected_identity):
    out = Path(out)
    result = json.loads((out / 'result.json').read_text())
    if result.get('job_identity') != expected_identity:
        raise ValueError('Completed neural result identity differs or is missing; preserve it and use a new run directory')
    artifacts = result.get('output_artifact_hashes')
    if not artifacts:
        raise ValueError('Completed neural result lacks verified output artifacts')
    for name, expected in artifacts.items():
        path = (out / name).resolve()
        if not path.is_relative_to(out.resolve()) or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f'Completed neural output is missing or changed: {name}')
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--spec',required=True); a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text()); project=Path(spec['project'])
    out=Path(spec['output_dir']); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(os.environ.get('BH_PROJECT_THREADS','1')))
    torch.set_num_interop_threads(1)
    identity, identity_inputs = job_identity(project, spec)
    if (out/'result.json').exists():
        verify_completed_result(out, identity)
        print('Trajectory already complete; verified identity and outputs before reuse.',flush=True); return
    if spec['config']['device']=='cuda': torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    result=train_trajectory(project,spec['train_ids'],spec.get('val_ids',[]),spec['config'],out)
    for candidate in result['candidates']:
        name=candidate['name']
        np.savez_compressed(out/(name+'_validation.npz'),row_ids=candidate.pop('val_ids'),
                            logits=candidate.pop('val_logits'))
    if spec.get('fit_diagnostic'):
        # Fitting only, not an outer generalization estimate.
        import pandas as pd
        from sklearn.metrics import log_loss,roc_auc_score
        from scipy.special import expit
        ids=spec['train_ids']
        logits=predict_checkpoint(project,ids,out/'best.pt')
        frame=pd.read_parquet(project/'data/processed/records.parquet').set_index('row_id')
        y=frame.loc[ids,'y'].to_numpy()
        result['training_fit_diagnostic']={'n':len(ids),'bce':float(log_loss(y,expit(logits),labels=[0,1])),
                                          'auroc':float(roc_auc_score(y,logits))}
    if spec.get('benchmark_inference_ids'):
        ids=spec['benchmark_inference_ids']; t=time.monotonic()
        predict_checkpoint(project,ids,out/'best.pt')
        result['inference_rows']=len(ids); result['inference_seconds']=time.monotonic()-t
    result['wall_seconds']=time.monotonic()-start
    result['peak_gpu_bytes']=int(torch.cuda.max_memory_allocated()) if spec['config']['device']=='cuda' else 0
    result['job_identity'] = identity
    result['identity_inputs'] = identity_inputs
    artifact_paths = [Path(c['checkpoint']) for c in result['candidates']]
    artifact_paths += [out / (c['name'] + '_validation.npz') for c in result['candidates']]
    artifact_paths.append(Path(result['resume_checkpoint']))
    result['output_artifact_hashes'] = {
        str(path.resolve().relative_to(out.resolve())): file_sha256(path) for path in artifact_paths
    }
    tmp=out/'result.json.tmp'; tmp.write_text(json.dumps(result,indent=2)); tmp.replace(out/'result.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ['candidates']},indent=2),flush=True)


if __name__=='__main__': main()
