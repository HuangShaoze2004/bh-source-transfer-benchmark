#!/usr/bin/env python3
"""Independently inspect all prescribed sensitivity RF fits, without inference.

Does not import the tabular training implementation. Requires full fit coverage;
checks locked inputs, training-only imputation and actual fitted RF parameters.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit():
    main_hash = sha256(ROOT / 'FINAL_EVAL_LOCK.json')
    verified = []
    for kind in ('author_labels', 'dd_original'):
        project = ROOT / 'sensitivity' / kind
        lock = json.loads((project / 'FINAL_EVAL_LOCK.json').read_text())
        require(lock['status'] == 'FROZEN' and lock['main_lock_sha256'] == main_hash,
                'Sensitivity lock does not inherit the frozen main protocol')
        for name, expected in lock['artifact_hashes'].items():
            require(sha256(project / name) == expected, 'Changed sensitivity artifact: ' + name)
        for name, expected in lock['source_code_hashes'].items():
            require(sha256(name) == expected, 'Changed sensitivity code dependency: ' + name)
        splits = json.loads((project / 'data/processed/splits.json').read_text())
        records_path = project / 'data/processed/records.parquet'
        records = pd.read_parquet(records_path)
        require(not records.row_id.duplicated().any(), 'Repeated record row identifiers')
        target = records.author_label if kind == 'author_labels' else (records.yield_raw >= 10).astype(int)
        require(np.array_equal(records.y.to_numpy(), target.to_numpy()), 'Sensitivity label mismatch')
        with np.load(project / 'data/processed/features.npz') as features:
            row_ids = features['row_ids'].astype(int)
            matrix = np.concatenate([features['author_substr'], features['descriptors']], axis=1).astype(np.float32)
        require(len(set(row_ids)) == len(row_ids) and set(row_ids) == set(records.row_id),
                'Feature row coverage differs from records')
        require(matrix.shape == (len(row_ids), 2069), 'Expected2048 substrate bits plus21 reagent descriptors')
        positions = {int(row): i for i, row in enumerate(row_ids)}
        feature_hash = hashlib.sha256(matrix.tobytes()).hexdigest()
        records_hash = sha256(records_path)
        for fold, part in splits['folds'].items():
            if not part['test_ids']:
                continue
            train_ids = part['train_ids']
            require(set(train_ids) == set(records.loc[records.source != part['source'], 'row_id']),
                    'RF training is not exactly the other sources')
            train = matrix[[positions[int(row)] for row in train_ids]]
            medians = np.nanmedian(train, axis=0)
            medians = np.where(np.isnan(medians), 0., medians)
            for seed in (17, 42, 2026):
                name = f'{fold}_R1_substitute_s{seed}'
                run_id = f'sens_{kind}_{name}'
                out = project / 'runs/tabular' / name
                result = json.loads((out / 'result.json').read_text())
                spec = json.loads((project / 'runs/specs' / (run_id + '.json')).read_text())
                status = json.loads((ROOT / 'runs' / run_id / 'job_status.json').read_text())
                for saved in (result, spec):
                    require(saved['train_ids'] == train_ids and saved['val_ids'] == [], 'Wrong final RF partition')
                    require(saved['seed'] == seed and saved['model'] == 'R1_substitute', 'Wrong RF method/seed')
                require(result['label'] == 'y' and result['val_metrics'] == {}, 'Unexpected final RF validation/target')
                require(status['status'] == 'completed' and not status['gpu'] and status['threads'] == 1,
                        'RF fit lacks successful one-thread CPU guard completion')
                identity_data = dict(train_ids=list(map(int, train_ids)), val_ids=[], model='R1_substitute',
                                     seed=seed, label='y', features_sha256=feature_hash,
                                     records_sha256=records_hash,
                                     code_sha256=sha256(ROOT / 'src/bh_transfer/tabular.py'))
                identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
                require(result['identity'] == identity, 'Saved fit does not bind frozen features/records/IDs/code')
                checkpoint = out / 'model.pkl'
                require(Path(result['checkpoint']) == checkpoint, 'Unexpected RF checkpoint path')
                with checkpoint.open('rb') as stream:
                    fitted = pickle.load(stream)  # Trusted local artifacts produced by this project.
                imputer, forest = fitted.steps[0][1], fitted.steps[1][1]
                require(isinstance(imputer, SimpleImputer) and imputer.strategy == 'median'
                        and imputer.keep_empty_features, 'Unexpected RF preprocessing')
                require(np.allclose(imputer.statistics_, medians, rtol=0, atol=0, equal_nan=True),
                        'Imputer statistics differ from independently reconstructed training medians')
                require(isinstance(forest, RandomForestClassifier), 'Not a fitted RF classifier')
                expected = dict(n_estimators=500, max_depth=None, min_samples_split=4, min_samples_leaf=1,
                                criterion='entropy', class_weight='balanced', max_features='log2',
                                random_state=seed, n_jobs=1)
                require(all(forest.get_params()[key] == value for key, value in expected.items()),
                        'Fitted RF differs from the fixed strong substitute')
                require(forest.n_features_in_ == 2069 and len(forest.estimators_) == 500
                        and np.array_equal(forest.classes_, [0, 1]), 'Unexpected fitted RF dimensions/classes/trees')
                with np.load(out / 'validation.npz') as validation:
                    require(len(validation['row_ids']) == len(validation['score']) == 0,
                            'Final RF fit unexpectedly contains validation predictions')
                verified.append(dict(run_id=run_id, training_rows=len(train_ids),
                                     checkpoint_sha256=sha256(checkpoint), result_sha256=sha256(out / 'result.json')))
                del fitted, imputer, forest
    require(len(verified) == 30, 'Expected all30 prespecified sensitivity RF fits')
    report = dict(status='PASS', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  scope='all30 sensitivity RF fits; no external predictions or metrics read',
                  main_lock_sha256=main_hash, runs_verified=len(verified), runs=verified)
    destination = ROOT / 'reports/ALL_SENSITIVITY_TREE_RUNTIME_AUDIT.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(destination)
    print(json.dumps(dict(status='PASS', runs_verified=len(verified), report=str(destination))))


if __name__ == '__main__':
    audit()
