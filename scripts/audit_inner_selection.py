#!/usr/bin/env python3
"""Verify a completed inner selection without accessing any outer predictions."""
import argparse
import datetime as dt
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np

import run_guard as guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock_path = root / 'FINAL_EVAL_LOCK.json'
    lock = json.loads(lock_path.read_text())
    splits = json.loads((root / 'data/processed/splits.json').read_text())
    part = splits['folds'][args.fold]
    selection_path = root / 'runs/selections' / (args.fold + '.json')
    selected = json.loads(selection_path.read_text())
    assert selected['lock_sha256'] == guard.sha256_file(lock_path)
    grids = list(range(len(lock['neural_grid'])))
    steps, checked = {}, {}
    model_names = {'erm': ('B', 'B_S'), 'bal': ('B_bal', 'B_bal_S'), 'dro': ('B_A', 'B_A_S')}
    for objective, names in model_names.items():
        choice = selected['choices'][objective]
        assert choice['inner_valid'] and choice['common_completed_grids'] == grids
        results = {}
        for grid in grids:
            results[grid] = []
            for ii, inner in enumerate(part['inner']):
                run_id = f'dev_{args.fold}_{objective}_g{grid}_i{ii}'
                spec = json.loads((root / 'runs/specs' / (run_id + '.json')).read_text())
                state = json.loads((root / 'runs' / run_id / 'job_status.json').read_text())
                directory = root / 'runs/trajectories' / run_id
                result_path = directory / 'result.json'
                result = json.loads(result_path.read_text())
                assert state['status'] == 'completed' and state['exit_code'] == 0
                assert spec['train_ids'] == inner['train_ids'] and spec['val_ids'] == inner['val_ids']
                expected = {**lock['neural_base_config'], **lock['neural_grid'][grid], 'objective': objective, 'seed': 42}
                assert spec['config'] == expected
                assert result['status'] == 'complete' and result['encoder_grad_verified']
                assert result['completed_steps'] == result['total_steps']
                assert result['identity_inputs']['evaluation_lock_sha256'] == selected['lock_sha256']
                for name, digest in result['output_artifact_hashes'].items():
                    path = (directory / name).resolve()
                    assert path.is_relative_to(directory.resolve()) and guard.sha256_file(path) == digest
                for candidate in result['candidates']:
                    with np.load(directory / (candidate['name'] + '_validation.npz'), allow_pickle=False) as predictions:
                        assert predictions['row_ids'].tolist() == inner['val_ids']
                        assert np.isfinite(predictions['logits']).all()
                steps.setdefault((grid, ii), []).append(result['total_steps'])
                checked[run_id] = guard.sha256_file(result_path)
                results[grid].append(result)
        def score(rows, name):
            return fmean(next(c['val_metrics']['auroc'] for c in row['candidates'] if c['name'] == name) for row in rows)
        best = max(grids, key=lambda grid: (score(results[grid], 'ordinary_best'), -grid))
        assert choice['grid_index'] == best
        fraction = fmean(r['ordinary_best_step'] / r['total_steps'] for r in results[best])
        assert math.isclose(choice['config']['ordinary_fraction'], fraction, abs_tol=1e-12)
        assert {key: value for key, value in choice['config'].items() if key != 'ordinary_fraction'} == {
            **lock['neural_base_config'], **lock['neural_grid'][best], 'objective': objective}
        swa = max(['swa_050', 'swa_075'], key=lambda name: (score(results[best], name), name))
        assert choice['swa_candidate'] == swa
        assert set(choice['temperatures']) == set(names)
        assert all(math.isfinite(t) and t > 0 for t in choice['temperatures'].values())
    assert all(len(values) == 3 and len(set(values)) == 1 for values in steps.values())
    for model in ['R0_RF', 'R0_HGB', 'R1_substitute']:
        temperature = selected['choices'][model]['temperature']
        assert math.isfinite(temperature) and temperature > 0
    report = {'status': 'PASS', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'fold': args.fold,
              'scope': 'complete inner candidates, saved artifacts, matched updates, deterministic selection and calibration parameter validity; no outer predictions read',
              'selection_sha256': guard.sha256_file(selection_path), 'lock_sha256': selected['lock_sha256'],
              'completed_neural_trajectories': len(checked), 'result_hashes': checked}
    guard.atomic_json(root / args.out, report)
    print(json.dumps({key: value for key, value in report.items() if key != 'result_hashes'}))


if __name__ == '__main__':
    main()
