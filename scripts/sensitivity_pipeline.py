#!/usr/bin/env python3
"""Run prespecified label/additive sensitivities without selecting on test scores.

All jobs use the MAIN project's resource guard and ledger. The two alternative
data roots receive separate immutable locks, specs, checkpoints and predictions.
The shared sensitivity GPU cap includes failed, repeated and live attempts.
Exhausting this auxiliary budget records incomplete coverage and exits normally,
so it does not prevent the main experiment's independent final evaluation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / '.venv/bin/python')
KINDS = ('author_labels', 'dd_original')
MODELS = ('R1_substitute', 'B', 'B_S', 'B_A', 'B_A_S')
SEEDS = (17, 42, 2026)
OBJECTIVES = {'erm': ('B', 'B_S'), 'dro': ('B_A', 'B_A_S')}
MAX_GPU_HOURS = 30.
SOURCE_FILES = (
    'src/bh_transfer/neural.py', 'src/bh_transfer/tokenizer.py',
    'src/bh_transfer/tabular.py', 'src/bh_transfer/data.py', 'src/bh_transfer/evaluate.py',
    'scripts/run_neural_job.py', 'scripts/run_tabular_job.py', 'scripts/predict_job.py',
    'scripts/run_guard.py', 'scripts/sensitivity_pipeline.py',
)
WEIGHT_FILES = ('manifest.json', 'tokenizer_extension_manifest.json', 'pytorch_model.bin',
                'config.json', 'vocab.txt', 'extended_vocab.txt')


class SensitivityBudgetStop(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def module(name):
    spec = importlib.util.spec_from_file_location('sensitivity_' + name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read_main_lock():
    path = ROOT / 'FINAL_EVAL_LOCK.json'
    lock = json.loads(path.read_text())
    if lock.get('status') != 'FROZEN':
        raise ValueError('The main protocol must be frozen before sensitivity preparation')
    for relative, expected in lock['artifact_hashes'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError('Main frozen artifact changed: ' + relative)
    plan = lock.get('sensitivity_plan', {})
    if list(plan.get('models', MODELS)) != list(MODELS) or list(plan.get('seeds', SEEDS)) != list(SEEDS):
        raise ValueError('Sensitivity models/seeds differ from the prespecified matrix')
    return lock, sha256(path)


def ensure_shared_links(alternative):
    # Only links INSIDE the sensitivity root are created. Main sources/weights
    # are read, never copied over or modified by this script.
    for name in ('weights', 'src'):
        target = ROOT / name
        link = alternative / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(os.path.relpath(target, alternative), target_is_directory=True)
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise ValueError(f'{link} must be a relative link to the main {name}')
        if os.path.isabs(os.readlink(link)):
            raise ValueError(f'{link} must use a relative symlink')


def inherited_choices(fold, main_lock, main_lock_hash, main_splits):
    selection_path = ROOT / 'runs/selections' / (fold + '.json')
    if selection_path.exists():
        saved = json.loads(selection_path.read_text())
        if saved.get('lock_sha256') != main_lock_hash:
            raise ValueError('Main selection cache belongs to a different lock: ' + fold)
        choices = {objective: saved['choices'][objective] for objective in OBJECTIVES}
        return {'rule': 'corresponding_primary_inner_selection', 'choices': choices,
                'primary_selection_path': str(selection_path), 'primary_selection_sha256': sha256(selection_path)}
    if fold in main_splits['folds'] and main_splits['folds'][fold]['test_ids']:
        raise ValueError('Primary inner selection is not complete for fold ' + fold)
    # Adding DD can introduce a source with no main-project counterpart. Its
    # defaults are fixed by the main lock, never borrowed from external scores.
    base = main_lock['neural_base_config']
    default = main_lock['neural_grid'][main_lock['default_grid_index']]
    choices = {objective: {'config': {**base, **default, 'objective': objective, 'ordinary_fraction': 1.},
                           'swa_candidate': 'swa_075', 'inner_valid': False}
               for objective in OBJECTIVES}
    return {'rule': 'new_source_uses_frozen_primary_default_no_search', 'choices': choices,
            'primary_selection_path': None, 'primary_selection_sha256': None}


def prepare(kind, main_lock, main_lock_hash):
    alternative = ROOT / 'sensitivity' / kind
    if alternative.resolve() == ROOT.resolve() or not alternative.resolve().is_relative_to((ROOT / 'sensitivity').resolve()):
        raise ValueError('Sensitivity data root must remain inside sensitivity/')
    alternative.mkdir(parents=True, exist_ok=True)
    ensure_shared_links(alternative)
    records = pd.read_parquet(alternative / 'data/processed/records.parquet')
    if records.row_id.duplicated().any():
        raise ValueError('Duplicate sensitivity row IDs')
    splits_path = alternative / 'data/processed/splits.json'
    splits = json.loads(splits_path.read_text())
    main_splits = json.loads((ROOT / 'data/processed/splits.json').read_text())
    if kind == 'author_labels' and sha256(splits_path) != sha256(ROOT / 'data/processed/splits.json'):
        raise ValueError('Author-label sensitivity must retain exactly the main split file')
    from bh_transfer.data import assert_split_integrity
    assert_split_integrity(records, splits)
    selections = {fold: inherited_choices(fold, main_lock, main_lock_hash, main_splits)
                  for fold, part in splits['folds'].items() if part['test_ids']}
    required_data = ('records.parquet', 'features.npz', 'splits.json', 'feature_manifest.json')
    artifact_paths = ['data/processed/' + name for name in required_data]
    artifact_paths += ['weights/rxnfp/' + name for name in WEIGHT_FILES]
    artifact_paths += [name for name in ('data/audit/token_audit.json', 'data/audit/verification.json')
                       if (alternative / name).exists()]
    lock = {
        'schema_version': 1, 'status': 'FROZEN', 'evaluation_allowed': True,
        'sensitivity_kind': kind, 'confirmatory_main_experiment': False,
        'main_lock_path': str(ROOT / 'FINAL_EVAL_LOCK.json'), 'main_lock_sha256': main_lock_hash,
        'artifact_hashes': {relative: sha256(alternative / relative) for relative in artifact_paths},
        'source_code_hashes': {str(ROOT / relative): sha256(ROOT / relative) for relative in SOURCE_FILES},
        'models': list(MODELS), 'seeds': list(SEEDS), 'minimum_seeds': 3, 'score_type': 'probability',
        'primary_label': 'author-provided labels' if kind == 'author_labels' else 'Yield>=10',
        'primary_metric': main_lock['primary_metric'],
        'folds': [{'fold': f, 'source': p['source'], 'eligible': p['eligible'], 'test_row_ids': p['test_ids']}
                  for f, p in splits['folds'].items()],
        'eligibility_basis': 'unchanged primary split eligibility' if kind == 'author_labels' else 'recomputed strict split eligibility',
        'selection_rule': 'Reuse corresponding main inner ERM/DRO hyperparameters, ordinary fraction and SWA start; no new search',
        'inherited_selections': selections,
        'calibration': 'temperature=1 for every model/seed; raw-score sensitivity only; no new calibration fitting',
        'temperature': 1., 'classification_threshold': .5,
        'gpu_hours_shared_cap': min(MAX_GPU_HOURS, float(main_lock.get('sensitivity_plan', {}).get('reserved_gpu_hours', MAX_GPU_HOURS))),
        'release_policy': 'No performance computation here; merge only after complete model/seed/row coverage and identity checks',
        'limitations': (['Original author label sensitivity; eligibility retained from primary labels'] if kind == 'author_labels'
                        else ['Original Dreher Doyle 2018 additive identities unresolved; conditions incomplete; sensitivity only']),
    }
    path = alternative / 'FINAL_EVAL_LOCK.json'
    if path.exists():
        saved = json.loads(path.read_text())
        expected = dict(saved)
        expected.pop('created_utc', None)
        if expected != lock:
            raise ValueError('Existing sensitivity lock differs; preserve previous artifacts and investigate: ' + kind)
        lock = saved
    else:
        lock['created_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        dump(path, lock)
        path.chmod(0o444)
    return alternative, lock, splits, records.set_index('row_id', drop=False)


def sensitivity_usage(guard):
    totals = guard.totals()  # Includes/reconciles interrupted central jobs.
    starts, finished = {}, {}
    for event in guard.ledger_records():
        attempt = event.get('attempt_id')
        if event.get('stage') != 'sensitivity' or not event.get('gpu') or not attempt:
            continue
        if event.get('event') == 'started':
            starts[attempt] = event
        elif event.get('event') == 'finished':
            finished[attempt] = max(finished.get(attempt, 0.), float(event.get('gpu_seconds', 0.)))
    now = time.time()
    seconds = sum(finished.values())
    for attempt, event in starts.items():
        if attempt not in finished:
            heartbeat = guard.read_heartbeat(event)
            seconds += max(float(heartbeat.get('gpu_seconds', 0.)), now - guard._started_unix(event), 0.)
    return dict(totals, sensitivity_gpu_hours=seconds / 3600.)


def stop_reason(guard, limit, reserve=0.):
    totals = sensitivity_usage(guard)
    if totals['sensitivity_gpu_hours'] + reserve / 3600. >= limit:
        return 'Shared sensitivity GPU-hour allowance exhausted', totals
    global_reason = guard.budget_reason(totals, True, 'sensitivity', reserve, 1)
    return global_reason, totals


def verify_job_output(script, spec):
    if script == 'run_neural_job':
        worker = module(script)
        identity, _ = worker.job_identity(spec['project'], spec)
        worker.verify_completed_result(spec['output_dir'], identity)
    elif script == 'predict_job':
        worker = module(script)
        identity, _ = worker.prediction_identity(spec['project'], spec)
        if not worker.verify_existing_prediction(spec['output'], identity):
            raise ValueError('Prediction worker did not produce an artifact')
    else:
        out = Path(spec['output_dir'])
        result = json.loads((out / 'result.json').read_text())
        if not Path(result['checkpoint']).is_file():
            raise ValueError('Missing sensitivity tabular checkpoint')


def job(alternative, kind, name, script, spec, gpu, guard, limit):
    run_id = f'sens_{kind}_{name}'
    spec_path = alternative / 'runs/specs' / (run_id + '.json')
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError('Sensitivity job specification changed: ' + run_id)
    else:
        dump(spec_path, spec)
    command = [PYTHON, str(ROOT / 'scripts' / (script + '.py')), '--spec', str(spec_path)]
    execution_identity, _ = guard.execution_identity(command)
    status_path = ROOT / 'runs' / run_id / 'job_status.json'
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    cached = guard.reusable_completion(previous, execution_identity)
    reserve = guard.TERM_GRACE_SECONDS + 2 * guard.POLL_SECONDS
    if not cached:
        reason, _ = stop_reason(guard, limit, reserve)
        if reason:
            raise SensitivityBudgetStop(reason)
    launcher = [PYTHON, str(ROOT / 'scripts/run_guard.py'), '--run-id', run_id,
                '--stage', 'sensitivity', '--threads', '1']
    if gpu:
        launcher.append('--gpu')
    launcher += ['--'] + command
    process = subprocess.Popen(launcher, cwd=ROOT)
    reason, termination_at = None, None
    while process.poll() is None:
        if not cached:
            current_reason, _ = stop_reason(guard, limit, reserve)
            if current_reason:
                reason = current_reason
                event = json.loads(status_path.read_text()) if status_path.exists() else {}
                # Verify this launcher's recorded session. Never signal foreign
                # GPU users or a previous run with a recycled PID.
                if event.get('launcher_pid') == process.pid and event.get('status') == 'running':
                    if termination_at is None:
                        guard.signal_workload(event, signal.SIGTERM)
                        termination_at = time.monotonic()
                    elif time.monotonic() - termination_at >= guard.TERM_GRACE_SECONDS:
                        guard.signal_workload(event, signal.SIGKILL)
        try:
            process.wait(timeout=guard.POLL_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    if reason:
        raise SensitivityBudgetStop(reason)
    if process.returncode:
        global_reason, _ = stop_reason(guard, limit, reserve)
        if global_reason:
            raise SensitivityBudgetStop(global_reason)
        raise subprocess.CalledProcessError(process.returncode, launcher)
    verify_job_output(script, spec)


def prediction_spec(alternative, part, fold, model, seed, checkpoint):
    return {'project': str(alternative), 'ids': part['test_ids'], 'model': model,
            'seed': seed, 'fold': fold, 'checkpoint': str(checkpoint), 'temperature': 1.,
            'output': str(alternative / 'artifacts/sealed_predictions' / f'{fold}_{model}_s{seed}.parquet')}


def run_kind(alternative, kind, lock, splits, guard):
    for fold, part in splits['folds'].items():
        if not part['test_ids']:
            continue
        for seed in SEEDS:
            for objective, models in OBJECTIVES.items():
                selected = lock['inherited_selections'][fold]['choices'][objective]
                name = f'{fold}_{objective}_s{seed}'
                out = alternative / 'runs/trajectories' / name
                config = dict(selected['config'], seed=seed, objective=objective)
                if not 0 < float(config.get('ordinary_fraction', 0)) <= 1:
                    raise ValueError('Sensitivity requires an internally preselected ordinary fraction')
                spec = {'project': str(alternative), 'train_ids': part['train_ids'], 'val_ids': [],
                        'config': config, 'output_dir': str(out)}
                job(alternative, kind, name, 'run_neural_job', spec, True, guard, lock['gpu_hours_shared_cap'])
                for model, filename in zip(models, ('best.pt', selected['swa_candidate'] + '.pt')):
                    pred = prediction_spec(alternative, part, fold, model, seed, out / filename)
                    job(alternative, kind, f'predict_{fold}_{model}_s{seed}', 'predict_job', pred, True,
                        guard, lock['gpu_hours_shared_cap'])
            model = 'R1_substitute'
            name = f'{fold}_{model}_s{seed}'
            out = alternative / 'runs/tabular' / name
            spec = {'project': str(alternative), 'train_ids': part['train_ids'], 'val_ids': [],
                    'model': model, 'seed': seed, 'output_dir': str(out)}
            job(alternative, kind, name, 'run_tabular_job', spec, False, guard, lock['gpu_hours_shared_cap'])
            pred = prediction_spec(alternative, part, fold, model, seed, out / 'model.pkl')
            job(alternative, kind, f'predict_{fold}_{model}_s{seed}', 'predict_job', pred, False,
                guard, lock['gpu_hours_shared_cap'])


def coverage(alternative, splits):
    expected = [alternative / 'artifacts/sealed_predictions' / f'{fold}_{model}_s{seed}.parquet'
                for fold, part in splits['folds'].items() if part['test_ids']
                for model, seed in itertools.product(MODELS, SEEDS)]
    missing = [str(path.relative_to(alternative)) for path in expected if not path.exists()]
    return {'expected_prediction_files': len(expected), 'present_prediction_files': len(expected) - len(missing),
            'missing_prediction_files': missing}


def combine(alternative, kind, lock, splits, records):
    report = coverage(alternative, splits)
    if report['missing_prediction_files']:
        raise ValueError('Sensitivity prediction matrix is incomplete')
    predictor = module('predict_job')
    files, frames = [], []
    for fold, part in splits['folds'].items():
        if not part['test_ids']:
            continue
        for model, seed in itertools.product(MODELS, SEEDS):
            spec_path = alternative / 'runs/specs' / f'sens_{kind}_predict_{fold}_{model}_s{seed}.json'
            spec = json.loads(spec_path.read_text())
            expected_path = alternative / 'artifacts/sealed_predictions' / f'{fold}_{model}_s{seed}.parquet'
            if Path(spec['output']) != expected_path or spec['ids'] != part['test_ids']:
                raise ValueError('Sensitivity prediction spec/path/row coverage mismatch')
            identity, _ = predictor.prediction_identity(alternative, spec)
            if not predictor.verify_existing_prediction(expected_path, identity):
                raise ValueError('Missing verified sensitivity prediction')
            files.append(expected_path)
            frames.append(pd.read_parquet(expected_path))
    if not frames:
        raise ValueError('No nonempty strict sensitivity test source; no score or macro average can be produced')
    predictions = pd.concat(frames, ignore_index=True)
    from bh_transfer.evaluate import validate_predictions
    validate_predictions(predictions, lock, alternative / 'FINAL_EVAL_LOCK.json')
    unique = predictions.drop_duplicates('row_id').set_index('row_id')
    truth = records.loc[unique.index]
    for observed, expected in (('y_true', 'y'), ('source', 'source'), ('aryl_id', 'aryl_id'),
                               ('amine_id', 'amine_id'), ('pair_id', 'pair_id'), ('condition_id', 'input_key')):
        if not np.array_equal(unique[observed].to_numpy(), truth[expected].to_numpy()):
            raise ValueError('Sensitivity prediction ground truth/chemistry mismatch: ' + observed)
    target = alternative / 'artifacts/predictions.parquet'
    temporary = target.with_name('predictions.tmp.parquet')
    predictions.to_parquet(temporary, index=False)
    temporary.replace(target)
    dump(alternative / 'artifacts/PREDICTION_COMPLETENESS.json', {
        **report, 'complete': True, 'rows': len(predictions), 'files': len(files),
        'lock_sha256': sha256(alternative / 'FINAL_EVAL_LOCK.json'), 'predictions_sha256': sha256(target),
        'truth_and_chemistry_verified': True, 'raw_scores_only': True, 'external_metrics_computed': False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['all', *KINDS], default='all')
    parser.add_argument('--phase', choices=['all', 'prepare', 'run', 'combine'], default='all')
    args = parser.parse_args()
    guard = module('run_guard')
    (ROOT / 'runs').mkdir(exist_ok=True)
    lease = (ROOT / 'runs/sensitivity_pipeline.lock').open('a+')
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A sensitivity coordinator is already active; no duplicate launched')
    main_lock, main_hash = read_main_lock()
    kinds = KINDS if args.kind == 'all' else (args.kind,)
    summary_path = ROOT / 'artifacts/SENSITIVITY_EXECUTION.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for kind_index, kind in enumerate(kinds):
        alternative, lock, splits, records = prepare(kind, main_lock, main_hash)
        try:
            if args.phase in ('all', 'run'):
                run_kind(alternative, kind, lock, splits, guard)
            if args.phase in ('all', 'combine'):
                combine(alternative, kind, lock, splits, records)
            status = 'COMPLETE' if args.phase in ('all', 'combine') else 'PREPARED' if args.phase == 'prepare' else 'PREDICTIONS_FINISHED'
            info = {'status': status, 'kind': kind, 'coverage': coverage(alternative, splits),
                    'budget': sensitivity_usage(guard), 'external_metrics_computed': False}
            dump(alternative / 'STATE.json', info)
            summary[kind] = info
        except SensitivityBudgetStop as exc:
            info = {'status': 'INCOMPLETE_BUDGET', 'kind': kind, 'reason': str(exc),
                    'coverage': coverage(alternative, splits), 'budget': sensitivity_usage(guard),
                    'main_formal_evaluation_must_continue': True, 'external_metrics_computed': False}
            dump(alternative / 'STATE.json', info)
            summary[kind] = info
            for remaining_kind in kinds[kind_index + 1:]:
                remaining_root = ROOT / 'sensitivity' / remaining_kind
                state_path = remaining_root / 'STATE.json'
                previous = json.loads(state_path.read_text()) if state_path.exists() else {}
                if previous.get('status') == 'COMPLETE':
                    summary[remaining_kind] = previous
                    continue
                pending = {'status': 'NOT_STARTED_BUDGET', 'kind': remaining_kind,
                           'reason': str(exc), 'budget': info['budget'],
                           'main_formal_evaluation_must_continue': True,
                           'external_metrics_computed': False}
                dump(state_path, pending)
                summary[remaining_kind] = pending
            # Auxiliary exhaustion deliberately returns success to its caller.
            break
        except Exception as exc:
            dump(alternative / 'STATE.json', {'status': 'INCOMPLETE_ERROR', 'kind': kind,
                 'reason': f'{type(exc).__name__}: {exc}', 'coverage': coverage(alternative, splits),
                 'external_metrics_computed': False, 'main_formal_evaluation_must_continue': True})
            raise
    dump(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
