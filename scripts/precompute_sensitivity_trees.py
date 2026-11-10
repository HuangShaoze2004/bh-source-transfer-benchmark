#!/usr/bin/env python3
"""Fit only the already-prespecified sensitivity RF controls on free CPU slots.

Use the sensitivity coordinator's exact job API, specifications and run IDs so
its later full run reuses these fits. Never produce or read outer predictions.
The coordinator lease prevents a simultaneous sensitivity runner.
"""
import datetime as dt
import fcntl
import json
import os
import resource
import time

import sensitivity_pipeline as pipeline


def main():
    root = pipeline.ROOT
    with (root / 'runs/sensitivity_pipeline.lock').open('a+') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A sensitivity coordinator is already active; no duplicate launched')
        guard = pipeline.module('run_guard')
        main_lock, main_hash = pipeline.read_main_lock()
        completed = []
        lock_hashes = {}
        for kind in pipeline.KINDS:
            alternative, lock, splits, _ = pipeline.prepare(kind, main_lock, main_hash)
            lock_hashes[kind] = pipeline.sha256(alternative / 'FINAL_EVAL_LOCK.json')
            for fold, part in splits['folds'].items():
                if not part['test_ids']:
                    continue
                for seed in pipeline.SEEDS:
                    model = 'R1_substitute'
                    name = f'{fold}_{model}_s{seed}'
                    out = alternative / 'runs/tabular' / name
                    spec = {'project': str(alternative), 'train_ids': part['train_ids'],
                            'val_ids': [], 'model': model, 'seed': seed, 'output_dir': str(out)}
                    pipeline.job(alternative, kind, name, 'run_tabular_job', spec, False,
                                 guard, lock['gpu_hours_shared_cap'])
                    completed.append({'kind': kind, 'fold': fold, 'seed': seed,
                                      'run_id': f'sens_{kind}_{name}',
                                      'result_sha256': pipeline.sha256(out / 'result.json')})
        manifest = {'status': 'FITS_COMPLETE', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'main_lock_sha256': main_hash, 'sensitivity_lock_sha256': lock_hashes,
                    'completed_fits': completed, 'external_predictions_generated': False,
                    'external_metrics_computed': False}
        pipeline.dump(root / 'artifacts/SENSITIVITY_TREE_PRECOMPUTE.json', manifest)
        print(json.dumps({'status': manifest['status'], 'completed_fits': len(completed)}), flush=True)


if __name__ == '__main__':
    # This lightweight coordinator must not hold a guard lease while launching
    # separately guarded children (which would count child CPU twice). Reserve
    # one minute of self-CPU up front, retaining the bound even on interruption.
    guard = pipeline.module('run_guard')
    started = time.time()
    event = {'event': 'finished', 'run_id': 'coord_sensitivity_tree_precompute_v1',
             'attempt_id': 'coord_sensitivity_tree_precompute_v1__' + str(time.time_ns()),
             'stage': 'review', 'gpu': False, 'threads': 1, 'pid': os.getpid(),
             'start_unix': started, 'gpu_seconds': 0, 'cpu_seconds': 60.,
             'status': 'overhead_reserved', 'cost_quality': 'upper_bound',
             'reason': 'Coordinator self-CPU allowance; child fits separately guarded'}
    guard.append(event)
    succeeded = False
    try:
        main()
        succeeded = True
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        event.update(status='completed' if succeeded else 'failed',
                     wall_seconds=time.time()-started,
                     cpu_seconds=max(60., usage.ru_utime+usage.ru_stime),
                     measured_coordinator_cpu_seconds=usage.ru_utime+usage.ru_stime)
        guard.append(event)
