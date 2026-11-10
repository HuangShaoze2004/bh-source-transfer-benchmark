"""CPU-only regression checks against stale training and prediction reuse."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location('tested_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def jobs():
    return load_script('run_neural_job'), load_script('predict_job')


@pytest.fixture
def fixture_project(tmp_path, jobs):
    training, _ = jobs
    for relative in training.IDENTITY_ARTIFACTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('fixture:' + relative).encode())
    (tmp_path / 'FINAL_EVAL_LOCK.json').write_text(json.dumps({'status': 'FROZEN', 'artifact_hashes': {}}))
    (tmp_path / 'checkpoint.pt').write_bytes(b'saved model fixture')
    return tmp_path


def base_spec(root):
    return {'project': str(root), 'train_ids': [11, 12], 'val_ids': [13],
            'config': {'seed': 42, 'lr': 1e-5, 'objective': 'erm'},
            'checkpoint': str(root / 'checkpoint.pt'), 'ids': [21, 22],
            'temperature': 1., 'output': str(root / 'prediction.parquet')}


def test_complete_spec_and_all_initialization_data_inputs_are_bound(fixture_project, jobs):
    training, prediction = jobs
    root = fixture_project
    spec = base_spec(root)
    for identity in (training.job_identity, prediction.prediction_identity):
        original, _ = identity(root, spec)
        changed = copy.deepcopy(spec)
        changed['config']['lr'] = 3e-5
        assert identity(root, changed)[0] != original
        changed = copy.deepcopy(spec)
        changed['ids'].reverse()
        assert identity(root, changed)[0] != original
        for relative in training.IDENTITY_ARTIFACTS:
            path = root / relative
            before = path.read_bytes()
            path.write_bytes(before + b'changed')
            assert identity(root, spec)[0] != original, relative
            path.write_bytes(before)
        assert identity(root, spec)[0] == original


def test_training_reuse_requires_identity_and_intact_outputs(fixture_project, jobs):
    training, _ = jobs
    root = fixture_project
    identity, _ = training.job_identity(root, base_spec(root))
    out = root / 'run'
    out.mkdir()
    artifact = out / 'best.pt'
    artifact.write_bytes(b'checkpoint')
    result = {'job_identity': identity, 'output_artifact_hashes': {'best.pt': training.file_sha256(artifact)}}
    (out / 'result.json').write_text(json.dumps(result))
    assert training.verify_completed_result(out, identity) == result
    with pytest.raises(ValueError, match='identity differs'):
        training.verify_completed_result(out, 'changed identity')
    artifact.write_bytes(b'different checkpoint')
    with pytest.raises(ValueError, match='missing or changed'):
        training.verify_completed_result(out, identity)
    (out / 'result.json').write_text(json.dumps({'status': 'complete'}))
    with pytest.raises(ValueError, match='identity differs'):
        training.verify_completed_result(out, identity)


def test_prediction_reuse_requires_sidecar_and_unchanged_bytes(fixture_project, jobs):
    _, prediction = jobs
    root = fixture_project
    spec = base_spec(root)
    identity, _ = prediction.prediction_identity(root, spec)
    out = Path(spec['output'])
    assert prediction.verify_existing_prediction(out, identity) is False
    out.write_bytes(b'prediction fixture')
    with pytest.raises(ValueError, match='incomplete'):
        prediction.verify_existing_prediction(out, identity)
    sidecar = prediction.sidecar_path(out)
    sidecar.write_text(json.dumps({'job_identity': identity, 'prediction_sha256': prediction.file_sha256(out)}))
    assert prediction.verify_existing_prediction(out, identity)
    with pytest.raises(ValueError, match='identity differs'):
        prediction.verify_existing_prediction(out, 'different identity')
    out.write_bytes(b'changed prediction bytes')
    with pytest.raises(ValueError, match='recorded hash'):
        prediction.verify_existing_prediction(out, identity)


def test_checkpoint_and_temperature_changes_invalidate_prediction_identity(fixture_project, jobs):
    _, prediction = jobs
    root = fixture_project
    spec = base_spec(root)
    original, _ = prediction.prediction_identity(root, spec)
    changed = dict(spec, temperature=2.)
    assert prediction.prediction_identity(root, changed)[0] != original
    Path(spec['checkpoint']).write_bytes(b'different model weights')
    assert prediction.prediction_identity(root, spec)[0] != original


def test_lock_is_validated_before_prediction_identity(fixture_project, jobs):
    _, prediction = jobs
    root = fixture_project
    lock_path = root / 'FINAL_EVAL_LOCK.json'
    relative = 'data/processed/records.parquet'
    lock = {'status': 'FROZEN', 'artifact_hashes': {relative: prediction.file_sha256(root / relative)}}
    lock_path.write_text(json.dumps(lock))
    prediction.prediction_identity(root, base_spec(root))
    (root / relative).write_bytes(b'changed frozen data')
    with pytest.raises(ValueError, match='Frozen artifact changed'):
        prediction.prediction_identity(root, base_spec(root))
    lock_path.write_text(json.dumps({'status': 'DRAFT'}))
    with pytest.raises(ValueError, match='not frozen'):
        prediction.prediction_identity(root, base_spec(root))


def test_executable_script_source_is_bound(fixture_project, jobs, monkeypatch):
    root = fixture_project
    script = root / 'script.py'
    script.write_text('# version one\n')
    for module, function in ((jobs[0], jobs[0].job_identity), (jobs[1], jobs[1].prediction_identity)):
        monkeypatch.setattr(module, '__file__', str(script))
        original, _ = function(root, base_spec(root))
        script.write_text(script.read_text() + '# code changed\n')
        assert function(root, base_spec(root))[0] != original
