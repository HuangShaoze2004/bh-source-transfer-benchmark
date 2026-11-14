import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest


PROJECT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_archive_round_trip_and_changed_source(tmp_path, monkeypatch):
    publisher = load_script("publish_github")
    reproduction = load_script("reproduce")
    zstd = shutil.which("zstd")
    if not zstd:
        pytest.skip("zstd command is needed for archive integration")
    source = tmp_path / "source"
    source.mkdir()
    path = source / "model.pt"
    payload = b"checkpoint contents" * 200
    path.write_bytes(payload)
    entry = dict(path="model.pt", kind="file", bytes=len(payload),
                 sha256=hashlib.sha256(payload).hexdigest(), mode=0o644, storage="release")
    output = tmp_path / "archives"
    output.mkdir()
    monkeypatch.setattr(publisher, "ROOT", source)
    monkeypatch.setattr(publisher, "OUTPUT", output)
    archive = publisher.build_archive(1, [entry], zstd)
    destination = tmp_path / "restored"
    destination.mkdir()
    monkeypatch.setattr(reproduction, "ROOT", destination)
    reproduction.extract_archive(output / archive["name"], archive, {"model.pt": entry}, zstd)
    assert (destination / "model.pt").read_bytes() == payload
    path.write_bytes(b"different contents")
    with pytest.raises(ValueError, match="Source changed"):
        publisher.build_archive(2, [entry], zstd)


def test_restore_rejects_archive_path_escape(tmp_path, monkeypatch):
    reproduction = load_script("reproduce")
    zstd = shutil.which("zstd")
    if not zstd:
        pytest.skip("zstd command is needed for archive integration")
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as archive:
        info = tarfile.TarInfo("../outside.pt")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    path = tmp_path / "bad.tar.zst"
    with path.open("wb") as handle:
        subprocess.run([zstd, "-q", "-c"], input=tar_bytes.getvalue(), stdout=handle, check=True)
    destination = tmp_path / "checkout"
    destination.mkdir()
    monkeypatch.setattr(reproduction, "ROOT", destination)
    metadata = dict(sha256=reproduction.sha256(path), members={"../outside.pt": "unused"})
    with pytest.raises(ValueError, match="escapes checkout"):
        reproduction.extract_archive(path, metadata, {"../outside.pt": dict(bytes=1)}, zstd)
    assert not (tmp_path / "outside.pt").exists()


def test_recorded_source_paths_relocate_without_reading_old_checkout(tmp_path, monkeypatch):
    reproduction = load_script("reproduce")
    monkeypatch.setattr(reproduction, "ROOT", tmp_path)
    metadata = dict(recorded_source_root="/previous/computer/project")
    assert reproduction.local_source_path("/previous/computer/project/src/example.py", metadata) == tmp_path / "src/example.py"
    with pytest.raises(ValueError):
        reproduction.local_source_path("/unrelated/private/file", metadata)


def test_replay_preserves_archived_spec_and_uses_isolated_guard(tmp_path, monkeypatch):
    reproduction = load_script("reproduce")
    monkeypatch.setattr(reproduction, "ROOT", tmp_path)
    (tmp_path / "runs/specs").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "release").mkdir()
    shutil.copyfile(PROJECT / "scripts/run_guard.py", tmp_path / "scripts/run_guard.py")
    original = dict(project="/previous/computer/project", train_ids=[1, 2], val_ids=[],
                    model="R0_RF", seed=17, output_dir="/previous/computer/project/runs/tabular/example")
    path = tmp_path / "runs/specs/example.json"
    path.write_text(json.dumps(original))
    (tmp_path / "release/MANIFEST.json").write_text(json.dumps(dict(recorded_source_root="/previous/computer/project")))
    captured = []
    monkeypatch.setattr(reproduction.subprocess, "run", lambda command, **kwargs: captured.append(command))
    reproduction.replay(argparse.Namespace(job="example", replayed_checkpoint=False))
    assert json.loads(path.read_text()) == original
    saved = json.loads((tmp_path / "reproduction/specs/example.json").read_text())
    assert saved["project"] == str(tmp_path)
    assert saved["output_dir"] == str(tmp_path / "reproduction/work/runs/tabular/example")
    assert captured[0][captured[0].index("--stage") + 1] == "formal"
    assert Path(captured[0][1]).is_relative_to(tmp_path / "reproduction/runtime")


@pytest.mark.parametrize("correct_checksum", [True, False])
def test_asset_resume_checks_remote_digest_before_any_write(tmp_path, monkeypatch, correct_checksum):
    publisher = load_script("publish_github")
    path = tmp_path / "checkpoint.tar.zst"
    path.write_bytes(b"verified release bytes")
    checksum = publisher.digest(path)
    remote = dict(state="uploaded", size=path.stat().st_size,
                  digest="sha256:" + (checksum if correct_checksum else "0" * 64))

    class Session:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(publisher.requests, "Session", Session)
    monkeypatch.setattr(publisher, "assets", lambda *args: {path.name: remote})
    monkeypatch.setattr(publisher, "api", lambda *args, **kwargs: pytest.fail("Verified resume must avoid a remote write"))
    if correct_checksum:
        assert publisher.upload_asset({}, {}, path)["sha256"] == checksum
    else:
        with pytest.raises(ValueError, match="digest mismatch"):
            publisher.upload_asset({}, {}, path)
