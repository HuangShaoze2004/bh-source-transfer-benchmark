#!/usr/bin/env python3
"""Package and publish the complete experiment to its owner's GitHub repository."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import tarfile
import tempfile
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "HuangShaoze2004/bh-source-transfer-benchmark"
TAG = "experiment-2026-11-11"
OUTPUT = ROOT / "release"
SCIENTIFIC_COMMIT = "618b1cc"
SOURCE_DIRECTORIES = ("data", "weights", "artifacts", "runs", "sensitivity", "logs")
SECRET_PATTERN = re.compile(rb"github_pat_[A-Za-z0-9_]{30,}|\bgh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path, scan=False):
    result = hashlib.sha256()
    overlap = b""
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
            if scan and SECRET_PATTERN.search(overlap + block):
                raise ValueError(f"Credential pattern detected in {path}")
            overlap = block[-256:]
    return result.hexdigest()


def progress(event, **details):
    record = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), event=event, **details)
    print(json.dumps(record), flush=True)
    with (OUTPUT / "publication_events.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def candidate_files():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    paths = {os.fsdecode(item) for item in tracked if item}
    for directory in SOURCE_DIRECTORIES:
        for base, directories, filenames in os.walk(ROOT / directory, followlinks=False):
            directories[:] = [name for name in directories if name != "__pycache__"]
            for filename in filenames:
                path = Path(base) / filename
                if (path.suffix == ".lock" and path.stat().st_size == 0) or path.name.endswith(".tmp"):
                    continue
                paths.add(str(path.relative_to(ROOT)))
            for name in directories:
                path = Path(base) / name
                if path.is_symlink():
                    paths.add(str(path.relative_to(ROOT)))
    return sorted(path for path in paths if not path.startswith("release/"))


def inventory():
    entries = []
    for index, name in enumerate(candidate_files()):
        path = ROOT / name
        if path.is_symlink():
            resolved = path.resolve()
            if not resolved.is_relative_to(ROOT):
                raise ValueError(f"External symlink in experiment: {name}")
            entries.append(dict(path=name, kind="symlink", target=str(path.readlink()), storage="git"))
            continue
        if not path.is_file():
            raise ValueError(f"Missing source file: {name}")
        stat = path.stat()
        storage = "release" if path.suffix in (".pt", ".pkl") else "git"
        if storage == "git" and stat.st_size >= 100 * 1024**2:
            storage = "release"
        entries.append(dict(path=name, kind="file", bytes=stat.st_size,
                            sha256=digest(path, scan=storage == "git"),
                            mode=stat.st_mode & 0o777, storage=storage))
        if index % 100 == 0:
            progress("inventory", inspected=index + 1)
    return entries


def build_archive(index, entries, zstd):
    name = f"checkpoints-{index:04d}.tar.zst"
    path = OUTPUT / name
    receipt = path.with_suffix(path.suffix + ".json")
    members = {item["path"]: item["sha256"] for item in entries}
    if receipt.exists() and path.exists():
        previous = json.loads(receipt.read_text())
        if previous["members"] == members and digest(path) == previous["sha256"]:
            return previous
        raise ValueError(f"Existing archive differs: {name}")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as output:
        process = subprocess.Popen([zstd, "--single-thread", "-1", "-q", "-c"], stdin=subprocess.PIPE, stdout=output)
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
                for entry in entries:
                    source = ROOT / entry["path"]
                    info = archive.gettarinfo(str(source), arcname=entry["path"])
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as handle:
                        reader = CheckedReader(handle)
                        archive.addfile(info, reader)
                        if reader.checksum.hexdigest() != entry["sha256"]:
                            raise ValueError(f"Source changed during packaging: {entry['path']}")
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("zstd archive creation failed")
        except BaseException:
            process.stdin.close()
            process.wait()
            raise
    if temporary.stat().st_size >= 2 * 1024**3:
        raise ValueError("Archive exceeds the GitHub Release asset limit")
    temporary.replace(path)
    record = dict(name=name, bytes=path.stat().st_size, sha256=digest(path), members=members)
    save(receipt, record)
    progress("archive_complete", name=name, bytes=record["bytes"], members=len(members))
    return record


class CheckedReader:
    """Hash the exact bytes fed into an archive member."""

    def __init__(self, handle):
        self.handle = handle
        self.checksum = hashlib.sha256()

    def read(self, size):
        block = self.handle.read(size)
        self.checksum.update(block)
        return block


def prepare(args):
    OUTPUT.mkdir(exist_ok=True)
    manifest_path = OUTPUT / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        progress("existing_manifest_reused", archives=len(manifest["archives"]))
        return
    started = time.monotonic()
    entries = inventory()
    selected = [item for item in entries if item["storage"] == "release"]
    groups, batch, total = [], [], 0
    for item in selected:
        if batch and total + item["bytes"] > args.archive_mib * 1024**2:
            groups.append(batch)
            batch, total = [], 0
        batch.append(item)
        total += item["bytes"]
    if batch:
        groups.append(batch)
    save(OUTPUT / "inventory.partial.json", entries)
    archives = [build_archive(index, group, args.zstd) for index, group in enumerate(groups, start=1)]
    manifest = dict(schema_version=1, repository=REPOSITORY, release_tag=TAG,
                    scientific_commit=SCIENTIFIC_COMMIT, recorded_source_root=str(ROOT),
                    created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    files=entries, archives=archives,
                    omitted_rebuildable_directories=[".venv", "cache", ".cache", ".pytest_cache", "__pycache__"],
                    omitted_host_state=["zero-byte resource lock files", "temporary incomplete writes"],
                    environment="Python 3.12; exact distributions in requirements.lock.txt")
    save(manifest_path, manifest)
    with (OUTPUT / "SHA256SUMS").open("w") as handle:
        handle.write(digest(manifest_path) + "  MANIFEST.json\n")
        for archive in archives:
            handle.write(archive["sha256"] + "  " + archive["name"] + "\n")
    with (OUTPUT / "git-paths.nul").open("wb") as handle:
        for item in entries:
            if item["storage"] == "git":
                handle.write(os.fsencode(item["path"]) + b"\0")
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    save(OUTPUT / "packaging_cost.json", dict(wall_seconds=time.monotonic() - started,
         cpu_seconds=own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime,
         gpu_seconds=0, compression_workers=1))
    progress("prepared", files=len(entries), archives=len(archives),
             archive_bytes=sum(item["bytes"] for item in archives))


def authenticated_session():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    private = Path("/root/.config/bh-source-transfer/github_token")
    if not token and private.is_file():
        token = private.read_text().strip()
    if not token:
        raise ValueError("Set GH_TOKEN/GITHUB_TOKEN or configure the external credential file")
    session = requests.Session()
    session.headers.update(Authorization="Bearer " + token, Accept="application/vnd.github+json",
                           **{"X-GitHub-Api-Version": "2022-11-28", "User-Agent": "bh-source-transfer-publication"})
    return session, token


def api(session, method, path, **kwargs):
    url = path if path.startswith("https://") else "https://api.github.com" + path
    response = session.request(method, url, timeout=(30, 300), **kwargs)
    if not response.ok:
        raise RuntimeError(f"GitHub {method} failed with HTTP {response.status_code}")
    return response.json() if response.content else None


def assets(session, release):
    result = []
    for page in range(1, 100):
        values = api(session, "GET", release["assets_url"], params=dict(per_page=100, page=page))
        result.extend(values)
        if len(values) < 100:
            return {item["name"]: item for item in result}
    raise RuntimeError("Unexpected release asset pagination")


def verify_remote_asset(session, remote, local):
    if remote["state"] != "uploaded" or remote["size"] != local.stat().st_size:
        raise ValueError(f"Remote asset size/state mismatch: {local.name}")
    expected = digest(local)
    recorded = remote.get("digest")
    if recorded:
        if recorded != "sha256:" + expected:
            raise ValueError(f"Remote asset digest mismatch: {local.name}")
        return expected
    result = hashlib.sha256()
    with session.get(remote["url"], headers={"Accept": "application/octet-stream"},
                     stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        for block in response.iter_content(4 * 1024 * 1024):
            result.update(block)
    if result.hexdigest() != expected:
        raise ValueError(f"Downloaded asset digest mismatch: {local.name}")
    return expected


def upload_asset(headers, release, path):
    with requests.Session() as session:
        session.headers.update(headers)
        for attempt in range(4):
            existing = assets(session, release).get(path.name)
            if existing and existing["state"] == "starter" and existing["size"] == 0:
                api(session, "DELETE", existing["url"])
                existing = None
            if existing:
                checksum = verify_remote_asset(session, existing, path)
                return dict(name=path.name, sha256=checksum, bytes=path.stat().st_size)
            try:
                progress("upload_started", name=path.name, bytes=path.stat().st_size, attempt=attempt + 1)
                with path.open("rb") as handle:
                    remote = api(session, "POST", release["upload_url"].split("{")[0],
                                 params=dict(name=path.name), data=handle,
                                 headers={"Content-Type": "application/octet-stream"})
                checksum = verify_remote_asset(session, remote, path)
                return dict(name=path.name, sha256=checksum, bytes=path.stat().st_size)
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))


def upload(args):
    manifest = json.loads((OUTPUT / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        if item["storage"] != "git":
            continue
        path = ROOT / item["path"]
        if item["kind"] == "symlink":
            if not path.is_symlink() or str(path.readlink()) != item["target"]:
                raise ValueError(f"Local symlink differs from the manifest: {item['path']}")
        elif not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError(f"Local file differs from the manifest: {item['path']}")
    session, token = authenticated_session()
    repository = api(session, "GET", "/repos/" + REPOSITORY)
    if repository["full_name"] != REPOSITORY or not repository["permissions"]["push"]:
        raise ValueError("Credential has no write permission for the requested repository")
    remotes = subprocess.check_output(["git", "remote"], cwd=ROOT, text=True).splitlines()
    remote_url = "https://github.com/" + REPOSITORY + ".git"
    if "origin" not in remotes:
        subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=ROOT, check=True)
    if subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=ROOT, text=True).strip() != remote_url:
        raise ValueError("Existing origin points to a different repository")
    with tempfile.TemporaryDirectory(prefix="bh-git-auth-") as directory:
        askpass = Path(directory) / "askpass"
        askpass.write_text('#!/bin/sh\ncase "$1" in *Username*) printf "%s" "HuangShaoze2004" ;; *) printf "%s" "$GITHUB_TOKEN" ;; esac\n')
        askpass.chmod(0o700)
        env = dict(os.environ, GITHUB_TOKEN=token, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT="0")
        subprocess.run(["git", "-c", "credential.helper=", "-c", "pack.threads=1", "push", "origin", "HEAD:main"], cwd=ROOT, env=env, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    endpoint = "/repos/" + REPOSITORY + "/releases/tags/" + TAG
    response = session.get("https://api.github.com" + endpoint, timeout=30)
    if response.status_code == 404:
        drafts = api(session, "GET", "/repos/" + REPOSITORY + "/releases", params=dict(per_page=100))
        matching = [item for item in drafts if item["tag_name"] == TAG]
        release = matching[0] if matching else api(session, "POST", "/repos/" + REPOSITORY + "/releases", json=dict(
            tag_name=TAG, target_commitish=commit, name="Complete experiment snapshot — 2026-11-11",
            draft=True, prerelease=False, body=(
                "Complete frozen three-seed experiment, both sensitivity analyses, predictions, logs and resumable checkpoints.\n\n"
                "Scientific outcome: NOT_READY; the predefined increment criterion was unmet.\n\n"
                "Core data and evidence are tracked in Git. Download checkpoint archives with scripts/reproduce.py restore. "
                "MANIFEST.json records every scientific artifact and SHA-256. See REPRODUCING.md for environment setup and cross-directory verification.")))
    else:
        response.raise_for_status()
        release = response.json()
    pending = [OUTPUT / item["name"] for item in manifest["archives"]]
    pending += [OUTPUT / "MANIFEST.json", OUTPUT / "SHA256SUMS"]
    verified = []
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = [executor.submit(upload_asset, dict(session.headers), release, path) for path in pending]
    try:
        for future in as_completed(futures):
            record = future.result()
            verified.append(record)
            save(OUTPUT / "UPLOAD_RECEIPT.json", dict(repository=REPOSITORY, release_url=release["html_url"],
                 source_commit=commit, status="UPLOADING", verified_assets=verified))
            progress("upload_verified", name=record["name"], completed=len(verified), expected=len(pending))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    remote_head = api(session, "GET", "/repos/" + REPOSITORY + "/git/ref/heads/main")["object"]["sha"]
    if remote_head != commit:
        raise ValueError("Remote main differs from the pushed commit")
    if release["draft"]:
        source_url = f"https://github.com/{REPOSITORY}/blob/{commit}"
        body = (
            "Complete controlled experiments for cross-source Buchwald–Hartwig feasibility prediction "
            "at Yield >= 10%, with both substrate entities individually unseen.\n\n"
            "Scientific status: NOT_READY. The frozen increment criteria were unmet. "
            "The main nine-method, three-seed comparison and both sensitivities are complete.\n\n"
            f"[Reproduction instructions]({source_url}/REPRODUCING.md) · "
            f"[Experimental report]({source_url}/reports/EXPERIMENT_REPORT.md) · "
            f"[Attribution]({source_url}/THIRD_PARTY_NOTICES.md)\n\n"
            "Git contains data, fixed features/splits, all predictions, specifications, logs and statistics. "
            "The 64 checkpoint archives preserve every neural/tree model and recovery state. "
            "MANIFEST.json and SHA256SUMS identify all artifacts and their checksums.\n\n"
            "After cloning and installing the pinned environment, run:\n\n"
            "```bash\npython scripts/reproduce.py verify\npython scripts/reproduce.py metrics\n"
            "python scripts/reproduce.py restore\npython scripts/reproduce.py verify --full\n```\n"
        )
        release = api(session, "PATCH", release["url"], json=dict(draft=False, target_commitish=commit, body=body))
    save(OUTPUT / "UPLOAD_RECEIPT.json", dict(repository=REPOSITORY, release_url=release["html_url"],
         source_commit=commit, status="COMPLETE", verified_assets=verified,
         verified_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    progress("publication_complete", commit=commit, release_url=release["html_url"], assets=len(verified))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--archive-mib", type=int, default=1536)
    prepare_parser.add_argument("--zstd", default="zstd")
    upload_parser = sub.add_parser("upload")
    upload_parser.add_argument("--workers", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    if args.action == "prepare":
        if not 1 <= args.archive_mib <= 1792:
            parser.error("archive-mib must lie between 1 and 1792")
        prepare(args)
    else:
        upload(args)


if __name__ == "__main__":
    main()
