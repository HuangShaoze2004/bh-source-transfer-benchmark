#!/usr/bin/env python3
"""Fetch only public, pinned initialization artifacts and verify every hash."""
import hashlib,json,os,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'weights/rxnfp/manifest.json').read_text())
for item in manifest['files']:
    path=ROOT/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Existing artifact hash mismatch: '+str(path))
        continue
    tmp=path.with_suffix(path.suffix+'.part')
    with urllib.request.urlopen(item['source'],timeout=120) as src,tmp.open('wb') as out:
        while chunk:=src.read(1024*1024):out.write(chunk)
    if hashlib.sha256(tmp.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError('Downloaded artifact hash mismatch: '+str(path))
    os.replace(tmp,path)
print('Pinned pretrained artifacts verified; no supervised reaction weights downloaded.')
