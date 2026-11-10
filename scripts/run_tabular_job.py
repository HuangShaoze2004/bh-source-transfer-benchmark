#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from bh_transfer.tabular import fit
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args()
s=json.loads(Path(a.spec).read_text())
r=fit(s['project'],s['train_ids'],s.get('val_ids',[]),s['model'],s['seed'],s['output_dir'],s.get('label','y'))
print(json.dumps({k:v for k,v in r.items() if k not in ['train_ids','val_ids','val_metrics']},indent=2))
