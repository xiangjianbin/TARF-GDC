"""Explicit, restartable orchestration of all 27 historical model evaluations.

This can take hours. Existing completed summaries must have the matching
checkpoint hash before they are skipped; incomplete folders cause an error.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def main():
    from release_runtime import registry
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    for split in ('validation_iid','test_locked'):
        for name,row in registry().items():
            target=a.output.resolve()/split/name
            if target.exists():
                saved=json.loads((target/'summary.json').read_text())
                if saved['checkpoint_sha256']!=row['sha256'] or saved['split']!=split or saved['model']!=name:
                    raise ValueError('Existing evaluation does not match: '+str(target))
                print('Verified completed run:',target,flush=True)
                continue
            subprocess.run([sys.executable,str(ROOT/'reproduce.py'),'evaluate','--assets',str(a.assets.resolve()),
                '--output',str(target),'--model',name,'--split',split,'--device',a.device],check=True,cwd=ROOT)


if __name__=='__main__':
    main()
