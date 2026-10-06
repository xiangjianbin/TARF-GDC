#!/usr/bin/env python3
"""Recompute training-only normalization, optional caches and operator derivatives.

The supplied assets are immutable. Recomputed products are written separately.
The operator stage preserves the historical preparation implementation, including
optional Tikhonov matrices that are NOT used by any manuscript model.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def main():
    import torch
    from threadpoolctl import threadpool_limits
    from preparation_core import audit_frozen, build_cache, build_normalization, prepare_operators
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--operators',action='store_true')
    p.add_argument('--cache-all',action='store_true',help='Explicitly cache validation and locked-test arrays too')
    a=p.parse_args();assets=a.assets.resolve();out=a.output.resolve()
    if assets==out or assets in out.parents:
        raise ValueError('Preparation outputs must be outside the frozen asset bundle')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        audit_frozen(assets/'dataset',out)
        build_cache(assets/'dataset',out,'train')
        result=build_normalization(assets/'dataset',out)
        from v024_inversion.utils import read_json, write_json
        expected=read_json(ROOT/'assets/normalization.json')['normalization']
        if result['normalization'] != expected:
            raise ValueError('Recomputed normalization differs from the frozen reference')
        write_json(out/'normalization_comparison.json',dict(passed=True,comparison='exact numerical equality'))
        if a.operators:
            prepare_operators(assets/'dataset',out)
        if a.cache_all:
            for split in ('validation_iid','test_locked'):
                build_cache(assets/'dataset',out,split)
    print('PASS',out,flush=True)


if __name__=='__main__':
    main()
