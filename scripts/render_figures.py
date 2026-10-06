#!/usr/bin/env python3
"""Replay the retained manuscript plots without modifying frozen source data.

Each group uses a new output directory and working copies of its arrays.
Chinese labels are retained where present in the audited Chinese manuscript.
The spatial routine emits a superseded section plot; group 09 is the final
common-threshold section plot. Field groups take explicitly supplied data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--group', choices=['01-02','03','04','05','06-07','08','09','10-11','12','13','14'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--field-source', type=Path,
                   help='Authorized processed field data, e.g. the output of reproduce.py field')
    args = p.parse_args()
    assets, out = args.assets.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    groups = {'01-02': ('dataset_figures','dataset'), '03': ('independent_example','independent_example'),
              '04': ('architecture',None), '05': ('training','statistics'),
              '06-07': ('pairs','pairs'), '08': ('spatial','statistics'),
              '09': ('sections','statistics'), '10-11': ('field_and_classical','classical'),
              '12': ('best_field_seed','restricted'), '13': ('field_metrics','restricted'),
              '14': ('field_and_classical','restricted')}
    module, data = groups[args.group]
    if data == 'restricted' and args.field_source is None:
        p.error('Field figures require --field-source; restricted field data are not bundled for publication')
    source = args.field_source.resolve() if data == 'restricted' else assets / ('figure_data/'+str(data))
    before = {}
    if data:
        if not source.is_dir():
            raise FileNotFoundError(str(source)+'; see docs/DATA.md for availability and field-data rights')
        before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
        shutil.copytree(source, out/'source_data')
    os.environ.update(TARF_FIGURE_OUTPUT=str(out), TARF_GDC_ASSETS=str(assets))
    sys.path.insert(0, str(ROOT/'figures'))
    # Legacy argparse blocks are bypassed. Scientific functions remain unchanged.
    ns = runpy.run_path(str(ROOT/'figures'/f'{module}.py'), run_name='release_figure')
    if args.group == '01-02':
        ns['figure1'](); ns['figure2']()
    elif args.group == '03':
        import numpy as np
        with np.load(out/'source_data/independent_example_arrays.npz',allow_pickle=False) as z:
            arrays = dict(z)
        metadata = json.loads((out/'source_data/independent_example_provenance.json').read_text())
        ns['render'](arrays, metadata)
    elif args.group in ('06-07','09','12'):
        ns['main']()
    elif args.group == '08':
        ns['spatial']()
    elif args.group == '10-11':
        ns['matched']()
    elif args.group == '13':
        sys.argv = ['field_metrics', '--source-dir', str(out/'source_data'), '--output-dir',str(out)]
        ns['main']()
    elif args.group == '14':
        ns['vinton']()
    for name, digest in before.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
    (out/'replay_provenance.json').write_text(json.dumps(dict(group=args.group,
        source_files=before, immutable_inputs_unchanged=True, numerical_transform_changes=False), indent=2))


if __name__ == '__main__':
    main()
