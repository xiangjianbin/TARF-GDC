"""Read-only scientific audit, writing only the requested JSON report.

Three independent levels: paper tables from per-seed summaries; frozen dataset
hashes/metadata; and all 27 checkpoint predictions against historical batch-16
BF16 r0 metrics. A batch check is not a complete retraining or test-set rerun.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from v024_inversion.utils import read_json, write_json, sha256_file


def check_tables():
    overall = pd.read_csv(ROOT / 'evidence/overall_seed_metrics.csv')
    pairs = pd.read_csv(ROOT / 'evidence/pair_seed_metrics.csv')
    control = pd.read_csv(ROOT / 'evidence/control_seed_metrics.csv')
    checks = []

    def check(label, values, expected_mean, expected_sd, decimals):
        values = np.asarray(values, float)
        if len(values) != 3 or not np.isfinite(values).all():
            raise ValueError('Expected three finite training-seed values: ' + label)
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        tolerance = .501 * 10 ** -decimals
        ok = abs(mean - expected_mean) <= tolerance and abs(sd - expected_sd) <= tolerance
        checks.append(dict(item=label, mean=mean, sample_sd=sd, expected_mean=expected_mean,
                           expected_sd=expected_sd, passed=bool(ok)))

    # Printed values are checked at their displayed precision, not as exact floats.
    table4 = {
        '0001': [(127.871,7.327),(0.3684,.0234),(.3413,.0214),(.2469,.0298)],
        '1001': [(106.442,5.417),(.3234,.0113),(.2940,.0076),(.2136,.0245)],
        '0101': [(91.352,4.562),(.2934,.0137),(.2681,.0134),(.1272,.0143)],
        '1101': [(79.228,2.740),(.2676,.0030),(.2416,.0012),(.1132,.0091)]}
    columns = ['density_sae','density_nrmse','foreground_nrmse','physics_nrms']
    for code, values in table4.items():
        part = overall[(overall.split == 'validation_iid') & (overall.code.astype(str) == code)]
        for i, (metric, (mean, sd)) in enumerate(zip(columns, values)):
            check('Table4/' + code + '/' + metric, part[metric], mean, sd, 3 if i == 0 else 4)
    for code, vals in {'0001C': [(126.669,6.308),(.2422,.0251)],
                       '1001C': [(106.097,5.465),(.2156,.0234)]}.items():
        part = overall[(overall.split == 'validation_iid') & (overall.code == code)]
        for metric, val, digits in zip(['density_sae','physics_nrms'], vals, [3,4]):
            check('Table5/' + code + '/' + metric, part[metric], *val, digits)
    for split, vals in {'validation_iid': [(82.936,2.538),(.1355,.0090)],
                        'test_locked': [(82.888,2.688),(.1365,.0088)]}.items():
        for metric, val, digits in zip(['SAE','Physics NRMS'], vals, [3,4]):
            part = control[(control.split == split) & (control.model == 'Obs-only') &
                           (control.group == 'all') & (control.metric == metric)]
            check('Table6/' + split + '/' + metric, part.value, *val, digits)
    table7 = {
      'separable': {'T0P': [(1.2141,.0379),(.2869,.0170)],'B0P': [(1.1638,.0109),(.3060,.0062)],
                    '0001': [(1.1407,.0352),(.2995,.0160)],'1101': [(1.0967,.0100),(.3670,.0055)]},
      'shape': {'T0P': [(1.1137,.0086),(.0027,.0030)],'B0P': [(1.0322,.0105),(.2157,.0291)],
                '0001': [(1.0418,.0131),(.1851,.0054)],'1101': [(.9721,.0097),(.2912,.0149)]}}
    for track, methods in table7.items():
        for code, vals in methods.items():
            part = pairs[(pairs.split == 'validation_iid') & (pairs.code == code) & (pairs.track == track)]
            for metric, val in zip(['difference_nrmse','difference_cosine'], vals):
                check('Table7/' + track + '/' + code + '/' + metric, part[metric], *val, 4)
    table8 = {'T0P': [(139.174,8.089),(.2557,.0187)], 'B0P': [(150.788,2.084),(.2765,.0120)],
      '0001': [(128.105,6.939),(.2488,.0289)], '1001': [(106.492,5.352),(.2140,.0239)],
      '0101': [(91.714,4.129),(.1288,.0139)], '1101': [(79.161,2.841),(.1143,.0090)],
      '0001C': [(127.688,6.688),(.2440,.0242)], '1001C': [(106.178,5.412),(.2159,.0226)]}
    for code, vals in table8.items():
        part = overall[(overall.split == 'test_locked') & (overall.code == code)]
        for metric, val, digits in zip(['density_sae','physics_nrms'], vals, [3,4]):
            check('Table8/' + code + '/' + metric, part[metric], *val, digits)
    return dict(passed=all(c['passed'] for c in checks), checks=checks,
                note='Noise is averaged within model; SD is across three training seeds (ddof=1).')


def check_dataset(assets):
    from v024_inversion.data import read_split_metadata
    root = assets / 'dataset'
    checked, errors = 0, []
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        path = (root / relative.strip()).resolve()
        if root.resolve() not in path.parents:
            raise ValueError('Unsafe dataset manifest path')
        if not path.is_file() or sha256_file(path) != digest:
            errors.append(relative)
        checked += 1
    split_ids, group_ids, sizes = {}, {}, {}
    for split in ('train', 'validation_iid', 'test_locked'):
        rows, pairs = read_split_metadata(root, split)
        split_ids[split] = {r['sample_id'] for r in rows}
        group_ids[split] = set(pairs)
        sizes[split] = {'samples': len(rows), 'pairs': len(pairs)}
    overlaps = {}
    for a, b in [('train','validation_iid'), ('train','test_locked'), ('validation_iid','test_locked')]:
        overlaps[a + '/' + b] = {'sample_ids': len(split_ids[a] & split_ids[b]),
                                 'pair_ids': len(group_ids[a] & group_ids[b])}
    return dict(passed=not errors and not any(sum(v.values()) for v in overlaps.values()),
                files_checked=checked, hash_errors=errors, split_sizes=sizes, overlaps=overlaps)


def check_predictions(assets, device):
    import gc
    import torch
    from torch.utils.data._utils.collate import default_collate
    from release_runtime import registry, load_model
    from v024_inversion.trainer import build_dataset
    from v024_inversion.metrics import metrics_per_sample
    torch.set_num_threads(2)
    expected = read_json(ROOT / 'evidence/first_batch_reference.json')
    results = []
    batch = None
    for identifier, row in registry().items():
        cfg, model, op = load_model(identifier, assets, device)
        if batch is None:
            dataset = build_dataset(cfg, 'validation_iid', 0)
            batch = default_collate([dataset[i] for i in range(16)])
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        reference = expected[identifier]['samples']
        if batch['sample_id'] != [r['sample_id'] for r in reference]:
            raise ValueError('Reference batch order differs')
        prepared = op.prepare_inputs(batch['raw6'], batch['receiver_mask'], cfg['task']['input_mode'])
        with torch.inference_mode(), torch.autocast(device_type=device,
                dtype=torch.bfloat16, enabled=device == 'cuda'):
            prediction = model(prepared, op)
        metrics = metrics_per_sample(prediction, batch, op, cfg['task']['physics_mode'])
        errors = {}
        ok = True
        for name in ('density_sae', 'density_nrmse', 'foreground_nrmse', 'physics_nrms'):
            actual = metrics[name].detach().cpu().numpy().astype(float)
            target = np.array([float(r[name]) for r in reference])
            errors[name] = float(np.max(np.abs(actual - target)))
            ok &= bool(np.allclose(actual, target, rtol=2e-5, atol=1e-5))
        results.append(dict(model=identifier, passed=ok, max_absolute_errors=errors))
        print(identifier, 'PASS' if ok else 'FAIL', errors, flush=True)
        del model, op, prediction, metrics
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    return dict(passed=all(r['passed'] for r in results), models=results, samples_per_model=16,
                noise_realization=0, dtype='bf16' if device == 'cuda' else 'float32')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--predictions', action='store_true')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/release_check.json')
    args = parser.parse_args()
    start = time.time()
    report = {'tables': check_tables(), 'full_retraining_executed': False}
    if args.assets:
        report['dataset'] = check_dataset(args.assets.resolve())
    if args.predictions:
        if not args.assets:
            parser.error('--predictions requires --assets')
        report['predictions'] = check_predictions(args.assets.resolve(), args.device)
    report['passed'] = all(v['passed'] for v in report.values() if isinstance(v, dict) and 'passed' in v)
    report['elapsed_seconds'] = time.time() - start
    write_json(args.output, report)
    print('PASS' if report['passed'] else 'FAIL', args.output)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
