#!/usr/bin/env python3
"""Inspect recipes, train, evaluate, and run frozen field inference.

Outputs are new directories. No command writes into the frozen asset bundle.
Training is never invoked by a check or evaluation command.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))


def evaluate(args):
    import torch
    from release_runtime import load_model, registry
    from v024_inversion.trainer import build_dataset, build_loader, _amp_dtype
    from v024_inversion.evaluation import evaluate_with_pairs
    from v024_inversion.statistics import collapse_realizations, summarize_metric_rows, summarize_pairs
    from v024_inversion.utils import write_csv, write_json
    cfg, model, op = load_model(args.model, args.assets, args.device, args.checkpoint)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows_all, pairs_all = [], []
    for realization in range(5):
        ds = build_dataset(cfg, args.split, realization)
        summary, rows, pairs = evaluate_with_pairs(model, op, build_loader(ds, cfg, False),
            torch.device(args.device), _amp_dtype(cfg['training']['amp_dtype']),
            cfg['task']['input_mode'], cfg['task']['physics_mode'])
        expected_n, expected_p = (2500, 650) if args.split == 'validation_iid' else (10710, 2855)
        if len(rows) != expected_n or len(pairs) != expected_p:
            raise ValueError('Incomplete endpoint/pair evaluation')
        for row in rows + pairs:
            row.update(noise_realization=realization, split=args.split)
        rows_all.extend(rows); pairs_all.extend(pairs)
        write_csv(output / f'samples_r{realization}.csv', rows)
        write_csv(output / f'pairs_r{realization}.csv', pairs)
        print(f'completed {args.model} {args.split} r{realization}', flush=True)
    rows = collapse_realizations(rows_all)
    pairs = collapse_realizations(pairs_all, key='ambiguity_group_id')
    write_csv(output / 'metrics_per_sample_noise_mean.csv', rows)
    write_csv(output / 'metrics_per_pair_noise_mean.csv', pairs)
    write_json(output / 'summary.json', dict(model=args.model, split=args.split,
        sample_count=len(rows), pair_count=len(pairs), realizations=list(range(5)),
        checkpoint_sha256=cfg['loaded_checkpoint_sha256'],
        historical_checkpoint=cfg['loaded_historical_checkpoint'],
        aggregate=summarize_metric_rows(rows, ['density_sae', 'density_nrmse', 'foreground_nrmse', 'physics_nrms']),
        pair_subgroups=summarize_pairs(pairs)))


def train(args):
    import torch
    from release_runtime import config_for, verify_operator
    from v024_inversion import trainer
    from v024_inversion.utils import write_json, sha256_file
    verify_operator(args.assets)
    cfg = config_for(args.model, args.assets, args.device)
    cfg['output_root'] = str(args.output.resolve())
    if args.stage1_checkpoint:
        if 'staged_refinement' not in cfg:
            raise ValueError('Stage-1 models do not accept --stage1-checkpoint')
        path = args.stage1_checkpoint.resolve()
        saved = torch.load(path, map_location='cpu', weights_only=False)
        cfg['staged_refinement'].update(checkpoint=str(path), checkpoint_sha256=sha256_file(path),
                                       source_epoch=int(saved['epoch']))
    if args.smoke:
        cfg['phase'] = 'smoke'
        cfg['experiment_id'] += '_smoke'
        cfg['training'].update(epochs=1, max_steps=1, num_workers=0)
        cfg['data'].update(train_limit=16, validation_limit=16)
    cfg_path = args.output.resolve() / (cfg['experiment_id'] + '_resolved.json')
    if cfg_path.exists() and not args.resume:
        raise FileExistsError(cfg_path)
    if args.resume:
        from v024_inversion.utils import read_json
        if read_json(cfg_path) != cfg:
            raise ValueError('Resume configuration differs')
    else:
        write_json(cfg_path, cfg)
    print(trainer.run_training(cfg, cfg_path, resume=args.resume))


def field(args):
    import numpy as np
    import torch
    from release_runtime import load_model
    from field_preprocessing import load_field_data, resample_to_model_grid, to_z_up, fit_metrics, structure_at_threshold
    from v024_inversion.utils import write_csv, write_json
    observed = to_z_up(resample_to_model_grid(load_field_data(args.field_directory)))
    args.output.mkdir(parents=True, exist_ok=False)
    rows, geometries, cuts = [], [], {}
    for seed in (1107, 1108, 1109):
        cfg, model, op = load_model(f'1101_s{seed}', args.assets, args.device)
        raw = torch.from_numpy(observed.transpose(1, 2, 0).reshape(1, -1, 6)).float().to(args.device)
        mask = torch.ones((1, 1089), device=args.device, dtype=torch.bool)
        # Match the archived precision boundary: input preparation is float32.
        prepared = op.prepare_inputs(raw, mask, 'roles')
        with torch.inference_mode(), torch.autocast(device_type=args.device,
                dtype=torch.bfloat16, enabled=args.device == 'cuda'):
            rho = model(prepared, op)['density'].float()
        predicted = op.density_to_d6(rho[:, 0])[0].detach().cpu().numpy().reshape(33, 33, 6).transpose(2, 0, 1)
        np.savez_compressed(args.output / f's{seed}.npz', density_zyx=rho[0, 0].cpu().numpy(),
                            observed_d6_33=observed, predicted_d6_33=predicted)
        density = rho[0, 0].cpu().numpy()
        np.savez_compressed(args.output / f'vinton_Full_s{seed}.npz', density_zyx=density,
                            observed_d6_33=observed, predicted_d6_33=predicted)
        rows.extend(dict(method='Full', seed=seed, **r) for r in fit_metrics(observed, predicted))
        z, y, x = map(int, np.unravel_index(np.argmax(density), density.shape))
        cuts[str(seed)] = dict(peak_z=z, peak_y=y, peak_x=x,
            peak_depth_m=(15.5-z)*62.5, peak_x_m=(x+.5)*125, peak_y_m=(y+.5)*125)
        geometries.append(dict(method='Full', seed=seed, peak_density_gcc=float(density.max()),
            **cuts[str(seed)], **structure_at_threshold(density, .05)))
    write_csv(args.output / 'component_metrics.csv', rows)
    write_csv(args.output / 'vinton_component_metrics.csv', rows)
    write_csv(args.output / 'vinton_geometry.csv', geometries)
    write_json(args.output / 'vinton_peak_cuts.json', cuts)
    write_json(args.output / 'scope.json', dict(density_ground_truth_available=False,
        selected_display_seed=1107, selection='historical mean six-component R2; all seeds retained',
        fine_tuning=False, posthoc_density_correction=False, receiver_grid=[33, 33]))


def main():
    from release_runtime import registry, LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List the 27 frozen model identities')
    for command in ('train', 'evaluate', 'field'):
        p = sub.add_parser(command)
        p.add_argument('--assets', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
        if command != 'field':
            p.add_argument('--model', choices=sorted(registry()), required=True)
        if command == 'train':
            p.add_argument('--smoke', action='store_true', help='One step on 16 samples; NOT a paper result')
            p.add_argument('--resume', action='store_true')
            p.add_argument('--stage1-checkpoint', type=Path, help='Best checkpoint from your matching stage-1 rerun')
        if command == 'evaluate':
            p.add_argument('--split', choices=['validation_iid', 'test_locked'], required=True)
            p.add_argument('--checkpoint', type=Path,
                           help='Trusted best checkpoint from your complete matching rerun; default is the historical registry')
        if command == 'field':
            p.add_argument('--field-directory', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'list':
        for key, row in registry().items():
            print(f'{key:20} {LABELS[row["code"]]:20} epoch={row["epoch"]}')
        return
    import torch
    torch.set_num_threads(2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for diagnostics')
    {'train': train, 'evaluate': evaluate, 'field': field}[args.command](args)


if __name__ == '__main__':
    main()
