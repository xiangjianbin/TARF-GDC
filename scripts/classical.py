"""Portable selected L2/IRLS comparison on the fixed 200-model r0 subset.

The paper uses L2 beta=3000 and IRLS beta=0.3. IRLS starts from the historical
short-budget solution, which is included as a separate, identified asset.
Cold-start solutions are a different experiment. Iteration caps are not
convergence certificates; both KKT and outer-change criteria are reported.
"""
from pathlib import Path
import argparse
import sys
import time
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code/classical_solvers'), str(ROOT/'code')]
from classical import Problem
from fft_operator import GridConvolution
from protocol import solve_full, continue_irls, SOLVER, calibration_indices
from release_runtime import config_for, verify_operator
from v024_inversion.physics import Normalization, build_operator
from v024_inversion.trainer import build_dataset
from v024_inversion.metrics import metrics_per_sample
from v024_inversion.utils import read_json, write_json, write_csv, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['L2','IRLS'], required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--limit', type=int, default=200, help='Less than 200 is diagnostic only')
    parser.add_argument('--max-rounds', type=int, default=10, help='Each IRLS continuation round has <=80 outer iterations')
    parser.add_argument('--calibrate', action='store_true', help='Search beta on 12 training models, not validation/test')
    args = parser.parse_args()
    if not 1 <= args.limit <= 200 or args.max_rounds < 1:
        parser.error('limit must be 1..200; max-rounds must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    seed_everything(20260918)
    verify_operator(args.assets)
    cfg = config_for('0001_s1107',args.assets,args.device)
    norm = Normalization.from_json(cfg['data']['normalization_path'])
    op = build_operator(cfg,norm,ROOT).to(args.device)
    # Preserve the original precision boundary: form roles in float32, solve in float64.
    matrix = torch.einsum('ac,rcv->rav',op.role_transform,op.g6.reshape(1089,6,16384))
    matrix = (matrix/op.role_scales[None,:,None]).reshape(5445,16384)
    problem = Problem(matrix.double(),spacing=(.5,1.,1.),floor=.001)
    problem.convolution = GridConvolution(problem.matrix)
    dataset = build_dataset(cfg,'train' if args.calibrate else 'validation_iid',0)
    subset = read_json(ROOT/'evidence/subset_200.json')
    indices = calibration_indices(dataset) if args.calibrate else subset['indices'][:args.limit]
    betas = ([.01,.1,.3,1.,3.,10.,30.,100.,300.,1000.,3000.,10000.] if args.method=='L2'
             else [.1,.3,1.,3.,10.]) if args.calibrate else [3000. if args.method=='L2' else .3]
    initial = None
    if args.method == 'IRLS' and not args.calibrate:
        import pandas as pd
        folder = args.assets/'evidence/classical/IRLS_warm_start'
        if pd.read_csv(folder/'sample_id.csv').sample_id.tolist() != subset['sample_ids']:
            raise ValueError('IRLS warm-start sample order differs')
        initial = np.load(folder/'density.npy')
    all_summary = []
    for beta in betas:
        rows, predictions, diagnostics = [], [], []
        started = time.time()
        for start in range(0,len(indices),4):
            batch = default_collate([dataset[i] for i in indices[start:start+4]])
            if not args.calibrate and batch['sample_id'] != subset['sample_ids'][start:start+len(batch['sample_id'])]:
                raise ValueError('Subset identity mismatch')
            batch = {k:v.to(args.device) if torch.is_tensor(v) else v for k,v in batch.items()}
            observed = op.observed_to_normalized(batch['raw6'],'roles').reshape(len(batch['sample_id']),-1)
            context = problem.batch(observed)
            if initial is not None:
                density, info = continue_irls(context,torch.from_numpy(initial[start:start+len(observed)].copy()),beta)
            else:
                density, info = solve_full(context,args.method,beta)
            rounds = 1
            # Do not pass off an exhausted solver as a converged solution.
            while not args.calibrate and args.method=='IRLS' and not bool(info['converged'].all()) and rounds<args.max_rounds:
                density, info = continue_irls(context,density,beta)
                rounds += 1
            pred = density.float()
            metrics = metrics_per_sample({'density':pred[:,None]},batch,op,'roles')
            support, truth = batch['support'], batch['density']
            scale = (torch.where(support,truth.square(),0.).sum((1,2,3))/support.sum((1,2,3)).clamp_min(1)).sqrt().clamp_min(1e-6)
            bg = (torch.where(~support,(density-truth).square(),0.).sum((1,2,3))/(~support).sum((1,2,3)).clamp_min(1)).sqrt()/scale
            for j,sid in enumerate(batch['sample_id']):
                rows.append(dict(sample_id=sid,beta=beta,
                    **{k:float(metrics[k][j]) for k in ('density_sae','density_nrmse','foreground_nrmse','physics_nrms')},
                    selection_score=float(metrics['foreground_nrmse'][j]+bg[j])))
                diagnostics.append(dict(sample_id=sid,beta=beta,rounds=rounds,
                    relative_kkt=float(info['relative_kkt'][j]),
                    relative_change=float(info['relative_change'][j]) if 'relative_change' in info else 0.,
                    converged=bool(info['converged'][j])))
            predictions.append(pred.detach().cpu().numpy())
            print(args.method,beta,start+len(observed),'/',len(indices),flush=True)
        tag = f'{args.method}_beta{beta:g}'
        write_csv(args.output/(tag+'_metrics.csv'),rows)
        write_csv(args.output/(tag+'_solver.csv'),diagnostics)
        np.save(args.output/(tag+'_density.npy'),np.concatenate(predictions))
        all_summary.append(dict(beta=beta,score=float(np.mean([r['selection_score'] for r in rows])),
                                converged=all(r['converged'] for r in diagnostics),elapsed_seconds=time.time()-started))
    eligible = [r for r in all_summary if r['converged']]
    selected = min(eligible,key=lambda r:(r['score'],-r['beta']))['beta'] if eligible else None
    write_json(args.output/'summary.json',dict(method=args.method,calibration=args.calibrate,
        samples=len(indices),diagnostic_only=not args.calibrate and args.limit!=200,
        protocol=SOLVER,irls_historical_warm_start=initial is not None,results=all_summary,
        selected_beta=selected if args.calibrate else betas[0],
        calibration_selection_requires_all_endpoints_converged=True))
    if (args.calibrate and not eligible) or (not args.calibrate and not all(r['converged'] for r in all_summary)):
        raise RuntimeError('One or more solves did not meet the stopping criteria; inspect solver CSV')


if __name__=='__main__':
    main()
