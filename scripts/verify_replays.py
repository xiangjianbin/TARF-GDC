"""Compare replayed validation/field outputs and all 200 classical source densities.

Classical comparisons use float64 reductions from the preserved original
densities, not a substituted beta or a newly optimized solution.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--validation',type=Path)
    p.add_argument('--field',type=Path)
    p.add_argument('--field-reference',type=Path,help='Authorized historical field source-data directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    a=p.parse_args();report={}
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    if a.validation:
        summary=json.loads((a.validation/'summary.json').read_text())
        code,seed=summary['model'].split('_s')
        ref=pd.read_csv(ROOT/'evidence/overall_seed_metrics.csv',dtype={'code':str})
        row=ref[(ref.code==code)&(ref.seed==int(seed))&(ref.split==summary['split'])].iloc[0]
        errors={k:abs(float(row[k])-v) for k,v in summary['aggregate'].items()}
        assert max(errors.values())<1e-7,errors
        ref=pd.read_csv(ROOT/'evidence/pair_seed_metrics.csv',dtype={'code':str})
        selected=ref[(ref.code==code)&(ref.seed==int(seed))&(ref.split==summary['split'])]
        for _,r in selected.iterrows():
            got=summary['pair_subgroups']['track/'+r.track]
            for old,new in [('difference_nrmse','difference_nrmse'),('difference_cosine','pdacc_cos')]:
                assert abs(r[old]-got[new])<1e-7
        report['validation']=dict(passed=True,samples=summary['sample_count'],pairs=summary['pair_count'],
            realizations=5,aggregate_max_errors=errors,paired_metrics_checked=True)
    if a.field:
        if a.field_reference is None:
            p.error('--field requires --field-reference')
        field=[]
        from field_preprocessing import structure_at_threshold
        for seed in (1107,1108,1109):
            with np.load(a.field/f's{seed}.npz') as got, np.load(a.field_reference/f'vinton_Full_s{seed}.npz') as ref:
                errors={k:float(np.max(np.abs(got[k]-ref[k]))) for k in got.files}
                assert all(v==0 for v in errors.values()),errors
                field.append(dict(seed=seed,array_max_errors=errors,
                    geometry=structure_at_threshold(got['density_zyx'],.05),
                    peak_density_gcc=float(got['density_zyx'].max())))
        report['field']=dict(passed=True,seeds=field)
    with np.load(a.assets/'figure_data/classical/matched_all_predictions.npz') as matched:
        truth=matched['truth'].astype(np.float64);raw=matched['raw6'].astype(np.float64)
        ids=matched['sample_ids'].tolist()
        predictions={f'TARF-GDC_s{s}':matched[f'Full_s{s}'].copy() for s in (1107,1108,1109)}
    for name,folder in [('Smooth_L2_beta3000','L2_beta3000'),('IRLS_original_beta0.3','IRLS_beta0.3')]:
        loc=a.assets/'evidence/classical'/folder
        assert pd.read_csv(loc/'sample_id.csv').sample_id.tolist()==ids
        predictions[name]=np.load(loc/'density.npy')
        info=pd.read_csv(loc/'info.csv')
        assert len(info)==200 and info.converged.all()
        assert info.relative_kkt.max()<=1e-5
        if 'IRLS' in folder:
            assert info.relative_change.max()<=1e-3
    g=torch.from_numpy(np.load(a.assets/'operator/g6.npy')).to(device=a.device,dtype=torch.float64)
    role=torch.tensor([[-1/3,0,0,-1/3,0,2/3],[0,0,1,0,0,0],[0,0,0,0,1,0],
                       [.5,0,0,-.5,0,0],[0,1,0,0,0,0]],device=a.device,dtype=torch.float64)
    observed=torch.from_numpy(raw).to(a.device)@role.T
    scales=observed.square().mean(1).sqrt().clamp_min(1e-6)
    reference=pd.read_csv(ROOT/'evidence/comparison_200_per_sample_float64.csv')
    checks=[]
    for name,rho in predictions.items():
        error=rho.astype(np.float64)-truth;axes=(1,2,3);mask=abs(truth)>=.005
        values=dict(density_sae=abs(error).sum(axes),
            density_nrmse=np.sqrt(np.square(error).sum(axes)/np.square(truth).sum(axes)),
            foreground_nrmse=np.sqrt(np.square(error*mask).sum(axes)/np.square(truth*mask).sum(axes)))
        physics=[]
        with torch.no_grad():
            for start in range(0,200,16):
                r=torch.from_numpy(rho[start:start+16].astype(np.float64)).to(a.device).flatten(1)
                pred=(r@g.T).reshape(-1,1089,6)@role.T
                nrms=((pred-observed[start:start+16])/scales[start:start+16,None,:]).square().mean((1,2)).sqrt()
                physics.extend(nrms.cpu().tolist())
        values['physics_nrms']=np.array(physics)
        saved=reference[reference.method==name]
        assert saved.sample_id.tolist()==ids,(name,reference.method.unique())
        errors={k:float(abs(v-saved[k].to_numpy()).max()) for k,v in values.items()}
        assert max(errors.values())<1e-7,errors
        checks.append(dict(method=name,n=200,max_errors=errors,means={k:float(v.mean()) for k,v in values.items()}))
    report.update(classical=dict(passed=True,methods=checks),passed=True)
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',a.output)


if __name__=='__main__':
    main()
