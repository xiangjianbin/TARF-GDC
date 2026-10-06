"""Replay the training-only four-batch physical-loss calibration without fitting."""
from pathlib import Path
import argparse
import statistics
import sys
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from release_runtime import config_for,verify_operator
from v024_inversion.data import SingleSampleDataset
from v024_inversion.physics import Normalization,build_operator
from v024_inversion.model import build_model
from v024_inversion.losses import _role_physics_loss
from v024_inversion.utils import seed_everything,write_json,read_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(2); seed_everything(1107); verify_operator(a.assets)
    cfg=config_for('0001_s1107',a.assets)
    norm=Normalization.from_json(cfg['data']['normalization_path'])
    operator=build_operator(cfg,norm,ROOT).cuda()
    model=build_model(cfg,norm).cuda().eval()
    ds=SingleSampleDataset(cfg['data']['directory'],'train','online',limit=32,
                           cache_directory=cfg['data']['cache_directory'])
    rows=[]
    for batch in DataLoader(ds,batch_size=8,shuffle=False,num_workers=0):
        batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        prepared=operator.prepare_inputs(batch['raw6'],batch['receiver_mask'],'roles')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            initial=model(prepared)['density']
        prediction=initial.float().detach().requires_grad_(True)
        ld=(prediction-batch['density'][:,None]).square().mean()
        lp=_role_physics_loss(prediction,batch,operator,cfg['loss'])
        gd=torch.autograd.grad(ld,prediction,retain_graph=True)[0].norm()
        gp=torch.autograd.grad(lp,prediction)[0].norm()
        error=(operator.density_to_d6(batch['density'])-batch['clean_raw6']).norm()/batch['clean_raw6'].norm()
        if error>2e-4: raise ValueError('Forward labels disagree')
        rows.append(dict(proposed_weight=.2*float(gd)/max(float(gp),1e-30),label_relative_error=float(error)))
    weight=statistics.median(row['proposed_weight'] for row in rows)
    expected=read_json(ROOT/'assets/physics_calibration.json')['physics_weight']
    passed=abs(weight-expected)<=abs(expected)*2e-6
    write_json(a.output,dict(passed=passed,physics_weight=weight,expected=expected,records=rows,
                            split='train',samples=32,validation_or_test_used=False))
    if not passed: raise ValueError('Calibration differs from frozen weight')
    print('PASS',weight)


if __name__=='__main__': main()
