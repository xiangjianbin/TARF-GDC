"""Self-contained CPU example: analytic prism, noisy roles, TARF, one update.

No external data, weights, CUDA, or 409-MB forward matrix are needed. This checks
installation and model I/O, NOT the paper's trained performance or full GDC.
"""
from pathlib import Path
import argparse
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'dataset_generation')]
from v017.physics import independent_prism_tensor
from v024_inversion.physics import tensor_roles, Normalization
from v024_inversion.model import build_model
from v024_inversion.utils import read_json, seed_everything, write_json


def run(output):
    start = time.time()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    seed_everything(1107)
    cfg = read_json(ROOT / 'configs/1001_s1107.json')
    cfg.pop('staged_refinement', None)
    norm = Normalization.from_json(ROOT / 'assets/normalization.json')
    model = build_model(cfg, norm).cpu().eval()
    x, y = np.meshgrid(np.linspace(0, 4000, 33), np.linspace(0, 4000, 33))
    receivers = np.c_[x.ravel(), y.ravel(), np.ones(x.size)]
    clean = independent_prism_tensor(receivers, ((1250,2250),(1250,2250),(-500,-250)), .5)
    sigma = np.array([.425,.31,.7,.425,.7,.74])
    observed = clean + np.random.default_rng(1107).normal(size=clean.shape) * sigma
    roles = tensor_roles(torch.from_numpy(observed).float()[None])
    scales = torch.tensor([norm.scalar_rms, norm.spin1_rms, norm.spin1_rms,
                           norm.spin2_rms, norm.spin2_rms])
    image = (roles/scales).transpose(1,2).reshape(1,5,33,33)
    prepared = dict(input_image=image, role_image=image, scalar=image[:,:1],
                    spin1=image[:,1:3], spin2=image[:,3:5], mask_image=torch.ones(1,1,33,33))
    truth = torch.zeros(1,1,16,32,32)
    truth[:,:,8:12,10:18,10:18] = .5
    before = model(prepared)['density']
    loss = (before-truth).square().mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    with torch.inference_mode():
        after = model(prepared)['density']
    if after.shape != truth.shape or not torch.isfinite(after).all():
        raise AssertionError('Invalid density output')
    torch.save(model.state_dict(), output / 'demo_state.pt')
    restored = build_model(cfg, norm).cpu().eval()
    restored.load_state_dict(torch.load(output / 'demo_state.pt', weights_only=True))
    with torch.inference_mode():
        replay = restored(prepared)['density']
    if not torch.equal(after, replay):
        raise AssertionError('Checkpoint replay differs')
    np.savez_compressed(output / 'demo_arrays.npz', density_true=truth[0,0].numpy(),
                        density_predicted=after[0,0].numpy(), clean_d6=clean, observed_d6=observed)
    result = dict(passed=True, trained_paper_model=False, gdc_exercised=False,
                  output_shape=list(after.shape), finite=True, checkpoint_replay_exact=True,
                  initial_mse=float(loss.detach()), final_mse=float((after-truth).square().mean()),
                  elapsed_seconds=time.time()-start)
    write_json(output / 'result.json', result)
    print(result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/demo')
    run(p.parse_args().output)
