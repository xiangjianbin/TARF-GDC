"""Re-render all six components of the explicitly requested best frozen seed.

Contract: one best-fit spatial example, not an average performance claim.
Archetype: quantitative grid. Rows: observation, prediction, signed residual.
All 1089 receivers and six components retained. No smoothing or clipping.
The seed is selected by six-component mean R2, before styling the figure.
"""
from pathlib import Path
import hashlib
import json
import string
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from figure_runtime import OUTPUT, SOURCE_DATA, ASSETS
HERE = OUTPUT
SOURCE = SOURCE_DATA
SEEDS = [1107, 1108, 1109]
COMPONENTS = ['Txx', 'Txy', 'Txz', 'Tyy', 'Tyz', 'Tzz']
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 7,
    'axes.linewidth': .6, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
})


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sources = {s: SOURCE / ('vinton_Full_s%d.npz' % s) for s in SEEDS}
    hashes = {str(s): sha(p) for s, p in sources.items()}
    arrays, rows = {}, []
    for seed, path in sources.items():
        with np.load(path) as a:
            obs = a['observed_d6_33'].astype(np.float64)
            pred = a['predicted_d6_33'].astype(np.float64)
            assert obs.shape == pred.shape == (6,33,33)
            assert np.isfinite(obs).all() and np.isfinite(pred).all()
            arrays[seed] = (obs, pred)
        for c, name in enumerate(COMPONENTS):
            o, p = obs[c].ravel(), pred[c].ravel()
            sst = np.square(o-o.mean()).sum()
            assert sst > 0
            rows.append(dict(seed=seed, component=name,
                r2=float(1-np.square(o-p).sum()/sst),
                nrms=float(np.linalg.norm(o-p)/np.linalg.norm(o)),
                rms_ratio=float(np.linalg.norm(p)/np.linalg.norm(o))))
    metrics = pd.DataFrame(rows)
    means = metrics.groupby('seed')[['r2','nrms','rms_ratio']].mean()
    selected = int(means.r2.idxmax())
    assert selected == 1107
    metrics.to_csv(HERE/'Figure12_all_seeds_per_component_recompute.csv', index=False)
    means.to_csv(HERE/'Figure12_seed_selection_basis.csv')
    obs, pred = arrays[selected]
    residual = obs-pred
    data = [obs, pred, residual]
    np.savez_compressed(HERE/'Figure12_source_data.npz',
                        seed=selected, observed_d6_33=obs,
                        predicted_d6_33=pred, residual_d6_33=residual,
                        x_m=np.linspace(0,4000,33), y_m=np.linspace(0,4000,33))
    width_inches, height_inches = 183/25.4, 114/25.4
    fig = plt.figure(figsize=(width_inches, height_inches))
    fig.text(.5, .973, 'Best-fit frozen seed %d  |  Mean R² = %.4f' % (selected, means.loc[selected,'r2']),
             ha='center', va='top', fontsize=8)
    width = .128
    starts = [.095 + c*.15 for c in range(6)]
    height = width*183/114
    bottoms = [.654, .403, .152]
    panels = []
    cmap = mpl.colormaps['RdBu_r']
    for c, name in enumerate(COMPONENTS):
        bound = float(np.max(np.abs(np.stack([d[c] for d in data]))))
        norm = Normalize(vmin=-bound, vmax=bound, clip=False)
        for r, d in enumerate(data):
            ax = fig.add_axes([starts[c],bottoms[r],width,height])
            ax.imshow(d[c], origin='lower', extent=(-.0625,4.0625,-.0625,4.0625),
                      cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
            ax.set(xticks=[0,2,4], yticks=[0,2,4])
            ax.tick_params(length=2, pad=1)
            # The half-cell extent places image pixels on the original 0..4 km receiver coordinates.
            ax.set_xlim(-.0625,4.0625); ax.set_ylim(-.0625,4.0625)
            if r == 0:
                ax.set_title(name, pad=9)
            if c == 0:
                ax.set_ylabel(['Observed','TARF-GDC','Residual'][r]+'\ny (km)', labelpad=2, fontsize=7)
            else:
                ax.set_yticklabels([])
            if r == 2:
                ax.set_xlabel('x (km)', labelpad=2, fontsize=6)
            else:
                ax.set_xticklabels([])
            ax.text(0,1.015,string.ascii_lowercase[r*6+c], transform=ax.transAxes,
                    fontsize=6, va='bottom', fontweight='bold')
            panels.append(dict(panel=string.ascii_lowercase[r*6+c], component=name,
                               role=['observation','prediction','observed-minus-predicted'][r],
                               seed=selected, receivers=1089, values_min=float(d[c].min()),
                               values_max=float(d[c].max()), color_min=-bound, color_max=bound,
                               smoothing=False, clipping=False,
                               uncertainty='None: single selected frozen prediction; aggregate spread in Figure 13'))
        cax = fig.add_axes([starts[c],.055,width,.012])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
        # Symmetric ticks stay within the full, unclipped component range.
        tick = float(np.floor(bound/5)*5)
        if tick <= 0: tick = bound
        cb.set_ticks([-tick,0,tick])
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.set_label('E', fontsize=6, labelpad=1)
    fig.canvas.draw()
    prefix = HERE/'Figure12_best_frozen_s1107_20261002'
    fig.savefig(str(prefix)+'.svg')
    fig.savefig(str(prefix)+'.pdf')
    fig.savefig(str(prefix)+'.png',dpi=600)
    fig.savefig(str(prefix)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(str(prefix)+'_preview.png',dpi=300)
    plt.close(fig)
    assert hashes == {str(s):sha(p) for s,p in sources.items()}
    record = dict(selected_seed=selected, selection_rule='maximum equally weighted six-component mean R2 among existing three frozen predictions',
                  selection_requested_by_author=True, previous_figure_seed=1107, source_values_changed=False,
                  field_refinement_used=False, evaluation_grid=[33,33], all_seeds=SEEDS,
                  all_seed_means=means.reset_index().to_dict(orient='records'),
                  aggregate_mean=means.mean().to_dict(), aggregate_sample_sd=means.std(ddof=1).to_dict(),
                  source_sha256=hashes, panels=panels, dimensions_mm=[183,114],
                  source_mapping={'observed_d6_33':'row1','predicted_d6_33':'row2','observed_d6_33 - predicted_d6_33':'row3'},
                  color_rule='same symmetric full-range scale within each component across all three rows',
                  exclusions='Only seed selection requested by author; no component or receiver exclusions',
                  visual_qa='Pending panel-by-panel review')
    (HERE/'Figure12_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:record[k] for k in ['selected_seed','all_seed_means','aggregate_mean']},ensure_ascii=False))


if __name__ == '__main__':
    main()
