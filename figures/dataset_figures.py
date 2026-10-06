"""Reproduce chapter 2 figures from the frozen V023 data and export source data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, FuncFormatter

from figure_runtime import OUTPUT, SOURCE_DATA, ASSETS
HERE = OUTPUT
DATA = ASSETS / 'dataset'
OUT = OUTPUT
SOURCE = SOURCE_DATA
SPLITS = ('train', 'validation_iid', 'test_locked')
TRACKS = ('ordinary', 'separable', 'shape')
SIGMA = np.array([.425, .31, .70, .425, .70, .74])
SUBTYPES = ('ellipsoid', 'superellipsoid', 'box', 'cylinder', 'lens', 'frustum',
            'dipping_slab', 'dike', 'wedge', 'faulted_body', 'curved_layer',
            'grf_levelset', 'harmonic_levelset', 'blob_union', 'body_flank',
            'notched_body', 'weak_appendage', 'density_gradient', 'zoned_density')
TITLES = ('Ellipsoid', 'Superellipsoid', 'Box', 'Cylinder', 'Lens', 'Frustum',
          'Dipping slab', 'Dike', 'Wedge', 'Faulted body', 'Curved layer',
          'GRF level set', 'Harmonic level set', 'Blob union', 'Body with flank',
          'Notched body', 'Weak appendage', 'Density gradient', 'Zoned density')
COMPONENTS = ('Txx', 'Txy', 'Txz', 'Tyy', 'Tyz', 'Tzz')
COLORS = {'ordinary': '#7B858D', 'separable': '#3079A5', 'shape': '#C77A41'}
CMAP = mpl.colormaps['RdBu_r']
CELL = np.array([.125, .125, .0625])
DEPTH = (15.5 - np.arange(16)) * .0625

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.unicode_minus': False,
    'savefig.facecolor': 'white', 'legend.frameon': False,
})


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def dump_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=2)
def shard_arrays(split, shard):
    with np.load(DATA / split / shard, allow_pickle=False) as data:
        return {k: data[k] for k in ('density_zyx', 'clean_d6', 'sample_id')}


def endpoint(split, index, record):
    spec = json.loads((DATA / split / 'index.json').read_text())
    shard = next(s for s in spec['shards'] if s['start'] <= index < s['stop'])
    raw = shard_arrays(split, shard['file'])
    local = index - shard['start']
    assert str(raw['sample_id'][local]) == record['sample_id']
    arrays = {k: raw[k][local].copy() for k in ('density_zyx', 'clean_d6')}
    for key, hashkey in [('density_zyx', 'density_sha256'), ('clean_d6', 'clean_d6_sha256')]:
        assert hashlib.sha256(arrays[key].tobytes()).hexdigest() == record[hashkey]
        assert np.isfinite(arrays[key]).all()
    assert (np.abs(arrays['density_zyx']) >= .005).sum() == record['support_cells']
    return arrays


def metrics(a, b):
    ra, rb = a['density_zyx'].astype(float), b['density_zyx'].astype(float)
    delta = a['clean_d6'].astype(float) - b['clean_d6'].astype(float)
    d0 = np.linalg.norm(delta[5] / SIGMA[5])
    d1 = np.linalg.norm(delta[[2, 4]] / SIGMA[[2, 4], None, None])
    q = (delta[0] - delta[3]) / 2
    sq = np.sqrt(SIGMA[0]**2 + SIGMA[3]**2) / 2
    d2 = np.sqrt(np.sum((q / sq)**2) + np.sum((delta[1] / SIGMA[1])**2))
    sa, sb = np.abs(ra) >= .005, np.abs(rb) >= .005
    return {'tzz_global_dprime': float(d0), 'spin1_global_dprime': float(d1),
            'spin2_global_dprime': float(d2),
            'density_nrmse': float(np.sqrt(np.mean((ra-rb)**2) / (.5*np.mean(ra**2+rb**2)))),
            'support_jaccard': float(np.sum(sa & sb) / np.sum(sa | sb))}


def quantiles(values):
    return dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                    np.quantile(values, [0, .25, .5, .75, 1]).tolist()))


def prepare():
    SOURCE.mkdir(parents=True, exist_ok=True)
    records = {s: read_rows(DATA / s / 'records.jsonl') for s in SPLITS}
    pairs = {s: read_rows(DATA / s / 'pairs.jsonl') for s in SPLITS}
    all_pair_rows, endpoint_rows, counts = [], [], []
    summary = {'dataset_id': 'V023_STANDALONE_FULL', 'counts': {}, 'tracks': {}}
    id_sets, support_sets, group_sets = {}, {}, {}
    for split in SPLITS:
        rows = records[split]
        id_sets[split] = {r['sample_id'] for r in rows}
        support_sets[split] = {r['support_sha256'] for r in rows}
        group_sets[split] = {r['ambiguity_group_id'] for r in rows if r['ambiguity_group_id']}
        assert len(id_sets[split]) == len(rows)
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            if r['ambiguity_group_id']:
                groups[r['ambiguity_group_id']].append((i, r))
        assert all(sorted(r['ambiguity_pair_role'] for _, r in group) == ['A', 'B']
                   for group in groups.values())
        for p in pairs[split]:
            group = groups[p['ambiguity_group_id']]
            ja = float(p.get('support_jaccard', 1.0))
            row = {'split': split, 'pair_id': p['ambiguity_group_id'], 'track': p['track'],
                   'tzz_dprime': p['tzz_global_dprime'], 'spin1_dprime': p['spin1_global_dprime'],
                   'spin2_dprime': p['spin2_global_dprime'],
                   'max_spin_dprime': max(p['spin1_global_dprime'], p['spin2_global_dprime']),
                   'density_nrmse': p['density_nrmse'], 'support_jaccard': ja,
                   'support_symmetric_difference': 1-ja,
                   'sample_A': next(r['sample_id'] for _, r in group if r['ambiguity_pair_role']=='A'),
                   'sample_B': next(r['sample_id'] for _, r in group if r['ambiguity_pair_role']=='B')}
            assert row['density_nrmse'] >= .12
            if row['track'] == 'shape':
                assert row['tzz_dprime'] <= 1 and row['max_spin_dprime'] >= 25 and ja <= .7
                assert p['shape_gate_tier'] == 1
            else:
                assert row['tzz_dprime'] >= 5 and row['max_spin_dprime'] >= 5 and ja == 1
            all_pair_rows.append(row)
        index = json.loads((DATA / split / 'index.json').read_text())
        for shard in index['shards']:
            # Every stored model contributes to depth and support summaries.
            with np.load(DATA / split / shard['file'], allow_pickle=False) as arrays:
                rho = arrays['density_zyx']
                ids = arrays['sample_id']
            mask = np.abs(rho) >= .005
            support_n = mask.sum(axis=(1, 2, 3))
            depth = (mask * DEPTH[None, :, None, None]).sum(axis=(1, 2, 3)) / support_n
            for k in range(len(rho)):
                r = rows[shard['start']+k]
                assert str(ids[k]) == r['sample_id']
                assert support_n[k] == r['support_cells']
                endpoint_rows.append({'split': split, 'sample_id': r['sample_id'],
                    'track': r['generation_track'], 'subtype': r['support_subtype'],
                    'support_fraction': float(support_n[k] / 16384),
                    'support_centroid_depth_km': float(depth[k]), 'density_sign': r['density_sign']})
        n = Counter(r['generation_track'] for r in rows)
        p = Counter(r['track'] for r in pairs[split])
        count = {'split': split, 'ordinary_models': n['ordinary'], 'separable_pairs': p['separable'],
                 'shape_pairs': p['shape'], 'total_models': len(rows),
                 'independent_units': n['ordinary']+sum(p.values())}
        summary['counts'][split] = count
        counts.append(count)
        print(f'Prepared {split}: {len(rows)} models, {sum(p.values())} pairs', flush=True)
    for i, s in enumerate(SPLITS):
        for t in SPLITS[i+1:]:
            assert not (id_sets[s] & id_sets[t])
            assert not (support_sets[s] & support_sets[t])
            assert not (group_sets[s] & group_sets[t])
    for track in TRACKS:
        er = [r for r in endpoint_rows if r['track']==track]
        summary['tracks'][track] = {'n_models': len(er),
            'depth_km': quantiles([r['support_centroid_depth_km'] for r in er]),
            'deep_fraction_by_split': {s: float(np.mean([r['support_centroid_depth_km'] >= .6
                for r in er if r['split']==s])) for s in SPLITS}}
        pr = [r for r in all_pair_rows if r['track']==track]
        if pr:
            summary['tracks'][track]['n_pairs'] = len(pr)
            for key in ('tzz_dprime', 'max_spin_dprime', 'density_nrmse', 'support_jaccard'):
                summary['tracks'][track][key] = quantiles([r[key] for r in pr])
            if track=='separable':
                bins = np.histogram([r['tzz_dprime'] for r in pr], [5, 10, 20, 40, np.inf])[0]
                summary['tracks'][track]['tzz_bin_counts'] = bins.tolist()
                summary['tracks'][track]['tzz_bin_percent'] = (100*bins/len(pr)).tolist()
            else:
                summary['tracks'][track]['tzz_ge_095_percent'] = 100*float(np.mean([r['tzz_dprime']>=.95 for r in pr]))
    dump_csv(SOURCE/'all_pairs.csv', all_pair_rows)
    dump_csv(SOURCE/'all_models.csv', endpoint_rows)
    dump_csv(SOURCE/'split_counts.csv', counts)
    train = records['train']
    selected, model_arrays = [], []
    for subtype in SUBTYPES:
        candidates = [(i, r) for i, r in enumerate(train)
                      if r['generation_track']=='ordinary' and r['support_subtype']==subtype]
        median = float(np.median([r['support_fraction'] for _, r in candidates]))
        i, r = min(candidates, key=lambda x: (abs(x[1]['support_fraction']-median), x[1]['sample_id']))
        model_arrays.append(endpoint('train', i, r)['density_zyx'])
        selected.append({'subtype': subtype, 'sample_id': r['sample_id'], 'index': i,
                         'n_candidates': len(candidates), 'support_fraction': r['support_fraction']})
    np.savez_compressed(SOURCE/'subtype_examples.npz', density_zyx=np.stack(model_arrays), subtypes=np.array(SUBTYPES))
    forward_examples, forward_arrays = [], []
    for subtype in ('faulted_body', 'cylinder'):
        candidates = [(i, r) for i, r in enumerate(train) if r['generation_track']=='ordinary'
                      and r['support_subtype']==subtype and .08<=r['support_fraction']<=.18]
        median = float(np.median([r['support_fraction'] for _, r in candidates]))
        i, r = min(candidates, key=lambda x: (abs(x[1]['support_fraction']-median), x[1]['sample_id']))
        forward_arrays.append(endpoint('train', i, r))
        forward_examples.append({'subtype': subtype, 'sample_id': r['sample_id'], 'index': i,
             'support_fraction': r['support_fraction'], 'density_sign': r['density_sign'], 'n_candidates':len(candidates)})
    np.savez_compressed(SOURCE/'forward_examples.npz',
        density_zyx=np.stack([a['density_zyx'] for a in forward_arrays]),
        clean_d6=np.stack([a['clean_d6'] for a in forward_arrays]))
    shape_pairs = [p for p in pairs['train'] if p['track']=='shape']
    values = np.array([[np.log10(p['tzz_global_dprime']),
        np.log10(max(p['spin1_global_dprime'],p['spin2_global_dprime'])),
        p['density_nrmse'],p['support_jaccard']] for p in shape_pairs])
    assert np.isfinite(values).all()
    med = np.median(values, axis=0)
    iqr = np.subtract(*np.quantile(values,[.75,.25],axis=0))
    assert np.all(iqr > 0)
    distances = np.sum(((values-med)/iqr)**2,axis=1)
    pi = min(range(len(shape_pairs)),key=lambda i:(distances[i],shape_pairs[i]['ambiguity_group_id']))
    pair = shape_pairs[pi]
    group = sorted([(i,r) for i,r in enumerate(train) if r['ambiguity_group_id']==pair['ambiguity_group_id']],
                   key=lambda v:v[1]['ambiguity_pair_role'])
    a,b = [endpoint('train',i,r) for i,r in group]
    recomputed = metrics(a,b)
    for key,value in recomputed.items():
        assert np.isclose(value,pair[key],rtol=2e-5,atol=1e-7), (key,value,pair[key])
    union = (np.abs(a['density_zyx'])>=.005) | (np.abs(b['density_zyx'])>=.005)
    iz = int(np.argmax(union.sum(axis=(1,2))))
    receiver_y = int(np.argmin(np.abs(np.linspace(0,4,33)-(np.argwhere(union)[:,1].mean()+.5)*.125)))
    np.savez_compressed(SOURCE/'shape_pair_example.npz',
        density_zyx=np.stack([a['density_zyx'],b['density_zyx']]),
        clean_d6=np.stack([a['clean_d6'],b['clean_d6']]),slice_iz=iz,receiver_y_index=receiver_y)
    line_rows = [{'east_km': float(x), 'north_km': receiver_y*.125, 'component': c,
                 'model_A_E':float(a['clean_d6'][ci,receiver_y,xi]),
                 'model_B_E':float(b['clean_d6'][ci,receiver_y,xi])}
                 for ci,c in enumerate(COMPONENTS) for xi,x in enumerate(np.linspace(0,4,33))]
    dump_csv(SOURCE/'shape_pair_profiles.csv',line_rows)
    pair_selection = {'pair_id': pair['ambiguity_group_id'],
        'sample_ids':[r['sample_id'] for _,r in group], 'metrics':recomputed,
        'slice_depth_km':float(DEPTH[iz]), 'slice_iz_bottom_up':iz,
        'profile_north_km':receiver_y*.125,'n_candidates':len(shape_pairs)}
    dump_json(SOURCE/'selection.json', {'subtypes':selected,'forward':forward_examples,'shape_pair':pair_selection})
    dump_json(SOURCE/'summary.json', summary)
    print(json.dumps({'summary':summary,'examples':{'forward':forward_examples,'shape':pair_selection}},ensure_ascii=False),flush=True)


def tag(ax, letter, title='', x=0, y=1.05):
    ax.text(x,y,letter,transform=ax.transAxes,fontweight='bold',fontsize=9,va='bottom')
    if title:
        ax.set_title(title,loc='center',pad=6)


def crop_bounds(densities):
    union = np.any(np.abs(densities)>=.005,axis=0)
    # Plotting d increases downward; stored z increases upward.
    idx = np.argwhere(union[::-1].transpose(2,1,0))
    lower = np.maximum(idx.min(axis=0)-1,0)*CELL
    upper = np.minimum(idx.max(axis=0)+2,[32,32,16])*CELL
    return lower,upper


def volume(ax,rho,norm,bounds=None,mask=None,small=False):
    xyz = rho[::-1].transpose(2,1,0)
    visible = (np.abs(rho)>=.005) if mask is None else mask
    visible = visible[::-1].transpose(2,1,0)
    corners = np.meshgrid(np.arange(33)*.125,np.arange(33)*.125,np.arange(17)*.0625,indexing='ij')
    ax.voxels(*corners,visible,facecolors=CMAP(norm(xyz)),shade=False,
              edgecolor='#697A87',linewidth=.08)
    lower,upper = crop_bounds(rho[None]) if bounds is None else bounds
    ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),zlim=(upper[2],lower[2]))
    ax.set_box_aspect(upper-lower)
    ax.set_proj_type('ortho')
    ax.view_init(elev=24,azim=-58)
    for i,axis in enumerate((ax.xaxis,ax.yaxis,ax.zaxis)):
        ticks=MaxNLocator(nbins=2,steps=[1,2,5]).tick_values(lower[i],upper[i])
        axis.set_ticks(ticks[(ticks>=lower[i])&(ticks<=upper[i])])
        axis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
        axis.pane.fill=False
        axis._axinfo['grid']['color']=(.87,.89,.9,1)
    # Fixed axis-name positions avoid collisions with varying 3D tick locations.
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.text2D(.22,.04,'x',transform=ax.transAxes,fontsize=6 if small else 7)
    ax.text2D(.90,.10,'y',transform=ax.transAxes,fontsize=6 if small else 7)
    ax.text2D(1.02,.78,'d',transform=ax.transAxes,fontsize=6 if small else 7)
    ax.tick_params(pad=-3 if small else -1,length=2,labelsize=5.5 if small else 6)


def save(fig,stem):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    # Native-backend final-size preview. PDF text sizes are audited separately.
    fig.savefig(OUT/f'{stem}.svg',dpi=600)
    fig.savefig(OUT/f'{stem}.pdf',dpi=600)
    fig.savefig(OUT/f'{stem}.png',dpi=300)
    fig.savefig(OUT/f'{stem}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    print(f'Exported {stem}',flush=True)


def figure1():
    data=np.load(SOURCE/'subtype_examples.npz')
    fig=plt.figure(figsize=(183/25.4,230/25.4))
    norm=Normalize(-.8,.8)
    for i,rho in enumerate(data['density_zyx']):
        row,col=divmod(i,4)
        left,bottom=.035+col*.242,.802-row*.186
        ax=fig.add_axes([left,bottom,.189,.148],projection='3d')
        volume(ax,rho,norm,small=True)
        fig.text(left,bottom+.159,chr(97+i),fontsize=9,fontweight='bold')
        fig.text(left+.024,bottom+.159,TITLES[i],fontsize=6.8)
    cax=fig.add_axes([.805,.102,.018,.095])
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=CMAP),cax=cax,ticks=[-.8,0,.8])
    cb.ax.tick_params(labelsize=7,length=2)
    cb.set_label('ρ (g cm⁻³)',fontsize=8,labelpad=6)
    fig.text(.776,.064,'x, y, d (km)\nDepth increases downward',fontsize=6.5,linespacing=1.5)
    save(fig,'fig1_model_subtypes_v023')


def figure2():
    data=np.load(SOURCE/'forward_examples.npz')
    rho,resp=data['density_zyx'],data['clean_d6']
    fig=plt.figure(figsize=(183/25.4,185/25.4))
    norm=Normalize(-.8,.8)
    limits=np.max(np.abs(resp),axis=(0,2,3))
    assert np.all(limits>0)
    limits=np.ceil(limits/10)*10
    for case in range(2):
        top=.92-case*.46
        ax=fig.add_axes([.028,top-.265,.265,.245],projection='3d')
        volume(ax,rho[case],norm)
        fig.text(.04,top+.025,'a' if case==0 else 'h',fontsize=9,fontweight='bold')
        fig.text(.065,top+.025,('Faulted body','Cylinder')[case],fontsize=8)
        for ci,component in enumerate(COMPONENTS):
            row,col=divmod(ci,3)
            left,bottom=.345+col*.218,top-.165-row*.209
            ax2=fig.add_axes([left,bottom,.154,.1523])
            im=ax2.pcolormesh(np.linspace(0,4,33),np.linspace(0,4,33),resp[case,ci],
                shading='nearest',cmap=CMAP,norm=Normalize(-limits[ci],limits[ci]),rasterized=True)
            ax2.set(aspect='equal',xlim=(0,4),ylim=(0,4),xticks=[0,2,4],yticks=[0,2,4])
            ax2.tick_params(length=2,pad=1,labelsize=6)
            tag(ax2,chr(98+case*7+ci),component)
            if row==1:
                ax2.set_xlabel('x (km)',labelpad=1)
            else:
                ax2.set_xticklabels([])
            if col==0:
                ax2.set_ylabel('y (km)',labelpad=1)
            else:
                ax2.set_yticklabels([])
            cax=fig.add_axes([left+.163,bottom+.016,.006,.121])
            cb=fig.colorbar(im,cax=cax,ticks=[-limits[ci],0,limits[ci]])
            cb.ax.tick_params(length=1.5,pad=1,labelsize=5.5)
            cb.ax.set_title('E',fontsize=6,pad=4)
    cax=fig.add_axes([.062,.114,.18,.011])
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=CMAP),cax=cax,
                   orientation='horizontal',ticks=[-.8,0,.8])
    cb.set_label('ρ (g cm⁻³)',fontsize=8,labelpad=2)
    cb.ax.tick_params(length=2,pad=1)
    fig.text(.048,.035,'x, y, d (km)\nDepth increases downward',fontsize=6.5,linespacing=1.4)
    dump_json(SOURCE/'forward_color_limits.json',dict(zip(COMPONENTS,limits.tolist())))
    save(fig,'fig2_geometry_forward_v023')


def figure3():
    data=np.load(SOURCE/'shape_pair_example.npz')
    rho,resp=data['density_zyx'],data['clean_d6']
    iz,iy=int(data['slice_iz']),int(data['receiver_y_index'])
    sel=json.loads((SOURCE/'selection.json').read_text())['shape_pair']
    fig=plt.figure(figsize=(183/25.4,210/25.4))
    bounds=crop_bounds(rho)
    union=np.any(np.abs(rho)>=.005,axis=0)
    delta=rho[0]-rho[1]
    limit=float(np.ceil(np.max(np.abs(delta))*100)/100)
    norms=[Normalize(-.8,.8),Normalize(-.8,.8),Normalize(-limit,limit)]
    fields=[rho[0],rho[1],delta]
    titles=['Model A','Model B','A − B']
    for i in range(3):
        left=.065+i*.31
        ax=fig.add_axes([left,.745,.235,.185],projection='3d')
        volume(ax,fields[i],norms[i],bounds,mask=union if i==2 else None)
        fig.text(left-.01,.94,chr(97+i),fontsize=9,fontweight='bold')
        fig.text(left+.085,.94,titles[i],fontsize=8,ha='center')
        ax2=fig.add_axes([left+.018,.492,.185,.1612])
        im=ax2.pcolormesh((np.arange(32)+.5)*.125,(np.arange(32)+.5)*.125,fields[i][iz],
            shading='nearest',cmap=CMAP,norm=norms[i],rasterized=True)
        ax2.set(aspect='equal',xlim=(0,4),ylim=(0,4),xticks=[0,2,4],yticks=[0,2,4],xlabel='x (km)')
        ax2.set_ylabel('y (km)',labelpad=1)
        ax2.tick_params(length=2,pad=1)
        ax2.axhline(iy*.125,color='#343A40',lw=.7,ls=':')
        tag(ax2,chr(100+i),titles[i])
        cax=fig.add_axes([left+.218,.512,.008,.12])
        cb=fig.colorbar(im,cax=cax,ticks=[norms[i].vmin,0,norms[i].vmax])
        cb.ax.tick_params(labelsize=6,length=2,pad=1)
    fig.text(.5,.695,f'Horizontal slice: d = {DEPTH[iz]:.5f} km    |    ρ and Δρ (g cm⁻³)',
             fontsize=7,ha='center')
    order=[5,0,3,1,2,4]
    for k,ci in enumerate(order):
        row,col=divmod(k,3)
        left,bottom=.085+col*.307,.271-row*.181
        ax=fig.add_axes([left,bottom,.242,.12])
        ax.plot(np.linspace(0,4,33),resp[0,ci,iy],color='#3079A5',lw=1.25,label='Model A')
        ax.plot(np.linspace(0,4,33),resp[1,ci,iy],color='#C77A41',lw=1.15,ls='--',label='Model B')
        ax.set(xlim=(0,4),xticks=[0,2,4])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(length=2,pad=1)
        ax.set_ylabel('E',labelpad=2)
        if row==1:
            ax.set_xlabel('x (km)',labelpad=1)
        else:
            ax.set_xticklabels([])
        tag(ax,chr(103+k),COMPONENTS[ci],y=1.04)
        if k==0:
            ax.legend(loc='upper left',fontsize=6,handlelength=1.6,borderaxespad=.2)
    m=sel['metrics']
    fig.text(.5,.435,f'Profiles at y = {iy*.125:.3f} km; identical line for all six components',ha='center',fontsize=7)
    fig.text(.5,.023,f'Tzz d′ = {m["tzz_global_dprime"]:.3f}   |   max spin d′ = {max(m["spin1_global_dprime"],m["spin2_global_dprime"]):.1f}'
             f'   |   Density NRMSE = {m["density_nrmse"]:.3f}   |   Jaccard = {m["support_jaccard"]:.3f}',ha='center',fontsize=6.5)
    save(fig,'fig3_shape_pair_v023')


def figureS1():
    with (SOURCE/'all_pairs.csv').open() as stream:
        pairs=list(csv.DictReader(stream))
    with (SOURCE/'all_models.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    fig=plt.figure(figsize=(183/25.4,80/25.4))
    a=fig.add_axes([.077,.23,.254,.63])
    b=fig.add_axes([.408,.23,.22,.63])
    c=fig.add_axes([.736,.23,.24,.63])
    for track,label in [('separable','Separable'),('shape','Shape ambiguity')]:
        subset=[r for r in pairs if r['track']==track]
        x=np.array([float(r['tzz_dprime']) for r in subset])
        y=np.array([float(r['max_spin_dprime']) for r in subset])
        assert np.all(x>0) and np.all(y>0)
        # All pairs are retained; rasterize marks rather than subsampling them.
        a.scatter(x,y,s=2,alpha=.32,c=COLORS[track],rasterized=True,linewidths=0,label=label)
        j=np.sort([float(r['support_jaccard']) for r in subset])
        b.step(j,np.arange(1,len(j)+1)/len(j),where='post',color=COLORS[track],lw=1.2,
               ls='-' if track=='separable' else '--')
        if track=='separable':
            b.plot([1,1],[0,1],color=COLORS[track],lw=1.2)
    a.set(xscale='log',yscale='log',xlabel='Tzz d′',ylabel='max(spin-1, spin-2) d′',xlim=(.3,1500),ylim=(4,3000))
    a.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    a.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    a.axvline(1,color='#777777',lw=.65,ls=':')
    a.axvline(5,color='#777777',lw=.65,ls=':')
    a.axhline(25,color='#777777',lw=.65,ls=':')
    a.legend(loc='lower right',fontsize=6,markerscale=3,handlelength=1)
    b.set(xlim=(.15,1.025),ylim=(0,1.025),xlabel='Support Jaccard',ylabel='Cumulative fraction')
    b.axvline(.7,color='#777777',lw=.65,ls=':')
    for track,label,ls in [('ordinary','Ordinary','-'),('separable','Separable','--'),('shape','Shape ambiguity',':')]:
        d=np.sort([float(r['support_centroid_depth_km']) for r in rows if r['track']==track])
        c.step(d,np.arange(1,len(d)+1)/len(d),where='post',color=COLORS[track],ls=ls,lw=1.4,label=label)
    c.axvline(.6,ymin=.34,color='#777777',lw=.65,ls=':')
    c.set(xlim=(0,1),ylim=(0,1.025),xlabel='Support centroid depth (km)',ylabel='Cumulative fraction')
    c.legend(loc='lower right',fontsize=6,handlelength=1.6)
    for ax,letter,title in [(a,'a','Response separation'),(b,'b','Support difference'),(c,'c','Depth coverage')]:
        tag(ax,letter,title,x=-.16,y=1.08)
        ax.tick_params(length=2,pad=2)
    fig.text(.5,.048,'All 10,005 pairs (a, b) and 38,210 models (c); no subsampling',ha='center',fontsize=7)
    save(fig,'figD1_dataset_distributions_v023')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--fig',choices=['1','2','3','S1','all'])
    args=parser.parse_args()
    if args.prepare:
        prepare()
    if args.fig:
        for key,fn in [('1',figure1),('2',figure2),('3',figure3),('S1',figureS1)]:
            if args.fig in (key,'all'):
                fn()
