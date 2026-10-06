"""Auditable independent forward example, adapted to the existing 12-panel layout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np

from rotation_helpers import rotate, relative_rms, dump, COMPONENTS

from figure_runtime import OUTPUT, SOURCE_DATA, ASSETS
ROOT = OUTPUT
OUT = OUTPUT
SOURCE = SOURCE_DATA
ANGLE = 22.5
BLUE, ORANGE = '#0072B2', '#D55E00'


def xy(x,y):
    c,s = np.cos(np.deg2rad(ANGLE)),np.sin(np.deg2rad(ANGLE))
    return 2+c*(x-2)-s*(y-2),2+s*(x-2)+c*(y-2)


def prepare(archive,operator):
    SOURCE.mkdir(parents=True,exist_ok=True)
    with np.load(archive/'selected_pair_arrays.npz',allow_pickle=False) as z:
        data = {k:z[k].copy() for k in z.files}
    provenance = json.loads((archive/'selected_pair_provenance.json').read_text())
    checks = {}
    g = np.load(operator,mmap_mode='r')
    rotation = data['rotation_matrix']
    assert np.allclose(rotation@rotation.T,np.eye(3),atol=1e-14)
    for role in ('A','B'):
        rho,raw = data['density_'+role],data['original_d6_'+role]
        for arr,key in ((rho,'density_sha256'),(raw,'clean_d6_sha256')):
            assert hashlib.sha256(arr.tobytes()).hexdigest()==provenance['endpoints'][role][key]
        predicted = (g@rho.reshape(-1)).reshape(33,33,6).transpose(2,0,1)
        checks[role+'_forward_max_error_E'] = float(np.max(np.abs(predicted-raw)))
        checks[role+'_forward_relative_L2'] = float(np.linalg.norm(predicted-raw)/np.linalg.norm(raw))
        assert checks[role+'_forward_relative_L2'] < 1e-6
        turned = rotate(raw.astype(float),ANGLE)
        assert np.allclose(turned,data['rotated_d6_'+role],rtol=1e-12,atol=1e-12)
        tensor = np.array([[raw[0],raw[1],raw[2]],[raw[1],raw[3],raw[4]],[raw[2],raw[4],raw[5]]],dtype=float)
        expected = np.einsum('ia,abxy,jb->ijxy',rotation,tensor,rotation)
        assert np.allclose(turned,expected[[0,0,0,1,1,2],[0,1,2,1,2,2]],atol=1e-12)
        assert np.array_equal(turned[5],raw[5])
        assert np.allclose(rotate(turned,-ANGLE),raw,rtol=1e-12,atol=1e-12)
        assert np.allclose(turned[[0,3,5]].sum(0),raw[[0,3,5]].sum(0),atol=2e-5)
        weights = np.array([1,2,2,1,2,1])[:,None,None]
        assert np.allclose((weights*turned**2).sum(0),(weights*raw.astype(float)**2).sum(0),rtol=1e-12)
        data['rotated_d6_'+role] = turned
    line = int(data['receiver_row'])
    rows=[]
    for frame in ('original','rotated'):
        for scope in ('grid','line'):
            a,b = [data[frame+'_d6_'+r].astype(float) for r in ('A','B')]
            if scope=='line': a,b=a[:,line],b[:,line]
            relative,absolute,scale=relative_rms(a,b,(1,2) if scope=='grid' else 1)
            if frame=='rotated': assert relative[5]<.01 and (relative[:5]>.20).all()
            rows.extend(dict(frame=frame,scope=scope,component='T'+c,relative_rms=float(relative[i]),
                             difference_rms_E=float(absolute[i]),endpoint_rms_E=float(scale[i])) for i,c in enumerate(COMPONENTS))
    a,b = [data['rotated_d6_'+r] for r in ('A','B')]
    _,_,line_scale=relative_rms(a,b,2)
    assert (line_scale[:,line]>=.5*line_scale.max(axis=1)).all()
    provenance.update(scope='Independent illustrative pair from V022 archive; not a member of the V023 train/validation/test counts or any reported performance statistics.',
                      selection='Existing archived example; recomputed all six grid/line metrics. V023 scan at 0 and 22.5 degrees had no pair satisfying all thresholds.',
                      current_operator_sha256=hashlib.sha256(operator.read_bytes()).hexdigest(),
                      original_Tzz_dprime_at_V023_sigma=float(np.linalg.norm((data['original_d6_A'][5].astype(float)-data['original_d6_B'][5])/.740)),
                      current_checks=checks,rotation_check='R T R^T; inverse, Tzz, trace and Frobenius invariance passed; both model cell vertices and receivers rotate together.',
                      uncertainty='No noise or uncertainty intervals; one illustrative pair, not a population estimate.',
                      difference_3d_threshold_gcc=.005,
                      selection_thresholds=dict(Tzz_relative_rms_max=.01,other5_relative_rms_min=.20,
                          apply_to='Both whole grid and one common 33-point line',
                          line_signal_floor='Each component RMS >= half its maximum across receiver rows'))
    np.savez_compressed(SOURCE/'independent_example_arrays.npz',**data)
    dump(SOURCE/'independent_example_provenance.json',provenance)
    with (SOURCE/'independent_example_component_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    profile=[]
    for k in range(33):
        r=dict(s_km=float(data['line_s_km'][k]),x_km=float(data['receiver_x_km'][line,k]),y_km=float(data['receiver_y_km'][line,k]))
        for role in ('A','B'):
            r.update({'T'+c+'_'+role+'_E':float(data['rotated_d6_'+role][j,line,k]) for j,c in enumerate(COMPONENTS)})
        profile.append(r)
    with (SOURCE/'independent_example_profiles.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=profile[0]);w.writeheader();w.writerows(profile)
    print(json.dumps({'forward_checks':checks,'metrics':rows,'source_scope':provenance['scope']},indent=2),flush=True)
    return data,provenance


def export(fig,stem):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    ignored=set()
    for ax in fig.axes:
        for axis in ((ax.xaxis,ax.yaxis,ax.zaxis) if hasattr(ax,'zaxis') else (ax.xaxis,ax.yaxis)):
            low,high=sorted(axis.get_view_interval())
            for loc,tick in zip(axis.get_majorticklocs(),axis.get_major_ticks()):
                if loc<low or loc>high: ignored.update((id(tick.label1),id(tick.label2)))
    for label in fig.findobj(match=mpl.text.Text):
        if not label.get_visible() or not label.get_text() or id(label) in ignored: continue
        box=label.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:
            outside.append(label.get_text())
    assert not outside, outside
    fig.savefig(OUT/(stem+'.svg'))
    fig.savefig(OUT/(stem+'.pdf'),dpi=600)
    fig.savefig(OUT/(stem+'.png'),dpi=300)
    fig.savefig(OUT/(stem+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    dump(OUT/(stem+'_render_checks.json'),dict(width_mm=fig.get_figwidth()*25.4,height_mm=fig.get_figheight()*25.4,
        canvas_text_overflow=outside,all_33_points_retained=True,smoothing=False,amplitude_rescaling=False,
        PDF_SVG_editable_text=True,raster_dpi=600))
    plt.close(fig)


def profiles(fig,data,frame,bottom,height,gap,letters):
    handles=None
    for j,c in enumerate((5,0,3,1,2,4)):
        row,col=divmod(j,3)
        ax=fig.add_axes([.09+col*.315,bottom-row*gap,.235,height])
        for role,color,ls in (('A',BLUE,'-'),('B',ORANGE,'--')):
            values=data[frame+'_d6_'+role][c,int(data['receiver_row'])]
            ax.plot(data['line_s_km'],values,color=color,ls=ls,lw=1.35,label='Model '+role)
        a,b=[data[frame+'_d6_'+r][c,int(data['receiver_row'])] for r in ('A','B')]
        relative=float(relative_rms(a,b,0)[0])
        span=max(a.max(),b.max())-min(a.min(),b.min())
        ax.set(xlim=(0,4),ylim=(min(a.min(),b.min())-.12*span,max(a.max(),b.max())+.28*span),xticks=[0,2,4])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_title('T'+COMPONENTS[c],fontsize=9,pad=4)
        ax.text(.97,.97,f'{100*relative:.2f}%',ha='right',va='top',transform=ax.transAxes,fontsize=8)
        ax.text(-.16,1.04,chr(letters+j),transform=ax.transAxes,weight='bold',fontsize=10)
        ax.set_ylabel('E',labelpad=1)
        if row==1: ax.set_xlabel('s (km)',labelpad=1)
        ax.grid(axis='y',color='#E4E7E8',lw=.5)
        handles=ax.get_legend_handles_labels()
    return handles


def render(data,metadata):
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Liberation Sans','Arial'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(7.204724409448819,8.425196850393702))  # 183 x 214 mm
    fig.text(.5,.981,'Independent example · horizontal rotation 22.5°',ha='center',fontsize=8)
    fields=[data['density_A'],data['density_B'],data['density_difference']]
    norms=[Normalize(-.7,.7),Normalize(-.7,.7),Normalize(-.4,.4)]
    titles=['Model A','Model B','A − B']
    supported=(np.abs(fields[0])>=.005)|(np.abs(fields[1])>=.005)
    iz,iy,ix=np.where(supported)
    xc,yc=xy((ix+.5)*.125,(iy+.5)*.125)
    xmin,xmax=float(xc.min()-.20),float(xc.max()+.20)
    ymin,ymax=float(yc.min()-.20),float(yc.max()+.20)
    dmax=float((15.5-iz.min())*.0625+.10)
    xx,yy,dd=np.meshgrid(np.linspace(0,4,33),np.linspace(0,4,33),np.linspace(0,1,17),indexing='ij')
    xx,yy=xy(xx,yy)
    vx,vy=np.meshgrid(np.linspace(0,4,33),np.linspace(0,4,33),indexing='xy')
    vx,vy=xy(vx,vy)
    for col,(field,norm,title) in enumerate(zip(fields,norms,titles)):
        ax=fig.add_axes([.027+col*.315,.744,.267,.213],projection='3d')
        displayed=field[::-1].transpose(2,1,0)
        ax.voxels(xx,yy,dd,np.abs(displayed)>=.005,facecolors=mpl.colormaps['RdBu_r'](norm(displayed)),
                  edgecolor='#89959C',linewidth=.15,shade=False)
        ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),zlim=(dmax,0))
        ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3));ax.zaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel('x',labelpad=-6);ax.set_ylabel('y',labelpad=-6)
        ax.set_zlabel('')
        ax.text2D(1.045,.70,'d',transform=ax.transAxes,fontsize=8)
        ax.tick_params(pad=-2,labelsize=8)
        ax.set_box_aspect((xmax-xmin,ymax-ymin,dmax));ax.set_proj_type('ortho');ax.view_init(elev=28,azim=-58)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.pane.fill=False;axis._axinfo['grid']['color']=(.85,.87,.88,1)
        ax.text2D(.02,1.02,chr(97+col),transform=ax.transAxes,fontsize=10,weight='bold')
        ax.text2D(.50,1.02,title,transform=ax.transAxes,ha='center',fontsize=9)
        ax2=fig.add_axes([.10+col*.315,.528,.224,.190])
        ax2.pcolormesh(vx,vy,field[int(data['slice_index'])],cmap='RdBu_r',norm=norm,shading='flat',rasterized=True)
        line=int(data['receiver_row'])
        ax2.plot(data['receiver_x_km'][line],data['receiver_y_km'][line],ls='--',color='#444444',lw=.9)
        ax2.set(xlim=(xmin,xmax),ylim=(ymin,ymax),aspect='equal')
        ax2.xaxis.set_major_locator(MaxNLocator(3));ax2.yaxis.set_major_locator(MaxNLocator(3))
        ax2.set_xlabel('x (km)',labelpad=1);ax2.set_ylabel('y (km)',labelpad=1)
        ax2.text(-.15,1.06,chr(100+col),transform=ax2.transAxes,fontsize=10,weight='bold')
    fig.text(.5,.731,'Section depth: 0.28125 km · d increases downward · coordinates in km',ha='center',fontsize=8)
    for rect,norm,label in (([.15,.450,.42,.010],norms[0],'Density contrast (g cm⁻³)'),
                            ([.735,.450,.20,.010],norms[2],'A − B (g cm⁻³)')):
        cax=fig.add_axes(rect)
        cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap='RdBu_r'),cax=cax,orientation='horizontal',ticks=[norm.vmin,0,norm.vmax])
        cb.set_label(label,fontsize=8,labelpad=2);cb.ax.xaxis.set_label_position('top');cb.ax.tick_params(pad=1,length=2)
    fig.text(.5,.406,'Same receiver line · relative RMS difference (%)',ha='center',fontsize=8)
    handles,labels=profiles(fig,data,'rotated',.250,.123,.178,103)
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.002),ncol=2,frameon=False)
    export(fig,'Figure03')
    control=plt.figure(figsize=(7.204724409448819,4.409448818897638))  # 183 x 112 mm
    control.text(.5,.97,'Same pair and line, original coordinates (no rotation)',ha='center',fontsize=9)
    handles,labels=profiles(control,data,'original',.585,.27,.43,97)
    control.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.002),ncol=2,frameon=False)
    export(control,'Figure03_original_coordinates_control')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--operator',type=Path)
    args=parser.parse_args()
    if args.archive and args.operator:
        arrays,metadata=prepare(args.archive,args.operator)
    else:
        with np.load(SOURCE/'independent_example_arrays.npz',allow_pickle=False) as z: arrays=dict(z)
        metadata=json.loads((SOURCE/'independent_example_provenance.json').read_text())
    render(arrays,metadata)
