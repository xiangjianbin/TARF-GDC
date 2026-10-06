"""Publication plots from the frozen recomputation bundle; no metric selection."""
from pathlib import Path
import json
import string
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from figure_runtime import OUTPUT, SOURCE_DATA, ASSETS
ROOT = OUTPUT
SOURCE = SOURCE_DATA
SRC = SOURCE_DATA
MATCHED_SOURCE = SRC/'matched_reconstructions_reoptimized.npz'
OUT = OUTPUT
CMAP = mpl.colormaps['RdBu_r']
DEPTH = (15.5-np.arange(16))*.0625
CENTRE = (np.arange(32)+.5)*.125
COLORS = {'Full':'#5B8E7D', 'L2':'#777777', 'IRLS':'#4C78A8'}
NAMES = {'Full':'TARF-GDC', 'L2':'Smooth L2', 'IRLS':'IRLS'}
PANELS = []
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
    'font.size':7, 'axes.labelsize':7, 'axes.titlesize':7, 'xtick.labelsize':6, 'ytick.labelsize':6,
    'legend.fontsize':6.5, 'axes.linewidth':.65, 'axes.spines.right':False, 'axes.spines.top':False,
    'legend.frameon':False, 'svg.fonttype':'none', 'pdf.fonttype':42, 'savefig.facecolor':'white'})


def save(fig, n):
    OUT.mkdir(exist_ok=True)
    fig.canvas.draw()
    for i,ax in enumerate(fig.axes):
        PANELS.append(dict(figure=n,panel=i+1,title=ax.get_title(),xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),
            interval='TARF-GDC mean ± sample SD and all 3 seeds' if n==13 and i<3 else 'none; fixed spatial array or color scale',
            source='vinton_component_metrics.csv' if n==13 else
                   MATCHED_SOURCE.name if n<12 else 'vinton_Full_s1107.npz'))
    fig.savefig(OUT/f'Figure{n:02d}.svg')
    fig.savefig(OUT/f'Figure{n:02d}.pdf')
    fig.savefig(OUT/f'Figure{n:02d}.png',dpi=600)
    plt.close(fig)
    print('Rendered', n, flush=True)


def colorbar(fig,norm,rect,label='Density contrast (g/cm³)'):
    ax=fig.add_axes(rect)
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=CMAP),cax=ax,orientation='horizontal')
    cb.set_label(label,fontsize=7,labelpad=2)
    cb.ax.tick_params(labelsize=6,length=2,pad=2)


def volume(ax,rho,norm,first=False):
    displayed=rho[::-1].transpose(2,1,0)
    filled=np.abs(displayed)>=.05
    padded=np.pad(filled,1)
    vertices,values=[],[]
    for axis in range(3):
        other=[i for i in range(3) if i!=axis]
        for side in (0,1):
            offset=[0,0,0]; offset[axis]=2*side-1
            neighbour=padded[tuple(slice(1+offset[i],1+offset[i]+filled.shape[i]) for i in range(3))]
            boundary=filled & ~neighbour
            indices=np.argwhere(boundary)
            corners=np.zeros((4,3)); corners[:,axis]=side
            corners[:,other]=[[0,0],[1,0],[1,1],[0,1]]
            vertices.append((indices[:,None,:]+corners)*np.array([.125,.125,.0625]))
            values.append(displayed[boundary])
    faces=np.concatenate(vertices); vals=np.concatenate(values)
    if vals.size:
        ax.add_collection3d(Poly3DCollection(faces,facecolors=CMAP(norm(vals)),edgecolors=(.2,.23,.25,.15),linewidths=.10))
    ax.set(xlim=(0,4),ylim=(0,4),zlim=(1,0),xticks=[0,4],yticks=[0,4],zticks=[0,1])
    ax.set_proj_type('ortho'); ax.view_init(elev=22,azim=-55); ax.set_box_aspect((4,4,2))
    ax.tick_params(pad=-1,labelsize=6)
    ax.set_xlabel('x (km)',labelpad=-6,fontsize=6)
    ax.set_ylabel('y (km)',labelpad=-6,fontsize=6)
    if first:
        ax.text2D(-.03,.6,'Depth (km)',transform=ax.transAxes,rotation=90,rotation_mode='anchor',fontsize=6,va='center')
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False; axis.pane.set_edgecolor('#CBD0D3'); axis._axinfo['grid']['color']=(.87,.88,.89,1)


def section(ax,rho,kind,index,norm,truth=None):
    if kind=='xy':
        a=rho[index]; t=truth[index] if truth is not None else None
        extent=(0,4,0,4); origin='lower'; ys=CENTRE; aspect=1
        ax.set(xlabel='x (km)',ylabel='y (km)',xticks=[0,2,4],yticks=[0,2,4])
    else:
        a=(rho[:,index,:] if kind=='xz' else rho[:,:,index])[::-1]
        t=(truth[:,index,:] if kind=='xz' else truth[:,:,index])[::-1] if truth is not None else None
        extent=(0,4,1,0); origin='upper'; ys=DEPTH[::-1]; aspect=2
        ax.set(xlabel='x (km)' if kind=='xz' else 'y (km)',ylabel='Depth (km)',xticks=[0,2,4],yticks=[0,.5,1])
    ax.imshow(a,origin=origin,extent=extent,cmap=CMAP,norm=norm,interpolation='nearest',aspect=aspect)
    for b,color,style in ((a,'#252525','-'),(t,'#707070','--')):
        if b is not None and np.min(np.abs(b)) < .005 < np.max(np.abs(b)):
            ax.contour(CENTRE,ys,np.abs(b),levels=[.005],colors=color,linestyles=style,linewidths=.65)


def matched():
    data=np.load(MATCHED_SOURCE)
    labels=['Ordinary','Small-volume','Shape pair A']
    keys=['truth','L2','IRLS','Full_s1107']; titles=['Truth','Smooth L2','IRLS','TARF-GDC']
    # Bounds retain the original shared-per-case convention; no values are clipped.
    fig=plt.figure(figsize=(183/25.4,167/25.4))
    for r in range(3):
        vals=[data[k][r] for k in keys]
        bound=float(np.ceil(max(np.max(np.abs(a)) for a in vals)*20)/20)
        norm=Normalize(-bound,bound)
        bottom=.10+(2-r)*.295
        fig.text(.015,bottom+.11,labels[r],rotation=90,rotation_mode='anchor',ha='left',va='center',fontsize=7)
        for c,rho in enumerate(vals):
            ax=fig.add_axes([.065+c*.23,bottom,.215,.22],projection='3d')
            volume(ax,rho,norm,first=c==0)
            if r==0: ax.set_title(titles[c],pad=9)
            ax.text2D(.03,.91,f'{string.ascii_lowercase[r]}{c+1}',transform=ax.transAxes,fontsize=6)
        colorbar(fig,norm,(.37,bottom-.018,.31,.009))
    save(fig,10)
    bound=float(np.ceil(max(np.max(np.abs(data[k])) for k in keys)*20)/20)
    norm=Normalize(-bound,bound)
    fig=plt.figure(figsize=(183/25.4,226/25.4))
    gs=fig.add_gridspec(6,4,left=.09,right=.975,bottom=.13,top=.95,wspace=.30,hspace=.52)
    cuts=[]
    for r in range(3):
        truth=data['truth'][r]; weight=np.abs(truth)
        z=int(np.argmax(weight.sum(axis=(1,2))))
        y=int(np.rint(np.sum(weight*np.arange(32)[None,:,None])/weight.sum()))
        x=int(np.rint(np.sum(weight*np.arange(32)[None,None,:])/weight.sum()))
        cuts.append(dict(sample_id=data['sample_ids'][r],z=z,y=y,x=x,depth_km=DEPTH[z]))
        for s,(kind,index) in enumerate((('xy',z),('xz',y))):
            for c,key in enumerate(keys):
                ax=fig.add_subplot(gs[2*r+s,c]); section(ax,data[key][r],kind,index,norm,truth if c else None)
                ax.set_xlabel(''); ax.set_ylabel(''); ax.set_xticks([0,4]); ax.set_yticks([0,4] if kind=='xy' else [0,1])
                if r==0 and s==0: ax.set_title(titles[c],pad=8)
                if c==0: ax.set_ylabel(labels[r]+'\n'+('y (km)' if kind=='xy' else 'Depth (km)'),fontsize=6.5)
                if r==2 and s==1: ax.set_xlabel('x (km)')
    colorbar(fig,norm,(.3,.043,.44,.011))
    fig.legend(handles=[Line2D([],[],color='#252525',lw=.7,label='Predicted support'),Line2D([],[],color='#707070',ls='--',lw=.7,label='True support')],loc='lower center',bbox_to_anchor=(.5,.078),ncol=2)
    pd.DataFrame(cuts).to_csv(ROOT/'matched_cuts.csv',index=False)
    save(fig,11)


def vinton():
    data=np.load(SRC/'vinton_Full_s1107.npz')
    observed,predicted,rho=data['observed_d6_33'],data['predicted_d6_33'],data['density_zyx']
    bound=max(.05,float(np.ceil(np.abs(rho).max()*20)/20)); norm=Normalize(-bound,bound)
    z,y,x=map(int,np.unravel_index(np.argmax(rho),rho.shape))
    cuts=json.loads((SRC/'vinton_peak_cuts.json').read_text())['1107']
    assert [z,y,x] == [cuts[k] for k in ('peak_z','peak_y','peak_x')]
    fig=plt.figure(figsize=(183/25.4,147/25.4))
    ax=fig.add_axes([.055,.47,.5,.48],projection='3d'); volume(ax,rho,norm,True)
    ax.set_title('a  TARF-GDC',loc='left',fontweight='bold')
    ax=fig.add_axes([.68,.56,.25,.32]); section(ax,rho,'xy',z,norm)
    ax.set_title(f'b  Depth = {DEPTH[z]*1000:.2f} m',loc='left',fontweight='bold')
    ax.axvline(CENTRE[x],ls='--',lw=.7,color='#555'); ax.axhline(CENTRE[y],ls='--',lw=.7,color='#555')
    for j,(kind,index,title) in enumerate((('yz',x,f'x = {CENTRE[x]:.4f} km'),('xz',y,f'y = {CENTRE[y]:.4f} km'))):
        ax=fig.add_axes([.085+j*.49,.225,.39,.205]); section(ax,rho,kind,index,norm)
        ax.set_title(('c','d')[j]+'  '+title,loc='left',fontweight='bold')
        ax.axhline(DEPTH[z],ls='--',lw=.7,color='#555')
    colorbar(fig,norm,(.29,.08,.43,.015))
    save(fig,14)


if __name__=='__main__':
    matched(); vinton()
    pd.DataFrame(PANELS).to_csv(ROOT/'panel_audit.csv',index=False)
