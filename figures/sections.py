"""Revision of the established section plot: common 0.05 contours, unchanged density images.
Contract: compare main-body position and amplitude across four fixed examples.
Archetype: image plate; five matched columns, horizontal/vertical paired rows.
No sample replacement, smoothing, masking, component removal or metric alteration.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from figure_runtime import OUTPUT, SOURCE_DATA, ASSETS
OUT=OUTPUT
PACKAGE=ASSETS/"figure_data"
SRC=SOURCE_DATA
ROOT=PACKAGE
CORE=['0001','1001','0101','1101']
LABEL={'0001':'Base','1001':'Base-TARF','0101':'Base-Correction','1101':'TARF-GDC'}
CONTOUR_THRESHOLD=.05
WIDTH_IN=183/25.4
CMAP=mpl.colormaps["RdBu_r"]
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6.5,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})
QA=[]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def save(fig,name,claim,roles):
    fig.canvas.draw()
    panels=[]
    for i,ax in enumerate(fig.axes):
        if ax.images:
            arr=np.asarray(ax.images[0].get_array())
            panels.append(dict(panel=i+1,all_values_shown=True,pixels=int(arr.size),
                               min=float(arr.min()),max=float(arr.max()),interpolation=ax.images[0].get_interpolation(),
                               xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),uncertainty="none: fixed seed 1107, fixed r0, no aggregation"))
    assert len(panels)==40
    fig.savefig(OUT/(name+".png"),dpi=600)
    fig.savefig(OUT/(name+".pdf"))
    fig.savefig(OUT/(name+".svg"))
    QA.append(dict(figure=name,claim=claim,roles=roles,width_mm=183,height_mm=245,
                   original_contour_threshold=.005,new_contour_threshold=CONTOUR_THRESHOLD,
                   evaluation_threshold_unchanged=.005,
                   array_modifications="none",sample_replacement=False,smoothing=False,
                   connected_component_filter=False,panel_audit=panels))
    plt.close(fig)

def main():
    sources=[SRC/"fixed_case_predictions.npz",SRC/"fixed_case_cuts.csv",PACKAGE/"Source_data/six_group_final_predictions.npz"]
    hashes={str(p):sha(p) for p in sources}
    d=np.load(sources[0]);old=np.load(sources[2]);cuts=pd.read_csv(sources[1])
    labels=['Ordinary','Shape pair A','Shape pair B','Deep model'];cols=['truth']+['pred_'+c for c in CORE]
    bound=np.ceil(max(float(abs(old[k]).max()) for k in old.files if k=='truth' or k.startswith('pred_'))*20)/20
    bound=max(bound,np.ceil(max(float(abs(d[k]).max()) for k in cols)*20)/20);norm=Normalize(-bound,bound)
    fig=plt.figure(figsize=(WIDTH_IN,245/25.4));gs=fig.add_gridspec(8,5,left=.105,right=.98,bottom=.13,top=.95,hspace=.58,wspace=.20,height_ratios=[1,.54]*4)
    centres=(np.arange(32)+.5)*.125;depth=(np.arange(16)+.5)*.0625
    for r in range(4):
      z=int(cuts.iloc[r].z_index);y=int(cuts.iloc[r].y_index)
      for s in [0,1]:
       truth=d['truth'][r,z] if s==0 else d['truth'][r,::-1,y,:]
       for c,key in enumerate(cols):
        ax=fig.add_subplot(gs[2*r+s,c]);a=d[key][r,z] if s==0 else d[key][r,::-1,y,:]
        ax.imshow(a,origin='lower' if s==0 else 'upper',extent=(0,4,0,4) if s==0 else (0,4,1,0),norm=norm,cmap=CMAP,interpolation='nearest',aspect='equal' if s==0 else 2)
        for value,color,ls in [(a,'#252525','-')]+([(truth,'#707070','--')] if c else []):
            mask=abs(value)>=CONTOUR_THRESHOLD
            if mask.any() and not mask.all():ax.contour(centres,centres if s==0 else depth,mask.astype(float),levels=[.5],colors=[color],linestyles=[ls],linewidths=.55)
        ax.set_xticks([0,4]);ax.set_yticks([0,4] if s==0 else [0,1]);ax.tick_params(pad=1,length=2,labelsize=6)
        if c==0:ax.set_ylabel(labels[r]+'\ny (km)' if s==0 else 'Depth (km)',fontsize=6)
        if r==0 and s==0:ax.set_title('Truth' if c==0 else LABEL[CORE[c-1]],fontsize=6.5,pad=6)
        if r==3 and s==1:ax.set_xlabel('x (km)',fontsize=6)
    ca=fig.add_axes([.31,.047,.44,.009]);bar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=CMAP),cax=ca,orientation='horizontal');bar.set_label('Density contrast (g/cm³)',fontsize=7);bar.ax.tick_params(labelsize=6)
    fig.legend(handles=[Line2D([],[],color='#252525',lw=.8,label='Prediction'),Line2D([],[],color='#707070',lw=.8,ls='--',label='Truth')],loc='lower center',bbox_to_anchor=(.5,.074),ncol=2)
    fig.text(.5,.009,'Contours: |density contrast| = 0.05 g/cm³; colour images retain all values.',ha='center',fontsize=6)
    save(fig,'Figure09_sections_major_contours_20261002','Identical truth-defined slices show internal signed density and support errors.',['Fixed horizontal slices','Fixed vertical slices'])
    assert all(sha(p)==h for p,h in hashes.items())
    record=dict(status="PASS",source_hashes_unchanged=True,source_sha256=hashes,figures=QA)
    (OUT/"Figure09_rendering_audit.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")

if __name__=="__main__":main()
