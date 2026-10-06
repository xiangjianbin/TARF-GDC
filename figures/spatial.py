"""Python-only, evidence-linked figure rebuild for the frozen four-model panel."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from figure_runtime import OUTPUT, SOURCE_DATA, ASSETS
ROOT=ASSETS/'figure_data/statistics'
SRC=SOURCE_DATA
OUT=OUTPUT
CORE=['0001','1001','0101','1101']
LABEL={'0001':'Base','1001':'Base-TARF','0101':'Base-Correction','1101':'TARF-GDC','0001C':'Base-cont.','1001C':'TARF-cont.','T0P':'SC','B0P':'MC'}
SHORT={'0001':'B','1001':'T','0101':'R','1101':'F'}
COLOR={'0001':'#777777','1001':'#4C78A8','0101':'#D09A53','1101':'#5B8E7D','0001C':'#777777','1001C':'#4C78A8','T0P':'#A8A8A8','B0P':'#9B8FAD'}
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6.5,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})
CMAP=mpl.colormaps['RdBu_r']
QA=[]
WIDTH_IN=7.20472440945

def load(name):return pd.read_csv(SRC/name,dtype={'code':str,'initial':str,'final':str,'reference':str,'candidate':str})

def save(fig,name,claim,roles):
    fig.canvas.draw()
    panel_checks=[]
    renderer=fig.canvas.get_renderer()
    for i,ax in enumerate(fig.axes):
        panel_checks.append(dict(panel=i+1,axes_type=ax.name,xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),title=ax.get_title()))
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.png',dpi=600)
    QA.append(dict(figure=name,claim=claim,roles=roles,width_mm=fig.get_figwidth()*25.4,height_mm=fig.get_figheight()*25.4,panels=panel_checks))
    plt.close(fig);print('rendered',name,flush=True)

def label(ax,s):ax.set_title(s,loc='left',fontweight='bold',pad=7)

def dot_summary(ax,values,x,color,open_marker=False):
    vals=np.asarray(values)
    ax.plot(x+np.linspace(-.12,.12,len(vals)),vals,linestyle='none',marker='o',ms=3,mfc='white' if open_marker else color,mec=color,mew=.7,alpha=.8)
    ax.errorbar(x,vals.mean(),yerr=vals.std(ddof=1),fmt='_',ms=12,color=color,lw=1.1,capsize=3,zorder=5)

def training():
    df=load('training_history.csv');assert (df.train_loss>0).all()
    zero=load('epoch_zero_validation.csv')
    fig,axs=plt.subplots(2,2,figsize=(WIDTH_IN,124/25.4))
    fig.subplots_adjust(left=.095,right=.985,bottom=.13,top=.925,wspace=.31,hspace=.48)
    for col,stage in enumerate([1,2]):
        codes=['T0P','B0P','0001','1001'] if stage==1 else ['0101','1101','0001C','1001C']
        for row,metric in enumerate(['validation_sae','train_loss']):
            ax=axs[row,col]
            for code in codes:
                part=df[(df.code==code)&(df.stage==stage)]
                if stage==2 and row==0:
                    part=pd.concat([zero[zero.code==code],part],ignore_index=True).sort_values(['seed','epoch'])
                for seed,g in part.groupby('seed'):
                    ax.plot(g.epoch,g[metric],color=COLOR[code],lw=.45,alpha=.28,ls='--' if code.endswith('C') else '-')
                mean=part.groupby('epoch')[metric].mean()
                ax.plot(mean.index,mean.values,color=COLOR[code],lw=1.15,ls='--' if code.endswith('C') else '-',label=LABEL[code])
            ax.set_xlim(0 if stage==2 and row==0 else 1,50);ax.set_xlabel('Epoch within stage')
            ax.set_ylabel('Validation SAE (g/cm³)' if row==0 else 'Training loss')
            if row:
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:.0e}'))
                ax.yaxis.set_minor_formatter(FuncFormatter(lambda value,pos:f'{value:.0e}' if value in [.0002,.0003,.0004,.0006] else ''))
            label(ax,f'{chr(97+row*2+col)}  Stage {stage}')
        axs[0,col].legend(loc='upper right',ncol=2,columnspacing=.8,handlelength=1.5)
    axs[0,1].set_ylim(73,161)
    save(fig,'fig05_training','Separate actual training traces by stage; continuation and correction use the same second-stage epoch budget.',['Validation selection trace','Training objective trace'])

def ablation():
    df=load('overall_seed_metrics.csv');df=df[df['split']=='validation_iid']
    cp=load('paired_seed_contrasts.csv');cp=cp[(cp['split']=='validation_iid')&(cp.metric=='density_sae')]
    fig,axs=plt.subplots(1,3,figsize=(WIDTH_IN,75/25.4));fig.subplots_adjust(left=.075,right=.985,bottom=.22,top=.91,wspace=.40)
    for ax,metric,title in [(axs[0],'density_sae','a  Density error'),(axs[2],'physics_nrms','c  Response fit')]:
        for x,code in enumerate(CORE):dot_summary(ax,df[df.code==code][metric],x,COLOR[code])
        ax.set_xticks(range(4),['B','T','R','F']);ax.set_xlim(-.5,3.5)
        ax.set_ylabel('SAE (g/cm³)' if metric=='density_sae' else 'Physics NRMS')
        label(ax,title)
    for x,(a,b) in enumerate([('0001','1001'),('0001','0101'),('1001','1101'),('0101','1101')]):
        dot_summary(axs[1],cp[(cp.reference==a)&(cp.candidate==b)].gain,x,COLOR[b])
    axs[1].axhline(0,lw=.7,color='#555555',ls='--');axs[1].set_xticks(range(4),['B−T','B−R','T−F','R−F']);axs[1].set_ylabel('Paired SAE gain (g/cm³)');label(axs[1],'b  Paired gains')
    fig.legend(handles=[Line2D([],[],marker='o',ls='none',color=COLOR[c],label=f'{SHORT[c]}: {LABEL[c]}',ms=4) for c in CORE],loc='lower center',ncol=4,bbox_to_anchor=(.5,.015),columnspacing=1.2)
    save(fig,'fig06_ablation','All four SAE contrasts favour the added module, and Physics NRMS must be considered separately.',['Overall density error','Four paired effects with zero reference','Response fitting trade-off'])

def pair_recovery():
    df=load('pair_seed_metrics.csv');df=df[df['split']=='validation_iid']
    fig,axs=plt.subplots(2,3,figsize=(WIDTH_IN,120/25.4));fig.subplots_adjust(left=.09,right=.985,bottom=.15,top=.94,wspace=.33,hspace=.55)
    for r,track in enumerate(['separable','shape']):
      for c,(metric,ylabel) in enumerate([('difference_nrmse','Difference NRMSE'),('difference_cosine','Difference cosine'),('difference_magnitude_ratio','DMR')]):
        ax=axs[r,c]
        for x,code in enumerate(CORE):dot_summary(ax,df[(df.code==code)&(df.track==track)][metric],x,COLOR[code])
        ax.set_xticks(range(4),['B','T','R','F']);ax.set_ylabel(ylabel)
        if metric!='difference_cosine':ax.axhline(1,color='#555555',lw=.7,ls='--')
        else:ax.axhline(0,color='#555555',lw=.7,ls='--')
        label(ax,f'{chr(97+r*3+c)}  '+('Separable pairs' if r==0 else 'Shape-ambiguous pairs'))
    fig.legend(handles=[Line2D([],[],marker='o',ls='none',color=COLOR[c],label=f'{SHORT[c]}: {LABEL[c]}',ms=4) for c in CORE],loc='lower center',ncol=4,bbox_to_anchor=(.5,.012),columnspacing=1.2)
    save(fig,'fig07_pair_recovery','Correction improves pair recovery, whereas adding TARF increases difference error in both pair classes.',['Complete difference error','Direction alignment','Difference norm'])

def volume(ax,rho,norm,first=False):
    displayed=rho[::-1].transpose(2,1,0);filled=abs(displayed)>=.05;padded=np.pad(filled,1)
    vertices=[];values=[]
    for axis in range(3):
      other=[i for i in range(3) if i!=axis]
      for side in [0,1]:
        offset=[0,0,0];offset[axis]=2*side-1
        adjacent=padded[tuple(slice(1+offset[i],1+offset[i]+filled.shape[i]) for i in range(3))]
        exposed=filled&~adjacent;indices=np.argwhere(exposed)
        corners=np.zeros((4,3));corners[:,axis]=side;corners[:,other]=[[0,0],[1,0],[1,1],[0,1]]
        vertices.append((indices[:,None,:]+corners)*[.125,.125,.0625]);values.append(displayed[exposed])
    vals=np.concatenate(values)
    if len(vals):ax.add_collection3d(Poly3DCollection(np.concatenate(vertices),facecolors=CMAP(norm(vals)),edgecolors=(.2,.2,.2,.15),linewidths=.08))
    ax.set(xlim=(0,4),ylim=(0,4),zlim=(1,0),xticks=[0,4],yticks=[0,4],zticks=[0,1]);ax.set_proj_type('ortho');ax.view_init(elev=22,azim=-55);ax.set_box_aspect((4,4,2))
    ax.tick_params(pad=-2,labelsize=6);ax.set_xlabel('x (km)',labelpad=-8,fontsize=6);ax.set_ylabel('y (km)',labelpad=-8,fontsize=6)
    if first:ax.text2D(-.04,.6,'Depth (km)',transform=ax.transAxes,rotation=90,rotation_mode='anchor',fontsize=6,ha='center',va='center')
    for a in [ax.xaxis,ax.yaxis,ax.zaxis]:a.pane.fill=False;a.pane.set_edgecolor('#CBD0D3');a._axinfo['grid']['color']=(.86,.86,.86,1)

def spatial():
    d=np.load(SRC/'fixed_case_predictions.npz');old=np.load(ROOT.parent/'Source_data/six_group_final_predictions.npz');cuts=load('fixed_case_cuts.csv')
    labels=['Ordinary','Shape pair A','Shape pair B','Deep model'];cols=['truth']+['pred_'+c for c in CORE]
    fig=plt.figure(figsize=(WIDTH_IN,205/25.4));limits=[]
    for r in range(4):
        prior=max(float(abs(old[k][r]).max()) for k in old.files if k=='truth' or k.startswith('pred_'))
        current=max(float(abs(d[k][r]).max()) for k in cols);bound=np.ceil(max(prior,current)*20)/20;limits.append(bound);norm=Normalize(-bound,bound)
        bottom=.10+(3-r)*.215
        fig.text(.017,bottom+.06,labels[r],rotation=90,rotation_mode='anchor',fontsize=7,va='center')
        for c,key in enumerate(cols):
            ax=fig.add_axes([.07+c*.18,bottom,.176,.177],projection='3d');volume(ax,d[key][r],norm,first=c==0)
            if r==0:ax.set_title('Truth' if c==0 else LABEL[CORE[c-1]],pad=4,fontsize=6.5)
            ax.text2D(.02,.94,f'{chr(97+r)}{c+1}',transform=ax.transAxes,fontsize=6)
        ca=fig.add_axes([.365,bottom-.014,.29,.007]);bar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=CMAP),cax=ca,orientation='horizontal');bar.ax.tick_params(labelsize=6,pad=1,length=2);bar.set_label('Density contrast (g/cm³)',fontsize=6,labelpad=1)
    save(fig,'fig08_reconstruction','Fixed examples expose spatial errors without changing truth, sample selection or display thresholds.',['Signed three-dimensional density at four fixed examples'])
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
            mask=abs(value)>=.005
            if mask.any() and not mask.all():ax.contour(centres,centres if s==0 else depth,mask.astype(float),levels=[.5],colors=[color],linestyles=[ls],linewidths=.55)
        ax.set_xticks([0,4]);ax.set_yticks([0,4] if s==0 else [0,1]);ax.tick_params(pad=1,length=2,labelsize=6)
        if c==0:ax.set_ylabel(labels[r]+'\ny (km)' if s==0 else 'Depth (km)',fontsize=6)
        if r==0 and s==0:ax.set_title('Truth' if c==0 else LABEL[CORE[c-1]],fontsize=6.5,pad=6)
        if r==3 and s==1:ax.set_xlabel('x (km)',fontsize=6)
    ca=fig.add_axes([.31,.038,.44,.009]);bar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=CMAP),cax=ca,orientation='horizontal');bar.set_label('Density contrast (g/cm³)',fontsize=7);bar.ax.tick_params(labelsize=6)
    fig.legend(handles=[Line2D([],[],color='#252525',lw=.8,label='Prediction'),Line2D([],[],color='#707070',lw=.8,ls='--',label='Truth')],loc='lower center',bbox_to_anchor=(.5,.061),ncol=2)
    save(fig,'fig09_sections','Identical truth-defined slices show internal signed density and support errors.',['Fixed horizontal slices','Fixed vertical slices'])
    pd.DataFrame({'case':labels,'volume_symmetric_bound':limits,'section_symmetric_bound':bound,'voxel_display_threshold':.05,'section_contour_threshold':.005}).to_csv(SRC/'figure08_09_display_contract.csv',index=False)

def diagnostics():
    df=load('initial_final_diagnostics.csv');df=df[df['split']=='validation_iid']
    fig,axs=plt.subplots(1,3,figsize=(WIDTH_IN,78/25.4));fig.subplots_adjust(left=.08,right=.98,bottom=.25,top=.91,wspace=.36)
    for ax,metric,title in [(axs[0],'density_sae','a  Density error'),(axs[1],'physics_nrms','b  Response fit')]:
        for x,code in enumerate(['0101','1101']):dot_summary(ax,100*df[(df.final==code)&(df.metric==metric)].degraded_fraction,x,COLOR[code])
        ax.set_xticks([0,1],['Base → R','TARF → F']);ax.set_ylabel('Degraded models (%)');label(ax,title);ax.set_ylim(bottom=0)
    for x,(track,code) in enumerate([('separable','0101'),('separable','1101'),('shape','0101'),('shape','1101')]):
        dot_summary(axs[2],100*df[(df.final==code)&(df.track==track)&(df.metric=='difference_nrmse')].degraded_fraction,x,COLOR[code])
    axs[2].set_xticks(range(4),['Sep. R','Sep. F','Shape R','Shape F']);axs[2].set_ylabel('Degraded pairs (%)');axs[2].set_ylim(bottom=0);label(axs[2],'c  Difference error')
    save(fig,'figS2_initial_final','Mean correction gains do not imply monotonic improvement for every model or pair.',['Density degradation frequency','Physics degradation frequency','Difference-error degradation frequency'])

if __name__=='__main__':
    training();ablation();pair_recovery();spatial();diagnostics()
    (OUT/'statistics_figure_qa.json').write_text(json.dumps(QA,ensure_ascii=False,indent=2))
