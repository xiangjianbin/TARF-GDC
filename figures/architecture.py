"""Audited TARF-GDC inference and training diagram; all dimensions in figure units."""
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from figure_runtime import OUTPUT, SOURCE_DATA, ASSETS
HERE=OUTPUT
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':7.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7,
    'savefig.facecolor':'white'})
COLORS={'initial':'#E5EEF5','correction':'#F6E6D7','physics':'#EFEFEF','white':'#FFFFFF'}
fig=plt.figure(figsize=(7.2047244,4.9606299))  # 183 x 126 mm
a=fig.add_axes([.015,.485,.97,.50])
b=fig.add_axes([.015,.055,.46,.385])
c=fig.add_axes([.525,.055,.46,.385])
texts=[]


def setup(ax,label,title):
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis('off')
    texts.append(ax.text(0,1,label,fontsize=9,fontweight='bold',va='top'))
    texts.append(ax.text(.055,1,title,fontsize=8.5,va='top'))


def box(ax,x,y,w,h,label,kind='initial',fontsize=7.5):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.006,rounding_size=0.014',
        linewidth=.75,edgecolor='#53616A',facecolor=COLORS[kind],zorder=3))
    texts.append(ax.text(x,y,label,ha='center',va='center',fontsize=fontsize,zorder=4))


def arrow(ax,start,end,dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=8,
        linewidth=.85,color='#59636A',linestyle='--' if dashed else '-',zorder=2))


def line(ax,points,dashed=False):
    for start,end in zip(points[:-2],points[1:-1]):
        ax.plot([start[0],end[0]],[start[1],end[1]],color='#59636A',lw=.85,
                linestyle='--' if dashed else '-',zorder=2)
    arrow(ax,points[-2],points[-1],dashed)


setup(a,'a','TARF-GDC: one-step density correction')
box(a,.065,.77,.115,.17,'Observed\nroles','physics')
box(a,.215,.77,.12,.17,'TARF')
box(a,.390,.77,.16,.17,'2D–3D\nbackbone')
box(a,.565,.77,.12,.17,'GLIN')
box(a,.725,.77,.13,.17,'Initial\n'+r'$\rho_0,\ p_0$',fontsize=8)
box(a,.920,.77,.13,.17,'Final\n'+r'$\hat\rho$',kind='correction',fontsize=8)
for st,en in [((.124,.77),(.153,.77)),((.277,.77),(.308,.77)),
              ((.472,.77),(.503,.77)),((.627,.77),(.658,.77)),((.793,.77),(.852,.77))]:
    arrow(a,st,en)
box(a,.725,.49,.13,.135,'Forward +\nrole map','physics',7.5)
arrow(a,(.725,.68),(.725,.562))
box(a,.215,.49,.15,.135,'Residual','physics')
line(a,[(.065,.68),(.065,.49),(.134,.49)])
arrow(a,(.658,.49),(.294,.49))
texts.append(a.text(.465,.515,'Observed − predicted',ha='center',va='bottom',fontsize=7))
box(a,.215,.18,.18,.16,'Weighted\nbackprojection','physics')
box(a,.435,.18,.17,.16,'3D correction\nnetwork','correction')
box(a,.680,.18,.20,.16,r'$p_0\odot0.25\tanh(q_\phi)$','correction',8)
arrow(a,(.215,.414),(.215,.266))
arrow(a,(.31,.18),(.344,.18))
arrow(a,(.526,.18),(.574,.18))
line(a,[(.786,.18),(.92,.18),(.92,.68)])
texts.append(a.text(.935,.415,'+',ha='left',va='center',fontsize=11))
line(a,[(.783,.71),(.819,.71),(.819,.32),(.68,.32),(.68,.266)])
texts.append(a.text(.803,.56,r'$p_0$',fontsize=8,ha='left'))
arrow(a,(.435,.35),(.435,.266))
texts.append(a.text(.435,.355,r'$\rho_0$',fontsize=8,ha='center',va='bottom'))
texts.append(a.text(.823,.80,r'$\rho_0$',fontsize=8,ha='center',va='bottom'))

setup(b,'b','Replacement-style TARF')
box(b,.11,.77,.15,.16,r'$\mathbf{h}_0$',fontsize=8)
box(b,.39,.77,.25,.16,'Scalar anchor')
arrow(b,(.192,.77),(.258,.77))
box(b,.12,.43,.18,.16,r'$\mathbf{h}_1,\mathbf{h}_2$',fontsize=8)
box(b,.45,.43,.31,.18,'Two gated\nresiduals')
arrow(b,(.216,.43),(.288,.43))
arrow(b,(.39,.68),(.39,.527))
box(b,.81,.60,.25,.24,'TARF\nfusion')
line(b,[(.519,.77),(.81,.77),(.81,.725)])
line(b,[(.611,.43),(.81,.43),(.81,.474)])
texts.append(b.text(.49,.15,r'$\mathbf{f}_0=\mathbf{a}+0.25\sum_r g_r\odot\mathbf{u}_r$',
    fontsize=8,ha='center',va='center'))

setup(c,'c','Two-stage supervision')
box(c,.27,.75,.47,.19,'Stage 1: initial network\n50 epochs')
box(c,.77,.75,.35,.19,'Density +\nclean response','physics')
arrow(c,(.586,.75),(.512,.75),True)
box(c,.27,.36,.47,.22,'Stage 2: correction only\n50 epochs','correction')
box(c,.77,.36,.35,.19,'Density +\nclean response','physics')
arrow(c,(.586,.36),(.512,.36),True)
arrow(c,(.27,.65),(.27,.481))
texts.append(c.text(.31,.555,'Freeze initial network',fontsize=7,va='center'))
texts.append(c.text(.50,.10,'Inference: fixed weights; no labels',fontsize=7,ha='center'))

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for text in texts:
    bbox=text.get_window_extent(renderer)
    assert bbox.width > 0 and bbox.height > 0
fig.savefig(HERE/'Figure04.svg')
fig.savefig(HERE/'Figure04.pdf')
fig.savefig(HERE/'Figure04.png',dpi=600)
plt.close(fig)
