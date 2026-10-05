# -*- coding: utf-8 -*-
import os,matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle,FancyBboxPatch,Polygon,Ellipse
_fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_fp);plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name();plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(13.2,14.6));ax.set_xlim(0,16.5);ax.set_ylim(0.2,17.4);ax.axis('off')
EC='#2b2b2b';DATA='#dadada';PROC='#f3f3f3';IO='#e7e7e7';OUT='#c4c4c4';VAL='#eef0f2';AC='#3a3a3a';LC='#555'
def txt(cx,cy,t,fs=14.4,c='#141414',b=False): ax.text(cx,cy,t,ha='center',va='center',fontsize=fs,color=c,linespacing=1.3,fontweight='bold' if b else 'normal')
def box(cx,cy,w,h,t,fc=PROC,lw=1.4,fs=14.4,b=False):
    ax.add_patch(Rectangle((cx-w/2,cy-h/2),w,h,fc=fc,ec=EC,lw=lw));txt(cx,cy,t,fs,b=b)
def obox(cx,cy,w,h,t,fc=OUT,lw=2.6,fs=15.4):
    ax.add_patch(FancyBboxPatch((cx-w/2+0.14,cy-h/2+0.14),w-0.28,h-0.28,boxstyle="round,pad=0.12,rounding_size=0.3",fc=fc,ec=EC,lw=lw));txt(cx,cy,t,fs,b=True)
def para(cx,cy,w,h,t,fc=IO,sk=0.33,lw=1.4,fs=14.4):
    ax.add_patch(Polygon([(cx-w/2+sk,cy+h/2),(cx+w/2+sk,cy+h/2),(cx+w/2-sk,cy-h/2),(cx-w/2-sk,cy-h/2)],closed=True,fc=fc,ec=EC,lw=lw));txt(cx,cy,t,fs)
def cyl(cx,cy,w,h,t,fc=DATA,lw=1.3,fs=13.0):
    e=h*0.17
    ax.add_patch(Rectangle((cx-w/2,cy-h/2+e),w,h-2*e,fc=fc,ec='none'))
    ax.plot([cx-w/2,cx-w/2],[cy-h/2+e,cy+h/2-e],color=EC,lw=lw);ax.plot([cx+w/2,cx+w/2],[cy-h/2+e,cy+h/2-e],color=EC,lw=lw)
    ax.add_patch(Ellipse((cx,cy-h/2+e),w,2*e,fc=fc,ec=EC,lw=lw));ax.add_patch(Ellipse((cx,cy+h/2-e),w,2*e,fc=fc,ec=EC,lw=lw));txt(cx,cy,t,fs)
def seg(pts,col=AC,lw=1.7,ls='-'): ax.plot([p[0] for p in pts],[p[1] for p in pts],color=col,lw=lw,ls=ls,solid_joinstyle='miter',solid_capstyle='butt')
def head(x,y,frm,col=AC,lw=1.7,ls='-'): ax.annotate("",xy=(x,y),xytext=frm,arrowprops=dict(arrowstyle='-|>',color=col,lw=lw,ls=ls,mutation_scale=15,shrinkA=0,shrinkB=1))
def ilbl(x,y,t,fs=12.2): ax.text(x,y,t,ha='center',va='center',fontsize=fs,color=LC,linespacing=1.24,bbox=dict(boxstyle='round,pad=0.28',fc='white',ec='#d2d2d2',lw=0.7))
def dvar(x,yb,yt,t): head(x,yt,(x,yb));ilbl(x,(yb+yt)/2,t)
def stage(y,t): ax.text(0.5,y,t,ha='center',va='center',fontsize=13.4,color='#9a9a9a',rotation=90,fontweight='bold')
ax.plot([1.05,1.05],[0.5,17.0],color='#dcdcdc',lw=1.0)
X=6.2
# ===== 자료 =====
for x,t in [(2.6,"Forest-type\n(species)"),(4.8,"Land\ncover"),(7.0,"DEM"),(9.2,"Admin.\nboundary")]: cyl(x,16.6,2.2,1.25,t)
cyl(14.2,16.6,2.4,1.25,"Independent validation\n2025 · 2022 Uljin")
for x in [2.6,4.8,7.0,9.2]: seg([(x,15.97),(x,15.55)])
seg([(2.6,15.55),(9.2,15.55)])
head(X,14.75,(X,15.55));ilbl(X,15.1,"Preprocessing: flammability from pre-fire (2019) forest-type map · DEM reprojection · clip to 3-province forest")
# ===== spine =====
para(X,14.05,6.2,1.35,"Input layers (30 m)\nspecies flammability · elevation · slope · northness")
box(X,11.95,5.2,1.35,"Selected variables\nspecies flammability + terrain")
box(X,9.95,5.2,1.35,"Analysis unit\n600 m blocks · 70,308")
box(X,7.75,5.2,1.35,"Single dendrogram")
box(X,5.8,5.2,1.35,"Micro-region candidates")
obox(X,3.5,6.3,1.55,"Fire-continuity regions\nL1 5  ⊃  L2 18  ⊃  L3 55")
dvar(X,13.37,12.63,"Contribution test (logistic AUC · ΔAUC)\nsoil · stand density · DBH · age class excluded")
dvar(X,11.27,10.63,"600 m block aggregation (≥25 px) · standardization")
dvar(X,9.27,8.43,"rook adjacency (connectivity)\nWard merge + W_SPACE · coordinate concat")
dvar(X,7.07,6.48,"Dendrogram cut")
dvar(X,5.12,4.28,"Boundary refinement: exclusivity · contiguity · 400 ha absorption\narea gates (L2 · L3) · hierarchical nesting")
# feedback loop
seg([(8.85,4.9),(9.75,4.9),(9.75,5.8)],col=LC,lw=1.4,ls=(0,(4,2)))
head(8.85,5.8,(9.75,5.8),col=LC,lw=1.4,ls=(0,(4,2)))
ax.text(10.75,5.35,"Iterate to\nconvergence",ha='center',va='center',fontsize=11.0,color=LC)
# ===== 산출 -> 검증·확인 =====
seg([(X,2.72),(X,2.15)]);ilbl(X,2.46,"Assign Pfafstetter codes")
seg([(2.5,2.15),(12.2,2.15)])
for x,t in [(2.5,"Discrimination\nin-sample AUC\n→ spatial-block CV"),(5.73,"Concentration\nvs geometry\nbaseline"),(8.96,"Independent transfer\n2022 Uljin\n(non-circular)"),(12.2,"Robustness\nSKATER · stability\n· MAUP")]:
    head(x,1.62,(x,2.15));box(x,0.98,3.05,1.42,t,fc=VAL,fs=12.8)
# 독립 검증자료 -> 검증행
seg([(14.2,15.97),(14.2,0.98),(13.75,0.98)],col='#888',lw=1.4,ls=(0,(3,2)))
head(13.75,0.98,(14.2,0.98),col='#888',lw=1.4,ls=(0,(3,2)))
ax.text(14.75,8.5,"Independent validation data",fontsize=12.0,color='#888',rotation=90,ha='center')
# ===== stage labels =====
for y,t in [(16.6,"Data"),(14.05,"Input"),(11.95,"Variable\nselection"),(9.95,"Aggregation"),(7.75,"Clustering"),(5.8,"Refinement"),(3.5,"Output"),(0.98,"Validation")]: stage(y,t)
plt.tight_layout();plt.savefig(os.path.expanduser("~/p3b/fig_flow_en.png"),dpi=300,facecolor='white',bbox_inches='tight');print("flow done")
