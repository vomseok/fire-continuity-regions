# -*- coding: utf-8 -*-
import os,numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg");import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_fp);plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name();plt.rcParams['axes.unicode_minus']=False;plt.rcParams['hatch.linewidth']=1.1
from matplotlib.colors import hsv_to_rgb,ListedColormap
from matplotlib.patches import PathPatch,Rectangle
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from rasterio.features import shapes
from affine import Affine
from shapely.geometry import shape as shp,Polygon,box
from shapely.ops import unary_union
import geopandas as gpd
TMP=os.path.expanduser("~/p3b");DATA=os.path.expanduser("~/mnt/Data/Source/1_boundary")
E0,N0,R=221910.0,323280.0,30.0
d=np.load(f"{TMP}/hier4_2019.npz");bid=d['bid'];wb=int(d['wb']);hb=int(d['hb']);B=int(d['B']);H=int(d['H']);W=int(d['W'])
N1=N0+H*R
l1,l2,l3=d['l1'].copy(),d['l2'].copy(),d['l3'].copy();AH=(B*30.0)**2/1e4
rr=bid//wb;cc=bid%wb;present=np.zeros((hb,wb),'uint8');present[rr,cc]=1
T2,T3,MMU=25000.,6000.,500.
def area_of(l):u=np.unique(l);return {v:(l==v).sum()*AH for v in u}
a1=area_of(l1);nl2=l2.copy();nx=l2.max()+1
for v in np.unique(l1):
    if a1[v]<T2: nl2[l1==v]=nx;nx+=1
a2=area_of(nl2);nl3=l3.copy();nx3=max(l3.max(),nl2.max())+1
for v in np.unique(nl2):
    if a2[v]<T3: nl3[nl2==v]=nx3;nx3+=1
l2,l3=nl2,nl3
df=pd.DataFrame({'l1':l1,'l2':l2,'l3':l3})
s1=df.groupby('l1').size().sort_values(ascending=False)
c1={v:str(int(v)) for v in s1.index};c2={};c3={}
L2ofL1={};L3ofL2={}
for a in s1.index:
    sa=df[df.l1==a];l2s=list(sa.groupby('l2').size().sort_values(ascending=False).index);L2ofL1[a]=l2s;s2=len(l2s)>1
    for j,b in enumerate(l2s):
        c2[b]=f"{c1[a]}.{j+1}" if s2 else c1[a]
        sb=sa[sa.l2==b];l3s=list(sb.groupby('l3').size().sort_values(ascending=False).index);L3ofL2[b]=l3s;s3=len(l3s)>1
        for k,e in enumerate(l3s): c3[e]=(f"{c2[b]}.{k+1}" if s3 else c2[b])
# ---------- encoding ----------
# L2 -> hue (distinct, spread); order by code
l2order=sorted(np.unique(l2),key=lambda v:[int(t) for t in c2[v].split('.')])
nh=len(l2order)
# spread hues with a shuffle so adjacent codes differ
hues=np.linspace(0,1,nh,endpoint=False)
perm=(np.arange(nh)*7)%nh   # decorrelate
hue_of={l2order[i]:float(hues[perm[i]]) for i in range(nh)}
# L3 -> saturation within its L2
sat_of={}
for b,l3s in L3ofL2.items():
    m=len(l3s)
    for k,e in enumerate(l3s):
        sat_of[e]= 0.20+0.42*(k/(m-1)) if m>1 else 0.40
# L1 -> hatch pattern
pats=['/','.','x','+','\\']
pat_of={v:pats[i%len(pats)] for i,v in enumerate(s1.index)}
l1tint={v:hsv_to_rgb([ (i*0.13)%1.0,0.18,0.97]) for i,v in enumerate(s1.index)}
# ---------- rasters ----------
def braster(l):a=np.zeros((hb,wb),'int32');a[rr,cc]=l;return a
bl1,bl2,bl3=braster(l1),braster(l2),braster(l3)
def rgb_from(level_hue,level_sat,valid):
    img=np.ones((hb,wb,4));img[...,3]=0.0
    ys,xs=np.where(valid)
    for y,x in zip(ys,xs): pass
    return img
# build HSV rgb images vectorized
def hsv_img(hue_arr,sat_arr,mask,val=0.95):
    img=np.zeros((hb,wb,4))
    hsv=np.zeros((hb,wb,3));hsv[...,0]=hue_arr;hsv[...,1]=sat_arr;hsv[...,2]=val
    rgb=hsv_to_rgb(hsv);img[...,:3]=rgb;img[...,3]=np.where(mask,1.0,0.0)
    return img
hueA=np.zeros((hb,wb));satA=np.zeros((hb,wb))
mask=present>0
# B: L2 hue, fixed sat
hB=np.zeros((hb,wb));sB=np.full((hb,wb),0.40)
for v in np.unique(l2): hB[bl2==v]=hue_of[v]
imgB=hsv_img(hB,sB,mask,val=0.80)
# C: L2 hue x L3 saturation
hC=np.zeros((hb,wb));sC=np.zeros((hb,wb))
# map each l3 to its parent l2 hue
l3_parent={}
for b,l3s in L3ofL2.items():
    for e in l3s: l3_parent[e]=b
for v in np.unique(l3):
    hC[bl3==v]=hue_of[l3_parent[v]];sC[bl3==v]=sat_of[v]
imgC=hsv_img(hC,sC,mask,val=0.84)
# ---------- polys ----------
def polys(l):
    a=braster(l);out={}
    for g,v in shapes(a,mask=a>0,transform=Affine.identity(),connectivity=4): out.setdefault(int(v),[]).append(shp(g))
    return {v:unary_union(x).buffer(0).simplify(0.8) for v,x in out.items()}
P1,P2,P3=polys(l1),polys(l2),polys(l3)
def biggest(g):return max((g.geoms if g.geom_type=='MultiPolygon' else [g]),key=lambda p:p.area)
def bounds_draw(ax,P,ls,lw,col,inter=False):
    for v,g in P.items():
        for p in (g.geoms if g.geom_type=='MultiPolygon' else [g]):
            if p.area*(B*B)*0.09<MMU: continue
            xs,ys=p.exterior.xy;ax.plot(np.array(xs)*B,np.array(ys)*B,ls=ls,color=col,lw=lw,solid_capstyle='round')
            if inter:
                for it in p.interiors:
                    if Polygon(it).area*(B*B)*0.09<MMU: continue
                    xi,yi=it.xy;ax.plot(np.array(xi)*B,np.array(yi)*B,ls=ls,color=col,lw=lw*0.8)
def hatch_patches(ax,P,hatch,ec='#333',lw=0.0,fc='none'):
    for v,g in P.items():
        for p in (g.geoms if g.geom_type=='MultiPolygon' else [g]):
            if p.area*(B*B)*0.09<MMU: continue
            verts=[];codes=[]
            ring=np.array(p.exterior.xy).T*B
            verts+=list(ring);codes+=[MPath.MOVETO]+[MPath.LINETO]*(len(ring)-2)+[MPath.CLOSEPOLY]
            for it in p.interiors:
                r2=np.array(it.xy).T*B
                if len(r2)<4: continue
                verts+=list(r2);codes+=[MPath.MOVETO]+[MPath.LINETO]*(len(r2)-2)+[MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(verts,codes),facecolor=fc,edgecolor=ec,hatch=hatch,lw=lw))
def labels(ax,l,P,code,amin,fs,fc,ec):
    for v in np.unique(l):
        if (l==v).sum()*AH<amin or v not in P or not code.get(v): continue
        g=biggest(P[v]); c=g.centroid
        pt=c if g.contains(c) else g.representative_point()
        ax.text(pt.x*B,pt.y*B,code[v],ha='center',va='center',fontsize=fs,fontweight='bold',color=fc,
                bbox=dict(boxstyle='round,pad=0.2',fc='white',ec=ec,lw=1.0,alpha=0.94))
# ---------- admin boundaries (EPSG:5181 -> 5186) clipped ----------
def to_px_coords(geom):
    # returns list of (col,row) arrays for plotting
    out=[]
    for p in (geom.geoms if geom.geom_type in ('MultiPolygon','GeometryCollection') else [geom]):
        if p.is_empty or p.geom_type!='Polygon': continue
        x,y=p.exterior.xy
        out.append(((np.array(x)-E0)/R,(N1-np.array(y))/R))
    return out
gbox=box(E0,N0,E0+W*R,N1)
sig=gpd.read_file(f"{DATA}/(센서스경계)시군구경계/Z_SOP_BND_SIGUNGU_PG/Z_SOP_BND_SIGUNGU_PG.shp")
sig=sig.set_crs(5181,allow_override=True).to_crs(5186)
sig=sig[sig.intersects(gbox)]
def draw_admin(ax,lw=0.6,col='#6b6b6b',alpha=0.55):
    for geom in sig.geometry:
        for cx,cy in to_px_coords(geom.intersection(gbox)):
            ax.plot(cx,cy,color=col,lw=lw,alpha=alpha)
gray=ListedColormap([(1,1,1,0),(0.9,0.9,0.9,1)])
def base(ax):
    ax.imshow(present,cmap=gray,interpolation='nearest',extent=(0,W,H,0),vmin=0,vmax=1,aspect='equal')
    ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off')
# ---------- FIGURE (1x3, no keymap) ----------
fig,axs=plt.subplots(1,3,figsize=(21,9.4));fig.patch.set_facecolor('white')
ax=axs[0];base(ax)
for v in s1.index: hatch_patches(ax,{v:P1[v]},pat_of[v],ec='#222',lw=0.0)
bounds_draw(ax,P1,'-',3.0,'black',True);draw_admin(ax);labels(ax,l1,P1,c1,MMU,24,'#b00020','#b00020')
ax.set_title("(a) L1 macro-regions — pattern",fontsize=18)
ax=axs[1];base(ax);ax.imshow(imgB,extent=(0,W,H,0),aspect='equal')
for v in s1.index: hatch_patches(ax,{v:P1[v]},pat_of[v],ec='#222',lw=0.0)
bounds_draw(ax,P2,'-',1.2,'#333');bounds_draw(ax,P1,'-',2.6,'black',True);draw_admin(ax);labels(ax,l2,P2,c2,MMU,22,'#10263b','#10263b')
ax.set_title("(b) L2 meso-regions — hue + L1 pattern",fontsize=18)
ax=axs[2];base(ax);ax.imshow(imgC,extent=(0,W,H,0),aspect='equal')
for v in s1.index: hatch_patches(ax,{v:P1[v]},pat_of[v],ec='#222',lw=0.0)
bounds_draw(ax,P3,'-',0.5,'#555');bounds_draw(ax,P2,'-',1.2,'#222');bounds_draw(ax,P1,'-',2.6,'black',True);draw_admin(ax,lw=0.5,alpha=0.45);labels(ax,l3,P3,c3,15000,17,'#222','#888')
ax.set_title("(c) L3 micro-regions — saturation · L1 pattern + L2 hue",fontsize=18)
from matplotlib.patches import Patch
legh=[Patch(facecolor='white',edgecolor='#555',hatch=pat_of[v],label=f"L1 {c1[v]}") for v in s1.index]
axs[0].legend(handles=legh,loc='lower left',fontsize=12,framealpha=0.9,title='L1 pattern')
fig.suptitle("Fire-continuity regions — hierarchical encoding (L1 pattern · L2 hue · L3 saturation) + admin. boundaries",fontsize=18,y=1.06)
plt.tight_layout(rect=[0,0,1,0.95]);plt.savefig(f"{TMP}/hier_encoded4_en.png",bbox_inches='tight',dpi=220,facecolor='white');plt.close()
print("saved hier_encoded 1x3  L1",len(set(l1)),"L2",len(set(l2)),"L3",len(set(l3)))
