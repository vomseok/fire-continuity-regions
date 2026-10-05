import os,numpy as np,rasterio,matplotlib
matplotlib.use("Agg");import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
_fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_fp);plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name();plt.rcParams['axes.unicode_minus']=False
TMP=os.path.expanduser("~/p3b")
F={'FVI':f"{TMP}/spec3_2019.tif",'elev':f"{TMP}/exp3_elev.tif",'slope':f"{TMP}/exp3_slope.tif",'north':f"{TMP}/exp3_north.tif"}
ds={k:rasterio.open(p) for k,p in F.items()};dfor=rasterio.open(f"{TMP}/forest3p.tif")
H=dfor.height;W=dfor.width;B=20;hb=(H+B-1)//B;wb=(W+B-1)//B
names=list(F);ssum=np.zeros((hb*wb,4));scnt=np.zeros(hb*wb)
STRIP=900
for r0 in range(0,H,STRIP):
    r1=min(r0+STRIP,H);win=rasterio.windows.Window(0,r0,W,r1-r0)
    fo=dfor.read(1,window=win)>0
    if not fo.any():continue
    feats=[];ok=fo.copy()
    for k in names:
        a=ds[k].read(1,window=win).astype('float32')
        if k=='FVI': a=np.where(a<0,np.nan,a)
        feats.append(a);ok&=np.isfinite(a)
    yy,xx=np.where(ok)
    if len(yy)==0:continue
    gr=(r0+yy)//B;gc=xx//B;bf=gr*wb+gc
    vals=np.column_stack([feats[j][yy,xx] for j in range(4)]).astype('float64')
    for j in range(4): np.add.at(ssum[:,j],bf,vals[:,j])
    np.add.at(scnt,bf,1.0)
sel=scnt>=25;bid=np.where(sel)[0];bf=(ssum[sel]/scnt[sel,None])
d=np.load(f"{TMP}/hier4_2019.npz");bid2=d['bid'];l2=d['l2']
assert len(bid)==len(bid2) and (bid==bid2).all()
rr=bid//wb;cc=bid%wb
km=KMeans(n_clusters=18,random_state=0,n_init=10).fit_predict(StandardScaler().fit_transform(bf))
rng=np.random.default_rng(3)
def recolor(lab):
    u=np.unique(lab);perm=rng.permutation(len(u));m={v:perm[i]%20 for i,v in enumerate(u)};return np.array([m[v] for v in lab])
def raster(lab):
    a=np.full((hb,wb),np.nan);a[rr,cc]=recolor(lab);return a
cmap=plt.get_cmap('tab20')
r0b,r1b,c0b,c1b=int(rr.min()),int(rr.max()),int(cc.min()),int(cc.max());pad=4
fig,axs=plt.subplots(1,2,figsize=(14,6.2))
for ax,lab,t in [(axs[0],km,"(a) Non-spatial clustering (k-means, k=18): fragmented"),(axs[1],l2,"(b) Spatially constrained clustering (L2, k=18): contiguous, non-overlapping")]:
    ax.imshow(raster(lab),cmap=cmap,vmin=0,vmax=19,interpolation='nearest',aspect='equal');ax.set_xlim(c0b-pad,c1b+pad);ax.set_ylim(r1b+pad,r0b-pad);ax.axis('off');ax.set_anchor('N');ax.set_title(t,fontsize=15,pad=6)
fig.suptitle("Same variables, same k (=18): non-spatial vs spatially constrained (species + terrain)",fontsize=16,y=0.99)
plt.tight_layout();plt.savefig(f"{TMP}/fig_compare4_en.png",dpi=220,facecolor='white',bbox_inches='tight');print("compare4 saved",len(bid),"blocks")
