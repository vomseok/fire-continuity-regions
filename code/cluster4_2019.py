# -*- coding: utf-8 -*-
import os,numpy as np,rasterio,pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
from scipy import sparse
TMP=os.path.expanduser("~/p3b")
F={'FVI':f"{TMP}/spec3_2019.tif",'elev':f"{TMP}/exp3_elev.tif",'slope':f"{TMP}/exp3_slope.tif",'north':f"{TMP}/exp3_north.tif"}
FOR=f"{TMP}/forest3p.tif"
ds={k:rasterio.open(p) for k,p in F.items()}; dfor=rasterio.open(FOR)
H=dfor.height; W=dfor.width; B=20
hb=(H+B-1)//B; wb=(W+B-1)//B; nblk=hb*wb
names=list(F)
ssum=np.zeros((nblk,len(names))); scnt=np.zeros(nblk)
gsum=np.zeros(len(names)); gsq=np.zeros(len(names)); gn=0.0
STRIP=600
for r0 in range(0,H,STRIP):
    r1=min(r0+STRIP,H); win=rasterio.windows.Window(0,r0,W,r1-r0)
    fo=dfor.read(1,window=win)>0
    if not fo.any(): continue
    feats=[]
    ok=fo.copy()
    for k in names:
        a=ds[k].read(1,window=win).astype('float32')
        if k=='FVI': a=np.where(a<0,np.nan,a)
        feats.append(a); ok&=np.isfinite(a)
    yy,xx=np.where(ok)
    if len(yy)==0: continue
    gr=(r0+yy)//B; gc=xx//B; bflat=gr*wb+gc
    vals=np.column_stack([feats[j][yy,xx] for j in range(len(names))]).astype('float64')
    for j in range(len(names)):
        np.add.at(ssum[:,j],bflat,vals[:,j])
        gsum[j]+=vals[:,j].sum(); gsq[j]+=(vals[:,j]**2).sum()
    np.add.at(scnt,bflat,1.0); gn+=len(yy)
pm=gsum/gn; pstd=np.sqrt(gsq/gn-pm**2)
sel=scnt>=25; bid=np.where(sel)[0]; cnt=scnt[sel]
bf=(ssum[sel]/cnt[:,None]-pm)/pstd
nb=len(bid); print("blocks",nb,"forest ha",int(scnt.sum()*0.09))
# compactness coords (standardized, weighted)
W_SPACE=float(os.environ.get("WSP","1.3"))
by=(bid//wb).astype(float); bx=(bid%wb).astype(float)
cy=(by-by.mean())/by.std(); cx=(bx-bx.mean())/bx.std()
bfx=np.column_stack([bf, W_SPACE*cx, W_SPACE*cy])
# adjacency rook
pos={b:i for i,b in enumerate(bid)}; idset=set(bid.tolist())
rows=[];cols=[]
for i,b in enumerate(bid):
    r,c=b//wb,b%wb
    for nbr,okk in ((b-1,c>0),(b+1,c<wb-1),(b-wb,r>0),(b+wb,r<hb-1)):
        if okk and nbr in idset: rows.append(i);cols.append(pos[nbr])
A=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(nb,nb)); A=((A+A.T)>0).astype(int)
adj=[[] for _ in range(nb)]; Aco=A.tocoo()
for i,j in zip(Aco.row,Aco.col): adj[i].append(j)
m=AgglomerativeClustering(n_clusters=None,distance_threshold=0,linkage='ward',connectivity=A,compute_full_tree=True).fit(bfx)
ch=m.children_;di=m.distances_;cu=np.array([1]*nb+[0]*ch.shape[0]);cn=np.zeros(ch.shape[0])
for i,(a,b2) in enumerate(ch): cn[i]=cu[a]+cu[b2];cu[nb+i]=cn[i]
Z=np.column_stack([ch,di,cn]).astype(float)
l3=fcluster(Z,55,'maxclust')
def comps(lab):
    seen=np.zeros(nb,bool);comp=np.full(nb,-1,int);cid=0
    for s in range(nb):
        if seen[s]:continue
        st=[s];seen[s]=True;mem=[]
        while st:
            u=st.pop();mem.append(u)
            for v in adj[u]:
                if not seen[v] and lab[v]==lab[u]: seen[v]=True;st.append(v)
        for q in mem: comp[q]=cid
        cid+=1
    return comp,cid
MIN=int(400/(B*B*0.09))
for _ in range(80):
    comp,nc=comps(l3); sz=np.bincount(comp); small=[c for c in range(nc) if sz[c]<MIN]
    if not small: break
    for c in sorted(small,key=lambda c:sz[c]):
        mem=np.where(comp==c)[0]; nl={}
        for u in mem:
            for v in adj[u]:
                if comp[v]!=c: nl.setdefault(l3[v],[]).append(v)
        if not nl: continue
        cm=bfx[mem].mean(0)
        best=min(nl,key=lambda lab:np.linalg.norm(bfx[np.where(l3==lab)[0]].mean(0)-cm))
        l3[mem]=best
comp,nc=comps(l3); l3=comp
rf=np.array([np.average(bfx[l3==r],axis=0,weights=cnt[l3==r]) for r in range(nc)])
rr=[];rc=[]
for i in range(nb):
    for j in adj[i]:
        if l3[i]!=l3[j]: rr.append(l3[i]);rc.append(l3[j])
RA=sparse.csr_matrix((np.ones(len(rr)),(rr,rc)),shape=(nc,nc));RA=((RA+RA.T)>0).astype(int)
m2=AgglomerativeClustering(n_clusters=None,distance_threshold=0,linkage='ward',connectivity=RA,compute_full_tree=True).fit(rf)
c2=m2.children_;d2=m2.distances_;cu=np.array([1]*nc+[0]*c2.shape[0]);co=np.zeros(c2.shape[0])
for i,(a,b2) in enumerate(c2): co[i]=cu[a]+cu[b2];cu[nc+i]=co[i]
Z2=np.column_stack([c2,d2,co]).astype(float)
cut55=fcluster(Z2,55,'maxclust');cut18=fcluster(Z2,18,'maxclust');cut5=fcluster(Z2,5,'maxclust')
piece=l3;l3=cut55[piece];l2=cut18[piece];l1=cut5[piece]
d=pd.DataFrame({'l1':l1,'l2':l2,'l3':l3})
print("L1/L2/L3",len(set(l1)),len(set(l2)),len(set(l3)),"nest",(d.groupby('l3').l2.nunique()==1).all(),(d.groupby('l2').l1.nunique()==1).all())
np.savez(f"{TMP}/hier4_2019.npz",bid=bid,wb=wb,hb=hb,B=B,l1=l1,l2=l2,l3=l3,bc=cnt,H=H,W=W,WSP=W_SPACE)
print("saved hier4_2019.npz")
