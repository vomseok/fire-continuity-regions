# -*- coding: utf-8 -*-
import os,time,numpy as np,pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
from scipy import sparse
TMP=os.path.expanduser("~/p3b");d=np.load(f"{TMP}/blocks_av4.npz")
bid=d['bid'];bf=d['bf'];cnt=d['cnt'].astype(float);burn=d['burn'].astype(float);wb=int(d['wb']);hb=int(d['hb']);B=int(d['B'])
nb=len(bid);AH=(B*30.0)**2/1e4
by=(bid//wb).astype(float);bx=(bid%wb).astype(float)
cy=(by-by.mean())/by.std();cx=(bx-bx.mean())/bx.std()
pos={int(b):i for i,b in enumerate(bid)};idset=set(int(b) for b in bid)
rows=[];cols=[]
for i,b in enumerate(bid):
    b=int(b);r,c=b//wb,b%wb
    for nbr,ok in ((b-1,c>0),(b+1,c<wb-1),(b-wb,r>0),(b+wb,r<hb-1)):
        if ok and nbr in idset: rows.append(i);cols.append(pos[nbr])
A=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(nb,nb));A=((A+A.T)>0).astype(int)
adj=[[] for _ in range(nb)];Aco=A.tocoo()
for i,j in zip(Aco.row,Aco.col): adj[i].append(j)
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
def cluster(W):
    bfx=np.column_stack([bf,W*cx,W*cy])
    m=AgglomerativeClustering(n_clusters=None,distance_threshold=0,linkage='ward',connectivity=A,compute_full_tree=True).fit(bfx)
    ch=m.children_;di=m.distances_;cu=np.array([1]*nb+[0]*ch.shape[0]);cnn=np.zeros(ch.shape[0])
    for i,(a,b2) in enumerate(ch): cnn[i]=cu[a]+cu[b2];cu[nb+i]=cnn[i]
    Z=np.column_stack([ch,di,cnn]).astype(float);l3=fcluster(Z,55,'maxclust')
    for _ in range(80):
        comp,nc=comps(l3);sz=np.bincount(comp);small=[c for c in range(nc) if sz[c]<MIN]
        if not small: break
        for c in sorted(small,key=lambda c:sz[c]):
            mem=np.where(comp==c)[0];nl={}
            for u in mem:
                for v in adj[u]:
                    if comp[v]!=c: nl.setdefault(l3[v],[]).append(v)
            if not nl: continue
            cm=bfx[mem].mean(0);best=min(nl,key=lambda lab:np.linalg.norm(bfx[np.where(l3==lab)[0]].mean(0)-cm));l3[mem]=best
    comp,nc=comps(l3);l3=comp
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
    return cut5[l3],cut18[l3],cut55[l3]
def ve(lab):  # variance explained of 4 features, weighted
    w=cnt;xbar=np.average(bf,axis=0,weights=w);tot=(w[:,None]*(bf-xbar)**2).sum()
    within=0.0
    for u in np.unique(lab):
        m=lab==u;ww=w[m];cm=np.average(bf[m],axis=0,weights=ww);within+=(ww[:,None]*(bf[m]-cm)**2).sum()
    return 1-within/tot
def area80(lab):
    us=np.unique(lab);n=np.array([cnt[lab==u].sum() for u in us]);b=np.array([burn[lab==u].sum() for u in us])
    keep=n>0;n=n[keep];b=b[keep];rate=b/n;o=np.argsort(-rate);cn=np.cumsum(n[o])/n.sum();cb=np.cumsum(b[o])/b.sum()
    return cn[np.searchsorted(cb,0.8)]
def compact(lab):  # mean fill ratio weighted by size (L1)
    tot=0;wsum=0
    for u in np.unique(lab):
        m=lab==u;r=by[m];c=bx[m];fill=m.sum()/(((r.max()-r.min()+1)*(c.max()-c.min()+1)));tot+=fill*m.sum();wsum+=m.sum()
    return tot/wsum
WS=[0.0,0.3,0.6,1.0,1.3,1.6,2.0]
out=f"{TMP}/sweep4.csv"
done=set()
if os.path.exists(out):
    done=set(round(float(x),2) for x in pd.read_csv(out)['W'])
t0=time.time()
rowsout=[]
for W in WS:
    if round(W,2) in done: continue
    l1,l2,l3=cluster(W)
    rec=dict(W=W,nL1=len(set(l1)),nL2=len(set(l2)),nL3=len(set(l3)),
             VE_L2=round(ve(l2),3),VE_L3=round(ve(l3),3),
             area80_L2=round(area80(l2),3),area80_L3=round(area80(l3),3),
             compact_L1=round(compact(l1),3),compact_L2=round(compact(l2),3))
    rowsout.append(rec);print(rec)
    hdr=not os.path.exists(out)
    pd.DataFrame([rec]).to_csv(out,mode='a',header=hdr,index=False)
    if time.time()-t0>120: print("time budget, exiting");break
print("done this run:",len(rowsout))
