# -*- coding: utf-8 -*-
import os,numpy as np,json,time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict
P=os.path.expanduser("~/p3b")
b=np.load(f"{P}/blocks_av4_2019.npz"); h=np.load(f"{P}/hier4_2019.npz")
bid=b['bid']; bf=b['bf'].astype('float64'); cnt=b['cnt']; burnpx=b['burn']; pm=b['pm']; pstd=b['pstd']
wb=int(b['wb']); hb=int(b['hb']); wardL1=h['l1']; n=len(bid)
flam=bf[:,0]*pstd[0]+pm[0]
brow=(bid//wb).astype(float); bcol=(bid%wb).astype(float)
pos={int(x):i for i,x in enumerate(bid)}; idset=set(bid.tolist())
def adist(i,j): return np.sqrt(((bf[i]-bf[j])**2).sum())
# rook edges (attribute-weighted)
ri=[];ci=[];wt=[]
for i,bb in enumerate(bid):
    bb=int(bb); r=bb//wb; c=bb%wb
    for nbr,ok in ((bb+1,c<wb-1),(bb+wb,r<hb-1)):
        if ok and nbr in idset:
            j=pos[nbr]; ri.append(i);ci.append(j);wt.append(adist(i,j)+1e-9)
ri=np.array(ri);ci=np.array(ci);wt=np.array(wt)
A=csr_matrix((wt,(ri,ci)),shape=(n,n)); A=A+A.T
ncomp,lab=connected_components(A,directed=False)
# bridge non-giant components to giant via nearest grid block (graph completion)
import collections; sz=collections.Counter(lab.tolist()); giant=sz.most_common(1)[0][0]
gmask=lab==giant; gidx=np.where(gmask)[0]
tree=cKDTree(np.column_stack([brow[gidx],bcol[gidx]]))
bri=[];brj=[];brw=[]
for cmp in set(lab.tolist()):
    if cmp==giant: continue
    members=np.where(lab==cmp)[0]
    dd,ii=tree.query(np.column_stack([brow[members],bcol[members]]))
    k=np.argmin(dd); u=members[k]; v=gidx[ii[k]]
    bri.append(u);brj.append(v);brw.append(adist(u,v)+1e-9)
if bri:
    Ab=csr_matrix((np.array(brw),(np.array(bri),np.array(brj))),shape=(n,n)); A=A+Ab+Ab.T
ncomp2,_=connected_components(A,directed=False)
print("blocks",n,"rook edges",len(wt),"orig comps",ncomp,"after bridge",ncomp2,"bridges",len(bri))
# MST + undirected edge list
MST=minimum_spanning_tree(A); MST=MST+MST.T; mco=MST.tocoo()
edges=list({(int(min(u,v)),int(max(u,v))) for u,v in zip(mco.row,mco.col)})
feat=bf; d=feat.shape[1]
def components_from_edges(active):
    adj=[[] for _ in range(n)]
    for (u,v) in active: adj[u].append(v); adj[v].append(u)
    comp=np.full(n,-1,int); cid=0
    for s in range(n):
        if comp[s]>=0: continue
        st=[s]; comp[s]=cid
        while st:
            x=st.pop()
            for y in adj[x]:
                if comp[y]<0: comp[y]=cid; st.append(y)
        cid+=1
    return comp,cid,adj
def best_cut_in_tree(nodes,adj):
    root=nodes[0]; order=[]; parent={root:-1}; st=[root]; seen={root}
    while st:
        u=st.pop(); order.append(u)
        for v in adj[u]:
            if v not in seen: seen.add(v); parent[v]=u; st.append(v)
    s={}; q={}; csub={}
    for u in nodes: s[u]=feat[u].copy(); q[u]=feat[u]**2; csub[u]=1.0
    for u in reversed(order):
        p=parent[u]
        if p!=-1: s[p]=s[p]+s[u]; q[p]=q[p]+q[u]; csub[p]+=csub[u]
    S=s[root]; Q=q[root]; N=csub[root]; parent_ssd=(Q-S*S/N).sum()
    best=None
    for u in order:
        if u==root: continue
        nb=csub[u]; na=N-nb
        if nb<1 or na<1: continue
        sb=s[u]; qb=q[u]; sa=S-sb; qa=Q-qb
        gain=parent_ssd-(qb-sb*sb/nb).sum()-(qa-sa*sa/na).sum()
        if best is None or gain>best[0]: best=(gain,(parent[u],u))
    return best
active=set(edges); comp,ncur,adj=components_from_edges(active); target=5; t0=time.time()
while ncur<target:
    groups=defaultdict(list)
    for i in range(n): groups[comp[i]].append(i)
    bestg=None
    for cidx,nodes in groups.items():
        if len(nodes)<2: continue
        bc=best_cut_in_tree(nodes,adj)
        if bc and (bestg is None or bc[0]>bestg[0]): bestg=bc
    gain,(u,v)=bestg; active.discard((min(u,v),max(u,v)))
    comp,ncur,adj=components_from_edges(active)
    print("cut (%d,%d) gain=%.1f -> %d trees [%.1fs]"%(u,v,gain,ncur,time.time()-t0))
skater=comp.copy()
ari=adjusted_rand_score(wardL1,skater); nmi=normalized_mutual_info_score(wardL1,skater)
print("SKATER k=%d ARI=%.4f NMI=%.4f"%(ncur,ari,nmi))
# simple largest-edge variant (cut 4 largest MST edges in giant tree)
ew=np.array([A[u,v] for (u,v) in edges]); simple=set(edges)
for idx in np.argsort(-ew)[:target-1]: simple.discard(edges[idx])
sc,snc,_=components_from_edges(simple)
ari_s=adjusted_rand_score(wardL1,sc); nmi_s=normalized_mutual_info_score(wardL1,sc)
print("SIMPLE clusters=%d ARI=%.4f NMI=%.4f"%(snc,ari_s,nmi_s))
def profile(labels):
    rows=[]
    for r in sorted(set(labels.tolist())):
        m=labels==r
        rows.append((int(r),int(m.sum()),float(np.average(flam[m],weights=cnt[m])),float(100.0*burnpx[m].sum()/cnt[m].sum())))
    rows.sort(key=lambda x:-x[2]); return rows
srows=profile(skater); wrows=profile(wardL1)
gflam=float(np.average(flam,weights=cnt)); gburn=float(100.0*burnpx.sum()/cnt.sum())
print("\nSKATER k=5 (sorted by flammability): reg nblk flam burn%")
for r,nb,fl,bu in srows: print("  %-4d %7d %7.1f %7.2f"%(r,nb,fl,bu))
print("global flam=%.1f burn%%=%.2f"%(gflam,gburn))
hi=[r for r in srows if r[2]>=gflam]; hs=sorted(hi,key=lambda x:-x[3])
twin=False; detail={}
if len(hi)>=2:
    burned=hs[0]; unburned=hs[-1]
    if unburned[3]<max(1.0,gburn*0.25) and burned[3]>gburn: twin=True
    detail={'high_flam_regions':[list(x) for x in hi],'most_burned':list(burned),'least_burned':list(unburned)}
print("FUEL-TWIN present:","YES" if twin else "NO")
out={'n_blocks':int(n),'rook_edges':int(len(wt)),'graph_components':int(ncomp),'bridges':int(len(bri)),
 'skater_k':int(ncur),'ari_skater_vs_ward':float(ari),'nmi_skater_vs_ward':float(nmi),
 'simple_largest_edge':{'clusters':int(snc),'ari':float(ari_s),'nmi':float(nmi_s)},
 'skater_profile':[{'reg':r,'nblk':nb,'flam':fl,'burn_pct':bu} for r,nb,fl,bu in srows],
 'ward_profile':[{'reg':r,'nblk':nb,'flam':fl,'burn_pct':bu} for r,nb,fl,bu in wrows],
 'global_flam':gflam,'global_burn_pct':gburn,'fuel_twin_present':bool(twin),'fuel_twin_detail':detail}
json.dump(out,open(f"{P}/_skater_part.json","w"),ensure_ascii=False,indent=1,default=float)
print("saved _skater_part.json")
