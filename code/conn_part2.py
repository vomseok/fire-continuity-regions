import numpy as np, json, pickle
from scipy import ndimage as ndi
d=np.load('_conn_tmp.npz')
flam=d['flam']; mask=d['mask']; gl2=d['gl2']; gl3=d['gl3']; burnfrac=d['burnfrac']; gburn=d['gburn']; gcnt=d['gcnt']
queen=np.ones((3,3),int); Ntot=mask.sum()
o=pickle.load(open('_conn_part1.pkl','rb'))

def adj_cross(occ,reg):
    # count adjacent occupied block-pairs (4-neigh) that cross a region boundary vs within
    within=0;cross=0
    # horizontal
    a=occ[:,:-1]&occ[:,1:]
    same=(reg[:,:-1]==reg[:,1:])&a
    within+=int(same.sum()); cross+=int((a&~same).sum())
    # vertical
    a=occ[:-1,:]&occ[1:,:]
    same=(reg[:-1,:]==reg[1:,:])&a
    within+=int(same.sum()); cross+=int((a&~same).sum())
    tot=within+cross
    return within,cross,(cross/tot if tot else 0.0)

# 3. Region alignment at key thresholds
align={}
for t in [80,65,60]:
    occ=(flam>=t)&mask
    w2,c2,f2=adj_cross(occ,gl2)
    w3,c3,f3=adj_cross(occ,gl3)
    # baseline: all forest-block adjacencies crossing L2/L3
    bw2,bc2,bf2=adj_cross(mask,gl2)
    bw3,bc3,bf3=adj_cross(mask,gl3)
    # per-L2: fraction of each global queen-cluster captured within a single L2
    lbl,n=ndi.label(occ,queen)
    capt=[]
    if n>0:
        sizes=np.bincount(lbl.ravel())
        for cid in range(1,n+1):
            if sizes[cid]<5: continue
            regs=gl2[lbl==cid]
            vals,cnts=np.unique(regs,return_counts=True)
            capt.append(cnts.max()/cnts.sum())  # dominant-L2 share
    align[str(t)]={
      'L2_cross_frac':round(f2,4),'L2_cross_frac_baseline':round(bf2,4),
      'L3_cross_frac':round(f3,4),'L3_cross_frac_baseline':round(bf3,4),
      'cross_reduction_vs_baseline_L2':round(1-f2/bf2 if bf2 else 0,3),
      'n_clusters_ge5':len(capt),
      'mean_dominant_L2_share':round(float(np.mean(capt)),3) if capt else None,
      'frac_clusters_single_L2_ge0.8':round(float(np.mean([c>=0.8 for c in capt])),3) if capt else None
    }
o['region_alignment']=align

# 4. Burn vs connectivity-component membership
burn_rel={}
for t in [80,65,60]:
    occ=(flam>=t)&mask
    lbl,n=ndi.label(occ,queen)
    sizes=np.bincount(lbl.ravel()) if n>0 else np.array([0])
    giant=int(np.argmax(sizes[1:]))+1 if n>0 else -1
    inG = (lbl==giant)
    # burn fraction groups (only forest blocks)
    bf_all=burnfrac[mask]
    grp_giant=burnfrac[inG & mask]
    grp_occ_nong=burnfrac[occ & ~inG & mask]
    grp_low=burnfrac[~occ & mask]
    # burned pixel totals
    def stat(a): return {'n':int(a.size),'mean_burnfrac':round(float(a.mean()),4),'frac_blocks_burned':round(float((a>0).mean()),4)}
    # burn concentration: share of total burned pixels in giant cluster
    totburn=gburn[mask].sum()
    burn_rel[str(t)]={
      'giant':stat(grp_giant),'occupied_nongiant':stat(grp_occ_nong),'unoccupied':stat(grp_low),
      'giant_share_of_total_burn':round(float(gburn[inG&mask].sum()/totburn),4),
      'giant_share_of_forest_area':round(float((inG&mask).sum()/Ntot),4),
      'occupied_share_of_total_burn':round(float(gburn[occ&mask].sum()/totburn),4),
      'occupied_share_of_forest_area':round(float((occ&mask).sum()/Ntot),4)
    }
# burn vs cluster size: correlate component size with mean burnfrac (th65)
occ=(flam>=65)&mask; lbl,n=ndi.label(occ,queen)
csize=np.zeros(n+1); cburn=np.zeros(n+1); ccnt=np.zeros(n+1)
flat=lbl.ravel(); bfl=burnfrac.ravel()
for cid in range(1,n+1):
    m=flat==cid; csize[cid]=m.sum(); cburn[cid]=bfl[m].sum(); ccnt[cid]=m.sum()
mb=cburn[1:]/np.maximum(ccnt[1:],1)
sz=csize[1:]
# spearman-ish: correlation of log size vs mean burnfrac
from scipy.stats import spearmanr
rho,pv=spearmanr(sz,mb)
burn_rel['size_vs_burn_th65']={'spearman_rho':round(float(rho),3),'p':round(float(pv),4),
   'giant_meanburn':round(float(mb[np.argmax(sz)]),4),
   'small_clusters(<10blk)_meanburn':round(float(mb[sz<10].mean()),4)}
o['burn_connectivity']=burn_rel

json.dump(o,open('connectivity.json','w'),ensure_ascii=False,indent=1,
   default=lambda x:int(x) if 'int' in type(x).__name__ else float(x))
print(json.dumps({'region_alignment':align,'burn_connectivity':burn_rel},ensure_ascii=False,indent=1,
   default=lambda x:int(x) if 'int' in type(x).__name__ else float(x)))
pickle.dump(o,open('_conn_full.pkl','wb'))
