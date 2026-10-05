# -*- coding: utf-8 -*-
import os,numpy as np,json,collections
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
P=os.path.expanduser("~/p3b")
b=np.load(f"{P}/blocks_av4_2019.npz")
bid=b['bid']; bf=b['bf'].astype('float64'); cnt=b['cnt']; burnpx=b['burn']; pm=b['pm']; pstd=b['pstd']
wb=int(b['wb']); hb=int(b['hb']); n=len(bid)
flam=bf[:,0]*pstd[0]+pm[0]
pos={int(x):i for i,x in enumerate(bid)}; idset=set(bid.tolist())
tot_area=cnt.sum(); tot_burn=burnpx.sum(); res={}
for TH in [65]:
    si=np.where(flam>=TH)[0]; sis=set(si.tolist())
    subpos={int(x):k for k,x in enumerate(si)}
    rr=[];cc=[]
    for k,i in enumerate(si):
        bb=int(bid[i]); r=bb//wb; c=bb%wb
        for nbr,ok in ((bb+1,c<wb-1),(bb+wb,r<hb-1)):
            if ok and nbr in idset:
                j=pos[nbr]
                if j in sis: rr.append(k);cc.append(subpos[j])
    Asub=csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(si),len(si))); Asub=Asub+Asub.T
    ncc,sl=connected_components(Asub,directed=False)
    csz=collections.Counter(sl.tolist()); big=csz.most_common(1)[0][0]; bbidx=si[sl==big]
    for name,idxs in [('backbone_largest',bbidx),('all_geTH',si)]:
        a=cnt[idxs].sum(); bu=burnpx[idxs].sum()
        res[f'{name}_TH{TH}']={'area_pct':float(100*a/tot_area),'burn_capture_pct':float(100*bu/tot_burn),
            'within_burn_rate_pct':float(100*bu/a),'lift':float((100*bu/tot_burn)/(100*a/tot_area)),
            'nblk':int(len(idxs)),'n_components':int(ncc)}
        print("%s: area%%=%.1f burn_capture%%=%.1f within%%=%.2f lift=%.2f nblk=%d ncomp=%d"%(
            name,res[f'{name}_TH{TH}']['area_pct'],res[f'{name}_TH{TH}']['burn_capture_pct'],
            res[f'{name}_TH{TH}']['within_burn_rate_pct'],res[f'{name}_TH{TH}']['lift'],len(idxs),ncc))
res['overall_burn_pct']=float(100*tot_burn/tot_area); print("overall burn%%=%.2f"%res['overall_burn_pct'])
json.dump(res,open(f"{P}/_backbone.json","w"),indent=1,default=float); print("saved")
