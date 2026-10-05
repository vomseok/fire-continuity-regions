# -*- coding: utf-8 -*-
import os,numpy as np,json
from sklearn.metrics import roc_auc_score
P=os.path.expanduser("~/p3b")
d=np.load(f"{P}/_transfer_arrays.npz"); pu=d['pu']; puS=d['puS']; yM=d['yM']; yA=d['yA']
rng=np.random.default_rng(42); N=len(yM); B=1000
def boot(prob,y):
    base=roc_auc_score(y,prob); vals=[]
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]
    for _ in range(B):
        # stratified resample preserves class balance; avoids degenerate all-one-class draws
        ii=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
        vals.append(roc_auc_score(y[ii],prob[ii]))
    vals=np.array(vals)
    return base,float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5)),float(vals.mean()),float(vals.std())
res={}
for name,prob,y in [('may_Fterrain',pu,yM),('may_species_only',puS,yM),('apr_Fterrain',pu,yA),('apr_species_only',puS,yA)]:
    base,lo,hi,mu,sd=boot(prob,y)
    res[name]={'auc':float(base),'ci95':[lo,hi],'boot_mean':mu,'boot_sd':sd,'n':int(len(y)),'pos':int(y.sum())}
    print("%-18s AUC=%.4f  95%%CI=[%.4f, %.4f]  (n=%d pos=%d)"%(name,base,lo,hi,len(y),y.sum()))
json.dump({'n_uljin_px':int(N),'n_boot':B,'results':res},open(f"{P}/_bootstrap.json","w"),indent=1)
print("saved _bootstrap.json")
