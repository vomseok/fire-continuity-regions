# -*- coding: utf-8 -*-
import numpy as np,rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
B="/sessions/placeholder"
import os
R=os.path.expanduser("~/mnt/대형산불_FVI")
P={'fuel':f'{R}/Claude outputs/fuel.tif','soil':f'{R}/data/clip/rasters/soil.tif',
'elev':f'{R}/Claude outputs/terrain_elev.tif','slope':f'{R}/Claude outputs/terrain_slope.tif','north':f'{R}/Claude outputs/terrain_northness.tif',
'spec':f'{R}/Claude outputs/paper3_layers/forest_spec.tif','dens':f'{R}/Claude outputs/paper3_layers/forest_dens.tif',
'diam':f'{R}/Claude outputs/paper3_layers/forest_diam.tif','age':f'{R}/Claude outputs/paper3_layers/forest_age.tif'}
def rd(p):
    with rasterio.open(p) as d:
        a=d.read(1).astype('float32'); nod=d.nodata
    if nod is not None: a=np.where(a==nod,np.nan,a)
    return a
arr={k:rd(p) for k,p in P.items()}
ims=rd(f'{R}/data/clip/rasters/imsang.tif')
with rasterio.open(f'{R}/Claude outputs/FVI/reproduction/results/rasters/burn_perimeter.tif') as d: burn=(d.read(1)==1)
mask=(ims>0)
for k in arr: mask&=np.isfinite(arr[k])
y=burn[mask].astype(int)
cols=['fuel','soil','elev','slope','north','spec','dens','diam','age']
X=np.column_stack([arr[k][mask] for k in cols]).astype('float64')
print("forest px",len(y),"burn%",round(100*y.mean(),2))
rng=np.random.default_rng(0); idx=rng.choice(len(y),min(200000,len(y)),replace=False)
Xs=StandardScaler().fit_transform(X[idx]); ys=y[idx]
Xtr,Xte,ytr,yte=train_test_split(Xs,ys,test_size=0.4,random_state=1,stratify=ys)
def auc(colsel):
    m=LogisticRegression(max_iter=2000,class_weight='balanced').fit(Xtr[:,colsel],ytr)
    return roc_auc_score(yte,m.predict_proba(Xte[:,colsel])[:,1])
ci={c:i for i,c in enumerate(cols)}
print("\n=== univariate AUC (each var alone) ===")
for c in cols: print(f"  {c:5s} {auc([ci[c]]):.3f}")
base=[ci['fuel'],ci['elev'],ci['slope'],ci['north']]
ab=auc(base);print(f"\nBASE (fuel+elev+slope+north) AUC={ab:.3f}")
print("=== incremental ΔAUC (add to BASE) ===")
for c in ['soil','spec','dens','diam','age']:
    print(f"  +{c:5s} {auc(base+[ci[c]]):.3f}  (Δ{auc(base+[ci[c]])-ab:+.3f})")
full=list(range(len(cols)));print(f"\nFULL (all 9) AUC={auc(full):.3f}")
# standardized coefficients (full)
m=LogisticRegression(max_iter=3000,class_weight='balanced').fit(Xtr,ytr)
print("=== full-model standardized |coef| (importance) ===")
for c,co in sorted(zip(cols,m.coef_[0]),key=lambda t:-abs(t[1])): print(f"  {c:5s} {co:+.3f}")
