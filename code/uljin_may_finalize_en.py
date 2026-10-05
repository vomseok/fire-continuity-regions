# -*- coding: utf-8 -*-
import os,json,numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score
P=os.path.expanduser("~/p3b")
us=np.load(f"{P}/uljin_spring.npz"); forest=us['forest']; burn_apr=us['burn']
spec=np.load(f"{P}/uljin_spec19.npy").astype('float32'); spec=np.where(spec<0,np.nan,spec)
ar=np.load(f"{P}/uljin_may_arrays.npz"); dnbr=ar['dnbr']; px_ha=0.09
evergreen=np.isin(np.round(spec),[100,95,82,68])
def mask(th,minha=50):
    m=np.isfinite(dnbr)&(dnbr>=th)&forest; lab,n=ndimage.label(m)
    if n: sz=ndimage.sum(np.ones_like(lab),lab,range(1,n+1));k=np.where(sz*px_ha>=minha)[0]+1;m=np.isin(lab,list(k))
    return m
burn5=mask(0.20); A5=burn5.sum()*px_ha
def auc(bm,sub=None):
    mm=forest&np.isfinite(spec)
    if sub is not None: mm=mm&sub
    y=bm[mm].astype(int); x=spec[mm]
    return float(roc_auc_score(y,x)) if y.sum()>10 and (y==0).sum()>10 else None
d=json.load(open(f"{P}/uljin_may.json"))
d['adopted_TH']=0.20; d['adopted_area_ha']=float(A5)
d['auc_may_TH020']={'all':auc(burn5),'evergreen':auc(burn5,evergreen)}
d['auc_robust_byTH']={('%.2f'%t):auc(mask(t)) for t in [0.10,0.15,0.20,0.27,0.35]}
d['evergreen_frac_burn']=float(evergreen[forest&burn5].mean())
d['evergreen_frac_unburn']=float(evergreen[forest&~burn5].mean())
json.dump(d,open(f"{P}/uljin_may.json","w"),ensure_ascii=False,indent=1)
# 미리보기 (영문 라벨)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
mean_core=d['mean_flam_core']; mean_all=d['mean_flam_all']; offset=d['offset']
from mpl_toolkits.axes_grid1 import make_axes_locatable
fy,fx=np.where(forest); r0,r1,c0,c1=int(fy.min()),int(fy.max()),int(fx.min()),int(fx.max()); pdv=12
def _crop(a): a.set_xlim(c0-pdv,c1+pdv); a.set_ylim(r1+pdv,r0-pdv); a.set_axis_off()
fig,ax=plt.subplots(1,3,figsize=(15,6.6))
ax[0].imshow(np.where(forest,0.85,np.nan),cmap='Greys',vmin=0,vmax=1)
im=ax[0].imshow(np.where(forest,dnbr,np.nan),cmap='RdYlGn_r',vmin=-0.1,vmax=0.8)
_crop(ax[0]); ax[0].set_title("(a) May dNBR (pre 2021-05-19 / post 2022-05-29)\nKey&Benson offset=%.3f"%offset,fontsize=14,pad=6)
plt.colorbar(im,cax=make_axes_locatable(ax[0]).append_axes("right",size="4%",pad=0.05))
ax[1].imshow(np.where(forest,0.9,np.nan),cmap='Greys',vmin=0,vmax=1)
ax[1].imshow(np.where(burn_apr&~burn5,1,np.nan),cmap=ListedColormap([(0.1,0.3,0.9)]),alpha=0.45)
ax[1].imshow(np.where(burn5,1,np.nan),cmap=ListedColormap([(0.85,0.07,0.09)]))
_crop(ax[1]); ax[1].set_title("(b) May burn %.0f ha (red, TH=0.20)\nApr-only footprint (blue)"%A5,fontsize=14,pad=6)
im2=ax[2].imshow(np.where(forest&np.isfinite(spec),spec,np.nan),cmap='YlOrRd',vmin=20,vmax=100)
_crop(ax[2]); ax[2].set_title("(c) 2019 species flammability\ncore burn mean=%.0f vs forest=%.0f"%(mean_core,mean_all),fontsize=14,pad=6)
plt.colorbar(im2,cax=make_axes_locatable(ax[2]).append_axes("right",size="4%",pad=0.05))
plt.tight_layout()
outpng=os.path.expanduser("~/mnt/\ub300\ud615\uc0b0\ubd88_FVI/\ub17c\ubb383/uljin_may_preview_en.png")
plt.savefig(outpng,dpi=220,facecolor="white",bbox_inches="tight"); print("saved preview",outpng,round(A5))
print(json.dumps(d,ensure_ascii=False,indent=1))
