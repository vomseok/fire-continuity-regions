# -*- coding: utf-8 -*-
import os,json,numpy as np,rasterio
from rasterio.warp import reproject,Resampling
from affine import Affine
from scipy import ndimage
from sklearn.metrics import roc_auc_score
P=os.path.expanduser("~/p3b"); M=f"{P}/may_tmp"
# --- 울진창 (5186, 30m, 1406x1838) : 4월 footprint와 동일 격자 ---
E0_f,N1_f,res=221910.0,615330.0,30.0
col_off,row_off,W,H=5309,2840,1406,1838
subtr=Affine(res,0,E0_f+col_off*res,0,-res,N1_f-row_off*res)   # (30,0,381180,0,-30,530130)
sh=(H,W); px_ha=res*res/1e4
# --- 참조 자료 (동일 창) ---
us=np.load(f"{P}/uljin_spring.npz")
forest=us['forest']            # bool
dnbr_apr=us['dnbr'].astype('float32')  # 4월 dNBR (offset 보정됨)
burn_apr=us['burn']            # 4월 소실 마스크
spec=np.load(f"{P}/uljin_spec19.npy").astype('float32')  # 2019 수종 가연성, -1=결측
spec=np.where(spec<0,np.nan,spec)
# --- 5월쌍 reproject: NBR_pre, NBR_post 각각 ---
def to5186(path):
    d=rasterio.open(path); a=d.read(1).astype('float32')
    a=np.where((a>0)&(a<65535),a,np.nan)       # 0/65535 nodata
    o=np.full(sh,np.nan,'float32')
    reproject(a,o,src_transform=d.transform,src_crs=d.crs,dst_transform=subtr,dst_crs="EPSG:5186",
              resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
    return o
pn=to5186(f"{M}/pre_B08.tiff"); ps=to5186(f"{M}/pre_B12.tiff")
on=to5186(f"{M}/post_B08.tiff"); osw=to5186(f"{M}/post_B12.tiff")
def nbr(n,s): v=n+s; return np.where(v!=0,(n-s)/v,np.nan)
NBR_pre=nbr(pn,ps); NBR_post=nbr(on,osw)
dnbr_raw=NBR_pre-NBR_post
valid=np.isfinite(dnbr_raw)&forest
cov=100*np.isfinite(dnbr_raw[forest]).sum()/max(forest.sum(),1)
print("5월 산림내 유효 화소 %%= %.1f"%cov)
# --- (a) offset 보정 (Key&Benson): 불연소 대조화소 median ---
# 대조화소 = 산림 & dnbr_raw<0.1 (연소 후보 제외)
ctrl=valid&(dnbr_raw<0.10)
offset=float(np.nanmedian(dnbr_raw[ctrl]))
dnbr=np.where(valid,dnbr_raw-offset,np.nan)
print("offset(Key&Benson 대조화소 median)= %.4f"%offset)
# --- (b) RdNBR = dnbr / sqrt(|NBR_pre|) ---
rdnbr=np.where(valid&(np.abs(NBR_pre)>1e-6), dnbr/np.sqrt(np.abs(NBR_pre)), np.nan)
# --- 3. 임계별 소실 면적 (연결성분 >=50ha) ---
def area(field,th,minha=50):
    m=np.isfinite(field)&(field>=th)&forest; lab,n=ndimage.label(m)
    if n==0: return m,0.0,0.0
    sz=ndimage.sum(np.ones_like(lab),lab,range(1,n+1)); keep=np.where(sz*px_ha>=minha)[0]+1
    m=np.isin(lab,list(keep)); return m,m.sum()*px_ha,sz.max()*px_ha
res_th={}
print("=== dNBR(offset보정) 임계별 면적 ===")
for th in [0.10,0.15,0.20,0.27,0.35]:
    _,a,b=area(dnbr,th); res_th["dnbr_%.2f"%th]=[round(a),round(b)]
    print("  TH%.2f: %8.0f ha (최대성분 %7.0f)"%(th,a,b))
print("=== dNBR(raw, offset 미보정) 임계별 면적 ===")
for th in [0.10,0.15,0.20,0.27,0.35]:
    _,a,b=area(dnbr_raw,th); res_th["raw_%.2f"%th]=[round(a),round(b)]
    print("  TH%.2f: %8.0f ha (최대성분 %7.0f)"%(th,a,b))
print("=== RdNBR 임계별 면적 ===")
for th in [0.20,0.30,0.40,0.50,0.65]:
    _,a,b=area(rdnbr,th); res_th["rdnbr_%.2f"%th]=[round(a),round(b)]
    print("  TH%.2f: %8.0f ha (최대성분 %7.0f)"%(th,a,b))
# 채택 소실마스크(5월): 4월 footprint(12,343ha)와 비교 위해 dnbr 기준 TH 자동탐색(최대성분 매칭 어려우니 면적 12343 근사)
grid=np.arange(0.05,0.8,0.005)
best=min(grid,key=lambda t:abs(area(dnbr,t)[1]-12343))
burn5,A5,big5=area(dnbr,float(best)); TH5=float(best)
print("채택 5월 TH=%.3f -> %.0f ha (최대성분 %.0f);  4월 footprint=12,343 ha"%(TH5,A5,big5))
# --- 4. 핵심검증: 2019 수종 가연성 -> 소실 AUC ---
def auc_spec(burnmask,subset=None):
    m=forest&np.isfinite(spec)
    if subset is not None: m=m&subset
    y=burnmask[m].astype(int); x=spec[m]
    if y.sum()<10 or (y==0).sum()<10: return float('nan'),int(y.sum())
    return float(roc_auc_score(y,x)), int(y.sum())
evergreen=np.isin(np.round(spec),[100,95,82,68])   # 상록침엽 상위 (소나무 등)
res_auc={}
for tag,bm in [("may",burn5),("apr",burn_apr)]:
    a_all,n_all=auc_spec(bm); a_ev,n_ev=auc_spec(bm,evergreen)
    res_auc[tag]={"all":a_all,"all_n":n_all,"evergreen":a_ev,"ev_n":n_ev}
    print("[검증 %s footprint] 수종가연성->소실 AUC 전체=%.3f(소실%d) / 상록하위집합=%.3f(소실%d)"%(tag,a_all,n_all,a_ev,n_ev))
# 소실 핵심부 평균 가연성 (2019): dnbr 상위 25% 핵심
core=forest&np.isfinite(spec)&np.isfinite(dnbr)
thr_core=np.nanpercentile(dnbr[core],75)
corem=core&(dnbr>=thr_core)
mean_core=float(np.nanmean(spec[corem])); mean_all=float(np.nanmean(spec[core]))
mean_burn5=float(np.nanmean(spec[forest&np.isfinite(spec)&burn5]))
mean_unb=float(np.nanmean(spec[forest&np.isfinite(spec)&~burn5]))
print("소실핵심부(dNBR상위25%%) 평균가연성(2019)=%.1f / 전체산림=%.1f / 소실=%.1f / 불연소=%.1f"%(mean_core,mean_all,mean_burn5,mean_unb))
# --- 5. 4월 vs 5월 phenology 오염 비교: 불연소 산림에서 낙엽수 가짜 dNBR ---
decid=np.isin(np.round(spec),[45,42,38])   # 낙엽활엽/혼효 저가연성(참나무류 등)
unburn=forest&np.isfinite(spec)&(~burn5)&(~burn_apr)  # 양쪽 모두 불연소
def phen(field):
    d_dec=np.nanmean(field[unburn&decid]); d_eg=np.nanmean(field[unburn&evergreen])
    # 불연소에서 낙엽수-상록 분리 AUC (phenology 가짜신호 지표; 0.5=깨끗)
    m=unburn&(decid|evergreen)&np.isfinite(field)
    y=decid[m].astype(int); x=field[m]
    au=roc_auc_score(y,x) if y.sum()>10 and (y==0).sum()>10 else float('nan')
    return float(d_dec),float(d_eg),float(au)
apr_dec,apr_eg,apr_auc=phen(dnbr_apr)
may_dec,may_eg,may_auc=phen(dnbr)
print("=== phenology 오염 (불연소 산림에서 낙엽수 가짜 dNBR) ===")
print("  4월: 낙엽수평균dNBR=%.4f 상록=%.4f  낙엽>상록분리AUC=%.3f"%(apr_dec,apr_eg,apr_auc))
print("  5월: 낙엽수평균dNBR=%.4f 상록=%.4f  낙엽>상록분리AUC=%.3f"%(may_dec,may_eg,may_auc))
print("  (0.5에 가까울수록 phenology 오염 적음)")
# --- 저장 ---
out=dict(coverage_pct=cov, offset=offset, TH5=TH5, area5_ha=A5, maxcomp5_ha=big5,
    area_by_th=res_th, auc=res_auc,
    mean_flam_core=mean_core, mean_flam_all=mean_all, mean_flam_burn=mean_burn5, mean_flam_unburn=mean_unb,
    phenology=dict(apr=dict(decid=apr_dec,evergreen=apr_eg,sep_auc=apr_auc),
                   may=dict(decid=may_dec,evergreen=may_eg,sep_auc=may_auc)),
    forest_px=int(forest.sum()), evergreen_px=int((forest&evergreen).sum()), decid_px=int((forest&decid).sum()))
json.dump(out,open(f"{P}/uljin_may.json","w"),ensure_ascii=False,indent=1)
np.savez(f"{P}/uljin_may_arrays.npz",dnbr=dnbr.astype('float32'),rdnbr=rdnbr.astype('float32'),
         burn5=burn5,NBR_pre=NBR_pre.astype('float32'))
print("saved uljin_may.json + arrays")
# --- 미리보기 PNG ---
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
fig,ax=plt.subplots(1,3,figsize=(16,8))
ax[0].imshow(np.where(forest,0.85,np.nan),cmap='Greys',vmin=0,vmax=1)
im=ax[0].imshow(np.where(forest,dnbr,np.nan),cmap='RdYlGn_r',vmin=-0.1,vmax=0.8)
ax[0].set_title("5월 dNBR (2021-05-19 / 2022-05-29)\noffset=%.3f"%offset);ax[0].axis('off');plt.colorbar(im,ax=ax[0],fraction=0.04)
ax[1].imshow(np.where(forest,0.9,np.nan),cmap='Greys',vmin=0,vmax=1)
ax[1].imshow(np.where(burn5,1,np.nan),cmap=ListedColormap([(0.85,0.07,0.09)]))
ax[1].imshow(np.where(burn_apr&~burn5,1,np.nan),cmap=ListedColormap([(0.1,0.3,0.9)]),alpha=0.5)
ax[1].set_title("5월 소실 %.0fha(적) TH=%.2f\n4월 전용부(청)"%(A5,TH5));ax[1].axis('off')
im2=ax[2].imshow(np.where(forest&np.isfinite(spec),spec,np.nan),cmap='YlOrRd',vmin=20,vmax=100)
ax[2].set_title("2019 수종 가연성\n(소실核평균=%.0f vs 전체=%.0f)"%(mean_core,mean_all));ax[2].axis('off');plt.colorbar(im2,ax=ax[2],fraction=0.04)
plt.tight_layout()
outpng=os.path.expanduser("~/mnt/대형산불_FVI/논문3/uljin_may_preview.png")
os.makedirs(os.path.dirname(outpng),exist_ok=True)
plt.savefig(outpng,dpi=110,facecolor='white'); print("saved preview",outpng)
