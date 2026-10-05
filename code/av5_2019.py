import os,numpy as np,rasterio
from rasterio.warp import reproject,Resampling as RS
from rasterio.transform import from_origin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
TMP=os.path.expanduser("~/p3b");CO=os.path.expanduser("~/mnt/대형산불_FVI/Claude outputs")
E0,N0,R,Wd,Hd=221910.0,323280.0,30.0,7185,9735;N1=N0+Hd*R;TR=from_origin(E0,N1,R,R);CRS="EPSG:5186"
d=np.load(f"{TMP}/hier4_2019.npz");bid=d['bid'];wb=int(d['wb']);hb=int(d['hb']);B=int(d['B'])
l1,l2,l3=d['l1'],d['l2'],d['l3'];rr=bid//wb;cc=bid%wb;nblk=hb*wb;bf=rr*wb+cc
BL1=np.zeros(nblk,'int32');BL2=np.zeros(nblk,'int32');BL3=np.zeros(nblk,'int32')
BL1[bf]=l1+1;BL2[bf]=l2+1;BL3[bf]=l3+1
with rasterio.open(f"{CO}/FVI/reproduction/results/rasters/burn_perimeter.tif") as bs:
    tmp=np.zeros((Hd,Wd),'float32');reproject(rasterio.band(bs,1),tmp,dst_transform=TR,dst_crs=CRS,src_transform=bs.transform,src_crs=bs.crs,resampling=RS.nearest,src_nodata=bs.nodata,dst_nodata=0)
burn=(tmp==1).astype('uint8');del tmp
us=np.load(f"{TMP}/uljin_spring.npz");keep=us['burn'];col_off,row_off=5309,2840
ubu=np.zeros((Hd,Wd),'uint8');hh,ww=keep.shape;ubu[row_off:row_off+hh,col_off:col_off+ww]=keep.astype('uint8')
dfor=rasterio.open(f"{TMP}/forest3p.tif");dF=rasterio.open(f"{TMP}/spec3_2019.tif")
dE=rasterio.open(f"{TMP}/exp3_elev.tif");dS=rasterio.open(f"{TMP}/exp3_slope.tif");dN=rasterio.open(f"{TMP}/exp3_north.tif")
from collections import defaultdict
A1=defaultdict(lambda:[0,0]);A2=defaultdict(lambda:[0,0]);A3=defaultdict(lambda:[0,0])
U2=defaultdict(lambda:[0,0]);U3=defaultdict(lambda:[0,0])
rng=np.random.default_rng(0);subs=[];STRIP=900;uly0,uly1=row_off,row_off+hh
for r0 in range(0,Hd,STRIP):
    r1=min(r0+STRIP,Hd);win=rasterio.windows.Window(0,r0,Wd,r1-r0)
    fo=dfor.read(1,window=win)>0
    if not fo.any():continue
    F=dF.read(1,window=win).astype('float32');F=np.where(F<0,np.nan,F)
    el=dE.read(1,window=win);sl=dS.read(1,window=win);no=dN.read(1,window=win)
    bu=burn[r0:r1,:];ub=ubu[r0:r1,:]
    ok=fo&np.isfinite(F)&np.isfinite(el)&np.isfinite(sl)&np.isfinite(no)
    yy,xx=np.where(ok)
    if len(yy)==0:continue
    gr=(r0+yy)//B;gc=xx//B;bb=gr*wb+gc
    r1l=BL1[bb];r2l=BL2[bb];r3l=BL3[bb];bv=bu[yy,xx].astype('int64');uv=ub[yy,xx].astype('int64')
    for keys,acc in ((r1l,A1),(r2l,A2),(r3l,A3)):
        for k in np.unique(keys):
            if k==0:continue
            m=keys==k;acc[int(k)][0]+=int(m.sum());acc[int(k)][1]+=int(bv[m].sum())
    inu=((r0+yy)>=uly0)&((r0+yy)<uly1)
    if inu.any():
        for keys,acc in ((r2l,U2),(r3l,U3)):
            kk=keys[inu];uu=uv[inu]
            for k in np.unique(kk):
                if k==0:continue
                m=kk==k;acc[int(k)][0]+=int(m.sum());acc[int(k)][1]+=int(uu[m].sum())
    gx=E0+(xx+0.5)*R;gy=N1-(r0+yy+0.5)*R
    sel=rng.random(len(yy))<0.04
    if sel.any():
        subs.append(np.column_stack([F[yy[sel],xx[sel]],el[yy[sel],xx[sel]],sl[yy[sel],xx[sel]],no[yy[sel],xx[sel]],
            r1l[sel],r2l[sel],r3l[sel],bv[sel],uv[sel],gx[sel],gy[sel],
            ((r0+yy[sel])>=uly0)&((r0+yy[sel])<uly1)]))
S=np.vstack(subs)
tn=sum(v[0] for v in A1.values());tb=sum(v[1] for v in A1.values())
print("forest px",tn,"2025burn%",round(100*tb/tn,3),"uljin sub px",int(S[:,8].sum()),"sub",S.shape)
F=S[:,0];el=S[:,1];sl=S[:,2];no=S[:,3];r1=S[:,4].astype(int);r2=S[:,5].astype(int);r3=S[:,6].astype(int)
y=S[:,7].astype(int);yu=S[:,8].astype(int);gx=S[:,9];gy=S[:,10];inuw=S[:,11].astype(bool)
X=np.column_stack([F,el,sl,no]);SC=StandardScaler().fit(X);Xs=SC.transform(X)
Xtr,Xte,ytr,yte,itr,ite=train_test_split(Xs,y,np.arange(len(y)),test_size=0.4,random_state=1,stratify=y)
def auc(cols): m=LogisticRegression(max_iter=2000,class_weight='balanced').fit(Xtr[:,cols],ytr);return roc_auc_score(yte,m.predict_proba(Xte[:,cols])[:,1])
def aucr(rl):
    oh=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(rl.reshape(-1,1))
    tr=np.hstack([Xtr,oh.transform(rl[itr].reshape(-1,1))]);te=np.hstack([Xte,oh.transform(rl[ite].reshape(-1,1))])
    sc=StandardScaler().fit(tr);m=LogisticRegression(max_iter=3000,class_weight='balanced').fit(sc.transform(tr),ytr)
    return roc_auc_score(yte,m.predict_proba(sc.transform(te))[:,1])
AUC_sp=auc([0]);AUC_ft=auc([0,1,2,3]);AUC_L1=aucr(r1);AUC_L2=aucr(r2)
print("[2025 split] 수종:%.3f F+지형:%.3f +L1:%.3f +L2:%.3f"%(AUC_sp,AUC_ft,AUC_L1,AUC_L2))
def tiles(gx,gy,nt=5):
    tx=np.clip(((gx-gx.min())/(gx.max()-gx.min()+1e-9)*nt).astype(int),0,nt-1)
    ty=np.clip(((gy-gy.min())/(gy.max()-gy.min()+1e-9)*nt).astype(int),0,nt-1);return tx*nt+ty
tl=tiles(gx,gy,5)
def spcv(cols,withL2=False):
    pr=np.full(len(y),np.nan)
    for t in np.unique(tl):
        te=tl==t;trn=~te
        if y[trn].sum()<5 or y[te].sum()<1:continue
        if not withL2:
            m=LogisticRegression(max_iter=2000,class_weight='balanced').fit(Xs[trn][:,cols],y[trn]);pr[te]=m.predict_proba(Xs[te][:,cols])[:,1]
        else:
            oh=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(r2[trn].reshape(-1,1))
            Tr=np.hstack([Xs[trn],oh.transform(r2[trn].reshape(-1,1))]);Te=np.hstack([Xs[te],oh.transform(r2[te].reshape(-1,1))])
            m=LogisticRegression(max_iter=3000,class_weight='balanced').fit(Tr,y[trn]);pr[te]=m.predict_proba(Te)[:,1]
    ok=np.isfinite(pr);return roc_auc_score(y[ok],pr[ok])
CV_ft=spcv([0,1,2,3]);CV_L2=spcv(None,withL2=True)
print("[C3 spCV] F+지형:%.3f +L2:%.3f (split +L2 %.3f)"%(CV_ft,CV_L2,AUC_L2))
def a80_region(acc):
    rows=np.array([[k,v[0],v[1]] for k,v in acc.items() if v[0]>0],float);n=rows[:,1];b=rows[:,2]
    if b.sum()==0:return np.nan
    o=np.argsort(-(b/np.maximum(n,1)));cn=np.cumsum(n[o])/n.sum();cb=np.cumsum(b[o])/b.sum();return cn[min(np.searchsorted(cb,0.8),len(cn)-1)]
def a80_lab(lab,yv):
    us=np.unique(lab);rows=np.array([[u,(lab==u).sum(),yv[lab==u].sum()] for u in us],float);rows=rows[rows[:,1]>0]
    n=rows[:,1];b=rows[:,2]
    if b.sum()==0:return np.nan
    o=np.argsort(-(b/np.maximum(n,1)));cn=np.cumsum(n[o])/n.sum();cb=np.cumsum(b[o])/b.sum();return cn[min(np.searchsorted(cb,0.8),len(cn)-1)]
k=len(set(r2[r2>0]))
kmA=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Xs)
kmG=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(np.column_stack([gx,gy]))
C_L2=a80_region(A2);C_L3=a80_region(A3);C_kmA=a80_lab(kmA,y);C_kmG=a80_lab(kmG,y)
print("[2025 A80] L2:%.1f L3:%.1f 비공간속성:%.1f 기하전용(C1):%.1f"%(100*C_L2,100*C_L3,100*C_kmA,100*C_kmG))
trn=~inuw
mT=LogisticRegression(max_iter=2000,class_weight='balanced').fit(Xs[trn][:,[0,1,2,3]],y[trn])
pu=mT.predict_proba(Xs[inuw][:,[0,1,2,3]])[:,1]
AUC_transfer=roc_auc_score(yu[inuw],pu) if yu[inuw].sum()>0 else float('nan')
def rate_map(acc):return {k:(v[1]/v[0] if v[0]>0 else 0) for k,v in acc.items()}
rate2=rate_map(A2);rate3=rate_map(A3)
def a80_transfer(Uacc,rate):
    rows=[[rate.get(k,0),v[0],v[1]] for k,v in Uacc.items() if v[0]>0];rows=np.array(rows,float)
    o=np.argsort(-rows[:,0]);n=rows[o,1];b=rows[o,2]
    if b.sum()==0:return np.nan
    cn=np.cumsum(n)/n.sum();cb=np.cumsum(b)/b.sum();return cn[min(np.searchsorted(cb,0.8),len(cn)-1)]
U_L2=a80_transfer(U2,rate2);U_L3=a80_transfer(U3,rate3)
def transfer_lab(lab,yv,yuv,restrict):
    us=np.unique(lab);rows=[]
    for u in us:
        m=lab==u;rate=yv[m].sum()/max(m.sum(),1);mu=m&restrict
        if mu.sum()>0:rows.append([rate,mu.sum(),yuv[mu].sum()])
    rows=np.array(rows,float);o=np.argsort(-rows[:,0]);n=rows[o,1];b=rows[o,2]
    if b.sum()==0:return np.nan
    cn=np.cumsum(n)/n.sum();cb=np.cumsum(b)/b.sum();return cn[min(np.searchsorted(cb,0.8),len(cn)-1)]
U_kmG=transfer_lab(kmG,y,yu,inuw);U_kmA=transfer_lab(kmA,y,yu,inuw)
print("[C2 transfer AUC] 2025->uljin: %.3f"%AUC_transfer)
print("[C2 uljin A80 by2025rate] L2:%.1f L3:%.1f 기하:%.1f 비공간:%.1f"%(100*U_L2,100*U_L3,100*U_kmG,100*U_kmA))
np.savez(f"{TMP}/review_final.npz",AUC_sp=AUC_sp,AUC_ft=AUC_ft,AUC_L1=AUC_L1,AUC_L2=AUC_L2,CV_ft=CV_ft,CV_L2=CV_L2,
    C_L2=C_L2,C_L3=C_L3,C_kmA=C_kmA,C_kmG=C_kmG,AUC_transfer=AUC_transfer,U_L2=U_L2,U_L3=U_L3,U_kmG=U_kmG,U_kmA=U_kmA,burnpct=100*tb/tn)
print("saved review_final.npz")
