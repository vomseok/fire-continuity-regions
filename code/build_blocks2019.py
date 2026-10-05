import os,numpy as np,rasterio
from rasterio.warp import reproject,Resampling as RS
from rasterio.transform import from_origin
TMP=os.path.expanduser("~/p3b");CO=os.path.expanduser("~/mnt/대형산불_FVI/Claude outputs")
H=np.load(f"{TMP}/hier4_2019.npz");BID=H['bid'];wb=int(H['wb']);hb=int(H['hb']);B=int(H['B'])
E0,N0,R,Wd,Hd=221910.0,323280.0,30.0,7185,9735;N1=N0+Hd*R;TR=from_origin(E0,N1,R,R)
F={'spec':f"{TMP}/spec3_2019.tif",'elev':f"{TMP}/exp3_elev.tif",'slope':f"{TMP}/exp3_slope.tif",'north':f"{TMP}/exp3_north.tif"}
ds={k:rasterio.open(p) for k,p in F.items()};dfor=rasterio.open(f"{TMP}/forest3p.tif")
with rasterio.open(f"{CO}/FVI/reproduction/results/rasters/burn_perimeter.tif") as bs:
    burnr=np.zeros((Hd,Wd),'float32');reproject(rasterio.band(bs,1),burnr,dst_transform=TR,dst_crs="EPSG:5186",src_transform=bs.transform,src_crs=bs.crs,resampling=RS.nearest,src_nodata=bs.nodata,dst_nodata=0)
burnr=(burnr==1)
nblk=hb*wb;names=['spec','elev','slope','north']
ssum=np.zeros((nblk,4));scnt=np.zeros(nblk);bsum=np.zeros(nblk)
gsum=np.zeros(4);gsq=np.zeros(4);gn=0.0;STRIP=900
for r0 in range(0,Hd,STRIP):
    r1=min(r0+STRIP,Hd);win=rasterio.windows.Window(0,r0,Wd,r1-r0)
    fo=dfor.read(1,window=win)>0
    if not fo.any():continue
    feats=[];ok=fo.copy()
    for k in names:
        a=ds[k].read(1,window=win).astype('float32')
        if k=='spec':a=np.where(a<0,np.nan,a)
        feats.append(a);ok&=np.isfinite(a)
    yy,xx=np.where(ok)
    if len(yy)==0:continue
    gr=(r0+yy)//B;gc=xx//B;bflat=gr*wb+gc
    vals=np.column_stack([feats[j][yy,xx] for j in range(4)]).astype('float64')
    for j in range(4):
        np.add.at(ssum[:,j],bflat,vals[:,j]);gsum[j]+=vals[:,j].sum();gsq[j]+=(vals[:,j]**2).sum()
    np.add.at(scnt,bflat,1.0);gn+=len(yy)
    bv=burnr[r0:r1,:][yy,xx].astype('float64');np.add.at(bsum,bflat,bv)
pm=gsum/gn;pstd=np.sqrt(gsq/gn-pm**2)
cnt=scnt[BID];bf=(ssum[BID]/cnt[:,None]-pm)/pstd;burn=bsum[BID]
np.savez(f"{TMP}/blocks_av4_2019.npz",bid=BID,bf=bf,cnt=cnt,burn=burn,pm=pm,pstd=pstd,wb=wb,hb=hb,B=B,H=Hd,W=Wd)
print("saved blocks_av4_2019.npz  blocks",len(BID),"burn blocks",int((burn>0).sum()),"pm",np.round(pm,2))
