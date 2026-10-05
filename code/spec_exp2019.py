# -*- coding: utf-8 -*-
import os,sys,time,json,numpy as np,pyogrio,zipfile
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from rasterio.transform import from_origin
D=os.path.expanduser("~/mnt/대형산불_FVI/data");TMP=os.path.expanduser("~/p3b")
E0,N1,R,W,H=221910.0,615330.0,30.0,7185,9735; GT=from_origin(E0,N1,R,R)
SPEC={'11':100,'15':100,'14':95,'12':82,'13':68,'77':65,'30':45,'31':45,'32':45,'33':45,'34':45,'37':42,'45':38,'46':42,'49':42,'81':20,'82':10,'83':55}
def sscore(code):
    code=str(code)
    if code in SPEC: return SPEC[code]
    return {'1':90,'3':45,'7':65,'4':42,'8':20}.get(code[:1],50)
def shp(z): return "/vsizip/"+os.path.abspath(f"{D}/{z}")+"/"+[x for x in zipfile.ZipFile(f"{D}/{z}").namelist() if x.endswith('.shp')][0]
SRC=[shp("47 (5).zip"),shp("27 (6).zip"),shp("43 (7).zip")]   # 2019판 경북·대구·충북
spp=f"{TMP}/specY2019.npy"; prog=f"{TMP}/specY2019_prog.json"
spec=np.load(spp) if os.path.exists(spp) else np.full((H,W),np.nan,'float32')
st=json.load(open(prog)) if os.path.exists(prog) else {"s":0,"off":0}
CH=30000; t0=time.time(); si=st["s"]; off=st["off"]
while si<len(SRC):
    p=SRC[si]; tot=pyogrio.read_info(p)['features']
    while off<tot:
        g=pyogrio.read_dataframe(p,columns=["KOFTR_GROU"],skip_features=off,max_features=CH)
        if len(g)==0: break
        g=g.to_crs("EPSG:5186")
        sc=g["KOFTR_GROU"].astype(str).map(sscore).astype('float32')
        shapes=[(geom,float(v)) for geom,v in zip(g.geometry,sc) if geom is not None and np.isfinite(v)]
        if shapes:
            r=rasterize(shapes,out_shape=(H,W),transform=GT,fill=np.nan,dtype='float32',merge_alg=MergeAlg.replace)
            m=np.isfinite(r); spec[m]=r[m]
        off+=CH
        if time.time()-t0>100:
            np.save(spp,spec);json.dump({"s":si,"off":off},open(prog,"w"))
            print(f"ckpt src{si} off{off}/{tot}");sys.exit(0)
    si+=1; off=0; np.save(spp,spec);json.dump({"s":si,"off":0},open(prog,"w"));print(f"done src{si}/{len(SRC)}")
np.save(spp,spec);json.dump({"s":len(SRC),"off":0},open(prog,"w"))
print("ALL DONE, finite px",int(np.isfinite(spec).sum()))
