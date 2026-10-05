import os,numpy as np,rasterio,geopandas as gpd,matplotlib
matplotlib.use("Agg");import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from rasterio.enums import Resampling
from rasterio.warp import reproject,Resampling as RS
from rasterio.transform import from_origin
from shapely.geometry import box
_fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_fp);plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name();plt.rcParams['axes.unicode_minus']=False
TMP=os.path.expanduser("~/p3b");CO=os.path.expanduser("~/mnt/대형산불_FVI/Claude outputs");DATA=os.path.expanduser("~/mnt/Data/Source/1_boundary")
E0,N0,R,Wd,Hd=221910.0,323280.0,30.0,7185,9735;N1=N0+Hd*R;TR=from_origin(E0,N1,R,R);CRS="EPSG:5186";F=6;hw=Wd//F;hh=Hd//F
def rd(p,cat=True):
    with rasterio.open(p) as d: a=d.read(1,out_shape=(Hd//F,Wd//F),resampling=Resampling.nearest if cat else Resampling.average).astype('float32'); nod=d.nodata
    return np.where(a==nod,np.nan,a) if nod is not None else a
forest=rd(f"{TMP}/forest3p.tif",True)>0
TRc=from_origin(E0,N1,R*F,R*F);burn=np.zeros((hh,hw),'float32')
with rasterio.open(f"{CO}/FVI/reproduction/results/rasters/burn_perimeter.tif") as bs:
    reproject(rasterio.band(bs,1),burn,dst_transform=TRc,dst_crs=CRS,src_transform=bs.transform,src_crs=bs.crs,resampling=RS.max,src_nodata=bs.nodata,dst_nodata=0)
burn=(burn==1)&forest
import pandas as pd
from rasterio.features import shapes as _rshapes
from affine import Affine as _Aff
from shapely.geometry import shape as _shape
from shapely.ops import unary_union as _uu
_d=np.load(f"{TMP}/hier4_2019.npz");_bid=_d['bid'];_wb=int(_d['wb']);_hb=int(_d['hb']);_B=int(_d['B']);_l1=_d['l1']
_rr=_bid//_wb;_cc=_bid%_wb
_s1=pd.Series(_l1).value_counts();_c1={v:str(int(v)) for v in np.unique(_l1)}
_bl1=np.zeros((_hb,_wb),'int32');_bl1[_rr,_cc]=_l1
_P1={}
for _g,_v in _rshapes(_bl1,mask=_bl1>0,transform=_Aff.identity(),connectivity=4): _P1.setdefault(int(_v),[]).append(_shape(_g))
_P1={v:_uu(x).buffer(0).simplify(0.8) for v,x in _P1.items()}
_SC=_B/F
def drawL1(ax):
    for v,g in _P1.items():
        for p in (g.geoms if g.geom_type=="MultiPolygon" else [g]):
            if p.area*(_B*_B)*0.09<500: continue
            xs,ys=p.exterior.xy;ax.plot(np.array(xs)*_SC,np.array(ys)*_SC,color="k",lw=2.0,solid_capstyle="round")
    for v in np.unique(_l1):
        g=_P1[v];big=max((g.geoms if g.geom_type=="MultiPolygon" else [g]),key=lambda p:p.area);pt=big.representative_point()
        ax.text(pt.x*_SC,pt.y*_SC,_c1[v],ha="center",va="center",fontsize=17,fontweight="bold",color="#b00020",bbox=dict(boxstyle="circle,pad=0.3",fc="white",ec="#b00020",lw=1.4,alpha=0.95))
sig=gpd.read_file(f"{DATA}/(센서스경계)시군구경계/Z_SOP_BND_SIGUNGU_PG/Z_SOP_BND_SIGUNGU_PG.shp").set_crs(5181,allow_override=True).to_crs(5186)
sido=gpd.read_file(f"{DATA}/(센서스경계)시도경계/Z_SOP_BND_SIDO_PG/Z_SOP_BND_SIDO_PG.shp").set_crs(5181,allow_override=True).to_crs(5186)
gbox=box(E0,N0,E0+Wd*R,N1)
def padm(ax,gdf,lw,col,al):
    for g in gdf.geometry:
        gg=g.intersection(gbox)
        for p in (gg.geoms if gg.geom_type in ('MultiPolygon','GeometryCollection') else [gg]):
            if p.is_empty or p.geom_type!='Polygon':continue
            x,y=p.exterior.xy;ax.plot((np.array(x)-E0)/(R*F),(N1-np.array(y))/(R*F),color=col,lw=lw,alpha=al)
ext=(0,hw,hh,0)
fig=plt.figure(figsize=(12.5,7))
ax=fig.add_subplot(1,2,1)
ax.imshow(np.where(forest,0.86,np.nan),cmap=ListedColormap([(0.86,0.86,0.86,1)]),extent=ext,vmin=0,vmax=1,aspect='equal')
ax.imshow(np.where(burn,1,np.nan),cmap=ListedColormap([(0.82,0.07,0.09,1)]),extent=ext,aspect='equal',interpolation='nearest')
padm(ax,sig,0.3,'#bbb',0.5);padm(ax,sido,0.8,'#aaa',0.7);drawL1(ax)
ax.set_xlim(0,hw);ax.set_ylim(hh,0);ax.axis('off')
ax.legend(handles=[Line2D([],[],marker='s',ls='',mfc='#d11212',mec='none',ms=11,label='2025 burn scar'),Line2D([],[],marker='s',ls='',mfc='#dbdbdb',mec='none',ms=11,label='Forest'),Line2D([],[],color='k',lw=2.0,label='Macro-region (L1) boundary')],loc='upper right',fontsize=12)
ax.set_title("(a) 2025 burn distribution with macro-region (L1) boundaries",fontsize=15)
axb=fig.add_subplot(1,2,2)
nm=['1','2','3','4','5'];area=[41489,398895,576564,347238,423497];pct=[0.00,10.70,0.11,0.00,3.42]
bha=[a*p/100 for a,p in zip(area,pct)]
cols=['#cccccc','#cc1a1f','#cccccc','#cccccc','#cc1a1f']
bars=axb.bar(nm,bha,color=cols,edgecolor='#333',lw=0.6)
for r,v,p in zip(bars,bha,pct): axb.text(r.get_x()+r.get_width()/2,v+500,f'{v:,.0f} ha\n({p:.2f}%)',ha='center',fontsize=11)
axb.set_ylabel('2025 forest burned area (ha)',fontsize=12);axb.set_ylim(0,48000);axb.set_title("(b) Burned area and burn rate by macro-region",fontsize=15);axb.grid(axis='y',alpha=0.3)
plt.tight_layout();plt.savefig(f"{TMP}/fig3_burn4_en.png",dpi=250,facecolor='white',bbox_inches='tight');print("burn4 saved")
