import os,numpy as np,rasterio,geopandas as gpd,matplotlib
matplotlib.use("Agg");import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from shapely.geometry import box
_fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_fp);plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name();plt.rcParams['axes.unicode_minus']=False
TMP=os.path.expanduser("~/p3b");DATA=os.path.expanduser("~/mnt/Data/Source/1_boundary")
E0,N0,R,Wd,Hd=221910.0,323280.0,30.0,7185,9735;N1=N0+Hd*R;F=6;hw=Wd//F;hh=Hd//F;ext=(0,hw,hh,0)
def rd(p,cat=True):
    with rasterio.open(p) as d: a=d.read(1,out_shape=(Hd//F,Wd//F),resampling=Resampling.nearest if cat else Resampling.average).astype('float32'); nod=d.nodata
    return np.where(a==nod,np.nan,a) if nod is not None else a
forest=rd(f"{TMP}/forest3p.tif",True)>0
fuel=rd(f"{TMP}/fuel3.tif",True); spec=rd(f"{TMP}/spec3_2019.tif",False)
elev=rd(f"{TMP}/exp3_elev.tif",False);slope=rd(f"{TMP}/exp3_slope.tif",False)
spec=np.where(forest&(spec>=0),spec,np.nan);elev=np.where(forest,elev,np.nan);slope=np.where(forest,slope,np.nan)
# burn
from rasterio.warp import reproject,Resampling as RS
TRc=from_origin(E0,N1,R*F,R*F);burn=np.zeros((hh,hw),'float32')
with rasterio.open(os.path.expanduser("~/mnt/대형산불_FVI/Claude outputs/FVI/reproduction/results/rasters/burn_perimeter.tif")) as bs:
    reproject(rasterio.band(bs,1),burn,dst_transform=TRc,dst_crs="EPSG:5186",src_transform=bs.transform,src_crs=bs.crs,resampling=RS.max,src_nodata=bs.nodata,dst_nodata=0)
burn=(burn==1)&forest
sig=gpd.read_file(f"{DATA}/(센서스경계)시군구경계/Z_SOP_BND_SIGUNGU_PG/Z_SOP_BND_SIGUNGU_PG.shp").set_crs(5181,allow_override=True).to_crs(5186)
sido=gpd.read_file(f"{DATA}/(센서스경계)시도경계/Z_SOP_BND_SIDO_PG/Z_SOP_BND_SIDO_PG.shp").set_crs(5181,allow_override=True).to_crs(5186)
gbox=box(E0,N0,E0+Wd*R,N1)
def padm(ax,gdf,lw,col,al):
    for g in gdf.geometry:
        gg=g.intersection(gbox)
        for p in (gg.geoms if gg.geom_type in ('MultiPolygon','GeometryCollection') else [gg]):
            if p.is_empty or p.geom_type!='Polygon':continue
            x,y=p.exterior.xy;ax.plot((np.array(x)-E0)/(R*F),(N1-np.array(y))/(R*F),color=col,lw=lw,alpha=al)
# ---- FIG1 ----
fig=plt.figure(figsize=(13,7.6))
axk=fig.add_subplot(1,2,1)
ne=gpd.read_file(os.path.expanduser("~/mnt/대형산불_FVI/논문3/ne/ne_50m_admin_0_countries.shp")).to_crs(4326)
s4=sido.to_crs(4326);ST={'37','22','33'}
ne.plot(ax=axk,facecolor='#eceae4',edgecolor='#9a9a9a',lw=0.5)
ys0,xs0=np.where(forest)
bx0=E0+xs0.min()*R*F;bx1=E0+(xs0.max()+1)*R*F;by1=N1-ys0.min()*R*F;by0=N1-(xs0*0+ys0.max()+1).max()*R*F
bb=gpd.GeoDataFrame(geometry=[box(bx0,by0,bx1,by1)],crs=5186).to_crs(4326)
mnx,mny,mxx,mxy=bb.total_bounds
axk.add_patch(Rectangle((mnx,mny),mxx-mnx,mxy-mny,fill=False,ec='#b00020',lw=1.8,zorder=6))
cx=(mnx+mxx)/2;cy=(mny+mxy)/2
for nm,lo,la in [("China",cx-4.2,cy-0.4),("N. Korea",cx-1.4,cy+3.6),("Japan",cx+3.6,cy-3.0),("East Sea",cx+3.3,cy+1.4),("Yellow Sea",cx-3.6,cy-2.2)]:
    axk.text(lo,la,nm,fontsize=11.5,ha='center',color='#4a5570',style='italic')
axk.set_facecolor('#d7e6f0');axk.set_xlim(cx-5.0,cx+5.0);axk.set_ylim(cy-5.4,cy+5.4);axk.set_aspect(1/np.cos(np.radians(cy)));axk.set_xticks([]);axk.set_yticks([])
axk.set_title("(a) Study area location (Northeast Asia)",fontsize=15)
axf=fig.add_subplot(1,2,2)
axf.imshow(np.where(forest,1,np.nan),cmap=ListedColormap([(0.82,0.88,0.80,1)]),extent=ext,aspect='equal')
axf.imshow(np.where(burn,1,np.nan),cmap=ListedColormap([(0.82,0.07,0.09,1)]),extent=ext,aspect='equal',interpolation='nearest')

# --- 2022 울진 산불 오버레이 + 발생시점 라벨 ---
from scipy import ndimage as _ndi
_us=np.load(f"{TMP}/uljin_spring.npz");_keep=_us['burn'];_co,_ro=5309,2840
_lab,_n=_ndi.label(_keep);_sz=_ndi.sum(np.ones_like(_lab),_lab,range(1,_n+1))
_big=np.isin(_lab,1+np.where(_sz*0.09>=100)[0])
_full=np.zeros((Hd,Wd),'uint8');_h2,_w2=_big.shape;_full[_ro:_ro+_h2,_co:_co+_w2]=_big.astype('uint8')
_fc=_full[:hh*F,:hw*F].reshape(hh,F,hw,F).max(axis=(1,3))
axf.imshow(np.where(_fc>0,1,np.nan),extent=ext,aspect='equal',cmap=ListedColormap([(0.95,0.55,0.0,1)]),interpolation='nearest')
_uy,_ux=np.where(_fc>0)
if len(_uy):
    axf.annotate("Mar 2022 Uljin",xy=(_ux.mean(),_uy.mean()),xytext=(min(_ux.mean()+95,hw-70),_uy.mean()-55),fontsize=9,fontweight="bold",color="#8a4b00",ha="left",arrowprops=dict(arrowstyle="-",color="#8a4b00",lw=0.9),bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="#c77700",lw=0.9,alpha=0.95),zorder=10)
_by,_bx=np.where(burn)
if len(_by):
    axf.annotate("Mar 2025 Uiseong-Andong",xy=(_bx.mean(),_by.mean()),xytext=(_bx.mean()-30,_by.mean()+135),fontsize=9,fontweight="bold",color="#b00020",ha="center",arrowprops=dict(arrowstyle="-",color="#b00020",lw=0.9),bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="#d11212",lw=0.9,alpha=0.95),zorder=10)

padm(axf,sig,0.5,'#8a8a8a',0.8);padm(axf,sido,1.6,'#000',0.9)
def dec(x):
    try:return str(x).encode('latin-1','ignore').decode('euc-kr','ignore')
    except:return str(x)
def px(x,y):return (x-E0)/(R*F),(N1-y)/(R*F)
ST2={'37','22','33'}
sigc=sig.copy();sigc['cx']=sigc.geometry.centroid.x;sigc['cy']=sigc.geometry.centroid.y
for _,r in sigc.iterrows():
    cd=str(r['SIGUNGU_CD'])
    if cd[:2] not in ST2:continue
    c,rw=px(r['cx'],r['cy'])
    if c<0 or c>hw or rw<0 or rw>hh:continue
    nm=dec(r['SIGUNGU_NM']);nm=nm.replace('경상북도','').replace('충청북도','').replace('대구광역시','').strip()
    pass
for _,r in sido[sido['SIDO_CD'].astype(str).isin(ST2)].iterrows():
    c=r.geometry.centroid;cc,rw=px(c.x,c.y);nm={'22':'Daegu','33':'Chungbuk','37':'Gyeongbuk'}.get(str(r['SIDO_CD']),'')
    axf.text(cc,rw,nm,fontsize=16,ha='center',va='center',color='#1f3864',fontweight='bold',alpha=0.35)
axf.set_xlim(0,hw);axf.set_ylim(hh,0);axf.axis('off')
axf.legend(handles=[Line2D([],[],marker='s',ls='',mfc='#d11212',mec='none',ms=11,label='2025 fire (Uiseong-Andong)'),Line2D([],[],marker='s',ls='',mfc=(0.95,0.55,0.0),mec='none',ms=11,label='2022 fire (Uljin)'),Line2D([],[],marker='s',ls='',mfc=(0.82,0.88,0.80),mec='none',ms=11,label='Forest')],loc='lower left',fontsize=12,framealpha=0.9)
axf.set_title("(b) Administrative units and burn scars (2025 Uiseong-Andong; 2022 Uljin)",fontsize=15)
plt.tight_layout();plt.savefig(f"{TMP}/fig1_studyarea4_en.png",dpi=220,facecolor='white',bbox_inches='tight');plt.close();print("fig1 ok")
# ---- FIG2 inputs (2x2) ----
north=rd(f"{TMP}/exp3_north.tif",False);north=np.where(forest,north,np.nan)
fy,fx=np.where(forest);rr0,rr1,cc0,cc1=fy.min(),fy.max(),fx.min(),fx.max();pd0=6
def cropax(ax):ax.set_xlim(cc0-pd0,cc1+pd0);ax.set_ylim(rr1+pd0,rr0-pd0);ax.axis("off")
fig,axs=plt.subplots(2,2,figsize=(11,10.2))
specs=[(spec,"YlOrRd",0,100,"(a) Species flammability (0\u2013100)"),(elev,"terrain",None,None,"(b) Elevation (m)"),(slope,"YlOrBr",0,40,"(c) Slope (\u00b0)"),(north,"coolwarm_r",-1,1,"(d) Northness (north +1 / south \u22121)")]
for ax,(arr,cm,vmn,vmx,ttl) in zip(axs.ravel(),specs):
    im=ax.imshow(arr,cmap=cm,vmin=vmn,vmax=vmx,extent=ext,aspect="equal");padm(ax,sido,0.8,"#000",0.6)
    cropax(ax);ax.set_title(ttl,fontsize=15);plt.colorbar(im,ax=ax,fraction=0.045,pad=0.02)
plt.tight_layout();plt.savefig(f"{TMP}/fig2_inputs4_en.png",dpi=220,facecolor="white",bbox_inches="tight");plt.close();print("fig2 ok")
