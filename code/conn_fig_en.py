import os,numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage as ndi
_fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_fp);plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name();plt.rcParams['axes.unicode_minus']=False
P=os.path.expanduser("~/p3b")
d=np.load(f'{P}/_conn_tmp.npz')
flam=d['flam']; mask=d['mask']; gl2=d['gl2']; burnfrac=d['burnfrac']
queen=np.ones((3,3),int)
fig,ax=plt.subplots(1,3,figsize=(15,7.4))
fl=np.where(mask,flam,np.nan)
im=ax[0].imshow(fl,cmap='YlOrRd',vmin=40,vmax=100,aspect='equal')
ax[0].set_title('(a) Species flammability (600 m blocks, 2019)',fontsize=14,pad=6)
plt.colorbar(im,cax=make_axes_locatable(ax[0]).append_axes("right",size="4.5%",pad=0.05),label='flammability')
occ=(flam>=65)&mask; lbl,n=ndi.label(occ,queen)
sizes=np.bincount(lbl.ravel()); giant=int(np.argmax(sizes[1:]))+1
rgb=np.ones(mask.shape+(3,))
rgb[mask]=[0.85,0.85,0.85]; rgb[occ]=[1.0,0.75,0.4]; rgb[lbl==giant]=[0.80,0.15,0.1]
burned=(burnfrac>0.1)&mask; rgb[burned]=[0.1,0.2,0.9]
ax[1].imshow(rgb,aspect='equal')
ax[1].set_title('(b) Giant high-flammability cluster (\u226565, red)\n+ 2025 burned (blue); S=0.42, 10,635 km\u00b2',fontsize=14,pad=6)
reg=np.where(mask,gl2.astype(float),np.nan)
ax[2].imshow(reg,cmap='tab20',aspect='equal')
hi=(flam>=80)&mask; ov=np.zeros(mask.shape+(4,)); ov[hi]=[0,0,0,1]
ax[2].imshow(ov,aspect='equal')
ax[2].set_title('(c) L2 regions (colors) + \u226580 clusters (black)\n95% nest in one region; giant crosses boundaries',fontsize=14,pad=6)
for a in ax: a.set_axis_off()
plt.tight_layout()
outp=os.path.expanduser('~/mnt/\ub300\ud615\uc0b0\ubd88_FVI/\ub17c\ubb383/connectivity_preview_en.png')
plt.savefig(outp,dpi=300,bbox_inches='tight',facecolor='white')
print('saved',outp)