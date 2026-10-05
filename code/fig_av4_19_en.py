import os,numpy as np,matplotlib
matplotlib.use("Agg");import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_fp);plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name();plt.rcParams['axes.unicode_minus']=False;plt.rcParams['hatch.linewidth']=0.8
TMP=os.path.expanduser("~/p3b")
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.6))
m=['Species','Species\n+Terrain','Species+Terrain\n+L1','Species+Terrain\n+L2'];v=[0.670,0.681,0.872,0.917]
cols=['0.9','0.9','0.9','0.9'];hatch1=['','//','xx','..']
b=ax1.bar(m,v,color=cols,edgecolor='black',lw=0.9)
for _bar,_h in zip(b,hatch1):_bar.set_hatch(_h)
for r,val in zip(b,v): ax1.text(r.get_x()+r.get_width()/2,val+0.008,f'{val:.3f}',ha='center',fontsize=12,fontweight='bold')
ax1.set_ylim(0.5,1.0);ax1.axhline(0.5,color='#999',ls=':',lw=1);ax1.set_ylabel('2025 burn discrimination (test AUC)',fontsize=12)
ax1.set_title('(a) Pixel discrimination: information added by regions',fontsize=13)
m2=['Geometry-only\n(coords only)','Non-spatial\nk-means\n(species+terrain)','Regions\nL2','Regions\nL3'];v2=[12.9,40.0,27.1,19.5]
# represent 수종F alone concentration as 100 baseline (few discrete values) -> use 100
v2[0]=100.0
cols2=['0.9','0.9','0.9','0.9'];hatch2=['','//','xx','..']
b2=ax2.bar(m2,v2,color=cols2,edgecolor='black',lw=0.9)
for _bar,_h in zip(b2,hatch2):_bar.set_hatch(_h)
for r,val in zip(b2,v2): ax2.text(r.get_x()+r.get_width()/2,val+1.5,f'{val:.0f}%',ha='center',fontsize=12,fontweight='bold')
ax2.set_ylim(0,108);ax2.set_ylabel('Area containing 80% of burning (%)  (lower = better)',fontsize=12)
ax2.set_title('(b) Burn concentration: effect of spatial constraint',fontsize=13)
plt.tight_layout();plt.savefig(f"{TMP}/fig_addedvalue4_en.png",dpi=300,facecolor='white',bbox_inches='tight');print("av4 saved")
