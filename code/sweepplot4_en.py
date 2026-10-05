import os,numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg");import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_fp);plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name();plt.rcParams['axes.unicode_minus']=False
TMP=os.path.expanduser("~/p3b");df=pd.read_csv(f"{TMP}/sweep4.csv").sort_values('W')
fig,ax=plt.subplots(figsize=(8.5,5.6))
ax.plot(df.W,df.VE_L2,'o-',color='#2c7fb8',lw=2,label='Internal homogeneity (VE, L2) ↑ better')
ax.plot(df.W,df.compact_L1,'s-',color='#d95f0e',lw=2,label='Spatial cohesion (fill-ratio, L1) ↑ better')
ax.plot(df.W,1-df.area80_L2,'^-',color='#31a354',lw=2,label='Burn concentration (1−area@80%, L2) ↑ better')
ax.axvline(1.3,color='#b00020',ls='--',lw=1.5);ax.text(1.32,0.02,'Adopted W=1.3',color='#b00020',fontsize=12)
ax.set_xlabel('Spatial weight W_SPACE (0 = attributes only … 2 = spatial emphasis)',fontsize=12);ax.set_ylabel('Index (0–1, higher = better)',fontsize=12)
ax.set_ylim(0,1);ax.grid(alpha=0.3);ax.legend(loc='upper right',fontsize=11)
ax.set_title('W_SPACE sensitivity: homogeneity vs spatial cohesion vs burn concentration',fontsize=13)
plt.tight_layout();plt.savefig(f"{TMP}/sweep4_en.png",dpi=300,facecolor='white');print("saved")
print(df.to_string(index=False))
