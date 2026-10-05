import numpy as np, json
from scipy import ndimage as ndi
b=np.load('blocks_av4_2019.npz'); h=np.load('hier4_2019.npz')
bid=b['bid']; bf=b['bf']; cnt=b['cnt']; burn=b['burn']; pm=b['pm']; pstd=b['pstd']
wb=int(b['wb']); hb=int(b['hb'])
l2=h['l2']; l3=h['l3']; l1=h['l1']
raw=bf[:,0]*pstd[0]+pm[0]
rr=bid//wb; cc=bid%wb
# crop to forest bounding box to save memory/time
r0,r1=rr.min(),rr.max()+1; c0,c1=cc.min(),cc.max()+1
H=r1-r0; Wd=c1-c0
def grid(vals,fill=np.nan,dt=float):
    g=np.full((H,Wd),fill,dt); g[rr-r0,cc-c0]=vals; return g
flam=grid(raw); mask=grid(1.0,0.0); mask=mask>0
gl2=grid(l2,-1,int); gl3=grid(l3,-1,int)
gburn=grid(burn,0.0); gcnt=grid(cnt,0.0)
burnfrac=np.where(gcnt>0,np.clip(gburn/np.maximum(gcnt,1),0,1),0.0)
Ntot=int(mask.sum())
BLK_KM2=0.36  # 600x600 m
rook=np.array([[0,1,0],[1,1,1],[0,1,0]]); queen=np.ones((3,3),int)

def spanning(lbl,comp_id,m):
    ys,xs=np.where(lbl==comp_id)
    topbot = (ys.min()==0 and ys.max()==m.shape[0]-1)
    lr = (xs.min()==0 and xs.max()==m.shape[1]-1)
    # relative to forest extent bbox
    return bool(topbot or lr)

out={'meta':{'n_forest_blocks':Ntot,'block_km2':BLK_KM2,'grid':[H,Wd],'wb':wb,'hb':hb,
      'flam_pctile':{str(q):float(np.percentile(raw,q)) for q in [50,65,75,80,90]}}}

# 1 & 2: threshold sweep for connectivity + percolation
ths=[90,85,80,75,70,65,60,55,50,45,40]
sweep=[]
giant_at={}
for t in ths:
    occ=(flam>=t)&mask
    p=occ.sum()/Ntot
    res={'th':t,'p':round(float(p),4)}
    for name,st in [('rook',rook),('queen',queen)]:
        lbl,n=ndi.label(occ,structure=st)
        if n==0:
            res[name]={'n_comp':0,'max_blocks':0,'max_km2':0.0,'S':0.0,'spanning':False}
            continue
        sizes=np.bincount(lbl.ravel())[1:]
        imax=int(np.argmax(sizes))+1
        mx=int(sizes.max())
        span=spanning(lbl,imax,occ)
        res[name]={'n_comp':int(n),'max_blocks':mx,'max_km2':round(mx*BLK_KM2,2),
                   'S':round(float(mx/Ntot),4),'mean_size':round(float(sizes.mean()),2),
                   'top5':[int(x) for x in np.sort(sizes)[::-1][:5]],'spanning':span}
        if name=='queen':
            giant_at[t]=(lbl,imax)
    sweep.append(res)
out['sweep']=sweep

# percolation p_c: first threshold (descending) where queen giant spans
pc=None; pc_p=None
for res in sweep:
    if res['queen']['spanning']:
        pc=res['th']; pc_p=res['p']; break
out['percolation']={'p_c_threshold':(int(pc) if pc else None),'p_c_occupied_fraction':(float(pc_p) if pc_p else None),
   'note':'p_c = occupied fraction at first appearance of a spanning (border-to-border) queen cluster as threshold is lowered'}

# correlation length (second-moment, queen) near/above p_c, excluding spanning cluster
def corr_len(occ):
    lbl,n=ndi.label(occ,queen)
    if n==0: return 0.0,0
    sizes=np.bincount(lbl.ravel())[1:]
    # identify spanning clusters to exclude
    span_ids=set()
    ys0,xs0=0,0
    # radius of gyration per cluster via center of mass
    coms=ndi.center_of_mass(occ,lbl,range(1,n+1))
    num=0.0; den=0.0
    for i in range(1,n+1):
        if spanning(lbl,i,occ): continue
        ys,xs=np.where(lbl==i)
        s=len(ys)
        if s<2: continue
        cy,cx=coms[i-1]
        Rg2=((ys-cy)**2+(xs-cx)**2).mean()  # in block units
        num+=Rg2*s*s; den+=s*s
    if den==0: return 0.0,0
    xi_blocks=np.sqrt(2*num/den)
    return xi_blocks, n
cl={}
for t in [int(pc) if pc else 65, (int(pc)+5 if pc else 70), (int(pc)-5 if pc else 60)]:
    if t is None or t<40 or t>90: continue
    occ=(flam>=t)&mask
    xi,_=corr_len(occ)
    cl[str(t)]={'xi_blocks':round(float(xi),2),'xi_km':round(float(xi*0.6),2)}
out['correlation_length']=cl

np.savez('_conn_tmp.npz',flam=flam,mask=mask,gl2=gl2,gl3=gl3,burnfrac=burnfrac,gburn=gburn,gcnt=gcnt)
print(json.dumps(out,ensure_ascii=False,indent=1,default=lambda o:int(o) if hasattr(o,'is_integer') or 'int' in type(o).__name__ else float(o)))
import pickle; pickle.dump(out,open('_conn_part1.pkl','wb'))
