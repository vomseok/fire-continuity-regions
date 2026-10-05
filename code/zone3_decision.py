# -*- coding: utf-8 -*-
import numpy as np, json
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

h=np.load('hier4_2019.npz'); b=np.load('blocks_av4_2019.npz')
bid=h['bid']; wb=int(h['wb']); hb=int(h['hb'])
l1,l2,l3=h['l1'],h['l2'],h['l3']
cnt=b['cnt'].astype(float); burn=b['burn'].astype(float)
bf=b['bf']; pm=b['pm']; pstd=b['pstd']
fuel = bf[:,0]*pstd[0]+pm[0]      # raw species flammability per block
nb=len(bid)
PXHA=0.09  # 30m px -> ha
tot_cnt=cnt.sum(); tot_burn=burn.sum()
overall_burn_rate = tot_burn/tot_cnt
overall_fuel = np.average(fuel,weights=cnt)
print("nb",nb,"overall burn rate %.4f"%overall_burn_rate,"overall fuel %.2f"%overall_fuel)

def zone_profile(lab):
    out=[]
    for z in sorted(set(lab.tolist())):
        m=lab==z
        c=cnt[m].sum(); br=burn[m].sum()
        out.append(dict(zone=int(z),
            nblocks=int(m.sum()),
            area_ha=round(c*PXHA,1),
            area_pct=round(100*c/tot_cnt,2),
            mean_fuel=round(float(np.average(fuel[m],weights=cnt[m])),2),
            fuel_sd=round(float(np.sqrt(np.average((fuel[m]-np.average(fuel[m],weights=cnt[m]))**2,weights=cnt[m]))),2),
            burn_rate_pct=round(100*br/c,3),
            burn_ha=round(br*PXHA,1)))
    return out

# ---- 1. L1 profile ----
L1prof=zone_profile(l1)
print("\n=== L1 PROFILE ===")
for r in L1prof: print(r)

# zone H = high fuel, ~0 burn. criterion: mean_fuel > overall_fuel and burn_rate < 0.5%
Hzones=[r['zone'] for r in L1prof if r['mean_fuel']>overall_fuel and r['burn_rate_pct']<0.5]
# fallback: pick highest-fuel zone among those with burn<0.5%
print("\nzone H (high fuel, ~0 burn):", Hzones)

# ---- 2. geometric vs attribute clustering at k=5,18 ----
brow=(bid//wb).astype(float); bcol=(bid%wb).astype(float)
coords=np.column_stack([(brow-brow.mean())/brow.std(),(bcol-bcol.mean())/bcol.std()])
# attribute non-spatial features = all 4 standardized attrs (includes fuel)
attr=bf.copy()

def cluster_quality(lab,name):
    """for each cluster: mean fuel, within fuel sd, burn rate; find best 'high-fuel low-burn' unit"""
    rows=[]
    for z in sorted(set(lab.tolist())):
        m=lab==z; c=cnt[m].sum()
        mf=float(np.average(fuel[m],weights=cnt[m]))
        sd=float(np.sqrt(np.average((fuel[m]-mf)**2,weights=cnt[m])))
        br=float(burn[m].sum()/c)
        rows.append((int(z),mf,sd,br*100,c*PXHA))
    rows.sort(key=lambda x:-x[1])  # by fuel desc
    # identify a unit that is high-fuel (top qu_fuel) AND low-burn (<0.5%)
    hf_thresh=overall_fuel
    hilo=[r for r in rows if r[1]>hf_thresh and r[3]<0.5]
    hilo_area=sum(r[4] for r in hilo)
    topfuel=rows[0]
    return dict(name=name,
        n_clusters=len(rows),
        mean_within_fuel_sd=round(float(np.mean([r[2] for r in rows])),2),
        top_fuel_cluster=dict(zone=topfuel[0],mean_fuel=round(topfuel[1],2),fuel_sd=round(topfuel[2],2),burn_rate_pct=round(topfuel[3],3),area_ha=round(topfuel[4],1)),
        n_highfuel_lowburn_units=len(hilo),
        highfuel_lowburn_area_ha=round(hilo_area,1),
        clusters=[dict(zone=r[0],mean_fuel=round(r[1],2),fuel_sd=round(r[2],2),burn_rate_pct=round(r[3],3),area_ha=round(r[4],1)) for r in rows])

step2={}
for k in (5,18):
    gl=KMeans(n_clusters=k,n_init=10,random_state=0).fit_predict(coords)
    al=KMeans(n_clusters=k,n_init=10,random_state=0).fit_predict(attr)
    geo=cluster_quality(gl,f'geometric_k{k}')
    att=cluster_quality(al,f'attribute_k{k}')
    # our zones: l1 for k~5, l2 for k~18
    ours=cluster_quality(l1 if k==5 else l2, f'ourzone_k{k}')
    step2[f'k{k}']=dict(geometric=geo,attribute=att,ourzone=ours)
    print(f"\n=== k={k} ===")
    print("geometric: mean within-fuel-sd %.2f | top-fuel cluster burn %.3f%% | highfuel-lowburn units %d (%.0f ha)"%(geo['mean_within_fuel_sd'],geo['top_fuel_cluster']['burn_rate_pct'],geo['n_highfuel_lowburn_units'],geo['highfuel_lowburn_area_ha']))
    print("attribute: mean within-fuel-sd %.2f | top-fuel cluster burn %.3f%% | highfuel-lowburn units %d (%.0f ha)"%(att['mean_within_fuel_sd'],att['top_fuel_cluster']['burn_rate_pct'],att['n_highfuel_lowburn_units'],att['highfuel_lowburn_area_ha']))
    print("ourzone  : mean within-fuel-sd %.2f | top-fuel cluster burn %.3f%% | highfuel-lowburn units %d (%.0f ha)"%(ours['mean_within_fuel_sd'],ours['top_fuel_cluster']['burn_rate_pct'],ours['n_highfuel_lowburn_units'],ours['highfuel_lowburn_area_ha']))

# ---- 3. decision change test: top-20% area priority under 3 schemes ----
K3=len(set(l3.tolist()))  # match granularity for geometric
geo55=KMeans(n_clusters=K3,n_init=10,random_state=0).fit_predict(coords)

def zone_mean_fuel_score(lab):
    score=np.zeros(nb)
    for z in set(lab.tolist()):
        m=lab==z
        score[m]=np.average(fuel[m],weights=cnt[m])
    return score

s_pixel = fuel.copy()                 # (i) block fuel
s_geo   = zone_mean_fuel_score(geo55)  # (ii) geometric zone mean
s_l3    = zone_mean_fuel_score(l3)     # (iii) our L3 zone mean

def top20_set(score):
    order=np.argsort(-score, kind='stable')
    cum=np.cumsum(cnt[order]); thr=0.20*tot_cnt
    k=np.searchsorted(cum,thr)+1
    sel=np.zeros(nb,bool); sel[order[:k]]=True
    return sel

sel_pixel=top20_set(s_pixel); sel_geo=top20_set(s_geo); sel_l3=top20_set(s_l3)

def jacc(a,b):
    # area-weighted Jaccard
    inter=cnt[a&b].sum(); uni=cnt[a|b].sum()
    return float(inter/uni)
def jacc_cnt(a,b):
    return float((a&b).sum()/(a|b).sum())

sp_pg=spearmanr(s_pixel,s_geo).correlation
sp_pl=spearmanr(s_pixel,s_l3).correlation
sp_gl=spearmanr(s_geo,s_l3).correlation

step3=dict(
  top20_area_target_ha=round(0.20*tot_cnt*PXHA,1),
  selected_area_ha=dict(pixel=round(cnt[sel_pixel].sum()*PXHA,1),geometric=round(cnt[sel_geo].sum()*PXHA,1),l3=round(cnt[sel_l3].sum()*PXHA,1)),
  jaccard_area=dict(pixel_vs_geo=round(jacc(sel_pixel,sel_geo),3),pixel_vs_l3=round(jacc(sel_pixel,sel_l3),3),geo_vs_l3=round(jacc(sel_geo,sel_l3),3)),
  jaccard_blockcount=dict(pixel_vs_geo=round(jacc_cnt(sel_pixel,sel_geo),3),pixel_vs_l3=round(jacc_cnt(sel_pixel,sel_l3),3),geo_vs_l3=round(jacc_cnt(sel_geo,sel_l3),3)),
  spearman=dict(pixel_vs_geo=round(float(sp_pg),3),pixel_vs_l3=round(float(sp_pl),3),geo_vs_l3=round(float(sp_gl),3)),
  burn_captured_pct=dict(  # how much of 2025 burn falls in each top20 set (sanity)
     pixel=round(100*burn[sel_pixel].sum()/tot_burn,1),
     geometric=round(100*burn[sel_geo].sum()/tot_burn,1),
     l3=round(100*burn[sel_l3].sum()/tot_burn,1)),
  geo_nclusters=int(K3), l3_nclusters=int(K3))
print("\n=== STEP3 decision change ===")
print(json.dumps(step3,indent=1,ensure_ascii=False))

# ---- 4. micro-priority within zone H ----
step4={}
for Hz in Hzones:
    mH=l1==Hz
    rows=[]
    for z in sorted(set(l3[mH].tolist())):
        m=(l3==z)&mH; c=cnt[m].sum()
        if c<=0: continue
        rows.append(dict(l3=int(z),area_ha=round(c*PXHA,1),
            mean_fuel=round(float(np.average(fuel[m],weights=cnt[m])),2),
            burn_rate_pct=round(100*burn[m].sum()/c,3)))
    rows.sort(key=lambda r:-r['mean_fuel'])
    step4[f'L1_{Hz}']=dict(total_area_ha=round(cnt[mH].sum()*PXHA,1),
        mean_fuel=round(float(np.average(fuel[mH],weights=cnt[mH])),2),
        burn_rate_pct=round(100*burn[mH].sum()/cnt[mH].sum(),3),
        n_L3=len(rows), top_L3=rows[:8], all_L3=rows)
print("\n=== STEP4 zone H micro priority ===")
for k,v in step4.items():
    print(k,"area %.0f ha fuel %.1f burn %.3f%%"%(v['total_area_ha'],v['mean_fuel'],v['burn_rate_pct']))
    for r in v['top_L3'][:5]: print("   L3",r)

out=dict(
  meta=dict(n_blocks=nb,total_forest_ha=round(tot_cnt*PXHA,1),total_burn_ha=round(tot_burn*PXHA,1),
            overall_burn_rate_pct=round(100*overall_burn_rate,3),overall_mean_fuel=round(float(overall_fuel),2),
            fuel_col='species_flammability_raw', px_ha=PXHA),
  step1_L1_profile=L1prof, zone_H=Hzones,
  step2_geom_vs_attr=step2,
  step3_decision_change=step3,
  step4_zoneH_micro=step4)
json.dump(out,open('zone3_decision.json','w'),ensure_ascii=False,indent=1)
print("\nSAVED zone3_decision.json")
