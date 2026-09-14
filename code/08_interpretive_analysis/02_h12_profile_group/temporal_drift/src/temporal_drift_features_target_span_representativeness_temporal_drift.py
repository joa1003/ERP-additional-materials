#!/usr/bin/env python3
"""Temporal drift features: 30-day target-span representativeness and temporal profile drift.

Read-only post-hoc analysis. It compares each receiving meter's first 30-day
training span with later training, validation and test periods, then examines
whether temporal profile drift is associated with Direct Transfer error,
30-day Scratch Limited error and Source Transfer Gain. No model is trained,
no prediction is changed, no prototype is refitted and no meter is reassigned.
"""
from __future__ import annotations

import argparse, hashlib, json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy import stats

ANALYSIS_ID = "temporal_drift_features"
ANALYSIS = "temporal_drift_features_target_span_representativeness_temporal_drift"
GROUPS = [1,2,3,4]
GROUP_LABELS = {1:"G1 Morning–evening double peak",2:"G2 Early evening peak",3:"G3 Late evening peak",4:"G4 Late night peak"}
PERIODS = {
    "limited_30day": (pd.Timestamp("2009-07-15"),1,pd.Timestamp("2009-08-13"),48),
    "later_train": (pd.Timestamp("2009-08-14"),1,pd.Timestamp("2010-07-24"),24),
    "validation": (pd.Timestamp("2010-07-24"),25,pd.Timestamp("2010-09-15"),48),
    "test": (pd.Timestamp("2010-09-16"),1,pd.Timestamp("2010-12-31"),48),
}
EXCLUDED_DATES={pd.Timestamp("2010-03-28")}
PRIMARY_PREDICTORS=[
    "test_z_clock_rmse","test_correlation_distance","test_mean_abs_change_pct",
    "test_std_abs_change_pct","test_low_share_abs_change_pp","test_high_share_abs_change_pp",
    "test_peak_shift_slots","validation_z_clock_rmse","later_train_z_clock_rmse",
]

@dataclass(frozen=True)
class Paths:
    root: Path; canonical: Path; gains: Path; tables: Path; metadata: Path

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True); p.add_argument("--check-only",action="store_true")
    p.add_argument("--batch-size",type=int,default=500000); p.add_argument("--bootstraps",type=int,default=500)
    return p.parse_args()

def paths(root:Path)->Paths:
    root=root.expanduser().absolute()
    return Paths(root,root/"data/processed/time_aligned/cer_canonical_local_grid.parquet",
        root/"outputs/tables/profile_group_analysis_profile_group_analysis/profile_group_analysis_native_meter_gains_with_group.csv",
        root/f"outputs/tables/{ANALYSIS}",root/f"outputs/metadata/{ANALYSIS}")

def ensure(c:bool,m:str):
    if not c: raise RuntimeError(m)

def require(df:pd.DataFrame, cols:Iterable[str], label:str):
    miss=sorted(set(cols)-set(df.columns)); ensure(not miss,f"{label} missing columns: {miss}")

def norm_id(x)->str:
    s=str(x).strip()
    try:
        f=float(s)
        if f.is_integer(): return str(int(f))
    except Exception: pass
    return s

def load_inputs(p:Paths):
    ensure(p.canonical.is_file(),f"Missing canonical CER parquet: {p.canonical}")
    ensure(p.gains.is_file(),f"Missing Profile group analysis gains: {p.gains}")
    gains=pd.read_csv(p.gains)
    require(gains,["entity_id","lead","criterion","support_scope","assigned_group","group_description","seed_count",
        "source_transfer_gain_mean","direct_transfer_value_mean","cer_scratch_limited_value_mean"],"Profile group analysis gains")
    ensure(len(gains)==8361,f"Expected 8,361 gain rows, found {len(gains):,}")
    ensure(gains.entity_id.nunique()==929,"Expected 929 meters")
    ensure(gains.support_scope.eq("native").all(),"Native support required")
    ensure(gains.seed_count.eq(4).all(),"Four formal seeds required")
    gains["meter_id"]=gains.entity_id.map(norm_id)
    counts=gains[["meter_id","assigned_group"]].drop_duplicates().assigned_group.value_counts().sort_index().to_dict()
    ensure(counts=={1:141,2:325,3:324,4:139},f"Unexpected group counts: {counts}")
    dataset=ds.dataset(str(p.canonical),format="parquet")
    req=["meter_id","local_date","local_slot","kwh"]
    ensure(not sorted(set(req)-set(dataset.schema.names)),f"Canonical parquet missing columns: {sorted(set(req)-set(dataset.schema.names))}")
    return gains,dataset

def mask_period(frame:pd.DataFrame, spec):
    sd,ss,ed,es=spec; d=pd.to_datetime(frame.local_date,errors="raise").dt.normalize(); s=pd.to_numeric(frame.local_slot,errors="raise").astype(int)
    return ((d>sd)|((d==sd)&(s>=ss))) & ((d<ed)|((d==ed)&(s<=es))) & (~d.isin(EXCLUDED_DATES))

def scan_periods(dataset,batch_size:int):
    profile_parts=[]; stat_parts=[]; scanned=0
    scanner=dataset.scanner(columns=["meter_id","local_date","local_slot","kwh"],batch_size=batch_size,use_threads=True)
    for batch in scanner.to_batches():
        f=batch.to_pandas(); scanned+=len(f); f["meter_id"]=f.meter_id.map(norm_id); f["kwh"]=pd.to_numeric(f.kwh,errors="coerce")
        for period,spec in PERIODS.items():
            q=f.loc[mask_period(f,spec),["meter_id","local_slot","kwh"]].copy()
            if q.empty: continue
            q=q[q.kwh.notna()]; q["local_slot"]=pd.to_numeric(q.local_slot).astype(int)
            ensure(q.local_slot.between(1,48).all(),"Non-standard local slot entered Temporal drift features")
            pr=q.groupby(["meter_id","local_slot"],observed=True,sort=False).kwh.agg(["sum","count"]).reset_index()
            pr["period"]=period; profile_parts.append(pr)
            x=q.kwh.to_numpy(float)
            q["x2"]=q.kwh*q.kwh; q["is_zero"]=(q.kwh==0).astype(int); q["le01"]=(q.kwh<=.1).astype(int)
            q["mid"]=(q.kwh.gt(.1)&q.kwh.le(.5)).astype(int); q["midhi"]=(q.kwh.gt(.5)&q.kwh.le(1)).astype(int); q["high"]=(q.kwh>1).astype(int)
            st=q.groupby("meter_id",observed=True,sort=False).agg(n=("kwh","size"),sum=("kwh","sum"),sumsq=("x2","sum"),zeros=("is_zero","sum"),le01=("le01","sum"),mid=("mid","sum"),midhi=("midhi","sum"),high=("high","sum")).reset_index()
            st["period"]=period; stat_parts.append(st)
    ensure(profile_parts and stat_parts,"No canonical rows selected")
    prof=pd.concat(profile_parts,ignore_index=True).groupby(["period","meter_id","local_slot"],observed=True,sort=True)[["sum","count"]].sum().reset_index()
    stat=pd.concat(stat_parts,ignore_index=True).groupby(["period","meter_id"],observed=True,sort=True)[["n","sum","sumsq","zeros","le01","mid","midhi","high"]].sum().reset_index()
    return prof,stat,scanned

def time_label(slot:int)->str:
    m=(slot-1)*30; return f"{m//60:02d}:{m%60:02d}"

def build_profiles(prof,stat,meter_ids):
    rows=[]; chars=[]
    for period in PERIODS:
        p=prof[prof.period==period].copy(); s=stat[stat.period==period].copy().set_index("meter_id")
        pivot=(p.assign(mean=p["sum"]/p["count"]).pivot(index="meter_id",columns="local_slot",values="mean").reindex(index=meter_ids,columns=range(1,49)))
        ensure(not pivot.isna().any().any(),f"Missing meter-slot mean in {period}")
        a=pivot.to_numpy(float); means=a.mean(1); stds=a.std(1,ddof=0); z=(a-means[:,None])/np.maximum(stds[:,None],1e-12)
        for i,m in enumerate(meter_ids):
            total=max(float(a[i].sum()),1e-12); peak=int(np.argmax(a[i])+1); ramp=float(np.max(np.abs(np.diff(a[i]))))
            morning=float(a[i,12:20].sum()/total); midday=float(a[i,20:32].sum()/total); evening=float(a[i,32:42].sum()/total); night=1-morning-midday-evening
            rr=s.loc[m]; n=float(rr.n); mean=float(rr["sum"]/n); var=max(float(rr.sumsq/n-mean*mean),0.0)
            chars.append({"meter_id":m,"period":period,"observed_rows":int(n),"raw_mean_kwh":mean,"raw_std_kwh_ddof0":math.sqrt(var),
                "zero_share":float(rr.zeros/n),"low_le_0_1_share":float(rr.le01/n),"mid_0_1_0_5_share":float(rr.mid/n),"mid_0_5_1_0_share":float(rr.midhi/n),"high_gt_1_share":float(rr.high/n),
                "profile_mean_kwh":float(means[i]),"profile_std_kwh_ddof0":float(stds[i]),"peak_slot":peak,"peak_time":time_label(peak),"peak_to_mean_ratio":float(a[i].max()/max(means[i],1e-12)),
                "max_adjacent_ramp_kwh":ramp,"morning_energy_share":morning,"midday_energy_share":midday,"evening_energy_share":evening,"night_energy_share":night})
            for slot in range(48): rows.append({"meter_id":m,"period":period,"local_slot":slot+1,"local_time":time_label(slot+1),"raw_mean_kwh":float(a[i,slot]),"profile_zscore":float(z[i,slot])})
    return pd.DataFrame(chars),pd.DataFrame(rows)

def corr_distance(a,b):
    if np.std(a)<1e-12 or np.std(b)<1e-12: return np.nan
    return float(1-np.corrcoef(a,b)[0,1])

def circular_slot_diff(a,b):
    d=abs(int(a)-int(b)); return min(d,48-d)

def build_drift(chars,profiles):
    cp=chars.set_index(["meter_id","period"]); pp=profiles.pivot(index=["meter_id","period"],columns="local_slot",values="raw_mean_kwh")
    zp=profiles.pivot(index=["meter_id","period"],columns="local_slot",values="profile_zscore")
    out=[]
    for m in sorted(chars.meter_id.unique()):
        base=cp.loc[(m,"limited_30day")]; br=pp.loc[(m,"limited_30day")].to_numpy(float); bz=zp.loc[(m,"limited_30day")].to_numpy(float)
        row={"meter_id":m}
        for period in ["later_train","validation","test"]:
            c=cp.loc[(m,period)]; r=pp.loc[(m,period)].to_numpy(float); z=zp.loc[(m,period)].to_numpy(float); pre=period+"_"
            row[pre+"z_clock_rmse"]=float(np.sqrt(np.mean((z-bz)**2)))
            row[pre+"correlation_distance"]=corr_distance(bz,z)
            row[pre+"raw_clock_mae_kwh"]=float(np.mean(np.abs(r-br)))
            row[pre+"raw_clock_rmse_kwh"]=float(np.sqrt(np.mean((r-br)**2)))
            row[pre+"mean_signed_change_pct"]=100*float(c.raw_mean_kwh-base.raw_mean_kwh)/max(abs(float(base.raw_mean_kwh)),1e-6)
            row[pre+"mean_abs_change_pct"]=abs(row[pre+"mean_signed_change_pct"])
            row[pre+"std_signed_change_pct"]=100*float(c.raw_std_kwh_ddof0-base.raw_std_kwh_ddof0)/max(abs(float(base.raw_std_kwh_ddof0)),1e-6)
            row[pre+"std_abs_change_pct"]=abs(row[pre+"std_signed_change_pct"])
            row[pre+"low_share_change_pp"]=100*float(c.low_le_0_1_share-base.low_le_0_1_share); row[pre+"low_share_abs_change_pp"]=abs(row[pre+"low_share_change_pp"])
            row[pre+"high_share_change_pp"]=100*float(c.high_gt_1_share-base.high_gt_1_share); row[pre+"high_share_abs_change_pp"]=abs(row[pre+"high_share_change_pp"])
            row[pre+"peak_shift_slots"]=circular_slot_diff(base.peak_slot,c.peak_slot)
            row[pre+"evening_share_change_pp"]=100*float(c.evening_energy_share-base.evening_energy_share)
            row[pre+"night_share_change_pp"]=100*float(c.night_energy_share-base.night_energy_share)
            row[pre+"peak_to_mean_abs_change"]=abs(float(c.peak_to_mean_ratio-base.peak_to_mean_ratio))
            row[pre+"ramp_abs_change_kwh"]=abs(float(c.max_adjacent_ramp_kwh-base.max_adjacent_ramp_kwh))
        out.append(row)
    return pd.DataFrame(out)

def safe_rel(num,den): return np.where(np.abs(den)>1e-12,100*np.asarray(num,float)/np.asarray(den,float),np.nan)

def build_outcome(gains,drift):
    h=gains[gains.lead.eq(12)].copy(); h["source_transfer_gain_relative_pct"]=safe_rel(h.source_transfer_gain_mean,h.cer_scratch_limited_value_mean)
    return h.merge(drift,on="meter_id",how="inner",validate="many_to_one")

def bh(p):
    p=np.asarray(p,float); n=len(p); order=np.argsort(p); out=np.empty(n); prev=1.0
    for rank,idx in reversed(list(enumerate(order,start=1))): prev=min(prev,p[idx]*n/rank); out[idx]=prev
    return np.clip(out,0,1)

def rank_residual(values,groups):
    y=pd.Series(values).rank(method="average").to_numpy(float); X=pd.get_dummies(pd.Series(groups).astype(int),drop_first=False).to_numpy(float); X=np.column_stack([np.ones(len(X)),X[:,1:]])
    return y-X@np.linalg.lstsq(X,y,rcond=None)[0]

def correlations(outcome):
    outcomes=["source_transfer_gain_mean","source_transfer_gain_relative_pct","direct_transfer_value_mean","cer_scratch_limited_value_mean"]
    rows=[]; adj=[]; within=[]
    for criterion in ["mae","rmse","smape"]:
        d=outcome[outcome.criterion.eq(criterion)].copy()
        for pred in PRIMARY_PREDICTORS:
            for ycol in outcomes:
                x=d[pred].to_numpy(float); y=d[ycol].to_numpy(float); mask=np.isfinite(x)&np.isfinite(y)
                rho,pv=stats.spearmanr(x[mask],y[mask]); rows.append({"criterion":criterion,"predictor":pred,"outcome":ycol,"n":int(mask.sum()),"rho":rho,"p_value":pv})
                rx=rank_residual(x[mask],d.loc[mask,"assigned_group"]); ry=rank_residual(y[mask],d.loc[mask,"assigned_group"]); ar,ap=stats.pearsonr(rx,ry)
                adj.append({"criterion":criterion,"predictor":pred,"outcome":ycol,"n":int(mask.sum()),"rho":ar,"p_value":ap})
                for g in GROUPS:
                    q=d.assigned_group.eq(g)&mask; wr,wp=stats.spearmanr(x[q],y[q]); within.append({"criterion":criterion,"predictor":pred,"outcome":ycol,"assigned_group":g,"n":int(q.sum()),"rho":wr,"p_value":wp})
    for df in (rows,adj,within):
        q=bh([r["p_value"] for r in df]); [r.update({"q_value_bh":float(v)}) for r,v in zip(df,q)]
    return pd.DataFrame(rows),pd.DataFrame(adj),pd.DataFrame(within)

def bootstrap_primary(outcome,bootstraps):
    rng=np.random.default_rng(20260728); rows=[]
    for criterion in ["mae","rmse","smape"]:
        d=outcome[outcome.criterion.eq(criterion)].reset_index(drop=True)
        y=d.source_transfer_gain_relative_pct.to_numpy(float)
        for pred in PRIMARY_PREDICTORS:
            x=d[pred].to_numpy(float); valid=np.isfinite(x)&np.isfinite(y); xv=x[valid]; yv=y[valid]; vals=[]
            for _ in range(bootstraps):
                idx=rng.integers(0,len(xv),len(xv)); vals.append(stats.spearmanr(xv[idx],yv[idx]).statistic)
            rho,pv=stats.spearmanr(xv,yv); rows.append({"criterion":criterion,"predictor":pred,"outcome":"source_transfer_gain_relative_pct","rho":rho,"p_value":pv,"bootstrap_ci_low":float(np.nanquantile(vals,.025)),"bootstrap_ci_high":float(np.nanquantile(vals,.975)),"bootstraps":bootstraps})
    df=pd.DataFrame(rows); df["q_value_bh"]=bh(df.p_value); return df

def quartiles(outcome):
    rows=[]
    for criterion in ["mae","rmse","smape"]:
        d=outcome[outcome.criterion.eq(criterion)].copy()
        for pred in ["test_z_clock_rmse","test_mean_abs_change_pct","test_low_share_abs_change_pp","test_high_share_abs_change_pp"]:
            d["quartile"]=pd.qcut(d[pred].rank(method="first"),4,labels=[1,2,3,4])
            for q,x in d.groupby("quartile",observed=True):
                y=x.source_transfer_gain_relative_pct
                rows.append({"criterion":criterion,"predictor":pred,"quartile":int(q),"meters":len(x),"predictor_mean":x[pred].mean(),"gain_mean_pct":y.mean(),"gain_median_pct":y.median(),"positive_share":(y>0).mean()})
    return pd.DataFrame(rows)

def group_summary(drift,outcome):
    mem=outcome[["meter_id","assigned_group","group_description"]].drop_duplicates(); d=drift.merge(mem,on="meter_id",validate="one_to_one")
    rows=[]
    for pred in PRIMARY_PREDICTORS:
        for g,x in d.groupby("assigned_group"):
            rows.append({"predictor":pred,"assigned_group":int(g),"group_description":x.group_description.iloc[0],"meters":len(x),"mean":x[pred].mean(),"median":x[pred].median(),"q25":x[pred].quantile(.25),"q75":x[pred].quantile(.75),"sample_std":x[pred].std(ddof=1)})
    return pd.DataFrame(rows)

def write_outputs(p,chars,profiles,drift,outcome,overall,adjusted,within,primary,quart,group,scanned,elapsed):
    p.tables.mkdir(parents=True,exist_ok=True); p.metadata.mkdir(parents=True,exist_ok=True)
    files=[]
    def csv(df,name):
        path=p.tables/name; df.to_csv(path,index=False); files.append(path); return path
    csv(chars,"temporal_drift_features_meter_period_characteristics.csv")
    path=p.tables/"temporal_drift_features_meter_period_profiles.parquet"; profiles.to_parquet(path,index=False); files.append(path)
    csv(drift,"temporal_drift_features_meter_temporal_drift_features.csv")
    csv(outcome,"temporal_drift_features_h12_drift_outcome_dataset.csv")
    csv(group,"temporal_drift_features_group_drift_summary.csv")
    csv(overall,"temporal_drift_features_overall_spearman_summary.csv")
    csv(adjusted,"temporal_drift_features_group_adjusted_rank_summary.csv")
    csv(within,"temporal_drift_features_within_group_spearman_summary.csv")
    csv(primary,"temporal_drift_features_h12_primary_drift_evidence.csv")
    csv(quart,"temporal_drift_features_drift_quartile_summary.csv")
    status={"analysis_id":ANALYSIS_ID,"status":"COMPLETE_PASS","meters":929,"period_characteristic_rows":len(chars),"period_profile_rows":len(profiles),"drift_rows":len(drift),"h12_outcome_rows":len(outcome),"overall_relationships":len(overall),"group_adjusted_relationships":len(adjusted),"within_group_relationships":len(within),"primary_relationships":len(primary),"quartile_rows":len(quart),"canonical_rows_scanned":int(scanned),"elapsed_seconds":elapsed,"training_performed":False,"predictions_modified":False,"prototypes_refitted":False,"meters_reassigned":False,"meters_excluded":False,"interpretation":"descriptive and non-causal"}
    sp=p.metadata/f"{ANALYSIS}_status.json"; sp.write_text(json.dumps(status,indent=2)+"\n"); files.append(sp)
    inv=[]
    for file in files:
        h=hashlib.sha256(file.read_bytes()).hexdigest(); inv.append({"relative_path":str(file.relative_to(p.root)),"bytes":file.stat().st_size,"sha256":h})
    pd.DataFrame(inv).to_csv(p.metadata/"temporal_drift_features_output_inventory.csv",index=False)

def main():
    a=args(); p=paths(a.root); gains,dataset=load_inputs(p)
    print("="*100); print("TEMPORAL DRIFT FEATURES TARGET-SPAN REPRESENTATIVENESS AND TEMPORAL PROFILE DRIFT"); print("="*100)
    print(f"Meters        : {gains.entity_id.nunique()}"); print(f"Canonical     : {p.canonical}"); print("Mode          : read-only; descriptive and non-causal")
    if a.check_only:
        print("INPUT CHECK: PASS"); return
    t=time.time(); prof,stat,scanned=scan_periods(dataset,a.batch_size); meter_ids=sorted(gains.meter_id.unique(),key=lambda x:int(x) if x.isdigit() else x)
    chars,profiles=build_profiles(prof,stat,meter_ids); ensure(len(chars)==3716,"Expected 3,716 meter-period rows"); ensure(len(profiles)==178368,"Expected 178,368 profile rows")
    drift=build_drift(chars,profiles); ensure(len(drift)==929,"Expected 929 drift rows")
    outcome=build_outcome(gains,drift); ensure(len(outcome)==2787,"Expected 2,787 h=12 outcome rows")
    overall,adjusted,within=correlations(outcome); primary=bootstrap_primary(outcome,a.bootstraps); quart=quartiles(outcome); group=group_summary(drift,outcome)
    elapsed=time.time()-t; write_outputs(p,chars,profiles,drift,outcome,overall,adjusted,within,primary,quart,group,scanned,elapsed)
    print("="*100); print("TEMPORAL DRIFT FEATURES COMPLETE — PASS"); print("="*100)
    print("Meters retained             : 929"); print(f"Canonical rows scanned      : {scanned:,}"); print("Meter-period rows           : 3,716"); print("Meter drift rows            : 929"); print("h=12 outcome rows           : 2,787"); print(f"Elapsed seconds             : {elapsed:.1f}"); print("Training performed          : False"); print("Predictions modified        : False"); print("Final status                : COMPLETE_PASS")

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"TEMPORAL DRIFT FEATURES FAILED: {e}",file=__import__('sys').stderr); raise
