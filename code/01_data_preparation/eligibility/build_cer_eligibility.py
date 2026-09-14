from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
LOOKBACK=48; HORIZONS=(1,12,48); R3=100; R4=.05
TRAIN_START=pd.Timestamp('2009-07-15 00:00'); TRAIN_END=pd.Timestamp('2010-07-24 11:30'); FULL_START=pd.Timestamp('2009-07-15 00:00'); FULL_END=pd.Timestamp('2010-12-31 23:30')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--project-root',type=Path,default=Path.cwd()); a=ap.parse_args(); root=a.project_root.resolve()
 grid=pd.read_parquet(root/'data/processed/time_aligned/cer_canonical_local_grid.parquet',columns=['meter_id','timestamp_local','kwh'])
 grid['timestamp_local']=pd.to_datetime(grid.timestamp_local); meters=sorted(grid.meter_id.unique()); assert len(meters)==929
 pivot=grid.pivot(index='timestamp_local',columns='meter_id',values='kwh').sort_index().reindex(columns=meters); idx=pivot.index; x=pivot.to_numpy(float); present=np.isfinite(x)
 full=(idx>=FULL_START)&(idx<=FULL_END); train=(idx>=TRAIN_START)&(idx<=TRAIN_END)
 cov=present[full].mean(axis=0); r1=cov>=.9
 distinct=[]
 for j in range(len(meters)):
  v=x[train,j]; v=v[np.isfinite(v)]; distinct.append(np.unique(np.round(v,3)).size)
 distinct=np.asarray(distinct); r2=distinct>1
 useful=np.zeros((len(meters),len(HORIZONS)),dtype=int); skip=np.zeros_like(useful,dtype=float)
 for q,h in enumerate(HORIZONS):
  targets=np.flatnonzero(train); targets=targets[targets-(h+LOOKBACK-1)>=0]
  # exact train target period already begins after one support day, so formal candidate count follows target period and h support
  target_times=idx[targets]; targets=targets[(target_times>=TRAIN_START)&(target_times<=TRAIN_END)]
  for j in range(len(meters)):
   valid=0; u=0
   for t in targets:
    inp=x[t-h-LOOKBACK+1:t-h+1,j]; y=x[t,j]
    ok=np.isfinite(y) and np.isfinite(inp).all()
    if ok:
     valid+=1
     if y>0 and np.any(np.diff(inp)!=0): u+=1
   useful[j,q]=u; skip[j,q]=(len(targets)-valid)/len(targets)
 minuse=useful.min(axis=1); maxskip=skip.max(axis=1); r3=minuse>=R3; r4=maxskip<=R4; final=r1&r2&r3&r4
 out=pd.DataFrame({'meter_id':meters,'coverage_full_period':cov,'train_distinct_values':distinct,'R1_pass':r1,'R2_pass':r2,'minimum_useful_transitions_formal_horizons':minuse,'maximum_train_skip_rate_formal_horizons':maxskip,'R3_pass':r3,'R4_pass':r4,'final_eligible':final})
 for q,h in enumerate(HORIZONS): out[f'train_useful_transitions_h{h}']=useful[:,q]; out[f'train_skip_rate_h{h}']=skip[:,q]
 assert final.sum()==929, f'Expected all 929 target households to pass; found {final.sum()}'
 od=root/'outputs/tables/eligibility'; od.mkdir(parents=True,exist_ok=True); out.to_csv(od/'cer_r1_r4_eligibility.csv',index=False)
 pd.DataFrame([{'stage':'Raw selected target cohort','retained_households':929},{'stage':'R1','retained_households':int(r1.sum())},{'stage':'R2','retained_households':int((r1&r2).sum())},{'stage':'R3','retained_households':int((r1&r2&r3).sum())},{'stage':'R4','retained_households':int(final.sum())}]).to_csv(od/'cer_r1_r4_eligibility_flow.csv',index=False)
 print('CER R1-R4 ELIGIBILITY: PASS')
if __name__=='__main__': main()
