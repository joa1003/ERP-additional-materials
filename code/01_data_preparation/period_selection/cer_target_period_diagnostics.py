from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
 p=argparse.ArgumentParser(); p.add_argument('--project-root',type=Path,default=Path.cwd()); a=p.parse_args(); root=a.project_root.resolve()
 grid=pd.read_parquet(root/'data/processed/time_aligned/cer_canonical_local_grid.parquet',columns=['meter_id','timestamp_local','kwh'])
 grid['timestamp_local']=pd.to_datetime(grid['timestamp_local']); meters=sorted(grid.meter_id.unique()); assert len(meters)==929
 periods=[('Full period (Jul 2009-Dec 2010)','2009-07-15 00:00','2010-12-31 23:30'),('First 12 months (Jul 2009-Jul 2010)','2009-07-15 00:00','2010-07-14 23:30'),('Year 2010','2010-01-01 00:00','2010-12-31 23:30')]
 rows=[]
 for label,start,end in periods:
  s=pd.Timestamp(start); e=pd.Timestamp(end); exp=int((e-s)/pd.Timedelta('30min'))+1; sub=grid[(grid.timestamp_local>=s)&(grid.timestamp_local<=e)]
  counts=sub.loc[sub.kwh.notna()].groupby('meter_id').size().reindex(meters,fill_value=0); cov=counts/exp
  rows.append({'candidate_window':label,'timestamps':exp,'mean_coverage_pct':100*cov.mean(),'median_coverage_pct':100*cov.median(),'p25_coverage_pct':100*cov.quantile(.25),'p75_coverage_pct':100*cov.quantile(.75),'minimum_coverage_pct':100*cov.min(),'households_ge_90pct':int((cov>=.9).sum()),'households_lt_50pct':int((cov<.5).sum()),'overall_missing_rate_pct':100*(1-counts.sum()/(exp*len(meters)))})
 out=root/'outputs/tables/period_selection/cer_candidate_period_comparison.csv'; out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); print('CER PERIOD DIAGNOSTIC: PASS'); print(out)
if __name__=='__main__': main()
