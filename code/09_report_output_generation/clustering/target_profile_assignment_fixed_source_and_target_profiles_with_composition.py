#!/usr/bin/env python3
"""Generate Appendix E target-profile assignment figure from reproduced outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
GROUP_TITLES={1:'Morning-evening double peak',2:'Early evening peak',3:'Late evening peak',4:'Late night peak'}
SOURCE_COLOR='#1f77b4'; TARGET_COLOR='#ff7f0e'; ZERO_COLOR='#1f77b4'
def root(): return Path(__file__).resolve().parents[3]
def args():
 p=argparse.ArgumentParser(); r=root()
 p.add_argument('--source-prototypes',type=Path,default=r/'outputs/tables/profile_clustering_clustering_final_decision_dissertation/profile_clustering_final_selected_source_prototypes.csv')
 p.add_argument('--source-summary',type=Path,default=r/'outputs/tables/profile_clustering_clustering_final_decision_dissertation/profile_clustering_final_selected_profile_groups_dissertation.csv')
 p.add_argument('--target-profiles',type=Path,default=r/'outputs/tables/target_profile_assignment_cer_assignment/target_profile_assignment_cer_training_profiles.csv')
 p.add_argument('--target-assignments',type=Path,default=r/'outputs/tables/target_profile_assignment_cer_assignment/target_profile_assignment_cer_source_defined_assignments.csv')
 p.add_argument('--output',type=Path,default=r/'outputs/figures/report/appendix_e_target_profile_assignment.png'); p.add_argument('--no-pdf',action='store_true'); return p.parse_args()
def local_time(slot):
 m=(int(slot)-1)*30; return f'{m//60:02d}:{m%60:02d}'
def prepare(a):
 src=pd.read_csv(a.source_prototypes); ss=pd.read_csv(a.source_summary); tp=pd.read_csv(a.target_profiles); ta=pd.read_csv(a.target_assignments)
 if 'assigned_group' not in ta.columns and 'profile_group' in ta.columns: ta=ta.rename(columns={'profile_group':'assigned_group'})
 merged=tp.merge(ta[['meter_id','assigned_group']],on='meter_id',validate='many_to_one')
 tg=merged.groupby(['assigned_group','local_slot'])['profile_zscore'].agg(['median',lambda x:x.quantile(.25),lambda x:x.quantile(.75)]).reset_index()
 tg.columns=['profile_group','local_slot','cer_median_z','cer_q1_z','cer_q3_z']
 src=src[['profile_group','local_slot','prototype_zscore']].copy(); src['local_time']=src['local_slot'].map(local_time)
 profiles=src.merge(tg,on=['profile_group','local_slot'],validate='one_to_one')
 counts=ta.groupby('assigned_group').size().rename('cer_meters').reset_index().rename(columns={'assigned_group':'profile_group'}); counts['cer_share']=counts.cer_meters/len(ta)
 ev=ss[['profile_group','all_assigned_n','all_assigned_share','peak_time']].rename(columns={'all_assigned_n':'source_households','all_assigned_share':'source_share','peak_time':'source_peak_time'}).merge(counts,on='profile_group')
 peak=merged.loc[merged.groupby('meter_id')['profile_zscore'].idxmax(),['meter_id','assigned_group','local_slot']]; peak['peak_time']=peak.local_slot.map(local_time)
 def median_peak(x):
  slots=np.sort(x.local_slot.to_numpy()); return local_time(int(np.median(slots)))
 med=peak.groupby('assigned_group').apply(median_peak,include_groups=False).rename('cer_median_peak_time').reset_index().rename(columns={'assigned_group':'profile_group'})
 ev=ev.merge(med,on='profile_group'); return profiles.sort_values(['profile_group','local_slot']),ev.sort_values('profile_group')
def maxima(v):
 out=[]
 for i in range(len(v)):
  left=v[i-1] if i else -np.inf; right=v[i+1] if i<len(v)-1 else -np.inf
  if v[i]>=left and v[i]>=right and (v[i]>left or v[i]>right): out.append(i)
 return out
def peak_label(title,times,vals,evidence):
 if 'double peak' not in title.lower(): return str(evidence)
 m=sorted(maxima(vals),key=lambda i: vals[i],reverse=True)[:2]; return ' and '.join(times[i] for i in sorted(m))
def make(profiles,ev,out,pdf=True):
 fig=plt.figure(figsize=(2048/150,1619/150),dpi=150); grid=fig.add_gridspec(4,2,height_ratios=[1,1,.10,.34],left=.045,right=.985,bottom=.020,top=.885,hspace=.34,wspace=.22)
 axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
 for group,ax in zip([1,2,3,4],axes):
  f=profiles[profiles.profile_group==group].sort_values('local_slot'); x=np.arange(48); times=f.local_time.astype(str).tolist(); lookup={v:i for i,v in enumerate(times)}; labels=[v for v in ['00:00','04:00','08:00','12:00','16:00','20:00','23:30'] if v in lookup]
  ax.plot(x,f.prototype_zscore,color=SOURCE_COLOR,lw=2,label='Fixed source prototype'); ax.plot(x,f.cer_median_z,color=TARGET_COLOR,lw=2,ls='--',label='Median assigned profile'); ax.fill_between(x,f.cer_q1_z,f.cer_q3_z,color=TARGET_COLOR,alpha=.18,label='Assigned-profile IQR')
  ax.set_title(f'Group {group} — {GROUP_TITLES[group]}',loc='left',fontsize=13,fontweight='bold',pad=7); ax.set_xlim(0,47); ax.set_xticks([lookup[v] for v in labels],labels,fontsize=10); ax.grid(True,alpha=.28,lw=.7); ax.axhline(0,color=ZERO_COLOR,alpha=.55,lw=.8)
  if group in (1,3): ax.set_ylabel('Standardised daily load shape',fontsize=11)
  if group in (3,4): ax.set_xlabel('Local time of day',fontsize=11)
 fig.suptitle('Fixed source prototypes and profiles assigned in the target dataset',fontsize=18,fontweight='bold',y=.982)
 h,l=axes[0].get_legend_handles_labels(); la=fig.add_subplot(grid[2,:]); la.axis('off'); la.legend(h,l,loc='center',ncol=3,frameon=False,fontsize=10.5)
 rows=[]
 for g in [1,2,3,4]:
  er=ev[ev.profile_group==g].iloc[0]; f=profiles[profiles.profile_group==g].sort_values('local_slot'); times=f.local_time.astype(str).tolist(); rows.append([f'G{g}',f"{int(er.source_households):,} ({100*float(er.source_share):.1f}%)",f"{int(er.cer_meters):,} ({100*float(er.cer_share):.1f}%)",peak_label(GROUP_TITLES[g],times,f.prototype_zscore.to_numpy(float),er.source_peak_time),str(er.cer_median_peak_time)])
 ta=fig.add_subplot(grid[3,:]); ta.axis('off'); table=ta.table(cellText=rows,colLabels=['Group','Source households, n (%)','Assigned target households, n (%)','Source peak','Target peak'],cellLoc='center',colLoc='center',loc='center',colWidths=[.08,.23,.25,.20,.20]); table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1,1.48)
 for (r,c),cell in table.get_celld().items(): cell.set_edgecolor('black'); cell.set_linewidth(.8); cell.set_facecolor('white'); cell.set_text_props(weight='bold' if r==0 else 'normal')
 out=out.resolve(); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=150,facecolor='white');
 if pdf: fig.savefig(out.with_suffix('.pdf'),facecolor='white'); plt.close(fig); print('TARGET PROFILE ASSIGNMENT FIGURE: PASS')
def main():
 a=args(); p,e=prepare(a); make(p,e,a.output,not a.no_pdf)
if __name__=='__main__': main()
