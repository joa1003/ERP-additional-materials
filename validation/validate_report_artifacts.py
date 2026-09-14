#!/usr/bin/env python3
"""Inventory check for every retained dissertation table and figure."""
from __future__ import annotations
import argparse
from pathlib import Path

TABLES=[
"Table_3_1.csv","Table_3_2.csv","Table_3_3.csv","Table_3_4.csv","Table_3_5.csv",
"Table_4_1.csv","Table_4_2.csv","Table_4_3.csv",
"Table_A_1.csv","Table_A_2.csv","Table_B_1.csv","Table_B_2.csv",
"Table_D_1.csv","Table_D_2.csv","Table_D_3.csv","Table_D_4.csv","Table_D_5.csv",
"Table_F_1.csv","Table_F_2.csv","Table_F_3.csv","Table_F_4.csv","Table_F_5.csv","Table_F_6.csv","Table_F_7.csv","Table_F_8.csv",
"Table_G_1.csv","Table_H_1.csv","Table_I_1.csv","Table_I_2.csv","Table_J_1.csv","Table_J_2.csv",
"Table_K_1.csv","Table_K_2.csv","Table_L_1.csv",
"Table_M_1.csv","Table_M_2.csv","Table_M_3.csv","Table_M_4.csv","Table_M_5.csv","Table_M_6.csv","Table_M_7.csv",
]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--project-root',type=Path,default=Path.cwd()); a=ap.parse_args(); root=a.project_root.resolve(); d=root/'outputs/report_tables'
 missing=[x for x in TABLES if not (d/x).is_file() or (d/x).stat().st_size<20]
 print('='*96); print('SUBMITTED DISSERTATION REPORT ARTIFACT INVENTORY'); print('='*96)
 print(f'Report-facing tables: {len(TABLES)-len(missing)}/{len(TABLES)}')
 if missing:
  print('REPORT TABLE INVENTORY: FAIL'); [print('-',x) for x in missing]; return 1
 print('REPORT TABLE INVENTORY: PASS')
 # figure validator remains the canonical 25-caption check
 import subprocess, sys
 r=subprocess.run([sys.executable,str(root/'validation/validate_report_figures.py'),'--project-root',str(root)])
 if r.returncode: return r.returncode
 print('\nREPORT ARTIFACT INVENTORY: PASS')
 print(f'Report tables covered : {len(TABLES)}')
 print('Figure captions covered: 25')
 print('Total retained report items represented: 66')
 return 0
if __name__=='__main__': raise SystemExit(main())
