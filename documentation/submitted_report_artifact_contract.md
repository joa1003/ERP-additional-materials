# Submitted dissertation report-artifact contract

The submitted dissertation is the authority for public-facing report artifacts. A complete reproduction must regenerate every retained table and figure that appears in Methodology, Results and Appendices A-M. Helper CSVs remain available under `outputs/tables/`, but the stable report-facing outputs are collected under `outputs/report_tables/` and the report figure directories.

## Scope

- Main methodology: Tables 3.1-3.5; Figures 3.1-3.2.
- Main results: Tables 4.1-4.3; Figures 4.1-4.13.
- Appendix A: Tables A.1-A.2.
- Appendix B: Tables B.1-B.2; Figure B.1.
- Appendix C: Figures C.1-C.2.
- Appendix D: Tables D.1-D.5; Figure D.1.
- Appendix E: Figures E.1-E.3.
- Appendix F: Tables F.1-F.8.
- Appendix G: Table G.1.
- Appendix H: Table H.1.
- Appendix I: Tables I.1-I.2.
- Appendix J: Tables J.1-J.2.
- Appendix K: Tables K.1-K.2; Figures K.1-K.3.
- Appendix L: Table L.1 only.
- Appendix M: Tables M.1-M.7.

Total retained report items: 66. No Appendix N and no Table L.2 are created.

## Figure fidelity

Reproduction includes the analytical information visible in the submitted figures, not only the underlying curves. Required content includes panel structure, legends, markers, line styles, no-gain/reference lines, IQR bands, callouts, displayed values, load-range shares and quartile labels where present.

The Appendix C displayed annotations are locked to the values printed in the submitted dissertation while the plotted curves and anchor locations are calculated from reproduced data. Figure D.1 retains the submitted white-background callouts for the selected h=12 and h=48 labels in panel (b). Figure B.1 is the submitted h=1 Rule-3 separation diagnostic; formal eligibility remains evaluated across h=1,12,48.

## Table outputs

`code/09_report_output_generation/generate_submitted_report_tables.py` materialises stable, dissertation-numbered CSV outputs. Static design tables are generated from the locked design; empirical tables are reshaped or collected from reproduced analytical-stage outputs. The full Stage-12 run fails if any retained report-facing table or figure is missing.
