#!/usr/bin/env bash
# Every number, table and figure in the paper from the predictions on this machine (about 20 minutes on the M-series Mac).
# Refresh the predictions first with deploy/pull_preds.py while the servers are up.
#   bash analysis/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY analysis/api_eval.py      # results/api_preds.parquet, api_metrics.csv (cache only, no API calls)
$PY analysis/collect.py       # results/metrics.parquet
$PY analysis/headline.py      # 95% CIs for every cell, results/paper_numbers.json
$PY analysis/compare.py       # Appendix H paired comparisons, Holm
$PY analysis/compare.py --cluster --b 2000  # same comparisons, cluster bootstrap (T1 source docs, T2 prompts)
$PY analysis/collapse.py      # constant-output runs (test-score SD < 0.01), supplementary Table S5
$PY analysis/figures.py       # figures 2-7 (4b reads headline's cells_ci.csv), tables 4, 6, 7
$PY analysis/tables.py        # tables 5, 8, 9, D
$PY analysis/label_source.py  # Table V: teacher quality, plateau gap, exchange rate, precision/recall skew
$PY analysis/truncation.py    # truncation per task x split (tokenizers from the local cache only)
$PY analysis/compute.py       # Appendix G: GPU-hours, energy, carbon (assumptions stated in the table)
$PY analysis/fig1_design.py   # Fig. 1 design overview and the 660 x 295 px graphical abstract
if [ -d paper/latex ]; then
  $PY analysis/paper_tables.py  # LaTeX table bodies in paper/latex/tables/
  $PY analysis/latex_numbers.py # paper/latex/numbers.tex: every number in the text as a macro
fi
