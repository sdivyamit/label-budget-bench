#!/usr/bin/env bash
# Revision analyses (D37), from the local predictions.
# Writes only results/revision/ (CSV tables and revision_numbers.json); about 25 minutes on the M-series Mac.
#   bash analysis/rev_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY analysis/rev_phase1.py --b 10000         # zero-shot, no-instruction ablation, encoder LR check, pilot cells
$PY analysis/rev_breakeven.py --b 2000       # 2a, 2c: deployable break-even, floors, crossing probabilities
$PY analysis/rev_compare.py --b 10000        # 2b: seed-aware intervals, 12 comparisons and Table S2
$PY analysis/rev_drift.py --draws 200        # 2d, 2f: drift restated, exchangeability band, recalibration, prior shift
$PY analysis/rev_calibration.py              # 2e: calibration tables, Jev, seed disagreement
$PY analysis/rev_overlap.py --b 10000        # 2g: T2 prompt overlap
$PY analysis/rev_teacher_family.py --b 10000 # 2h: open-teacher family check
$PY analysis/rev_api_repeats.py --analyze     # hosted-reference repeat SD (needs preds/api_repeats/)
$PY analysis/rev_val_subsets.py --draws 1000 # N=0 thresholds chosen on validation subsets
$PY analysis/rev_final.py                    # prevalence gap, throughput ratios, overlap denominators
$PY analysis/rev_audit.py --score            # extended label-alignment audit kappas
$PY analysis/rev_accounting.py               # run counts, compute, collapse, API spend
$PY analysis/rev_figures.py                  # revised figures + compliant graphical abstract
