# From Label Budget to Deployed Threshold (`label-budget-bench`)

Code, frozen prompts, nested label-budget splits, alignment-audit labels, and aggregated results for:

> **From Label Budget to Deployed Threshold: How Gold and LLM-Distilled Labels Shape Discrimination, Calibration, and Shift Robustness of Fine-Tuned Encoders and Small Decoders**  
> Divya Singaravelu and Suresh Rajendran (2026)

Per-item predictions (`preds/runs/<run_id>/preds.parquet`) and full metric tables (`results/metrics.parquet`, `results/api_preds.parquet`) for all 2,342 completed runs are archived on Zenodo, [doi:10.5281/zenodo.24243899](https://doi.org/10.5281/zenodo.24243899), and unpack directly into the root of this repository.

## Content & Licensing Notice

- **Code and scripts:** MIT License (`LICENSE`).
- **No dataset text is redistributed:** Because LLM-AggreFact is licensed under CC BY-ND 4.0 and BeaverTails under CC BY-NC 4.0, all split definitions (`data/splits/`), audit files (`docs/audit/`), teacher outputs (`preds/teacher/`, `preds/api/`), and per-item model predictions (`preds/runs/`) contain **dataset item identifiers (`uid`) and numeric scores/verdicts only**. Raw datasets are fetched directly from their upstream sources via `data/download.py`.
- **Content warning:** Tasks T2 (BeaverTails / WildGuardTest) and T3 (ToxiGen / Implicit Hate Corpus) contain harmful, violent, and hateful text in the upstream datasets.

## Repository Layout

- `configs/` — Pinned model identifiers (`models.yaml`), task definitions (`tasks.yaml`), training hyperparameters (`train.yaml`), sweep matrix (`sweep.yaml`), API pricing (`budget.yaml`), and the 12 primary comparisons (`comparisons.yaml`).
- `data/` — Upstream dataset downloader (`download.py`), deterministic cleaner (`prepare.py`), nested stratified split generator (`splits.py`), dataset manifest (`MANIFEST.md`), and frozen chain/subset definitions (`data/splits/`).
- `labels/` — Frozen task prompts (`prompts/t1_rag_hallucination.yaml`, `t2_response_safety.yaml`, `t3_implicit_toxicity.yaml`; versions `t1-v1`, `t2-v1`, `t3-v1`) and budget-guarded batch/sync API runner (`api.py`).
- `train/` — Fine-tuning pipelines for TF-IDF + logistic regression (`classical.py`), ModernBERT / DeBERTa-v3 (`encoder.py`), LoRA / full-SFT Qwen3.5 & Gemma 4 decoders (`decoder.py`), and vLLM open-teacher labelling (`teacher.py`).
- `detect/` & `serve/` — Off-the-shelf zero-label detector runner (`detect/run_detectors.py`) and A6000 throughput/latency benchmark (`serve/bench.py`).
- `eval/` — GPU-free evaluation library: PR-AUC, max-F1, operating-point recall/precision, ECE, Brier, Balanced Brier Score (`metrics.py`), temperature / Platt / isotonic calibration (`calibrate.py`), split-conformal FPR control (`conformal.py`), Krippendorff's $\alpha_K$ (`stability.py`), break-even budgets $N^\star$ (`breakeven.py`), and paired item/cluster bootstrap (`fastboot.py`).
- `analysis/` — One-command paper reproduction pipeline (`run_all.sh`) generating every CSV table (`results/tables/`), vector PDF figure (`results/figures/`), and `results/paper_numbers.json`.
- `docs/` — Frozen protocol decision log (`decisions.md`), prior-art search log (`prior-art-log.md`), and anonymized label-semantics alignment audit sheets (`docs/audit/`).

## Quickstart

### 1. Environment & Unit Tests (CPU / macOS or Linux)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests/
```

### 2. Download Upstream Datasets & Rebuild Processed Splits

Accept the upstream dataset terms on Hugging Face for gated sources (`lytang/LLM-AggreFact` and `allenai/wildguardmix`), export `HF_TOKEN`, and run:

```bash
export HF_TOKEN=hf_...
.venv/bin/python data/download.py
.venv/bin/python data/prepare.py
.venv/bin/python analysis/audit_kappa.py   # reproduces Section IV-A1 alignment-audit kappas
```

### 3. Reproduce Every Table and Figure from the Zenodo Predictions Archive

Download `label-budget-bench-predictions-v1.tar.gz` from https://doi.org/10.5281/zenodo.24243899, unpack it into the repository root, then run:

```bash
tar -xzf label-budget-bench-predictions-v1.tar.gz
bash analysis/run_all.sh
```

`analysis/run_all.sh` recomputes all metrics, 10,000-resample paired bootstrap confidence intervals, cluster-bootstrap checks, Holm-adjusted $p$-values, CSV tables (`results/tables/`), and figures (`results/figures/`) on CPU in ~20 minutes on an Apple M-series laptop.

### 4. Running Training Sweeps on GPU

```bash
uv pip install --python .venv/bin/python -r requirements-gpu.txt
# Optional: build causal-conv1d wheel for faster Qwen3.5 steps
bash deploy/build_causal_conv1d.sh

# Single run example (ModernBERT-base, T2, N=1000, gold labels, seed 0)
.venv/bin/python train/run.py --family encoder --model modernbert_base --task t2_response_safety --n 1000 --seed 0

# Worker queue across all jobs in queue/jobs.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python queue/worker.py
```
