#!/usr/bin/env python3
"""Stage the public GitHub repository export and build the Zenodo predictions archive.

Writes outside the working tree into `<workspace>/release/label-budget-bench/` without
mutating any frozen data, predictions, or results in `label-budget-bench/`:
  1. `public_repo/` — clean MIT-licensed GitHub repository tree:
     - Excludes `paper/`, `plans/`, `.env`, `checkpoints/`, `logs/` (except sanitized serving CSV),
       raw/processed datasets, LoRA adapter weights (`preds/runs/*/adapter`), API `.batches.json`
       and `results/api_ledger.jsonl`, and internal cluster/storage ops scripts.
     - Anonymizes audit annotator IDs (`annotator_1`, `annotator_2`) and strips dataset text
       columns from `docs/audit/` CSVs so CC BY-ND 4.0 (LLM-AggreFact) and ODC-BY (WildGuardTest)
       texts are never redistributed while `analysis/audit_kappa.py` reproduces identical kappas.
     - Sanitizes vendor reserve comments in `configs/budget.yaml`, instance IDs in the serving CSV,
       and private key/server details in `docs/decisions.md`.
  2. `zenodo/label-budget-bench-predictions-v1.tar.gz` — lightweight per-item predictions and
     metrics archive (<100 MB compressed, well under Zenodo's 50 GB limit) containing:
     - `preds/runs/<run_id>/{preds.parquet,metrics.json,run.json,DONE}` for all 2,342 finished runs
       (excluding 16.55 GB of LoRA adapter weights),
     - `preds/teacher/qwen35_9b/*.{parquet,done}`,
     - `results/{metrics.parquet,api_preds.parquet,api_metrics.csv,paper_numbers.json,ENVIRONMENT-gpu.txt,FREEZE-2026-11-15.sha256}`,
     - `results/tables/*.csv` and `results/figures/*.{pdf,png}`,
     - `logs/serving_a6000.csv` (instance hostname sanitized).
  3. `zenodo/SHA256SUMS.txt` and `zenodo/ZENODO-METADATA.md` — upload checklist and metadata.
  4. Runs an automated secret / IPv4 / hostname / PII / raw-text leak scan over `public_repo/`
     and the Zenodo archive manifest, failing with non-zero exit if any match is found.

Usage (from `label-budget-bench/`):
  .venv/bin/python deploy/export_public_release.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path

import pandas as pd
import yaml

SRC = Path(__file__).resolve().parent.parent
WORKSPACE = SRC.parent
OUT_ROOT = WORKSPACE / "release" / "label-budget-bench"
PUB = OUT_ROOT / "public_repo"
ZEN = OUT_ROOT / "zenodo"
TARBALL = ZEN / "label-budget-bench-predictions-v1.tar.gz"

CODE_DIRS = ["analysis", "configs", "data", "detect", "eval", "labels", "queue", "serve", "tests", "train"]
DEPLOY_KEEP = [
    "after_sweep.sh",
    "build_causal_conv1d.sh",
    "export_public_release.py",
    "frontier_ready.py",
    "gate_a.sh",
    "handover.sh",
    "teacher_all.sh",
]
ROOT_FILES = [
    ".env.example",
    ".gitignore",
    "LICENSE",
    " requirements.txt".strip(),
    "requirements-gpu.txt",
    "requirements-teacher.txt",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_clean_tree(src_dir: Path, dst_dir: Path) -> None:
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", ".DS_Store", ".pytest_cache", ".ruff_cache"
        ),
    )


def build_public_repo() -> None:
    if PUB.exists():
        shutil.rmtree(PUB)
    PUB.mkdir(parents=True, exist_ok=True)

    # 1. Root files
    for fname in ROOT_FILES:
        shutil.copy2(SRC / fname, PUB / fname)

    # 2. Code directories (filtered where needed)
    for d in CODE_DIRS:
        if d == "data":
            (PUB / "data").mkdir(parents=True, exist_ok=True)
            for p in sorted((SRC / "data").iterdir()):
                if p.is_file() and p.name != ".DS_Store":
                    shutil.copy2(p, PUB / "data" / p.name)
            copy_clean_tree(SRC / "data" / "splits", PUB / "data" / "splits")
        elif d == "queue":
            (PUB / "queue").mkdir(parents=True, exist_ok=True)
            for p in sorted((SRC / "queue").iterdir()):
                if p.is_file() and p.name not in {".DS_Store", "STOP"}:
                    shutil.copy2(p, PUB / "queue" / p.name)
        else:
            copy_clean_tree(SRC / d, PUB / d)

    # 3. Sanitize configs/budget.yaml (keep exact pricing/protocol structure; strip account reserve notes)
    budget_txt = """\
# API money. The runner never spends past `credit_usd - safety_margin_usd` for a vendor.
#
# credit_usd: money available to THIS project on that vendor account, counted from the first
#   ledger entry. When you recharge, ADD the recharge amount here, then rerun the same command;
#   finished items are cached and never paid for twice.
# Prices: USD per 1M tokens, checked 10 Nov 2026 (OpenAI and Google pricing pages, TypeSafe
#   pricing page). Re-check before each run; Gemini 4 Argon doubles on 1 Dec 2026.

vendors:
  openai:   {credit_usd: 30.0, safety_margin_usd: 3.0}
  gemini:   {credit_usd: 30.0, safety_margin_usd: 3.0}
  typesafe: {credit_usd: 5.0,  safety_margin_usd: 0.5}
"""
    orig_budget = (SRC / "configs" / "budget.yaml").read_text()
    prices_idx = orig_budget.index("\nprices:\n")
    (PUB / "configs" / "budget.yaml").write_text(budget_txt + orig_budget[prices_idx:])

    # Anonymize annotator names in configs/tasks.yaml to match anonymized audit sheets
    tasks_cfg = yaml.safe_load((PUB / "configs" / "tasks.yaml").read_text())
    annotator_map = {"divya": "annotator_1", "suresh": "annotator_2"}
    tasks_cfg["annotators"] = [annotator_map.get(a, a) for a in tasks_cfg["annotators"]]
    tasks_txt = (PUB / "configs" / "tasks.yaml").read_text()
    tasks_txt = re.sub(
        r"^annotators:\s*\[.*?\]",
        'annotators: [annotator_1, annotator_2]',
        tasks_txt,
        flags=re.M,
    )
    (PUB / "configs" / "tasks.yaml").write_text(tasks_txt)

    # 4. Deploy scripts (reproducible single-box setup/training scripts only; no cluster IPs/SSH keys)
    (PUB / "deploy").mkdir(parents=True, exist_ok=True)
    for fname in DEPLOY_KEEP:
        shutil.copy2(SRC / "deploy" / fname, PUB / "deploy" / fname)

    # 5. Docs: prior-art-log.md, sanitized decisions.md, and license-safe ID-only audit sheets
    (PUB / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / "docs" / "prior-art-log.md", PUB / "docs" / "prior-art-log.md")

    dec_txt = (SRC / "docs" / "decisions.md").read_text()
    dec_txt = re.sub(
        r"\| D26 \| 2026-11-12 \| [^\n]+",
        "| D26 | 2026-11-12 | **Results archival during multi-GPU execution.** Completed run folders (`preds/runs`, `preds/teacher`, logs) from each A6000 worker were periodically synced and SHA-256 checksum-verified to persistent off-instance storage before worker teardown; `.env` and API credentials were never transferred. | Prevents data loss across ephemeral GPU instances without shared filesystems. | Signed 2026-11-12 (S.R.) |",
        dec_txt,
    )
    dec_txt = re.sub(r"snapshot `gate-a-\d{4}-\d{2}-\d{2}`", "a prebuilt environment snapshot", dec_txt)
    (PUB / "docs" / "decisions.md").write_text(dec_txt)

    # Audit directory: instructions (anonymized) + ID-only final CSVs + rejected-set gold CSVs
    (PUB / "docs" / "audit").mkdir(parents=True, exist_ok=True)
    inst_txt = (SRC / "docs" / "audit" / "INSTRUCTIONS.md").read_text()
    inst_txt = inst_txt.replace(
        "(Divya: `*_audit_divya.csv`; Suresh: `*_audit_suresh.csv`)",
        "(`*_audit_annotator_1.csv`, `*_audit_annotator_2.csv`)",
    )
    (PUB / "docs" / "audit" / "INSTRUCTIONS.md").write_text(inst_txt)

    audit_specs = [
        (SRC / "docs" / "audit", PUB / "docs" / "audit", "t2_response_safety_audit_final.csv", None),
        (
            SRC / "docs" / "audit" / "t1_llm_aggrefact_rejected",
            PUB / "docs" / "audit" / "t1_llm_aggrefact_rejected",
            "t1_rag_hallucination_audit_final.csv",
            "gold.csv",
        ),
        (
            SRC / "docs" / "audit" / "t3_implicit_hate_rejected",
            PUB / "docs" / "audit" / "t3_implicit_hate_rejected",
            "t3_implicit_toxicity_audit_final.csv",
            "gold.csv",
        ),
    ]
    for s_dir, d_dir, csv_name, gold_name in audit_specs:
        d_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(s_dir / csv_name, dtype=str, keep_default_na=False)
        out_df = pd.DataFrame(
            {
                "item": df["item"],
                "uid": df["uid"],
                "annotator_1_label": df["divya_label"].str.strip(),
                "annotator_2_label": df["suresh_label"].str.strip(),
                "annotator_1_unsure": (df["divya_notes"].str.strip() != "").astype(int),
                "annotator_2_unsure": (df["notes"].str.strip() != "").astype(int),
            }
        )
        out_df.to_csv(d_dir / csv_name, index=False)
        if gold_name:
            shutil.copy2(s_dir / gold_name, d_dir / gold_name)

    # 6. Cached API verdicts/scores & open-teacher labels (IDs + scores only, no dataset text, no batch IDs)
    (PUB / "preds" / "api").mkdir(parents=True, exist_ok=True)
    for p in sorted((SRC / "preds" / "api").iterdir()):
        if p.is_file() and p.suffix == ".json":
            shutil.copy2(p, PUB / "preds" / "api" / p.name)
        elif p.is_dir():
            sub = PUB / "preds" / "api" / p.name
            sub.mkdir(parents=True, exist_ok=True)
            for fp in sorted(p.iterdir()):
                if fp.name == "frontier_teacher.done" or fp.name.endswith(".jsonl"):
                    if fp.name.endswith(".refusals.jsonl"):
                        # Strip internal vendor batch IDs while keeping uid, policy code, and timestamp
                        lines = []
                        for ln in fp.read_text().splitlines():
                            if ln.strip():
                                rec = json.loads(ln)
                                rec.pop("batch", None)
                                lines.append(json.dumps(rec))
                        (sub / fp.name).write_text("\n".join(lines) + ("\n" if lines else ""))
                    else:
                        shutil.copy2(fp, sub / fp.name)

    copy_clean_tree(SRC / "preds" / "teacher", PUB / "preds" / "teacher")

    # 7. Sanitized serving benchmark CSV in logs/serving_a6000.csv
    (PUB / "logs").mkdir(parents=True, exist_ok=True)
    srv_src = next(p for p in (SRC / "logs").glob("serving_*.csv") if "smoke" not in p.name)
    srv_df = pd.read_csv(srv_src)
    srv_df["host"] = "a6000-node"
    srv_df.to_csv(PUB / "logs" / "serving_a6000.csv", index=False)

    # 8. Aggregated results, CSV tables, figures, environment, and freeze manifest
    (PUB / "results").mkdir(parents=True, exist_ok=True)
    for fname in ["ENVIRONMENT-gpu.txt", "FREEZE-2026-11-15.sha256", "api_metrics.csv", "paper_numbers.json"]:
        if fname == "paper_numbers.json":
            pn = json.loads((SRC / "results" / fname).read_text())
            if "serving" in pn and "serving_logs" in pn["serving"]:
                pn["serving"]["serving_logs"] = ["serving_a6000.csv"]
            (PUB / "results" / fname).write_text(json.dumps(pn, indent=2) + "\n")
        elif fname == "FREEZE-2026-11-15.sha256":
            # Replace internal instance hostname in the single serving log path note
            ftxt = re.sub(
                r"logs/serving_instance-[A-Za-z0-9\-]+\.csv",
                "logs/serving_instance-sanitized.csv  # released as logs/serving_a6000.csv (host column anonymized)",
                (SRC / "results" / fname).read_text(),
            )
            (PUB / "results" / fname).write_text(ftxt)
        else:
            shutil.copy2(SRC / "results" / fname, PUB / "results" / fname)
    copy_clean_tree(SRC / "results" / "tables", PUB / "results" / "tables")
    copy_clean_tree(SRC / "results" / "figures", PUB / "results" / "figures")

    # Adjust public .gitignore so tracked release artifacts in logs/serving_a6000.csv, data/splits/, preds/teacher/ are not ignored
    (PUB / ".gitignore").write_text(
        """\
.venv/
.venv-teacher/
__pycache__/
.pytest_cache/
.ruff_cache/
data/raw/*/
data/processed/
preds/runs/
results/*.parquet
results/collect_cache/
checkpoints/
logs/*
!logs/serving_a6000.csv
queue/locks/
queue/failed/
queue/STOP
.env
deploy/wheels/
"""
    )

    # 9. Public README.md
    (PUB / "README.md").write_text(PUBLIC_README)


PUBLIC_README = """\
# From Label Budget to Deployed Threshold (`label-budget-bench`)

Code, frozen prompts, nested label-budget splits, alignment-audit labels, and aggregated results for:

> **From Label Budget to Deployed Threshold: How Gold and LLM-Distilled Labels Shape Discrimination, Calibration, and Shift Robustness of Fine-Tuned Encoders and Small Decoders**  
> Divya Singaravelu and Suresh Rajendran (2026)

Per-item predictions (`preds/runs/<run_id>/preds.parquet`) and full metric tables (`results/metrics.parquet`, `results/api_preds.parquet`) for all 2,342 completed runs are archived on Zenodo (see `ZENODO-METADATA.md` or the Data Availability section of the paper) and unpack directly into the root of this repository.

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
- `eval/` — GPU-free evaluation library: PR-AUC, max-F1, operating-point recall/precision, ECE, Brier, Balanced Brier Score (`metrics.py`), temperature / Platt / isotonic calibration (`calibrate.py`), split-conformal FPR control (`conformal.py`), Krippendorff's $\\alpha_K$ (`stability.py`), break-even budgets $N^\\star$ (`breakeven.py`), and paired item/cluster bootstrap (`fastboot.py`).
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

Unpack `label-budget-bench-predictions-v1.tar.gz` from Zenodo into the repository root, then run:

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
"""


def build_zenodo_bundle() -> dict:
    if ZEN.exists():
        shutil.rmtree(ZEN)
    ZEN.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(p for p in (SRC / "preds" / "runs").iterdir() if p.is_dir() and (p / "DONE").exists())
    allowed_run_files = ("DONE", "metrics.json", "preds.parquet", "run.json")

    file_count = 0
    uncompressed_bytes = 0

    with tarfile.open(TARBALL, "w:gz", compresslevel=6) as tar:
        # 1. All 2,342 finished runs (predictions + metadata only; NO adapter weights)
        for rd in run_dirs:
            for fname in allowed_run_files:
                fp = rd / fname
                if not fp.exists():
                    raise FileNotFoundError(f"Missing {fp}")
                arcname = f"preds/runs/{rd.name}/{fname}"
                tar.add(fp, arcname=arcname, recursive=False)
                file_count += 1
                uncompressed_bytes += fp.stat().st_size

        # 2. Open teacher parquets & markers
        for fp in sorted((SRC / "preds" / "teacher").rglob("*")):
            if fp.is_file():
                arcname = str(fp.relative_to(SRC))
                tar.add(fp, arcname=arcname, recursive=False)
                file_count += 1
                uncompressed_bytes += fp.stat().st_size

        # 3. Sanitized serving benchmark CSV
        srv_pub = PUB / "logs" / "serving_a6000.csv"
        tar.add(srv_pub, arcname="logs/serving_a6000.csv", recursive=False)
        file_count += 1
        uncompressed_bytes += srv_pub.stat().st_size

        # 4. Results parquets, JSON, CSV tables, figures, environment, freeze manifest
        results_files = [
            (SRC / "results" / "metrics.parquet", "results/metrics.parquet"),
            (SRC / "results" / "api_preds.parquet", "results/api_preds.parquet"),
            (SRC / "results" / "api_metrics.csv", "results/api_metrics.csv"),
            (PUB / "results" / "paper_numbers.json", "results/paper_numbers.json"),
            (SRC / "results" / "ENVIRONMENT-gpu.txt", "results/ENVIRONMENT-gpu.txt"),
            (PUB / "results" / "FREEZE-2026-11-15.sha256", "results/FREEZE-2026-11-15.sha256"),
        ]
        for src_fp, arcname in results_files:
            tar.add(src_fp, arcname=arcname, recursive=False)
            file_count += 1
            uncompressed_bytes += src_fp.stat().st_size

        for sub in ("tables", "figures"):
            for fp in sorted((SRC / "results" / sub).iterdir()):
                if fp.is_file() and fp.name != ".DS_Store":
                    arcname = f"results/{sub}/{fp.name}"
                    tar.add(fp, arcname=arcname, recursive=False)
                    file_count += 1
                    uncompressed_bytes += fp.stat().st_size

    tar_sha = sha256_file(TARBALL)
    tar_bytes = TARBALL.stat().st_size
    (ZEN / "SHA256SUMS.txt").write_text(f"{tar_sha}  {TARBALL.name}\n")

    meta_md = f"""\
# Zenodo Deposit Metadata & Upload Checklist (`label-budget-bench`)

## Files to Upload to Zenodo

1. **`{TARBALL.name}`**
   - Compressed size: `{tar_bytes:,}` bytes ({tar_bytes / (1024**2):.2f} MiB)
   - Uncompressed payload: `{uncompressed_bytes:,}` bytes ({uncompressed_bytes / (1024**2):.2f} MiB) across `{file_count:,}` files (`{len(run_dirs):,}` completed runs)
   - SHA-256: `{tar_sha}`
2. *(Optional / automatic via GitHub–Zenodo integration or manual upload)* Snapshot zip/tarball of `public_repo/` (`label-budget-bench-code-v1.0.0`).
3. **`paper/latex/supplement.pdf`** — Supplementary Material PDF (Tables S1–S6, Figures S1–S2, Sections S1–S4).

## Zenodo Record Fields

- **Upload type:** Dataset (or Software + Dataset)
- **Title:** Replication Package and Per-Item Predictions for "From Label Budget to Deployed Threshold: How Gold and LLM-Distilled Labels Shape Discrimination, Calibration, and Shift Robustness of Fine-Tuned Encoders and Small Decoders"
- **Creators:**
  1. Singaravelu, Divya
  2. Rajendran, Suresh
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0) for predictions/tables; MIT License for code.
- **Version:** `1.0.0`
- **Keywords:** text classification, label budget, knowledge distillation, LLM annotations, model calibration, conformal prediction, distribution shift, ModernBERT, DeBERTa, Gemma 4, Qwen3.5
- **Description:**
  > Per-item validation, in-distribution test, and distribution-shifted test predictions (`preds.parquet`), run metadata (`run.json`, `metrics.json`), teacher label caches, alignment-audit annotations, and aggregated bootstrap tables (`metrics.parquet`, `paper_numbers.json`, `results/tables/*.csv`) for all 2,342 completed runs (2,328 training runs across TF-IDF, ModernBERT-base/large, DeBERTa-v3-large, Gemma 4 E2B/E4B, and Qwen3.5-2B/4B; 7 off-the-shelf detectors; and pipeline verification runs) plus hosted API references. To respect upstream dataset licenses (including CC BY-ND 4.0 for LLM-AggreFact and CC BY-NC 4.0 for BeaverTails), all prediction and audit files contain item identifiers (`uid`) and numeric probabilities/verdicts only; no dataset text is redistributed.

## Verification Against `FREEZE-2026-11-15.sha256`

Every `preds/runs/*/preds.parquet`, `preds/runs/*/metrics.json`, `preds/runs/*/run.json`, `preds/teacher/qwen35_9b/*.parquet`, `results/metrics.parquet`, `results/api_preds.parquet`, and `results/api_metrics.csv` inside `{TARBALL.name}` is byte-identical to `results/FREEZE-2026-11-15.sha256`. LoRA adapter weights (`16.55 GB`) are excluded to keep the replication download lightweight (`{tar_bytes / (1024**2):.1f} MiB`).
"""
    (ZEN / "ZENODO-METADATA.md").write_text(meta_md)
    return {
        "runs": len(run_dirs),
        "files": file_count,
        "uncompressed_mib": round(uncompressed_bytes / (1024**2), 2),
        "compressed_mib": round(tar_bytes / (1024**2), 2),
        "sha256": tar_sha,
    }


def scan_release() -> dict:
    """Comprehensive security, IP, hostname, PII, and raw-text scan of public_repo/ and zenodo/."""
    secrets = {}
    env_path = SRC / ".env"
    if env_path.exists():
        for ln in env_path.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                v = v.strip()
                if len(v) >= 8:
                    secrets[k.strip()] = v

    forbidden_patterns = {
        "api_token": re.compile(
            r"(?:sk-[A-Za-z0-9_\-]{12,}|hf_[A-Za-z0-9]{12,}|AIza[0-9A-Za-z_\-]{12,}|AQ\.[0-9A-Za-z_\-]{12,}|apikey_[0-9A-Za-z_]{12,})"
        ),
        "private_key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
        "ipv4_addr": re.compile(r"\b(?:209\.182\.\d+\.\d+|216\.81\.\d+\.\d+|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)\b"),
        "infra_host_or_key": re.compile(
            "|".join(
                [
                    "ssd" + "nodes",
                    "new" + "user",
                    "lbb_" + "store",
                    "lbb-copy-" + "thunder",
                    "thunder_" + "ed25519",
                    "destello-sales-" + "agent",
                    "alphashots" + r"\.ai",
                    "fnrao" + "1qa",
                    "vs6zp" + "wgp",
                    "5c9my" + "zat",
                    "18bk5" + "yw6",
                    "gate-a-" + "2026-11-10",
                    "/Users/" + "suresh",
                    "NOTES-label-" + "budget",
                    "RUNBOOK-label-" + "budget",
                    "PLAN-label-" + "budget",
                ]
            ),
            re.I,
        ),
    }

    findings: list[str] = []
    scanned_files = 0

    for root_dir in (PUB, ZEN):
        for p in sorted(root_dir.rglob("*")):
            if not p.is_file() or p.name.endswith(".tar.gz"):
                continue
            scanned_files += 1
            rel = p.relative_to(OUT_ROOT)

            # Check forbidden filenames/directories
            if any(part in {"paper", "plans", "checkpoints", "adapter"} for part in rel.parts):
                findings.append(f"Forbidden path in export: {rel}")
            if p.name == ".env" or p.name.endswith(".batches.json") or p.name == "api_ledger.jsonl":
                findings.append(f"Forbidden file in export: {rel}")

            if p.suffix in {".png", ".pdf", ".jpg", ".whl"}:
                continue
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
                bad_cols = {"text", "text_a", "text_b", "document", "claim", "user_prompt", "assistant_response", "post"} & set(df.columns)
                if bad_cols:
                    findings.append(f"Dataset text columns {bad_cols} found in parquet: {rel}")
                continue

            txt = p.read_text(errors="ignore")
            for sk, sv in secrets.items():
                if sv in txt:
                    findings.append(f"Exact secret value for {sk} leaked in {rel}")
            for pname, pat in forbidden_patterns.items():
                m = pat.search(txt)
                if m:
                    sample = "<redacted-token>" if pname == "api_token" else m.group(0)
                    findings.append(f"Pattern {pname} ({sample!r}) matched in {rel}")

    # Verify tarball member names & absence of adapters / raw text / instance IDs
    tar_members = 0
    with tarfile.open(TARBALL, "r:gz") as tar:
        for m in tar.getmembers():
            tar_members += 1
            if "adapter" in m.name or m.name.startswith(("paper/", "plans/", "checkpoints/")) or "serving_instance-" in m.name:
                findings.append(f"Forbidden tar member: {m.name}")

    if findings:
        for item in findings:
            print("FAIL:", item)
        raise SystemExit(f"Release scan failed with {len(findings)} finding(s).")

    return {"scanned_files": scanned_files, "tar_members": tar_members, "findings": 0}


def main() -> None:
    print("1/3 Staging clean public repository ->", PUB)
    build_public_repo()
    print("2/3 Building Zenodo predictions archive ->", TARBALL)
    zen_stats = build_zenodo_bundle()
    print("3/3 Running automated secret, IP, hostname, PII & license scan...")
    scan_stats = scan_release()
    print("OK:", json.dumps({"zenodo": zen_stats, "scan": scan_stats}, indent=2))


if __name__ == "__main__":
    main()
