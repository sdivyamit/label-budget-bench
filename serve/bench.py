"""Serving table (RQ4, reduced to one table): scoring speed of each fine-tunable model on one A6000.

  python serve/bench.py                       # all models, all tasks -> logs/serving_<host>.csv
  python serve/bench.py --models qwen35_2b --tasks t2_response_safety --limit 64   # smoke test

Per model and task: (a) batched throughput over the whole test set with the same length-sorted batching and
token budget as the evaluation of every run; (b) single-request latency (batch size 1, sequential) over the
first `--latency-n` test items, p50 / p95 / p99. Weights are the base checkpoints with a fresh 2-way head
(encoders) or the LM head (decoders; a merged LoRA adapter has the base model's cost), in bf16 as in evaluation.
Speed does not depend on the trained weights. Cost per million items uses the measured A6000 price (D27).
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
from common import MODELS, TRAIN_CFG, device, load_all  # noqa: E402

USD_PER_GPU_HOUR = 0.366
TASKS = ["t1_rag_hallucination", "t2_response_safety", "t3_implicit_toxicity"]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def encoder_bench(model_key, task, df, dev, latency_n):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import encoder
    cfg = MODELS["open"]["encoders"][model_key]
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    # as in evaluation: fp32 weights under bf16 autocast (DeBERTa-v3's checkpoint would otherwise load as fp16)
    model = AutoModelForSequenceClassification.from_pretrained(cfg["hf_id"], num_labels=2,
                                                               dtype=torch.float32).to(dev).eval()
    L = min(cfg["max_len"], TRAIN_CFG["max_len"][task])
    enc = encoder._encode(tok, df, L)
    budget = TRAIN_CFG["eval_batch_tokens"]
    encoder.predict(model, {k: v[:8] for k, v in enc.items()}, dev, budget)          # warm-up
    sync(); t0 = time.time()
    encoder.predict(model, enc, dev, budget)
    sync(); thr = len(df) / (time.time() - t0)
    lat = []
    for i in range(min(latency_n, len(df))):
        one = {k: v[i:i + 1] for k, v in enc.items()}
        sync(); t0 = time.time()
        encoder.predict(model, one, dev, budget)
        sync(); lat.append(time.time() - t0)
    n_tok = float(np.mean([len(x) for x in enc["input_ids"]]))
    return thr, lat, n_tok


def decoder_bench(model_key, task, df, dev, latency_n):
    from transformers import AutoTokenizer
    import decoder_lora as dl
    cfg = MODELS["open"]["decoders"][model_key]
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    model = dl.load_model(cfg["hf_id"], dev, torch.bfloat16 if dev.type == "cuda" else torch.float32).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    yes, no = dl.answer_ids(tok)
    seqs = dl.PromptEncoder(tok, task, TRAIN_CFG["max_len"][task]).many(df)
    budget = TRAIN_CFG["eval_batch_tokens"]
    dl.predict(model, seqs[:8], pad, yes, no, dev, budget)                                # warm-up
    sync(); t0 = time.time()
    dl.predict(model, seqs, pad, yes, no, dev, budget)
    sync(); thr = len(df) / (time.time() - t0)
    lat = []
    for i in range(min(latency_n, len(df))):
        sync(); t0 = time.time()
        dl.predict(model, seqs[i:i + 1], pad, yes, no, dev, budget)
        sync(); lat.append(time.time() - t0)
    return thr, lat, float(np.mean([len(s) for s in seqs]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(list(MODELS["open"]["encoders"]) +
                                                 [m for m, c in MODELS["open"]["decoders"].items() if c["sweep"] != "none"]))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--limit", type=int, help="smoke test: first k test items")
    a = ap.parse_args()
    dev = device()
    host = socket.gethostname()
    out = ROOT / "logs" / f"serving_{host}{'_smoke' if a.limit else ''}.csv"
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev.model, prev.task))
    for m in a.models.split(","):
        fam = "encoder" if m in MODELS["open"]["encoders"] else "decoder"
        for task in a.tasks.split(","):
            if (m, task) in done:
                continue
            df = load_all(task)
            df = df[df.split == "test"].reset_index(drop=True)
            if a.limit:
                df = df.head(a.limit)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            thr, lat, ntok = (encoder_bench if fam == "encoder" else decoder_bench)(m, task, df, dev, a.latency_n)
            lat_ms = np.asarray(lat) * 1000
            row = dict(model=m, family=fam, task=task, items=len(df), mean_input_tokens=round(ntok, 1),
                       items_per_s=round(thr, 2), p50_ms=round(float(np.percentile(lat_ms, 50)), 1),
                       p95_ms=round(float(np.percentile(lat_ms, 95)), 1), p99_ms=round(float(np.percentile(lat_ms, 99)), 1),
                       usd_per_million_items=round(1e6 / thr / 3600 * USD_PER_GPU_HOUR, 3),
                       gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                       torch=torch.__version__, host=host, ts=time.strftime("%FT%T"))
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(row, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
