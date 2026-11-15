"""Open teacher (Qwen3.5-9B, D13): zero-shot labels with the detailed prompt, no training.

Labels the training pool plus val/test/shift (the last three only for the teacher-quality table).
Score = renormalized first-token P(Yes) at temperature 0; verdict = P(Yes) >= 0.5.
Uses vLLM when installed (GPU), else Hugging Face forward passes (CPU smoke tests).
Resumable: already-labelled uids in the output file are skipped.

  python train/teacher.py --task t2_response_safety                 # GPU, vLLM
  python train/teacher.py --task t3_implicit_toxicity --model qwen35_08b_smoke --limit 20 --max-len 256
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, ROOT, TRAIN_CFG, device, load_all  # noqa: E402

CHUNK = 2000


def model_cfg(key):
    return MODELS["open"]["teacher"].get(key) or MODELS["open"]["decoders"][key]


def out_path(key, task):
    return ROOT / "preds" / "teacher" / key / f"{task}.parquet"


_LLM = {}


def score_vllm(hf_id, prompts, yes_no, enc_len):
    from vllm import LLM, SamplingParams
    if hf_id not in _LLM:
        # enforce_eager: CUDA-graph capture hangs on Thunder's virtualized GPU, and one-token scoring is all prefill.
        _LLM[hf_id] = LLM(model=hf_id, max_model_len=enc_len + 128, dtype="bfloat16", enable_prefix_caching=True,
                          gpu_memory_utilization=0.92, enforce_eager=True,
                          limit_mm_per_prompt={"image": 0, "video": 0})
    llm = _LLM[hf_id]
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    res = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
    out = []
    for r in res:
        lp = r.outputs[0].logprobs[0]
        y = math.exp(lp[yes_no[0]].logprob) if yes_no[0] in lp else 0.0
        n = math.exp(lp[yes_no[1]].logprob) if yes_no[1] in lp else 0.0
        out.append(y / (y + n) if y + n > 0 else float("nan"))
    return out


def score_hf(hf_id, prompts, yes_no, pad_id):
    import torch

    import decoder_lora as dl
    dev = device()
    model = dl.load_model(hf_id, dev, torch.bfloat16 if dev.type == "cuda" else torch.float32).eval()
    return list(dl.predict(model, prompts, pad_id, yes_no[0], yes_no[1], dev, TRAIN_CFG["eval_batch_tokens"]))


def main() -> int:
    from transformers import AutoTokenizer

    import decoder_lora as dl
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default="qwen35_9b")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    a = ap.parse_args()
    cfg = model_cfg(a.model)
    L = a.max_len or TRAIN_CFG["max_len"][a.task]
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    yes_no = dl.answer_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = dl.PromptEncoder(tok, a.task, L, kind="detailed")

    df = load_all(a.task)
    order = {"pool": 0, "val": 1, "test": 2, "shift": 3}
    df = df.sort_values("split", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)
    path = out_path(a.model, a.task)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["uid", "split", "p_yes", "verdict"])
    todo = df[~df.uid.isin(set(done.uid))]
    backend = a.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except ImportError:
            backend = "hf"
    print(f"{a.task}: {len(todo)} items to label with {cfg['hf_id']} ({backend})", flush=True)
    for start in range(0, len(todo), CHUNK):
        chunk = todo.iloc[start:start + CHUNK]
        prompts = enc.many(chunk)
        p = score_vllm(cfg["hf_id"], prompts, yes_no, enc.L) if backend == "vllm" else score_hf(cfg["hf_id"], prompts, yes_no, pad_id)
        new = pd.DataFrame({"uid": chunk.uid.values, "split": chunk.split.values, "p_yes": p})
        new["verdict"] = (new.p_yes >= 0.5).astype("Int64").where(new.p_yes.notna())
        done = pd.concat([done, new], ignore_index=True)
        done.to_parquet(path, index=False)
        print(f"  labelled {len(done)}/{len(df)}", flush=True)
    if not a.limit and set(df.uid) <= set(done.uid):
        path.with_suffix(".done").write_text(f"{len(done)}\n")
    lab = done.merge(df[["uid", "label"]], on="uid")
    for split, g in lab.groupby("split"):
        acc = (g.verdict == g.label).mean()
        print(f"  teacher vs gold on {split}: accuracy {acc:.3f}, positive rate {g.verdict.mean():.3f} (gold {g.label.mean():.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
