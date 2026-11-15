"""Project total A6000-hours for queue/jobs.yaml from measured Gate A runs (§6.8 rule 8).

Seconds per token are measured on the Gate A cells (T2, N=1k) for one encoder and one decoder,
training at steady state (one-off kernel compilation excluded via the step log) and eval over
all eval tokens. Other cells scale by parameter ratio x real tokenized length: the decoder
prompt + chat template overhead is included, training batches are padded to their longest
item, and each task's eval sets are counted at their real sizes. Attention's quadratic term is
ignored (linear in tokens), which under-states T1 slightly.

  python analysis/gpu_projection.py
  python analysis/gpu_projection.py --decoder-train-tag nockpt_conv --decoder-eval-tag nockpt
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from common import MODELS, TRAIN_CFG, RunSpec, hyperparams, load_all, n_steps  # noqa: E402

PARAMS_B = {"modernbert_base": 0.149, "modernbert_large": 0.395, "deberta_v3_large": 0.435,
            "gemma4_e2b": 2.0, "qwen35_2b": 2.0, "gemma4_e4b": 4.0, "qwen35_4b": 4.0}
REF = {"encoder": "modernbert_base", "decoder": "qwen35_2b"}
TOKENIZER = {"encoder": "answerdotai/ModernBERT-base", "decoder": "Qwen/Qwen3.5-2B"}
SAMPLE = 1500
# measured step time relative to the parameter-ratio estimate (D24): Gemma 4 E2B 3.04 s (4 x 4 micro-batches, no
# checkpointing) vs Qwen3.5-2B 1.44 s per step at T2; E4B assumed to carry the same factor
STEP_FACTOR = {"gemma4_e2b": 3.04 / 1.44, "gemma4_e4b": 3.04 / 1.44}


def run_dir(run_id: str) -> Path:
    """Gate A reference runs live in preds/archive_gate_a once the main sweep has replaced them."""
    for d in (ROOT / "preds" / "runs" / run_id, ROOT / "preds" / "archive_gate_a" / run_id):
        if (d / "run.json").exists():
            return d
    return ROOT / "preds" / "runs" / run_id


def steady_step_seconds(run_id: str, fallback: float) -> float:
    log = ROOT / "logs" / f"{run_id}.log"
    pts = [(int(m[1]), float(m[2])) for m in re.finditer(r"step (\d+)/\d+ .*?\(([\d.]+) s/step\)",
                                                          log.read_text())] if log.exists() else []
    pts = [p for p in pts if p[0] >= 50]
    if len(pts) < 2:
        return fallback
    (k, a), (n, b) = pts[0], pts[-1]
    return (n * b - k * a) / (n - k)


def lengths(family: str, task: str, cap: int) -> tuple[np.ndarray, dict[str, tuple[int, float]]]:
    """Token lengths of a pool sample, and (count, mean tokens) per eval split."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER[family])
    df = load_all(task)
    if family == "decoder":
        from decoder_lora import PromptEncoder
        enc = PromptEncoder(tok, task, cap)
        count = lambda d: np.array([len(x) for x in enc.many(d)])  # noqa: E731
    else:
        def count(d):
            b = d.text_b.fillna("") if "text_b" in d else [""] * len(d)
            return np.array([min(cap, len(tok(a, s)["input_ids"]) if s else len(tok(a)["input_ids"]))
                             for a, s in zip(d.text_a, b)])
    sample = lambda d: d.sample(min(SAMPLE, len(d)), random_state=0)  # noqa: E731
    pool = count(sample(df[df.split == "pool"]))
    ev = {s: (int((df.split == s).sum()), float(count(sample(df[df.split == s])).mean()))
          for s in ("val", "test", "shift")}
    return pool, ev


def padded_per_item(lens: np.ndarray, batch: int, micro: int, rng) -> float:
    draws = [lens[rng.choice(len(lens), micro)].max() for _ in range(400)]
    return float(np.mean(draws))


def main():
    ap = argparse.ArgumentParser()
    for fam in REF:
        ap.add_argument(f"--{fam}-train-tag", default="", help="run tag of the reference cell for step time")
        ap.add_argument(f"--{fam}-eval-tag", default="", help="run tag of the reference cell for eval time")
    ap.add_argument("--show", nargs="*", default=[], help="job ids whose estimate to print (compare with measured)")
    ap.add_argument("--out", help="write the per-job estimate (id, task, family, model, hours) to this CSV")
    a = vars(ap.parse_args())
    jobs = yaml.safe_load((ROOT / "queue" / "jobs.yaml").read_text())
    rng = np.random.default_rng(0)
    enc_cfg = MODELS["open"]["encoders"]
    cache, rows = {}, []

    def stats(family, model, task):
        cap = TRAIN_CFG["max_len"][task]
        if family == "encoder":
            cap = min(cap, enc_cfg[model]["max_len"])
        key = (family, task, cap)
        if key not in cache:
            pool, ev = lengths(family, task, cap)
            cache[key] = (pool, ev, cap)
        return cache[key]

    meas = {}
    for fam, model in REF.items():
        rids = {}
        for part in ("train", "eval"):
            tag = a[f"{fam}_{part}_tag"]
            rids[part] = f"t2_response_safety__{model}__gold__n1000__s0" + (f"__{tag}" if tag else "")
            if not run_dir(rids[part]).exists():
                print(f"missing Gate A run for {model}: {rids[part]}")
                return 1
        rt, re_ = (json.loads((run_dir(rids[p]) / "run.json").read_text()) for p in ("train", "eval"))
        spec = RunSpec(fam, model, "t2_response_safety", 1000, 0, "gold")
        hp = hyperparams(spec)
        pool, ev, _ = stats(fam, model, "t2_response_safety")
        micro = max(1, min(hp["batch"], TRAIN_CFG.get("train_batch_tokens", 16384) // max(int(np.max(pool)), 1)))
        step = steady_step_seconds(rids["train"], rt["train_seconds"] / rt["train_steps"])
        tok_step = hp["batch"] * padded_per_item(pool, hp["batch"], micro, rng)
        ev_tok = sum(n * m for n, m in ev.values())
        meas[fam] = dict(train=step / tok_step, eval=re_["eval_seconds"] / ev_tok)
        print(f"{model}: {step:.2f} s/step steady ({rids['train']}), {tok_step:.0f} tokens/step; "
              f"eval {re_['eval_seconds']:.0f} s for {ev_tok / 1e6:.2f}M tokens ({rids['eval']})")

    for j in jobs:
        if j["family"] == "classical":
            continue
        spec = RunSpec(j["family"], j["model"], j["task"], j["n"], j["seed"], j["labels"])
        hp = hyperparams(spec)
        pool, ev, _ = stats(j["family"], j["model"], j["task"])
        L = int(np.max(pool))
        micro = max(1, min(hp["batch"], TRAIN_CFG.get("train_batch_tokens", 16384) // max(L, 1)))
        scale = PARAMS_B[j["model"]] / PARAMS_B[REF[j["family"]]] * STEP_FACTOR.get(j["model"], 1.0)
        m = meas[j["family"]]
        train_tok = n_steps(spec, hp["batch"]) * hp["batch"] * padded_per_item(pool, hp["batch"], micro, rng)
        eval_tok = sum(n * mt for n, mt in ev.values())
        rows.append(dict(id=j["id"], task=j["task"][:2], family=j["family"], model=j["model"],
                         train_s_per_step=train_tok * m["train"] * scale / n_steps(spec, hp["batch"]),
                         train_h=train_tok * m["train"] * scale / 3600,
                         eval_h=eval_tok * m["eval"] * scale / 3600))
    df = pd.DataFrame(rows)
    df["hours"] = df.train_h + df.eval_h
    if a.get("out"):
        df[["id", "task", "family", "model", "hours"]].to_csv(a["out"], index=False)
    if a["show"]:
        print(df[df.id.isin(a["show"])][["id", "train_s_per_step", "train_h", "eval_h"]].round(3).to_string(index=False))
    print()
    print(df.groupby(["task", "family"])[["train_h", "eval_h", "hours"]].sum().round(1).to_string())
    tot = df.hours.sum()
    print(f"\nprojected total: {tot:.0f} A6000-hours (train {df.train_h.sum():.0f}, eval {df.eval_h.sum():.0f}) "
          f"({'<= 450 h: launch the 2nd A6000 in Week 3' if tot <= 450 else '> 450 h: apply §6.8 levers before scaling'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
