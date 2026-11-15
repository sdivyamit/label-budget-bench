"""Family A: full fine-tuning of encoders (ModernBERT, DeBERTa-v3) with a 2-way head."""
from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import loop
from common import MODELS, TRAIN_CFG, device, eval_sets, gpu_info, hyperparams, max_len, n_steps, set_seed, training_set


def _encode(tok, df, L):
    b = df.get("text_b")
    has_b = b is not None and b.fillna("").str.len().gt(0).any()
    return tok(df.text_a.tolist(), b.fillna("").tolist() if has_b else None,
               truncation="longest_first", max_length=L)


@torch.no_grad()
def predict(model, enc, dev, budget_tokens: int) -> np.ndarray:
    lens = np.array([len(x) for x in enc["input_ids"]])
    order = np.argsort(-lens)
    out = np.empty(len(lens))
    i = 0
    while i < len(order):
        bs = max(1, budget_tokens // max(lens[order[i]], 1))
        idx = order[i:i + bs]
        batch = {k: [enc[k][j] for j in idx] for k in enc.keys()}
        batch = _pad(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(**batch).logits.float()
        out[idx] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        i += bs
    return out


def _pad(batch: dict, dev) -> dict:
    L = max(len(x) for x in batch["input_ids"])
    res = {}
    for k, seqs in batch.items():
        pad = 0
        res[k] = torch.tensor([list(s) + [pad] * (L - len(s)) for s in seqs], device=dev)
    return res


def _windows(tok, df, L, overlap: int = 64):
    """Chunk-and-pool ablation (T1): split text_a (the context) into overlapping windows, each paired
    with the whole response (text_b, cut to L/2). Returns per-window encodings and their item index."""
    resp = [tok.decode(x[: L // 2]) for x in tok(list(df.text_b.fillna("")), add_special_tokens=False).input_ids]
    e = tok(list(df.text_a.fillna("")), resp, truncation="only_first", max_length=L, stride=overlap,
            return_overflowing_tokens=True)
    return ({"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]},
            np.asarray(e["overflow_to_sample_mapping"]))


def _pooled_margin(model, enc, owner, items, dev):
    """Per-item logit margin z1 - z0 = min over that item's windows (any supporting window suffices)."""
    sel = np.flatnonzero(np.isin(owner, items))
    batch = _pad({k: [enc[k][j] for j in sel] for k in enc}, dev)
    z = model(**batch).logits.float()
    margin = z[:, 1] - z[:, 0]
    pos = torch.as_tensor(np.searchsorted(items, owner[sel]), device=dev)
    out = torch.full((len(items),), float("inf"), device=dev)
    return out.scatter_reduce(0, pos, margin, reduce="amin")


def run_chunk_pool(spec, cfg, hp, dev, L) -> dict:
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["hf_id"], num_labels=2,
                                                               dtype=torch.float32).to(dev)
    train_df = training_set(spec)
    enc, owner = _windows(tok, train_df, L)
    labels = torch.tensor(train_df.label.to_numpy(dtype="float32"), device=dev)
    weights = torch.tensor((train_df.w / train_df.w.mean()).to_numpy(dtype="float32"), device=dev)
    steps = n_steps(spec, hp["batch"])
    micro = max(1, int(hp.get("chunk_micro", 1)))         # items per forward; a T1 item has up to ~11 windows

    def collate(idx):
        return np.sort(idx), labels[np.sort(idx)], weights[np.sort(idx)]

    def loss_fn(m, b):
        items, y, w = b
        bce = torch.nn.functional.binary_cross_entropy_with_logits(_pooled_margin(m, enc, owner, items, dev), y,
                                                                   reduction="none")
        return (bce * w).mean()

    info = loop.train(spec, model, hp, steps, len(train_df), micro, collate, loss_fn, dev)
    t0 = time.time()
    preds = {}
    with torch.no_grad():
        for split, df in eval_sets(spec).items():
            e, o = _windows(tok, df, L)
            p = np.empty(len(df))
            for s in range(0, len(df), 16):
                items = np.arange(s, min(s + 16, len(df)))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    p[items] = torch.sigmoid(_pooled_margin(model, e, o, items, dev)).cpu().numpy()
            preds[split] = (df, p)
    info.update(eval_seconds=round(time.time() - t0, 1), n_train=len(train_df), max_len=L, n_windows=len(owner),
                hf_id=cfg["hf_id"], hp=hp, **gpu_info())
    return preds, info


def run(spec) -> dict:
    set_seed(spec.seed)
    cfg = MODELS["open"]["encoders"][spec.model]
    hp = hyperparams(spec)
    if hp.get("chunk_pool"):
        dev = device()
        return run_chunk_pool(spec, cfg, hp, dev, max_len(spec, cfg["max_len"]))
    dev = device()
    L = max_len(spec, cfg["max_len"])
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    # fp32 master weights under bf16 autocast; transformers 5 would load DeBERTa-v3's fp16 checkpoint as fp16.
    model = AutoModelForSequenceClassification.from_pretrained(cfg["hf_id"], num_labels=2,
                                                               dtype=torch.float32).to(dev)
    train_df = training_set(spec)
    enc = _encode(tok, train_df, L)
    labels = torch.tensor(train_df.label.to_numpy(dtype="int64"), device=dev)
    weights = torch.tensor((train_df.w / train_df.w.mean()).to_numpy(dtype="float32"), device=dev)
    steps = n_steps(spec, hp["batch"])
    micro = max(1, min(hp["batch"], hp.get("micro_max", hp["batch"]), TRAIN_CFG.get("train_batch_tokens", 16384) // L))

    def collate(idx):
        return _pad({k: [enc[k][j] for j in idx] for k in enc.keys()}, dev), labels[idx], weights[idx]

    def loss_fn(m, b):
        inputs, y, w = b
        return (torch.nn.functional.cross_entropy(m(**inputs).logits.float(), y, reduction="none") * w).mean()

    info = loop.train(spec, model, hp, steps, len(train_df), micro, collate, loss_fn, dev)
    t0 = time.time()
    preds = {}
    for split, df in eval_sets(spec).items():
        preds[split] = (df, predict(model, _encode(tok, df, L), dev, TRAIN_CFG["eval_batch_tokens"]))
    info.update(eval_seconds=round(time.time() - t0, 1), n_train=len(train_df), max_len=L,
                hf_id=cfg["hf_id"], hp=hp, **gpu_info())
    return preds, info
