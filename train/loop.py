"""Crash-safe training loop shared by encoder and decoder families.

Resume semantics: the batch order for all steps is fixed from the seed before training, and a
checkpoint (trainable weights, optimizer, scheduler, step, RNG) is written atomically every
`ckpt_every` steps. After a kill -9 the same command resumes at the last checkpoint and replays
exactly the batches it would have seen.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from common import TRAIN_CFG, batch_plan, load_ckpt, save_ckpt


def trainable_state(model) -> dict:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def train(spec, model, hp: dict, steps: int, n_items: int, micro: int, collate, loss_fn, dev) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp.get("weight_decay", 0.0))
    sched = get_linear_schedule_with_warmup(opt, int(hp.get("warmup", 0.06) * steps), steps)
    start, losses = 0, []
    st = load_ckpt(spec.ckpt)
    if st is not None:
        missing = model.load_state_dict(st["model"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng"])
        start, losses = st["step"], st["losses"]
        print(f"  resumed from checkpoint at step {start}/{steps}", flush=True)
    plan = batch_plan(n_items, hp["batch"], steps, spec.seed)
    every = TRAIN_CFG["ckpt_every"] if not spec.max_steps else max(1, steps // 3)
    use_amp = dev.type == "cuda"
    model.train()
    t0 = time.time()
    for step in range(start, steps):
        idx = plan[step]
        total = 0.0
        for chunk in np.array_split(idx, max(1, int(np.ceil(len(idx) / micro)))):
            batch = collate(chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model, batch) * (len(chunk) / len(idx))
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(total)
        if (step + 1) % 25 == 0 or step + 1 == steps:
            print(f"  step {step + 1}/{steps} loss {np.mean(losses[-25:]):.4f} "
                  f"({(time.time() - t0) / (step + 1 - start):.2f} s/step)", flush=True)
        if (step + 1) % every == 0 and step + 1 < steps:
            save_ckpt(spec.ckpt, dict(model=trainable_state(model), opt=opt.state_dict(),
                                      sched=sched.state_dict(), step=step + 1, losses=losses,
                                      rng=torch.get_rng_state()))
    model.eval()
    return dict(train_steps=steps, resumed_from=start, final_loss=float(np.mean(losses[-25:])),
                train_seconds=round(time.time() - t0, 1))
