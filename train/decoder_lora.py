"""Family B: LoRA fine-tuning of small decoders as Yes/No classifiers.

The input is the task's concise prompt through the model's chat template (thinking disabled).
The score is the renormalized first-token probability P(Yes) / (P(Yes) + P(No)) at the
generation position; training minimizes cross-entropy on that 2-way distribution, so the
training objective and the evaluated probability are the same quantity.
"""
from __future__ import annotations

import re
import time

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

import loop
from common import (MODELS, ROOT, TRAIN_CFG, device, eval_sets, gpu_info, hyperparams, max_len,
                    n_steps, set_seed, training_set)

VISION = r".*(vision|visual|audio|image|multi_modal_projector|embed_vision|embed_audio).*"


def load_model(hf_id: str, dev, dtype):
    try:
        m = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(hf_id, dtype=dtype)
    return m.to(dev)


def answer_ids(tok) -> tuple[int, int]:
    ids = {}
    for w in ("Yes", "No"):
        t = tok.encode(w, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"'{w}' is {len(t)} tokens for this tokenizer; pick another model")
        ids[w] = t[0]
    return ids["Yes"], ids["No"]


class PromptEncoder:
    """`item_len` budgets the item text only; the instruction and chat template come on top
    (they exceed T3's 256-token budget on their own). Sequences are never cut at the end,
    because the generation prompt there is where P(Yes) is read."""

    def __init__(self, tok, task: str, item_len: int, kind: str = "concise"):
        p = yaml.safe_load((ROOT / "labels" / "prompts" / f"{task}.yaml").read_text())
        self.tok, self.instr, self.tpl = tok, p[kind], p["item_template"]
        self.overhead = len(self._ids("", ""))
        self.L = self.overhead + item_len

    def _text(self, a: str, b: str) -> str:
        msgs = [{"role": "system", "content": self.instr},
                {"role": "user", "content": self.tpl.format(text_a=a, text_b=b)}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except Exception:
            msgs = [{"role": "user", "content": self.instr + "\n\n" + msgs[1]["content"]}]
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _ids(self, a: str, b: str) -> list[int]:
        return self.tok(self._text(a, b), add_special_tokens=False)["input_ids"]

    def _cut(self, s: str, over: int) -> str:
        t = self.tok(s, add_special_tokens=False)["input_ids"]
        return self.tok.decode(t[: max(16, len(t) - over - 8)]) + " [...]"

    def __call__(self, a: str, b: str) -> list[int]:
        b = b or ""
        ids = self._ids(a, b)
        if len(ids) > self.L:                      # truncate the context (text_a) first
            a = self._cut(a, len(ids) - self.L)
            ids = self._ids(a, b)
        if len(ids) > self.L and b:                # then the judged text (text_b)
            b = self._cut(b, len(ids) - self.L)
            ids = self._ids(a, b)
        assert len(ids) <= self.L + 64, (len(ids), self.L)
        return ids

    def many(self, df) -> list[list[int]]:
        return [self(a, b) for a, b in zip(df.text_a, df.text_b.fillna("") if "text_b" in df else [""] * len(df))]


def _pad_left(seqs, pad_id, dev):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        mask[i, L - len(s):] = 1
    return ids.to(dev), mask.to(dev)


def yes_no_logits(model, ids, mask, yes_id, no_id):
    try:
        out = model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=ids, attention_mask=mask)
    last = out.logits[:, -1, :].float()
    return torch.stack([last[:, no_id], last[:, yes_id]], dim=-1)


class HeadModel(torch.nn.Module):
    """Classification-head ablation: a 2-way linear head on the last hidden state at the generation
    position, instead of reading P(Yes) from the LM head."""

    def __init__(self, base, hidden: int):
        super().__init__()
        self.base = base
        self.head = torch.nn.Linear(hidden, 2).to(next(base.parameters()).device)
        torch.nn.init.normal_(self.head.weight, std=0.02)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, logits_to_keep=1)
        return self.head(out.hidden_states[-1][:, -1, :].float())

    def gradient_checkpointing_enable(self, **kw):
        self.base.gradient_checkpointing_enable(**kw)

    def enable_input_require_grads(self):
        self.base.enable_input_require_grads()


@torch.no_grad()
def predict(model, seqs, pad_id, yes_id, no_id, dev, budget_tokens: int, score2=None) -> np.ndarray:
    lens = np.array([len(s) for s in seqs])
    order = np.argsort(-lens)
    out = np.empty(len(seqs))
    i = 0
    while i < len(order):
        bs = max(1, budget_tokens // max(lens[order[i]], 1))
        idx = order[i:i + bs]
        ids, mask = _pad_left([seqs[j] for j in idx], pad_id, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = score2(model, ids, mask) if score2 else yes_no_logits(model, ids, mask, yes_id, no_id)
        out[idx] = torch.softmax(lg.float(), -1)[:, 1].cpu().numpy()
        i += bs
    return out


def fits_without_ckpt(model, collate, loss_fn, seqs, micro, headroom=0.9) -> bool:
    """One forward/backward on the `micro` longest items without gradient checkpointing.
    Checkpointing changes speed and memory only, not the gradients; RNG state is preserved."""
    idx = np.argsort([-len(s) for s in seqs])[:micro]
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model, collate(idx))
            loss.backward()
        ok = torch.cuda.max_memory_allocated() < headroom * total
    except torch.cuda.OutOfMemoryError:
        ok = False
    model.zero_grad(set_to_none=True)
    loss = None
    torch.cuda.empty_cache()
    return ok


def run(spec) -> dict:
    from peft import LoraConfig, get_peft_model

    set_seed(spec.seed)
    cfg = MODELS["open"]["decoders"].get(spec.model) or MODELS["open"]["teacher"][spec.model]
    hp = hyperparams(spec)
    dev = device()
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    L = max_len(spec, 32768)
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    yes_id, no_id = answer_ids(tok)
    full_sft = not hp.get("lora", True)                    # ablation: full fine-tuning, fp32 master weights
    model = load_model(cfg["hf_id"], dev, torch.float32 if full_sft and dev.type == "cuda" else dtype)
    if full_sft:
        for n, p_ in model.named_parameters():
            p_.requires_grad = not re.match(VISION, n)
    else:
        lora = LoraConfig(r=hp["lora_r"], lora_alpha=hp["lora_alpha"], lora_dropout=hp["lora_dropout"],
                          target_modules="all-linear", exclude_modules=VISION, task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    head = hp.get("scoring", "logprob") == "head"          # ablation: classification head
    if head:
        mc = model.config
        model = HeadModel(model, getattr(mc, "text_config", mc).hidden_size)
        score2 = lambda m, ids, mask: m(ids, mask)  # noqa: E731
    else:
        score2 = lambda m, ids, mask: yes_no_logits(m, ids, mask, yes_id, no_id)  # noqa: E731
    enc = PromptEncoder(tok, spec.task, L)
    train_df = training_set(spec)
    seqs = enc.many(train_df)
    labels = torch.tensor(train_df.label.to_numpy(dtype="int64"), device=dev)
    weights = torch.tensor((train_df.w / train_df.w.mean()).to_numpy(dtype="float32"), device=dev)
    steps = n_steps(spec, hp["batch"])
    micro = max(1, min(hp["batch"], hp.get("micro_max", hp["batch"]), TRAIN_CFG.get("train_batch_tokens", 16384) // enc.L))

    def collate(idx):
        ids, mask = _pad_left([seqs[j] for j in idx], pad_id, dev)
        return ids, mask, labels[idx], weights[idx]

    def loss_fn(m, b):
        ids, mask, y, w = b
        return (torch.nn.functional.cross_entropy(score2(m, ids, mask).float(), y, reduction="none") * w).mean()

    ckpt = hp.get("grad_ckpt", "auto")
    if ckpt == "auto":
        # the probe runs before AdamW allocates its fp32 states, which full fine-tuning cannot afford to ignore
        ckpt = dev.type == "cuda" and (full_sft or not fits_without_ckpt(model, collate, loss_fn, seqs, micro))
    if ckpt and dev.type == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    info = loop.train(spec, model, hp, steps, len(train_df), micro, collate, loss_fn, dev)
    info["grad_ckpt"] = bool(ckpt)
    t0 = time.time()
    preds = {}
    for split, df in eval_sets(spec).items():
        preds[split] = (df, predict(model, enc.many(df), pad_id, yes_id, no_id, dev,
                                    TRAIN_CFG["eval_batch_tokens"], score2))
    if not full_sft:
        (model.base if head else model).save_pretrained(spec.out / "adapter")
        if head:
            torch.save(model.head.state_dict(), spec.out / "adapter" / "head.pt")
    info.update(eval_seconds=round(time.time() - t0, 1), n_train=len(train_df), max_len=L,
                hf_id=cfg["hf_id"], hp=hp, trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                **gpu_info())
    return preds, info
