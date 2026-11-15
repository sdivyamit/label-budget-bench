"""Truncation table: share of items per task x split whose tokenized input exceeds the length the runs used.

Encoders: the (text_a, text_b) pair with special tokens against min(model max_len, task max_len), as
train/encoder.py feeds it (truncation longest_first). Decoders and the open teacher: the item text against
the task's item budget in configs/train.yaml (D19a: instruction and chat template come on top and are never
cut), counted as the tokens of the prompt's item template filled with the item minus those of the empty
template. Tokenizers come from the local Hugging Face cache only (transformers, else the cached
tokenizer.json); a model whose tokenizer is not cached is reported as not computed, never downloaded.

  python analysis/truncation.py        # -> results/tables/tableC_truncation.csv
"""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "tables"
MODELS = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["open"]
TRAIN = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())
TASKS = list(yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["tasks"])
SPLITS = ("pool", "val", "test", "shift")


class Lengths:
    """Token counts from a transformers tokenizer or a raw tokenizers.Tokenizer, no truncation."""

    def __init__(self, tok, raw: bool):
        self.tok, self.raw = tok, raw

    def __call__(self, a: list[str], b: list[str] | None, special: bool) -> np.ndarray:
        if self.raw:
            enc = self.tok.encode_batch(list(zip(a, b)) if b is not None else a, add_special_tokens=special)
            return np.array([len(e.ids) for e in enc])
        ids = self.tok(a, b, add_special_tokens=special, truncation=False, verbose=False)["input_ids"]
        return np.array([len(x) for x in ids])


def load(hf_id: str) -> tuple[Lengths | None, str]:
    from transformers import AutoTokenizer
    try:
        return Lengths(AutoTokenizer.from_pretrained(hf_id, local_files_only=True), raw=False), "transformers"
    except (OSError, ValueError, KeyError):
        pass
    from huggingface_hub import try_to_load_from_cache
    from tokenizers import Tokenizer
    f = try_to_load_from_cache(hf_id, "tokenizer.json")
    if not isinstance(f, str):
        return None, "not computed: tokenizer not in the local Hugging Face cache"
    tok = Tokenizer.from_file(f)
    tok.no_truncation()
    tok.no_padding()
    return Lengths(tok, raw=True), "tokenizer.json"


def main() -> int:
    from transformers.utils import logging
    logging.set_verbosity_error()
    data = {t: pd.read_parquet(ROOT / "data" / "processed" / f"{t}.parquet", columns=["text_a", "text_b", "split"])
            for t in TASKS}
    tpl = {t: yaml.safe_load((ROOT / "labels" / "prompts" / f"{t}.yaml").read_text())["item_template"] for t in TASKS}
    specs = [("encoder", m, s["hf_id"], s["max_len"]) for m, s in MODELS["encoders"].items()]
    specs += [("decoder", m, s["hf_id"], None) for m, s in MODELS["decoders"].items() if s.get("sweep") != "none"]
    specs += [("teacher", m, s["hf_id"], None) for m, s in MODELS["teacher"].items()]
    rows, cache = [], {}
    for family, model, hf_id, model_max in specs:
        if hf_id not in cache:
            cache[hf_id] = load(hf_id)
        count, how = cache[hf_id]
        for task in TASKS:
            d = data[task]
            L = min(model_max, TRAIN["max_len"][task]) if model_max else TRAIN["max_len"][task]
            base = dict(task=task, family=family, model=model, hf_id=hf_id, max_len=L,
                        counted="pair + special tokens" if family == "encoder" else "item text")
            if count is None:
                rows += [dict(base, split=s, n_items=int((d.split == s).sum()), status=how) for s in SPLITS]
                continue
            a, b = d.text_a.fillna("").tolist(), d.text_b.fillna("")
            if family == "encoder":
                n = count(a, b.tolist() if b.str.len().gt(0).any() else None, special=True)
            else:
                filled = [tpl[task].format(text_a=x, text_b=y) for x, y in zip(a, b)]
                n = count(filled, None, special=False) - count([tpl[task].format(text_a="", text_b="")], None, False)[0]
            for s in SPLITS:
                k = n[(d.split == s).to_numpy()]
                rows.append(dict(base, split=s, n_items=len(k), n_truncated=int((k > L).sum()),
                                 share_truncated=float((k > L).mean()), p50_tokens=int(np.median(k)),
                                 p95_tokens=int(np.percentile(k, 95)), max_tokens=int(k.max()),
                                 status=f"computed ({how})"))
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "tableC_truncation.csv", index=False, float_format="%.4f")
    print(t.pivot_table(index=["family", "model"], columns=["task", "split"], values="share_truncated")
          .round(3).to_string() if "share_truncated" in t else t.to_string())
    missing = t[t.status.str.startswith("not")].model.unique()
    print(f"tableC_truncation: {len(t)} rows -> results/tables/tableC_truncation.csv"
          + (f"; not computed for {', '.join(missing)}" if len(missing) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
