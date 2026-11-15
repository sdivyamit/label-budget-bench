"""Normalize raw sources into data/processed/{task}.parquet with a common schema.

Columns: uid, task, text_a, text_b, label (1 = positive class per configs/tasks.yaml),
group (stratification key), split in {pool, val, test, shift}, plus source-specific extras.
Leakage and duplicate handling is logged to data/processed/{task}.log.json.

Shift sets whose raw files are absent (gated or survey-gated) are skipped with a message;
rerun after data/download.py fetches them.

Usage: python data/prepare.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
CFG = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())

from splits import stratified_order  # noqa: E402  (same directory)


def require(df: pd.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: expected columns {missing} not found; got {list(df.columns)}")


def frame(uid, text_a, text_b, label, group, split, **extra) -> pd.DataFrame:
    d = {"uid": uid, "text_a": text_a, "text_b": text_b, "label": np.asarray(label, dtype="int8"),
         "group": group, "split": split}
    d.update(extra)
    return pd.DataFrame(d)


def carve_val(train: pd.DataFrame, strata: list[str], rng, group_key: str | None) -> pd.DataFrame:
    """Mark val_size items of `train` as split='val'; by whole groups if group_key is set."""
    n = CFG["val_size"]
    if group_key is None:
        order = stratified_order(train, strata, rng)
        val_idx = train.index[order[:n]]
    else:
        units = train.groupby(group_key).agg(group=("group", "first"), size=("uid", "size")).reset_index()
        order = stratified_order(units, ["group"], rng)
        chosen = units.iloc[order][group_key][units.iloc[order]["size"].cumsum().values <= n]
        val_idx = train.index[train[group_key].isin(set(chosen))]
    train = train.copy()
    train.loc[val_idx, "split"] = "val"
    return train


def cap_pool(df: pd.DataFrame, strata: list[str], rng) -> pd.DataFrame:
    pool = df[df.split == "pool"]
    if len(pool) <= CFG["pool_cap"]:
        return df
    keep = pool.index[stratified_order(pool, strata, rng)[: CFG["pool_cap"]]]
    return df[(df.split != "pool") | df.index.isin(keep)]


def cap_shift(shift: pd.DataFrame, cap, strata: list[str], rng) -> pd.DataFrame:
    if cap is None or len(shift) <= cap:
        return shift
    return shift.iloc[stratified_order(shift, strata, rng)[:cap]]


# ---------- T1: RAGTruth QA + Summary -> RAGTruth Data2txt (held-out task type, D21) ----------
def t1(cfg: dict, rng, log: dict) -> pd.DataFrame:
    r = pd.read_json(RAW / "ragtruth" / "response.jsonl", lines=True)
    s = pd.read_json(RAW / "ragtruth" / "source_info.jsonl", lines=True)
    require(r, ["id", "source_id", "labels", "split", "quality", "response", "model"], "RAGTruth response")
    require(s, ["source_id", "task_type", "prompt"], "RAGTruth source_info")
    assert s.groupby("source_id").task_type.nunique().max() == 1, "a source document spans task types"
    log["ragtruth_dropped_quality"] = int(r.quality.isin(cfg["exclude_quality"]).sum())
    r = r[~r.quality.isin(cfg["exclude_quality"])].merge(s[["source_id", "task_type", "prompt"]], on="source_id")
    held = r.task_type == cfg["shift_task_type"]
    df = frame(uid="rt-" + r.id.astype(str), text_a=r.prompt, text_b=r.response,
               label=(r.labels.str.len() > 0), group=r.task_type,
               split=np.where(held, "shift", np.where(r.split == "test", "test", "pool")),
               source_id=r.source_id.astype(str), gen_model=r.model, ragtruth_split=r.split)
    assert not set(df[df.split == "test"].source_id) & set(df[df.split == "pool"].source_id)
    assert not set(df[df.split == "shift"].source_id) & set(df[df.split != "shift"].source_id)
    log["ragtruth_heldout_task_type"] = cfg["shift_task_type"]
    log["ragtruth_heldout_responses"] = int(held.sum())
    train = carve_val(df[df.split == "pool"], cfg["strata"], rng, cfg["val_group_key"])
    shift = cap_shift(df[df.split == "shift"], cfg["shift_cap"], ["label"], rng)
    return pd.concat([train, df[df.split == "test"], shift], ignore_index=True)


# ---------- T2: BeaverTails-30k -> WildGuardTest ----------
def t2(cfg: dict, rng, log: dict) -> pd.DataFrame:
    tr = pd.read_json(RAW / "beavertails" / "train.jsonl.gz", lines=True)
    te = pd.read_json(RAW / "beavertails" / "test.jsonl.gz", lines=True)
    require(tr, ["prompt", "response", "is_safe"], "BeaverTails")
    key = lambda d: d.prompt.str.strip() + "\x00" + d.response.str.strip()
    tr, te = tr.assign(key=key(tr)), te.assign(key=key(te))
    conflicting = tr.groupby("key").is_safe.nunique().pipe(lambda x: set(x[x > 1].index))
    log["beavertails_conflicting_pairs_dropped"] = len(conflicting)
    tr = tr[~tr.key.isin(conflicting)]
    log["beavertails_duplicate_pairs_dropped"] = int(tr.key.duplicated().sum())
    tr = tr.drop_duplicates("key")
    log["beavertails_train_test_overlap_dropped"] = int(tr.key.isin(set(te.key)).sum())
    tr = tr[~tr.key.isin(set(te.key))]
    log["beavertails_test_prompts_seen_in_train"] = int(te.prompt.isin(set(tr.prompt)).sum())
    df = pd.concat([
        frame(uid=[f"bt-tr-{i}" for i in tr.index], text_a=tr.prompt.values, text_b=tr.response.values,
              label=~tr.is_safe.values, group="all", split="pool"),
        frame(uid=[f"bt-te-{i}" for i in te.index], text_a=te.prompt.values, text_b=te.response.values,
              label=~te.is_safe.values, group="all", split="test"),
    ], ignore_index=True)
    train = carve_val(df[df.split == "pool"], cfg["strata"], rng, None)
    df = cap_pool(pd.concat([train, df[df.split == "test"]], ignore_index=True), cfg["strata"], rng)

    files = glob.glob(str(RAW / "wildguardtest" / "**" / "*.parquet"), recursive=True)
    if not files:
        print("  [pending] WildGuardTest not downloaded (gated)")
        return df
    w = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    require(w, ["prompt", "response", "response_harm_label"], "WildGuardTest")
    w = w[w.response.notna() & w.response_harm_label.isin(["harmful", "unharmful"])]
    sh = frame(uid=[f"wg-{i}" for i in range(len(w))], text_a=w.prompt.values, text_b=w.response.values,
               label=(w.response_harm_label.values == "harmful"),
               group=np.where(w.get("adversarial", pd.Series(False, index=w.index)).values, "adversarial", "vanilla"),
               split="shift")
    return pd.concat([df, cap_shift(sh, cfg["shift_cap"], ["label"], rng)], ignore_index=True)


# ---------- T3: ToxiGen (human-annotated), held-out target groups (D22) ----------
# The annotated test file names groups differently ("asian folks") from the train file ("asian").
TOXIGEN_GROUPS = {
    "asian folks": "asian", "black folks / african-americans": "black", "black/african-american folks": "black",
    "chinese folks": "chinese", "folks with mental disabilities": "mental_dis",
    "folks with physical disabilities": "physical_dis", "jewish folks": "jewish", "latino/hispanic folks": "latino",
    "lgbtq+ folks": "lgbtq", "mexican folks": "mexican", "middle eastern folks": "middle_east",
    "muslim folks": "muslim", "native american folks": "native_american",
    "native american/indigenous folks": "native_american",
}


def t3(cfg: dict, rng, log: dict) -> pd.DataFrame:
    tr = pd.read_parquet(RAW / "toxigen" / "annotated_train.parquet")
    te = pd.read_parquet(RAW / "toxigen" / "annotated_test.parquet")
    require(tr, ["text", "target_group", "toxicity_human"], "ToxiGen annotated")
    th = cfg["toxicity_threshold"]
    log["toxigen_threshold"] = f"toxicity_human >= {th}"
    log["toxigen_test_texts_in_train_dropped"] = int(tr.text.isin(set(te.text)).sum())
    tr = tr[~tr.text.isin(set(te.text))].reset_index(drop=True)
    df = pd.concat([
        frame(uid=[f"tg-tr-{i}" for i in tr.index], text_a=tr.text.values, text_b=None,
              label=tr.toxicity_human.values >= th, group=tr.target_group.replace(TOXIGEN_GROUPS).values,
              split="pool", toxicity_human=tr.toxicity_human.values, toxigen_split="train"),
        frame(uid=[f"tg-te-{i}" for i in te.index], text_a=te.text.values, text_b=None,
              label=te.toxicity_human.values >= th, group=te.target_group.replace(TOXIGEN_GROUPS).values,
              split="test", toxicity_human=te.toxicity_human.values, toxigen_split="test"),
    ], ignore_index=True)
    assert df.group.nunique() == 13, sorted(df.group.unique())
    held = set(cfg["shift_heldout_groups"])
    assert held <= set(df.group), held - set(df.group)
    df.loc[df.group.isin(held), "split"] = "shift"
    log["toxigen_heldout_groups"] = sorted(held)
    log["toxigen_heldout_items"] = int((df.split == "shift").sum())
    train = carve_val(df[df.split == "pool"], cfg["strata"], rng, None)
    shift = cap_shift(df[df.split == "shift"], cfg["shift_cap"], ["label", "group"], rng)
    return pd.concat([train, df[df.split == "test"], shift], ignore_index=True)


BUILDERS = {"t1_rag_hallucination": t1, "t2_response_safety": t2, "t3_implicit_toxicity": t3}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for task, build in BUILDERS.items():
        print(task)
        rng = np.random.default_rng(CFG["split_seed"])
        log: dict = {}
        df = build(CFG["tasks"][task], rng, log).assign(task=task)
        assert df.uid.is_unique, f"{task}: duplicate uids"
        df.to_parquet(OUT / f"{task}.parquet", index=False)
        log["sizes"] = df.split.value_counts().to_dict()
        log["prevalence"] = df.groupby("split").label.mean().round(4).to_dict()
        (OUT / f"{task}.log.json").write_text(json.dumps(log, indent=2))
        print("  ", log["sizes"], "prevalence", log["prevalence"])


if __name__ == "__main__":
    main()
