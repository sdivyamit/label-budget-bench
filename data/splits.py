"""Nested training chains, API evaluation subsamples, and alignment-audit samples.

Chain design (DECISION D1): per task, a fixed stratified sub-pool S of `frontier_subpool`
items is drawn once from the training pool P. For every seed, the chain is a stratified
random ordering of S followed by a stratified random ordering of P \\ S, so
  D_100 ⊂ D_250 ⊂ ... ⊂ D_Nmax  (nested within a seed), and
  D_N ⊆ S for every seed whenever N <= |S|.
The frontier teacher therefore labels only S, while gold, open-teacher, and mixed
conditions reuse the same chains (only the labels change).

Outputs, per task, in data/splits/{task}/:
  chains.parquet   uid, seed, rank (0-based position), in_subpool
  subsets.json     n_levels, seeds per level, API subsample uids, audit uids
and docs/audit/{task}_audit_{annotator}.csv, one blind copy per annotator (blank `label`).

Usage: python data/splits.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
AUDIT = ROOT / "docs" / "audit"
AUDIT_COLUMNS = {
    "t1_rag_hallucination": {"text_a": "document", "text_b": "claim"},
    "t2_response_safety": {"text_a": "user_prompt", "text_b": "assistant_response"},
    "t3_implicit_toxicity": {"text_a": "post", "text_b": "unused"},
}


def stratified_order(df: pd.DataFrame, strata: list[str], rng: np.random.Generator) -> np.ndarray:
    """Random permutation of df's rows whose every prefix is close to proportionally stratified.

    Each item gets a fractional rank (j + u) / n_k within its stratum k (j = random position,
    u ~ U[0,1)); sorting by that rank interleaves strata at their population rates.
    """
    keys = df[strata].astype(str).agg("|".join, axis=1).to_numpy()
    frac = np.empty(len(df))
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        perm = rng.permutation(len(idx))
        frac[idx[perm]] = (np.arange(len(idx)) + rng.random(len(idx))) / len(idx)
    return np.argsort(frac, kind="stable")


def n_levels_for(pool_size: int) -> list[int]:
    levels = [n for n in CFG["n_levels"] if n <= pool_size]
    if pool_size < max(CFG["n_levels"]) and pool_size not in levels:
        levels.append(pool_size)
    return levels


def seeds_for(n: int) -> list[int]:
    k = CFG["seeds_small_n"] if n <= CFG["small_n_max"] else CFG["seeds_large_n"]
    return CFG["chain_seeds"][:k]


def build_chains(pool: pd.DataFrame, strata: list[str]) -> pd.DataFrame:
    sub_rng = np.random.default_rng([CFG["split_seed"], 999])
    sub_n = min(CFG["frontier_subpool"], len(pool))
    in_sub = np.zeros(len(pool), dtype=bool)
    in_sub[stratified_order(pool, strata, sub_rng)[:sub_n]] = True
    S, R = pool[in_sub], pool[~in_sub]
    rows = []
    for seed in CFG["chain_seeds"]:
        rng = np.random.default_rng([CFG["split_seed"], seed])
        order = np.concatenate([S.uid.to_numpy()[stratified_order(S, strata, rng)],
                                R.uid.to_numpy()[stratified_order(R, strata, rng)]])
        rows.append(pd.DataFrame({"uid": order, "seed": seed, "rank": np.arange(len(order)),
                                  "in_subpool": np.arange(len(order)) < sub_n}))
    return pd.concat(rows, ignore_index=True)


def chain(task: str, seed: int, n: int) -> pd.DataFrame:
    """Training items D_n for (task, seed), in chain order."""
    data = pd.read_parquet(PROCESSED / f"{task}.parquet").set_index("uid")
    c = pd.read_parquet(SPLITS / task / "chains.parquet")
    uids = c[(c.seed == seed) & (c["rank"] < n)].sort_values("rank").uid
    return data.loc[uids].reset_index()


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    for task, tcfg in CFG["tasks"].items():
        path = PROCESSED / f"{task}.parquet"
        if not path.exists():
            print(f"[skip] {task}: run data/prepare.py first")
            continue
        df = pd.read_parquet(path)
        strata = tcfg["strata"]
        pool = df[df.split == "pool"].reset_index(drop=True)
        out = SPLITS / task
        out.mkdir(parents=True, exist_ok=True)
        build_chains(pool, strata).to_parquet(out / "chains.parquet", index=False)

        rng = np.random.default_rng([CFG["split_seed"], 2024])
        test = df[df.split == "test"].reset_index(drop=True)
        test_sub = test.uid.to_numpy()[stratified_order(test, strata, rng)][: CFG["api_subsample_detailed"]]
        subsets = {
            "n_levels": n_levels_for(len(pool)),
            "seeds_per_level": {n: seeds_for(n) for n in n_levels_for(len(pool))},
            "pool_size": len(pool),
            "subpool_size": min(CFG["frontier_subpool"], len(pool)),
            "test_api_detailed": test_sub.tolist(),
        }
        shift = df[df.split == "shift"].reset_index(drop=True)
        if len(shift):
            s_strata = ["label", "group"] if shift.group.nunique() > 1 else ["label"]
            s_order = shift.uid.to_numpy()[stratified_order(shift, s_strata, rng)]
            subsets["shift_api_concise"] = s_order[: CFG["api_subsample_shift"]].tolist()
            subsets["shift_api_detailed"] = s_order[: CFG["api_subsample_detailed"]].tolist()
            if task in CFG["audit_size"]:
                audit_uids = s_order[: CFG["audit_size"][task]]
                subsets["alignment_audit"] = audit_uids.tolist()
                a = shift.set_index("uid").loc[audit_uids, ["text_a", "text_b"]].reset_index()
                a = a.sample(frac=1, random_state=CFG["split_seed"])  # hide stratification order from annotators
                a.insert(0, "item", range(1, len(a) + 1))
                a["label"], a["notes"] = "", ""
                a = a.rename(columns=AUDIT_COLUMNS[task]).drop(columns=["unused"], errors="ignore")
                for who in CFG["annotators"]:
                    dest = AUDIT / f"{task}_audit_{who}.csv"
                    if dest.exists():  # annotators' work may sit in any column; never overwrite
                        print(f"  [keep] {dest.name} exists; not overwritten")
                        continue
                    a.to_csv(dest, index=False)
        else:
            print(f"  [pending] {task}: no shifted set yet; API shift subsamples and audit sheet not built")
        (out / "subsets.json").write_text(json.dumps(subsets, indent=1))
        print(f"{task}: pool {len(pool)}, levels {subsets['n_levels']}, sub-pool {subsets['subpool_size']}, "
              f"test API subsample {len(test_sub)}")


if __name__ == "__main__":
    main()
