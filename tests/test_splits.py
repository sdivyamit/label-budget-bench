import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
import splits  # noqa: E402

TASKS = list(splits.CFG["tasks"])


def toy_pool(n=12000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "uid": [f"u{i}" for i in range(n)],
        "label": (rng.random(n) < 0.3).astype(int),
        "group": rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1]),
    })


def test_stratified_order_is_permutation_and_prefixes_are_proportional():
    pool = toy_pool()
    order = splits.stratified_order(pool, ["label", "group"], np.random.default_rng(1))
    assert sorted(order) == list(range(len(pool)))
    full = pool.groupby(["label", "group"]).size() / len(pool)
    for n in (100, 250, 1000):
        prefix = pool.iloc[order[:n]].groupby(["label", "group"]).size() / n
        assert (prefix.reindex(full.index, fill_value=0) - full).abs().max() < 2.5 / n + 1e-9


def test_chains_nested_subpool_and_deterministic():
    pool = toy_pool()
    c1 = splits.build_chains(pool, ["label", "group"])
    c2 = splits.build_chains(pool, ["label", "group"])
    pd.testing.assert_frame_equal(c1, c2)
    sub = splits.CFG["frontier_subpool"]
    subpools = []
    for seed, c in c1.groupby("seed"):
        c = c.sort_values("rank")
        assert sorted(c.uid) == sorted(pool.uid)                      # a permutation of the pool
        assert c.in_subpool.sum() == sub and c.in_subpool.iloc[:sub].all()
        subpools.append(frozenset(c.uid.iloc[:sub]))
    assert len(set(subpools)) == 1                                    # same sub-pool for every seed
    orders = [tuple(c.sort_values("rank").uid.iloc[:1000]) for _, c in c1.groupby("seed")]
    assert len(set(orders)) == len(orders)                            # seeds give different chains


def test_levels_and_seed_counts():
    assert splits.n_levels_for(7840)[-1] == 7840
    assert splits.n_levels_for(20000)[-1] == 20000
    assert len(splits.seeds_for(2000)) == 5 and len(splits.seeds_for(5000)) == 3


processed = [t for t in TASKS if (splits.PROCESSED / f"{t}.parquet").exists()
             and (splits.SPLITS / t / "chains.parquet").exists()]


@pytest.mark.parametrize("task", processed)
def test_real_data_disjoint_and_chained(task):
    df = pd.read_parquet(splits.PROCESSED / f"{task}.parquet")
    by = {s: set(df[df.split == s].uid) for s in df.split.unique()}
    for a in by:
        for b in by:
            if a < b:
                assert not by[a] & by[b], f"{task}: {a}/{b} overlap"
    if task.startswith("t1"):
        src = {s: set(df[df.split == s].source_id) for s in ("pool", "val", "test")}
        assert not src["pool"] & src["val"] and not src["pool"] & src["test"] and not src["val"] & src["test"]
    text = df.text_a.fillna("") + "\x00" + df.text_b.fillna("")
    assert not set(text[df.split == "test"]) & set(text[df.split == "pool"]), f"{task}: test text in pool"
    c = pd.read_parquet(splits.SPLITS / task / "chains.parquet")
    assert set(c.uid) == by["pool"]
    d_small = set(splits.chain(task, 0, 1000).uid)
    d_large = set(splits.chain(task, 0, 2000).uid)
    assert d_small < d_large
