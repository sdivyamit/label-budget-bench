"""T2 prompt overlap (revision 2g; review point 8).

BeaverTails pairs one prompt with several responses, so a T2 test item's prompt can also appear in the
training pool with a different response. A prompt is "seen" by a run if its normalized text (lower case,
whitespace collapsed; as in analysis/compare.py) is the prompt of any item in that run's D_N, i.e. the first
N items of the seed's chain (data/splits.py).

  1. overlap table: share of test (and shifted) items whose prompt is seen, by N and seed;
  2. H1c (ModernBERT-large, 20,000 gold, against GPT-6.1 Sol, F1 at 0.5) on unseen-prompt test items, with
     a two-level bootstrap interval (seeds and items);
  3. the T2 gold curves of every model on each run's unseen-prompt test items, against the better frontier
     model on the same items (seed mean of the per-seed difference), and the break-even N on that subset;
     control: every budget on the fixed items whose prompt is in no training set (unseen at N = 20,000,
     where D_N is the whole pool for every seed), which separates prompt familiarity from item difficulty.

  python analysis/rev_overlap.py [--b 10000]
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd

import rev_common as rc
from eval import fastboot as fb  # noqa: E402

TASK = "t2_response_safety"


def norm(s) -> str:
    return " ".join(str(s).lower().split())


def prompt_keys() -> pd.Series:
    d = pd.read_parquet(rc.ROOT / "data" / "processed" / f"{TASK}.parquet", columns=["uid", "text_a"])
    return d.set_index("uid").text_a.map(norm)


def seen_masks() -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """(N, seed) -> eval set -> boolean mask over hl.items order: prompt appears in D_N."""
    keys = prompt_keys()
    chains = pd.read_parquet(rc.ROOT / "data" / "splits" / TASK / "chains.parquet")
    levels = rc.json.loads((rc.ROOT / "data" / "splits" / TASK / "subsets.json").read_text())["seeds_per_level"]
    items = {es: keys.loc[rc.hl.items(TASK, es)[0]].to_numpy() for es in ("test", "shift_api", "shift")}
    out = {}
    for n, seeds in levels.items():
        for s in seeds:
            c = chains[(chains.seed == s) & (chains["rank"] < int(n))]
            train = set(keys.loc[c.uid])
            out[(int(n), s)] = {es: np.array([k in train for k in v]) for es, v in items.items()}
    return out


def overlap_table(masks) -> pd.DataFrame:
    rows = [dict(n=n, seed=s, eval_set=es, items=len(m), seen=int(m.sum()), share=float(m.mean()))
            for (n, s), d in masks.items() for es, m in d.items()]
    return pd.DataFrame(rows)


def h1c_unseen(masks, b: int) -> dict:
    ref = rc.api_scores(TASK, "gpt61_sol", "test")
    _, y = rc.hl.items(TASK, "test")
    seeds = rc.cell_seeds(TASK, "encoder", "modernbert_large", "gold", 20000)
    out = {}
    for subset in ("unseen", "seen", "all"):
        per_seed, per_ref = [], []
        for s, d in sorted(seeds.items()):
            m = masks[(20000, s)]["test"]
            keep = ~m if subset == "unseen" else (m if subset == "seen" else np.ones_like(m))
            idx = np.flatnonzero(keep)
            w = np.vstack([np.ones((1, len(idx)), np.float32),
                           fb.resamples(len(idx), b, zlib.crc32(f"h1c/{subset}/{s}".encode()))])
            per_seed.append(fb.f1_at(w, y[idx], rc.run(d)["test"][idx]))
            per_ref.append(fb.f1_at(w, y[idx], ref[idx]))
        r, rr = np.vstack(per_seed), np.vstack(per_ref)
        diff = r - rr
        two = rc.two_level(diff, zlib.crc32(f"h1c/{subset}".encode()))
        out[subset] = dict(items=int(len(idx)), model=float(r[:, 0].mean()), ref=float(rr[:, 0].mean()),
                           diff=float(two[0]), lo=rc.ci(two)[1], hi=rc.ci(two)[2],
                           p=float(min(1, 2 * min((two[1:] <= 0).mean(), (two[1:] >= 0).mean()))))
    return out


def curves_unseen(masks) -> pd.DataFrame:
    _, y = rc.hl.items(TASK, "test")
    ref_name = rc.ref_model(TASK, "test")
    ref = rc.api_scores(TASK, ref_name, "test")
    i = rc.index()
    g = i[(i.task == TASK) & (i.tag == "") & (i.labels == "gold")]
    never = ~masks[(20000, 0)]["test"]
    rows = []
    for r in g.itertuples():
        m = masks[(int(r.n), int(r.seed))]["test"]
        x = rc.run(r.dir)
        for subset, keep in (("all", np.ones_like(m)), ("unseen", ~m), ("unseen_at_nmax", never)):
            yy, ss, rf = y[keep].astype(int), x["test"][keep], ref[keep]
            rows.append(dict(model=r.model, family=r.family, n=int(r.n), seed=int(r.seed), subset=subset,
                             items=int(keep.sum()),
                             f1_valthr=rc.M.f1_at(yy, ss, x["t_val"]), f1_at_05=rc.M.f1_at(yy, ss, 0.5),
                             max_f1=rc.M.max_f1(yy, ss)[0], pr_auc=rc.M.pr_auc(yy, ss),
                             ref_f1=rc.M.f1_at(yy, rf, 0.5), ref_max_f1=rc.M.max_f1(yy, rf)[0]))
    df = pd.DataFrame(rows)
    for mtr in ("f1_valthr", "f1_at_05"):
        df[f"{mtr}_minus_ref"] = df[mtr] - df.ref_f1
    df["max_f1_minus_ref"] = df.max_f1 - df.ref_max_f1
    return df


def first_sustained(ns, ok):
    for k in range(len(ns)):
        if ok[k:].all():
            return int(ns[k])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=10_000)
    a = ap.parse_args()
    rc.OUT.mkdir(parents=True, exist_ok=True)
    masks = seen_masks()
    ov = overlap_table(masks)
    ov.to_csv(rc.OUT / "t2_prompt_overlap.csv", index=False, float_format="%.4f")
    ovs = ov.groupby(["eval_set", "n"]).share.agg(["mean", "min", "max"]).reset_index()
    print(ovs.round(4).to_string(index=False))

    h1c = h1c_unseen(masks, a.b)
    print("H1c by prompt subset:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in h1c.items()})

    cu = curves_unseen(masks)
    cu.to_csv(rc.OUT / "t2_curves_unseen_prompts.csv", index=False, float_format="%.4f")
    agg = cu.groupby(["model", "n", "subset"])[["items", "f1_valthr", "f1_at_05", "max_f1", "pr_auc", "ref_f1",
                                                 "f1_valthr_minus_ref", "f1_at_05_minus_ref", "max_f1_minus_ref"]].mean().reset_index()
    agg.to_csv(rc.OUT / "t2_curves_unseen_prompts_summary.csv", index=False, float_format="%.4f")
    be = []
    for (model, subset), g in agg.groupby(["model", "subset"]):
        g = g.sort_values("n")
        for mtr in ("f1_valthr", "f1_at_05", "max_f1"):
            be.append(dict(model=model, subset=subset, metric=mtr,
                           n_star_mean=first_sustained(g.n.to_numpy(), (g[f"{mtr}_minus_ref"] >= 0).to_numpy())))
    be = pd.DataFrame(be)
    be.to_csv(rc.OUT / "t2_breakeven_unseen_prompts.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(be.pivot_table(index="model", columns=["metric", "subset"], values="n_star_mean", aggfunc="first").to_string())
    print(agg[agg.n.isin([100, 1000, 5000, 20000])].pivot_table(index=["model", "n"], columns="subset",
                                                            values=["items", "f1_valthr_minus_ref"]).round(3).to_string())

    numbers = {f"overlap/{r.eval_set}/n{r.n}": dict(mean=rc.r4(r.mean), min=rc.r4(r.min), max=rc.r4(r.max))
               for r in ovs.itertuples()}
    numbers |= {f"h1c/{k}": {kk: (rc.r4(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in h1c.items()}
    numbers |= {f"breakeven/{r.model}/{r.metric}/{r.subset}": ("not reached" if pd.isna(r.n_star_mean) else int(r.n_star_mean))
                for r in be.itertuples()}
    numbers |= {f"curve/{r.model}/n{r.n}/{r.subset}": dict(items=rc.r4(r.items), f1_valthr=rc.r4(r.f1_valthr),
                                                         ref=rc.r4(r.ref_f1), diff=rc.r4(r.f1_valthr_minus_ref))
                for r in agg.itertuples()}
    rc.save_numbers("t2_overlap", numbers)
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
