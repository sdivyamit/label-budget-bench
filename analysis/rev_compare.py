"""Seed-aware intervals (revision 2b; review point 3).

1. The 12 primary comparisons of configs/comparisons.yaml with three intervals side by side:
   item bootstrap of the seed means (as published), two-level bootstrap (seeds and items), and two-level
   with item clusters (T1 source documents, T2 prompts) in place of items. Seeds are paired when both
   sides are the same model at the same N (teacher against gold); otherwise they are drawn independently.
   Holm correction over the 12, separately for each interval type.
2. Table S2: gold minus teacher-label PR-AUC at the same budget and seeds (seed-paired two-level CI).

  python analysis/rev_compare.py [--b 10000]
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd
import yaml

import rev_common as rc
import compare as cmp  # noqa: E402  (analysis/compare.py, for the cluster ids)
from eval.bootstrap import holm  # noqa: E402

TOP = {"t1_rag_hallucination": 8651, "t2_response_safety": 20000, "t3_implicit_toxicity": 5920}


def cluster_chunks(task: str, es: str, b: int, size: int = 1000):
    _, y = rc.hl.items(task, es)
    g = cmp.clusters(task, es)
    n_g = int(g.max()) + 1
    rng = np.random.default_rng(rc.hl.boot_seed(task, es) + 7)
    out = [np.ones((1, len(y)), np.float32)]
    for start in range(0, b, size):
        out.append(rng.multinomial(n_g, np.full(n_g, 1.0 / n_g), size=min(size, b - start)).astype(np.float32)[:, g])
    return out


def resolve(spec: str, task: str, es: str) -> tuple[str, dict[int, tuple] | None, np.ndarray | None]:
    """-> (name, {seed: (scores, t_val)}, hosted scores)."""
    p = spec.split("/")
    if p[0] == "api":
        return spec, None, rc.api_scores(task, p[1], es)
    family, model, labels, n = p
    i = rc.index()
    c = i[(i.task == task) & (i.family == family) & (i.model == model) & (i.labels == labels) & (i.tag == "")]
    n = int(c.n.max()) if n == "top" else int(n)
    seeds = rc.cell_seeds(task, family, model, labels, n)
    return f"{family}/{model}/{labels}/n{n}", {k: (rc.run(d)[es], rc.run(d)["t_val"]) for k, d in seeds.items()}, None


def side_reps(side, task, es, b, metric, ch) -> tuple[np.ndarray, list[int]]:
    name, seeds, hosted = side
    if hosted is not None:
        return rc.item_reps([(hosted, 0.5)], task, es, b, metric, ch), [0]
    ks = sorted(seeds)
    return rc.item_reps([seeds[k] for k in ks], task, es, b, metric, ch), ks


def diff_reps(ra, ka, rb, kb, seed) -> tuple[np.ndarray, np.ndarray]:
    """(item-only diff of seed means, two-level diff); paired seeds when both sides have the same seed ids."""
    item = ra.mean(0) - rb.mean(0)
    cols = np.arange(ra.shape[1])[:, None]
    if ka == kb and len(ka) > 1:
        idx = rc.seed_draws(len(ka), ra.shape[1] - 1, seed)
        return item, ra[idx, cols].mean(1) - rb[idx, cols].mean(1)
    ia = rc.seed_draws(len(ka), ra.shape[1] - 1, seed)
    ib = rc.seed_draws(len(kb), rb.shape[1] - 1, seed + 1)
    return item, ra[ia, cols].mean(1) - rb[ib, cols].mean(1)


def pval(d: np.ndarray) -> float:
    r = d[1:][~np.isnan(d[1:])]
    return float(min(1.0, 2 * min((r <= 0).mean(), (r >= 0).mean())))


def comparisons(b: int) -> pd.DataFrame:
    cfg = yaml.safe_load((rc.ROOT / "configs" / "comparisons.yaml").read_text())
    rows = []
    for c in cfg["comparisons"]:
        task, es, metric = c["task"], c["eval_set"], c.get("metric", cfg["metric_default"])
        sides = {k: resolve(c[k], task, es) for k in ("a", "b")}
        seed = zlib.crc32(c["id"].encode())
        row = dict(id=c["id"], task=task, eval_set=es, metric=metric, a=sides["a"][0], b=sides["b"][0])
        for kind, ch in (("item", rc.weight_chunks(task, es, b)), ("cluster", cluster_chunks(task, es, b))):
            (ra, ka), (rb, kb) = (side_reps(sides[k], task, es, b, metric, ch) for k in ("a", "b"))
            item, two = diff_reps(ra, ka, rb, kb, seed)
            if kind == "item":
                row.update(k_a=len(ka), k_b=len(kb), value_a=float(ra[:, 0].mean()), value_b=float(rb[:, 0].mean()),
                           diff=float(item[0]), item_lo=rc.ci(item)[1], item_hi=rc.ci(item)[2], item_p=pval(item),
                           two_lo=rc.ci(two)[1], two_hi=rc.ci(two)[2], two_p=pval(two))
            else:
                row.update(cl_lo=rc.ci(two)[1], cl_hi=rc.ci(two)[2], cl_p=pval(two))
        rows.append(row)
    df = pd.DataFrame(rows)
    for k in ("item", "two", "cl"):
        df[f"{k}_p_holm"] = holm(df[f"{k}_p"].to_numpy())
    return df


def table_s2(b: int) -> pd.DataFrame:
    rows = []
    i = rc.index()
    teachers = [l for l in i[i.tag == ""].labels.unique() if "teacher" in l]
    for task in rc.TASKS:
        for es in ("test", "shift"):
            ch = rc.weight_chunks(task, es, b)
            for (family, model), _ in i[(i.task == task) & (i.tag == "") & (i.labels == "gold")].groupby(["family", "model"]):
                for n in sorted({5000, TOP[task]}):
                    gold = rc.cell_seeds(task, family, model, "gold", n)
                    for lab in teachers:
                        t = rc.cell_seeds(task, family, model, lab, n)
                        ks = sorted(set(gold) & set(t))
                        if not ks:
                            continue
                        rg = rc.item_reps(rc.per_seed([gold[k] for k in ks], es), task, es, b, "pr_auc", ch)
                        rt = rc.item_reps(rc.per_seed([t[k] for k in ks], es), task, es, b, "pr_auc", ch)
                        item, two = diff_reps(rg, ks, rt, ks, zlib.crc32(f"{task}/{es}/{model}/{n}/{lab}".encode()))
                        rows.append(dict(task=task, eval_set=es, family=family, model=model, n=n, labels=lab, k=len(ks),
                                         gold=float(rg[:, 0].mean()), teacher=float(rt[:, 0].mean()), gap=float(item[0]),
                                         item_lo=rc.ci(item)[1], item_hi=rc.ci(item)[2],
                                         two_lo=rc.ci(two)[1], two_hi=rc.ci(two)[2], two_p=pval(two)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=10_000)
    a = ap.parse_args()
    rc.OUT.mkdir(parents=True, exist_ok=True)
    df = comparisons(a.b)
    df.to_csv(rc.OUT / "comparisons_two_level.csv", index=False, float_format="%.4f")
    s2 = table_s2(a.b)
    s2.to_csv(rc.OUT / "table_s2_two_level.csv", index=False, float_format="%.4f")
    numbers = {"meta": dict(bootstrap_resamples=a.b)}
    for r in df.to_dict("records"):
        numbers[f"cmp/{r['id']}"] = {k: (rc.r4(v) if isinstance(v, float) else v) for k, v in r.items() if k != "id"}
    for r in s2.itertuples():
        numbers[f"s2/{rc.SHORT[r.task]}/{r.eval_set}/{r.model}/n{r.n}/{r.labels}"] = dict(
            gap=rc.r4(r.gap), item_lo=rc.r4(r.item_lo), item_hi=rc.r4(r.item_hi), two_lo=rc.r4(r.two_lo),
            two_hi=rc.r4(r.two_hi), k=int(r.k))
    rc.save_numbers("seed_aware", numbers)
    pd.set_option("display.width", 250)
    print(df[["id", "eval_set", "metric", "a", "b", "k_a", "k_b", "diff", "item_lo", "item_hi", "two_lo", "two_hi",
              "cl_lo", "cl_hi", "item_p_holm", "two_p_holm", "cl_p_holm"]].round(4).to_string(index=False))
    w = (s2.two_hi - s2.two_lo) / (s2.item_hi - s2.item_lo)
    print(f"Table S2: {len(s2)} gaps; two-level / item CI width: median {w.median():.2f}, max {w.max():.2f}; "
          f"sign-excludes-0 item {((s2.item_lo > 0) | (s2.item_hi < 0)).sum()}, two-level {((s2.two_lo > 0) | (s2.two_hi < 0)).sum()}")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
