"""Results of the revision runs (D36): zero-shot decoders, the no-instruction ablation, the encoder
learning-rate check on T1, and the two pilot cells.

  python analysis/rev_phase1.py [--b 10000]

Writes results/revision/{zeroshot,noinstr,lr_check,pilot_cells}.csv and sections of revision_numbers.json.
Intervals: zero-shot and hosted references are single deterministic runs (item bootstrap); fine-tuned
cells use the two-level bootstrap (seeds and items); ablation differences pair seeds (same D_n) and items.
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd

import rev_common as rc

DECODERS = ["qwen35_2b", "gemma4_e2b", "qwen35_4b", "gemma4_e4b"]
ENCODERS = ["modernbert_base", "modernbert_large", "deberta_v3_large"]
SETS = ("test", "shift_api", "shift")


def s(*parts) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode())


def ref_model(task: str, eval_set: str = "test") -> str:
    """The frontier-tier hosted model with the higher F1 on this eval set (q_ref of Table 4)."""
    return rc.ref_model(task, eval_set)


def zeroshot(b: int) -> pd.DataFrame:
    rows, numbers = [], {}
    for task in rc.TASKS:
        for es in SETS:
            ref = ref_model(task, es)
            ch = rc.weight_chunks(task, es, b)
            rs = rc.api_scores(task, ref, es)
            ref_r = {m: rc.item_reps([(rs, 0.5)], task, es, b, m, ch)[0] for m in rc.F1S} if rs is not None else {}
            for model in DECODERS:
                cells = {f"zeroshot-{p}": rc.cell_runs(task, "decoder", model, "none", 0, t)
                         for p, t in (("concise", "zeroshot"), ("detailed", "zeroshot-detailed"))}
                cells |= {f"ft-n{n}": rc.cell_runs(task, "decoder", model, "gold", n) for n in (100, 1000)}
                for cond, dirs in cells.items():
                    ps = rc.per_seed(dirs, es)
                    if not ps:
                        continue
                    for m in rc.F1S + ("pr_auc", "ece"):
                        r = rc.item_reps(ps, task, es, b, m, ch)
                        v = rc.two_level(r, s(task, es, model, cond, m))
                        mean, lo, hi = rc.ci(v)
                        row = dict(task=task, eval_set=es, model=model, cond=cond, k=len(ps), metric=m,
                                   mean=mean, lo=lo, hi=hi, ref=ref, floor=rc.floor(task, es))
                        if m in ref_r:
                            d = v - ref_r[m]
                            row.update(ref_value=float(ref_r[m][0]), diff=float(d[0]),
                                       diff_lo=rc.ci(d)[1], diff_hi=rc.ci(d)[2], p_ge_ref=float((d[1:] >= 0).mean()))
                        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(rc.OUT / "zeroshot.csv", index=False, float_format="%.4f")
    for r in df.itertuples():
        key = f"{rc.SHORT[r.task]}/{r.eval_set}/{r.model}/{r.cond}/{r.metric}"
        numbers[key] = dict(mean=rc.r4(r.mean), lo=rc.r4(r.lo), hi=rc.r4(r.hi), k=int(r.k),
                            **({"ref": rc.r4(r.ref_value), "diff": rc.r4(r.diff), "diff_lo": rc.r4(r.diff_lo),
                                "diff_hi": rc.r4(r.diff_hi), "p_ge_ref": rc.r4(r.p_ge_ref)}
                               if not np.isnan(getattr(r, "ref_value", np.nan)) else {}))
    rc.save_numbers("zeroshot", numbers)
    return df


def paired_ablation(task, family, model, n, tag, es, metric, b, ch, hp_main_tag=""):
    """Ablation minus main configuration on the seeds both have: (ablation, main, diff) two-level reps."""
    a = rc.cell_seeds(task, family, model, "gold", n, tag)
    m = rc.cell_seeds(task, family, model, "gold", n, hp_main_tag)
    seeds = sorted(set(a) & set(m))
    if not seeds:
        return None
    ra = rc.item_reps(rc.per_seed([a[k] for k in seeds], es), task, es, b, metric, ch)
    rm = rc.item_reps(rc.per_seed([m[k] for k in seeds], es), task, es, b, metric, ch)
    idx = rc.seed_draws(len(seeds), ra.shape[1] - 1, s(task, es, model, n, tag, metric))
    col = np.arange(ra.shape[1])[:, None]
    return ra[idx, col].mean(1), rm[idx, col].mean(1), len(seeds)


def noinstr(b: int) -> pd.DataFrame:
    rows, numbers = [], {}
    for task in rc.TASKS:
        for es in ("test", "shift"):
            ch = rc.weight_chunks(task, es, b)
            for n in (100, 1000, 5000):
                for m in ("pr_auc", "f1_valthr", "max_f1", "ece"):
                    res = paired_ablation(task, "decoder", "qwen35_2b", n, "abl-noinstr", es, m, b, ch)
                    if res is None:
                        continue
                    va, vm, k = res
                    d = va - vm
                    rows.append(dict(task=task, eval_set=es, n=n, metric=m, k=k, noinstr=float(va[0]), main=float(vm[0]),
                                     diff=float(d[0]), lo=rc.ci(d)[1], hi=rc.ci(d)[2]))
    df = pd.DataFrame(rows)
    df.to_csv(rc.OUT / "noinstr.csv", index=False, float_format="%.4f")
    for r in df.itertuples():
        numbers[f"{rc.SHORT[r.task]}/{r.eval_set}/n{r.n}/{r.metric}"] = dict(
            noinstr=rc.r4(r.noinstr), main=rc.r4(r.main), diff=rc.r4(r.diff), lo=rc.r4(r.lo), hi=rc.r4(r.hi), k=int(r.k))
    rc.save_numbers("noinstr", numbers)
    return df


def lr_check(b: int) -> pd.DataFrame:
    task, n = "t1_rag_hallucination", 8651
    rows, numbers = [], {}
    for model in ENCODERS:
        for tag in ("abl-lr-div3", "", "abl-lr-x3"):
            dirs = rc.cell_runs(task, "encoder", model, "gold", n, tag)
            if not dirs:
                continue
            lr = rc.json.loads((rc.Path(dirs[0]) / "run.json").read_text())["hp"]["lr"]
            sd = [float(np.std(rc.run(d)["test"])) for d in dirs]
            for es in ("test", "shift_api"):
                ch = rc.weight_chunks(task, es, b)
                rs = rc.api_scores(task, ref_model(task, es), es)
                for m in ("pr_auc", "f1_valthr", "max_f1"):
                    r = rc.item_reps(rc.per_seed(dirs, es), task, es, b, m, ch)
                    v = rc.two_level(r, s(task, es, model, tag, m))
                    mean, lo, hi = rc.ci(v)
                    rref = rc.item_reps([(rs, 0.5)], task, es, b, m, ch)[0] if m != "pr_auc" else None
                    rows.append(dict(model=model, setting=tag or "selected", lr=lr, eval_set=es, metric=m, k=len(dirs),
                                     mean=mean, lo=lo, hi=hi, per_seed=";".join(f"{x:.3f}" for x in r[:, 0]),
                                     collapsed=sum(x < 0.01 for x in sd),
                                     ref=None if rref is None else float(rref[0]),
                                     p_ge_ref=None if rref is None else float(((v - rref)[1:] >= 0).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(rc.OUT / "lr_check.csv", index=False, float_format="%.4f")
    for r in df.itertuples():
        numbers[f"{r.model}/{r.setting}/{r.eval_set}/{r.metric}"] = dict(
            lr=r.lr, mean=rc.r4(r.mean), lo=rc.r4(r.lo), hi=rc.r4(r.hi), k=int(r.k), collapsed=int(r.collapsed),
            per_seed=r.per_seed, ref=rc.r4(r.ref), p_ge_ref=rc.r4(r.p_ge_ref))
    rc.save_numbers("lr_check", numbers)
    return df


def pilot_cells(b: int) -> pd.DataFrame:
    task, n = "t2_response_safety", 1000
    rows, numbers = [], {}
    for fam, model in (("encoder", "modernbert_base"), ("decoder", "qwen35_2b")):
        dirs = rc.cell_runs(task, fam, model, "gold", n)
        old = [d for d in dirs if not d.endswith("__s0")]
        for es in ("test", "shift_api"):
            ch = rc.weight_chunks(task, es, b)
            for m in ("pr_auc", "max_f1", "f1_valthr", "f1_at_05", "ece"):
                r_new = rc.item_reps(rc.per_seed(dirs, es), task, es, b, m, ch)
                r_old = rc.item_reps(rc.per_seed(old, es), task, es, b, m, ch)
                seed0 = [i for i, d in enumerate(dirs) if d.endswith("__s0")][0]
                rows.append(dict(model=model, eval_set=es, metric=m, k_new=len(dirs), mean_5=float(r_new[:, 0].mean()),
                                 mean_4=float(r_old[:, 0].mean()), seed0=float(r_new[seed0, 0])))
    df = pd.DataFrame(rows)
    df.to_csv(rc.OUT / "pilot_cells.csv", index=False, float_format="%.4f")
    for r in df.itertuples():
        numbers[f"{r.model}/{r.eval_set}/{r.metric}"] = dict(mean_5=rc.r4(r.mean_5), mean_4=rc.r4(r.mean_4), seed0=rc.r4(r.seed0))
    rc.save_numbers("pilot_cells", numbers)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=10_000)
    ap.add_argument("--only", default="zeroshot,noinstr,lr_check,pilot_cells")
    a = ap.parse_args()
    rc.OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    for part in a.only.split(","):
        df = globals()[part](a.b)
        print(f"== {part}: {len(df)} rows -> results/revision/{part}.csv")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
