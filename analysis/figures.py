"""Figures and tables regenerated from results/metrics.parquet (plan §7, V-A ... V-C).

  python analysis/collect.py && python analysis/headline.py && python analysis/figures.py
                                                                  # -> results/figures/, results/tables/

Fig. 2 / 3: max-F1 vs N (gold labels), test / shifted set; one line per model (seed mean, +-1 SD band),
frontier-API band shaded (concise prompt), Jev as the zero-label score reference line, N*_mean markers.
Table 4: N*_mean / N*_var (D17) per task x model x eval set against the lower edge of the frontier band.
Fig. 4b: alpha_K of the 0.5 verdicts across seeds vs N (results/tables/cells_ci.csv, test set).
Fig. 6: reliability diagrams from the per-run predictions (preds/runs/).
A figure built while some API sets are still missing says so in its title.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import breakeven, calibrate, metrics  # noqa: E402

FIG, TAB, RUNS = ROOT / "results" / "figures", ROOT / "results" / "tables", ROOT / "preds" / "runs"
MODELS = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
TIER = {m: s["tier"] for m, s in MODELS["api"].items()}
TASK_TITLE = {"t1_rag_hallucination": "T1 RAG hallucination", "t2_response_safety": "T2 response safety",
              "t3_implicit_toxicity": "T3 implicit toxicity"}
API_SET = {"test": "test", "shift_api": "shift_api_concise"}
STYLE = {"tfidf_lr": ("TF-IDF + LR", "0.45", "-"),
         "modernbert_base": ("ModernBERT-base", "#9ecae1", "-"), "modernbert_large": ("ModernBERT-large", "#3182bd", "-"),
         "deberta_v3_large": ("DeBERTa-v3-large", "#08519c", "--"),
         "gemma4_e2b": ("Gemma 4 E2B", "#fdae6b", "-"), "qwen35_2b": ("Qwen3.5-2B", "#e6550d", "-"),
         "gemma4_e4b": ("Gemma 4 E4B", "#fd8d3c", "--"), "qwen35_4b": ("Qwen3.5-4B", "#a63603", "--")}


def load() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "results" / "metrics.parquet")


def curves(df, metric: str, eval_set: str, labels: str = "gold", calib: str = "raw") -> pd.DataFrame:
    """task, model, n, seed, q for main-sweep runs."""
    d = df[(df.tag == "") & (df.family != "api") & (df.labels == labels) & (df.metric == metric)
           & (df.eval_set == eval_set) & (df.calib == calib)]
    return d[["task", "family", "model", "n", "seed", "value"]].rename(columns={"value": "q"})


def api_refs(df, metric: str, eval_set: str) -> pd.DataFrame:
    """task, model, tier, q (concise prompt) for every complete API result set."""
    d = df[(df.family == "api") & (df.tag == "concise") & (df.metric == metric) & (df.eval_set == API_SET[eval_set])]
    d = d[["task", "model", "value"]].rename(columns={"value": "q"})
    d["tier"] = d.model.map(TIER)
    return d


def band(refs: pd.DataFrame, task: str):
    f = refs[(refs.task == task) & (refs.tier == "frontier")]
    expected = sum(t == "frontier" for t in TIER.values())
    return (f.q.min(), f.q.max(), len(f) < expected) if len(f) else (None, None, True)


def learning_curves(df, eval_set: str, out: str, metric: str = "max_f1"):
    c, refs = curves(df, metric, eval_set), api_refs(df, metric, eval_set)
    tasks = [t for t in TASK_TITLE if t in set(c.task)]
    if not tasks:
        print(f"{out}: no runs yet")
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.2), squeeze=False)
    provisional = False
    for ax, task in zip(axes[0], tasks):
        lo, hi, missing = band(refs, task)
        provisional |= missing
        if lo is not None:
            ax.axhspan(lo, hi if hi > lo else lo + 0.002, color="0.85", zorder=0, label="frontier API band")
        jev = refs[(refs.task == task) & (refs.tier == "hosted_score")]
        if len(jev):
            ax.axhline(jev.q.iloc[0], color="purple", ls=":", lw=1.4, label="Jev (zero-label)")
        for model, g in c[c.task == task].groupby("model"):
            name, color, ls = STYLE.get(model, (model, None, "-"))
            s = breakeven._summary(g[["n", "seed", "q"]])
            ax.plot(s.index, s["mean"], ls, color=color, marker="o", ms=3, lw=1.6, label=name)
            ax.fill_between(s.index, s["mean"] - s["sd"], s["mean"] + s["sd"], color=color, alpha=0.15, lw=0)
            if hi is not None:
                n_star = breakeven.n_star_mean(g[["n", "seed", "q"]], hi)
                if n_star:
                    ax.plot([n_star], [s.loc[n_star, "mean"]], marker="*", ms=12, color=color, mec="k", mew=0.6)
        ax.set_xscale("log")
        ax.set_title(TASK_TITLE[task])
        ax.set_xlabel("gold labels N")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(f"{metric.replace('_', '-')} ({'in-distribution test' if eval_set == 'test' else 'shifted test'})")
    h, lab = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(6, len(lab)), fontsize=8, frameon=False)
    if provisional:
        fig.suptitle("PROVISIONAL (API sets missing)")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95 if provisional else 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{out}.pdf")
    fig.savefig(FIG / f"{out}.png", dpi=150)
    plt.close(fig)
    print(f"{out}: {len(tasks)} tasks{' (provisional)' if provisional else ''} -> results/figures/{out}.pdf")


def table_breakeven(df, metric: str = "max_f1") -> pd.DataFrame:
    rows = []
    for eval_set in ("test", "shift_api"):
        c, refs = curves(df, metric, eval_set), api_refs(df, metric, eval_set)
        for (task, family, model), g in c.groupby(["task", "family", "model"]):
            lo, hi, missing = band(refs, task)
            if lo is None:
                continue
            g = g[["n", "seed", "q"]]
            rows.append(dict(task=task, eval_set=eval_set, family=family, model=model, q_ref=round(hi, 4),
                             n_star_mean=breakeven.n_star_mean(g, hi), n_star_var=breakeven.n_star_var(g, hi),
                             q_ref_weaker=round(lo, 4), n_star_mean_weaker=breakeven.n_star_mean(g, lo),
                             n_star_var_weaker=breakeven.n_star_var(g, lo),
                             n_max=int(g.n.max()), best_mean=round(g.groupby("n").q.mean().max(), 4),
                             provisional=missing))
    t = pd.DataFrame(rows)
    if len(t):
        TAB.mkdir(parents=True, exist_ok=True)
        t.to_csv(TAB / "table4_breakeven.csv", index=False)
        print(t.to_string(index=False))
    return t


LABEL_STYLE = {"gold": "-", "open_teacher": "--", "frontier_teacher": ":"}
REPRESENTATIVE = ["tfidf_lr", "modernbert_large", "qwen35_2b"]


def _mix_marker(src: str) -> dict:
    """Diamond = open-teacher mix, square = frontier-teacher mix; hollow = 250 gold anchor, filled = 1,000."""
    return dict(marker="D" if "open" in src else "s", s=22, zorder=3, label=src.replace("_", " "),
                **(dict(facecolors="none", edgecolors="k") if src.startswith("mix250") else dict(color="k")))


def _save(fig, out: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{out}.pdf")
    fig.savefig(FIG / f"{out}.png", dpi=150)
    plt.close(fig)
    print(f"{out} -> results/figures/{out}.pdf")


def label_sources(df, out: str = "fig4_label_sources", metric: str = "max_f1"):
    """RQ2: gold vs open-teacher vs frontier-teacher curves for representative models; mixes as markers."""
    d = df[(df.tag == "") & (df.family != "api") & (df.metric == metric) & (df.eval_set == "test") & (df.calib == "raw")]
    tasks = [t for t in TASK_TITLE if t in set(d.task)]
    models = [m for m in REPRESENTATIVE if m in set(d.model)]
    if not tasks or not models:
        print(f"{out}: no runs yet")
        return
    fig, axes = plt.subplots(len(tasks), len(models), figsize=(4.3 * len(models), 3.3 * len(tasks)), squeeze=False)
    for r, task in enumerate(tasks):
        for c, model in enumerate(models):
            ax = axes[r][c]
            name, color, _ = STYLE[model]
            g = d[(d.task == task) & (d.model == model)]
            for src, ls in LABEL_STYLE.items():
                h = g[g.labels == src]
                if len(h):
                    s = h.groupby("n").value.agg(["mean", "std"]).fillna(0)
                    ax.plot(s.index, s["mean"], ls, color=color, marker="o", ms=2.5, label=src.replace("_", " "))
            for src, h in g[g.labels.str.startswith("mix")].groupby("labels"):
                s = h.groupby("n").value.mean()
                ax.scatter(s.index, s.values, **_mix_marker(src))
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            ax.set_title(f"{TASK_TITLE[task]} - {name}", fontsize=9)
            if r == len(tasks) - 1:
                ax.set_xlabel("training labels N")
            if c == 0:
                ax.set_ylabel(f"{metric.replace('_', '-').upper() if metric == 'pr_auc' else metric} (test)")
    seen = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=len(seen), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out)


def seed_agreement(out: str = "fig4b_seed_agreement", metric: str = "alpha_k"):
    """RQ2 companion to Fig. 4: Krippendorff's alpha_K of the 0.5 verdicts across seeds vs N, same panels
    (test set; K = 5 seeds up to N = 2,000, 3 above). Data for every model in results/tables/<out>.csv."""
    f = TAB / "cells_ci.csv"
    if not f.exists():
        print(f"{out}: no cells_ci.csv yet (run analysis/headline.py)")
        return
    c = pd.read_csv(f, keep_default_na=False, na_values=[""])
    c["tag"] = c["tag"].fillna("")
    d = c[(c.tag == "") & c.family.isin(["classical", "encoder", "decoder"]) & (c.eval_set == "test")
          & c.metric.isin([metric, "pair_disagree"])]
    tasks = [t for t in TASK_TITLE if t in set(d.task)]
    models = [m for m in REPRESENTATIVE if m in set(d.model)]
    if not tasks or not models:
        print(f"{out}: no multi-seed cells yet")
        return
    TAB.mkdir(parents=True, exist_ok=True)
    t = d.pivot_table(index=["task", "family", "model", "labels", "n", "k"], columns="metric", values="mean")
    t.reset_index().to_csv(TAB / f"{out}.csv", index=False, float_format="%.4f")
    d = d[d.metric == metric]
    fig, axes = plt.subplots(len(tasks), len(models), figsize=(4.3 * len(models), 3.3 * len(tasks)), squeeze=False)
    for r, task in enumerate(tasks):
        for col, model in enumerate(models):
            ax = axes[r][col]
            name, color, _ = STYLE[model]
            g = d[(d.task == task) & (d.model == model)]
            for src, ls in LABEL_STYLE.items():
                s = g[g.labels == src].set_index("n")["mean"].sort_index()
                if len(s):
                    ax.plot(s.index, s.values, ls, color=color, marker="o", ms=2.5, label=src.replace("_", " "))
            for src, h in g[g.labels.str.startswith("mix")].groupby("labels"):
                ax.scatter(h.n, h["mean"], **_mix_marker(src))
            ax.axvline(np.sqrt(2000 * 5000), color="0.6", lw=0.8, ls=":")
            ax.text(np.sqrt(2000 * 5000), 0.02, " K=5 | K=3", transform=ax.get_xaxis_transform(), fontsize=6,
                    color="0.4", ha="center")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            ax.set_title(f"{TASK_TITLE[task]} - {name}", fontsize=9)
            if r == len(tasks) - 1:
                ax.set_xlabel("training items N")
            if col == 0:
                ax.set_ylabel(r"seed agreement $\alpha_K$ (test, verdicts at 0.5)")
            ax.legend(fontsize=6)
    fig.tight_layout()
    _save(fig, out)


REL_TASK, REL_MODELS, REL_BINS = "t2_response_safety", ("modernbert_large", "qwen35_2b"), 10


def _pooled(task: str, model: str, n: int, eval_set: str):
    """Gold-label main-sweep runs of one cell, seeds pooled: y, raw p, temperature-scaled p, k, mean T."""
    ys, raw, temp, ts = [], [], [], []
    pat = re.compile(rf"{task}__{model}__gold__n{n}__s\d+")
    for d in sorted(RUNS.glob(f"{task}__{model}__gold__n{n}__s*")):
        if not (pat.fullmatch(d.name) and (d / "DONE").exists() and (d / "preds.parquet").exists()):
            continue
        r = json.loads((d / "run.json").read_text())
        if r.get("eval_limit") or r.get("max_steps"):
            continue
        pp = pd.read_parquet(d / "preds.parquet")
        val, g = pp[pp.split == "val"], pp[pp.split == eval_set]
        cal = calibrate.fit("temperature", val.p.to_numpy(), val.label.to_numpy())
        ys.append(g.label.to_numpy()); raw.append(g.p.to_numpy()); temp.append(cal(g.p.to_numpy())); ts.append(cal.T)
    if not ys:
        return None
    return np.concatenate(ys), np.concatenate(raw), np.concatenate(temp), len(ys), float(np.mean(ts))


def _bins(y, p, n_bins: int = REL_BINS) -> pd.DataFrame:
    """Top-label reliability (D16): confidence max(p, 1-p), accuracy of the 0.5 verdict, equal-width bins."""
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    t = pd.DataFrame(dict(bin=b, conf=conf, correct=correct)).groupby("bin").agg(
        count=("conf", "size"), mean_conf=("conf", "mean"), accuracy=("correct", "mean")).reset_index()
    t["lo"], t["hi"], t["share"] = t.bin / n_bins, (t.bin + 1) / n_bins, t["count"] / len(p)
    return t


def reliability(out: str = "fig6_reliability", task: str = REL_TASK, models=REL_MODELS, eval_set: str = "test"):
    """RQ3: reliability diagrams, raw vs temperature-scaled (T fitted on each run's validation set), at N = 1,000
    and the top N, gold labels, seeds pooled, 10 equal-width bins; legend ECE is that of the pooled items."""
    tops = {}
    for m in models:
        ns = [int(x.group(1)) for d in RUNS.glob(f"{task}__{m}__gold__n*__s*")
              if (x := re.fullmatch(rf"{task}__{m}__gold__n(\d+)__s\d+", d.name)) and (d / "DONE").exists()]
        if ns:
            tops[m] = max(ns)
    models = [m for m in models if m in tops]
    if not models:
        print(f"{out}: no runs yet")
        return
    fig, axes = plt.subplots(len(models), 2, figsize=(6.8, 3.0 * len(models)), squeeze=False)
    rows = []
    for r, model in enumerate(models):
        name, color, _ = STYLE[model]
        for col, n in enumerate((1000, tops[model])):
            ax = axes[r][col]
            got = _pooled(task, model, n, eval_set)
            ax.plot([0.5, 1], [0.5, 1], color="0.5", lw=0.8, ls="--")
            if got is None:
                ax.set_title(f"{name}, N = {n:,}: no runs", fontsize=8.5)
                continue
            y, p_raw, p_temp, k, t_mean = got
            sides = (("raw", p_raw, "-", "o", "raw"), ("temperature", p_temp, "--", "s", f"temp. (mean T {t_mean:.2f})"))
            for i, (calib, p, ls, mk, lab) in enumerate(sides):
                t = _bins(y, p)
                e = metrics.ece(y, p, REL_BINS)
                ax.bar(t.lo + 0.005 + 0.045 * i, t.share, width=0.04, align="edge", color=color, zorder=1,
                       alpha=0.2 if i == 0 else 0.45, lw=0, label="share (raw)" if i == 0 else "share (temp.)")
                ax.plot(t.mean_conf, t.accuracy, ls, marker=mk, ms=3.5, color=color, lw=1.4, zorder=3,
                        label=f"{lab}, ECE {e:.3f}")
                rows += [dict(task=task, model=model, labels="gold", n=n, eval_set=eval_set, calib=calib, k=k,
                              mean_T=t_mean, ece=e, **row) for row in t.to_dict("records")]
            ax.set_xlim(0.5, 1.0)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
            ax.set_title(f"{name}, N = {n:,}, {k} seeds pooled", fontsize=8.5)
            ax.legend(fontsize=6, loc="upper left")
            if r == len(models) - 1:
                ax.set_xlabel("confidence max(p, 1 - p)")
            if col == 0:
                ax.set_ylabel("accuracy (bars: share of items)")
    where = "in-distribution test" if eval_set == "test" else f"{eval_set} set"
    fig.suptitle(f"Reliability, {TASK_TITLE[task]}, gold labels, {where}", fontsize=9)
    fig.tight_layout()
    _save(fig, out)
    TAB.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TAB / f"{out}_bins.csv", index=False, float_format="%.4f")


def calibration_vs_n(df, out: str = "fig5_calibration"):
    """RQ3: ECE and BS-Error (raw scores, test) vs N; colour = model, line style = label source."""
    d = df[(df.tag == "") & (df.family.isin(["encoder", "decoder", "classical"])) & (df.eval_set == "test")
           & (df.calib == "raw") & (df.labels.isin(list(LABEL_STYLE)))]
    tasks = [t for t in TASK_TITLE if t in set(d.task)]
    if not tasks:
        print(f"{out}: no runs yet")
        return
    fig, axes = plt.subplots(2, len(tasks), figsize=(5 * len(tasks), 7), squeeze=False)
    for c, task in enumerate(tasks):
        for r, metric in enumerate(("ece", "bs_error")):
            ax = axes[r][c]
            g = d[(d.task == task) & (d.metric == metric)]
            for (model, src), h in g.groupby(["model", "labels"]):
                name, color, _ = STYLE.get(model, (model, None, "-"))
                s = h.groupby("n").value.mean()
                ax.plot(s.index, s.values, LABEL_STYLE[src], color=color, lw=1.3,
                        label=f"{name}" if src == "gold" else None)
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            ax.set_title(TASK_TITLE[task] if r == 0 else "", fontsize=10)
            ax.set_ylabel({"ece": "ECE", "bs_error": "BS-Error"}[metric] if c == 0 else "")
            if r == 1:
                ax.set_xlabel("training labels N")
    h, lab = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(8, len(lab)), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out)


def threshold_drift(df, out: str = "fig7_threshold_drift"):
    """RQ3: realized FPR at a validation-set FPR target of 0.05, by calibration method, family and eval set."""
    d = df[(df.tag == "") & (df.family.isin(["encoder", "decoder", "classical"])) & (df.metric == "at_fpr05_fpr")
           & (df.eval_set.isin(["test", "shift"]))]
    tasks = [t for t in TASK_TITLE if t in set(d.task)]
    if not tasks:
        print(f"{out}: no runs yet")
        return
    methods = ["raw", "temperature", "isotonic", "conformal"]
    fams = [f for f in ("classical", "encoder", "decoder") if f in set(d.family)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 3.8), squeeze=False)
    width = 0.8 / (2 * len(fams))
    for ax, task in zip(axes[0], tasks):
        g = d[d.task == task]
        k = 0
        for fam in fams:
            for es, hatch in (("test", ""), ("shift", "//")):
                h = g[(g.family == fam) & (g.eval_set == es)]
                vals = [h[h.calib == m].value for m in methods]
                pos = [i - 0.4 + width * (k + 0.5) for i in range(len(methods))]
                ax.boxplot(vals, positions=pos, widths=width * 0.9, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor={"classical": "0.7", "encoder": "#6baed6", "decoder": "#fd8d3c"}[fam],
                                         hatch=hatch), medianprops=dict(color="k"))
                k += 1
        ax.axhline(TARGET_FPR := 0.05, color="r", lw=1, ls="--")
        ax.set_xticks(range(len(methods)), methods)
        ax.set_title(TASK_TITLE[task])
        ax.set_ylabel("realized FPR (target 0.05)")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out)


def tables_rq3(df):
    d = df[(df.tag == "") & (df.family.isin(["encoder", "decoder", "classical"]))]
    TAB.mkdir(parents=True, exist_ok=True)
    cal = d[d.metric.isin(["ece", "brier", "bbs", "bs_error"]) & d.calib.isin(["raw", "temperature"])
            & d.eval_set.isin(["test", "shift"])]
    top = cal.groupby("task").n.transform("max")
    cal = cal[cal.n.isin([1000, 5000]) | (cal.n == top)]
    t6 = cal.pivot_table(index=["task", "model", "labels", "n", "eval_set", "calib"], columns="metric", values="value",
                         aggfunc="mean").round(4)
    t6.to_csv(TAB / "table6_calibration.csv")
    dr = d[d.metric.isin(["at_fpr05_fpr", "at_prec90_precision", "at_prec90_recall"]) & d.eval_set.isin(["test", "shift"])]
    dr = dr.assign(dev=lambda x: (x.value - 0.05).abs().where(x.metric == "at_fpr05_fpr", x.value))
    t7 = dr.pivot_table(index=["task", "family", "labels", "eval_set", "calib"], columns="metric", values="dev",
                        aggfunc="mean").round(4)
    t7 = t7.rename(columns={"at_fpr05_fpr": "mean |FPR - 0.05|", "at_prec90_precision": "precision @ target 0.90",
                            "at_prec90_recall": "recall @ target 0.90"})
    t7.to_csv(TAB / "table7_threshold_drift.csv")
    print(f"table6: {len(t6)} rows, table7: {len(t7)} rows -> results/tables/")


def main() -> int:
    df = load()
    learning_curves(df, "test", "fig2_learning_curves_test")
    learning_curves(df, "shift_api", "fig3_learning_curves_shift")
    table_breakeven(df)
    label_sources(df, metric="pr_auc")
    seed_agreement()
    calibration_vs_n(df)
    reliability()
    threshold_drift(df)
    tables_rq3(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
