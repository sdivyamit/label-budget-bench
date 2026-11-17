"""Revision figures -> results/revision/figures/*.{pdf,png} + paper/submission/fig1_graphical_abstract.jpg.

Leaves every file in results/figures/ untouched (preserving results/FREEZE-2026-11-15.sha256).
Addresses review items 1, 2, 9 and minor items on figure density and Fig. S2 pooled vs seed-mean ECE:
  - Fig. 1 + graphical abstract: adds N=0 decoders, val-thr F1 / F1@0.5, and removes unqualified "(aligned labels)".
  - Figs. 2 & 3: deployable F1 (validation-chosen threshold) instead of oracle max-F1, with N=0 zero-shot points
    for all four decoders, constant-positive floor line, and N*_mean stars from results/revision/breakeven.csv.
  - Figs. 4 & S1 (fig4b): distinct colour per label source + 1k-gold anchors only (5 series instead of 7 monochrome),
    incorporating the two completed T2 N=1,000 seed-0 pilot runs.
  - Fig. 5: 6 series per panel instead of 24 (top row: raw ECE by label source for ModernBERT-large & Qwen3.5-2B;
    bottom row: raw Balanced Brier Score on gold labels across the 6 full-sweep models).
  - Fig. S2 (fig6): includes completed seed 0 at N=1,000 and prints both pooled ECE and seed-mean ECE in the legend.
  - Fig. 7: includes the two completed pilot cells.
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
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import fig1_design as f1d  # noqa: E402
import rev_common as rc  # noqa: E402
from eval import calibrate, metrics, stability  # noqa: E402

REV = ROOT / "results" / "revision"
FIG = REV / "figures"
SUB = ROOT / "paper" / "submission"
RUNS = ROOT / "preds" / "runs"

TASK_TITLE = {
    "t1_rag_hallucination": "T1 RAG hallucination",
    "t2_response_safety": "T2 response safety",
    "t3_implicit_toxicity": "T3 implicit toxicity",
}
STYLE = {
    "tfidf_lr": ("TF-IDF + LR", "0.45", "-"),
    "modernbert_base": ("ModernBERT-base", "#9ecae1", "-"),
    "modernbert_large": ("ModernBERT-large", "#3182bd", "-"),
    "deberta_v3_large": ("DeBERTa-v3-large", "#08519c", "--"),
    "gemma4_e2b": ("Gemma 4 E2B", "#fdae6b", "-"),
    "qwen35_2b": ("Qwen3.5-2B", "#e6550d", "-"),
    "gemma4_e4b": ("Gemma 4 E4B", "#fd8d3c", "--"),
    "qwen35_4b": ("Qwen3.5-4B", "#a63603", "--"),
}
REPRESENTATIVE = ["tfidf_lr", "modernbert_large", "qwen35_2b"]
FULL_SWEEP = ["tfidf_lr", "modernbert_base", "modernbert_large", "deberta_v3_large", "gemma4_e2b", "qwen35_2b"]
DECODERS = ["gemma4_e2b", "qwen35_2b", "gemma4_e4b", "qwen35_4b"]

SRC_STYLE = {
    "gold": ("gold", "#1b9e77", "-"),
    "open_teacher": ("open teacher", "#d95f02", "--"),
    "frontier_teacher": ("frontier teacher", "#7570b3", ":"),
}
MIX_1K_STYLE = {
    "mix1000_open_teacher": ("1k gold + open", "#d95f02", "D"),
    "mix1000_frontier_teacher": ("1k gold + frontier", "#7570b3", "s"),
}


def _save(fig, name: str, dpi: int = 200) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=dpi)
    plt.close(fig)
    print(f"wrote {FIG.relative_to(ROOT)}/{name}.pdf")


def build_fig1() -> None:
    """Updated Fig. 1 and IEEE Access 660x295 graphical abstract (< 45 KB JPEG)."""
    f1d.FACTORS = [
        ("Model family", "#deebf7", "#3182bd",
         ["TF-IDF + LR", "3 encoders (full fine-tuning)", "4 small decoders, 2B-4B (LoRA)"]),
        ("Label budget N", "#fee6ce", "#e6550d",
         ["N = 0 (decoders), 100 ... 20k", "5 seeds up to 2k, 3 above"]),
        ("Label source", "#e5f5e0", "#31a354",
         ["gold", "open teacher (Qwen3.5-9B)", "frontier teacher (GPT-6.1 Sol, N \u2264 5k)",
          "gold-anchored mixes (250 or 1k gold)"]),
    ]
    f1d.OUTPUTS = [
        ("Discrimination", ["PR-AUC, val F1", "F1@0.5, N*"]),
        ("Calibration", ["ECE, Brier, BBS", "raw / temp. / iso."]),
        ("Threshold drift", ["\u0394FPR, \u0394precision", "val \u2192 test, shift"]),
        ("Cost", ["GPU-hours", "USD / M items"]),
    ]

    # Single-column Fig. 1 with updated header (no unqualified "aligned labels" on all 3 tasks)
    w, fs = 3.5, 6.5
    m, inner, hd, arr, task_rh, out_h = 0.04, w - 0.08, 0.2, 0.24, 0.2, 0.5
    title_w, width = 0.98, 46
    fh = f1d.factors_height(fs, width)
    h = m + hd + 3 * task_rh + 0.1 + arr + hd + fh + arr + out_h + m
    fig, ax = f1d.canvas(w, h)
    top = h - m
    f1d.header(ax, m, top - hd + 0.03, "Tasks: in-distribution \u2192 shifted test", fs)
    top = f1d.tasks(ax, m, top - hd, inner, task_rh, fs, title_w=1.3)
    f1d.arrow(ax, (w / 2, top - 0.02), (w / 2, top - arr + 0.02), "fine-tune on N labelled pool items", fs - 0.5)
    top -= arr
    f1d.header(ax, m, top - hd + 0.03, "Crossed design: family \u00d7 N \u00d7 label source", fs)
    top = f1d.factors(ax, m, top - hd, inner, fs, title_w, width)
    f1d.arrow(ax, (w / 2, top - 0.02), (w / 2, top - arr + 0.02), "score val / test / shift; calibrate on val", fs - 0.5)
    gap = 0.05
    bw = (inner - 3 * gap) / 4
    for i, (name, lines) in enumerate(f1d.OUTPUTS):
        f1d.output_box(ax, m + i * (bw + gap), m, bw, out_h, name, lines, fs)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig1_design.pdf")
    fig.savefig(FIG / "fig1_design.png", dpi=300)
    plt.close(fig)
    print(f"wrote {FIG.relative_to(ROOT)}/fig1_design.pdf")

    # Wide graphical abstract (660 x 295 px)
    fig_w = f1d.wide()
    fig_w.savefig(FIG / "fig1_graphical_abstract.pdf")
    fig_w.savefig(FIG / "fig1_graphical_abstract.png", dpi=100)
    hi_png = FIG / "_ga_4x.png"
    fig_w.savefig(hi_png, dpi=400)
    plt.close(fig_w)
    im = Image.open(hi_png).convert("RGB").resize((660, 295), Image.Resampling.LANCZOS)
    hi_png.unlink(missing_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    jpg_path = SUB / "fig1_graphical_abstract.jpg"
    for q in range(92, 60, -2):
        im.save(jpg_path, format="JPEG", quality=q, subsampling=0, optimize=True)
        if jpg_path.stat().st_size < 44_500:
            break
    print(f"wrote {jpg_path.relative_to(ROOT)} ({jpg_path.stat().st_size:,} bytes, 660x295)")


def build_learning_curves(eval_set: str, out: str) -> None:
    """Deployable F1 (f1_valthr) vs N (gold labels) with N=0 zero-shot decoder points, floor, and N*_mean stars."""
    bc = pd.read_csv(REV / "breakeven_curves.csv")
    be = pd.read_csv(REV / "breakeven.csv")
    zs = pd.read_csv(REV / "zeroshot.csv")
    api_df = pd.read_csv(ROOT / "results" / "api_metrics.csv")

    bc = bc[(bc.eval_set == eval_set) & (bc.metric == "f1_valthr") & (bc.ref_kind == "better")]
    be_zs = be[(be.eval_set == eval_set) & (be.metric == "f1_valthr") & (be.ref_kind == "better") & (be.with_zeroshot)]
    be_no = be[(be.eval_set == eval_set) & (be.metric == "f1_valthr") & (be.ref_kind == "better") & (~be.with_zeroshot)]
    zs_c = zs[(zs.eval_set == eval_set) & (zs.cond == "zeroshot-concise") & (zs.metric == "f1_valthr")]

    x0 = 45.0  # log-scale x coordinate representing N = 0 (left of N = 100)
    x_sep = 67.0
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.2), squeeze=False)
    for ax, task in zip(axes[0], rc.TASKS):
        api_set = "test" if eval_set == "test" else "shift_api_concise"
        fa = api_df[(api_df.task == task) & (api_df.set == api_set) & (api_df.kind == "concise")]
        fr = fa[fa.model.isin(["gpt61_sol", "gemini4_argon"])]["f1_at_05"]
        lo, hi = float(fr.min()), float(fr.max())
        ax.axhspan(lo, hi if hi > lo + 1e-4 else lo + 0.002, color="0.86", zorder=0, label="frontier API band")

        jev = fa[fa.model == "jev"]["f1_at_05"]
        if len(jev):
            ax.axhline(float(jev.iloc[0]), color="purple", ls=":", lw=1.35, label="Jev (zero-label)")

        fl = rc.floor(task, eval_set)
        ax.axhline(fl, color="0.55", ls="-.", lw=1.0, zorder=1, label="all-positive floor")

        ax.axvline(x_sep, color="0.75", lw=0.8, ls=":")

        for model in STYLE:
            name, color, ls = STYLE[model]
            g = bc[(bc.task == task) & (bc.model == model)].sort_values("n")
            if g.empty:
                continue
            xs = g["n"].to_numpy(float)
            ys = g["q"].to_numpy(float)
            sds = g["seed_sd"].fillna(0.0).to_numpy(float)
            ax.plot(xs, ys, ls, color=color, marker="o", ms=3.0, lw=1.55, label=name)
            ax.fill_between(xs, ys - sds, ys + sds, color=color, alpha=0.14, lw=0)

            # Plot N = 0 point for decoders (concise prompt, raw 0.5 or val-thr? On N=0 there is no D_N,
            # zeroshot.csv f1_valthr uses the 1,000-item gold val set, matching D38!)
            zrow = zs_c[(zs_c.task == task) & (zs_c.model == model)]
            y0 = None
            if len(zrow):
                y0 = float(zrow["mean"].iloc[0])
                ax.plot([x0, xs[0]], [y0, ys[0]], ls=":", lw=1.0, color=color, alpha=0.75)
                ax.plot([x0], [y0], marker="o", ms=4.2, mfc="white", mec=color, mew=1.4, zorder=4)

            # Break-even star (with_zeroshot for decoders, standard for others)
            brow = be_zs[(be_zs.task == task) & (be_zs.model == model)] if model in DECODERS else be_no[(be_no.task == task) & (be_no.model == model)]
            if len(brow):
                ns = float(brow["n_star_mean"].iloc[0])
                if np.isfinite(ns):
                    if ns == 0 and y0 is not None:
                        ax.plot([x0], [y0], marker="*", ms=11.5, color=color, mec="k", mew=0.6, zorder=5)
                    elif int(ns) in set(g["n"]):
                        yq = float(g[g["n"] == int(ns)]["q"].iloc[0])
                        ax.plot([ns], [yq], marker="*", ms=11.5, color=color, mec="k", mew=0.6, zorder=5)

        ax.set_xscale("log")
        top_n = int(bc[bc.task == task]["n"].max())
        ticks = [x0, 100, 250, 500, 1000, 2000, 5000] + ([10000, 20000] if top_n == 20000 else [top_n])
        tlabels = ["0", "100", "250", "500", "1k", "2k", "5k"] + (["10k", "20k"] if top_n == 20000 else [f"{top_n/1000:.1f}k"])
        ax.set_xticks(ticks, tlabels, fontsize=7.5)
        ax.minorticks_off()
        ax.set_xlim(34, top_n * 1.22)
        ax.set_title(TASK_TITLE[task], fontsize=10)
        ax.set_xlabel("gold training labels N (0 = zero-shot decoder)")
        ax.grid(alpha=0.28)

    where = "in-distribution test" if eval_set == "test" else "shifted test"
    axes[0][0].set_ylabel(f"deployable F1 ({where})")
    h, lab = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    _save(fig, out)


def _all_main_runs_metrics() -> pd.DataFrame:
    """Load metrics.parquet and supplement the 2 re-run pilot cells (T2, gold, N=1000, seed 0) so no cell is 4-seed."""
    m = pd.read_parquet(ROOT / "results" / "metrics.parquet")
    m["tag"] = m["tag"].fillna("")
    # Add the 2 completed pilot cells if missing from metrics.parquet
    existing = set(m["run_id"].unique())
    extra_rows = []
    for mod, fam in (("modernbert_base", "encoder"), ("qwen35_2b", "decoder")):
        rid = f"t2_response_safety__{mod}__gold__n1000__s0"
        d = RUNS / rid
        if rid not in existing and (d / "DONE").exists() and (d / "metrics.json").exists():
            mj = json.loads((d / "metrics.json").read_text())
            # metrics.json stores nested eval_set -> calib -> metric
            for es, cal_dict in mj.items():
                if not isinstance(cal_dict, dict):
                    continue
                for cal, met_dict in cal_dict.items():
                    if not isinstance(met_dict, dict):
                        continue
                    for met, val in met_dict.items():
                        if isinstance(val, (int, float)):
                            extra_rows.append(dict(
                                run_id=rid, task="t2_response_safety", family=fam, model=mod,
                                labels="gold", n=1000, seed=0, tag="", eval_set=es, calib=cal,
                                metric=met, value=float(val)
                            ))
    if extra_rows:
        m = pd.concat([m, pd.DataFrame(extra_rows)], ignore_index=True)
    return m


def build_label_sources(df: pd.DataFrame) -> None:
    """Fig. 4: PR-AUC vs N for 3 representative models; 3 coloured curves + 2 1k-anchor markers (5 series, not 7)."""
    d = df[(df.tag == "") & (df.family != "api") & (df.metric == "pr_auc") & (df.eval_set == "test") & (df.calib == "raw")]
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 9.2), squeeze=False)
    for r, task in enumerate(rc.TASKS):
        for c, model in enumerate(REPRESENTATIVE):
            ax = axes[r][c]
            mname, _, _ = STYLE[model]
            g = d[(d.task == task) & (d.model == model)]
            for src, (sname, scolor, sls) in SRC_STYLE.items():
                h = g[g.labels == src]
                if len(h):
                    s = h.groupby("n").value.agg(["mean", "std"]).fillna(0)
                    ax.plot(s.index, s["mean"], sls, color=scolor, marker="o", ms=2.8, lw=1.55, label=sname)
                    ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=scolor, alpha=0.11, lw=0)
            for src, (m_label, mcolor, mk) in MIX_1K_STYLE.items():
                h = g[g.labels == src]
                if len(h):
                    s = h.groupby("n").value.mean()
                    ax.scatter(s.index, s.values, marker=mk, s=30, color=mcolor, edgecolors="k", linewidths=0.5,
                               zorder=4, label=m_label)
            ax.set_xscale("log")
            ax.grid(alpha=0.28)
            ax.set_title(f"{TASK_TITLE[task]} \u2014 {mname}", fontsize=9)
            if r == 2:
                ax.set_xlabel("training labels N")
            if c == 0:
                ax.set_ylabel("PR-AUC (in-distribution test)")
    seen = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    _save(fig, "fig4_label_sources")


def build_seed_agreement() -> None:
    """Fig. S1 (fig4b_seed_agreement): Krippendorff's alpha_K with distinct colours & 1k anchors only (5 series)."""
    c = pd.read_csv(ROOT / "results" / "tables" / "cells_ci.csv", keep_default_na=False, na_values=[""])
    c["tag"] = c["tag"].fillna("")
    d = c[(c.tag == "") & c.family.isin(["classical", "encoder", "decoder"]) & (c.eval_set == "test") & (c.metric == "alpha_k")].copy()

    # Update the 2 re-run pilot cells at T2 gold N=1000 so K=5 there too
    _, y_t2 = rc.hl.items("t2_response_safety", "test")
    for mod, fam in (("modernbert_base", "encoder"), ("qwen35_2b", "decoder")):
        dirs = rc.cell_runs("t2_response_safety", fam, mod, "gold", 1000)
        if len(dirs) == 5:
            verdicts = np.vstack([(rc.run(dr)["test"] >= 0.5).astype(int) for dr in dirs])
            ak = float(stability.alpha_k(verdicts))
            mask = (d.task == "t2_response_safety") & (d.model == mod) & (d.labels == "gold") & (d.n == 1000)
            d.loc[mask, "mean"] = ak

    fig, axes = plt.subplots(3, 3, figsize=(12.6, 9.2), squeeze=False)
    for r, task in enumerate(rc.TASKS):
        for col, model in enumerate(REPRESENTATIVE):
            ax = axes[r][col]
            mname, _, _ = STYLE[model]
            g = d[(d.task == task) & (d.model == model)]
            for src, (sname, scolor, sls) in SRC_STYLE.items():
                s = g[g.labels == src].set_index("n")["mean"].sort_index()
                if len(s):
                    ax.plot(s.index, s.values, sls, color=scolor, marker="o", ms=2.8, lw=1.55, label=sname)
            for src, (m_label, mcolor, mk) in MIX_1K_STYLE.items():
                h = g[g.labels == src]
                if len(h):
                    ax.scatter(h.n, h["mean"], marker=mk, s=30, color=mcolor, edgecolors="k", linewidths=0.5,
                               zorder=4, label=m_label)
            ax.axvline(np.sqrt(2000 * 5000), color="0.6", lw=0.8, ls=":")
            ax.text(np.sqrt(2000 * 5000), 0.03, " K=5 | K=3", transform=ax.get_xaxis_transform(),
                    fontsize=6.5, color="0.4", ha="center")
            ax.set_xscale("log")
            ax.grid(alpha=0.28)
            ax.set_title(f"{TASK_TITLE[task]} \u2014 {mname}", fontsize=9)
            if r == 2:
                ax.set_xlabel("training items N")
            if col == 0:
                ax.set_ylabel(r"seed agreement $\alpha_K$ (test, verdicts at 0.5)")
    seen = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    _save(fig, "fig4b_seed_agreement")


def build_calibration_vs_n(df: pd.DataFrame) -> None:
    """Fig. 5: 6 series per panel instead of 24.
    Top row: Raw ECE vs N for ModernBERT-large and Qwen3.5-2B across 3 label sources (6 curves).
    Bottom row: Raw BBS (Balanced Brier Score) vs N on gold labels across the 6 full-sweep models (6 curves).
    """
    d = df[(df.tag == "") & df.family.isin(["encoder", "decoder", "classical"]) & (df.eval_set == "test") & (df.calib == "raw")]
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 6.8), squeeze=False)

    # Row 0: Raw ECE by label source (ModernBERT-large vs Qwen3.5-2B)
    ls_map = {"gold": ("-", "gold"), "open_teacher": ("--", "open"), "frontier_teacher": (":", "frontier")}
    for c, task in enumerate(rc.TASKS):
        ax = axes[0][c]
        g = d[(d.task == task) & (d.metric == "ece")]
        for mod in ("modernbert_large", "qwen35_2b"):
            mname, color, _ = STYLE[mod]
            for src, (ls, slabel) in ls_map.items():
                h = g[(g.model == mod) & (g.labels == src)]
                if len(h):
                    s = h.groupby("n").value.mean()
                    ax.plot(s.index, s.values, ls, color=color, marker="o", ms=2.6, lw=1.5,
                            label=f"{mname} ({slabel})")
        ax.set_xscale("log")
        ax.grid(alpha=0.28)
        ax.set_title(TASK_TITLE[task], fontsize=10)
        if c == 0:
            ax.set_ylabel("raw ECE (test, by label source)")
        if c == 2:
            h_top, lab_top = ax.get_legend_handles_labels()  # legend drawn below the panels (never over data)

    # Row 1: Raw BBS on gold labels across the 6 full-sweep models
    for c, task in enumerate(rc.TASKS):
        ax = axes[1][c]
        g = d[(d.task == task) & (d.metric == "bbs") & (d.labels == "gold")]
        for mod in FULL_SWEEP:
            mname, color, ls = STYLE[mod]
            h = g[g.model == mod]
            if len(h):
                s = h.groupby("n").value.mean()
                ax.plot(s.index, s.values, ls, color=color, marker="o", ms=2.6, lw=1.5, label=mname)
        ax.set_xscale("log")
        ax.grid(alpha=0.28)
        ax.set_xlabel("training labels N")
        if c == 0:
            ax.set_ylabel("raw BBS (test, gold labels)")

    h_bot, lab_bot = axes[1][-1].get_legend_handles_labels()
    fig.legend(h_top, lab_top, loc="lower center", ncol=6, fontsize=8, frameon=False,
               title="Top row (raw ECE; solid gold, dashed open teacher, dotted frontier teacher):",
               title_fontsize=8, bbox_to_anchor=(0.5, 0.075))
    fig.legend(h_bot, lab_bot, loc="lower center", ncol=6, fontsize=8, frameon=False,
               title="Bottom row (raw BBS, gold labels):", title_fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    _save(fig, "fig5_calibration")


def _pooled_and_seed_mean(task: str, model: str, n: int, eval_set: str = "test"):
    ys, raw, temp, ts, ece_raw_s, ece_tmp_s = [], [], [], [], [], []
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
        yi = g.label.to_numpy()
        pri = g.p.to_numpy()
        pti = cal(pri)
        ys.append(yi); raw.append(pri); temp.append(pti); ts.append(cal.T)
        ece_raw_s.append(metrics.ece(yi, pri, 10))
        ece_tmp_s.append(metrics.ece(yi, pti, 10))
    return (np.concatenate(ys), np.concatenate(raw), np.concatenate(temp),
            len(ys), float(np.mean(ts)), float(np.mean(ece_raw_s)), float(np.mean(ece_tmp_s)))


def _bins(y, p, n_bins: int = 10) -> pd.DataFrame:
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    t = pd.DataFrame(dict(bin=b, conf=conf, correct=correct)).groupby("bin").agg(
        count=("conf", "size"), mean_conf=("conf", "mean"), accuracy=("correct", "mean")).reset_index()
    t["lo"], t["hi"], t["share"] = t.bin / n_bins, (t.bin + 1) / n_bins, t["count"] / len(p)
    return t


def build_reliability() -> None:
    """Fig. S2 (fig6_reliability): includes all 5 seeds at N=1,000 and prints pooled + seed-mean ECE."""
    task = "t2_response_safety"
    models = ("modernbert_large", "qwen35_2b")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), squeeze=False)
    for r, model in enumerate(models):
        name, color, _ = STYLE[model]
        for col, n in enumerate((1000, 20000)):
            ax = axes[r][col]
            y, p_raw, p_temp, k, t_mean, sm_raw, sm_tmp = _pooled_and_seed_mean(task, model, n, "test")
            ax.plot([0.5, 1], [0.5, 1], color="0.5", lw=0.8, ls="--")
            sides = (
                ("raw", p_raw, "-", "o", "raw", sm_raw),
                ("temperature", p_temp, "--", "s", f"temp. (T={t_mean:.2f})", sm_tmp),
            )
            for i, (_, p, ls, mk, lab, sm_e) in enumerate(sides):
                t = _bins(y, p, 10)
                ep = metrics.ece(y, p, 10)
                ax.bar(t.lo + 0.005 + 0.045 * i, t.share, width=0.04, align="edge", color=color, zorder=1,
                       alpha=0.2 if i == 0 else 0.45, lw=0, label="share (raw)" if i == 0 else "share (temp.)")
                ax.plot(t.mean_conf, t.accuracy, ls, marker=mk, ms=3.5, color=color, lw=1.4, zorder=3,
                        label=f"{lab}: pooled {ep:.3f} (seed mean {sm_e:.3f})")
            ax.set_xlim(0.5, 1.0)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
            ax.set_title(f"{name}, N = {n:,} ({k} seeds pooled)", fontsize=8.5)
            ax.legend(fontsize=5.8, loc="upper left")
            if r == 1:
                ax.set_xlabel("confidence max(p, 1 - p)")
            if col == 0:
                ax.set_ylabel("accuracy (bars: share of items)")
    fig.suptitle("Reliability, T2 response safety, gold labels, in-distribution test", fontsize=9.5)
    fig.tight_layout()
    _save(fig, "fig6_reliability")


def build_threshold_drift(df: pd.DataFrame) -> None:
    """Fig. 7: realized FPR at target 0.05 across all main-sweep runs (including the 2 completed pilot runs)."""
    d = df[(df.tag == "") & df.family.isin(["encoder", "decoder", "classical"])
           & (df.metric == "at_fpr05_fpr") & df.eval_set.isin(["test", "shift"])]
    methods = ["raw", "temperature", "isotonic", "conformal"]
    mlabels = ["raw", "temp.", "isotonic", "conformal"]
    fams = ["classical", "encoder", "decoder"]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 3.8), squeeze=False)
    width = 0.8 / (2 * len(fams))
    for ax, task in zip(axes[0], rc.TASKS):
        g = d[d.task == task]
        k = 0
        for fam in fams:
            for es, hatch in (("test", ""), ("shift", "//")):
                h = g[(g.family == fam) & (g.eval_set == es)]
                vals = [h[h.calib == m].value for m in methods]
                pos = [i - 0.4 + width * (k + 0.5) for i in range(len(methods))]
                ax.boxplot(vals, positions=pos, widths=width * 0.9, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor={"classical": "0.7", "encoder": "#6baed6", "decoder": "#fd8d3c"}[fam],
                                         hatch=hatch),
                           medianprops=dict(color="k"))
                k += 1
        ax.axhline(0.05, color="r", lw=1, ls="--")
        ax.set_xticks(range(len(methods)), mlabels)
        ax.set_title(TASK_TITLE[task])
        if task == rc.TASKS[0]:
            ax.set_ylabel("realized FPR (target 0.05)")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "fig7_threshold_drift")


def main() -> int:
    build_fig1()
    build_learning_curves("test", "fig2_learning_curves_test")
    build_learning_curves("shift_api", "fig3_learning_curves_shift")
    df = _all_main_runs_metrics()
    build_label_sources(df)
    build_seed_agreement()
    build_calibration_vs_n(df)
    build_reliability()
    build_threshold_drift(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
