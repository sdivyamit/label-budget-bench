"""Fig. 1 (plan §5 I): the crossed design, model family x label budget N x label source, on three tasks with
their shift pairs, evaluated for discrimination, calibration, threshold drift and cost.

Two layouts from the same content: a vertical one for the IEEE single column (3.5 in wide; PDF + PNG) and a
left-to-right one exported at exactly 660 x 295 px for the IEEE Access graphical abstract.

  python analysis/fig1_design.py      # -> results/figures/fig1_design.{pdf,png}, fig1_graphical_abstract.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"

TITLE = "Label budgets, distilled labels and calibrated scores for high-ambiguity text classification"
TASKS = [("T1 RAG hallucination", "RAGTruth QA+Summary", "Data2txt"),
         ("T2 Response safety", "BeaverTails", "WildGuardTest"),
         ("T3 Implicit toxicity", "ToxiGen, 10 groups", "3 held-out groups")]
FACTORS = [("Model family", "#deebf7", "#3182bd",
            ["TF-IDF + LR", "3 encoders (full fine-tuning)", "4 small decoders, 2B-4B (LoRA)"]),
           ("Label budget N", "#fee6ce", "#e6550d", ["N = 100 ... 20k, log-spaced", "5 seeds up to 2k, 3 above"]),
           ("Label source", "#e5f5e0", "#31a354",
            ["gold", "open teacher (Qwen3.5-9B)", "frontier teacher (GPT-6.1 Sol, N \u2264 5k)",
             "gold-anchored mixes (250 or 1k gold)"])]
REFS = ["Zero-label references: 4 hosted LLMs (verdicts only),", "Jev (graded scores), off-the-shelf detectors"]
OUTPUTS = [("Discrimination", ["PR-AUC, max-F1", "break-even N*"]),
           ("Calibration", ["ECE, Brier, BBS", "raw / temp. / iso."]),
           ("Threshold drift", ["\u0394FPR, \u0394precision", "val \u2192 test, shift"]),
           ("Cost", ["GPU-hours", "USD / M items"])]
TASK_FC, TASK_EC, OUT_FC, OUT_EC = "#f2f2f2", "0.45", "#efedf5", "#756bb1"
SEP, PAD, GAP = "  \u00b7  ", 0.65, 0.14


def line_h(fs: float) -> float:
    return fs * 1.38 / 72


def wrap(atoms: list[str], width: int) -> list[str]:
    """Greedy: join atoms with a middle dot while the line stays within `width` characters."""
    lines = [atoms[0]]
    for a in atoms[1:]:
        if len(lines[-1]) + 3 + len(a) <= width:
            lines[-1] += SEP + a
        else:
            lines.append(a)
    return lines


def box(ax, x, y, w, h, fc, ec, ls="-", lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=ec, ls=ls, lw=lw))


def arrow(ax, xy0, xy1, label=None, fs=6.0):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=7, lw=0.9, color="0.3"))
    if label:
        ax.text((xy0[0] + xy1[0]) / 2 + 0.07, (xy0[1] + xy1[1]) / 2, label, fontsize=fs, va="center",
                color="0.25", style="italic")


def header(ax, x, y, text, fs):
    ax.text(x, y, text, fontsize=fs, weight="bold", va="bottom", color="0.15")


def tasks(ax, x, top, w, rh, fs, title_w=None) -> float:
    """Three stacked task rows from `top` down; title_w None puts the name above the pair. Returns the bottom."""
    gap = 0.05
    for i, (name, src, shift) in enumerate(TASKS):
        y = top - (i + 1) * rh - i * gap
        box(ax, x, y, w, rh, TASK_FC, TASK_EC)
        pair = f"{src} \u2192 {shift}"
        if title_w is None:
            ax.text(x + 0.06, y + rh * 0.68, name, fontsize=fs + 0.5, weight="bold", va="center")
            ax.text(x + 0.06, y + rh * 0.30, pair, fontsize=fs, va="center")
        else:
            ax.text(x + 0.06, y + rh / 2, name, fontsize=fs + 0.5, weight="bold", va="center")
            ax.text(x + title_w, y + rh / 2, pair, fontsize=fs, va="center")
    return top - 3 * rh - 2 * gap


def factors(ax, x, top, w, fs, title_w, width: int) -> float:
    """Factor rows joined by a cross, then the dashed reference row; heights follow the wrapped text."""
    lh = line_h(fs)
    for i, (name, fc, ec, atoms) in enumerate(FACTORS):
        lines = wrap(atoms, width)
        rh = lh * (len(lines) + PAD)
        top -= rh
        box(ax, x, top, w, rh, fc, ec)
        ax.text(x + 0.06, top + rh / 2, name, fontsize=fs + 0.5, weight="bold", va="center", color=ec)
        ax.text(x + title_w, top + rh / 2, "\n".join(lines), fontsize=fs, va="center", linespacing=1.38)
        if i < len(FACTORS) - 1:
            ax.text(x + w / 2, top - GAP / 2, "\u00d7", fontsize=fs + 2, weight="bold", ha="center", va="center",
                    color="0.25")
        top -= GAP
    rh = lh * (len(REFS) + PAD)
    top -= rh
    box(ax, x, top, w, rh, "white", "0.5", ls=(0, (3, 2)))
    ax.text(x + 0.06, top + rh / 2, "\n".join(REFS), fontsize=fs - 0.3, va="center", color="0.25", linespacing=1.38)
    return top


def factors_height(fs, width) -> float:
    rows = [len(wrap(a, width)) for *_, a in FACTORS] + [len(REFS)]
    return sum(line_h(fs) * (k + PAD) for k in rows) + GAP * len(FACTORS)


def output_box(ax, x, y, w, h, name, lines, fs):
    box(ax, x, y, w, h, OUT_FC, OUT_EC)
    ax.text(x + w / 2, y + h * 0.71, name, fontsize=fs, weight="bold", ha="center", va="center", color=OUT_EC)
    ax.text(x + w / 2, y + h * 0.33, "\n".join(lines), fontsize=fs - 0.4, ha="center", va="center", linespacing=1.3)


def canvas(w: float, h: float):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def column(w: float = 3.5, fs: float = 6.5):
    """Single-column layout: tasks, crossed factors, outputs, top to bottom; height follows the content."""
    m, inner, hd, arr, task_rh, out_h = 0.04, w - 0.08, 0.2, 0.24, 0.2, 0.5
    title_w, width = 0.98, 46
    fh = factors_height(fs, width)
    h = m + hd + 3 * task_rh + 0.1 + arr + hd + fh + arr + out_h + m
    fig, ax = canvas(w, h)
    top = h - m
    header(ax, m, top - hd + 0.03, "Tasks: in-distribution \u2192 shifted test (aligned labels)", fs)
    top = tasks(ax, m, top - hd, inner, task_rh, fs, title_w=1.3)
    arrow(ax, (w / 2, top - 0.02), (w / 2, top - arr + 0.02), "fine-tune on N labelled pool items", fs - 0.5)
    top -= arr
    header(ax, m, top - hd + 0.03, "Crossed design: family \u00d7 N \u00d7 label source", fs)
    top = factors(ax, m, top - hd, inner, fs, title_w, width)
    arrow(ax, (w / 2, top - 0.02), (w / 2, top - arr + 0.02), "score val / test / shift; calibrate on val", fs - 0.5)
    gap = 0.05
    bw = (inner - 3 * gap) / 4
    for i, (name, lines) in enumerate(OUTPUTS):
        output_box(ax, m + i * (bw + gap), m, bw, out_h, name, lines, fs)
    return fig


def wide(w: float = 6.6, h: float = 2.95, fs: float = 6.8):
    """Graphical-abstract layout: tasks -> crossed factors -> outputs, left to right."""
    fig, ax = canvas(w, h)
    top = h - 0.4
    ax.text(w / 2, h - 0.06, TITLE, fontsize=fs + 1.6, weight="bold", ha="center", va="top", color="0.1")
    xt, wt = 0.05, 1.84
    header(ax, xt, top + 0.03, "Tasks: in-distribution \u2192 shift", fs + 0.2)
    t_bottom = tasks(ax, xt, top, wt, 0.4, fs - 0.3)
    ax.text(xt, t_bottom - 0.08, "Fixed 1,000-item gold validation\nset per task fits temperature,\n"
            "isotonic and conformal thresholds", fontsize=fs - 0.6, va="top", color="0.3", linespacing=1.35)
    xd = xt + wt + 0.25
    xo = w - 0.05 - 1.2
    wd = xo - 0.25 - xd
    header(ax, xd, top + 0.03, "Crossed design", fs + 0.2)
    f_bottom = factors(ax, xd, top, wd, fs - 0.2, title_w=1.04, width=39)
    header(ax, xo, top + 0.03, "Outputs", fs + 0.2)
    gap, ow = 0.05, w - 0.05 - xo
    oh = (top - f_bottom - 3 * gap) / 4
    for i, (name, lines) in enumerate(OUTPUTS):
        output_box(ax, xo, top - (i + 1) * oh - i * gap, ow, oh, name, lines, fs)
    mid = (top + t_bottom) / 2
    arrow(ax, (xt + wt + 0.03, mid), (xd - 0.03, mid))
    arrow(ax, (xd + wd + 0.03, mid), (xo - 0.03, mid))
    ax.text(xo, f_bottom - 0.06, "per task, family, N and\nlabel source; 95% CIs", fontsize=fs - 0.6,
            va="top", color="0.3", linespacing=1.35)
    return fig


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    fig = column()
    fig.savefig(FIG / "fig1_design.pdf")
    fig.savefig(FIG / "fig1_design.png", dpi=300)
    plt.close(fig)
    fig = wide()
    fig.savefig(FIG / "fig1_graphical_abstract.png", dpi=100)
    fig.savefig(FIG / "fig1_graphical_abstract.pdf")
    plt.close(fig)
    print("fig1_design -> results/figures/fig1_design.pdf (3.5 in wide), fig1_graphical_abstract.png (660 x 295 px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
