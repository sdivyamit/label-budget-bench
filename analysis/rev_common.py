"""Shared pieces for the revision analyses (Phase 2; D36).

Everything here reads the per-item predictions directly; nothing in results/ outside results/revision/ is
written. Item order, item bootstrap and hosted-model alignment are those of analysis/headline.py.

Deployable F1 (review point 1): each run's threshold is the one that maximizes F1 on its own validation
set (the 1,000-item gold validation set; for budget-honest runs the held-out 20% of D_n), and is then
applied unchanged to test and shifted items. `f1_at_05` uses the model's own 0.5 cut-off and no labels.
Hosted LLMs return verdicts, so for them all three F1 variants are the same number.

Two-level bootstrap (review point 3): seeds are drawn with replacement within each cell and items with
replacement within each eval set, jointly; every cell of a task x eval set uses the same item draws, so
differences between cells (and against a hosted reference) are paired on items.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import headline as hl  # noqa: E402
from eval import fastboot as fb  # noqa: E402
from eval import metrics as M  # noqa: E402

OUT = ROOT / "results" / "revision"
NUMBERS = OUT / "revision_numbers.json"
TASKS = ["t1_rag_hallucination", "t2_response_safety", "t3_implicit_toxicity"]
SHORT = {"t1_rag_hallucination": "t1", "t2_response_safety": "t2", "t3_implicit_toxicity": "t3"}
FRONTIER = ["gpt61_sol", "gemini4_argon"]
F1S = ("f1_valthr", "f1_at_05", "max_f1")


@lru_cache(maxsize=None)
def index() -> pd.DataFrame:
    return hl.run_index()


@lru_cache(maxsize=4096)
def run(d: str) -> dict:
    """Validation scores, the validation-chosen threshold and aligned scores on test / shift / shift_api."""
    pp = pd.read_parquet(Path(d) / "preds.parquet")
    task = json.loads((Path(d) / "run.json").read_text())["task"]
    val = pp[pp.split == "val"]
    yv, pv = val.label.to_numpy(int), val.p.to_numpy(float)
    out = {"val": (yv, pv), "t_val": M.max_f1(yv, pv)[1] if 0 < yv.sum() < len(yv) else 0.5}
    for st in hl.SETS:
        g = pp[pp.split == ("test" if st == "test" else "shift")]
        if st == "shift_api":
            g = g[g.uid.isin(hl.api_uids(task))]
        uid, _ = hl.items(task, st)
        if len(g) == len(uid):
            out[st] = hl.align(uid, g.uid.to_numpy(), g.p.to_numpy())[0]
    return out


@lru_cache(maxsize=None)
def api() -> pd.DataFrame:
    a = pd.read_parquet(ROOT / "results" / "api_preds.parquet")
    return a[a.kind == "concise"]


def api_scores(task: str, model: str, eval_set: str) -> np.ndarray | None:
    """A hosted model's scores aligned to hl.items (concise prompt); None if it was not run on that set."""
    st = {"test": "test", "shift_api": "shift_api_concise"}.get(eval_set)
    if st is None:
        return None
    g = api()[(api().task == task) & (api().model == model) & (api().set == st)]
    uid, _ = hl.items(task, eval_set)
    return hl.align(uid, g.uid.to_numpy(), g.p.to_numpy())[0] if len(g) == len(uid) else None


@lru_cache(maxsize=None)
def ref_model(task: str, eval_set: str, weaker: bool = False) -> str | None:
    """Frontier-tier hosted model with the higher (weaker=True: lower) F1 on this eval set; None if not run there."""
    es = "shift_api" if eval_set == "shift" else eval_set
    _, y = hl.items(task, es)
    f = {m: M.f1_at(y, s, 0.5) for m in FRONTIER if (s := api_scores(task, m, es)) is not None}
    return (min if weaker else max)(f, key=f.get) if f else None


def floor(task: str, eval_set: str) -> float:
    """F1 of flagging every item (always-positive)."""
    _, y = hl.items(task, eval_set)
    pi = float(y.mean())
    return 2 * pi / (1 + pi)


def metric_fn(metric: str):
    if metric == "f1_at_05":
        return lambda w, y, s, t: fb.f1_at(w, y, s, 0.5)
    if metric == "f1_valthr":
        return lambda w, y, s, t: fb.f1_at(w, y, s, t)
    if metric.startswith("ece"):
        return lambda w, y, s, t: fb.ece(w, y, s)
    return lambda w, y, s, t: fb.METRICS[metric](w, y, s)


def weight_chunks(task: str, eval_set: str, b: int):
    """The item resamples of headline.py (row 0 of the first chunk = the full set)."""
    _, y = hl.items(task, eval_set)
    return list(fb.chunks(len(y), b, hl.boot_seed(task, eval_set)))


def item_reps(scores: list[tuple[np.ndarray, float]], task: str, eval_set: str, b: int, metric: str,
              chunks=None) -> np.ndarray:
    """[k seeds x (b+1)] metric values; scores = [(aligned scores, validation threshold)] per seed."""
    _, y = hl.items(task, eval_set)
    fn = metric_fn(metric)
    chunks = chunks if chunks is not None else weight_chunks(task, eval_set, b)
    return np.hstack([np.vstack([fn(w, y, s.astype(np.float32), t) for s, t in scores]) for w in chunks])


def seed_draws(k: int, b: int, seed: int) -> np.ndarray:
    """[b+1, k] seed indices; row 0 = every seed once (the point estimate)."""
    rng = np.random.default_rng(seed)
    return np.vstack([np.arange(k)[None, :], rng.integers(0, k, size=(b, k))])


def two_level(r: np.ndarray, seed: int) -> np.ndarray:
    """Seed mean per bootstrap replicate with seeds resampled too: r [k x (b+1)] -> [b+1]."""
    k, cols = r.shape
    idx = seed_draws(k, cols - 1, seed)
    return r[idx, np.arange(cols)[:, None]].mean(1)


def ci(reps: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    r = reps[1:][~np.isnan(reps[1:])]
    lo, hi = np.percentile(r, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]) if len(r) else (np.nan, np.nan)
    return float(reps[0]), float(lo), float(hi)


def cell_runs(task: str, family: str, model: str, labels: str, n: int, tag: str = "") -> list[str]:
    i = index()
    c = i[(i.task == task) & (i.family == family) & (i.model == model) & (i.labels == labels) & (i.n == n) & (i.tag == tag)]
    return sorted(c.dir)


def cell_seeds(task: str, family: str, model: str, labels: str, n: int, tag: str = "") -> dict[int, str]:
    i = index()
    c = i[(i.task == task) & (i.family == family) & (i.model == model) & (i.labels == labels) & (i.n == n) & (i.tag == tag)]
    return dict(zip(c.seed.astype(int), c.dir))


def per_seed(dirs: list[str], eval_set: str) -> list[tuple[np.ndarray, float]]:
    return [(r[eval_set], r["t_val"]) for r in map(run, dirs) if eval_set in r]


def save_numbers(section: str, values: dict) -> None:
    """Merge one section into results/revision/revision_numbers.json (keys are stable names for numbers.tex)."""
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(NUMBERS.read_text()) if NUMBERS.exists() else {}
    d[section] = values
    NUMBERS.write_text(json.dumps(d, indent=1, sort_keys=True, default=float))


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
