"""Shared pieces for every training family: data for a run, label sources, outputs, metrics."""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
from eval import calibrate, metrics  # noqa: E402
from splits import chain  # noqa: E402

TRAIN_CFG = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())
MODELS = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
TASKS_CFG = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["tasks"]
RUNS = ROOT / "preds" / "runs"
CKPT = ROOT / "checkpoints"


@dataclass
class RunSpec:
    family: str            # encoder | decoder | classical
    model: str             # key in configs/models.yaml
    task: str
    n: int
    seed: int
    labels: str = "gold"
    hp: dict = field(default_factory=dict)   # overrides of configs/train.yaml
    tag: str = ""                            # e.g. hp-sweep id or ablation name
    max_len: int | None = None               # override (smoke tests)
    max_steps: int | None = None             # override (smoke tests)
    eval_limit: int | None = None            # evaluate only the first k items per split (smoke tests)

    @property
    def run_id(self) -> str:
        base = f"{self.task}__{self.model}__{self.labels}__n{self.n}__s{self.seed}"
        tag = self.tag or ("smoke" if (self.max_len or self.max_steps or self.eval_limit) else "")
        return base + (f"__{tag}" if tag else "")

    @property
    def out(self) -> Path:
        return RUNS / self.run_id

    @property
    def ckpt(self) -> Path:
        return CKPT / self.run_id

    def done(self) -> bool:
        return (self.out / "DONE").exists()


def hyperparams(spec: RunSpec) -> dict:
    hp = TRAIN_CFG["hp"][spec.family]
    base = hp.get(spec.model, hp.get("default", {}))
    return {**base, **spec.hp}


def n_steps(spec: RunSpec, batch: int) -> int:
    if spec.max_steps:
        return spec.max_steps
    ep = next(e["epochs"] for e in TRAIN_CFG["epochs"] if spec.n <= e["n_max"])
    steps = max(TRAIN_CFG["min_steps"], math.ceil(ep * spec.n / batch))
    caps = TRAIN_CFG["max_steps"]
    cap = caps.get(spec.n) or caps[min((k for k in caps if k >= spec.n), default=max(caps))]
    return min(steps, cap)


def max_len(spec: RunSpec, model_max: int) -> int:
    return spec.max_len or min(model_max, TRAIN_CFG["max_len"][spec.task])


def load_all(task: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / f"{task}.parquet")


def _frontier_records(task: str) -> list[dict]:
    cfg = TRAIN_CFG["label_sources"]["frontier_teacher"]
    prompt_v = yaml.safe_load((ROOT / "labels" / "prompts" / f"{task}.yaml").read_text())["version"]
    path = ROOT / "preds" / "api" / task / f"{cfg['api_model']}__{prompt_v}_{cfg['kind']}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def refused_uids(task: str, source: str) -> set[str]:
    """Items the teacher's provider refused to label (content filter); only the frontier teacher can refuse."""
    return {r["uid"] for r in _frontier_records(task) if r.get("refused")} if source == "frontier_teacher" else set()


def teacher_labels(task: str, source: str, drop_refused: bool = False) -> pd.Series:
    """uid -> 0/1 teacher verdict (unparseable verdicts dropped).

    A refused item (D25) takes the task's `refusal_label` from configs/tasks.yaml (T2: 1 = harmful). It has no
    label, and is dropped from training, if the task sets none or with drop_refused (robustness ablation).
    """
    ls = TRAIN_CFG["label_sources"]
    if source == "open_teacher":
        df = pd.read_parquet(ROOT / ls["open_teacher"]["path"].format(task=task))
        return df.set_index("uid").verdict.dropna().astype(int)
    if source == "frontier_teacher":
        fill = None if drop_refused else TASKS_CFG[task].get("refusal_label")
        s = pd.Series({r["uid"]: fill if r.get("refused") else r["verdict"] for r in _frontier_records(task)},
                      dtype="float64")
        return s.dropna().astype(int)
    raise ValueError(source)


def _labelled_chain(spec: RunSpec) -> pd.DataFrame:
    """D_n in chain order with labels from spec.labels and a per-item loss weight `w`.

    gold | open_teacher | frontier_teacher | mix{m}_{teacher}: first m chain items gold, rest teacher.
    hp `gold_weight` (mixtures only, ablation) up-weights the gold anchor in the loss.
    """
    d = chain(spec.task, spec.seed, spec.n)
    d["gold"] = d["label"]
    d["w"] = 1.0
    src = spec.labels
    m = re.fullmatch(r"mix(\d+)_(open_teacher|frontier_teacher)", src)
    if src == "gold":
        return d
    teacher = m.group(2) if m else src
    drop = bool(hyperparams(spec).get("drop_refused"))
    t = teacher_labels(spec.task, teacher, drop_refused=drop)
    d["label"] = d.uid.map(t)
    if m:
        k = int(m.group(1))
        d.loc[: k - 1, "label"] = d.loc[: k - 1, "gold"]
        d.loc[: k - 1, "w"] = float(hyperparams(spec).get("gold_weight", 1.0))
    unlabelled = d["label"].isna()
    if drop:                                   # refused items are left out on purpose, not missing
        unlabelled &= ~d.uid.isin(refused_uids(spec.task, teacher))
    missing = int(unlabelled.sum())
    if missing > max(5, 0.02 * len(d)):
        raise RuntimeError(f"{src}: {missing} of {len(d)} training items have no teacher label "
                           "(teacher file incomplete?)")
    d = d.dropna(subset=["label"]).copy()
    d["label"] = d["label"].astype(int)
    return d


def _honest_split(d: pd.DataFrame, seed: int, frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Budget-honest ablation: a stratified 20% of D_n becomes the validation set (plan §6.4)."""
    rng = np.random.default_rng([seed, 11])
    val_idx = []
    for _, g in d.groupby("label"):
        k = int(round(frac * len(g)))
        val_idx += list(rng.choice(g.index.to_numpy(), size=k, replace=False))
    val = d.loc[sorted(val_idx)]
    return d.drop(index=val.index), val.reset_index(drop=True)


def training_set(spec: RunSpec) -> pd.DataFrame:
    """Training items (chain order, teacher-dropped items removed; counted in run.json)."""
    d = _labelled_chain(spec)
    if hyperparams(spec).get("budget_honest"):
        d, _ = _honest_split(d, spec.seed)
    return d


def eval_sets(spec: RunSpec) -> dict[str, pd.DataFrame]:
    df = load_all(spec.task)
    out = {s: df[df.split == s].reset_index(drop=True) for s in ("val", "test", "shift")}
    if hyperparams(spec).get("budget_honest"):
        out["val"] = _honest_split(_labelled_chain(spec), spec.seed)[1]
    if spec.eval_limit:
        out = {k: v.head(spec.eval_limit) for k, v in out.items()}
    return out


def write_outputs(spec: RunSpec, preds: dict[str, tuple[pd.DataFrame, np.ndarray]], info: dict) -> None:
    """Per-item predictions + raw and temperature-calibrated metrics on every split."""
    spec.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, (df, p) in preds.items():
        rows.append(pd.DataFrame({"uid": df.uid, "split": split, "label": df.label.astype(int),
                                  "group": df.get("group"), "p": p}))
    pp = pd.concat(rows, ignore_index=True)
    pp.to_parquet(spec.out / "preds.parquet", index=False)
    val = pp[pp.split == "val"]
    temp = calibrate.fit("temperature", val.p, val.label)
    m = {}
    for split, g in pp.groupby("split"):
        m[split] = {"raw": metrics.all_metrics(g.label, g.p),
                    "temperature": metrics.all_metrics(g.label, temp(g.p))}
    m["temperature_T"] = temp.T
    (spec.out / "metrics.json").write_text(json.dumps(m, indent=1))
    (spec.out / "run.json").write_text(json.dumps({**asdict(spec), "run_id": spec.run_id, **info}, indent=1, default=str))
    (spec.out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_info() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"gpu": "cpu"}
    return {"gpu": torch.cuda.get_device_name(0),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}


def set_seed(seed: int) -> None:
    import random

    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_ckpt(path: Path, state: dict) -> None:
    """Atomic: write to a temp file, then rename, so a kill -9 never leaves a torn checkpoint."""
    import torch
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "state.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path / "state.pt")


def load_ckpt(path: Path):
    import torch
    f = path / "state.pt"
    return torch.load(f, map_location="cpu", weights_only=False) if f.exists() else None


def batch_plan(n_items: int, batch: int, steps: int, seed: int) -> np.ndarray:
    """(steps, batch) item indices fixed up front from the seed, so resume replays the same order."""
    rng = np.random.default_rng([seed, 7])
    need = steps * batch
    order = np.concatenate([rng.permutation(n_items) for _ in range(math.ceil(need / n_items))])
    return order[:need].reshape(steps, batch)
