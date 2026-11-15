"""Appendix G: compute and carbon from run.json (train_seconds + eval_seconds) over every finished run in preds/runs.

GPU-hours count runs whose run.json names a GPU; CPU runs (TF-IDF, on the Mac, D30) are counted separately and
add no GPU energy. Runs are split into the main sweep, ablations (tag abl-*), hyperparameter selection (hp-*)
and setup probes (eval limit, step cap or other tags).
ASSUMPTIONS, stated with the table: energy = GPU-hours x the board power of one RTX A6000 (--watts, default
300 W, its TDP, drawn for the whole run); host CPU, memory, cooling (PUE), idle boxes, the open-teacher pass,
serving benchmarks and hosted-API calls are excluded. Carbon = energy x grid intensity (--grid, default
0.4 kg CO2e / kWh, a placeholder, not a measured value for the provider's region). Cost = GPU-hours x the
measured USD 0.366 per A6000-hour (D27); run time only, so below the invoiced box-hours.

  python analysis/compute.py [--watts 300] [--grid 0.4]     # -> results/tables/tableG_compute.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS, OUT = ROOT / "preds" / "runs", ROOT / "results"
USD_PER_GPU_HOUR = 0.366
KINDS = ("main", "ablation", "hp_selection", "probe")


def kind(r: dict, tag: str) -> str:
    if r.get("eval_limit") or r.get("max_steps"):
        return "probe"
    if not tag:
        return "main"
    return "ablation" if tag.startswith("abl-") else "hp_selection" if tag.startswith("hp-") else "probe"


def runs() -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not (d / "DONE").exists() or not (d / "run.json").exists():
            continue
        r = json.loads((d / "run.json").read_text())
        base = f"{r['task']}__{r['model']}__{r['labels']}__n{r['n']}__s{r['seed']}"
        tag = d.name[len(base) + 2:]
        gpu = str(r.get("gpu") or "cpu")
        rows.append(dict(task=r["task"], family=r["family"], model=r["model"], kind=kind(r, tag),
                         device="cpu" if gpu == "cpu" else gpu, on_gpu=gpu != "cpu",
                         train_h=float(r.get("train_seconds") or 0) / 3600,
                         eval_h=float(r.get("eval_seconds") or 0) / 3600, resumed=bool(r.get("resumed_from"))))
    return pd.DataFrame(rows)


def table(df: pd.DataFrame, watts: float, grid: float) -> pd.DataFrame:
    df = df.assign(h=df.train_h + df.eval_h)
    gpu = df[df.on_gpu]

    def agg(g: pd.DataFrame, h: pd.DataFrame) -> dict:
        out = dict(runs=len(g), gpu_runs=int(g.on_gpu.sum()), cpu_runs=int((~g.on_gpu).sum()),
                   gpu_train_h=h.train_h.sum(), gpu_eval_h=h.eval_h.sum(), gpu_hours=h.h.sum(),
                   cpu_hours=g[~g.on_gpu].h.sum())
        out.update({f"gpu_h_{k}": h[h.kind == k].h.sum() for k in KINDS})
        return out

    rows = []
    for (fam, task), g in df.groupby(["family", "task"]):
        rows.append(dict(family=fam, task=task, **agg(g, gpu[(gpu.family == fam) & (gpu.task == task)])))
    for fam, g in df.groupby("family"):
        rows.append(dict(family=fam, task="all", **agg(g, gpu[gpu.family == fam])))
    for task, g in df.groupby("task"):
        rows.append(dict(family="all", task=task, **agg(g, gpu[gpu.task == task])))
    rows.append(dict(family="all", task="all", **agg(df, gpu)))
    t = pd.DataFrame(rows)
    t["energy_kwh"] = t.gpu_hours * watts / 1000
    t["co2e_kg"] = t.energy_kwh * grid
    t["usd_run_time"] = t.gpu_hours * USD_PER_GPU_HOUR
    t["assumed_gpu_watts"], t["assumed_grid_kg_co2e_per_kwh"] = watts, grid
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watts", type=float, default=300.0, help="assumed board power of one A6000 (W)")
    ap.add_argument("--grid", type=float, default=0.4, help="assumed grid intensity (kg CO2e per kWh)")
    a = ap.parse_args()
    df = runs()
    t = table(df, a.watts, a.grid)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "tables" / "tableG_compute.csv", index=False, float_format="%.3f")
    tot = t[(t.family == "all") & (t.task == "all")].iloc[0]
    note = (f"GPU-hours = train_seconds + eval_seconds from run.json over {int(tot.runs)} finished runs "
            f"({int(tot.gpu_runs)} on GPU); ASSUMED {a.watts:g} W per RTX A6000 for the whole run time and "
            f"{a.grid:g} kg CO2e/kWh grid intensity; excludes host, cooling, idle boxes, the open-teacher pass, "
            f"serving and API calls; cost at the measured USD {USD_PER_GPU_HOUR} per A6000-hour (D27), run time only")
    f = OUT / "paper_numbers.json"
    numbers = json.loads(f.read_text()) if f.exists() else {}
    numbers["compute"] = dict(note=note, assumed_gpu_watts=a.watts, assumed_grid_kg_co2e_per_kwh=a.grid,
                              usd_per_gpu_hour=USD_PER_GPU_HOUR, devices=sorted(df.device.unique()),
                              resumed_runs=int(df.resumed.sum()),
                              rows=json.loads(t.to_json(orient="records", double_precision=3)))
    f.write_text(json.dumps(numbers, indent=1, sort_keys=True))
    cols = ["family", "task", "runs", "gpu_runs", "cpu_runs", "gpu_hours", "gpu_h_main", "gpu_h_ablation", "energy_kwh",
            "co2e_kg", "usd_run_time"]
    print(t[cols].round(2).to_string(index=False))
    print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
