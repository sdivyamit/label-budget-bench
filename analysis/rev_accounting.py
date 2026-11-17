"""Run accounting, compute, collapse, and API spend across all 2,425 finished runs (Phase 4/5 revision).

Writes only into results/revision/ (leaving frozen results/ untouched):
  * results/revision/compute_revision.csv
  * results/revision/collapse_revision.csv
  * results/revision/collapse_summary_revision.csv
  * section "accounting" in results/revision/revision_numbers.json

  python analysis/rev_accounting.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import rev_common as rc

RUNS = rc.ROOT / "preds" / "runs"
USD_PER_GPU_HOUR = 0.366
WATTS = 300.0
GRID_CO2 = 0.4
SD_MIN = 0.01
KINDS = ("main", "ablation", "hp_selection", "zeroshot", "detector", "probe")


def run_kind(r: dict, tag: str) -> str:
    if r.get("eval_limit") or r.get("max_steps"):
        return "probe"
    if r["family"] == "detector":
        return "detector"
    if not tag:
        return "main"
    if tag.startswith("zeroshot"):
        return "zeroshot"
    if tag.startswith("abl-"):
        return "ablation"
    if tag.startswith("hp-"):
        return "hp_selection"
    return "probe"


def scan_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    comp_rows, col_rows = [], []
    for d in sorted(RUNS.iterdir()):
        if not (d / "DONE").exists() or not (d / "run.json").exists():
            continue
        r = json.loads((d / "run.json").read_text())
        base = f"{r['task']}__{r['model']}__{r['labels']}__n{r['n']}__s{r['seed']}"
        tag = d.name[len(base) + 2:] if len(d.name) > len(base) else ""
        k = run_kind(r, tag)
        gpu = str(r.get("gpu") or "cpu")
        on_gpu = gpu != "cpu"
        train_h = float(r.get("train_seconds") or 0) / 3600.0
        eval_h = float(r.get("eval_seconds") or 0) / 3600.0
        comp_rows.append(dict(
            run_id=d.name, task=r["task"], family=r["family"], model=r["model"],
            labels=r["labels"], n=int(r["n"]), seed=int(r["seed"]), tag=tag, kind=k,
            on_gpu=on_gpu, train_h=train_h, eval_h=eval_h, h=train_h + eval_h,
        ))
        if k in ("main", "ablation") and r["family"] in ("encoder", "decoder") and (d / "preds.parquet").exists():
            p = pd.read_parquet(d / "preds.parquet", columns=["split", "p"])
            sd = float(p[p.split == "test"].p.std())
            col_rows.append(dict(
                run_id=d.name, task=r["task"], family=r["family"], model=r["model"],
                labels=r["labels"], n=int(r["n"]), seed=int(r["seed"]), tag=tag,
                test_score_sd=sd, constant=bool(sd < SD_MIN),
            ))
    return pd.DataFrame(comp_rows), pd.DataFrame(col_rows)


def compute_table(df: pd.DataFrame) -> pd.DataFrame:
    gpu = df[df.on_gpu]

    def agg(g: pd.DataFrame, h: pd.DataFrame) -> dict:
        out = dict(
            runs=len(g),
            gpu_runs=int(g.on_gpu.sum()),
            cpu_runs=int((~g.on_gpu).sum()),
            gpu_train_h=float(h.train_h.sum()),
            gpu_eval_h=float(h.eval_h.sum()),
            gpu_hours=float(h.h.sum()),
            cpu_hours=float(g[~g.on_gpu].h.sum()),
        )
        out.update({f"gpu_h_{k}": float(h[h.kind == k].h.sum()) for k in KINDS})
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
    t["energy_kwh"] = t.gpu_hours * WATTS / 1000.0
    t["co2e_kg"] = t.energy_kwh * GRID_CO2
    t["usd_run_time"] = t.gpu_hours * USD_PER_GPU_HOUR
    return t


def main() -> int:
    rc.OUT.mkdir(parents=True, exist_ok=True)
    comp_df, col_df = scan_runs()
    ct = compute_table(comp_df)
    ct.to_csv(rc.OUT / "compute_revision.csv", index=False, float_format="%.3f")

    col_df[col_df.constant].to_csv(rc.OUT / "collapse_revision.csv", index=False, float_format="%.5f")
    main_sweep = col_df[col_df.tag == ""]
    orig_sum = pd.read_csv(rc.ROOT / "results" / "tables" / "tableB_collapse_summary.csv")
    summ = []
    for (task, model), g in main_sweep.groupby(["task", "model"]):
        o = orig_sum[(orig_sum.task == task) & (orig_sum.model == model)]
        summ.append(dict(
            task=task, model=model, runs=len(g), constant=int(g.constant.sum()),
            gold_top_pr_auc=float(o.gold_top_pr_auc.iloc[0]) if len(o) and pd.notna(o.gold_top_pr_auc.iloc[0]) else None,
            gold_top_pr_auc_without=float(o.gold_top_pr_auc_without.iloc[0]) if len(o) and pd.notna(o.gold_top_pr_auc_without.iloc[0]) else None,
            gold_top_k_without=int(o.gold_top_k_without.iloc[0]) if len(o) and pd.notna(o.gold_top_k_without.iloc[0]) else None,
        ))
    cs = pd.DataFrame(summ)
    cs.to_csv(rc.OUT / "collapse_summary_revision.csv", index=False, float_format="%.4f")

    rep_path = rc.OUT / "ref_repeat_sd.csv"
    rep_openai, rep_gemini = 0.0, 0.0
    if rep_path.exists():
        rep_df = pd.read_csv(rep_path)
        rep_openai = float(rep_df[rep_df.model == "gpt61_sol"].repeat_cost_usd.sum())
        rep_gemini = float(rep_df[rep_df.model == "gemini4_argon"].repeat_cost_usd.sum())

    trained_mask = comp_df.kind.isin(["main", "ablation", "hp_selection"])
    nums = {
        "runs/total_finished": int(len(comp_df)),
        "runs/trained": int(trained_mask.sum()),
        "runs/main": int((comp_df.kind == "main").sum()),
        "runs/ablation": int((comp_df.kind == "ablation").sum()),
        "runs/hp": int((comp_df.kind == "hp_selection").sum()),
        "runs/zeroshot": int((comp_df.kind == "zeroshot").sum()),
        "runs/detector": int((comp_df.kind == "detector").sum()),
        "runs/pipeline_test": int((comp_df.kind == "probe").sum()),
        "collapse/main_runs": int(len(main_sweep)),
        "collapse/main_constant": int(main_sweep.constant.sum()),
        "collapse/ablation_constant": int(col_df[col_df.tag != ""].constant.sum()),
        "spend/openai_orig": 18.72,
        "spend/gemini_orig": 20.59,
        "spend/jev": 0.85,
        "spend/openai_repeats": round(rep_openai, 2),
        "spend/gemini_repeats": round(rep_gemini, 2),
        "spend/openai_total": round(18.72 + rep_openai, 2),
        "spend/gemini_total": round(20.59 + rep_gemini, 2),
    }
    rc.save_numbers("accounting", nums)
    print(f"accounting: {nums['runs/total_finished']} total finished ({nums['runs/trained']} trained, "
          f"{nums['runs/zeroshot']} zero-shot, {nums['runs/detector']} detectors, {nums['runs/pipeline_test']} probes)")
    tot = ct[(ct.family == "all") & (ct.task == "all")].iloc[0]
    print(f"compute: {tot.gpu_hours:.1f} GPU-h, USD {tot.usd_run_time:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
