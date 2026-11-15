"""Write data/MANIFEST.md from configs/tasks.yaml, data/raw/downloads.json, and processed logs.

Usage: python data/manifest.py
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())
DL = ROOT / "data" / "raw" / "downloads.json"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"

RAW_DIR = {"ragtruth": "ragtruth", "beavertails_30k": "beavertails", "toxigen_annotated": "toxigen",
           "llm_aggrefact": "llm_aggrefact", "wildguardtest": "wildguardtest", "implicit_hate_stg1": "implicit_hate"}


def main() -> None:
    downloads = json.loads(DL.read_text()) if DL.exists() else {}
    lines = [f"# Data manifest", "", f"Generated {dt.date.today().isoformat()} by `data/manifest.py`. "
             "Do not edit by hand.", "", "## Sources", "",
             "| Source | License | Access | Files (sha256, first 12) | Accessed |", "|---|---|---|---|---|"]
    for name, s in CFG["sources"].items():
        files = downloads.get(RAW_DIR[name], {})
        access = s.get("access") or ("gated (accept terms on HF; HF_TOKEN)" if s.get("gated") else "open")
        if files:
            f_txt = "<br>".join(f"`{Path(p).name}` {v['sha256'][:12]} ({v['bytes']:,} B)" for p, v in files.items())
            when = max(v["accessed"] for v in files.values())
        else:
            f_txt, when = "**not yet downloaded**", "—"
        lines.append(f"| [{name}]({s['url']}) | {s['license']} | {access} | {f_txt} | {when} |")

    lines += ["", "## Processed tasks", ""]
    for task in CFG["tasks"]:
        p = PROCESSED / f"{task}.parquet"
        if not p.exists():
            lines += [f"### {task}", "", "_not processed_", ""]
            continue
        df = pd.read_parquet(p)
        log = json.loads((PROCESSED / f"{task}.log.json").read_text())
        sub = json.loads((SPLITS / task / "subsets.json").read_text()) if (SPLITS / task / "subsets.json").exists() else {}
        lines += [f"### {task}", "", f"Positive class: {CFG['tasks'][task]['positive']}.", "",
                  "| Split | Items | Positive rate |", "|---|---|---|"]
        for split in ("pool", "val", "test", "shift"):
            d = df[df.split == split]
            lines.append(f"| {split} | {len(d):,} | {d.label.mean():.3f} |" if len(d) else f"| {split} | pending | — |")
        if sub:
            lines += ["", f"N levels: {sub['n_levels']}; frontier sub-pool: {sub['subpool_size']:,}; "
                          f"detailed-prompt test subsample: {len(sub['test_api_detailed']):,}."]
        groups = df[df.split == "pool"].groupby("group").label.agg(["size", "mean"])
        if len(groups) > 1:
            lines += ["", "Pool by stratification group: " + "; ".join(
                f"{g} {int(r['size']):,} ({r['mean']:.2f} pos)" for g, r in groups.iterrows()) + "."]
        cleaning = {k: v for k, v in log.items() if k not in ("sizes", "prevalence")}
        if cleaning:
            lines += ["", "Cleaning: " + "; ".join(f"{k} = {v}" for k, v in cleaning.items()) + "."]
        lines.append("")
    (ROOT / "data" / "MANIFEST.md").write_text("\n".join(lines))
    print("wrote data/MANIFEST.md")


if __name__ == "__main__":
    main()
