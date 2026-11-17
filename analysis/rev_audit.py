"""Extended label-alignment audit (Phase 3b; review point 7).

Three 100-item blind audit sets (same definitions as docs/audit/INSTRUCTIONS.md):
  1. t2_beavertails_pool : 100 stratified items from T2 BeaverTails training pool
  2. t1_data2txt_shift   : 100 stratified items from T1 RAGTruth Data2txt shifted set
  3. t3_heldout_groups   : 100 stratified items from T3 ToxiGen held-out target groups shifted set

Usage:
  python analysis/rev_audit.py --prepare   # write blind CSVs + INSTRUCTIONS.md into docs/audit/extended/
  python analysis/rev_audit.py --score     # score completed sheets against source gold + inter-annotator kappa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.splits import stratified_order  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())
PROCESSED = ROOT / "data" / "processed"
EXT_AUDIT = ROOT / "docs" / "audit" / "extended"
REV_OUT = ROOT / "results" / "revision"
THRESHOLD = 0.6
SAMPLE_SIZE = 100
AUDIT_SEED = 20261116

SPECS = {
    "t2_beavertails_pool": {
        "task": "t2_response_safety",
        "split": "pool",
        "strata": ["label"],
        "columns": {"text_a": "user_prompt", "text_b": "assistant_response"},
        "drop_cols": [],
        "description": "BeaverTails 30k training pool (T2 in-distribution source)",
    },
    "t1_data2txt_shift": {
        "task": "t1_rag_hallucination",
        "split": "shift",
        "strata": ["label"],
        "columns": {"text_a": "document", "text_b": "response"},
        "drop_cols": [],
        "description": "RAGTruth Data2txt shifted set (T1 held-out task type)",
    },
    "t3_heldout_groups": {
        "task": "t3_implicit_toxicity",
        "split": "shift",
        "strata": ["label", "group"],
        "columns": {"text_a": "post", "text_b": "unused"},
        "drop_cols": ["unused"],
        "description": "ToxiGen held-out target groups (T3 shifted set: native_american, latino, mexican)",
    },
}

INSTRUCTIONS_TEXT = """# Extended label-alignment audit — annotator instructions

**Purpose.** Pre-submission review (point 7) asked to audit the **source** labels of BeaverTails
(T2 training pool) and the two within-dataset shifted sets (**T1 Data2txt** and **T3 held-out
target groups**). Label 100 items from each set using the exact same training definitions as the
initial audit (`../INSTRUCTIONS.md`). We then compute Cohen's κ against the source gold labels and
between annotators.

**Content warning.** T2 and T3 contain hateful, violent, and otherwise offensive text.

## Your files

Label only your own files (Divya: `*_audit_divya.csv`; Suresh: `*_audit_suresh.csv`):

| Order | File | Items | Source |
|---|---|---|---|
| 1 | `t3_heldout_groups_audit_<you>.csv` | 100 short posts | ToxiGen held-out groups (shift) |
| 2 | `t2_beavertails_pool_audit_<you>.csv` | 100 prompt + response pairs | BeaverTails pool |
| 3 | `t1_data2txt_shift_audit_<you>.csv` | 100 JSON business record + overview pairs | RAGTruth Data2txt (shift) |

## Rules

1. **Work independently.** Don't look at the other person's file and don't discuss any item until both of you have finished all three files.
2. **No AI assistance and no looking up the source dataset.**
3. **Fill the `label` column with `1` or `0` for every row.** If you genuinely can't decide, pick the more likely answer and write `unsure` in `notes`. Don't leave blanks.
4. **Don't edit any other column**, reorder rows, or rename the file.
5. Open in Google Sheets or Excel, then save/export as **CSV** with the exact same file name in this folder.

## Definitions (same as `../INSTRUCTIONS.md`)

### T3 — implicit toxicity (ToxiGen human ratings)
- **1** = the post is offensive, demeaning, or harmful toward a group of people, **including subtle forms**: stereotypes, coded language or dog-whistles, "grievance" framing, irony or sarcasm aimed at a group, implicit threats or incitement. No slur is needed.
- **0** = neutral, factual, positive, or unrelated to any group, even if it mentions a group or a political topic.

### T2 — response safety (BeaverTails `is_safe`)
Judge the **assistant response**, given the prompt. The prompt being harmful does not by itself make the response harmful.
- **1** = the response is harmful: it helps with, encourages, or contains content that could cause harm (violence, weapons, crime/fraud, drugs, self-harm, hate/harassment, privacy violations, sexual content involving minors or non-consent, dangerous misinformation).
- **0** = the response is harmless: it refuses, redirects, warns, gives safe general information, or the exchange was benign to begin with.

### T1 — RAG hallucination (RAGTruth Data2txt)
Read the `response` and check every factual claim against the `document` (the structured JSON record and customer reviews in `document`).
- **1** = the response contains anything **not supported by the document**: it contradicts the document, or adds a fact, number, attribute, or review detail the document does not state, **even if plausible**.
- **0** = everything the response says is stated in or directly implied by the document.

## When you're done

Run `.venv/bin/python analysis/rev_audit.py --score` (or tell the agent). It reports κ per person
and between annotators with 95% bootstrap CIs and saves `results/revision/extended_audit_kappa.csv`.
"""


def kappa_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    idx = (rng.integers(0, len(a), len(a)) for _ in range(n_boot))
    vals = [cohen_kappa_score(a[i], b[i]) for i in idx]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(cohen_kappa_score(a, b)), (float(lo), float(hi))


def prepare() -> None:
    EXT_AUDIT.mkdir(parents=True, exist_ok=True)
    (EXT_AUDIT / "INSTRUCTIONS.md").write_text(INSTRUCTIONS_TEXT)
    rng = np.random.default_rng([CFG["split_seed"], AUDIT_SEED])
    manifest = {}
    for name, spec in SPECS.items():
        df = pd.read_parquet(PROCESSED / f"{spec['task']}.parquet")
        sub = df[df.split == spec["split"]].reset_index(drop=True)
        order = stratified_order(sub, spec["strata"], rng)[:SAMPLE_SIZE]
        picked = sub.iloc[order][["uid", "text_a", "text_b"]].copy()
        picked = picked.sample(frac=1, random_state=CFG["split_seed"]).reset_index(drop=True)
        picked.insert(0, "item", range(1, len(picked) + 1))
        picked["label"], picked["notes"] = "", ""
        picked = picked.rename(columns=spec["columns"]).drop(columns=spec["drop_cols"], errors="ignore")
        manifest[name] = {
            "task": spec["task"],
            "split": spec["split"],
            "n": len(picked),
            "uids": picked["uid"].tolist(),
        }
        for who in CFG["annotators"]:
            dest = EXT_AUDIT / f"{name}_audit_{who}.csv"
            if dest.exists():
                print(f"[keep] {dest.relative_to(ROOT)} exists; not overwritten")
                continue
            picked.to_csv(dest, index=False)
            print(f"[write] {dest.relative_to(ROOT)} ({len(picked)} items)")
    (EXT_AUDIT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[write] {(EXT_AUDIT / 'INSTRUCTIONS.md').relative_to(ROOT)}")
    print(f"[write] {(EXT_AUDIT / 'manifest.json').relative_to(ROOT)}")


def _load_annotator_labels(name: str) -> tuple[pd.Index, dict[str, np.ndarray], dict[str, int]] | None:
    final_f = EXT_AUDIT / f"{name}_audit_final.csv"
    if final_f.exists():
        df = pd.read_csv(final_f, dtype=str, keep_default_na=False)
        labels, unsure = {}, {}
        for who in CFG["annotators"]:
            v = df[f"{who}_label"].str.strip()
            bad = v[~v.isin(["0", "1"])]
            if len(bad):
                raise ValueError(f"{final_f.name}: {who}_label must be 0 or 1; bad items {df.item[bad.index].tolist()[:5]}")
            labels[who] = v.astype(int).to_numpy()
            notes_col = f"{who}_notes" if f"{who}_notes" in df.columns else "notes"
            unsure[who] = int(df[notes_col].str.lower().str.contains("unsure").sum()) if notes_col in df.columns else 0
        return df["uid"], labels, unsure

    annot_dfs = {}
    for who in CFG["annotators"]:
        f = EXT_AUDIT / f"{name}_audit_{who}.csv"
        if not f.exists():
            print(f"  missing {f.relative_to(ROOT)}")
            return None
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        v = df["label"].str.strip()
        if (v == "").any():
            print(f"  pending {f.relative_to(ROOT)} ({int((v == '').sum())} blank labels)")
            return None
        bad = v[~v.isin(["0", "1"])]
        if len(bad):
            raise ValueError(f"{f.name}: label must be 0 or 1 (no blanks); bad items {df.item[bad.index].tolist()[:5]}")
        annot_dfs[who] = df

    base_uids = annot_dfs[CFG["annotators"][0]]["uid"]
    labels, unsure = {}, {}
    for who, df in annot_dfs.items():
        if not df["uid"].equals(base_uids):
            df = df.set_index("uid").loc[base_uids].reset_index()
        labels[who] = df["label"].str.strip().astype(int).to_numpy()
        unsure[who] = int(df["notes"].str.lower().str.contains("unsure").sum()) if "notes" in df.columns else 0
    return base_uids, labels, unsure


def _score_complete_annotators_only(name: str, spec: dict) -> dict | None:
    """Score annotators whose sheet is complete when the other is still pending (no between-annotator kappa)."""
    entry = None
    for who in CFG["annotators"]:
        f = EXT_AUDIT / f"{name}_audit_{who}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        v = df["label"].str.strip()
        if not v.isin(["0", "1"]).all():
            continue
        g = pd.read_parquet(PROCESSED / f"{spec['task']}.parquet", columns=["uid", "label"]).set_index("uid").loc[df["uid"], "label"].astype(int).to_numpy()
        lab = v.astype(int).to_numpy()
        k, (lo, hi) = kappa_ci(lab, g)
        entry = entry or {"n": len(g), "gold_pos_rate": float(g.mean()), "annotators": {}, "partial": True}
        entry["annotators"][who] = dict(kappa=round(k, 4), lo=round(lo, 4), hi=round(hi, 4),
                                        agreement=round(float(np.mean(lab == g)), 4), pos_rate=round(float(lab.mean()), 4), unsure=0)
        print(f"  [partial] {who} kappa vs gold {k:.3f} [{lo:.3f}, {hi:.3f}] (other annotator pending)")
    return entry


def score() -> int:
    rows = []
    summary = {}
    for name, spec in SPECS.items():
        print(f"\n{name} — {spec['description']}")
        loaded = _load_annotator_labels(name)
        if loaded is None:
            partial = _score_complete_annotators_only(name, spec)
            if partial:
                summary[name] = partial
            continue
        uids, labels, unsure = loaded
        gold_df = pd.read_parquet(PROCESSED / f"{spec['task']}.parquet", columns=["uid", "label"]).set_index("uid")
        g = gold_df.loc[uids, "label"].astype(int).to_numpy()
        passed = True
        entry = {"n": len(g), "gold_pos_rate": float(g.mean()), "annotators": {}, "between": {}}
        for who, lab in labels.items():
            k, (lo, hi) = kappa_ci(lab, g)
            agree = float(np.mean(lab == g))
            passed &= k >= THRESHOLD
            print(f"  {who:8s} kappa vs gold {k:.3f} [{lo:.3f}, {hi:.3f}]  agreement {agree:.1%}  "
                  f"pos {lab.mean():.1%} (gold {g.mean():.1%})  unsure {unsure[who]}  "
                  f"{'PASS' if k >= THRESHOLD else 'BELOW 0.6'}")
            rows.append(dict(audit=name, task=spec["task"], split=spec["split"], pair=f"{who}_vs_gold",
                             n=len(g), kappa=round(k, 4), lo=round(lo, 4), hi=round(hi, 4),
                             agreement=round(agree, 4), pos_a=round(float(lab.mean()), 4),
                             pos_b=round(float(g.mean()), 4), unsure=unsure[who],
                             passes_06=bool(k >= THRESHOLD)))
            entry["annotators"][who] = dict(kappa=round(k, 4), lo=round(lo, 4), hi=round(hi, 4),
                                            agreement=round(agree, 4), pos_rate=round(float(lab.mean()), 4),
                                            unsure=unsure[who])
        w0, w1 = CFG["annotators"][:2]
        k_btw, (lo_b, hi_b) = kappa_ci(labels[w0], labels[w1])
        agree_b = float(np.mean(labels[w0] == labels[w1]))
        print(f"  between annotators kappa {k_btw:.3f} [{lo_b:.3f}, {hi_b:.3f}]  agreement {agree_b:.1%}  on {len(g)} items")
        print(f"  => {'PASS (>= 0.6)' if passed else 'BELOW 0.6'}")
        rows.append(dict(audit=name, task=spec["task"], split=spec["split"], pair="between_annotators",
                         n=len(g), kappa=round(k_btw, 4), lo=round(lo_b, 4), hi=round(hi_b, 4),
                         agreement=round(agree_b, 4), pos_a=round(float(labels[w0].mean()), 4),
                         pos_b=round(float(labels[w1].mean()), 4), unsure=unsure[w0] + unsure[w1],
                         passes_06=bool(passed)))
        entry["between"] = dict(kappa=round(k_btw, 4), lo=round(lo_b, 4), hi=round(hi_b, 4),
                                agreement=round(agree_b, 4))
        entry["passes_06"] = bool(passed)
        summary[name] = entry

    if rows:
        REV_OUT.mkdir(parents=True, exist_ok=True)
        out_csv = REV_OUT / "extended_audit_kappa.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n[write] {out_csv.relative_to(ROOT) if out_csv.is_relative_to(ROOT) else out_csv}")
        num_path = REV_OUT / "revision_numbers.json"
        nums = json.loads(num_path.read_text()) if num_path.exists() else {}
        nums["extended_audit"] = summary
        num_path.write_text(json.dumps(nums, indent=2))
        print(f"[write] {num_path.relative_to(ROOT) if num_path.is_relative_to(ROOT) else num_path} (extended_audit)")
    else:
        print("\nNo completed extended audit sheets found yet; fill labels in docs/audit/extended/ and re-run --score.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if not (args.prepare or args.score):
        ap.error("specify --prepare and/or --score")
    if args.prepare:
        prepare()
    if args.score:
        return score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
