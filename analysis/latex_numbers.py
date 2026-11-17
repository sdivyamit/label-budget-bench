"""results/paper_numbers.json + results/tables/*.csv + results/revision/* -> paper/latex/numbers.tex.

Every numeric value in the manuscript and supplement is pulled through \\pn, \\pnci, \\pnsd, \\pcmp, or \\pv;
nothing in results/ outside paper/latex/numbers.tex is written, preserving results/FREEZE-2026-11-15.sha256.

  \\pn{t2/test/encoder/modernbert_large/gold/n20000/pr_auc}     0.947
  \\pnci{...}                                                   0.947 (95% CI 0.939--0.955)
  \\pnsd{...}                                                   seed SD
  \\pcmp{H1a/diff}, \\pcmp{H1a/lo}, \\pcmp{H1a/p_holm}            Appendix H two-level bootstrap comparisons
  \\pcmp{cl/H1a/...}, \\pcmp{it/H1a/...}                         two-level cluster / item-only bootstrap
  \\pv{be/t2/test/qwen35_2b/n_star_mean}                        deployable val-thr break-even ("0", "100", "n.r.")
  \\pv{revbe/<t>/<es>/<m>/<metric>/<ref>/<zs>/<col>}            full break-even grid (valthr / f1_at_05 / max_f1)
  \\pv{zs/<t>/<es>/<cond>/<m>/<metric>/<col>}                   N=0 zero-shot small decoders
  \\pv{noinstr/<t>/<es>/<n>/<metric>/<col>}                     Qwen3.5-2B no-guideline prompt ablation
  \\pv{lrchk/<m>/<setting>/<es>/<metric>/<col>}                 T1 encoder learning-rate check
  \\pv{cal/<t>/<es>/<m>/<labels>/<n>/<calib>/<metric>}          6-metric calibration (ECE-10, ECE-15em, Brier, BBS, BS-C/E)
  \\pv{driftfpr/<t>/<m>/<es>/<method>/<col>}                    conformal / raw / temp / iso FPR drift
  \\pv{exch/<t>/<es>/<col>}, \\pv{recal/<t>/<m>/<k>/<col>}       exchangeability bands & target-negative recalibration
  \\pv{priorshift/<t>/<es>/<m>/<method>/<col>}                  Saerens EM prior-shift precision adjustment
  \\pv{refsd/<t>/<es>/<m>/<col>}                                repeat-API reference noise
  \\pv{t2ov/...}                                                PKU-SafeRLHF prompt-overlap checks

  python analysis/latex_numbers.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "results" / "paper_numbers.json"
REV = ROOT / "results" / "revision"
REV_JSON = REV / "revision_numbers.json"
DST = ROOT / "paper" / "latex" / "numbers.tex"
TAB = ROOT / "results" / "tables"
RUNS = ROOT / "preds" / "runs"

TASK = re.compile(r"\b(t[123])_[a-z_]+?(?=/|$)")
API_SET = {"test": "test", "shift_api_concise": "shift", "test_api_detailed": "test_detailed",
           "shift_api_detailed": "shift_detailed", "probe": "probe", "subpool": "subpool"}
DECODERS = {"gemma4_e2b", "qwen35_2b", "gemma4_e4b", "qwen35_4b"}
FAMILY = {"tfidf_lr": "classical", "modernbert_base": "encoder", "modernbert_large": "encoder",
          "deberta_v3_large": "encoder", "gemma4_e2b": "decoder", "qwen35_2b": "decoder",
          "gemma4_e4b": "decoder", "qwen35_4b": "decoder"}

TABLES = {
    "be_orig": ("table4_breakeven.csv", ["task", "eval_set", "model"]),
    "tq": ("table5a_teacher_quality.csv", ["task", "teacher", "scope"]),
    "skew": ("table5d_pr_skew.csv", ["task", "model", "eval_set", "n", "labels"]),
    "agree": ("fig4b_seed_agreement.csv", ["task", "model", "labels", "n"]),
    "drift": ("table7_threshold_drift.csv", ["task", "family", "labels", "eval_set", "calib"]),
    "serve": ("table8_serving.csv", ["model", "task"]),
    "trunc": ("tableC_truncation.csv", ["task", "model", "split"]),
    "abl": ("tableD_ablations.csv", ["tag", "task", "model", "labels", "n", "eval_set", "metric"]),
}
BUDGET_COLS = ("n_star", "n_max")
SIGNED = ("d_", "gap", "diff", "tminusg", "family_diff", "two_lo", "two_hi", "item_lo", "item_hi", "cl_lo", "cl_hi")
INT_COLS = ("n", "runs", "gpu_runs", "cpu_runs", "n_items", "n_truncated", "n_unlabelled", "n_refused",
            "refused", "unparseable", "k", "k_new", "p50_tokens", "p95_tokens", "max_tokens", "max_len",
            "constant", "gold_top_k_without", "n_val_neg", "n_test_neg", "models", "models_above",
            "seeds", "seeds_above", "n_collapsed", "collapsed", "items", "seen", "m", "n_sub", "n_runs")
TOP = {"t1_rag_hallucination": 8651, "t2_response_safety": 20000, "t3_implicit_toxicity": 5920}


def short(key: str) -> str:
    return TASK.sub(r"\1", key)


def half_up3(v) -> str:
    """Three decimals, half rounded up (the source value is already rounded to four places)."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(str(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def f3(v) -> str:
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.3f}"


def pval(p: float, b: int, *, two_sided: bool = True, holm_m: int | None = None) -> str:
    """Bootstrap p-value with the replicate count taken into account.

    Zero exceedances among b replicates only bounds a two-sided p by 2/b (one-sided / probabilities: 1/b); a Holm
    adjustment over holm_m comparisons multiplies that bound by holm_m (12 x 2/b = 0.0024 for b = 10,000).
    """
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if float(p) == 0:
        bound = (2 if two_sided else 1) * (holm_m or 1) / b
        return f"$<${bound:.4f}"
    if float(p) < 0.001:
        return f"{float(p):.4f}"
    return f"{float(p):.3f}"


N_HOLM = 12  # primary comparisons (Appendix A)
B_BOOT = 10_000  # replicates of the primary two-level / cluster bootstraps (rev_compare.py --b default)


def tex_escape(s: str) -> str:
    return re.sub(r"([%&_#$])", r"\\\1", s)


def fmt_budget(v) -> str:
    if v is None:
        return "n.r."
    if isinstance(v, str):
        if v.strip().lower() in ("not reached", "n.r.", "nan", "none", "inf", ""):
            return "n.r."
        v = float(v)
    if not math.isfinite(float(v)):
        return "n.r."
    return f"{int(round(float(v))):,}".replace(",", "{,}")


def fmt(col: str, v) -> str:
    if isinstance(v, bool) or (isinstance(v, str) and v in ("True", "False")):
        return str(v)
    if col.startswith(BUDGET_COLS):
        return fmt_budget(v)
    if isinstance(v, str):
        return tex_escape(v)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if col.startswith(SIGNED):
        x = float(v)
        return f"{0.0 if abs(x) < 0.0005 else x:+.3f}"
    if col in INT_COLS:
        return f"{int(round(float(v))):,}".replace(",", "{,}")
    if col == "items_per_s":
        return f"{int(round(float(v))):,}".replace(",", "{,}")
    if col == "usd_per_million_items":
        return f"{float(v):,.0f}".replace(",", "{,}") if float(v) >= 16 else f"{float(v):.2f}"
    if col.endswith("_ms"):
        return f"{int(round(float(v)))}"
    if col in ("gpu_hours", "gpu_train_h", "gpu_eval_h", "gpu_h_main", "gpu_h_ablation",
               "gpu_h_hp_selection", "gpu_h_zeroshot", "gpu_h_detector", "gpu_h_probe",
               "energy_kwh", "co2e_kg"):
        return f"{float(v):,.1f}".replace(",", "{,}")
    if col in ("usd_run_time", "cpu_hours") or col.startswith("spend_") or col.endswith("_usd"):
        return f"{float(v):,.2f}".replace(",", "{,}")
    if col == "sd_ref":
        x = float(v)
        return f"{x:.4f}" if x < 0.001 else f"{x:.3f}"
    return f"{float(v):.3f}"


def col_key(c: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_")


def _pilot_5seed_overrides() -> dict[tuple[str, str], dict[str, float]]:
    """Compute true 5-seed means & SDs for T2 gold N=1000 modernbert_base & qwen35_2b across test/shift/shift_api."""
    from eval import stability
    overrides: dict[tuple[str, str], dict[str, float]] = {}
    pc_path = REV / "pilot_cells.csv"
    if pc_path.exists():
        pc = pd.read_csv(pc_path)
        for r in pc.itertuples():
            fam = FAMILY[r.model]
            overrides[("m", f"t2/{r.eval_set}/{fam}/{r.model}/gold/n1000/{r.metric}")] = {"mean": float(r.mean_5)}

    # Also read metrics.json across all 5 seeds on test & shift for pr_auc, max_f1, f1_at_05, ece, brier, bbs, bs_error
    for mod in ("modernbert_base", "qwen35_2b"):
        fam = FAMILY[mod]
        seed_jsons = []
        for s in range(5):
            mj_p = RUNS / f"t2_response_safety__{mod}__gold__n1000__s{s}" / "metrics.json"
            if mj_p.exists():
                seed_jsons.append(json.loads(mj_p.read_text()))
        if len(seed_jsons) == 5:
            for es in ("test", "shift"):
                for cal_prefix, cal_key in (("", "raw"), ("temperature/", "temperature")):
                    keys_set = set()
                    for mj in seed_jsons:
                        keys_set.update(mj.get(es, {}).get(cal_key, {}).keys())
                    for met in keys_set:
                        vals = [float(mj[es][cal_key][met]) for mj in seed_jsons
                                if isinstance(mj.get(es, {}).get(cal_key, {}).get(met), (int, float))]
                        if len(vals) == 5:
                            arr = np.asarray(vals, dtype=float)
                            kname = f"t2/{es}/{fam}/{mod}/gold/n1000/{cal_prefix}{met}"
                            overrides[("m", kname)] = {
                                "mean": float(arr.mean()),
                                "sd": float(arr.std(ddof=1)),
                            }
            # 5-seed Krippendorff alpha at 0.5 cut-off (matching fig4b_seed_agreement.csv convention)
            for es in ("test", "shift"):
                mats = []
                for s in range(5):
                    pp = pd.read_parquet(RUNS / f"t2_response_safety__{mod}__gold__n1000__s{s}" / "preds.parquet")
                    g = pp[pp.split == es].sort_values("uid")
                    mats.append((g.p.to_numpy(float) >= 0.5).astype(int))
                alpha = float(stability.krippendorff_alpha_nominal(np.stack(mats, axis=0)))
                overrides[("m", f"t2/{es}/{fam}/{mod}/gold/n1000/alpha_k")] = {"mean": alpha}
    return overrides


def main() -> int:
    d = json.loads(SRC.read_text())
    rn = json.loads(REV_JSON.read_text()) if REV_JSON.exists() else {}

    # Store macros in a dict so revision overrides cleanly replace stale 4-seed/item-only definitions
    defs: dict[tuple[str, str], str] = {}

    def put(kind: str, key: str, val: str):
        val = re.sub(r"(?<![0-9A-Za-z-])-(?=[0-9.])", r"\\ensuremath{-}", str(val))
        defs[(kind, short(key))] = val

    # 1. Base cell means, CIs, SDs from results/paper_numbers.json
    for group, cells in d["cells"].items():
        for name, v in cells.items():
            key = f"{group}/{name}"
            put("m", key, f3(v["mean"]))
            if v.get("lo") is not None:
                put("ci", key, f"{f3(v['mean'])} (95\\% CI {f3(v['lo'])}--{f3(v['hi'])})")
            if v.get("sd") is not None:
                put("sd", key, f3(v["sd"]))

    # 1b. Add f1_valthr cell values & override 5-seed pilot cells
    bc_path = REV / "breakeven_curves.csv"
    if bc_path.exists():
        bc = pd.read_csv(bc_path)
        bc_b = bc[bc.ref_kind == "better"]
        for r in bc_b.itertuples():
            fam = FAMILY.get(r.model, r.family)
            k_cell = f"{r.task}/{r.eval_set}/{fam}/{r.model}/gold/n{int(r.n)}/{r.metric}"
            put("m", k_cell, f3(r.q))
            if pd.notna(r.seed_sd):
                put("sd", k_cell, f3(r.seed_sd))
            base_c = f"becurve/{r.task}/{r.eval_set}/{r.model}/{int(r.n)}/{r.metric}"
            put("v", f"{base_c}/q", f3(r.q))
            put("v", f"{base_c}/diff", fmt("diff", r.diff))
            put("v", f"{base_c}/lo", fmt("two_lo", r.diff_lo))
            put("v", f"{base_c}/hi", fmt("two_hi", r.diff_hi))
            put("v", f"{base_c}/ci", f"{fmt('two_lo', r.diff_lo)} to {fmt('two_hi', r.diff_hi)}")
            put("v", f"{base_c}/diffci", f"{fmt('diff', r.diff)} (95\\% CI {fmt('two_lo', r.diff_lo)} to {fmt('two_hi', r.diff_hi)})")
            put("v", f"{base_c}/p_ge_ref", pval(r.p_ge_ref, 2000, two_sided=False))

    for (_, kname), vals in _pilot_5seed_overrides().items():
        put("m", kname, f3(vals["mean"]))
        if "sd" in vals:
            put("sd", kname, f3(vals["sd"]))

    # 2. Pre-specified comparisons: two-level primary ("") + cluster ("cl/") + item-only ("it/")
    cmp_tl = REV / "comparisons_two_level.csv"
    if cmp_tl.exists():
        cdf = pd.read_csv(cmp_tl)
        b_res = B_BOOT
        for r in cdf.to_dict("records"):
            cid = r["id"]
            specs = [
                ("", r["two_lo"], r["two_hi"], r["two_p"], r["two_p_holm"]),
                ("cl/", r["cl_lo"], r["cl_hi"], r["cl_p"], r["cl_p_holm"]),
                ("it/", r["item_lo"], r["item_hi"], r["item_p"], r["item_p_holm"]),
            ]
            for pref, lo, hi, pv_raw, pv_holm in specs:
                pcid = pref + cid
                put("c", f"{pcid}/value_a", f3(r["value_a"]))
                put("c", f"{pcid}/value_b", f3(r["value_b"]))
                put("c", f"{pcid}/diff", fmt("diff", r["diff"]))
                put("c", f"{pcid}/lo", fmt("two_lo", lo))
                put("c", f"{pcid}/hi", fmt("two_hi", hi))
                put("c", f"{pcid}/p", pval(pv_raw, b_res))
                put("c", f"{pcid}/p_holm", pval(pv_holm, b_res, holm_m=N_HOLM))
                put("c", f"{pcid}/ci", f"{fmt('two_lo', lo)} to {fmt('two_hi', hi)}")
                put("c", f"{pcid}/diffci", f"{fmt('diff', r['diff'])} (95\\% CI {fmt('two_lo', lo)} to {fmt('two_hi', hi)})")
    else:
        for block, prefix in (("comparisons", ""), ("comparisons_cluster", "cl/")):
            cmp = d.get(block, {})
            b = cmp.get("bootstrap_resamples", 10_000)
            for cid, r in cmp.get("rows", {}).items():
                cid = prefix + cid
                for k in ("value_a", "value_b", "diff", "lo", "hi"):
                    put("c", f"{cid}/{k}", f"{r[k]:+.3f}" if k in ("diff", "lo", "hi") else f3(r[k]))
                put("c", f"{cid}/p", pval(r["p"], b))
                put("c", f"{cid}/p_holm", pval(r["p_holm"], b, holm_m=N_HOLM))
                put("c", f"{cid}/diffci", f"{r['diff']:+.3f} (95\\% CI {r['lo']:+.3f} to {r['hi']:+.3f})")
                put("c", f"{cid}/ci", f"{r['lo']:+.3f} to {r['hi']:+.3f}")

    # 3. Hosted API metrics
    api = ROOT / "results" / "api_metrics.csv"
    if api.exists():
        a = pd.read_csv(api)
        for r in a.to_dict("records"):
            base = f"api/{r['task']}/{API_SET.get(r['set'], r['set'])}/{r['kind']}/{r['model']}"
            for c, v in r.items():
                if c not in ("task", "set", "kind", "model", "scores"):
                    put("v", f"{base}/{col_key(c)}", fmt(c, v))
        sub = a[a["set"].isin(["test_api_detailed", "shift_api_detailed"])]
        piv = sub.pivot(index=["task", "set", "model"], columns="kind", values=["f1_at_05", "pr_auc"])
        for model in ("gpt61_sol", "gemini4_argon", "jev"):
            for metric in ("f1_at_05", "pr_auc"):
                if (metric, "detailed") in piv and (metric, "concise") in piv:
                    s = (piv[(metric, "detailed")] - piv[(metric, "concise")]).xs(model, level="model").dropna()
                    if len(s):
                        put("v", f"api/promptdiff/{model}/{metric}/absmax", f3(float(s.abs().max())))

    # 4. Base CSV tables (excluding break-even, plateau gap, compute, collapse which are superseded by revision CSVs)
    for name, (fname, keys) in TABLES.items():
        p = TAB / fname
        if not p.exists():
            continue
        t = pd.read_csv(p)
        for r in t.to_dict("records"):
            base = "/".join([name] + [str(int(r[k])) if isinstance(r[k], float) and r[k] == int(r[k]) else str(r[k])
                                      for k in keys])
            for c, v in r.items():
                if c not in keys:
                    put("v", f"{base}/{col_key(c)}", fmt(c, v))

    # Update agree table for 5-seed T2 pilot cells
    for (_, kname), vals in _pilot_5seed_overrides().items():
        if kname.endswith("/alpha_k"):
            parts = kname.split("/")  # t2/test/decoder/qwen35_2b/gold/n1000/alpha_k
            if parts[1] == "test":
                put("v", f"agree/t2/{parts[3]}/gold/1000/alpha_k", f3(vals["mean"]))

    # 5. Revision Break-Even (deployable f1_valthr primary + full grid + budget-honest)
    be_rev = REV / "breakeven.csv"
    if be_rev.exists():
        bdf = pd.read_csv(be_rev)
        for r in bdf.to_dict("records"):
            zs_tag = "zs" if r["with_zeroshot"] else "nozs"
            base = f"revbe/{r['task']}/{r['eval_set']}/{r['model']}/{r['metric']}/{r['ref_kind']}/{zs_tag}"
            for c in ("q_ref", "sd_ref", "floor", "n_star_mean", "n_star_var", "n_star_var_sdref0",
                      "n_star_p05", "n_star_p95", "p_reached", "q_at_nmax", "p_ge_ref_at_nmax"):
                put("v", f"{base}/{c}", fmt(c, r[c]))
            # Primary \pv{be/<task>/<es>/<model>/...} -> f1_valthr, better, zs for decoders else nozs
            is_primary_zs = bool(r["with_zeroshot"]) == (r["model"] in DECODERS)
            if r["metric"] == "f1_valthr" and r["ref_kind"] == "better" and is_primary_zs:
                pbase = f"be/{r['task']}/{r['eval_set']}/{r['model']}"
                for c in ("q_ref", "sd_ref", "floor", "n_star_mean", "n_star_var", "n_star_var_sdref0",
                          "n_star_p05", "n_star_p95", "p_reached", "q_at_nmax", "p_ge_ref_at_nmax"):
                    put("v", f"{pbase}/{c}", fmt(c, r[c]))
            if r["metric"] == "f1_valthr" and r["ref_kind"] == "weaker" and is_primary_zs:
                pbase = f"be/{r['task']}/{r['eval_set']}/{r['model']}"
                put("v", f"{pbase}/n_star_mean_weaker", fmt("n_star_mean", r["n_star_mean"]))
                put("v", f"{pbase}/n_star_var_weaker", fmt("n_star_var", r["n_star_var"]))
                put("v", f"{pbase}/q_ref_weaker", fmt("q_ref", r["q_ref"]))

    beh = REV / "breakeven_honest.csv"
    if beh.exists():
        hdf = pd.read_csv(beh)
        for r in hdf.to_dict("records"):
            base = f"behon/{r['task']}/{r['eval_set']}/{r['model']}/{int(r['n'])}/{r['against']}"
            for c in ("honest", "main", "diff", "lo", "hi"):
                put("v", f"{base}/{c}", fmt("two_lo" if c in ("lo", "hi") else c, r[c]))
        # Summary across f1_valthr cells
        sub_vt = hdf[hdf.against == "f1_valthr"]
        put("v", "behon/summary/f1_valthr/median_diff", fmt("diff", sub_vt["diff"].median()))
        put("v", "behon/summary/f1_valthr/min_diff", fmt("diff", sub_vt["diff"].min()))
        put("v", "behon/summary/f1_valthr/max_diff", fmt("diff", sub_vt["diff"].max()))
        sub_n1000 = sub_vt[sub_vt.n == 1000]
        put("v", "behon/summary/f1_valthr_n1000/absmax", f3(sub_n1000["diff"].abs().max()))

    # 6. Zero-shot small decoders (N = 0)
    zs_p = REV / "zeroshot.csv"
    if zs_p.exists():
        zdf = pd.read_csv(zs_p)
        for r in zdf.to_dict("records"):
            base = f"zs/{r['task']}/{r['eval_set']}/{r['cond']}/{r['model']}/{r['metric']}"
            put("v", f"{base}/mean", f3(r["mean"]))
            put("v", f"{base}/lo", f3(r["lo"]))
            put("v", f"{base}/hi", f3(r["hi"]))
            put("v", f"{base}/ci", f"{f3(r['mean'])} (95\\% CI {f3(r['lo'])}--{f3(r['hi'])})")
        for (task, es, cond, met), g in zdf.groupby(["task", "eval_set", "cond", "metric"]):
            sbase = f"zssum/{task}/{es}/{cond}/{met}"
            put("v", f"{sbase}/min", f3(g["mean"].min()))
            put("v", f"{sbase}/max", f3(g["mean"].max()))
    if "zeroshot" in rn:
        for zk, zv in rn["zeroshot"].items():
            # zk is t1/test/qwen35_2b/zeroshot-concise/f1_valthr
            parts = zk.split("/")
            if len(parts) == 5:
                zt, zes, zm, zc, zmet = parts
                base = f"zs/{zt}/{zes}/{zc}/{zm}/{zmet}"
                if "diff" in zv and zv["diff"] is not None:
                    put("v", f"{base}/diff", fmt("diff", zv["diff"]))
                    put("v", f"{base}/diff_lo", fmt("two_lo", zv["diff_lo"]))
                    put("v", f"{base}/diff_hi", fmt("two_hi", zv["diff_hi"]))
                    put("v", f"{base}/diffci", f"{fmt('diff', zv['diff'])} (95\\% CI {fmt('two_lo', zv['diff_lo'])} to {fmt('two_hi', zv['diff_hi'])})")
                if "p_ge_ref" in zv and zv["p_ge_ref"] is not None:
                    put("v", f"{base}/p_ge_ref", f"{float(zv['p_ge_ref']):.2f}")
                if "ref" in zv and zv["ref"] is not None:
                    put("v", f"{base}/ref", f3(zv["ref"]))

    if "pilot_cells" in rn:
        pc_d = rn["pilot_cells"]
        diffs_all = [abs(float(v["mean_5"]) - float(v["mean_4"])) for v in pc_d.values()]
        diffs_disc = [abs(float(v["mean_5"]) - float(v["mean_4"])) for k, v in pc_d.items() if not k.endswith("/ece")]
        if diffs_all:
            put("v", "pilot/absmax_all", f3(max(diffs_all)))
        if diffs_disc:
            put("v", "pilot/absmax_disc", f3(max(diffs_disc)))

    # Gate 1 zero-shot positive rates & prevalence
    g1_p = REV / "gate1_zeroshot_vs_reference.csv"
    if g1_p.exists():
        g1 = pd.read_csv(g1_p)
        for r in g1[g1["cond"].isin(["zeroshot", "zeroshot-detailed"])].to_dict("records"):
            ck = "zeroshot-concise" if r["cond"] == "zeroshot" else "zeroshot-detailed"
            base = f"g1zs/{r['task']}/{ck}/{r['model']}"
            put("v", f"{base}/pos_rate", f3(r["pos_rate"]))
            put("v", f"{base}/prev", f3(r["prev"]))

    # 7. No-guideline prompt ablation (Qwen3.5-2B)
    ni_p = REV / "noinstr.csv"
    if ni_p.exists():
        ndf = pd.read_csv(ni_p)
        for r in ndf.to_dict("records"):
            base = f"noinstr/{r['task']}/{r['eval_set']}/{int(r['n'])}/{r['metric']}"
            put("v", f"{base}/noinstr", f3(r["noinstr"]))
            put("v", f"{base}/main", f3(r["main"]))
            put("v", f"{base}/diff", fmt("diff", r["diff"]))
            put("v", f"{base}/lo", fmt("two_lo", r["lo"]))
            put("v", f"{base}/hi", fmt("two_hi", r["hi"]))
            put("v", f"{base}/diffci", f"{fmt('diff', r['diff'])} (95\\% CI {fmt('two_lo', r['lo'])} to {fmt('two_hi', r['hi'])})")

    # 8. T1 encoder LR check
    lr_p = REV / "lr_check.csv"
    if lr_p.exists():
        ldf = pd.read_csv(lr_p)
        for r in ldf.to_dict("records"):
            base = f"lrchk/{r['model']}/{r['setting']}/{r['eval_set']}/{r['metric']}"
            put("v", f"{base}/mean", f3(r["mean"]))
            put("v", f"{base}/lo", f3(r["lo"]))
            put("v", f"{base}/hi", f3(r["hi"]))
            put("v", f"{base}/ci", f"{f3(r['mean'])} (95\\% CI {f3(r['lo'])}--{f3(r['hi'])})")
            put("v", f"{base}/collapsed", fmt("collapsed", r["collapsed"]))
            if pd.notna(r["ref"]):
                put("v", f"{base}/ref", f3(r["ref"]))
                put("v", f"{base}/p_ge_ref", pval(r["p_ge_ref"], 10000, two_sided=False))

    # 9. Two-level plateau gaps (replaces item-only \pv{gap/...})
    s2_p = REV / "table_s2_two_level.csv"
    if s2_p.exists():
        s2 = pd.read_csv(s2_p)
        for r in s2.to_dict("records"):
            base = f"gap/{r['task']}/{r['model']}/{r['eval_set']}/{int(r['n'])}/{r['labels']}"
            put("v", f"{base}/gold", f3(r["gold"]))
            put("v", f"{base}/teacher", f3(r["teacher"]))
            put("v", f"{base}/gap", fmt("gap", r["gap"]))
            put("v", f"{base}/tminusg", fmt("tminusg", -r["gap"]))
            put("v", f"{base}/lo", fmt("two_lo", r["two_lo"]))
            put("v", f"{base}/hi", fmt("two_hi", r["two_hi"]))
            put("v", f"{base}/tminusg_lo", fmt("two_lo", -r["two_hi"]))
            put("v", f"{base}/tminusg_hi", fmt("two_hi", -r["two_lo"]))
            put("v", f"{base}/item_lo", fmt("item_lo", r["item_lo"]))
            put("v", f"{base}/item_hi", fmt("item_hi", r["item_hi"]))
            put("v", f"{base}/p", pval(r["two_p"], B_BOOT))
        put("v", "s2sum/total", fmt("k", len(s2)))
        put("v", "s2sum/two_sig", fmt("k", int(((s2["two_lo"] > 0) | (s2["two_hi"] < 0)).sum())))
        put("v", "s2sum/item_sig", fmt("k", int(((s2["item_lo"] > 0) | (s2["item_hi"] < 0)).sum())))
        w_ratio = (s2["two_hi"] - s2["two_lo"]) / (s2["item_hi"] - s2["item_lo"])
        put("v", "s2sum/width_ratio_median", f"{float(w_ratio.median()):.2f}")
        put("v", "s2sum/width_ratio_max", f"{float(w_ratio.max()):.2f}")

    # 9b. Pre-registered comparison cluster vs two-level vs item-only CI summaries
    if "seed_aware" in rn:
        sa = rn["seed_aware"]
        cl_ratios = []
        two_ratios = []
        for ck, cv in sa.items():
            if not ck.startswith("cmp/"):
                continue
            cid = ck.split("/")[1]
            put("v", f"cmpci/{cid}/diff", fmt("diff", cv["diff"]))
            put("v", f"cmpci/{cid}/two_lo", fmt("two_lo", cv["two_lo"]))
            put("v", f"cmpci/{cid}/two_hi", fmt("two_hi", cv["two_hi"]))
            put("v", f"cmpci/{cid}/item_lo", fmt("item_lo", cv["item_lo"]))
            put("v", f"cmpci/{cid}/item_hi", fmt("item_hi", cv["item_hi"]))
            put("v", f"cmpci/{cid}/cl_lo", fmt("two_lo", cv["cl_lo"]))
            put("v", f"cmpci/{cid}/cl_hi", fmt("two_hi", cv["cl_hi"]))
            w_item = cv["item_hi"] - cv["item_lo"]
            w_two = cv["two_hi"] - cv["two_lo"]
            w_cl = cv["cl_hi"] - cv["cl_lo"]
            if w_item > 0:
                two_ratios.append(w_two / w_item)
            if w_two > 0:
                cl_ratios.append(w_cl / w_two)
        if two_ratios:
            put("v", "cmpci/summary/two_vs_item_median", f"{float(np.median(two_ratios)):.2f}")
            put("v", "cmpci/summary/two_vs_item_max", f"{float(np.max(two_ratios)):.2f}")
        if cl_ratios:
            put("v", "cmpci/summary/cl_vs_two_min", f"{float(np.min(cl_ratios)):.2f}")
            put("v", "cmpci/summary/cl_vs_two_median", f"{float(np.median(cl_ratios)):.2f}")
            put("v", "cmpci/summary/cl_vs_two_max", f"{float(np.max(cl_ratios)):.2f}")

    # 10. Teacher-family check (Qwen3.5-2B same-family vs Gemma 4 E2B cross-family)
    tf_p = REV / "teacher_family.csv"
    if tf_p.exists():
        tf = pd.read_csv(tf_p)
        for r in tf.to_dict("records"):
            base = f"tfam/{r['task']}/{r['eval_set']}/{r['teacher']}/{int(r['n'])}/{r['metric']}"
            put("v", f"{base}/cost_qwen", fmt("gap", r["cost_qwen"]))
            put("v", f"{base}/cost_gemma", fmt("gap", r["cost_gemma"]))
            put("v", f"{base}/family_diff", fmt("family_diff", r["family_diff"]))
            put("v", f"{base}/lo", fmt("two_lo", r["lo"]))
            put("v", f"{base}/hi", fmt("two_hi", r["hi"]))
        for (teacher, met), g in tf.groupby(["teacher", "metric"]):
            sbase = f"tfamsum/{teacher}/{met}"
            put("v", f"{sbase}/min", fmt("family_diff", g["family_diff"].min()))
            put("v", f"{sbase}/median", fmt("family_diff", g["family_diff"].median()))
            put("v", f"{sbase}/max", fmt("family_diff", g["family_diff"].max()))
            sig = int(((g["lo"] > 0) | (g["hi"] < 0)).sum())
            put("v", f"{sbase}/sig_cells", fmt("k", sig))
            put("v", f"{sbase}/total_cells", fmt("k", len(g)))

    # 11. Expanded 6-metric calibration
    cf_p = REV / "calibration_full.csv"
    if cf_p.exists():
        cf = pd.read_csv(cf_p)
        cf_main = cf[cf["tag"].isna() | (cf["tag"] == "concise")]
        for r in cf_main.to_dict("records"):
            base = f"cal/{r['task']}/{r['eval_set']}/{r['model']}/{r['labels']}/{int(r['n'])}/{r['calib']}"
            for mcol in ("bbs", "brier", "bs_correct", "bs_error", "ece", "ece_mass15"):
                put("v", f"{base}/{mcol}", f3(r[mcol]))
                if pd.notna(r.get(f"{mcol}_sd")):
                    put("v", f"{base}/{mcol}_sd", f3(r[f"{mcol}_sd"]))

    cfr_p = REV / "calibration_family_ranges.csv"
    if cfr_p.exists():
        cfr = pd.read_csv(cfr_p)
        for r in cfr.to_dict("records"):
            base = f"calfam/{r['task']}/{r['eval_set']}/{r['metric']}/{r['family']}"
            put("v", f"{base}/min", f3(r["min"]))
            put("v", f"{base}/max", f3(r["max"]))

    # 12. Threshold drift, exchangeability bands, T3 within-val check, recalibration, and prior shift
    dfpr_p = REV / "drift_fpr.csv"
    if dfpr_p.exists():
        dfpr = pd.read_csv(dfpr_p)
        for r in dfpr.to_dict("records"):
            base = f"driftfpr/{r['task']}/{r['model']}/{r['eval_set']}/{r['method']}"
            for c in ("mean", "median", "sd", "mean_without_collapsed", "n_collapsed"):
                if c in r and pd.notna(r[c]):
                    put("v", f"{base}/{c}", fmt(c, r[c]))
            if "per_seed" in r and pd.notna(r["per_seed"]):
                for si, sv in enumerate(str(r["per_seed"]).split(";")):
                    put("v", f"{base}/s{si}", f3(float(sv.rstrip("*"))))
    if "drift" in rn:
        dr_d = rn["drift"]
        if "t3_split" in dr_d:
            t3s = dr_d["t3_split"]
            put("v", "t3val/tox_neg_val", f"{float(t3s['tox_neg_val']):.2f}")
            put("v", "t3val/tox_neg_test", f"{float(t3s['tox_neg_test']):.2f}")
            put("v", "t3val/tox_neg_pool", f"{float(t3s['tox_neg_pool']):.2f}")
            put("v", "t3val/group_tv_neg", f3(t3s["group_tv_neg"]))
            put("v", "t3val/prev_val", half_up3(t3s["prev_val"]))
            put("v", "t3val/prev_test", half_up3(t3s["prev_test"]))
        if "method_spread" in dr_d:
            for ms_es, ms_vals in dr_d["method_spread"].items():
                for ms_k, ms_v in ms_vals.items():
                    put("v", f"driftspread/{ms_es}/{ms_k}", f3(ms_v))

    dprec_p = REV / "drift_precision_target.csv"
    if dprec_p.exists():
        dprec = pd.read_csv(dprec_p)
        for r in dprec.to_dict("records"):
            base = f"driftprec/{r['task']}/{r['model']}/{r['eval_set']}/{r['calib']}"
            for c in ("mean_flag_rate", "mean_precision", "mean_recall", "median_precision", "median_recall"):
                put("v", f"{base}/{c}", f3(r[c]))

    exch_p = REV / "drift_exchangeability_band.csv"
    if exch_p.exists():
        exdf = pd.read_csv(exch_p)
        for r in exdf.to_dict("records"):
            base = f"exch/{r['task']}/{r['eval_set']}"
            for c in ("n_val_neg", "n_test_neg", "expected", "band_lo", "band_hi",
                      "models", "models_above", "seeds", "seeds_above", "model_min", "model_max"):
                put("v", f"{base}/{c}", fmt(c, r[c]))

    t3v_p = REV / "drift_t3_within_val.csv"
    if t3v_p.exists():
        t3v = pd.read_csv(t3v_p)
        put("v", "t3val/within_min", f3(t3v["within_val_fpr"].min()))
        put("v", "t3val/within_max", f3(t3v["within_val_fpr"].max()))
        put("v", "t3val/within_mean", f3(t3v["within_val_fpr"].mean()))
        put("v", "t3val/test_min", f3(t3v["test_fpr"].min()))
        put("v", "t3val/test_max", f3(t3v["test_fpr"].max()))
        for mod, g in t3v.groupby("model"):
            put("v", f"t3val/{mod}/within_mean", f3(g["within_val_fpr"].mean()))
            put("v", f"t3val/{mod}/test_mean", f3(g["test_fpr"].mean()))

    recal_p = REV / "recal_target_negatives_summary.csv"
    if recal_p.exists():
        rcdf = pd.read_csv(recal_p)
        for r in rcdf.to_dict("records"):
            base = f"recal/{r['task']}/{r['model']}/{int(r['m'])}"
            for c in ("fpr", "fpr_p05", "fpr_p95", "recall", "p_over_target"):
                put("v", f"{base}/{c}", f3(r[c]))
        neural_rc = rcdf[rcdf.model != "tfidf_lr"]
        for (task, m_neg), g in neural_rc.groupby(["task", "m"]):
            sbase = f"recalsum/{task}/{int(m_neg)}"
            put("v", f"{sbase}/fpr_min", f3(g["fpr"].min()))
            put("v", f"{sbase}/fpr_max", f3(g["fpr"].max()))
            put("v", f"{sbase}/rec_min", f3(g["recall"].min()))
            put("v", f"{sbase}/rec_max", f3(g["recall"].max()))

        # all neural models: spread of the mean FPR, share of the 200 draws above alpha, 95th-percentile FPR, labelled items
        n_shift = {"t1_rag_hallucination": 3000, "t2_response_safety": 1709, "t3_implicit_toxicity": 2108}
        if exch_p.exists():
            for task, ntot in n_shift.items():
                er = exdf[(exdf.task == task) & (exdf.eval_set == "shift")].iloc[0]
                neg_rate = float(er.n_test_neg) / ntot
                put("v", f"recalsum/{task}/neg_rate", f3(neg_rate))
                for m_neg in (50, 100, 200):
                    put("v", f"recalsum/{task}/{m_neg}/items", f"{m_neg / neg_rate:,.0f}".replace(",", "{,}"))
        for (task, m_neg), g in neural_rc[neural_rc.m > 0].groupby(["task", "m"]):
            sbase = f"recalsum/{task}/{int(m_neg)}"
            put("v", f"{sbase}/pover_min", f"{100 * g['p_over_target'].min():.0f}")
            put("v", f"{sbase}/pover_max", f"{100 * g['p_over_target'].max():.0f}")
            put("v", f"{sbase}/p95_max", f3(g["fpr_p95"].max()))
            put("v", f"{sbase}/p05_min", f3(g["fpr_p05"].min()))

    # 12b. Zero-shot decoders: threshold chosen on random validation subsets (Table S10; exploratory)
    vs_p = REV / "zeroshot_val_subsets.csv"
    if vs_p.exists():
        vs = pd.read_csv(vs_p)
        for r in vs.to_dict("records"):
            base = f"valsub/{short(r['task'])}/{r['eval_set']}/{r['model']}/m{int(r['m'])}"
            for c in ("f1_mean", "f1_sd", "f1_p05", "f1_p95", "f1_full_v", "f1_raw05", "q_ref", "floor"):
                put("v", f"{base}/{c}", f3(r[c]))
            put("v", f"{base}/p_ge_ref", f3(r["p_ge_ref"]))
            put("v", f"{base}/p_ge_floor", f3(r["p_ge_floor"]))
        ts = vs[vs.eval_set == "test"]
        for t, g in ts.groupby("task"):
            for m_sub, gm in g.groupby("m"):
                put("v", f"valsub/{short(t)}/test/m{int(m_sub)}/sd_max", f3(gm["f1_sd"].max()))
                put("v", f"valsub/{short(t)}/test/m{int(m_sub)}/loss_max", f3((gm["f1_full_v"] - gm["f1_mean"]).max()))

    # 12c. Final revision: validation-vs-test prevalence gap, throughput ratios, overlap denominators, prompt-overlap audit
    fin = rn.get("final", {})
    for k, v in fin.items():
        if k.startswith("serve/"):
            put("v", k, f"{float(v):.0f}")
        elif k.startswith("thrgap/"):
            put("v", k, f3(v) if "prev" not in k else half_up3(v))
        elif k.startswith("overlap/"):
            put("v", f"t2ov/{k.split('overlap/')[1]}", f"{int(v):,}".replace(",", "{,}") if "pct" not in k else f"{float(v):.1f}")
    _sv = pd.read_csv(TAB / "table8_serving.csv")
    _sv = _sv[_sv.family != "api"].set_index(["model", "task"]).items_per_s
    for _t in ("t1_rag_hallucination", "t2_response_safety", "t3_implicit_toxicity"):
        put("v", f"serve/deberta_vs_qwen35_2b/{short(_t)}", f"{_sv[('deberta_v3_large', _t)] / _sv[('qwen35_2b', _t)]:.0f}")
    for k, v in rn.get("prompt_overlap", {}).items():
        if isinstance(v, dict):
            continue
        put("v", f"promptov/{k}", f"{int(v):,}" if isinstance(v, int) else f"{float(v):.2f}")
    for k, v in rn.get("prompt_overlap", {}).get("corpus_texts", {}).items():
        put("v", f"promptov/{k}/corpus", f"{int(v):,}".replace(",", "{,}"))

    ps_p = REV / "prior_shift_precision_summary.csv"
    if ps_p.exists():
        psdf = pd.read_csv(ps_p)
        for r in psdf.to_dict("records"):
            base = f"priorshift/{r['task']}/{r['eval_set']}/{r['model']}/{r['method']}"
            for c in ("prev_val", "prev_true", "prev_est", "precision", "recall", "flag_rate"):
                put("v", f"{base}/{c}", f3(r[c]))

    # 13. Repeat-API reference noise
    rsd_p = REV / "ref_repeat_sd.csv"
    if rsd_p.exists():
        rsd = pd.read_csv(rsd_p)
        for r in rsd.to_dict("records"):
            base = f"refsd/{r['task']}/{r['eval_set']}/{r['model']}"
            put("v", f"{base}/sd_ref", fmt("sd_ref", r["sd_ref"]))
            put("v", f"{base}/f1_mean", f3(r["f1_mean_3runs"]))
            put("v", f"{base}/f1_run0", f3(r["f1_full_run0"]))
            put("v", f"{base}/agree", f3(r["pairwise_agree"]))
            put("v", f"{base}/flip_pct", f"{100.0 * float(r['pairwise_flip_rate']):.1f}")
        put("v", "refsd/summary/sd_min", fmt("sd_ref", rsd["sd_ref"].min()))
        put("v", "refsd/summary/sd_max", fmt("sd_ref", rsd["sd_ref"].max()))
        put("v", "refsd/summary/flip_min_pct", f"{100.0 * float(rsd['pairwise_flip_rate'].min()):.1f}")
        put("v", "refsd/summary/flip_max_pct", f"{100.0 * float(rsd['pairwise_flip_rate'].max()):.1f}")

    # 14. T2 prompt overlap & unseen-prompt checks
    if "t2_overlap" in rn:
        t2o = rn["t2_overlap"]
        for k, v in t2o.items():
            if k.startswith("h1c/"):
                sub = k.split("/")[1]
                for field in ("items", "model", "ref", "diff", "lo", "hi"):
                    put("v", f"t2ov/h1c/{sub}/{field}",
                        fmt("two_lo" if field in ("lo", "hi") else field, v[field]))
                put("v", f"t2ov/h1c/{sub}/p", pval(v["p"], 10000))
            elif k.startswith("overlap/"):
                _, es, ntag = k.split("/")
                put("v", f"t2ov/share/{es}/{ntag}/mean", f3(v["mean"]))
                put("v", f"t2ov/share/{es}/{ntag}/pct", f"{100.0 * float(v['mean']):.1f}")
                put("v", f"t2ov/share/{es}/{ntag}/max_pct", f"{100.0 * float(v['max']):.1f}")
            elif k.startswith("breakeven/"):
                _, mod, met, subset = k.split("/")
                put("v", f"t2ov/be/{mod}/{met}/{subset}", fmt_budget(v))
        if "h1c/seen" in t2o and "h1c/all" in t2o:
            seen_i = int(t2o["h1c/seen"]["items"])
            all_i = int(t2o["h1c/all"]["items"])
            put("v", "t2ov/nmax/seen_items", fmt("items", seen_i))
            put("v", "t2ov/nmax/unseen_items", fmt("items", all_i - seen_i))
            put("v", "t2ov/nmax/all_items", fmt("items", all_i))
            put("v", "t2ov/nmax/seen_pct", f"{100.0 * seen_i / all_i:.1f}")
    t2c_p = REV / "t2_curves_unseen_prompts_summary.csv"
    if t2c_p.exists():
        t2c = pd.read_csv(t2c_p)
        for r in t2c.to_dict("records"):
            base = f"t2ov/curve/{r['model']}/{int(r['n'])}/{r['subset']}"
            for c in ("f1_valthr", "f1_at_05", "max_f1", "pr_auc", "ref_f1"):
                put("v", f"{base}/{c}", f3(r[c]))
            for c in ("f1_valthr_minus_ref", "f1_at_05_minus_ref", "max_f1_minus_ref"):
                put("v", f"{base}/{c}", fmt("diff", r[c]))

    # 15. Pairwise seed disagreement
    sd_p = REV / "seed_disagreement.csv"
    if sd_p.exists():
        sddf = pd.read_csv(sd_p)
        for r in sddf.to_dict("records"):
            put("v", f"sdis/{r['task']}/{r['eval_set']}/{r['model']}/{int(r['n'])}/mean", f3(r["mean"]))

    # 16. Ablation min/max summaries (from original tableD_ablations.csv)
    abl = TAB / "tableD_ablations.csv"
    if abl.exists():
        a = pd.read_csv(abl)
        for keys in (["tag", "model", "eval_set", "metric"], ["tag", "eval_set", "metric"]):
            for k, g in a.groupby(keys):
                base = "ablsum/" + "/".join(k if len(keys) == 4 else (k[0], "all", *k[1:]))
                put("v", f"{base}/min", fmt("diff", g["diff"].min()))
                put("v", f"{base}/max", fmt("diff", g["diff"].max()))
                put("v", f"{base}/absmax", fmt("x", g["diff"].abs().max()))

    # 17. Updated Compute, Collapse, Run Counts, and API Spend (from revision accounting)
    comp_rev = REV / "compute_revision.csv"
    if comp_rev.exists():
        ct = pd.read_csv(comp_rev)
        for r in ct.to_dict("records"):
            base = f"gpu/{r['family']}/{r['task']}"
            for c, v in r.items():
                if c not in ("family", "task"):
                    put("v", f"{base}/{col_key(c)}", fmt(c, v))

    col_rev = REV / "collapse_summary_revision.csv"
    if col_rev.exists():
        cs = pd.read_csv(col_rev)
        for r in cs.to_dict("records"):
            base = f"collapse/{r['task']}/{r['model']}"
            for c, v in r.items():
                if c not in ("task", "model"):
                    put("v", f"{base}/{col_key(c)}", fmt(c, v))

    if "accounting" in rn:
        acc = rn["accounting"]
        put("v", "runs/total_finished", fmt("runs", acc["runs/total_finished"]))
        put("v", "runs/trained", fmt("runs", acc["runs/trained"]))
        put("v", "runs/main", fmt("runs", acc["runs/main"]))
        put("v", "runs/ablation", fmt("runs", acc["runs/ablation"]))
        put("v", "runs/hp", fmt("runs", acc["runs/hp"]))
        put("v", "runs/zeroshot", fmt("runs", acc["runs/zeroshot"]))
        put("v", "runs/detector", fmt("runs", acc["runs/detector"]))
        put("v", "runs/pipeline_test", fmt("runs", acc["runs/pipeline_test"]))
        put("v", "collapse/total/constant", fmt("runs", acc["collapse/main_constant"]))
        put("v", "collapse/total/runs", fmt("runs", acc["collapse/main_runs"]))
        put("v", "collapse/ablation/constant", fmt("runs", acc["collapse/ablation_constant"]))
        for sk in ("openai_orig", "openai_repeats", "openai_total", "gemini_orig", "gemini_repeats", "gemini_total", "jev"):
            put("v", f"spend/{sk}", fmt(f"spend_{sk}", acc[f"spend/{sk}"]))
        put("v", "spend/all_api_total",
            fmt("spend_total", acc["spend/openai_total"] + acc["spend/gemini_total"] + acc["spend/jev"]))

    # 18. Operating-point (at_fpr05 / at_prec90) numbers from metrics.parquet at N_max
    met = ROOT / "results" / "metrics.parquet"
    if met.exists():
        m = pd.read_parquet(met)
        m = m[(m.tag.fillna("") == "") & (m.labels == "gold") & m.metric.str.startswith(("at_fpr05", "at_prec90"))
              & m.eval_set.isin(["test", "shift"])]
        m = m[m.n == m.task.map(TOP)]
        for (task, model, es, calib, metric), g in m.groupby(["task", "model", "eval_set", "calib", "metric"]):
            put("v", f"op/{task}/{model}/{es}/{calib}/{metric}", fmt(metric, g.value.mean()))

    # 19. Human label-alignment audits (initial cross-dataset + optional extended source/shift hook)
    if "extended_audit" in rn and rn["extended_audit"]:
        ea = rn["extended_audit"]
        if all(k in ea and not ea[k].get("partial") for k in ("t2_beavertails_pool", "t1_data2txt_shift", "t3_heldout_groups")):
            put("v", "extaudit/ready", "1")
        for aname, entry in ea.items():
            put("v", f"extaudit/{aname}/n", str(int(entry["n"])))
            for who, ad in entry.get("annotators", {}).items():
                put("v", f"extaudit/{aname}/{who}/kappa", f"{ad['kappa']:.2f}")
                put("v", f"extaudit/{aname}/{who}/kappaci", f"{ad['kappa']:.2f} [{ad['lo']:.2f}, {ad['hi']:.2f}]")
                put("v", f"extaudit/{aname}/{who}/unsure", str(int(ad.get("unsure", 0))))
            if "between" in entry and "kappa" in entry["between"]:
                bd = entry["between"]
                put("v", f"extaudit/{aname}/between/kappa", f"{bd['kappa']:.2f}")
                put("v", f"extaudit/{aname}/between/kappaci", f"{bd['kappa']:.2f} [{bd['lo']:.2f}, {bd['hi']:.2f}]")

    out = [
        "% generated by analysis/latex_numbers.py from results/paper_numbers.json, results/tables, and results/revision; do not edit",
        r"\makeatletter",
        r"\def\pn@get#1#2{\ifcsname pn@#1@#2\endcsname\csname pn@#1@#2\endcsname\else\textcolor{red}{??\detokenize{#2}}\fi}",
        r"\newcommand{\pn}[1]{\pn@get{m}{#1}}",
        r"\newcommand{\pnci}[1]{\pn@get{ci}{#1}}",
        r"\newcommand{\pnsd}[1]{\pn@get{sd}{#1}}",
        r"\newcommand{\pcmp}[1]{\pn@get{c}{#1}}",
        r"\newcommand{\pv}[1]{\pn@get{v}{#1}}",
        r"\makeatother",
    ]
    head = len(out)
    for (kind, k), val in defs.items():
        out.append(rf"\expandafter\def\csname pn@{kind}@{k}\endcsname{{{val}}}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text("\n".join(out) + "\n")
    print(f"{len(out) - head:,} numbers -> {DST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
