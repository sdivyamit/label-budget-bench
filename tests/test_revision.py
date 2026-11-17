"""Revision jobs (configs/sweep.yaml `revision`) are added without changing any existing job."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "queue"))
sys.path.insert(0, str(ROOT / "deploy"))


def test_revision_jobs_and_existing_queue_unchanged():
    import make_jobs
    import plan_revision
    before = yaml.safe_load((ROOT / "queue" / "jobs.yaml").read_text())
    jobs, s = plan_revision.plan(8)
    rev = [j for j in jobs if j["stage"].startswith("rev-b")]
    assert len(rev) == s["jobs"] == 12 + 12 + 39 + 18 + 2
    assert {j["stage"] for j in rev} == {f"rev-b{k}" for k in range(8)}
    rerun = set(make_jobs.SWEEP["revision_rerun"])
    after = {j["id"]: j for j in jobs}
    for j in before:
        if j["id"] not in rerun and not j["stage"].startswith("rev-b"):
            assert after[j["id"]] == j
    zs = [j for j in rev if j["n"] == 0]
    assert len(zs) == 24 and all(j["labels"] == "none" and j["seed"] == 0 and j["family"] == "decoder" for j in zs)
    noinstr = [j for j in rev if j.get("tag") == "abl-noinstr"]
    assert all(j["model"] == "qwen35_2b" and j["hp"] == {"prompt": "none"} for j in noinstr)
    lr = [j for j in rev if j.get("tag", "").startswith("abl-lr")]
    assert all(j["task"] == "t1_rag_hallucination" and j["n"] == 8651 and j["family"] == "encoder" for j in lr)
    assert sorted({(j["model"], j["hp"]["lr"]) for j in lr}) == [
        ("deberta_v3_large", 6.67e-6), ("deberta_v3_large", 6.0e-5), ("modernbert_base", 1.0e-5),
        ("modernbert_base", 9.0e-5), ("modernbert_large", 3.33e-6), ("modernbert_large", 3.0e-5)]


def _annotators() -> list[str]:
    import yaml
    return yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["annotators"]


def test_extended_audit_sheets_are_blind_and_stratified():
    import json
    import pandas as pd
    ext = ROOT / "docs" / "audit" / "extended"
    manifest = json.loads((ext / "manifest.json").read_text())
    assert set(manifest) == {"t2_beavertails_pool", "t1_data2txt_shift", "t3_heldout_groups"}
    for name, info in manifest.items():
        assert info["n"] == 100
        assert len(set(info["uids"])) == 100
        for who in _annotators():
            df = pd.read_csv(ext / f"{name}_audit_{who}.csv", dtype=str, keep_default_na=False)
            assert len(df) == 100
            assert list(df["uid"]) == info["uids"]
            assert "group" not in df.columns
            # sheets were blank when prepared; after annotation they must hold only 0/1
            assert df["label"].isin(["", "0", "1"]).all()


def test_rev_audit_scoring_logic(tmp_path, monkeypatch):
    import pandas as pd
    sys.path.insert(0, str(ROOT / "analysis"))
    import rev_audit

    monkeypatch.setattr(rev_audit, "EXT_AUDIT", tmp_path)
    monkeypatch.setattr(rev_audit, "REV_OUT", tmp_path / "out")

    real_ext = ROOT / "docs" / "audit" / "extended"
    for name, spec in rev_audit.SPECS.items():
        gold = pd.read_parquet(ROOT / "data" / "processed" / f"{spec['task']}.parquet", columns=["uid", "label"]).set_index("uid")
        for who in _annotators():
            df = pd.read_csv(real_ext / f"{name}_audit_{who}.csv", dtype=str, keep_default_na=False)
            df["label"] = gold.loc[df["uid"], "label"].astype(int).astype(str).to_numpy()
            df.to_csv(tmp_path / f"{name}_audit_{who}.csv", index=False)

    assert rev_audit.score() == 0
    scored = pd.read_csv(tmp_path / "out" / "extended_audit_kappa.csv")
    assert len(scored) == 9
    assert (scored["kappa"] == 1.0).all()


def test_rev_api_repeats_subsample_and_verdict_array():
    sys.path.insert(0, str(ROOT / "analysis"))
    import rev_api_repeats as rar

    for task in rar.api.TASK_NAMES:
        for es in ("test", "shift_api"):
            uids = rar.subsample_uids(task, es)
            assert len(uids) == 500
            assert len(set(uids)) == 500

    recs = {
        "u1": {"verdict": 1, "refused": None, "cost_usd": 0.001},
        "u2": {"verdict": None, "refused": "cyber_policy", "cost_usd": 0.0},
        "u3": {"verdict": None, "refused": None, "cost_usd": 0.001},
        "u4": {"verdict": 0, "refused": None, "cost_usd": 0.001},
    }
    v, unp, ref, cost = rar._verdict_array(recs, ["u1", "u2", "u3", "u4"], refusal_label=1)
    assert list(v) == [1.0, 1.0, 0.0, 0.0]
    assert unp == 1 and ref == 1 and abs(cost - 0.003) < 1e-9

