"""Money-safety behaviour of labels/api.py, with fake vendors (no network, no spend)."""
import json
import sys
from pathlib import Path

import pytest
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "labels"))
import api  # noqa: E402

TASK = "t3_implicit_toxicity"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PREDS", tmp_path / "preds")
    monkeypatch.setattr(api, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(api, "FACTORS", tmp_path / "preds" / "_f.json")
    b = yaml.safe_load(api.BUDGET_PATH.read_text())
    b["vendors"]["openai"] = {"credit_usd": 10.0, "safety_margin_usd": 0.0}
    path = tmp_path / "budget.yaml"
    path.write_text(yaml.safe_dump(b))
    monkeypatch.setattr(api, "BUDGET_PATH", path)
    api._item_tok.clear()
    return path


class FakeClient:
    calls = 0
    fail_after = None

    def __init__(self, model_key):
        pass

    def sync(self, instr, user, cache_key):
        FakeClient.calls += 1
        if FakeClient.fail_after is not None and FakeClient.calls > FakeClient.fail_after:
            raise api.OutOfCredit("openai HTTP 429: insufficient_quota")
        return "Yes", {"input": 1000, "cached": 0, "output": 1}, None


def resp(code, body):
    r = requests.Response()
    r.status_code = code
    r._content = json.dumps(body).encode()
    return r


def test_classify_money_vs_rate_limit():
    with pytest.raises(api.OutOfCredit):
        api.classify("openai", resp(429, {"error": {"code": "insufficient_quota"}}))
    with pytest.raises(api.OutOfCredit):
        api.classify("typesafe", resp(402, {"error": "payment required"}))
    with pytest.raises(api.Transient):
        api.classify("openai", resp(429, {"error": {"code": "rate_limit_exceeded"}}))
    with pytest.raises(api.Transient):
        api.classify("gemini", resp(503, {"error": "unavailable"}))
    api.classify("openai", resp(200, {}))


def test_parse_verdict():
    assert api.parse_verdict("Yes") == 1
    assert api.parse_verdict(" no.") == 0
    assert api.parse_verdict("**Yes**") == 1
    assert api.parse_verdict("Maybe") is None
    assert api.parse_verdict("") is None


def test_out_of_credit_midrun_then_resume_pays_once(sandbox, monkeypatch):
    monkeypatch.setitem(api.CLIENTS, "openai", FakeClient)
    FakeClient.calls, FakeClient.fail_after = 0, 7
    job = api.Job("gpt56_luna", TASK, "test", "concise", mode="sync")
    code = api.run_job(job, limit=20, concurrency=1)
    assert code == api.EXIT_RECHARGE
    assert len(job.cache.records) == 7
    paid_first = api.spent("openai")
    assert paid_first > 0

    FakeClient.calls, FakeClient.fail_after = 0, None
    job2 = api.Job("gpt56_luna", TASK, "test", "concise", mode="sync")
    assert api.run_job(job2, limit=13, concurrency=1) == api.EXIT_DONE
    assert FakeClient.calls == 13                      # --limit counts unpaid items only
    assert set(job2.cache.records) == set(job2.rows.uid[:20])
    assert api.spent("openai") == pytest.approx(paid_first * 20 / 7, rel=1e-3)   # ledger rounds to 1e-6 USD


def test_budget_prefix_and_recharge_message(sandbox, monkeypatch, capsys):
    monkeypatch.setitem(api.CLIENTS, "openai", FakeClient)
    FakeClient.calls, FakeClient.fail_after = 0, None
    per_item = api.price_of("gpt-5.6-luna", "sync", {"input": 1, "cached": 0, "output": 0})
    job = api.Job("gpt56_luna", TASK, "test", "concise", mode="sync")
    costs = api.estimate_item_costs("gpt56_luna", TASK, "concise", job.pending().head(30), "sync")
    b = yaml.safe_load(sandbox.read_text())
    b["vendors"]["openai"]["credit_usd"] = sum(costs[:12]) + per_item  # room for exactly 12 items
    sandbox.write_text(yaml.safe_dump(b))
    code = api.run_job(job, limit=30, concurrency=1)
    assert code == api.EXIT_RECHARGE
    assert FakeClient.calls == 12
    assert "RECHARGE NEEDED (openai)" in capsys.readouterr().out


def test_cache_survives_restart(sandbox):
    c = api.Cache("gpt56_luna", TASK, "concise")
    c.add({"uid": "x1", "verdict": 1})
    c.add({"uid": "x1", "verdict": 0})                 # duplicate ignored
    c2 = api.Cache("gpt56_luna", TASK, "concise")
    assert c2.records == {"x1": {"uid": "x1", "verdict": 1}}


def test_refusal_code():
    assert api.refusal_code('[cyber_policy] {"message": "This content was flagged for possible cybersecurity risk."}') == "cyber_policy"
    assert api.refusal_code('{"message": "This content was flagged for possible biological risk. If"}') == "biological_flagged"
    assert api.refusal_code('[rate_limit_exceeded] {"message": "slow down"}') is None
    assert api.refusal_code(None) is None


def test_refusal_settles_after_two_attempts_and_is_not_retried(sandbox):
    job = api.Job("gpt56_luna", TASK, "test", "concise")
    uid = job.rows.uid[0]
    job.log_refusal(uid, "cyber_policy", "batch_a")
    job.log_refusal(uid, "cyber_policy", "batch_a")            # same attempt logged twice counts once
    assert job.settle_refusals() == 0 and uid in set(job.pending().uid)
    job.log_refusal(uid, "cyber_policy", "batch_b")
    assert job.settle_refusals() == 1 and job.settle_refusals() == 0
    rec = api.Cache("gpt56_luna", TASK, "concise").records[uid]
    assert rec["refused"] == "cyber_policy" and rec["verdict"] is None and rec["cost_usd"] == 0.0
    assert uid not in set(job.pending().uid)
    job.record(uid, "Yes", {"input": 1, "cached": 0, "output": 1}, None, "batch")   # a late answer does not replace it
    assert api.Cache("gpt56_luna", TASK, "concise").records[uid]["refused"] == "cyber_policy"
