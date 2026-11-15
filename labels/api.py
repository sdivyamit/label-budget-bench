"""Resumable, budget-guarded API labeling for Families C (OpenAI, Gemini) and D (TypeSafe Jev).

Money safety:
  * Every paid answer is appended to preds/api/{task}/{model}__{kind}.jsonl the moment it
    arrives. Reruns skip cached uids, so nothing is paid for twice.
  * Spend is recorded in results/api_ledger.jsonl from the usage each API reports, priced
    with configs/budget.yaml. Open batches reserve their projected cost.
  * Before sending anything the runner projects the cost of the pending items and sends only
    the prefix that fits in `credit_usd - safety_margin - spent - reserved`. If the job does
    not fit, or the vendor reports exhausted credit/quota mid-run, it stops cleanly and prints
    how much to recharge. Recharge, add the amount to credit_usd, rerun the same command.

Exit codes: 0 job complete, 2 batches still running (rerun later), 3 recharge needed.

Usage:
  python labels/api.py estimate                       # cost of the whole §6.7 protocol
  python labels/api.py status
  python labels/api.py dry-run --model gpt56_luna --task t2_response_safety --set test --kind concise
  python labels/api.py run     --model gpt56_luna --task t2_response_safety --set test --kind concise [--mode batch|sync] [--wait]
  python labels/api.py run-all --tier low_cost|frontier|hosted_score|teacher [--mode batch] [--wait]
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
TASKS = yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())
MODELS = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
BUDGET_PATH = ROOT / "configs" / "budget.yaml"
PROMPTS = ROOT / "labels" / "prompts"
PREDS = ROOT / "preds" / "api"
LEDGER = ROOT / "results" / "api_ledger.jsonl"
FACTORS = PREDS / "_token_factors.json"
TASK_NAMES = list(TASKS["tasks"])
EXIT_DONE, EXIT_WAITING, EXIT_RECHARGE = 0, 2, 3
GEMINI_BATCH_MAX_BYTES = 15_000_000
OPENAI_BATCH_MAX_ITEMS = 5_000


class OutOfCredit(Exception):
    """Vendor says the account cannot pay (credit, quota, or spend limit exhausted)."""


class Transient(Exception):
    """Retryable failure (rate limit, 5xx, timeout)."""


# ---------------------------------------------------------------------------------------------
# configuration helpers


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def budget() -> dict:
    return yaml.safe_load(BUDGET_PATH.read_text())


def spec(model_key: str) -> dict:
    return MODELS["api"][model_key]


@functools.lru_cache(maxsize=None)
def prompt(task: str) -> dict:
    return yaml.safe_load((PROMPTS / f"{task}.yaml").read_text())


def render(task: str, kind: str, row) -> tuple[str, str]:
    p = prompt(task)
    user = p["item_template"].format(text_a=row["text_a"], text_b=row.get("text_b") or "")
    return p[kind], user


def item_set(task: str, name: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / f"{task}.parquet")
    subsets = json.loads((ROOT / "data" / "splits" / task / "subsets.json").read_text())
    if name == "test":
        return df[df.split == "test"].reset_index(drop=True)
    if name in ("test_api_detailed", "shift_api_concise", "shift_api_detailed"):
        uids = subsets[name]
    elif name == "probe":
        uids = subsets["test_api_detailed"][:200]
    elif name == "subpool":
        chains = pd.read_parquet(ROOT / "data" / "splits" / task / "chains.parquet")
        uids = chains.loc[chains.in_subpool & (chains.seed == 0)].sort_values("rank").uid.tolist()
    else:
        raise ValueError(f"unknown item set {name}")
    return df.set_index("uid").loc[uids].reset_index()


def protocol_jobs() -> list[dict]:
    """The §6.7 API protocol as a job list (model, task, set, kind)."""
    jobs = []
    for task in TASK_NAMES:
        for m, s in MODELS["api"].items():
            if s["tier"] not in ("frontier", "low_cost", "hosted_score"):
                continue
            for st in ("test", "shift_api_concise"):
                jobs.append(dict(model=m, task=task, set=st, kind="concise", tier=s["tier"]))
            if s["tier"] in ("frontier", "hosted_score"):
                for st in ("test_api_detailed", "shift_api_detailed"):
                    jobs.append(dict(model=m, task=task, set=st, kind="detailed", tier=s["tier"]))
        jobs.append(dict(model=MODELS["frontier_teacher"], task=task, set="subpool",
                         kind="detailed", tier="teacher"))
    return jobs


# ---------------------------------------------------------------------------------------------
# cache, ledger, pricing


class Cache:
    def __init__(self, model_key: str, task: str, kind: str):
        self.path = PREDS / task / f"{model_key}__{prompt(task)['version']}_{kind}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.records[r["uid"]] = r

    def add(self, rec: dict) -> None:
        with self.lock:
            if rec["uid"] in self.records:
                return
            self.records[rec["uid"]] = rec
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")


_ledger_lock = threading.Lock()


def ledger_add(vendor: str, model: str, job: str, cost: float, n: int, note: str = "") -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock, LEDGER.open("a") as f:
        f.write(json.dumps(dict(ts=now(), vendor=vendor, model=model, job=job, n=n,
                                cost_usd=round(cost, 6), note=note)) + "\n")


def spent(vendor: str) -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["cost_usd"] for l in LEDGER.read_text().splitlines()
               if l.strip() and json.loads(l)["vendor"] == vendor)


def price_of(model: str, mode: str, usage: dict) -> float:
    table = budget()["prices"][model]
    p = table.get(mode) or table["sync"]
    fresh = max(usage.get("input", 0) - usage.get("cached", 0), 0)
    return (fresh * p["input"] + usage.get("cached", 0) * p["cached"]
            + usage.get("output", 0) * p["output"]) / 1e6


@functools.lru_cache(maxsize=1)
def _encoder():
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


@functools.lru_cache(maxsize=200_000)
def n_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


_item_tok: dict[str, dict[str, int]] = {}


def item_tokens(task: str, rows: pd.DataFrame) -> dict[str, int]:
    """o200k token count of each rendered item, cached on disk per task (prompt-version keyed)."""
    path = PREDS / f"_tokens_{task}_{prompt(task)['version']}.json"
    if task not in _item_tok:
        _item_tok[task] = json.loads(path.read_text()) if path.exists() else {}
    cache = _item_tok[task]
    missing = rows[~rows.uid.isin(cache)]
    if len(missing):
        for r in missing.to_dict("records"):
            cache[r["uid"]] = n_tokens(render(task, "concise", r)[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    return cache


def expected_output_tokens(model_key: str) -> int:
    return spec(model_key).get("expected_output_tokens", 5)


def token_factor(model_key: str) -> float:
    """Ratio actual/estimated input tokens, learned from dry runs (vendor tokenizers differ)."""
    if FACTORS.exists():
        return json.loads(FACTORS.read_text()).get(model_key, 1.15)
    return 1.15


def estimate_item_costs(model_key: str, task: str, kind: str, rows: pd.DataFrame,
                        mode: str) -> list[float]:
    s = spec(model_key)
    instr = prompt(task)[kind]
    n_instr = n_tokens(instr)
    cacheable = n_instr >= 1024 and s["vendor"] in ("openai", "gemini")
    f = token_factor(model_key)
    out = expected_output_tokens(model_key)
    user_tokens = item_tokens(task, rows)
    costs = []
    for i, uid in enumerate(rows.uid):
        tot = int((n_instr + user_tokens[uid]) * f)
        cached = int(n_instr * f * 0.8) if cacheable and i > 0 else 0
        costs.append(price_of(s["model"], mode, dict(input=tot, cached=cached, output=out)))
    return costs


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------
# vendor clients


def parse_verdict(text: str | None) -> int | None:
    if not text:
        return None
    w = text.strip().strip("*`\"' .").split()
    if not w:
        return None
    w0 = w[0].strip(".,:;!*").lower()
    return 1 if w0 == "yes" else 0 if w0 == "no" else None


def _post(url, headers, payload, timeout=120, files=None, method="POST"):
    try:
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=timeout)
        elif files is not None:
            r = requests.post(url, headers=headers, files=files, data=payload, timeout=timeout)
        else:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise Transient(str(e))
    return r


def classify(vendor: str, r: requests.Response) -> None:
    """Raise OutOfCredit / Transient / RuntimeError for a failed response; return if ok."""
    if r.status_code < 400:
        return
    body = r.text[:800]
    low = body.lower()
    money_words = ("insufficient_quota", "billing", "credit", "balance", "spend limit",
                   "spending limit", "payment", "budget", "exceeded your current quota",
                   "prepay", "hard limit")
    if r.status_code == 402 or (r.status_code in (400, 403, 429) and any(w in low for w in money_words)):
        raise OutOfCredit(f"{vendor} HTTP {r.status_code}: {body}")
    if r.status_code in (408, 409, 429) or r.status_code >= 500:
        raise Transient(f"{vendor} HTTP {r.status_code}: {body[:200]}")
    raise RuntimeError(f"{vendor} HTTP {r.status_code}: {body}")


def with_retries(fn, tries: int = 7):
    for k in range(tries):
        try:
            return fn()
        except Transient as e:
            if k == tries - 1:
                raise OutOfCredit(f"gave up after {tries} retries (persistent rate/quota limit?): {e}")
            time.sleep(min(60, 2 ** k) + random.random())


class OpenAIClient:
    vendor = "openai"
    base = "https://api.openai.com/v1"

    def __init__(self, model_key: str):
        self.s = spec(model_key)
        self.h = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

    def body(self, instr: str, user: str, cache_key: str) -> dict:
        b = {"model": self.s["model"],
             "messages": [{"role": "developer", "content": instr},
                          {"role": "user", "content": user}],
             "prompt_cache_key": cache_key, **self.s.get("request", {})}
        if self.s.get("logprobs", True):
            b.update(logprobs=True, top_logprobs=5)
        return b

    @staticmethod
    def parse(resp: dict) -> tuple[str, dict, float | None]:
        ch = resp["choices"][0]
        text = ch["message"].get("content") or ""
        u = resp.get("usage", {})
        usage = dict(input=u.get("prompt_tokens", 0), output=u.get("completion_tokens", 0),
                     cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        p_yes = None
        lp = (ch.get("logprobs") or {}).get("content") or []
        if lp:
            import math as _m
            probs = {t["token"].strip().lower(): _m.exp(t["logprob"]) for t in lp[0].get("top_logprobs", [])}
            if "yes" in probs or "no" in probs:
                y, n = probs.get("yes", 0.0), probs.get("no", 0.0)
                p_yes = y / (y + n) if y + n > 0 else None
        return text, usage, p_yes

    def sync(self, instr, user, cache_key):
        def go():
            r = _post(f"{self.base}/chat/completions", {**self.h, "Content-Type": "application/json"},
                      self.body(instr, user, cache_key))
            classify(self.vendor, r)
            return self.parse(r.json())
        return with_retries(go)

    def submit_batch(self, items: list[tuple[str, str, str]], cache_key: str) -> str:
        lines = "\n".join(json.dumps({"custom_id": uid, "method": "POST", "url": "/v1/chat/completions",
                                      "body": self.body(i, u, cache_key)}) for uid, i, u in items)

        def upload():
            r = _post(f"{self.base}/files", self.h, {"purpose": "batch"},
                      files={"file": ("batch.jsonl", lines.encode())}, timeout=600)
            classify(self.vendor, r)
            return r.json()["id"]
        fid = with_retries(upload)

        def create():
            r = _post(f"{self.base}/batches", {**self.h, "Content-Type": "application/json"},
                      {"input_file_id": fid, "endpoint": "/v1/chat/completions", "completion_window": "24h"})
            classify(self.vendor, r)
            return r.json()["id"]
        return with_retries(create)

    def poll_batch(self, batch_id: str):
        """Return (state, results) with state in {running, done}; results: (uid, text, usage, p_yes, err)."""
        r = with_retries(lambda: (lambda x: (classify(self.vendor, x), x)[1])(
            _post(f"{self.base}/batches/{batch_id}", self.h, None, method="GET")))
        b = r.json()
        if b["status"] in ("validating", "in_progress", "finalizing", "cancelling"):
            return "running", []
        if b["status"] == "failed":
            errs = json.dumps(b.get("errors"))
            if any(w in errs.lower() for w in ("quota", "billing", "credit", "limit")):
                raise OutOfCredit(f"openai batch {batch_id} failed: {errs[:500]}")
            return "done", [(None, None, None, None, f"batch failed: {errs[:300]}")]
        results = []
        for key in ("output_file_id", "error_file_id"):
            if b.get(key):
                content = with_retries(lambda: _post(f"{self.base}/files/{b[key]}/content", self.h, None,
                                                      method="GET", timeout=600)).text
                for line in content.splitlines():
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    resp = o.get("response") or {}
                    if resp.get("status_code") == 200:
                        results.append((o["custom_id"], *self.parse(resp["body"]), None))
                    else:
                        body = resp.get("body")
                        err = o.get("error") or (body.get("error") if isinstance(body, dict) else None) or body or {}
                        code = err.get("code") if isinstance(err, dict) else None
                        results.append((o["custom_id"], None, None, None,
                                        (f"[{code}] " if code else "") + json.dumps(err)[:300]))
        return "done", results


class GeminiClient:
    vendor = "gemini"
    base = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model_key: str):
        self.s = spec(model_key)
        self.h = {"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"}

    def body(self, instr: str, user: str) -> dict:
        return {"systemInstruction": {"parts": [{"text": instr}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0, **self.s.get("request", {})}}

    @staticmethod
    def parse(resp: dict) -> tuple[str, dict, float | None]:
        cands = resp.get("candidates") or [{}]
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        u = resp.get("usageMetadata", {})
        usage = dict(input=u.get("promptTokenCount", 0), cached=u.get("cachedContentTokenCount", 0),
                     output=u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0))
        return text, usage, None

    def sync(self, instr, user, cache_key):
        def go():
            r = _post(f"{self.base}/models/{self.s['model']}:generateContent", self.h, self.body(instr, user))
            classify(self.vendor, r)
            return self.parse(r.json())
        return with_retries(go)

    def submit_batch(self, items, cache_key):
        reqs = [{"request": self.body(i, u), "metadata": {"key": uid}} for uid, i, u in items]
        payload = {"batch": {"display_name": f"lbb-{cache_key}-{int(time.time())}",
                             "input_config": {"requests": {"requests": reqs}}}}

        def go():
            r = _post(f"{self.base}/models/{self.s['model']}:batchGenerateContent", self.h, payload, timeout=600)
            classify(self.vendor, r)
            return r.json()["name"]
        return with_retries(go)

    def poll_batch(self, name: str):
        r = with_retries(lambda: (lambda x: (classify(self.vendor, x), x)[1])(
            _post(f"{self.base}/{name}", self.h, None, method="GET")))
        op = r.json()
        meta = op.get("metadata", {})
        state = meta.get("state", "")
        if not op.get("done") and state not in ("BATCH_STATE_SUCCEEDED", "BATCH_STATE_FAILED",
                                                 "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED"):
            return "running", []
        if op.get("error") or state == "BATCH_STATE_FAILED":
            err = json.dumps(op.get("error") or meta)[:500]
            if any(w in err.lower() for w in ("billing", "credit", "quota", "spend")):
                raise OutOfCredit(f"gemini batch {name} failed: {err}")
            return "done", [(None, None, None, None, f"batch failed: {err}")]
        out = (op.get("response") or {}).get("inlinedResponses") or (meta.get("output") or {}).get("inlinedResponses") or {}
        results = []
        for o in out.get("inlinedResponses", []):
            uid = (o.get("metadata") or {}).get("key")
            if "response" in o:
                results.append((uid, *self.parse(o["response"]), None))
            else:
                results.append((uid, None, None, None, json.dumps(o.get("error"))[:300]))
        return "done", results


class JevClient:
    vendor = "typesafe"
    url = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, model_key: str):
        self.s = spec(model_key)
        self.h = {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}", "Content-Type": "application/json"}

    def sync(self, instr, user, cache_key):
        def go():
            r = _post(self.url, self.h, {"model": self.s["model"], "state": user,
                                         "questions": {"q": {"type": "noul", "instructions": instr}}})
            classify(self.vendor, r)
            d = r.json()
            score = d["answers"]["q"]["noul"]
            u = d.get("usage", {})
            usage = dict(input=u.get("input_tokens", 0), output=u.get("output_tokens", 0), cached=0)
            return json.dumps(score), usage, float(score)
        return with_retries(go)


CLIENTS = {"openai": OpenAIClient, "gemini": GeminiClient, "typesafe": JevClient}


def jev_instructions(task: str, kind: str) -> str:
    p = prompt(task)
    return p["jev_question"] if kind == "concise" else p["detailed"] + "\n\nQuestion: " + p["jev_question"]


# ---------------------------------------------------------------------------------------------
# runner

REFUSAL_ATTEMPTS = 2     # D25: rejected this many times by the provider's content filter -> recorded as refused


def refusal_code(err: str | None) -> str | None:
    """Policy code if an item failed because the provider's content filter rejected it (not a transient error)."""
    m = re.match(r"\[(\w*policy\w*)\]", err or "")
    if m:
        return m.group(1)
    m = re.search(r"was flagged for possible (\w+) risk", err or "")
    return f"{m.group(1)}_flagged" if m else None



class Job:
    def __init__(self, model: str, task: str, set: str, kind: str, mode: str = "batch", **_):
        self.model, self.task, self.set, self.kind = model, task, set, kind
        self.s = spec(model)
        self.vendor = self.s["vendor"]
        self.mode = "sync" if self.vendor == "typesafe" else mode
        self.cache = Cache(model, task, kind)
        self.rows = item_set(task, set)
        self.name = f"{model}/{task}/{set}/{kind}"
        self.state_path = self.cache.path.with_suffix(".batches.json")

    def pending(self) -> pd.DataFrame:
        done = set(self.cache.records)
        return self.rows[~self.rows.uid.isin(done)].reset_index(drop=True)

    def prompt_pair(self, row) -> tuple[str, str]:
        instr, user = render(self.task, self.kind, row)
        if self.vendor == "typesafe":
            instr = jev_instructions(self.task, self.kind)
        return instr, user

    def batches(self) -> list[dict]:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else []

    def save_batches(self, b: list[dict]) -> None:
        self.state_path.write_text(json.dumps(b, indent=1))

    @property
    def refusal_log(self) -> Path:
        return self.cache.path.with_suffix(".refusals.jsonl")

    def refusals(self) -> list[dict]:
        f = self.refusal_log
        return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []

    def log_refusal(self, uid: str, code: str, batch_id: str) -> None:
        """One line per (item, attempt) the provider's content filter rejected (audit trail)."""
        if any(r["uid"] == uid and r["batch"] == batch_id for r in self.refusals()):
            return
        with self.refusal_log.open("a") as f:
            f.write(json.dumps(dict(uid=uid, code=code, batch=batch_id, ts=now())) + "\n")

    def settle_refusals(self) -> int:
        """An item rejected REFUSAL_ATTEMPTS times gets a final record with no verdict and `refused` = the
        policy code. It is not retried; consumers decide what a refusal means (D25)."""
        seen: dict[str, list[str]] = {}
        for r in self.refusals():
            seen.setdefault(r["uid"], []).append(r["code"])
        n = 0
        for uid, codes in seen.items():
            if len(codes) >= REFUSAL_ATTEMPTS and uid not in self.cache.records:
                self.cache.add(dict(uid=uid, verdict=None, p_yes=None, raw="", refused=codes[-1], attempts=len(codes),
                                    usage={}, cost_usd=0.0, mode="batch", model_id=self.s["model"], ts=now()))
                n += 1
        return n

    def record(self, uid, text, usage, p_yes, mode) -> float:
        cost = price_of(self.s["model"], mode, usage)
        verdict = None
        if self.vendor == "typesafe":
            verdict = int(p_yes >= 0.5)
        else:
            verdict = parse_verdict(text)
        self.cache.add(dict(uid=uid, verdict=verdict, p_yes=p_yes, raw=(text or "")[:200], usage=usage,
                            cost_usd=round(cost, 7), mode=mode, model_id=self.s["model"], ts=now()))
        return cost


def reserved(vendor: str) -> float:
    tot = 0.0
    for f in PREDS.glob("*/*.batches.json"):
        for b in json.loads(f.read_text()):
            if b["vendor"] == vendor and b["status"] == "running":
                tot += b["projected_usd"]
    return tot


def available(vendor: str) -> float:
    v = budget()["vendors"][vendor]
    return v["credit_usd"] - v["safety_margin_usd"] - spent(vendor) - reserved(vendor)


def recharge_msg(vendor: str, need: float) -> str:
    amt = max(5, math.ceil(need + budget()["vendors"][vendor]["safety_margin_usd"]))
    return (f"\nRECHARGE NEEDED ({vendor}): about ${need:.2f} more is needed to finish this job "
            f"(includes the ${budget()['vendors'][vendor]['safety_margin_usd']:.2f} safety margin if it is already used).\n"
            f"  1. Add at least ${amt} on the {vendor} billing page.\n"
            f"  2. Add the same amount to vendors.{vendor}.credit_usd in configs/budget.yaml.\n"
            f"  3. Rerun the same command; finished items are cached and will not be paid again.\n")


def collect_batches(job: Job) -> int:
    """Poll open batches; ingest finished results. Return number still running."""
    bs = job.batches()
    client = CLIENTS[job.vendor](job.model)
    running = 0
    for b in bs:
        if b["status"] != "running":
            continue
        state, results = client.poll_batch(b["id"])
        if state == "running":
            running += 1
            continue
        cost, n_ok, n_err, errs = 0.0, 0, 0, []
        for uid, text, usage, p_yes, err in results:
            if err is None and uid:
                cost += job.record(uid, text, usage, p_yes, "batch")
                n_ok += 1
            else:
                n_err += 1
                errs.append(err)
                if uid and refusal_code(err):
                    job.log_refusal(uid, refusal_code(err), b["id"])
        ledger_add(job.vendor, job.s["model"], job.name, cost, n_ok, note=f"batch {b['id']}")
        b.update(status="done", actual_usd=round(cost, 4), n_ok=n_ok, n_err=n_err, finished=now())
        job.save_batches(bs)
        print(f"  batch {b['id']}: {n_ok} ok, {n_err} failed (retried on next run), ${cost:.3f}")
        if any(e and any(w in e.lower() for w in ("quota", "billing", "credit")) for e in errs):
            raise OutOfCredit(f"{job.vendor} rejected items inside batch {b['id']} for credit/quota")
    settled = job.settle_refusals()
    if settled:
        print(f"  {settled} item(s) refused by the provider's content filter {REFUSAL_ATTEMPTS}+ times: recorded as refused")
    return running


def backfill_refusals(job: Job) -> tuple[int, int]:
    """Re-read the error files of this job's finished batches (collected before refusals were logged)."""
    if job.vendor != "openai":
        return 0, 0
    client = CLIENTS[job.vendor](job.model)
    before = len(job.refusals())
    for b in job.batches():
        if b["status"] != "done":
            continue
        _, results = client.poll_batch(b["id"])
        for uid, _text, _usage, _p, err in results:
            if uid and err is not None and refusal_code(err):
                job.log_refusal(uid, refusal_code(err), b["id"])
    return len(job.refusals()) - before, job.settle_refusals()


def run_job(job: Job, wait: bool = False, concurrency: int = 8, limit: int | None = None) -> int:
    print(f"[{job.name}] mode={job.mode}")
    try:
        if job.mode == "batch":
            running = collect_batches(job)
        in_batch = {u for b in job.batches() if b["status"] == "running" for u in b["uids"]}
        todo = job.pending()
        todo = todo[~todo.uid.isin(in_batch)].reset_index(drop=True)
        if limit is not None:
            todo = todo.head(limit)
        if len(todo):
            costs = estimate_item_costs(job.model, job.task, job.kind, todo, job.mode)
            avail = available(job.vendor)
            cum, k = 0.0, 0
            for c in costs:
                if cum + c > avail:
                    break
                cum += c
                k += 1
            total = sum(costs)
            print(f"  pending {len(todo)} items, projected ${total:.2f}; available ${avail:.2f}")
            if total > budget()["max_job_usd"] and limit is None:
                print(f"  projected cost exceeds max_job_usd={budget()['max_job_usd']}; raise it in budget.yaml to proceed")
                return EXIT_RECHARGE
            short = (sum(costs[k:]) + max(0.0, -avail)) if k < len(costs) else 0.0
            todo, costs = todo.head(k), costs[:k]
            if len(todo):
                if job.mode == "batch":
                    submit_batches(job, todo, costs)
                else:
                    run_sync(job, todo, concurrency)
            if short > 0:
                print(recharge_msg(job.vendor, short))
                return EXIT_RECHARGE
        if job.mode == "batch":
            while True:
                running = collect_batches(job)
                if not running or not wait:
                    break
                print(f"  {running} batch(es) running; polling again in 60 s")
                time.sleep(60)
            if running:
                print(f"  {running} batch(es) still running; rerun later to collect")
                return EXIT_WAITING
        left = len(job.pending())
        why = "not requested (--limit)" if limit is not None else "failed; they retry on rerun"
        print(f"  done {len(job.rows) - left}/{len(job.rows)}" + (f" ({left} {why})" if left else ""))
        return EXIT_DONE
    except OutOfCredit as e:
        print(f"  STOPPED: {e}")
        pend = job.pending()
        need = sum(estimate_item_costs(job.model, job.task, job.kind, pend, job.mode)) if len(pend) else 0
        print(recharge_msg(job.vendor, need))
        return EXIT_RECHARGE


def submit_batches(job: Job, todo: pd.DataFrame, costs: list[float]) -> None:
    client = CLIENTS[job.vendor](job.model)
    bs = job.batches()
    chunk, ccost, size = [], 0.0, 0

    def flush():
        nonlocal chunk, ccost, size
        if not chunk:
            return
        bid = client.submit_batch([(u, i, x) for u, i, x in chunk], f"{job.task}-{job.kind}")
        bs.append(dict(id=bid, vendor=job.vendor, uids=[u for u, _, _ in chunk], status="running",
                       projected_usd=round(ccost, 4), submitted=now()))
        job.save_batches(bs)
        print(f"  submitted batch {bid}: {len(chunk)} items, projected ${ccost:.3f}")
        chunk, ccost, size = [], 0.0, 0

    for r, c in zip(todo.to_dict("records"), costs):
        instr, user = job.prompt_pair(r)
        b = len(instr) + len(user) + 200
        if chunk and ((job.vendor == "gemini" and size + b > GEMINI_BATCH_MAX_BYTES)
                      or len(chunk) >= OPENAI_BATCH_MAX_ITEMS):
            flush()
        chunk.append((r["uid"], instr, user))
        ccost += c
        size += b
    flush()


def run_sync(job: Job, todo: pd.DataFrame, concurrency: int) -> None:
    client = CLIENTS[job.vendor](job.model)
    stop = threading.Event()
    spent_here = [0.0]
    lock = threading.Lock()
    err: list[Exception] = []

    def one(r):
        if stop.is_set():
            return
        instr, user = job.prompt_pair(r)
        try:
            text, usage, p_yes = client.sync(instr, user, f"{job.task}-{job.kind}")
        except OutOfCredit as e:
            stop.set()
            err.append(e)
            return
        c = job.record(r["uid"], text, usage, p_yes, "sync")
        with lock:
            spent_here[0] += c

    try:
        with ThreadPoolExecutor(concurrency) as ex:
            list(ex.map(one, todo.to_dict("records")))
    finally:
        n = len(todo) - len(job.pending().merge(todo[["uid"]]))
        ledger_add(job.vendor, job.s["model"], job.name, spent_here[0], n, note="sync")
        print(f"  sync: {n} items, ${spent_here[0]:.4f}")
    if err:
        raise err[0]


def dry_run(job: Job) -> int:
    """Run `dry_run_items` items synchronously; learn the token factor; project the full job."""
    n = budget()["dry_run_items"]
    sample = job.pending().head(n)
    est_in = [n_tokens(" ".join(job.prompt_pair(r))) for r in sample.to_dict("records")]
    mode = job.mode
    job.mode = "sync"
    code = run_job(job, limit=n, concurrency=4)
    job.mode = mode
    got = [job.cache.records[u]["usage"]["input"] for u in sample.uid if u in job.cache.records]
    if got:
        f = sum(got) / sum(est_in[:len(got)])
        fac = json.loads(FACTORS.read_text()) if FACTORS.exists() else {}
        fac[job.model] = round(f, 3)
        FACTORS.parent.mkdir(parents=True, exist_ok=True)
        FACTORS.write_text(json.dumps(fac, indent=1))
        recs = [job.cache.records[u] for u in sample.uid if u in job.cache.records]
        unparsed = sum(r["verdict"] is None for r in recs)
        outs = sorted(r["usage"]["output"] for r in recs)
        print(f"  token factor {f:.3f}; unparseable {unparsed}/{len(recs)}; output tokens median {outs[len(outs)//2]}")
        rest = job.pending()
        proj = sum(estimate_item_costs(job.model, job.task, job.kind, rest, mode)) if len(rest) else 0
        print(f"  projected cost of the remaining {len(rest)} items ({mode}): ${proj:.2f}")
    return code


def estimate(mode: str = "batch") -> pd.DataFrame:
    rows = []
    for j in protocol_jobs():
        s = spec(j["model"])
        m = "sync" if s["vendor"] == "typesafe" else mode
        items = item_set(j["task"], j["set"])
        cache = Cache(j["model"], j["task"], j["kind"])
        todo = items[~items.uid.isin(set(cache.records))]
        cost = sum(estimate_item_costs(j["model"], j["task"], j["kind"], todo, m)) if len(todo) else 0.0
        rows.append(dict(vendor=s["vendor"], model=j["model"], tier=j["tier"], task=j["task"][:2],
                         set=j["set"], kind=j["kind"], items=len(items), todo=len(todo), usd=round(cost, 2)))
    return pd.DataFrame(rows)


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("dry-run", "run"):
        p = sub.add_parser(c)
        p.add_argument("--model", required=True, choices=list(MODELS["api"]))
        p.add_argument("--task", required=True, choices=TASK_NAMES)
        p.add_argument("--set", required=True)
        p.add_argument("--kind", required=True, choices=["concise", "detailed"])
        p.add_argument("--mode", default="batch", choices=["batch", "sync"])
        p.add_argument("--wait", action="store_true")
        p.add_argument("--limit", type=int)
    p = sub.add_parser("run-all")
    p.add_argument("--tier", required=True, choices=["frontier", "low_cost", "hosted_score", "teacher"])
    p.add_argument("--vendor")
    p.add_argument("--mode", default="batch", choices=["batch", "sync"])
    p.add_argument("--wait", action="store_true")
    sub.add_parser("estimate").add_argument("--mode", default="batch", choices=["batch", "sync"])
    sub.add_parser("status")
    sub.add_parser("refusals")      # backfill content-filter refusals from finished batches, then settle them
    a = ap.parse_args()

    if a.cmd == "estimate":
        df = estimate(a.mode)
        pd.set_option("display.width", 200)
        print(df.to_string(index=False))
        print("\nby vendor:\n", df.groupby("vendor").usd.sum().round(2).to_string())
        print("\nby model:\n", df.groupby(["vendor", "model"]).usd.sum().round(2).to_string())
        return 0
    if a.cmd == "status":
        for v in budget()["vendors"]:
            print(f"{v:9s} spent ${spent(v):8.3f}  reserved ${reserved(v):7.3f}  available ${available(v):8.3f}")
        df = estimate()
        print(df[df.todo < df["items"]].to_string(index=False) if (df.todo < df["items"]).any() else "no API jobs started")
        return 0
    if a.cmd == "refusals":
        done = set()
        for j in protocol_jobs():
            job = Job(**j)
            if job.vendor != "openai" or job.cache.path in done:
                continue
            done.add(job.cache.path)
            logged, settled = backfill_refusals(job)
            codes = pd.Series([r.get("refused") for r in job.cache.records.values() if r.get("refused")])
            print(f"{job.model}/{job.task}/{job.kind}: +{logged} logged, +{settled} settled; refused records: "
                  f"{codes.value_counts().to_dict() if len(codes) else 0}")
        return 0
    if a.cmd == "run-all":
        worst = EXIT_DONE
        for j in protocol_jobs():
            if j["tier"] != a.tier or (a.vendor and spec(j["model"])["vendor"] != a.vendor):
                continue
            code = run_job(Job(**j, mode=a.mode), wait=a.wait)
            if code == EXIT_RECHARGE:
                return code
            worst = max(worst, code)
        return worst
    job = Job(a.model, a.task, a.set, a.kind, a.mode)
    if a.cmd == "dry-run":
        return dry_run(job)
    return run_job(job, wait=a.wait, limit=a.limit)


if __name__ == "__main__":
    sys.exit(main())
