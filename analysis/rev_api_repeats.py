"""Phase 3a: Repeat runs of frontier hosted references (GPT-6.1 Sol and Gemini 4 Argon).

Runs two additional passes (rep1, rep2) using the concise prompt on a 500-item proportionally
stratified subsample of each of the six evaluation sets (test and shift_api_concise for T1, T2, T3):
  * test_500  = subsets.json["test_api_detailed"][:500]
  * shift_500 = subsets.json["shift_api_concise"][:500]

Never modifies the frozen preds/api/ cache; writes to preds/api_repeats/{task}/ and records
actual API spend in results/api_ledger.jsonl.

Usage:
  python analysis/rev_api_repeats.py --estimate
  python analysis/rev_api_repeats.py --run --mode batch --wait
  python analysis/rev_api_repeats.py --run --mode sync
  python analysis/rev_api_repeats.py --analyze
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "labels"))
sys.path.insert(0, str(ROOT / "analysis"))
import api  # noqa: E402
from eval import metrics as M  # noqa: E402

REPEATS_DIR = ROOT / "preds" / "api_repeats"
REV_OUT = ROOT / "results" / "revision"
FRONTIER = ["gpt61_sol", "gemini4_argon"]
REPS = (1, 2)
SUB_N = 500


def subsample_uids(task: str, eval_set: str) -> list[str]:
    subsets = json.loads((ROOT / "data" / "splits" / task / "subsets.json").read_text())
    if eval_set == "test":
        return subsets["test_api_detailed"][:SUB_N]
    if eval_set == "shift_api":
        return subsets["shift_api_concise"][:SUB_N]
    raise ValueError(f"unknown eval_set {eval_set}")


def subsample_rows(task: str, eval_set: str) -> pd.DataFrame:
    uids = subsample_uids(task, eval_set)
    df = pd.read_parquet(ROOT / "data" / "processed" / f"{task}.parquet").set_index("uid")
    return df.loc[uids].reset_index()


class RepeatCache:
    def __init__(self, model_key: str, task: str, rep: int, kind: str = "concise"):
        ver = api.prompt(task)["version"]
        self.path = REPEATS_DIR / task / f"{model_key}__{ver}_{kind}__rep{rep}.jsonl"
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


class RepeatJob:
    def __init__(self, model: str, task: str, eval_set: str, rep: int, mode: str = "batch"):
        self.model = model
        self.task = task
        self.eval_set = eval_set
        self.rep = rep
        self.kind = "concise"
        self.mode = mode
        self.s = api.spec(model)
        self.vendor = self.s["vendor"]
        self.cache = RepeatCache(model, task, rep, self.kind)
        self.rows = subsample_rows(task, eval_set)
        self.name = f"repeat{rep}/{model}/{task}/{eval_set}/{self.kind}"
        self.state_path = self.cache.path.with_suffix(f".{eval_set}.batches.json")

    def pending(self) -> pd.DataFrame:
        done = set(self.cache.records)
        return self.rows[~self.rows.uid.isin(done)].reset_index(drop=True)

    def prompt_pair(self, row) -> tuple[str, str]:
        return api.render(self.task, self.kind, row)

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
        if any(r["uid"] == uid and r["batch"] == batch_id for r in self.refusals()):
            return
        with self.refusal_log.open("a") as f:
            f.write(json.dumps(dict(uid=uid, code=code, batch=batch_id, ts=api.now())) + "\n")

    def settle_refusals(self) -> int:
        seen: dict[str, list[str]] = {}
        for r in self.refusals():
            seen.setdefault(r["uid"], []).append(r["code"])
        n = 0
        for uid, codes in seen.items():
            if len(codes) >= api.REFUSAL_ATTEMPTS and uid not in self.cache.records:
                self.cache.add(dict(
                    uid=uid, verdict=None, p_yes=None, raw="", refused=codes[-1],
                    attempts=len(codes), usage={}, cost_usd=0.0, mode=self.mode,
                    model_id=self.s["model"], ts=api.now(),
                ))
                n += 1
        return n

    def record(self, uid: str, text: str | None, usage: dict, p_yes: float | None, mode: str) -> float:
        cost = api.price_of(self.s["model"], mode, usage)
        verdict = api.parse_verdict(text)
        self.cache.add(dict(
            uid=uid, verdict=verdict, p_yes=p_yes, raw=(text or "")[:200], usage=usage,
            cost_usd=round(cost, 7), mode=mode, model_id=self.s["model"], ts=api.now(),
        ))
        return cost


def total_reserved(vendor: str) -> float:
    tot = api.reserved(vendor)
    for f in REPEATS_DIR.glob("*/*.batches.json"):
        for b in json.loads(f.read_text()):
            if b["vendor"] == vendor and b["status"] == "running":
                tot += b["projected_usd"]
    return tot


def available_usd(vendor: str) -> float:
    v = api.budget()["vendors"][vendor]
    return v["credit_usd"] - v["safety_margin_usd"] - api.spent(vendor) - total_reserved(vendor)


def empirical_item_costs(model: str, task: str, uids: list[str], mode: str) -> float:
    ver = api.prompt(task)["version"]
    orig_path = ROOT / "preds" / "api" / task / f"{model}__{ver}_concise.jsonl"
    recs = {}
    for line in orig_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["uid"]] = r
    tot = 0.0
    for u in uids:
        r = recs[u]
        if r.get("usage"):
            tot += api.price_of(api.spec(model)["model"], mode, r["usage"])
    return tot


def all_jobs(mode: str = "batch", models=None, reps=None) -> list[RepeatJob]:
    models = models or FRONTIER
    reps = reps or REPS
    jobs = []
    for rep in reps:
        for model in models:
            for task in api.TASK_NAMES:
                for es in ("test", "shift_api"):
                    jobs.append(RepeatJob(model, task, es, rep, mode=mode))
    return jobs


def estimate_cmd() -> int:
    rows = []
    for model in FRONTIER:
        for task in api.TASK_NAMES:
            for es in ("test", "shift_api"):
                sub = subsample_rows(task, es)
                uids = sub["uid"].tolist()
                emp_b = empirical_item_costs(model, task, uids, "batch")
                emp_s = empirical_item_costs(model, task, uids, "sync")
                est_b = sum(api.estimate_item_costs(model, task, "concise", sub, "batch"))
                est_s = sum(api.estimate_item_costs(model, task, "concise", sub, "sync"))
                pending_reps = sum(
                    1 for rep in REPS
                    if len(RepeatJob(model, task, es, rep).pending()) > 0
                )
                rows.append(dict(
                    vendor=api.spec(model)["vendor"], model=model, task=task[:2], set=es,
                    n=len(sub), pending_reps=pending_reps,
                    emp_batch_2x=round(emp_b * pending_reps, 3),
                    est_batch_2x=round(est_b * pending_reps, 3),
                    emp_sync_2x=round(emp_s * pending_reps, 3),
                    est_sync_2x=round(est_s * pending_reps, 3),
                ))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nSummary by vendor (for pending repeats):")
    for v, g in df.groupby("vendor"):
        print(f"  {v:7s} emp_batch=${g.emp_batch_2x.sum():.2f}  est_batch=${g.est_batch_2x.sum():.2f}  "
              f"emp_sync=${g.emp_sync_2x.sum():.2f}  est_sync=${g.est_sync_2x.sum():.2f}  "
              f"available=${available_usd(v):.3f} (spent=${api.spent(v):.3f}, reserved=${total_reserved(v):.3f})")
    return 0


def collect_batches(job: RepeatJob) -> int:
    bs = job.batches()
    client = api.CLIENTS[job.vendor](job.model)
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
                if uid and api.refusal_code(err):
                    job.log_refusal(uid, api.refusal_code(err), b["id"])
        api.ledger_add(job.vendor, job.s["model"], job.name, cost, n_ok, note=f"batch {b['id']}")
        b.update(status="done", actual_usd=round(cost, 4), n_ok=n_ok, n_err=n_err, finished=api.now())
        job.save_batches(bs)
        print(f"  batch {b['id']}: {n_ok} ok, {n_err} failed, ${cost:.3f}")
        if any(e and any(w in e.lower() for w in ("quota", "billing", "credit")) for e in errs):
            raise api.OutOfCredit(f"{job.vendor} rejected items inside batch {b['id']} for credit/quota")
    settled = job.settle_refusals()
    if settled:
        print(f"  {settled} item(s) refused {api.REFUSAL_ATTEMPTS}+ times: recorded as refused")
    return running


def submit_batches(job: RepeatJob, todo: pd.DataFrame, costs: list[float]) -> None:
    client = api.CLIENTS[job.vendor](job.model)
    bs = job.batches()
    chunk, ccost, size = [], 0.0, 0

    def flush():
        nonlocal chunk, ccost, size
        if not chunk:
            return
        bid = client.submit_batch([(u, i, x) for u, i, x in chunk], f"{job.task}-{job.kind}-rep{job.rep}")
        bs.append(dict(id=bid, vendor=job.vendor, uids=[u for u, _, _ in chunk], status="running",
                       projected_usd=round(ccost, 4), submitted=api.now()))
        job.save_batches(bs)
        print(f"  submitted batch {bid}: {len(chunk)} items, projected ${ccost:.3f}")
        chunk, ccost, size = [], 0.0, 0

    for r, c in zip(todo.to_dict("records"), costs):
        instr, user = job.prompt_pair(r)
        b = len(instr) + len(user) + 200
        if chunk and ((job.vendor == "gemini" and size + b > api.GEMINI_BATCH_MAX_BYTES)
                      or len(chunk) >= api.OPENAI_BATCH_MAX_ITEMS):
            flush()
        chunk.append((r["uid"], instr, user))
        ccost += c
        size += b
    flush()


def run_sync(job: RepeatJob, todo: pd.DataFrame, concurrency: int = 8) -> None:
    client = api.CLIENTS[job.vendor](job.model)
    stop = threading.Event()
    spent_here = [0.0]
    lock = threading.Lock()
    err: list[Exception] = []

    def one(r):
        if stop.is_set():
            return
        instr, user = job.prompt_pair(r)
        for attempt in range(api.REFUSAL_ATTEMPTS):
            try:
                text, usage, p_yes = client.sync(instr, user, f"{job.task}-{job.kind}-rep{job.rep}")
                c = job.record(r["uid"], text, usage, p_yes, "sync")
                with lock:
                    spent_here[0] += c
                return
            except api.OutOfCredit as e:
                stop.set()
                err.append(e)
                return
            except RuntimeError as e:
                code = api.refusal_code(str(e))
                if code:
                    with lock:
                        job.log_refusal(r["uid"], code, f"sync-attempt-{attempt + 1}")
                    continue
                raise

    try:
        with ThreadPoolExecutor(concurrency) as ex:
            list(ex.map(one, todo.to_dict("records")))
    finally:
        settled = job.settle_refusals()
        n = len(todo) - len(job.pending().merge(todo[["uid"]])) - settled
        api.ledger_add(job.vendor, job.s["model"], job.name, spent_here[0], max(n, 0), note="sync")
        print(f"  sync: {n} ok, {settled} refused, ${spent_here[0]:.4f}")
    if err:
        raise err[0]


def run_jobs(jobs: list[RepeatJob], wait: bool = False, concurrency: int = 8) -> int:
    worst = api.EXIT_DONE
    for job in jobs:
        print(f"[{job.name}] mode={job.mode}")
        try:
            if job.mode == "batch":
                collect_batches(job)
            in_batch = {u for b in job.batches() if b["status"] == "running" for u in b["uids"]}
            todo = job.pending()
            todo = todo[~todo.uid.isin(in_batch)].reset_index(drop=True)
            if len(todo):
                # Prefer empirical cost from the identical items when available; fallback to estimate_item_costs
                emp = empirical_item_costs(job.model, job.task, todo["uid"].tolist(), job.mode)
                est = sum(api.estimate_item_costs(job.model, job.task, job.kind, todo, job.mode))
                proj = emp if emp > 0 else est
                scale = (proj / est) if est > 0 else 1.0
                costs = [c * scale for c in api.estimate_item_costs(job.model, job.task, job.kind, todo, job.mode)]
                avail = available_usd(job.vendor)
                print(f"  pending {len(todo)} items, projected ${proj:.2f} (est ${est:.2f}); available ${avail:.2f}")
                if proj > avail:
                    print(api.recharge_msg(job.vendor, proj - avail))
                    return api.EXIT_RECHARGE
                if job.mode == "batch":
                    submit_batches(job, todo, costs)
                else:
                    run_sync(job, todo, concurrency=concurrency)
        except api.OutOfCredit as e:
            print(f"  STOPPED: {e}")
            return api.EXIT_RECHARGE

    if jobs and jobs[0].mode == "batch":
        while True:
            running = sum(collect_batches(j) for j in jobs)
            if not running or not wait:
                if running:
                    print(f"  {running} batch(es) still running; rerun with --wait to collect")
                    worst = max(worst, api.EXIT_WAITING)
                break
            print(f"  {running} batch(es) still running; polling again in 60 s")
            time.sleep(60)
    return worst


def _verdict_array(records: dict[str, dict], uids: list[str], refusal_label: int | None) -> tuple[np.ndarray, int, int, float]:
    verdicts, unparseable, refused, cost = [], 0, 0, 0.0
    for u in uids:
        r = records[u]
        cost += float(r.get("cost_usd", 0.0))
        is_ref = bool(r.get("refused")) and refusal_label is not None
        if is_ref:
            refused += 1
            verdicts.append(int(refusal_label))
        elif r.get("verdict") is None:
            unparseable += 1
            verdicts.append(0)
        else:
            verdicts.append(int(r["verdict"]))
    return np.array(verdicts, dtype=float), unparseable, refused, cost


def analyze_cmd() -> int:
    rows = []
    summary = {}
    for task in api.TASK_NAMES:
        refusal_label = api.TASKS["tasks"][task].get("refusal_label")
        for es in ("test", "shift_api"):
            sub = subsample_rows(task, es)
            uids = sub["uid"].tolist()
            y = sub["label"].to_numpy(int)
            for model in FRONTIER:
                ver = api.prompt(task)["version"]
                orig_path = ROOT / "preds" / "api" / task / f"{model}__{ver}_concise.jsonl"
                orig_recs = {json.loads(l)["uid"]: json.loads(l) for l in orig_path.read_text().splitlines() if l.strip()}
                v0, u0, rf0, _ = _verdict_array(orig_recs, uids, refusal_label)
                runs = [v0]
                unp_list = [u0]
                ref_list = [rf0]
                rep_cost = 0.0
                complete = True
                for rep in REPS:
                    c = RepeatCache(model, task, rep, "concise")
                    if not all(u in c.records for u in uids):
                        complete = False
                        continue
                    vr, ur, rfr, cr = _verdict_array(c.records, uids, refusal_label)
                    runs.append(vr)
                    unp_list.append(ur)
                    ref_list.append(rfr)
                    rep_cost += cr
                if not complete:
                    print(f"[pending] {task}/{es}/{model}: missing repeat records")
                    continue
                f1s = [float(M.f1_at(y, r, 0.5)) for r in runs]
                mean_f1 = float(np.mean(f1s))
                sd_ref = float(np.std(f1s, ddof=1))
                # Pairwise exact verdict agreement across the 3 runs on the 500 items
                agrees = [
                    float(np.mean(runs[i] == runs[j]))
                    for i in range(len(runs)) for j in range(i + 1, len(runs))
                ]
                flip_rate = float(1.0 - np.mean(agrees))
                # Also full-set Run 0 F1 for comparison
                full_set = "test" if es == "test" else "shift_api_concise"
                full_df = api.item_set(task, full_set)
                vf, _, _, _ = _verdict_array(orig_recs, full_df["uid"].tolist(), refusal_label)
                full_f1 = float(M.f1_at(full_df["label"].to_numpy(int), vf, 0.5))

                row = dict(
                    task=task, eval_set=es, model=model, n_sub=len(uids), n_runs=len(runs),
                    f1_full_run0=round(full_f1, 6),
                    f1_rep0=round(f1s[0], 6), f1_rep1=round(f1s[1], 6), f1_rep2=round(f1s[2], 6),
                    f1_mean_3runs=round(mean_f1, 6), sd_ref=round(sd_ref, 6),
                    pairwise_agree=round(float(np.mean(agrees)), 6),
                    pairwise_flip_rate=round(flip_rate, 6),
                    unparseable=",".join(map(str, unp_list)),
                    refused=",".join(map(str, ref_list)),
                    repeat_cost_usd=round(rep_cost, 4),
                )
                rows.append(row)
                summary.setdefault(task, {}).setdefault(es, {})[model] = dict(
                    f1_full_run0=round(full_f1, 4), f1_runs=[round(x, 4) for x in f1s],
                    mean=round(mean_f1, 4), sd_ref=round(sd_ref, 4),
                    pairwise_flip_rate=round(flip_rate, 4),
                )
    if not rows:
        print("No completed repeat sets to analyze yet.")
        return 1
    df = pd.DataFrame(rows)
    REV_OUT.mkdir(parents=True, exist_ok=True)
    out_csv = REV_OUT / "ref_repeat_sd.csv"
    df.to_csv(out_csv, index=False)
    print(df[["task", "eval_set", "model", "f1_full_run0", "f1_rep0", "f1_rep1", "f1_rep2", "sd_ref", "pairwise_flip_rate", "repeat_cost_usd"]].to_string(index=False))
    print(f"\n[write] {out_csv.relative_to(ROOT)}")

    num_path = REV_OUT / "revision_numbers.json"
    nums = json.loads(num_path.read_text()) if num_path.exists() else {}
    nums["ref_repeats"] = summary
    num_path.write_text(json.dumps(nums, indent=2))
    print(f"[write] {num_path.relative_to(ROOT)} (ref_repeats)")
    return 0


def main() -> int:
    api.load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--mode", default="batch", choices=["batch", "sync"])
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--vendor", choices=["openai", "gemini"])
    ap.add_argument("--rep", type=int, nargs="+", choices=[1, 2])
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    if not (args.estimate or args.run or args.analyze):
        ap.error("specify --estimate, --run, and/or --analyze")
    if args.estimate:
        return estimate_cmd()
    if args.run:
        models = [m for m in FRONTIER if not args.vendor or api.spec(m)["vendor"] == args.vendor]
        jobs = all_jobs(mode=args.mode, models=models, reps=args.rep)
        code = run_jobs(jobs, wait=args.wait, concurrency=args.concurrency)
        if code != api.EXIT_DONE:
            return code
    if args.analyze:
        return analyze_cmd()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
