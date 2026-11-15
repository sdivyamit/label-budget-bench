"""GPU worker: claims the next unfinished job from queue/jobs.yaml and runs it, forever.

  CUDA_VISIBLE_DEVICES=0 python queue/worker.py                    # all GPU jobs in order
  python queue/worker.py --families classical                      # CPU-only jobs
  python queue/worker.py --stage gate_a --once                     # just Gate A
  python queue/worker.py --shard 1/3                               # machine 1 of 3 (no shared disk)
  python queue/worker.py --shard 5/8 --reverse                     # help machine 5 from the end of its list
  python queue/worker.py --shard 5/8 --count                       # print remaining work as JSON, then exit
  touch queue/STOP                                                 # finish current job, then exit

Claiming uses an exclusive lock file per job with a heartbeat; a lock whose heartbeat is older
than 10 minutes (crashed or deleted machine) is taken over. A killed job resumes from its last
checkpoint when claimed again. A job failing 3 times is parked in queue/failed/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "queue"
LOCKS, FAILED, LOGS = Q / "locks", Q / "failed", ROOT / "logs"
STALE_S, MAX_FAIL = 600, 3


def done(j) -> bool:
    return (ROOT / "preds" / "runs" / j["id"] / "DONE").exists()


def failures(j) -> int:
    f = FAILED / f"{j['id']}.json"
    return json.loads(f.read_text())["count"] if f.exists() else 0


def labels_ready(j) -> bool:
    """Teacher-label jobs wait (instead of failing) until that teacher's file has been copied here."""
    if "open_teacher" in j["labels"]:
        return (ROOT / "preds" / "teacher" / "qwen35_9b" / f"{j['task']}.done").exists()
    if "frontier_teacher" in j["labels"]:
        return (ROOT / "preds" / "api" / j["task"] / "frontier_teacher.done").exists()   # written once the sub-pool is covered
    return True


def claim(j) -> Path | None:
    lock = LOCKS / f"{j['id']}.lock"
    if lock.exists() and time.time() - lock.stat().st_mtime > STALE_S:
        lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), ts=time.time(),
                                 gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))).encode())
    os.close(fd)
    return lock


def heartbeat(lock: Path, stop: threading.Event):
    while not stop.wait(60):
        try:
            os.utime(lock)
        except FileNotFoundError:
            return


def run(j, extra: list[str]) -> int:
    LOGS.mkdir(exist_ok=True)
    cmd = [sys.executable, str(ROOT / "train" / "run.py"), "--family", j["family"], "--model", j["model"],
           "--task", j["task"], "--n", str(j["n"]), "--seed", str(j["seed"]), "--labels", j["labels"], *extra]
    if j.get("hp"):
        cmd += ["--hp", json.dumps(j["hp"]), "--tag", j["tag"]]
    with (LOGS / f"{j['id']}.log").open("a") as log:
        log.write(f"\n=== {time.strftime('%F %T')} {socket.gethostname()} {' '.join(cmd)}\n")
        log.flush()
        return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="encoder,decoder")
    ap.add_argument("--stage", help="regex on job stage, e.g. gate_a or 0-t2")
    ap.add_argument("--filter", help="regex on job id")
    ap.add_argument("--shard", help="i/k: only jobs whose id hashes to i mod k (machines without a shared disk)")
    ap.add_argument("--once", action="store_true", help="exit when no claimable job is left")
    ap.add_argument("--extra", default="", help="extra args for train/run.py (smoke tests)")
    ap.add_argument("--reverse", action="store_true", help="take jobs from the end (work stealing)")
    ap.add_argument("--count", action="store_true", help="print claimable / waiting / running counts and exit")
    a = ap.parse_args()
    for d in (LOCKS, FAILED):
        d.mkdir(parents=True, exist_ok=True)
    fams = set(a.families.split(","))
    shard = tuple(map(int, a.shard.split("/"))) if a.shard else None
    while True:
        if (Q / "STOP").exists() and not a.count:
            print("STOP file present; exiting")
            return 0
        jobs = yaml.safe_load((Q / "jobs.yaml").read_text())
        mine = [j for j in jobs if j["family"] in fams and not done(j) and failures(j) < MAX_FAIL
                and (not a.stage or re.search(a.stage, j["stage"]))
                and (not a.filter or re.search(a.filter, j["id"]))
                and (not shard or zlib.crc32(j["id"].encode()) % shard[1] == shard[0] % shard[1])]
        todo = [j for j in mine if labels_ready(j)]
        if a.reverse:
            todo.reverse()
        if a.count:
            locked = {p.stem for p in LOCKS.glob("*.lock") if time.time() - p.stat().st_mtime <= STALE_S}
            print(json.dumps(dict(claimable=sum(j["id"] not in locked for j in todo),
                                  running=sum(j["id"] in locked for j in todo),
                                  waiting_labels=len(mine) - len(todo))))
            return 0
        picked = None
        for j in todo:
            lock = claim(j)
            if lock:
                picked = (j, lock)
                break
        if not picked:
            if a.once:
                print("no claimable jobs left")
                return 0
            time.sleep(60)
            continue
        j, lock = picked
        stop = threading.Event()
        threading.Thread(target=heartbeat, args=(lock, stop), daemon=True).start()
        t0 = time.time()
        print(f"[{time.strftime('%T')}] start {j['id']}", flush=True)
        code = run(j, a.extra.split())
        stop.set()
        if code == 0 and done(j):
            print(f"[{time.strftime('%T')}] done  {j['id']} in {(time.time() - t0) / 60:.1f} min", flush=True)
        else:
            f = FAILED / f"{j['id']}.json"
            f.write_text(json.dumps(dict(count=failures(j) + 1, last_code=code, ts=time.time())))
            print(f"[{time.strftime('%T')}] FAIL  {j['id']} (exit {code}); see logs/{j['id']}.log", flush=True)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
