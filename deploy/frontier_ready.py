"""Write preds/api/{task}/frontier_teacher.done once the frontier teacher has a final record (an answer, or
a content-filter refusal settled per D25) for every chain item used by frontier_teacher /
mix*_frontier_teacher cells (N <= 5k, all seeds) and no batch holding unsettled items is still running. Workers only start those cells after the marker exists (queue/worker.labels_ready).

  python deploy/frontier_ready.py          # prints one line per task; exit 0
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "data"))
import yaml  # noqa: E402
from common import RunSpec, _labelled_chain, chain  # noqa: E402
from splits import seeds_for  # noqa: E402

SWEEP = yaml.safe_load((ROOT / "configs" / "sweep.yaml").read_text())
N_MAX = SWEEP["frontier_teacher_n_max"]


def check(task: str) -> str:
    marker = ROOT / "preds" / "api" / task / "frontier_teacher.done"
    if marker.exists():
        return "ready (marker exists)"
    seeds = sorted(set(seeds_for(100)) | set(seeds_for(N_MAX)))
    uids = set().union(*(set(chain(task, s, N_MAX).uid) for s in seeds))
    running = 0
    for f in (ROOT / "preds" / "api" / task).glob("gpt61_sol__*_detailed.batches.json"):
        cache = Path(str(f).replace(".batches.json", ".jsonl"))
        final = {json.loads(ln)["uid"] for ln in cache.read_text().splitlines() if ln.strip()} if cache.exists() else set()
        # a batch still running only matters if it holds needed items without a final record (answer or refusal, D25)
        running += sum(1 for b in json.loads(f.read_text())
                       if b["status"] == "running" and (uids & set(b["uids"])) - final)
    if running:
        return f"waiting: {running} teacher batch(es) running"
    try:
        for s in seeds:
            _labelled_chain(RunSpec(family="classical", model="tfidf_lr", task=task, n=N_MAX, seed=s,
                                    labels="frontier_teacher"))
    except (RuntimeError, FileNotFoundError) as e:
        return f"waiting: {e}"
    marker.write_text(f"{len(uids)}\n")
    return "ready (marker written)"


if __name__ == "__main__":
    for t in SWEEP["task_order"]:
        print(f"{t}: {check(t)}")
