"""Expand configs/sweep.yaml into queue/jobs.yaml (ordered: Gate A, then tasks in §8 order,
then classical < encoder < decoder < partial decoders, small N first).

  python queue/make_jobs.py            # writes queue/jobs.yaml and prints counts
"""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from splits import seeds_for  # noqa: E402

SWEEP = yaml.safe_load((ROOT / "configs" / "sweep.yaml").read_text())


def levels(task: str) -> list[int]:
    return [int(n) for n in json.loads((ROOT / "data" / "splits" / task / "subsets.json").read_text())["n_levels"]]


def job(family, model, task, n, seed, labels, stage, hp=None, tag=""):
    jid = f"{task}__{model}__{labels}__n{n}__s{seed}" + (f"__{tag}" if tag else "")
    j = dict(id=jid, stage=stage, family=family, model=model, task=task, n=n, seed=seed, labels=labels)
    if hp:
        j.update(hp=hp, tag=tag)
    return j


def hp_jobs() -> list[dict]:
    h = SWEEP["hp_selection"]
    out = []
    for family, g in h["grids"].items():
        for model in g["models"]:
            for lr in g["lr"]:
                for batch in g.get("batch", [None]):
                    hp = {"lr": lr} | ({"batch": batch} if batch else {})
                    tag = f"hp-lr{lr:g}" + (f"-b{batch}" if batch else "")
                    out.append(job(family, model, h["task"], h["n"], h["seed"], "gold", "hp", hp, tag))
    return out


def mix_cells(task: str, top: int):
    for m in SWEEP["mix_gold_anchors"]:
        for src, n in (("open_teacher", top), ("frontier_teacher", SWEEP["frontier_teacher_n_max"])):
            for s in seeds_for(n):
                yield n, s, f"mix{m}_{src}"


def ablation_jobs() -> list[dict]:
    family_of = {m: f.get("family", k) for k, f in SWEEP["families"].items() for m in f["models"]}
    out = []
    for a in SWEEP.get("ablations", []):
        for task in a.get("tasks", SWEEP["task_order"]):
            lv = levels(task)
            for model in a["models"]:
                for labels in a["labels"]:
                    if labels == "mixes":
                        cells = list(mix_cells(task, lv[-1]))
                    else:
                        ns = lv if a["n"] == "all" else [lv[-1] if n == "top" else n for n in a["n"]]
                        cells = [(n, s, labels) for n in ns for s in seeds_for(n)]
                    out += [job(family_of[model], model, task, n, s, lab, "ablation", a["hp"], a["tag"])
                            for n, s, lab in cells]
    return out


def expand() -> list[dict]:
    jobs = [job(**{k: v for k, v in g.items()}, stage="gate_a") for g in SWEEP["gate_a"]]
    jobs += hp_jobs()
    fam_rank = {"classical": 0, "encoder": 1, "decoder": 2, "decoder_partial": 3}
    for t_i, task in enumerate(SWEEP["task_order"]):
        lv = levels(task)
        top = lv[-1]
        for fam_key, f in sorted(SWEEP["families"].items(), key=lambda kv: fam_rank[kv[0]]):
            family = f.get("family", fam_key)
            ns = lv if f["n"] == "all" else [top if n == "top" else n for n in f["n"]]
            for model in f["models"]:
                for labels in f["labels"]:
                    if labels == "mixes":
                        jobs += [job(family, model, task, n, s, lab, f"{t_i}-{task}") for n, s, lab in mix_cells(task, top)]
                        continue
                    for n in ns:
                        if labels == "frontier_teacher" and n > SWEEP["frontier_teacher_n_max"]:
                            continue
                        for s in seeds_for(n):
                            jobs.append(job(family, model, task, n, s, labels, f"{t_i}-{task}"))
    jobs += ablation_jobs()
    seen, out = set(), []
    for j in jobs:
        if j["id"] not in seen:
            seen.add(j["id"])
            out.append(j)
    first = SWEEP.get("run_first", [])
    assert seen >= set(first), set(first) - seen
    head = [j for j in out if j["stage"] in ("gate_a", "hp")]
    return head + [j for j in out if j["id"] in first] + [j for j in out if j not in head and j["id"] not in first]


def main():
    jobs = expand()
    (ROOT / "queue" / "jobs.yaml").write_text(yaml.safe_dump(jobs, sort_keys=False))
    print(f"{len(jobs)} jobs -> queue/jobs.yaml")
    c = Counter((j["task"][:2], j["family"]) for j in jobs if j["stage"] != "gate_a")
    for k in sorted(c):
        print(f"  {k[0]} {k[1]:10s} {c[k]:5d}")


if __name__ == "__main__":
    main()
