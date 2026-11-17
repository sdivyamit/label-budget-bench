"""Do the 10 worked examples in the frozen detailed prompts (labels/prompts/t{1,2,3}_*.yaml) appear in the data?

  python analysis/check_prompt_overlap.py [--threshold 0.8]

For each task, every quoted example string (context, text, prompt, response, post) is compared against

  (a) all splits of the processed task data (pool, val, test, shift; text_a and text_b), and
  (b) the raw source corpora of the benchmark, including items removed during processing:
        T1 RAGTruth responses and source documents, LLM-AggreFact documents and claims;
        T2 BeaverTails (train and test) prompts and responses, WildGuardTest prompts and responses;
        T3 ToxiGen annotated train and test posts, Implicit Hate Corpus posts,

by two methods:

  exact   normalized (lower-case, alphanumerics only, whitespace collapsed) equality or containment of the example
          in a corpus text, or of a corpus text (>= 4 words) in the example;
  near    maximum TF-IDF cosine similarity (word 1-2-grams) to any corpus text, with the closest text printed and
          a difflib character ratio on the 5 closest texts.

An example is flagged when it matches exactly or when cosine >= --threshold. Lexical matching cannot exclude a
paraphrase with no word overlap. Writes results/revision/prompt_overlap_check.csv; exit status 1 if anything is
flagged. Requires the raw data under data/raw/ and data/processed/.
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import json
import re
from pathlib import Path

import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROMPTS = ROOT / "labels" / "prompts"
OUT = ROOT / "results" / "revision" / "prompt_overlap_check.csv"
TASKS = ["t1_rag_hallucination", "t2_response_safety", "t3_implicit_toxicity"]
MIN_WORDS = 3  # shorter quoted strings are field names or one-word answers of the structured-data example, not examples
FIELD = re.compile(r"^(Example \d+:?|Context[^:]*:|Text[^:]*:|Prompt:|Response:|Question[^:]*:|Answer:)\s*")


CORPUS_SIZES: dict[str, int] = {}


def norm(s) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())


def examples(task: str) -> list[tuple[int, str]]:
    """(example number, quoted string) for every quoted field of the '## Worked examples' section."""
    text = yaml.safe_load((PROMPTS / f"{task}.yaml").read_text())["detailed"]
    sec = text.split("## Worked examples", 1)[1].split("## Output", 1)[0]
    out, num, field, buf = [], 0, None, []

    def flush():
        if field is not None and field != "Answer:":
            out.extend((num, q) for q in re.findall(r'"([^"]{3,})"', " ".join(buf)) if len(q.split()) >= MIN_WORDS)

    for line in sec.splitlines():
        s = line.strip()
        m = FIELD.match(s)
        if m:
            flush()
            field, buf = m.group(1), [s[m.end():]]
            if field.startswith("Example"):
                num = int(re.search(r"\d+", field).group())
                field = "Text:" if s[m.end():].startswith('"') else "Example"  # T3 examples are inline posts
        elif s:
            buf.append(s)
    flush()
    seen, uniq = set(), []
    for n, q in out:
        if (n, q) not in seen:
            seen.add((n, q))
            uniq.append((n, q))
    return uniq


def corpora(task: str) -> dict[str, list[str]]:
    p = pd.read_parquet(PROC / f"{task}.parquet")
    c = {"processed": [t for col in ("text_a", "text_b") for t in p[col].dropna().astype(str)]}
    if task.startswith("t1"):
        r = pd.read_json(RAW / "ragtruth" / "response.jsonl", lines=True)
        s = pd.read_json(RAW / "ragtruth" / "source_info.jsonl", lines=True)
        c["raw RAGTruth"] = list(r.response.astype(str)) + [json.dumps(x) if not isinstance(x, str) else x for x in s.source_info]
        a = pd.concat([pd.read_parquet(f) for f in sorted((RAW / "llm_aggrefact" / "data").glob("*.parquet"))])
        c["raw LLM-AggreFact"] = list(a.doc.astype(str)) + list(a.claim.astype(str))
    elif task.startswith("t2"):
        bt = []
        for f in ("train", "test"):
            with gzip.open(RAW / "beavertails" / f"{f}.jsonl.gz", "rt") as fh:
                bt += [x for l in fh for x in (json.loads(l)["prompt"], json.loads(l)["response"])]
        c["raw BeaverTails"] = bt
        w = pd.read_parquet(RAW / "wildguardtest" / "test" / "wildguard_test.parquet")
        c["raw WildGuardTest"] = list(w.prompt.astype(str)) + list(w.response.astype(str))
    else:
        c["raw ToxiGen"] = [t for f in ("annotated_train", "annotated_test")
                            for t in pd.read_parquet(RAW / "toxigen" / f"{f}.parquet").text.astype(str)]
        c["raw Implicit Hate"] = list(pd.read_csv(RAW / "implicit_hate" / "implicit_hate.csv").post.astype(str))
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.8)
    a = ap.parse_args()
    rows = []
    for task in TASKS:
        ex = examples(task)
        corp = corpora(task)
        texts, where, normed_by = [], [], {}
        for name, lst in corp.items():
            normed_by[name] = [norm(t) for t in lst]
            texts += normed_by[name]
            where += [name] * len(normed_by[name])
        long_by = {name: {t for t in v if len(t.split()) >= 4} for name, v in normed_by.items()}
        big = {name: " " + " ".join(v) + " " for name, v in normed_by.items()}
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, dtype=np.float32)
        x = vec.fit_transform(list(texts) + [norm(q) for _, q in ex])
        xc, xe = x[: len(texts)], x[len(texts):]
        sims = (xe @ xc.T).toarray()
        CORPUS_SIZES[task[:2]] = len(texts)
        print(f"\n{task}: {len(ex)} example strings vs {len(texts):,} corpus texts")
        for i, (n, q) in enumerate(ex):
            nq = norm(q)
            exact_in = [name for name, b in big.items() if len(nq.split()) >= 4 and f" {nq} " in b]
            padded = f" {nq} "
            contains = [name for name, v in long_by.items() if any(f" {t} " in padded for t in v)]
            top = np.argsort(-sims[i])[:5]
            ratio = max(difflib.SequenceMatcher(None, nq, texts[j][:4000]).ratio() for j in top)
            best = int(top[0])
            flagged = bool(exact_in or contains or sims[i, best] >= a.threshold)
            review = sims[i, best] >= 0.7 and not flagged  # printed for manual review
            rows.append(dict(task=task, example=n, text=q[:90], exact_in=";".join(exact_in), contains=";".join(contains),
                             max_cosine=round(float(sims[i, best]), 3), difflib_top5=round(ratio, 3),
                             closest_source=where[best], closest=texts[best][:90], flagged=flagged, manual_review=review))
            print(f"  ex{n:>2} cos={sims[i, best]:.2f} diff={ratio:.2f} exact={bool(exact_in)} contains={bool(contains)}  "
                  f"{q[:60]!r} ~ {texts[best][:60]!r}")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n{len(df)} strings checked; flagged: {int(df.flagged.sum())}; manual review (cos >= 0.7): {int(df.manual_review.sum())}; max cosine per task:")
    print(df.groupby("task").max_cosine.max().to_string())
    nums = {}
    for task, g in df.groupby("task"):
        t = task[:2]
        nums[f"{t}/strings"] = int(len(g))
        nums[f"{t}/max_cosine"] = round(float(g.max_cosine.max()), 3)
        nums[f"{t}/flagged"] = int(g.flagged.sum())
        nums[f"{t}/review"] = int(g.manual_review.sum())
        nums[f"{t}/exact"] = int(((g.exact_in.fillna("") != "") | (g.contains.fillna("") != "")).sum())
    nums["strings"] = int(len(df))
    nums["threshold"] = a.threshold
    nums["corpus_texts"] = {t: n for t, n in CORPUS_SIZES.items()}
    import rev_common as rc
    rc.save_numbers("prompt_overlap", nums)
    return int(df.flagged.any())


if __name__ == "__main__":
    raise SystemExit(main())
