"""Fetch raw datasets into data/raw/ and record provenance in data/raw/downloads.json.

Open sources download directly. Gated Hugging Face datasets need HF_TOKEN (after
accepting each dataset's terms on the Hub). The Implicit Hate Corpus requires the
authors' survey; place implicit-hate-corpus.zip in data/raw/implicit_hate/ by hand.

Usage: python data/download.py [--only SOURCE ...]
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
HF = "https://huggingface.co/datasets"

DIRECT = {
    "ragtruth": {
        "response.jsonl": "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl",
        "source_info.jsonl": "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl",
    },
    "beavertails": {
        "train.jsonl.gz": f"{HF}/PKU-Alignment/BeaverTails/resolve/main/round0/30k/train.jsonl.gz",
        "test.jsonl.gz": f"{HF}/PKU-Alignment/BeaverTails/resolve/main/round0/30k/test.jsonl.gz",
    },
    "toxigen": {
        "annotated_train.parquet": f"{HF}/toxigen/toxigen-data/resolve/main/annotated/train-00000-of-00001.parquet",
        "annotated_test.parquet": f"{HF}/toxigen/toxigen-data/resolve/main/annotated/test-00000-of-00001.parquet",
    },
}

GATED = {
    "llm_aggrefact": ("lytang/LLM-AggreFact", ["*.parquet", "*.csv", "*.json*"]),
    "wildguardtest": ("allenai/wildguardmix", ["test/*"]),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(log: dict, source: str, path: Path, url: str) -> None:
    log.setdefault(source, {})[str(path.relative_to(RAW))] = {
        "url": url,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "accessed": dt.date.today().isoformat(),
    }


def fetch_direct(source: str, log: dict) -> None:
    out = RAW / source
    out.mkdir(parents=True, exist_ok=True)
    for name, url in DIRECT[source].items():
        dest = out / name
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
        record(log, source, dest, url)
    print(f"[ok] {source}")


def fetch_gated(source: str, log: dict) -> None:
    repo, patterns = GATED[source]
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(f"[skip] {source}: set HF_TOKEN after accepting terms at {HF}/{repo}")
        return
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    try:
        local = Path(snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns,
                                       local_dir=RAW / source, token=token))
    except GatedRepoError:
        print(f"[skip] {source}: token's account has not accepted the terms; click 'Agree' at {HF}/{repo}")
        return
    for p in sorted(local.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and ".cache" not in p.parts:
            record(log, source, p, f"{HF}/{repo}")
    print(f"[ok] {source}")


def check_implicit_hate(log: dict) -> None:
    out = RAW / "implicit_hate"
    zpath = out / "implicit-hate-corpus.zip"
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    files = sorted(out.rglob("implicit_hate_v1_*.tsv"))
    if not files:
        print("[skip] implicit_hate: complete the survey at https://github.com/SALT-NLP/implicit-hate, "
              f"then save the zip (or its unzipped folder) in {out}")
        return
    for p in files:
        record(log, "implicit_hate", p, "https://github.com/SALT-NLP/implicit-hate (survey-gated Dropbox)")
    print("[ok] implicit_hate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    wanted = set(args.only or [*DIRECT, *GATED, "implicit_hate"])

    log_path = RAW / "downloads.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for s in DIRECT:
        if s in wanted:
            fetch_direct(s, log)
    for s in GATED:
        if s in wanted:
            fetch_gated(s, log)
    if "implicit_hate" in wanted:
        check_implicit_hate(log)
    log_path.write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
