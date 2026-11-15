"""Decoder inputs must end with the generation prompt, where P(Yes) is read, for every task and
prompt kind, including items longer than the task's token budget."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    except Exception as e:  # offline without a cached tokenizer
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.mark.parametrize("kind", ["concise", "detailed"])
@pytest.mark.parametrize("task,budget", [("t1_rag_hallucination", 64), ("t2_response_safety", 32),
                                         ("t3_implicit_toxicity", 16)])
def test_long_items_keep_generation_prompt(tok, task, budget, kind):
    from decoder_lora import PromptEncoder
    enc = PromptEncoder(tok, task, budget, kind)
    tail = enc._ids("", "")[-6:]
    long = "word " * 3000
    for a, b in [(long, long), (long, ""), ("short", long), ("short", "short")]:
        ids = enc(a, b)
        assert ids[-6:] == tail
        assert len(ids) <= enc.L + 64
