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


@pytest.mark.parametrize("kind", ["concise", "detailed", "none"])
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


@pytest.mark.parametrize("task", ["t1_rag_hallucination", "t2_response_safety", "t3_implicit_toxicity"])
def test_no_instruction_input_is_the_item_only(tok, task):
    import yaml
    from decoder_lora import ROOT, PromptEncoder
    p = yaml.safe_load((ROOT / "labels" / "prompts" / f"{task}.yaml").read_text())
    question = p["item_template"].strip().splitlines()[-1]
    full, bare = PromptEncoder(tok, task, 512, "concise"), PromptEncoder(tok, task, 512, "none")
    t_full, t_bare = full._text("AAA first text", "BBB second text"), bare._text("AAA first text", "BBB second text")
    assert "AAA first text\n\nBBB second text" in t_bare
    tag = p["item_template"].splitlines()[0]
    for s in (p["concise"].strip()[:60], question, tag):
        assert s in t_full
        assert s not in t_bare
    assert bare.overhead < full.overhead
