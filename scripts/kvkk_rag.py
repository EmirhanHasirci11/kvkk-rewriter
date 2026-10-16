"""Paths into the pinned kvkk-rag submodule, and its scripts/ on sys.path.

Import this before any kvkk-rag module: `import kvkk_rag` then `from retriever import HybridRetriever`.
The retriever, corpus, eval sets, cached teacher rewrites and eval code all come from the submodule,
so the numbers here are computed by exactly the code that produced the kvkk-rag README.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAG = ROOT / "external/kvkk-rag"
RESULTS = ROOT / "results"

if not (RAG / "scripts/retriever.py").exists():
    raise RuntimeError(f"{RAG} is empty; run: git submodule update --init")
# appended, not prepended: this project's own scripts (llm.py in particular) win on a name clash
if str(RAG / "scripts") not in sys.path:
    sys.path.append(str(RAG / "scripts"))

DEV = RAG / "eval/golden.jsonl"           # 50 questions, used while building kvkk-rag
HELDOUT = RAG / "eval/test_v2.jsonl"      # 20 questions; in this project, one run only
EVAL_SETS = (DEV, HELDOUT, RAG / "eval/test_v1.jsonl", RAG / "eval/abstention.jsonl", RAG / "eval/abstention_v2.jsonl")
TEST_V3 = ROOT / "eval/test_v3.jsonl"     # 20 questions, this project's own (gate 7); one run only
TEACHER_REWRITES = {"dev": RAG / "results/rewrites_v1.jsonl", "heldout": RAG / "results/rewrites_heldout_v2.jsonl"}
TEACHER_TEST_V3 = RESULTS / "rewrites_teacher_test_v3_v1.jsonl"    # made in this project (gate 7), same teacher


def corpus_texts() -> dict[str, str]:
    # same order and reading as kvkk-rag (sorted file names, read_text with universal newlines)
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted((RAG / "corpus/text").glob("*.txt"))}
