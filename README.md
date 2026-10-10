# kvkk-rewriter

Distilling the Gemini query-rewrite step of [kvkk-rag](https://github.com/EmirhanHasirci11/kvkk-rag) into a
small model trained with QLoRA, and measuring its effect on retrieval, latency and cost.

Status: scaffold, sanity check, model shortlist and data plan (spending gate 1, $0). Plan and rules:
`docs/ROADMAP.md`; candidates: `docs/MODELS.md`; data generation: `docs/DATA_PLAN.md`.

## Setup

```
git clone --recurse-submodules https://github.com/EmirhanHasirci11/kvkk-rewriter.git
cd kvkk-rewriter
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

kvkk-rag is pinned as a submodule in `external/kvkk-rag`; its retriever, corpus, eval sets, cached Gemini
rewrites and eval code are used as they are.

LLM calls (data generation only) need `GEMINI_API_KEY` in `.env` at the repo root; it is git-ignored and is the
only place the key is read from. Every call is logged in `results/llm_usage.csv`; `scripts/llm.py` refuses calls
once this project's total reaches $6.

```
python -m pytest tests -q                   # no LLM call
python scripts/sanity_check.py              # kvkk-rag teacher numbers from the cached rewrites, no LLM call
```

## Sanity check

`scripts/sanity_check.py` recomputes kvkk-rag's retrieval with the cached Gemini rewrites (no reranker,
bm25-p5) and requires every question and every k to equal the saved kvkk-rag files. It passes: 70 questions
x 3 systems x 4 k identical. From `results/sanity_v1_summary.csv`:

| Set | System | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| dev (50 q) | orig | 0.38 | 0.56 | 0.64 | 0.73 |
| | rw-only | 0.37 | 0.67 | 0.75 | 0.78 |
| | fuse | 0.47 | 0.67 | 0.71 | 0.81 |
| held-out (20 q) | orig | 0.15 | 0.30 | 0.35 | 0.65 |
| | rw-only | 0.50 | 0.65 | 0.70 | 0.75 |
| | fuse | 0.20 | 0.55 | 0.60 | 0.70 |
