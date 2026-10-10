# kvkk-rewriter roadmap

Goal: distill the Gemini query-rewrite step of [kvkk-rag](https://github.com/EmirhanHasirci11/kvkk-rag) into a
small model trained with QLoRA, and measure what it does to retrieval, latency and cost.

kvkk-rag is a git submodule (`external/kvkk-rag`) pinned to `f03c563`; the retriever, corpus, eval sets,
cached Gemini rewrites and eval code all come from it.

## Systems
1. **no-rewrite**: the question alone (`orig`).
2. **Gemini teacher**: kvkk-rag's cached rewrites (`gemini-3.8-flash`, its rewrite prompt, default thinking).
3. **small model, zero-shot**: the shortlisted model with the teacher's prompt, no training.
4. **small model + QLoRA**: trained on teacher rewrites of generated questions.

## Metrics
- R@1 / R@3 / R@5 / R@10 for `rw-only` (rewrite alone) and `fuse` (RRF over question + rewrite), with
  kvkk-rag's eval code: e5 + BM25 (bm25-p5) hybrid, 192/48, depth 50, **no reranker**.
- Rewrite latency per question (p50 / p95), on the hardware it ran on.
- Cost per question: Gemini from token counts (`results/llm_usage.csv`); small model from Colab compute units
  per question (`results/colab_usage.csv`).

## Eval sets
- **dev**: kvkk-rag `eval/golden.jsonl` (50 q). Used for every choice (model, prompt, hyperparameters).
- **held-out**: kvkk-rag `eval/test_v2.jsonl` (20 q). Run once, at the end, with everything frozen.
- No training question comes from a chunk that holds dev or held-out evidence or from the same article /
  section, and every training question passes a leakage filter against all eval questions (`docs/DATA_PLAN.md`).

## Rules (same as kvkk-rag)
- A rule (adoption threshold, filter, frozen config) is written down before the run it applies to.
- Never change settings after looking at a held-out result; the held-out set is run once.
- Save each run as a new versioned file in `results/`, never overwrite old ones.
- When code moves, old numbers must reproduce exactly before new runs (`scripts/sanity_check.py`).
- One question is noise: with 50 questions, gains of 2-5 points are not claimed as wins.
- `.env` is never committed; check `git status` before every commit.

## Compute
- Local CPU: data generation (API calls), leakage filter, retrieval eval.
- Colab: training and small-model inference. Every Colab run appends a row to `results/colab_usage.csv`
  (date, notebook, GPU type, minutes, compute units, purpose).

## Spending gates (each needs the user's explicit approval)
1. Scaffold, sanity check, model shortlist, data plan: **$0**. Done.
2. Pilot: 10 questions + 10 rewrites; check question quality, leakage filter and real cost: **~$0.05**.
3. Full data generation (~1000-1500 pairs): **~$4-6**. Project LLM budget: $6, enforced in `scripts/llm.py`.
4. Colab: zero-shot baseline first, then QLoRA.
