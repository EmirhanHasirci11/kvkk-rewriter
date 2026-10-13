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
- Training questions come from the usable kvkk-rag chunks plus official KVKK guides that are not in the
  retrieval corpus (generation only; the retrieval corpus does not change).

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
   Done: $0.036 (+ $0.003 for the prompt v2 check).
3. Full data generation, 1000 pairs (decided 2026-10-10): **~$3.8** expected, $6.0 worst case.
   Project LLM budget: $6, enforced in `scripts/llm.py`. Done 2026-10-11: 1,018 pairs, $3.995 project total.
4. Colab: zero-shot baseline first, then QLoRA.

## Gate 4a: zero-shot baseline (rules written 2026-10-13, before any run)
- Models (user's choice): `Qwen/Qwen3-4B-Instruct-2507`, `vngrs-ai/Kumru-2B`, `meta-llama/Llama-3.1-8B-Instruct`
  (8B, above the 1-4B range, added by the user as a larger reference; gated, the user has access on their
  Hugging Face account; read in Colab from the `HF_TOKEN` secret).
- Hardware: Colab L4 (24 GB, bf16), paid compute units; each run is approved first and logged in
  `results/colab_usage.csv`.
- Prompt: kvkk-rag's rewrite prompt unchanged (`SYSTEM` in `external/kvkk-rag/scripts/rewrite.py`) as the system
  message, the question as the user message, each model's own chat template. No examples.
- Decoding: bf16 weights (no quantization), greedy, `max_new_tokens` 256, batch 1. The output is
  whitespace-normalized as in kvkk-rag's `rewrite()` and nothing else: no stripping of preambles or quotes.
  Outputs that hit the token limit are flagged (`truncated`), not dropped.
- Set: dev only (50 q). The held-out set is not touched in gate 4.
- Latency: wall clock of `generate()` per question on the L4, after one warm-up call on a training question.
- Retrieval: `scripts/eval_rewrites.py`, the same code and settings as the dev part of the sanity check
  (fused depth 10), on local CPU.
- This is a baseline only; nothing is adopted from it. Which models go on to QLoRA is decided with the user
  after it.
