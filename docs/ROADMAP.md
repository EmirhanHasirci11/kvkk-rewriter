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
5. Held-out: inference only on a Colab L4 (the adapters from gate 4b, no training): ~1 CU expected.

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

## Gate 4b: QLoRA (rules written 2026-10-13, before any run)
- Models (user's choice, all three from 4a): Qwen3-4B-Instruct-2507, Kumru-2B, Llama-3.1-8B-Instruct. Colab L4.
- Data: `data/pairs_v1.jsonl`, train split (966) for training, val split (52) for validation loss only.
  Each example is the inference prompt as is (kvkk-rag's `SYSTEM` + the question, the model's chat template)
  followed by the teacher rewrite and the template's end-of-turn token; loss on the rewrite tokens only.
- Base in 4-bit NF4 (double quantization, bf16 compute); LoRA r 16, alpha 32, dropout 0.05, on all linear
  layers except the LM head. AdamW, lr 2e-4, cosine with 3% warmup, 3 epochs, batch 16 (8 x 2 accumulation),
  max length 512 tokens (the longest example is 445 Qwen tokens), gradient checkpointing, seed 42. One configuration for all three; nothing is tuned on dev.
- The adapter after the last epoch is used (no checkpoint picked by dev or val). Val loss per epoch is recorded.
- Inference: the adapter merged into the bf16 base, then exactly the gate 4a decoding (greedy,
  `max_new_tokens` 256, batch 1, whitespace normalization only), dev only, latency on the L4.
- Scored with `scripts/eval_rewrites.py`. Adapters are kept on the user's Google Drive, not in git.

## Gate 5: held-out (rules written 2026-10-15, before any run)
- Frozen: everything from gate 4b, nothing changed after looking at the dev results. Models: all three students
  (Kumru-2B, Qwen3-4B-Instruct-2507, Llama-3.1-8B-Instruct; their dev differences are within noise, so none is
  picked on dev). Adapters: the `qlora_v1` last-epoch adapters on the user's Drive, as trained; the notebook
  checks that each adapter's `train_record.json` equals `results/train_record_qlora_v1_<model>.json`.
- Hardware and decoding: exactly gate 4b's inference on a Colab L4 (adapter merged into the bf16 base, greedy,
  `max_new_tokens` 256, batch 1, whitespace normalization only, one warm-up call on a training question, latency
  of `generate()` per question). The local RTX 3080 (10 GB) is not used: Llama-3.1-8B in bf16 does not fit, and
  all three run on the hardware their dev numbers came from.
- Set: kvkk-rag `eval/test_v2.jsonl` (20 q), run once (`notebooks/03_heldout.ipynb`). If the runtime disconnects,
  only the models without an output file are run again; a finished model's rewrites are never regenerated.
- Scoring: `scripts/eval_rewrites.py --set heldout`: kvkk-rag's held-out retrieval settings (as its
  `run_heldout.py` and `scripts/sanity_check.py`: no reranker, bm25-p5, fused depth 30). Teacher and no-rewrite
  references: `results/eval_teacher_heldout_v1_summary.csv` (rw R@5 0.70, fuse R@5 0.60) and
  `results/sanity_v1_summary.csv` (orig R@5 0.35), computed before the run.
- The numbers are reported as they come out, whichever way they go; nothing is retuned or rerun afterwards.
  One held-out question is 5 points: differences under 10 points (two questions) are not claimed.
