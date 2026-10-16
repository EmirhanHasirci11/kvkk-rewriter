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

## Gate 6: students in kvkk-rag's full retrieval pipeline (rules written 2026-10-15, before any run)
- Question: does kvkk-rag lose retrieval if its Gemini rewrite is replaced by the QLoRA Llama-3.1-8B (the user's
  pick after gate 5)? Gates 4-5 scored retrieval without the reranker; kvkk-rag's final pipeline has one.
- Pipeline: kvkk-rag's frozen final retrieval (`FINAL` in its `scripts/run_heldout.py`): question + rewrite, e5 +
  BM25 hybrid with bm25-snow, RRF, top-30, then `bge-reranker-v2-m3` scoring the original question against each
  candidate; recall on the reranked list. Also recorded: R@30 of the candidate pool before reranking.
- Systems: no rewrite (the question alone through the same pipeline), Gemini teacher (kvkk-rag's cached rewrites),
  and the three gate 4b students (Llama-3.1-8B, plus Kumru-2B and Qwen3-4B for reference). Rewrites as already
  generated in gates 4b and 5; nothing is regenerated, so no Colab run.
- Sets: dev (50 q) and held-out (20 q). The held-out rewrites are gate 5's frozen outputs and kvkk-rag's pipeline
  is frozen, so this scores fixed outputs with fixed settings; nothing is chosen from the held-out numbers.
- Check: the teacher rows must reproduce kvkk-rag's saved per-question numbers exactly (dev: `pipe-snow` in
  `results/stemming_v1_per_question.csv`, R@5 0.86; held-out: `fuse+bge+snow (final)` in
  `results/heldout_v2_retrieval_per_question.csv`, R@5 0.55). Otherwise the run stops.
- Decision: Llama-3.1-8B can replace the teacher in kvkk-rag's retrieval if its dev pipeline R@5 is at most 0.04
  (two questions) under the teacher's. The held-out numbers are reported next to it and do not change the decision.
- Script: `scripts/eval_pipeline.py`, local (the reranker may run on the local GPU; the check above guards it).

## Gate 7: Llama-3.1-8B v2 (rules written 2026-10-16, before any v2 data or run)
- Why: in gate 6 the v1 Llama misses legal terms the reranked pipeline needs. Decided with the user: Llama-3.1-8B
  only, +500 training pairs, the best-validation epoch instead of the last, and a GGUF Q6_K copy that runs on the
  user's RTX 3080 (Q8_0 does not fit 10 GB next to the desktop).
- New test set `eval/test_v3.jsonl` (20 q, q071-q090), because held-out `test_v2` is spent. Topics from corpus
  sections whose chunks no v1 training pair comes from (6 Kanun, 4 security guide, 4 special-category guide,
  3 communiqué, 3 regulation); questions in everyday user wording, without the legal terms of the evidence.
  Every evidence sentence is verbatim in the corpus (`check_golden.py` passes), no id or question repeats an eval
  set. Committed before any v2 data; scored once, at the end. Dev stays the set for all checks.
- Exclusion: test_v3 joins dev and held-out in the chunk rules (8-word evidence overlap, section labels, topic
  threshold 0.8846 unchanged) and in the leakage filter. v1 pairs are kept only if they still pass under the
  extended rules: 95 pairs from 20 chunks drop out, plus every v1 question the leakage filter flags against
  test_v3 (one known before the run: g35-006_3, cosine 0.935 to q075).
- New data: about 500 pairs from usable chunks that no v1 pair comes from, with the v1 pipeline unchanged (same
  generation order rule and seed, prompt v2, 5 questions per chunk, `gemini-3.8-flash`, teacher = kvkk-rag's
  rewrite step), into new versioned files. The validation split rule is unchanged (5% of the new chunks); v1
  val pairs stay val. LLM budget stays $6: if the guard stops the run early, v2 trains on what exists.
- Teacher reference on test_v3: kvkk-rag's rewrite of the 20 questions (about $0.07), made before the data
  generation so the budget guard cannot block it. Not scored until the end.
- Training (Colab L4): gate 4b's QLoRA configuration unchanged except that an adapter is saved after each of the
  3 epochs and the one with the lowest validation loss is used (val split, never dev). Merged into the bf16 base.
- Quantization (Colab): merged bf16 -> GGUF f16 (llama.cpp `convert_hf_to_gguf.py`) -> Q6_K (`llama-quantize`),
  saved to Drive. Check on dev, run on the local RTX 3080 with llama.cpp (the HF chat template as the prompt
  text, greedy, `max_new_tokens` 256, whitespace normalization only): Q6_K is accepted if its dev rw R@5 (gate 4b
  scoring) and dev full-pipeline R@5 (gate 6 scoring) are each within 0.02 (one question) of bf16 v2; the share of
  rewrites identical to bf16 is reported. If not accepted, Q8_0 with part of the layers on the CPU, as agreed.
- Decision, on dev with the accepted model: v2 can replace the teacher in kvkk-rag's retrieval if its dev
  full-pipeline R@5 is at most 0.04 (two questions) under the teacher's (0.86), the same rule as gate 6.
- Final, once, everything frozen: test_v3 through gate 4b scoring (rw, fuse) and gate 6 scoring (full pipeline)
  for no rewrite, the teacher, v1 (bf16, the gate 4b adapter, rewrites made on Colab in the v2 session) and v2
  (the accepted local model). Reported as they come out; one test_v3 question is 5 points, differences under
  10 points are not claimed.
- Data done 2026-10-16: teacher test_v3 rewrites $0.071; 100 new chunks -> 437 questions, 7 dropped by the leakage
  filter, 430 teacher rewrites; project LLM total $5.76 of $6 (the budget ended the run at 430 instead of 500).
  `data/pairs_v2.jsonl`: 1,347 pairs (1,284 train / 63 val) from v1 + v2; dropped 100 v1 questions whose chunk is
  excluded under split v3, 6 flagged against test_v3 (all close to q075), 10 leaks from generation.
- Colab done 2026-10-16 (`notebooks/04_qlora_v2.ipynb`, commit 96f3afc, log `results/colab_qlora_v2.log`): the
  runtime disconnected once during epoch 1 and training restarted from the start; val loss 1.657 -> 0.865 / 0.798 /
  0.870, so the epoch 2 adapter is used (`results/train_record_qlora_v2_llama3.1-8b.json`); Q6_K on Drive, 6.60 GB.
- v2 bf16 on dev: gate 4b scoring rw R@5 0.69, fuse 0.73 (v1 0.61 / 0.69, teacher 0.75 / 0.71); gate 6 full
  pipeline R@5 0.84 (v1 0.80, teacher 0.86, no rewrite 0.81), lost only q014 against the teacher
  (`results/pipeline_v2_dev_*`). `eval_pipeline.py` with its default arguments still reproduces
  `results/pipeline_v1_*` byte for byte.
- Before the final run (rules fixed now): test_v3 is scored by `eval_rewrites.py --set test_v3` with the held-out
  fused depth (30), and by `eval_pipeline.py --sets test_v3` (kvkk-rag has no saved numbers for it, so there is no
  reproduction check; the teacher rewrites are `results/rewrites_teacher_test_v3_v1.jsonl`).
- Local run: `scripts/gguf_rewrites.py` with the prebuilt llama.cpp b11696 (CUDA 13.4) in `tools/` (not in git);
  the prompt goes in as the HF tokenizer's token ids of the chat template, so the input equals the bf16 run's.
