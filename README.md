# kvkk-rewriter

Distilling the Gemini query-rewrite step of [kvkk-rag](https://github.com/EmirhanHasirci11/kvkk-rag) into a
small model trained with QLoRA, and measuring its effect on retrieval, latency and cost.

Status: gates 1-3 done. Training data: 1,018 (question, Gemini teacher rewrite) pairs in `data/pairs_v1.jsonl`
(966 train / 52 val), from kvkk-rag chunks and generation-only KVKK guides; LLM spend $3.995. Gates 4a (zero-shot
baseline), 4b (QLoRA, all three models) and 5 (held-out, run once) done on a Colab L4, see below. Plan and rules:
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

# generation-only source texts and the frozen chunk split (no LLM call)
python scripts/fetch_gen_corpus.py          # 35 KVKK guide PDFs -> gen_corpus/raw (git-ignored)
python scripts/extract_gen_corpus.py        # -> gen_corpus/text
python scripts/split_chunks.py --tag v2     # -> data/chunks_<tag>.jsonl (v1 is committed; never overwritten)
```

Student rewrites run on Colab (`notebooks/`, which clone this repo); their output files are scored locally:

```
python scripts/eval_rewrites.py results/rewrites_<name>.jsonl   # dev retrieval, no LLM call, CPU is enough
python scripts/eval_rewrites.py results/rewrites_<name>.jsonl --set heldout   # held-out (gate 5 only)
python scripts/cost.py                      # cost and latency, from the logged runs
```

`scripts/eval_rewrites.py` on kvkk-rag's cached teacher rewrites gives the sanity-check numbers exactly
(`results/eval_teacher_dev_v1_summary.csv`, `results/eval_teacher_heldout_v1_summary.csv`).

## Zero-shot baseline (gate 4a, dev, 50 q)

kvkk-rag's rewrite prompt, each model's chat template, bf16, greedy, on a Colab L4 (`notebooks/01_zeroshot.ipynb`,
`results/eval_zs_*_dev_v1_summary.csv`). Latency is per question on the L4; the teacher's is not comparable
(API, with thinking).

| System | rw R@5 | rw R@10 | fuse R@5 | fuse R@10 | latency p50 / p95 |
|---|---|---|---|---|---|
| no rewrite (orig) | 0.64 | 0.73 | | | |
| Gemini teacher | 0.75 | 0.78 | 0.71 | 0.81 | |
| Qwen3-4B-Instruct-2507 | 0.42 | 0.72 | 0.67 | 0.77 | 4.6 / 7.5 s |
| Kumru-2B | 0.44 | 0.60 | 0.55 | 0.67 | 0.7 / 2.2 s |
| Llama-3.1-8B-Instruct | 0.49 | 0.58 | 0.59 | 0.69 | 3.2 / 7.8 s |

Zero-shot, all three break the prompt's rules: they mostly answer the question instead of rewriting it, often
with numbers that are not in the question ("30 gün", "3 ay"), and their outputs are longer than the teacher's
(median 249 / 148 / 215 characters vs 143). Kumru-2B also echoes the question or the prompt's rules. Only
Qwen3-4B's fuse (0.67) is above no rewrite (0.64), within noise.

## QLoRA students (gate 4b, dev, 50 q)

One configuration for all three (`scripts/qlora.py`): 4-bit NF4 base, LoRA r 16 on all linear layers, 3 epochs on
the 966 training pairs; the last-epoch adapter merged into the bf16 base, then the gate 4a decoding
(`notebooks/02_qlora.ipynb`, `results/eval_qlora_v1_*_dev_v1_summary.csv`, `results/train_record_qlora_v1_*.json`).

| System | rw R@5 | rw R@10 | fuse R@1 | fuse R@3 | fuse R@5 | fuse R@10 | latency p50 / p95 | train |
|---|---|---|---|---|---|---|---|---|
| no rewrite (orig) | 0.64 | 0.73 | | | | | | |
| Gemini teacher | 0.75 | 0.78 | 0.47 | 0.67 | 0.71 | 0.81 | | |
| Kumru-2B | 0.49 | 0.74 | 0.34 | 0.55 | 0.69 | 0.81 | 0.5 / 1.0 s | 8 min, 4.7 GB |
| Qwen3-4B-Instruct-2507 | 0.61 | 0.70 | 0.33 | 0.59 | 0.69 | 0.77 | 2.5 / 3.8 s | 31 min, 12.6 GB |
| Llama-3.1-8B-Instruct | 0.61 | 0.78 | 0.43 | 0.67 | 0.69 | 0.81 | 2.3 / 3.2 s | 36 min, 16.0 GB |

After training all three rewrite instead of answering:
median length 142 / 143 / 145 characters (teacher 143), no truncations, and new numbers only as "6698 sayılı Kanun".
Fuse R@5 is 0.69 for all three, from 0.55 / 0.67 / 0.59 zero-shot, 0.02 (one question) under the teacher.
Rewrite alone is still well under the teacher (R@5 0.49-0.61 vs 0.75) and, for Kumru-2B, under no rewrite: some
rewrites drift from the question (Kumru-2B turned "Anonim verilerle araştırma" into "Arama motorları üzerinden").
Llama-3.1-8B is closest to the teacher at the top ranks (fuse R@1 0.43, R@3 0.67); Kumru-2B matches it at
R@5 / R@10 at a fifth of the latency. Val loss is lowest after epoch 2 for all three and rises slightly after
epoch 3 (the rules use the last epoch). Differences between the students are within noise at 50 questions.

## Held-out (gate 5, 20 q, run once)

The three gate 4b students, unchanged, on kvkk-rag's held-out set `eval/test_v2.jsonl` (`notebooks/03_heldout.ipynb`,
Colab L4, `results/eval_qlora_v1_*_heldout_v1_summary.csv`). Retrieval with kvkk-rag's held-out settings (fused
depth 30); teacher and no-rewrite numbers from its cached rewrites (`results/eval_teacher_heldout_v1_summary.csv`).
One question is 5 points.

| System | rw R@1 | rw R@3 | rw R@5 | rw R@10 | fuse R@1 | fuse R@3 | fuse R@5 | fuse R@10 | latency p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|
| no rewrite (orig) | 0.15 | 0.30 | 0.35 | 0.65 | | | | | |
| Gemini teacher | 0.50 | 0.65 | 0.70 | 0.75 | 0.20 | 0.55 | 0.60 | 0.70 | |
| Kumru-2B | 0.25 | 0.55 | 0.60 | 0.60 | 0.20 | 0.55 | 0.60 | 0.60 | 0.5 / 1.0 s |
| Qwen3-4B-Instruct-2507 | 0.25 | 0.50 | 0.55 | 0.60 | 0.35 | 0.50 | 0.55 | 0.65 | 2.8 / 4.0 s |
| Llama-3.1-8B-Instruct | 0.40 | 0.60 | 0.65 | 0.70 | 0.20 | 0.55 | 0.70 | 0.75 | 2.4 / 4.1 s |

On held-out the question alone is weak (R@5 0.35), and every student's rewrite lifts it to 0.55-0.65, four to six
questions. Against the teacher's rewrite alone (R@5 0.70) Llama-3.1-8B is one question under, Kumru-2B two and
Qwen3-4B three; the teacher stays clearly ahead at R@1 (0.50 vs 0.25-0.40). With fusion Llama-3.1-8B reaches 0.70
against the teacher's 0.60: two questions (q060, q064), at the noise limit, so it is read as on par with the
teacher, not better. Kumru-2B matches the teacher's fuse R@5 at the lowest latency. The order matches dev: Llama-3.1-8B
closest to the teacher, Kumru-2B the cheapest that keeps up at R@5, Qwen3-4B last. No truncations; median rewrite
length 151 / 162 / 155 characters (teacher 154). One kind of miss is a misread question: "İsimsiz ihbar mektupları
geçerli mi" became a question about the personal data inside the letter (Kumru-2B, Qwen3-4B) instead of whether the
Authority accepts anonymous complaints.

## Cost and latency

`scripts/cost.py` (`results/cost_v2_summary.csv`), dev + held-out latencies (70 q). The two sides are billed
differently. The teacher is paid per call: the cost kvkk-rag logged for each cached rewrite (gemini-3.8-flash,
default thinking), about $0.0032 a question; its latency is from this project's 1,023 teacher calls (API, home
connection). A student is paid per started GPU hour, however few questions run in it: Colab L4 at 1.4 CU/h, priced
at Colab's pay-as-you-go list price ($9.99 per 100 CU, about $0.10 per CU; an assumption, not an invoice), so
$0.14 per started hour. Capacity is 3600 s over the mean L4 latency (batch 1, one question after another); model
loading, under 2.5 min per model, is left out.

| System | latency p50 / p95 | capacity per hour | 20 q | 100 q | 1,000 q | 10,000 q | always on |
|---|---|---|---|---|---|---|---|
| Gemini teacher | 6.7 / 9.4 s | | $0.06 | $0.32 | $3.19 | $31.89 | |
| Kumru-2B | 0.5 / 1.1 s | ~6,500 | $0.14 | $0.14 | $0.14 | $0.28 | $3.36 / day |
| Qwen3-4B-Instruct-2507 | 2.7 / 3.9 s | ~1,350 | $0.14 | $0.14 | $0.14 | $1.12 | $3.36 / day |
| Llama-3.1-8B-Instruct | 2.4 / 3.6 s | ~1,500 | $0.14 | $0.14 | $0.14 | $0.98 | $3.36 / day |

A started L4 hour costs as much as 44 teacher calls, so a student is cheaper only when each GPU hour holds more than
44 questions. For batch jobs that is easy: 1,000 questions fit in one hour for every student ($0.14 vs $3.19). For
small runs it is not: the 20-question held-out set costs $0.14 on the GPU against $0.06 from Gemini. For live
traffic the GPU has to stay on, $3.36 a day, which pays off above about 1,050 questions a day; under that the
teacher is cheaper. On speed the students win at any volume: Kumru-2B is 13x and Llama-3.1-8B 3x faster than the
teacher's API call. One-off cost of the students: $3.995 of training data plus 4 CU ($0.40) of training for all
three. Project spend: $3.995 of Gemini calls and 6 Colab CU (gates 4a 1, 4b 4, 5 1). `results/cost_v1_summary.csv`
priced GPU time per second and is superseded.

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
