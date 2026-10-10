# Training data plan

Pairs of (everyday question, teacher rewrite). **Target: 1000 pairs.** Source chunks are frozen; no question
has been generated and no API call has been made yet (gates 2 and 3 need approval).

## Decision (2026-10-10): option B, 1000 pairs

The exclusion rule leaves only 32 of kvkk-rag's 153 retrieval chunks (81 hold eval evidence, 40 cover an
eval question's article / section; nothing usable from the security guide or the communiqué), about 150
questions. Chosen: keep the rule and add **generation-only texts**, the official KVKK guides that are not in
the retrieval corpus. They seed training questions only and are never added to the retrieval corpus.
Not chosen: A (only ~150 pairs), C (relax the section rule).

## Pipeline

0. Source texts (done, $0)
   - `scripts/fetch_gen_corpus.py`: every PDF on the KVKK guides page (https://www.kvkk.gov.tr/Icerik/2030/rehberler,
     5 pages, read 2026-10-10), except the two the retrieval corpus already holds (special-category guide,
     data security guide). Rule fixed before reading any of them; nothing is dropped by content at this step.
     35 PDFs, URLs and sha256 in `gen_corpus/sources.json`; the PDFs themselves are git-ignored.
   - `scripts/extract_gen_corpus.py`: generic PyMuPDF cleaner (body font size, contents pages and entries,
     covers and publisher notes dropped). 34 documents, 253,887 words in `gen_corpus/text/`
     (`gen_corpus/manifest.json`). Skipped: g05, the payment / e-money guide, a scan without a text layer (no OCR).
1. Frozen chunk split (done, $0): `scripts/split_chunks.py` -> `data/chunks_v1.jsonl`, `results/chunk_split_v1.json`.
   All chunks are 192/48 words (kvkk-rag's chunker).
   - Corpus chunk excluded if it (a) shares an 8-word sequence with any dev / held-out evidence sentence, or
     (b) covers an article / section a dev / held-out question is sourced from (`Madde 8-9` = 8 and 9).
   - Guide chunk excluded if it (a) shares an 8-word sequence with eval evidence, (b) has e5 cosine >= 0.8846
     to any eval evidence sentence ("topic", standing in for the section rule), (c) is under 60 words,
     (d) is a document's first chunk (cover, abbreviations), or (e) holds 3+ URLs / "Erişim Adresi/Tarihi"
     (footnote references).
   - Topic threshold, defined before use as the median topic score of the corpus chunks excluded by the
     section rule. On the corpus it flags 50% of section-excluded chunks, 83% of evidence chunks and 22% of
     chunks the corpus rules keep (AUC 0.78 section vs kept). So it is a partial stand-in: it removes the
     guide passages closest to eval evidence, but a guide passage on the same rule as an eval question can
     still pass, which is why the question-level leakage filter below runs on every question.
   - Order: corpus and guide chunks alternate until the 32 corpus chunks run out; guide chunks go round-robin
     over documents (seed 2026). The pilot's two calls get one of each.
2. `scripts/gen_questions.py --limit N`: one call per chunk in that order, 5 questions per call.
   `gemini-3.8-flash`, temperature 1.0, thinking LOW, JSON output (the generator is not the teacher).
   Prompt: everyday wording from the point of view of an employee, customer, patient, small-business owner,
   HR person, developer...; avoid the passage's legal terms; no article numbers or document names; mix
   scenarios, direct questions and short sloppy ones; answerable from the passage.
   - Leakage filter on every question, against all 85 eval questions (golden, test_v2, both abstention sets):
     word 3-gram Jaccard >= 0.30 or e5 cosine >= 0.92 -> kept in the file with `"leak": true`, never used.
     Calibration (`results/leakage_calibration_v1.json`): between distinct eval questions the nearest-neighbour
     cosine is 0.88 median / 0.91 p90 / 0.93 max, 3-gram Jaccard <= 0.11.
   - Output: `data/questions_v1.jsonl` (resumable).
3. `scripts/gen_rewrites.py --limit N`: teacher rewrite of every non-leaked question. Teacher = kvkk-rag's
   rewrite step, unchanged (`SYSTEM` imported from the submodule, same model, temperature 0, default thinking,
   AFC off, whitespace-normalized); `tests/test_teacher.py` checks the request equals kvkk-rag's.
   Output: `data/rewrites_v1.jsonl` with tokens, cost and latency per call (resumable).

Both scripts check `GEMINI_API_KEY` in this repo's `.env` first, print the call count and estimated cost and
wait for `yes`. `scripts/llm.py` logs every call to `results/llm_usage.csv` (separate from kvkk-rag's) and
refuses calls past the $6 project budget; a run also stops when it passes its approved amount by 10%.

## Chunks

| | all | excluded | usable |
|---|---|---|---|
| kvkk-rag corpus | 153 | 121 (evidence 81, section 40) | 32 |
| KVKK guides (34 docs) | 1,769 | 706 (evidence 335, topic 318, references 27, front 23, short 3) | 1,063 |
| **total** | 1,922 | 827 | **1,095** |

1000 pairs need ~223 calls (1,111 questions at 10% assumed leakage), so a fifth of the usable chunks:
32 corpus + 191 guide chunks from 36 documents (~7 per guide; the short guides fewer). Per-document counts
are in `results/chunk_split_v1.json`.

## Cost estimate (1000 pairs)

Per call: teacher rewrite $0.00316 mean / $0.00453 max (kvkk-rag's 85 cached rewrites, ~227 input and ~800
output tokens incl. thinking). Question generation, not measured yet: ~850 input tokens (prompt + 192-word
chunk at ~2.2 tokens per word, measured on kvkk-rag's answer calls) and ~600 output tokens with LOW thinking
-> $0.0029 mean, $0.0064 assumed max.

| Run | Gen calls | Questions | Rewrites | Expected | Upper (every call at max) |
|---|---|---|---|---|---|
| Pilot (gate 2) | 2 | 10 | 10 | $0.04 | $0.06 |
| Full (gate 3), incl. the pilot's | 223 | ~1,111 | 1,000 | $3.81 | $5.95 |

- Expected total $3.85 with the pilot, within the $6 budget; the worst case touches it.
- The rewrites are 83% of the cost and cannot be made cheaper without changing the teacher.
- `confirm_spend` refuses a run whose upper estimate would cross the budget, so gate 3 runs in batches
  (e.g. `--limit 300`). After the pilot, estimates use this project's measured mean and max.

## Pilot (gate 2), after approval

```
python scripts/gen_questions.py --limit 10      # 2 calls: one corpus chunk, one guide chunk
python scripts/gen_rewrites.py --limit 10       # 10 calls
```

Check: question quality by hand (everyday wording, answerable from the chunk, no legal terms copied),
leakage flags and their scores, real tokens and cost per call (`results/llm_usage.csv`) against this table.
