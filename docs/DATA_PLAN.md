# Training data plan

Pairs of (everyday question, teacher rewrite). Written for spending gates 2 and 3; nothing here has been run.

## Pipeline

1. `scripts/gen_questions.py --limit N`: everyday Turkish questions from kvkk-rag corpus chunks.
   - Source: kvkk-rag's 153 retrieval chunks (192/48), minus every chunk that
     (a) shares an 8-word sequence with any dev or held-out evidence sentence, or
     (b) covers an article / section that a dev or held-out question is sourced from (`Madde 8-9` = 8 and 9).
   - One call per chunk, 3-5 questions per call (default 5). `gemini-3.8-flash`, temperature 1.0,
     thinking LOW, JSON output. The generator is not the teacher, so its settings may differ.
   - Prompt: everyday wording from the point of view of an employee, customer, patient, small-business owner,
     HR person, developer...; avoid the legal terms of the passage; no article numbers or document names;
     mix scenarios, direct questions and short sloppy ones; answerable from the passage.
   - Leakage filter on every question, against all 85 eval questions (golden, test_v2, both abstention sets):
     word 3-gram Jaccard >= 0.30 or e5 cosine >= 0.92 -> kept in the file with `"leak": true`, never used.
     Calibration (`scripts/calibrate_leakage.py`, `results/leakage_calibration_v1.json`): between distinct
     eval questions the nearest-neighbour cosine is 0.88 median / 0.91 p90 / 0.93 max, 3-gram Jaccard <= 0.11.
     e5 cosines are compressed, so cosine alone cannot separate a paraphrase from a new question on a nearby
     topic; the chunk exclusion is the main guard, the filter is the second.
   - Output: `data/questions_<tag>.jsonl` (resumable).
2. `scripts/gen_rewrites.py --limit N`: teacher rewrite of every non-leaked question.
   - Teacher = kvkk-rag's rewrite step, unchanged: `SYSTEM` imported from the pinned submodule, same model,
     temperature 0, the model's default thinking (no thinking config), AFC off, whitespace-normalized output.
     `tests/test_teacher.py` checks with fake clients that the request equals kvkk-rag's (no API call).
   - Output: `data/rewrites_<tag>.jsonl` with tokens, cost and latency per call (resumable).

Both scripts check `GEMINI_API_KEY` in this repo's `.env` first, print the call count and estimated cost and
wait for `yes`. `scripts/llm.py` logs every call to `results/llm_usage.csv` (separate from kvkk-rag's) and
refuses calls past the $6 project budget; a run also stops when it passes its approved amount by 10%.

## Problem: the exclusion rule leaves 32 chunks

| | chunks |
|---|---|
| all | 153 |
| excluded, holds eval evidence | 81 |
| excluded, same article / section as an eval question | 40 |
| **usable** | **32** (kanun 18, özel nitelikli rehberi 12, silme yönetmeliği 2) |

`rehber_veri_guvenligi` and `teblig_aydinlatma` have no usable chunk. At 3-5 questions per chunk that is
**about 100-160 questions**, not 1000-1500. Options (your choice; nothing is decided):

- **A. Keep the rule, accept ~150 pairs** (or ~300 with two rounds of different personas per chunk).
  Cheapest (~$0.55). Small and narrow: the student never sees the security guide or the communiqué.
- **B. Keep the rule and add generation-only texts (recommended):** other official KVKK texts that are not in
  the retrieval corpus (e.g. the two listed as gaps in kvkk-rag: Veri Sorumlusuna Başvuru Usul ve Esasları
  Hakkında Tebliğ, Aydınlatma Yükümlülüğünün Yerine Getirilmesi Rehberi, plus other Kurum guides and
  regulations, to be listed and checked first). They use the same legal vocabulary, cannot hold eval
  evidence, and go through the same leakage filter. Reaches 1000-1500 pairs. $0 extra to prepare (PDF
  extraction with kvkk-rag's cleaner); the corpus used for retrieval does not change.
- **C. Relax the rule to evidence-only** (drop the same-section part): 72 chunks, ~360 questions.
  Not recommended: questions on the same article as an eval question are the leakage the rule targets.

## Cost estimate

Per call: teacher rewrite $0.00316 mean / $0.00453 max (kvkk-rag's 85 cached rewrites, ~227 input and ~800
output tokens incl. thinking). Question generation, not measured yet: ~850 input tokens (prompt + 192-word
chunk at ~2.2 tokens per word, measured on kvkk-rag's answer calls) and ~600 output tokens with LOW thinking
-> $0.0029 mean, $0.0064 assumed max. 10% of questions assumed lost to the leakage filter.

| Run | Gen calls | Questions | Rewrites | Expected | Upper (every call at max) |
|---|---|---|---|---|---|
| Pilot (gate 2) | 2 | 10 | 10 | $0.04 | $0.06 |
| A: strict, 32 chunks x 5 | 32 | ~160 | ~144 | $0.55 | $0.86 |
| 1000 pairs (needs B) | 223 | ~1,111 | 1,000 | $3.81 | $5.95 |
| 1200 pairs (needs B) | 267 | ~1,334 | 1,200 | $4.56 | $7.14 |
| 1500 pairs (needs B) | 334 | ~1,667 | 1,500 | $5.71 | $8.93 |

- **1500 pairs does not fit the $6 budget safely**: expected $5.71 leaves almost nothing for the pilot and noise,
  and with default thinking for question generation it would be $6.83.
- 1000-1200 pairs fit on the expected cost. The rewrites dominate (83%) and cannot be made cheaper without
  changing the teacher.
- `confirm_spend` refuses a run whose upper estimate would cross the budget, so a large run goes in batches
  (e.g. `--limit 400` at a time). After the pilot, the estimates use this project's own measured mean and max.

## Pilot (gate 2), after approval

```
python scripts/gen_questions.py --limit 10      # 2 calls
python scripts/gen_rewrites.py --limit 10       # 10 calls
```

Check: question quality by hand (everyday wording, answerable from the chunk, no legal terms copied),
leakage flags and their scores, real tokens and cost per call (`results/llm_usage.csv`) against this table.
