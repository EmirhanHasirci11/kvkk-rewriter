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

## Pilot (gate 2), run 2026-10-10

```
python scripts/gen_questions.py --limit 10      # 2 calls: c029 (kanun, Madde 24-25), g01-091 (lawyers' guide)
python scripts/gen_rewrites.py --limit 10       # 10 calls
```

Files: `data/questions_v1.jsonl`, `data/rewrites_v1.jsonl`, `results/llm_usage.csv`. Spent: **$0.0364**.

| | estimated per call | measured per call |
|---|---|---|
| question generation (LOW thinking) | $0.0029 (850 in / 600 out) | $0.0012 (~718 in / ~175 out) |
| teacher rewrite | $0.0032 | $0.0034 mean, $0.0054 max (~240 in / ~860 out); p50 latency 6.7 s |

- Teacher: input and output tokens match kvkk-rag's cached rewrites (216-245 input tokens there), and the
  rewrites read the same: legal terms ("ilgili kişi", "açık rıza", "veri sorumlusu"), 1-2 sentences, ~19 words.
- Questions: everyday wording, scenarios, answerable from the chunk in 9 of 10 (g01-091_2 asks whether it is
  a crime; the chunk only gives the legal basis). Weak spots: no short or sloppy question although the prompt
  asks for some; c029 is about the Kurum's organisation (who deputises for the president), which real users
  rarely ask; c029_2 copies "veri sorumluları sicil".
- Leakage filter: 0 of 10 flagged; max e5 cosine 0.889 (nearest q023), 3-gram Jaccard 0 for all. By hand, no
  question is a paraphrase of an eval question, but g01-091 states the same principle as q023's evidence
  (no consent is needed when another processing condition applies): the topic-level risk described above.
- Full run, re-estimated with the measured means (max per call kept at least at the default): 221 more
  generation calls $0.26 + 990 rewrites $3.36 = **$3.63** expected (upper $6.71), $3.67 project total.

## After the pilot: prompt v2, chunk split v2 (2026-10-10, before the full run)

Goal set by the user: training questions should look like real use. Two changes, both fixed before gate 3:

- **Prompt v2** (`scripts/gen_questions.py`, `PROMPT_VERSION`; every new row records it, the pilot's rows are v1):
  questions as people type them into a KVKK assistant or a search box; a wider set of askers (job seeker,
  student, building manager, accountant, e-commerce owner...); no quiz-style questions about the text
  ("how many ...", "which unit ..."); everyday words instead of the legal terms; a mix of 1-3 sentence
  situations, short direct questions and at least one sloppy one per chunk (lowercase, no punctuation, a
  typo, or unaccented letters). The examples in the prompt are templates without a topic, so they cannot
  steer the generator towards an eval topic. The model may return fewer questions, or none, for a passage
  with nothing a real person would ask (a marker row records the chunk as done).
- **Chunk split v2** (`data/chunks_v2.jsonl`, `results/chunk_split_v2.json`; v1 kept): two more rules.
  Corpus: a chunk whose labels are all administrative articles of the Kanun (Madde 19, 21, 23-27, 29-33,
  Geçici Madde 1-3: Kurum / Kurul organisation, president, staff, budget, amendments, transition, entry into
  force) is dropped (15 chunks; the duties in Madde 20 and 22 stay). Guides: four background documents chosen
  by title are dropped (g24 constitutional right, g32 international regulations, g33 need for the law,
  g37 Kurul structure; 19 chunks).

| | all | excluded | usable |
|---|---|---|---|
| kvkk-rag corpus | 153 | 136 (evidence 81, section 40, admin 15) | 17 |
| KVKK guides | 1,769 | 725 (evidence 335, topic 318, references 27, front 23, background 19, short 3) | 1,044 |
| **total** | 1,922 | 861 | **1,061** |

The topic threshold is unchanged (0.8846, from the section-excluded chunks). In the v2 report its AUC against
the kept corpus chunks is 0.66 instead of 0.78 only because the 15 admin chunks, which are far from any
evidence, left the kept set; v1's 0.78 over 32 chunks is the better description of the rule.

Pilot data: the 5 pairs from c029 (now an admin chunk) are not used for training; the 5 from g01-091 are.

Full run, re-estimated: 995 more pairs. Prompt v2 may return fewer than 5 questions per chunk; at an assumed
4 kept per call, ~276 generation calls x $0.0012 = $0.33, plus 995 rewrites x $0.0034 = $3.38:
**~$3.7 expected**, ~$3.75 project total with the pilot.

### Prompt v2 check (2 calls, $0.0026)

Chunks c020 (kanun, Madde 20-21) and g01-133 (lawyers' guide, generative AI): 9 questions, 0 flagged, $0.0013 per
call. Two of the nine are sloppy as asked ("kvkk kurulu uyelerini kim seciyor cumhurbaskani mi meclis mi",
"yapay zekaya sirket evragi yuklemek verileri yurtdisina gondermek sayilirmi"); the AI-upload questions read
like real ones. Weak spot: c020 keeps Madde 21 (Kurul membership) because the chunk also covers Madde 20, and
two of its questions ask who can become a Kurul member, which few users would; with 17 corpus chunks out of
~250 calls this is left as is.
Nearest eval question for three AI questions is a011 (held-out abstention: training an own AI model on
customer data), cosine up to 0.894: related topic, different question. The abstention sets are not part of
this project's metrics; if they are used later, note that the training data covers AI tools while the
retrieval corpus does not.
