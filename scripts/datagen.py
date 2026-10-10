"""Shared pieces of the training-data pipeline: source chunks, eval exclusion, leakage filter. No LLM call.

Two chunk sources, both cut 192/48 words with kvkk-rag's scripts/chunking.py:
  corpus  kvkk-rag's retrieval chunks (ids c000-c152), with the article / section labels the retriever uses
  gen     generation-only texts: official KVKK guides not in the retrieval corpus (gen_corpus/text, ids g05-012)

Exclusion (fixed before any question is generated). Eval sets: dev eval/golden.jsonl and held-out
eval/test_v2.jsonl (eval/test_v1.jsonl is q031-q050 of golden). A corpus chunk is dropped when
  1. it shares an 8-word sequence with any evidence sentence (or alternative) of an eval set, or
  2. any article / section label it covers equals the source label of an eval question
     ("Madde 8-9" counts as Madde 8 and Madde 9), or
  3. every label it covers is an administrative article of the Kanun (ADMIN_ARTICLES: the Kurum and Kurul
     organisation, president, staff, budget, amendments, transition, entry into force), which real users
     do not ask about. The Kurum's and Kurul's duties (Madde 20, 22) stay. Added after the pilot.
A gen chunk has no labels; it is dropped when
  1. as above (8-word sequence shared with eval evidence), or
  2. its e5 cosine to any eval evidence sentence ("passage: " on both sides) is >= the topic threshold, or
  3. it has fewer than MIN_WORDS words (a document's tail), or
  4. it is the first chunk of its document (cover, abbreviations, publisher notes), or
  5. it is mostly references: REFS_MIN or more URLs / "Erişim Adresi" / "Erişim Tarihi" (footnotes), or
  6. its document is background, chosen by title (BACKGROUND_DOCS: the constitutional right, international
     regulations, the need for the law, the Kurul's structure). Added after the pilot.
Rule 2 stands in for the section rule. The threshold is defined on the corpus chunks, where both are known:
the median topic score of the corpus chunks excluded by the section rule (so half of them would also be
caught by rule 2). The cosine separates section-excluded from kept corpus chunks only partly; the rates are
in results/chunk_split_<tag>.json (scripts/split_chunks.py).

Leakage filter (applied to every generated question): drop it when, against any question of the eval sets
(golden, test_v1, test_v2, abstention, abstention_v2),
  - word 3-gram Jaccard >= NGRAM_MAX, or
  - multilingual-e5-base cosine ("query: " prefix, as the retriever) >= COSINE_MAX.
Thresholds from scripts/calibrate_leakage.py (results/leakage_calibration_v1.json): between distinct eval
questions the nearest-neighbour cosine is 0.88 at the median, 0.91 at p90 and 0.93 at most (two questions on
the same retention rule), and the 3-gram Jaccard is at most 0.11. e5 cosines are compressed, so 0.92 flags
questions as close as the closest distinct eval pairs; it cannot separate every paraphrase from a new question
on a nearby topic, which is why the section exclusion above comes first.
"""
import json
import random
import re

import kvkk_rag
from chunking import chunk_words
from golden import facts, load_golden, norm
from sections import chunk_sections
from textproc import tr_lower

SIZE, OVERLAP = 192, 48
SHARED_WORDS = 8
MIN_WORDS = 60
REFS_MIN = 3
ADMIN_ARTICLES = {f"Madde {n}" for n in (19, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33)} | {
    f"Geçici Madde {n}" for n in (1, 2, 3)}
BACKGROUND_DOCS = {"g24", "g32", "g33", "g37"}
REFS_RE = re.compile(r"https?://|Erişim (?:Adresi|Tarihi)")
SEED = 2026
GEN_TEXT = kvkk_rag.ROOT / "gen_corpus/text"
NGRAM_N, NGRAM_MAX = 3, 0.30
COSINE_MAX = 0.92
EXCLUDE_SETS = (kvkk_rag.DEV, kvkk_rag.HELDOUT)


def all_chunks() -> list[dict]:
    """Every retrieval chunk, in the retriever's order: {"chunk_id", "source", "doc", "text", "section", "sections"}."""
    out = []
    for doc, t in kvkk_rag.corpus_texts().items():
        for c, meta in zip(chunk_words(t, SIZE, OVERLAP), chunk_sections(t, SIZE, OVERLAP), strict=True):
            out.append({"chunk_id": f"c{len(out):03d}", "source": "corpus", "doc": doc, "text": c, **meta})
    return out


def gen_chunks() -> list[dict]:
    """Every chunk of the generation-only texts, document by document."""
    out = []
    for p in sorted(GEN_TEXT.glob("*.txt")):
        for i, c in enumerate(chunk_words(p.read_text(encoding="utf-8"), SIZE, OVERLAP)):
            out.append({"chunk_id": f"{p.stem}-{i:03d}", "source": "gen", "doc": p.stem, "text": c,
                        "section": None, "sections": []})
    return out


def _labels(source: str) -> tuple[str, set[str]]:
    """'yonetmelik_x, Madde 8-9' -> ('yonetmelik_x', {'Madde 8', 'Madde 9'})."""
    doc, label = (s.strip() for s in source.split(",", 1))
    m = re.fullmatch(r"(Madde) (\d+)-(\d+)", label)
    if m:
        return doc, {f"Madde {i}" for i in range(int(m[2]), int(m[3]) + 1)}
    return doc, {label}


def _word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = tr_lower(norm(text)).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def eval_exclusions() -> tuple[set[tuple[str, str]], set[tuple[str, ...]]]:
    """(doc, label) pairs and evidence 8-grams of every question in the dev and held-out sets."""
    labels, grams = set(), set()
    for path in EXCLUDE_SETS:
        for q in load_golden(path):
            doc, ls = _labels(q["source"])
            labels |= {(doc, label) for label in ls}
            for alts in facts(q):
                for alt in alts:
                    grams |= _word_ngrams(alt, SHARED_WORDS)
    return labels, grams


def split_chunks() -> tuple[list[dict], list[dict]]:
    """Corpus chunks: (kept, excluded); every excluded chunk gets a "reason"."""
    labels, grams = eval_exclusions()
    kept, excluded = [], []
    for c in all_chunks():
        hit_labels = sorted(label for label in c["sections"] if (c["doc"], label) in labels)
        if _word_ngrams(c["text"], SHARED_WORDS) & grams:
            excluded.append({**c, "reason": "evidence"})
        elif hit_labels:
            excluded.append({**c, "reason": "section " + ", ".join(hit_labels)})
        elif c["doc"] == "kvkk_kanun_6698" and c["sections"] and set(c["sections"]) <= ADMIN_ARTICLES:
            excluded.append({**c, "reason": "admin"})
        else:
            kept.append(c)
    return kept, excluded


def evidence_sentences() -> list[str]:
    return [alt for path in EXCLUDE_SETS for q in load_golden(path) for alts in facts(q) for alt in alts]


def encode_passages(model, texts: list[str]):
    return model.encode(["passage: " + t for t in texts], normalize_embeddings=True, batch_size=16,
                        show_progress_bar=False)


def topic_scores(model, chunks: list[dict]):
    """Max e5 cosine of each chunk to any eval evidence sentence."""
    ev = encode_passages(model, evidence_sentences())
    return (encode_passages(model, [c["text"] for c in chunks]) @ ev.T).max(1)


def split_gen_chunks(model, topic_max: float) -> tuple[list[dict], list[dict]]:
    """Gen chunks: (kept, excluded); every chunk gets its "topic" score, excluded ones a "reason"."""
    _, grams = eval_exclusions()
    chunks = gen_chunks()
    scores = topic_scores(model, chunks)
    kept, excluded = [], []
    for c, score in zip(chunks, scores):
        c = {**c, "topic": round(float(score), 4)}
        if _word_ngrams(c["text"], SHARED_WORDS) & grams:
            excluded.append({**c, "reason": "evidence"})
        elif score >= topic_max:
            excluded.append({**c, "reason": "topic"})
        elif len(c["text"].split()) < MIN_WORDS:
            excluded.append({**c, "reason": "short"})
        elif c["chunk_id"].endswith("-000"):
            excluded.append({**c, "reason": "front"})
        elif len(REFS_RE.findall(c["text"])) >= REFS_MIN:
            excluded.append({**c, "reason": "references"})
        elif c["doc"] in BACKGROUND_DOCS:
            excluded.append({**c, "reason": "background"})
        else:
            kept.append(c)
    return kept, excluded


def generation_order(corpus_kept: list[dict], gen_kept: list[dict]) -> list[dict]:
    """Corpus chunks (seeded shuffle) and gen chunks (round-robin over documents, each document seeded-shuffled)
    alternate until the corpus chunks run out; then the rest of the gen chunks. A run cut off by --limit
    (the pilot) still sees both sources and many documents."""
    rng = random.Random(SEED)
    first = corpus_kept[:]
    rng.shuffle(first)
    by_doc = {}
    for c in gen_kept:
        by_doc.setdefault(c["doc"], []).append(c)
    queues = []
    for doc in sorted(by_doc):
        q = by_doc[doc][:]
        rng.shuffle(q)
        queues.append(q)
    rest = []
    while any(queues):
        for q in queues:
            if q:
                rest.append(q.pop(0))
    woven = [c for pair in zip(first, rest) for c in pair]
    n = min(len(first), len(rest))
    return woven + first[n:] + rest[n:]


def eval_questions() -> list[dict]:
    out = []
    for path in kvkk_rag.EVAL_SETS:
        out += [{"id": q["id"], "question": q["question"], "file": path.name}
                for q in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if q]
    # test_v1 repeats q031-q050 of golden
    return list({q["question"]: q for q in out}.values())


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


class LeakageFilter:
    """Checks generated questions against every eval question (n-gram and embedding similarity)."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from retriever import MODEL_NAME
        self.evals = eval_questions()
        self.eval_grams = [_word_ngrams(q["question"], NGRAM_N) for q in self.evals]
        self.model = SentenceTransformer(MODEL_NAME)
        self.eval_vecs = self._encode([q["question"] for q in self.evals])

    def _encode(self, texts):
        return self.model.encode(["query: " + t for t in texts], normalize_embeddings=True,
                                 batch_size=32, show_progress_bar=False)

    def check(self, questions: list[str]) -> list[dict]:
        """For each question: {"leak", "max_ngram", "max_cosine", "nearest_id"}."""
        sims = self._encode(questions) @ self.eval_vecs.T
        out = []
        for q, row in zip(questions, sims):
            g = _word_ngrams(q, NGRAM_N)
            ngram = max(_jaccard(g, e) for e in self.eval_grams)
            j = int(row.argmax())
            out.append({"leak": ngram >= NGRAM_MAX or float(row[j]) >= COSINE_MAX,
                        "max_ngram": round(ngram, 3), "max_cosine": round(float(row[j]), 4),
                        "nearest_id": self.evals[j]["id"]})
        return out
