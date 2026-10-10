"""Shared pieces of the training-data pipeline: source chunks, eval exclusion, leakage filter. No LLM call.

Source chunks are kvkk-rag's retrieval chunks (192/48 words, scripts/chunking.py + scripts/sections.py),
so every chunk carries the same doc and article / section labels the retriever uses.

Exclusion (fixed before any question is generated): a chunk is dropped when
  1. it shares an 8-word sequence with any evidence sentence (or alternative) of an eval set, or
  2. any article / section label it covers equals the source label of an eval question
     ("Madde 8-9" counts as Madde 8 and Madde 9).
Eval sets: dev eval/golden.jsonl and held-out eval/test_v2.jsonl (eval/test_v1.jsonl is q031-q050 of golden).

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
import re

import kvkk_rag
from chunking import chunk_words
from golden import facts, load_golden, norm
from sections import chunk_sections
from textproc import tr_lower

SIZE, OVERLAP = 192, 48
SHARED_WORDS = 8
NGRAM_N, NGRAM_MAX = 3, 0.30
COSINE_MAX = 0.92
EXCLUDE_SETS = (kvkk_rag.DEV, kvkk_rag.HELDOUT)


def all_chunks() -> list[dict]:
    """Every retrieval chunk, in the retriever's order: {"chunk_id", "doc", "text", "section", "sections"}."""
    out = []
    for doc, t in kvkk_rag.corpus_texts().items():
        for c, meta in zip(chunk_words(t, SIZE, OVERLAP), chunk_sections(t, SIZE, OVERLAP), strict=True):
            out.append({"chunk_id": len(out), "doc": doc, "text": c, **meta})
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
    """(kept, excluded); every excluded chunk gets a "reason"."""
    labels, grams = eval_exclusions()
    kept, excluded = [], []
    for c in all_chunks():
        hit_labels = sorted(label for label in c["sections"] if (c["doc"], label) in labels)
        if _word_ngrams(c["text"], SHARED_WORDS) & grams:
            excluded.append({**c, "reason": "evidence"})
        elif hit_labels:
            excluded.append({**c, "reason": "section " + ", ".join(hit_labels)})
        else:
            kept.append(c)
    return kept, excluded


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
