"""Exclusion of eval chunks. No LLM call, no model load."""
import datagen
from golden import facts, load_golden, norm

import kvkk_rag


def test_labels_expand_ranges():
    assert datagen._labels("yonetmelik_x, Madde 8-9") == ("yonetmelik_x", {"Madde 8", "Madde 9"})
    assert datagen._labels("rehber_ozel_nitelikli, 2.8") == ("rehber_ozel_nitelikli", {"2.8"})


def test_chunks_match_retriever_chunking():
    chunks = datagen.all_chunks()
    assert len(chunks) == 153     # 192/48 over the 5 documents, as in kvkk-rag
    assert [c["chunk_id"] for c in chunks] == list(range(153))


def test_no_kept_chunk_holds_eval_evidence_or_section():
    kept, excluded = datagen.split_chunks()
    assert len(kept) + len(excluded) == 153
    labels, _ = datagen.eval_exclusions()
    for path in datagen.EXCLUDE_SETS:
        for q in load_golden(path):
            for alts in facts(q):
                for alt in alts:
                    assert not any(norm(alt) in norm(c["text"]) for c in kept), q["id"]
    for c in kept:
        assert not any((c["doc"], label) in labels for label in c["sections"])


def test_eval_questions_dedup():
    qs = datagen.eval_questions()
    assert len(qs) == len({q["question"] for q in qs}) == 85     # 50 golden + 20 test_v2 + 10 + 5 abstention
