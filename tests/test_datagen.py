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
    assert [c["chunk_id"] for c in chunks] == [f"c{i:03d}" for i in range(153)]


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


def test_generation_order_alternates_sources_and_round_robins_docs():
    corpus = [{"chunk_id": f"c{i:03d}", "doc": "kanun"} for i in range(2)]
    gen = [{"chunk_id": f"{d}-{i:03d}", "doc": d} for d in ("g01", "g02") for i in range(2)]
    order = datagen.generation_order(corpus, gen)
    assert [c["doc"] for c in order] == ["kanun", "g01", "kanun", "g02", "g01", "g02"]
    assert sorted(c["chunk_id"] for c in order) == sorted(c["chunk_id"] for c in corpus + gen)
    assert order == datagen.generation_order(corpus, gen)     # seeded


def test_admin_articles_excluded_but_duties_kept():
    kept, excluded = datagen.split_chunks()
    for c in kept:
        if c["doc"] == "kvkk_kanun_6698" and c["sections"]:
            assert not set(c["sections"]) <= datagen.ADMIN_ARTICLES, c["chunk_id"]
    assert all(set(c["sections"]) <= datagen.ADMIN_ARTICLES for c in excluded if c["reason"] == "admin")
    assert not {"Madde 20", "Madde 22"} & datagen.ADMIN_ARTICLES


def test_parse_questions_distinguishes_empty_from_invalid():
    import gen_questions
    assert gen_questions.parse_questions('{"sorular": ["a  b", " "]}') == ["a b"]
    assert gen_questions.parse_questions('{"sorular": []}') == []
    assert gen_questions.parse_questions("not json") is None
