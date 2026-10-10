"""Freeze the source chunks for question generation, before any question exists. No LLM call.

Run from the repo root (after fetch_gen_corpus.py and extract_gen_corpus.py): python scripts/split_chunks.py [--tag v1]

Applies the exclusion rules of scripts/datagen.py to the corpus chunks and the gen chunks, computes the topic
threshold from the corpus chunks (median topic score of the section-excluded ones), and writes
  data/chunks_<tag>.jsonl         every chunk: kept ones with their generation "order", excluded ones with "reason"
  results/chunk_split_<tag>.json  counts per source and document, the threshold and how it behaves on the corpus
Never overwrites. scripts/gen_questions.py reads data/chunks_<tag>.jsonl.
"""
import argparse
import collections
import json
import sys

import numpy as np

import datagen
import kvkk_rag


def rate(x, t) -> float:
    return round(float((np.asarray(x) >= t).mean()), 3)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()
    out_chunks = kvkk_rag.ROOT / f"data/chunks_{args.tag}.jsonl"
    out_sum = kvkk_rag.RESULTS / f"chunk_split_{args.tag}.json"
    if out_chunks.exists() or out_sum.exists():
        sys.exit(f"{out_chunks.name} or {out_sum.name} already exists; use a new --tag")

    from sentence_transformers import SentenceTransformer
    from retriever import MODEL_NAME
    model = SentenceTransformer(MODEL_NAME)

    c_kept, c_exc = datagen.split_chunks()
    c_sec = [c for c in c_exc if c["reason"].startswith("section")]
    c_ev = [c for c in c_exc if c["reason"] == "evidence"]
    s_kept, s_sec, s_ev = (datagen.topic_scores(model, cs) for cs in (c_kept, c_sec, c_ev))
    topic_max = round(float(np.median(s_sec)), 4)
    auc = float(np.mean([[a > b for b in s_kept] for a in s_sec]))    # P(section chunk scores above a kept one)

    g_kept, g_exc = datagen.split_gen_chunks(model, topic_max)
    order = datagen.generation_order(c_kept, g_kept)
    rank = {c["chunk_id"]: i for i, c in enumerate(order)}

    out_chunks.parent.mkdir(exist_ok=True)
    with out_chunks.open("w", encoding="utf-8") as f:
        for c in c_kept + c_exc + g_kept + g_exc:
            row = {**c, "keep": c["chunk_id"] in rank, "order": rank.get(c["chunk_id"])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_doc = collections.defaultdict(collections.Counter)
    for c in c_kept + g_kept:
        per_doc[c["doc"]]["kept"] += 1
    for c in c_exc + g_exc:
        per_doc[c["doc"]][c["reason"].split()[0]] += 1
    summary = {
        "topic_threshold": topic_max,
        "topic_threshold_rule": "median topic score of corpus chunks excluded by the section rule",
        "corpus_calibration": {
            "auc_section_vs_kept": round(auc, 3),
            "share_at_or_above_threshold": {"kept": rate(s_kept, topic_max), "section": rate(s_sec, topic_max),
                                            "evidence": rate(s_ev, topic_max)},
            "n": {"kept": len(c_kept), "section": len(c_sec), "evidence": len(c_ev)},
        },
        "corpus": dict(collections.Counter(c["reason"].split()[0] for c in c_exc), kept=len(c_kept)),
        "gen": dict(collections.Counter(c["reason"] for c in g_exc), kept=len(g_kept)),
        "usable_chunks": len(order),
        "per_doc": {d: dict(v) for d, v in sorted(per_doc.items())},
    }
    out_sum.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_doc"}, ensure_ascii=False, indent=1))
    print(f"wrote {out_chunks.relative_to(kvkk_rag.ROOT)} and {out_sum.relative_to(kvkk_rag.ROOT)}")


if __name__ == "__main__":
    main()
