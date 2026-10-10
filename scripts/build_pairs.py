"""Training pairs from the generated questions and teacher rewrites. No LLM call.

Run from the repo root: python scripts/build_pairs.py [--tag v1] [--chunks v2] [--out pairs_v1]

A pair is kept when
  - the question passed the leakage filter ("leak": false),
  - its chunk is usable in the chunk split given by --chunks (so the pilot's c029 pairs drop out under v2),
  - it has a non-empty teacher rewrite,
  - the question is not a duplicate (case- and whitespace-insensitive) of an earlier one.
Validation split: VAL_SHARE of the chunks (not of the pairs, so no chunk is in both), chosen with seed 2026.
Writes data/<out>.jsonl ({"qid", "chunk_id", "source", "doc", "question", "rewrite", "split"}) and
results/<out>_summary.json (counts, per source, lengths, share of sloppy questions). Never overwrites.
"""
import argparse
import collections
import json
import random
import statistics
import sys

import kvkk_rag

VAL_SHARE = 0.05
SEED = 2026


def read_jsonl(path) -> list[dict]:
    return [r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r]


def is_sloppy(q: str) -> bool:
    """Lowercase start, no final punctuation, or no Turkish letters at all in a question that needs them."""
    return q[:1].islower() or q[-1:] not in "?.!" or not any(ch in q for ch in "çğıöşüÇĞİÖŞÜ")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1", help="data/questions_<tag>.jsonl and data/rewrites_<tag>.jsonl")
    parser.add_argument("--chunks", default="v2", help="data/chunks_<chunks>.jsonl")
    parser.add_argument("--out", default="pairs_v1")
    args = parser.parse_args()
    out = kvkk_rag.ROOT / f"data/{args.out}.jsonl"
    out_sum = kvkk_rag.RESULTS / f"{args.out}_summary.json"
    if out.exists() or out_sum.exists():
        sys.exit(f"{out.name} or {out_sum.name} already exists; use a new --out")

    usable = {c["chunk_id"] for c in read_jsonl(kvkk_rag.ROOT / f"data/chunks_{args.chunks}.jsonl") if c["keep"]}
    questions = [q for q in read_jsonl(kvkk_rag.ROOT / f"data/questions_{args.tag}.jsonl") if q["question"]]
    rewrites = {r["qid"]: r for r in read_jsonl(kvkk_rag.ROOT / f"data/rewrites_{args.tag}.jsonl")}

    dropped = collections.Counter()
    pairs, seen = [], set()
    for q in questions:
        key = " ".join(q["question"].lower().split())
        if q["leak"]:
            dropped["leak"] += 1
        elif q["chunk_id"] not in usable:
            dropped[f"chunk not usable in {args.chunks}"] += 1
        elif q["qid"] not in rewrites:
            dropped["no rewrite yet"] += 1
        elif not rewrites[q["qid"]]["rewrite"]:
            dropped["empty rewrite"] += 1
        elif key in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(key)
            pairs.append({"qid": q["qid"], "chunk_id": q["chunk_id"], "source": q["source"], "doc": q["doc"],
                          "question": q["question"], "rewrite": rewrites[q["qid"]]["rewrite"]})

    chunks = sorted({p["chunk_id"] for p in pairs})
    random.Random(SEED).shuffle(chunks)
    val = set(chunks[:max(1, round(len(chunks) * VAL_SHARE))]) if chunks else set()
    for p in pairs:
        p["split"] = "val" if p["chunk_id"] in val else "train"

    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    words = lambda key: round(statistics.mean(len(p[key].split()) for p in pairs), 1) if pairs else None
    summary = {
        "pairs": len(pairs),
        "split": dict(collections.Counter(p["split"] for p in pairs)),
        "chunks": len(chunks), "docs": len({p["doc"] for p in pairs}),
        "by_source": dict(collections.Counter(p["source"] for p in pairs)),
        "dropped": dict(dropped),
        "mean_words": {"question": words("question"), "rewrite": words("rewrite")},
        "sloppy_question_share": round(sum(map(is_sloppy, (p["question"] for p in pairs))) / len(pairs), 3) if pairs else None,
    }
    out_sum.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
