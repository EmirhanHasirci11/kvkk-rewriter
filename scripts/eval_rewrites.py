"""Retrieval with a file of rewrites (a student's, or any other). No LLM call; runs on CPU.

Run from the repo root: python scripts/eval_rewrites.py results/rewrites_<name>.jsonl [...] [--tag <name>]
                        [--set dev|heldout|test_v3]

Each input is a JSONL file with one row per question of the set: {"id", "question", "rewrite", ...} and, for
student runs, "latency_s" / "truncated" (from the Colab notebook). Retrieval is the matching part of
scripts/sanity_check.py: kvkk-rag's retriever (no reranker, bm25-p5, e5 + BM25 RRF, 192/48), systems rw
(rewrite alone) and fuse (RRF over question + rewrite), fused depth 10 on dev (as kvkk-rag's eval_rewrite.py)
and 30 on held-out (as its run_heldout.py). The teacher's numbers are in results/sanity_v1_summary.csv.
The held-out set is scored once, at the end (docs/ROADMAP.md, gate 5); test_v3 likewise (gate 7), with the
held-out depth of 30.

Writes results/eval_<tag>_per_question.csv and results/eval_<tag>_summary.csv (never overwrites); the tag
defaults to the input file name without "rewrites_".
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import kvkk_rag
from golden import facts, load_golden, recall_at_k
from retriever import HybridRetriever

KS = (1, 3, 5, 10)
SETS = {"dev": (kvkk_rag.DEV, 10), "heldout": (kvkk_rag.HELDOUT, 30),    # set -> (questions, fused depth)
        "test_v3": (kvkk_rag.TEST_V3, 30)}


def load_rows(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--tag", help="only with a single input file")
    parser.add_argument("--set", choices=SETS, default="dev")
    args = parser.parse_args()
    if args.tag and len(args.files) > 1:
        sys.exit("--tag works with one input file only")

    qpath, depth = SETS[args.set]
    questions = load_golden(qpath)
    jobs = []
    for path in args.files:
        tag = args.tag or path.stem.removeprefix("rewrites_")
        out_pq = kvkk_rag.RESULTS / f"eval_{tag}_per_question.csv"
        out_sum = kvkk_rag.RESULTS / f"eval_{tag}_summary.csv"
        if out_pq.exists() or out_sum.exists():
            sys.exit(f"{out_pq.name} or {out_sum.name} already exists; use a new --tag")
        rows = load_rows(path)
        missing = [q["id"] for q in questions if q["id"] not in rows]
        if missing:
            sys.exit(f"{path.name}: no rewrite for {missing}")
        for q in questions:
            assert rows[q["id"]]["question"] == q["question"], f"{path.name} {q['id']}: different question"
        jobs.append((tag, rows, out_pq, out_sum))

    retriever = HybridRetriever(kvkk_rag.corpus_texts())    # default stem = prefix5, as in the sanity check
    for tag, rows, out_pq, out_sum in jobs:
        per_q = []
        for q in questions:
            r = rows[q["id"]]
            row = {"id": q["id"], "rewrite": r["rewrite"], "latency_s": r.get("latency_s"),
                   "truncated": r.get("truncated")}
            for system, qs in {"rw": [r["rewrite"]], "fuse": [q["question"], r["rewrite"]]}.items():
                ranked = [c["text"] for c in retriever.search_multi(qs, k=depth)]
                for k in KS:
                    row[f"{system} R@{k}"] = recall_at_k(ranked, facts(q), k)
            per_q.append(row)
        per_q = pd.DataFrame(per_q)
        lat = per_q["latency_s"].dropna()
        summary = pd.DataFrame([{
            "tag": tag, "n": len(per_q), "system": system,
            **{f"R@{k}": round(per_q[f"{system} R@{k}"].mean(), 2) for k in KS},
            "latency_p50_s": round(lat.quantile(0.5), 3) if len(lat) else None,
            "latency_p95_s": round(lat.quantile(0.95), 3) if len(lat) else None,
            "truncated": int(per_q["truncated"].fillna(False).sum()),
        } for system in ("rw", "fuse")])
        print(summary.to_string(index=False), "\n")
        per_q.to_csv(out_pq, index=False)
        summary.to_csv(out_sum, index=False)
        print(f"wrote {out_pq.name}, {out_sum.name}\n")


if __name__ == "__main__":
    main()
