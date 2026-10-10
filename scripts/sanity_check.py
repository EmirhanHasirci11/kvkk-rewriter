"""Sanity check: kvkk-rag's teacher numbers must reproduce exactly from the submodule. No LLM call.

Run from the repo root: python scripts/sanity_check.py [--tag sanity_v1]

Recomputes recall of three systems with the cached Gemini rewrites (no reranker, bm25-p5, e5 + BM25 RRF,
192/48, depth 50), with the same settings as the kvkk-rag script that wrote each saved file:
  dev      eval/golden.jsonl (50 q), rewrites_v1.jsonl;        as scripts/eval_rewrite.py (top-10)
  heldout  eval/test_v2.jsonl (20 q), rewrites_heldout_v2.jsonl; as scripts/run_heldout.py (top-30)
Systems: orig (question alone), rw (rewrite alone, "rw-only"), fuse (RRF over both).

Every question and every k must equal results/rewrite_v1_per_question.csv and
results/heldout_v2_retrieval_per_question.csv in the submodule, and the summary must equal the
kvkk-rag README (dev rw R@5 0.75, fuse R@5 0.71; held-out rw-only R@5 0.70). Exits non-zero otherwise.
Writes results/<tag>_per_question.csv and results/<tag>_summary.csv (never overwrites).
"""
import argparse
import json
import sys

import pandas as pd

import kvkk_rag
from golden import facts, load_golden, recall_at_k
from retriever import HybridRetriever

KS = (1, 3, 5, 10)
SYSTEMS = ("orig", "rw", "fuse")
# set -> (questions, cached rewrites, fused list depth, saved per-question CSV, its column for each system)
SETS = {
    "dev": (kvkk_rag.DEV, kvkk_rag.TEACHER_REWRITES["dev"], 10, "results/rewrite_v1_per_question.csv",
            {"orig": "orig", "rw": "rw", "fuse": "fuse"}),
    "heldout": (kvkk_rag.HELDOUT, kvkk_rag.TEACHER_REWRITES["heldout"], 30,
                "results/heldout_v2_retrieval_per_question.csv", {"orig": "orig", "rw": "rw-only", "fuse": "fuse"}),
}
# kvkk-rag README, rounded to 2 decimals
README = {("dev", "rw", 5): 0.75, ("dev", "fuse", 5): 0.71, ("heldout", "rw", 5): 0.70,
          ("dev", "orig", 5): 0.64, ("heldout", "orig", 5): 0.35, ("heldout", "fuse", 5): 0.60}


def load_rewrites(path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="sanity_v1")
    args = parser.parse_args()
    out_pq = kvkk_rag.RESULTS / f"{args.tag}_per_question.csv"
    out_sum = kvkk_rag.RESULTS / f"{args.tag}_summary.csv"
    if out_pq.exists() or out_sum.exists():
        sys.exit(f"{out_pq.name} or {out_sum.name} already exists; use a new --tag")

    retriever = HybridRetriever(kvkk_rag.corpus_texts())    # default stem = prefix5, as in both saved runs
    rows = []
    for set_name, (qpath, rwpath, depth, _, _) in SETS.items():
        rewrites = load_rewrites(rwpath)
        for q in load_golden(qpath):
            rw = rewrites[q["id"]]
            assert rw["question"] == q["question"], f"{q['id']}: cached rewrite is for a different question"
            queries = {"orig": [q["question"]], "rw": [rw["rewrite"]], "fuse": [q["question"], rw["rewrite"]]}
            row = {"set": set_name, "id": q["id"]}
            for system, qs in queries.items():
                ranked = [c["text"] for c in retriever.search_multi(qs, k=depth)]
                for k in KS:
                    row[f"{system} R@{k}"] = recall_at_k(ranked, facts(q), k)
            rows.append(row)
    per_q = pd.DataFrame(rows)

    failures = []
    for set_name, (_, _, _, saved_path, cols) in SETS.items():
        saved = pd.read_csv(kvkk_rag.RAG / saved_path).set_index("id")
        mine = per_q[per_q["set"] == set_name].set_index("id")
        for system, col in cols.items():
            for k in KS:
                diff = (mine[f"{system} R@{k}"] - saved.loc[mine.index, f"{col} R@{k}"]).abs()
                bad = diff[diff > 1e-9].index.tolist()
                if bad:
                    failures.append(f"{set_name} {system} R@{k} differs from {saved_path} on {bad}")

    summary = pd.DataFrame([{"set": s, "n": int((per_q["set"] == s).sum()), "system": system,
                             **{f"R@{k}": per_q.loc[per_q["set"] == s, f"{system} R@{k}"].mean() for k in KS}}
                            for s in SETS for system in SYSTEMS]).round(2)
    for (s, system, k), expected in README.items():
        got = summary.loc[(summary["set"] == s) & (summary["system"] == system), f"R@{k}"].item()
        if abs(got - expected) > 1e-9:
            failures.append(f"{s} {system} R@{k} = {got}, README says {expected}")

    print(summary.to_string(index=False))
    if failures:
        print("\nSANITY CHECK FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    kvkk_rag.RESULTS.mkdir(exist_ok=True)
    per_q.to_csv(out_pq, index=False)
    summary.to_csv(out_sum, index=False)
    print(f"\nOK: all {len(per_q)} questions x {len(SYSTEMS)} systems x {len(KS)} k reproduce the saved kvkk-rag "
          f"values exactly, and the summary matches the README. Wrote {out_pq.name}, {out_sum.name}.")


if __name__ == "__main__":
    main()
