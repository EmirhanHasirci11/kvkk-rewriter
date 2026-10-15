"""Rewrites in kvkk-rag's full retrieval pipeline, with the reranker (docs/ROADMAP.md, gate 6). No LLM call.

Run from the repo root: python scripts/eval_pipeline.py [--tag pipeline_v1]

Pipeline: kvkk-rag's frozen final retrieval (FINAL in its scripts/run_heldout.py): question + rewrite through the
e5 + BM25 hybrid with bm25-snow, RRF, top-30, then bge-reranker-v2-m3 scoring the original question; recall on
the reranked list, plus R@30 of the candidate pool before reranking. Systems: no rewrite (question alone), the
Gemini teacher (kvkk-rag's cached rewrites) and the three gate 4b students (results/rewrites_qlora_v1_*).
Sets: dev (50 q) and held-out (20 q). The teacher rows must equal kvkk-rag's saved per-question numbers, or the
run stops. Writes results/<tag>_per_question.csv and results/<tag>_summary.csv (never overwrites).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import kvkk_rag
from golden import facts, load_golden, recall_at_k
from reranker import Reranker
from retriever import HybridRetriever

KS = (1, 3, 5, 10)
N_CAND = 30
STUDENTS = ("kumru-2b", "qwen3-4b", "llama3.1-8b")
# set -> (questions, saved kvkk-rag per-question file, its column prefix for the teacher's final pipeline)
SETS = {
    "dev": (kvkk_rag.DEV, "results/stemming_v1_per_question.csv", "pipe-snow"),
    "heldout": (kvkk_rag.HELDOUT, "results/heldout_v2_retrieval_per_question.csv", "fuse+bge+snow (final)"),
}


def load_rewrites(path: Path) -> dict[str, str]:
    return {r["id"]: r["rewrite"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="pipeline_v1")
    tag = parser.parse_args().tag
    out_pq, out_sum = (kvkk_rag.RESULTS / f"{tag}_{x}.csv" for x in ("per_question", "summary"))
    if out_pq.exists() or out_sum.exists():
        sys.exit(f"{out_pq.name} or {out_sum.name} already exists; use a new --tag")

    retriever = HybridRetriever(kvkk_rag.corpus_texts(), stem="snow")
    reranker = Reranker()
    rows = []
    for set_name, (qpath, saved_path, saved_col) in SETS.items():
        rewrites = {"teacher": load_rewrites(kvkk_rag.TEACHER_REWRITES[set_name]),
                    **{s: load_rewrites(kvkk_rag.RESULTS / f"rewrites_qlora_v1_{s}_{set_name}_v1.jsonl") for s in STUDENTS}}
        for q in load_golden(qpath):
            row = {"set": set_name, "id": q["id"]}
            for system in ("orig", *rewrites):
                queries = [q["question"]] + ([] if system == "orig" else [rewrites[system][q["id"]]])
                cands = retriever.search_multi(queries, k=N_CAND)
                row[f"{system} R@{N_CAND}"] = recall_at_k([c["text"] for c in cands], facts(q), N_CAND)
                ranked = [c["text"] for c in reranker.rerank(q["question"], cands)]
                for k in KS:
                    row[f"{system} R@{k}"] = recall_at_k(ranked, facts(q), k)
            rows.append(row)
        print(f"{set_name} done", flush=True)

        # the teacher's rows must reproduce kvkk-rag's final pipeline exactly
        mine = pd.DataFrame([r for r in rows if r["set"] == set_name]).set_index("id")
        saved = pd.read_csv(kvkk_rag.RAG / saved_path).set_index("id").loc[mine.index]
        for k in KS:
            diff = (mine[f"teacher R@{k}"] - saved[f"{saved_col} R@{k}"]).abs().max()
            assert diff < 1e-9, f"{set_name}: teacher R@{k} differs from {saved_col} in {saved_path}"
        print(f"{set_name}: teacher reproduces {saved_col} in kvkk-rag {saved_path}", flush=True)

    per_q = pd.DataFrame(rows)
    systems = ["orig", "teacher", *STUDENTS]
    summary = pd.DataFrame([{"set": s, "n": int((per_q["set"] == s).sum()), "system": system,
                             **{f"R@{k}": per_q.loc[per_q["set"] == s, f"{system} R@{k}"].mean() for k in KS + (N_CAND,)}}
                            for s in SETS for system in systems]).round(2)
    per_q.to_csv(out_pq, index=False)
    summary.to_csv(out_sum, index=False)
    print(summary.to_string(index=False))
    for s in SETS:
        d = per_q[per_q["set"] == s]
        for system in ("orig", *STUDENTS):
            a, b = d[f"{system} R@5"], d["teacher R@5"]
            print(f"{s} {system} vs teacher at R@5: gained {d.loc[a > b, 'id'].tolist()}, lost {d.loc[a < b, 'id'].tolist()}")
    print(f"wrote {out_pq.name}, {out_sum.name}")


if __name__ == "__main__":
    main()
