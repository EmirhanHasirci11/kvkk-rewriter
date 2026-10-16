"""Teacher rewrites for the generated training questions (data plan step 2). COSTS MONEY.

Run from the repo root: python scripts/gen_rewrites.py --limit N [--tag v1]

The teacher is kvkk-rag's rewrite step, unchanged: the system prompt is imported from the pinned submodule
(external/kvkk-rag/scripts/rewrite.py, SYSTEM), the request is llm.generate with its defaults
(gemini-3.8-flash, temperature 0, the model's default thinking, AFC disabled; tests/test_teacher.py checks
that it equals kvkk-rag's request), and the output is whitespace-normalized as in kvkk-rag's rewrite().

Reads data/questions_<tag>.jsonl (questions with "leak": false only), one LLM call per question, and appends
to data/rewrites_<tag>.jsonl; questions already rewritten are skipped, so a run can be resumed.
Before any call: checks GEMINI_API_KEY in .env, prints the number of calls and the estimated cost
(from kvkk-rag's 85 cached teacher rewrites until this project has its own), and waits for "yes".
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import kvkk_rag
import llm
import rewrite as rag_rewrite    # kvkk-rag's module; its own `from llm import generate` resolves to ours

PURPOSE = "teacher_rewrite"
assert Path(rag_rewrite.__file__).resolve().parent == kvkk_rag.RAG / "scripts", "rewrite.py must come from the submodule"
SYSTEM = rag_rewrite.SYSTEM


def teacher_rewrite(question: str) -> dict:
    out = llm.generate(question, SYSTEM, purpose=PURPOSE)
    return {**out, "text": " ".join(out["text"].split())}


def kvkk_rag_estimate() -> tuple[float, float]:
    """(mean, max) cost of kvkk-rag's cached teacher rewrites (85 calls, same prompt and model)."""
    costs = [r["cost_usd"] for p in kvkk_rag.RAG.glob("results/rewrites_*.jsonl")
             for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r]
    return statistics.mean(costs), max(costs)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, required=True, help="rewrite at most N questions (pilot: 10)")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--eval", type=Path, help="gate 7: rewrite an eval file (e.g. eval/test_v3.jsonl) instead, "
                                                  "into results/rewrites_teacher_<stem>_v1.jsonl")
    args = parser.parse_args()
    llm.api_key()

    if args.eval:
        qpath = kvkk_rag.ROOT / args.eval
        out_path = kvkk_rag.RESULTS / f"rewrites_teacher_{qpath.stem}_v1.jsonl"
        questions = [{**q, "qid": q["id"], "chunk_id": None} for q in
                     map(json.loads, qpath.read_text(encoding="utf-8").splitlines()) if q]
    else:
        qpath = kvkk_rag.ROOT / f"data/questions_{args.tag}.jsonl"
        out_path = kvkk_rag.ROOT / f"data/rewrites_{args.tag}.jsonl"
        if not qpath.exists():
            sys.exit(f"{qpath} not found; run scripts/gen_questions.py first")
        questions = [q for q in map(json.loads, qpath.read_text(encoding="utf-8").splitlines())
                     if q and q["question"] and not q["leak"]]
    done = set()
    if out_path.exists():
        done = {r["qid"] for r in map(json.loads, out_path.read_text(encoding="utf-8").splitlines()) if r}
    todo = [q for q in questions if q["qid"] not in done][:args.limit]
    if not todo:
        sys.exit(f"nothing to do: {len(done)} of {len(questions)} questions already rewritten")

    mean, mx = llm.per_call_estimate(PURPOSE, *kvkk_rag_estimate())
    approved = llm.confirm_spend(f"Teacher rewrites ({len(todo)} questions, {len(done)} already done)",
                                 len(todo), mean, mx)
    cap = llm.RunCap(approved)
    with out_path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            out = teacher_rewrite(q["question"])
            row = {"qid": q["qid"], **({"id": q["qid"]} if args.eval else {}), "chunk_id": q["chunk_id"],
                   "question": q["question"], "rewrite": out["text"],
                   "model": llm.MODEL, "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"],
                   "cost_usd": out["cost_usd"], "latency_s": round(out["latency_s"], 2)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {q['qid']} ${out['cost_usd']:.4f} {out['latency_s']:.1f}s  {out['text'][:90]}")
            cap.add(out["cost_usd"])
    print(f"done: run cost ${cap.spent:.4f}; project total ${llm.spent_usd():.4f} of ${llm.BUDGET_USD:.2f}")


if __name__ == "__main__":
    main()
