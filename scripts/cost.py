"""Cost and latency per question: Gemini teacher vs the QLoRA students. No LLM call, no GPU.

Run from the repo root: python scripts/cost.py [--tag cost_v1]

Teacher: kvkk-rag's cached rewrites of the dev + held-out questions carry each call's cost (gemini-3.8-flash,
default thinking); its latency comes from this project's 1,023 teacher-rewrite calls in results/llm_usage.csv
(same model and prompt, API wall clock from a home connection).
Students: GPU time per question on the Colab L4 (the latency of generate(), dev + held-out, 70 questions) priced
at CU_PER_HOUR x USD_PER_CU. This is the marginal cost with the GPU busy; model loading, setup and idle time are
not in it (the real sessions are in results/colab_usage.csv). One-off cost: the training data (all LLM spend in
results/llm_usage.csv, shared by the three students) plus each student's training time at the same rate.
Break-even: the number of questions after which the one-off cost is paid back by the saving per question.
Writes results/<tag>_summary.csv (never overwrites).
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAG_RESULTS = ROOT / "external/kvkk-rag/results"
STUDENTS = ("kumru-2b", "qwen3-4b", "llama3.1-8b")

CU_PER_HOUR = 1.4     # Colab L4 burn rate, as shown to the user (2026-10-15)
USD_PER_CU = 0.10     # Colab pay-as-you-go list price, $9.99 per 100 CU (assumed; not the user's own invoice)


def pct(values: list[float], q: float) -> float:
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(q * 100) - 1], 2)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="cost_v1")
    out = RESULTS / f"{parser.parse_args().tag}_summary.csv"
    if out.exists():
        sys.exit(f"{out.name} already exists; use a new --tag")

    usage = list(csv.DictReader((RESULTS / "llm_usage.csv").open(encoding="utf-8")))
    data_usd = sum(float(r["cost_usd"]) for r in usage)
    teacher_lat = [float(r["latency_s"]) for r in usage if r["purpose"] == "teacher_rewrite"]
    teacher_cost = [json.loads(l)["cost_usd"] for f in ("rewrites_v1.jsonl", "rewrites_heldout_v2.jsonl")
                    for l in (RAG_RESULTS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    teacher_usd = statistics.mean(teacher_cost)

    rows = [{"system": "gemini-teacher", "n": len(teacher_cost), "latency_p50_s": pct(teacher_lat, 0.5),
             "latency_p95_s": pct(teacher_lat, 0.95), "usd_per_question": round(teacher_usd, 6),
             "usd_per_1000": round(1000 * teacher_usd, 2), "one_off_usd": None, "break_even_questions": None}]
    for name in STUDENTS:
        lat = []
        for s in ("dev", "heldout"):
            with (RESULTS / f"eval_qlora_v1_{name}_{s}_v1_per_question.csv").open(encoding="utf-8") as f:
                lat += [float(r["latency_s"]) for r in csv.DictReader(f)]
        usd = statistics.mean(lat) / 3600 * CU_PER_HOUR * USD_PER_CU
        train_min = json.loads((RESULTS / f"train_record_qlora_v1_{name}.json").read_text(encoding="utf-8"))["train_min"]
        one_off = data_usd + train_min / 60 * CU_PER_HOUR * USD_PER_CU
        rows.append({"system": name, "n": len(lat), "latency_p50_s": pct(lat, 0.5), "latency_p95_s": pct(lat, 0.95),
                     "usd_per_question": round(usd, 6), "usd_per_1000": round(1000 * usd, 2),
                     "one_off_usd": round(one_off, 2), "break_even_questions": round(one_off / (teacher_usd - usd))})

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"\nCU_PER_HOUR {CU_PER_HOUR}, USD_PER_CU {USD_PER_CU}, training data ${data_usd:.3f}; wrote {out.name}")


if __name__ == "__main__":
    main()
