"""Cost and latency: Gemini teacher vs the QLoRA students. No LLM call, no GPU.

Run from the repo root: python scripts/cost.py [--tag cost_v2]

Teacher: pay per call. kvkk-rag's cached rewrites of the dev + held-out questions carry each call's cost
(gemini-3.8-flash, default thinking); its latency comes from this project's 1,023 teacher-rewrite calls in
results/llm_usage.csv (same model and prompt, API wall clock from a home connection).
Students: pay per GPU hour. The Colab L4 is billed by the started hour (CU_PER_HOUR x USD_PER_CU), however few
questions run in it. Capacity per hour is 3600 s over the mean generate() latency on the L4 (dev + held-out, 70 q,
batch 1, one question after another); model loading (under 2.5 min per model in gate 5) is left out.
A batch of N questions then costs ceil(N / capacity) hours; a GPU kept on for live traffic costs 24 hours a day.
The GPU is cheaper than the teacher once a started hour holds more than BREAK_EVEN questions.
One-off: the training data (all LLM spend in results/llm_usage.csv) plus the gate 4b Colab CU, for all three.
Writes results/<tag>_summary.csv (never overwrites).
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAG_RESULTS = ROOT / "external/kvkk-rag/results"
STUDENTS = ("kumru-2b", "qwen3-4b", "llama3.1-8b")
BATCHES = (20, 100, 1000, 10000)

CU_PER_HOUR = 1.4     # Colab L4 rate, as shown to the user (2026-10-15)
USD_PER_CU = 0.10     # Colab pay-as-you-go list price, $9.99 per 100 CU (assumed; not the user's own invoice)
USD_PER_HOUR = CU_PER_HOUR * USD_PER_CU


def pct(values: list[float], q: float) -> float:
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(q * 100) - 1], 2)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="cost_v2")
    out = RESULTS / f"{parser.parse_args().tag}_summary.csv"
    if out.exists():
        sys.exit(f"{out.name} already exists; use a new --tag")

    usage = list(csv.DictReader((RESULTS / "llm_usage.csv").open(encoding="utf-8")))
    data_usd = sum(float(r["cost_usd"]) for r in usage)
    train_cu = sum(float(r["compute_units"]) for r in csv.DictReader((RESULTS / "colab_usage.csv").open(encoding="utf-8"))
                   if r["notebook"] == "notebooks/02_qlora.ipynb")
    teacher_lat = [float(r["latency_s"]) for r in usage if r["purpose"] == "teacher_rewrite"]
    teacher_cost = [json.loads(l)["cost_usd"] for f in ("rewrites_v1.jsonl", "rewrites_heldout_v2.jsonl")
                    for l in (RAG_RESULTS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    teacher_usd = statistics.mean(teacher_cost)

    rows = [{"system": "gemini-teacher", "n": len(teacher_cost), "latency_p50_s": pct(teacher_lat, 0.5),
             "latency_p95_s": pct(teacher_lat, 0.95), "billing": f"{teacher_usd:.6f} USD per question",
             "capacity_per_hour": None,
             **{f"usd_{b}_questions": round(b * teacher_usd, 2) for b in BATCHES},
             "usd_per_day_always_on": None}]
    for name in STUDENTS:
        lat = []
        for s in ("dev", "heldout"):
            with (RESULTS / f"eval_qlora_v1_{name}_{s}_v1_per_question.csv").open(encoding="utf-8") as f:
                lat += [float(r["latency_s"]) for r in csv.DictReader(f)]
        capacity = int(3600 / statistics.mean(lat))
        rows.append({"system": name, "n": len(lat), "latency_p50_s": pct(lat, 0.5), "latency_p95_s": pct(lat, 0.95),
                     "billing": f"{USD_PER_HOUR:.2f} USD per started L4 hour", "capacity_per_hour": capacity,
                     **{f"usd_{b}_questions": round(math.ceil(b / capacity) * USD_PER_HOUR, 2) for b in BATCHES},
                     "usd_per_day_always_on": round(24 * USD_PER_HOUR, 2)})

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"\nL4 {USD_PER_HOUR:.2f} USD per started hour ({CU_PER_HOUR} CU x {USD_PER_CU} USD); the GPU beats the teacher "
          f"above {USD_PER_HOUR / teacher_usd:.0f} questions per started hour ({24 * USD_PER_HOUR / teacher_usd:.0f} a day "
          f"if always on). One-off: training data {data_usd:.3f} USD + gate 4b {train_cu:g} CU "
          f"({train_cu * USD_PER_CU:.2f} USD). Wrote {out.name}")


if __name__ == "__main__":
    main()
