"""Calibrate the leakage filter thresholds on the eval questions themselves. No LLM call.

Run from the repo root: python scripts/calibrate_leakage.py [--tag v1]

Reports, for the 85 distinct eval questions (golden, test_v2, abstention sets):
  - nearest other eval question: e5 cosine and word 3-gram Jaccard (distinct questions = should pass)
  - each question against its own cached teacher rewrite: e5 cosine (same meaning, other words)
  - how many eval questions the current thresholds would flag against the other eval questions
Writes results/leakage_calibration_<tag>.json (never overwrites).
"""
import argparse
import json
import sys

import numpy as np

import datagen
import kvkk_rag


def pct(x) -> dict:
    return {f"p{p}": round(float(np.percentile(x, p)), 4) for p in (10, 50, 90, 99)} | {"max": round(float(np.max(x)), 4)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()
    out = kvkk_rag.RESULTS / f"leakage_calibration_{args.tag}.json"
    if out.exists():
        sys.exit(f"{out.name} already exists; use a new --tag")

    f = datagen.LeakageFilter()
    qs = [q["question"] for q in f.evals]
    sims = f.eval_vecs @ f.eval_vecs.T
    np.fill_diagonal(sims, -1)
    nn_cos = sims.max(1)
    nn_jac = [max(datagen._jaccard(f.eval_grams[a], f.eval_grams[b]) for b in range(len(qs)) if b != a)
              for a in range(len(qs))]
    i, j = np.unravel_index(sims.argmax(), sims.shape)

    rws = [r for p in kvkk_rag.TEACHER_REWRITES.values()
           for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r]
    own = (f._encode([r["question"] for r in rws]) * f._encode([r["rewrite"] for r in rws])).sum(1)

    report = {
        "thresholds": {"ngram_n": datagen.NGRAM_N, "ngram_max": datagen.NGRAM_MAX, "cosine_max": datagen.COSINE_MAX},
        "n_eval_questions": len(qs),
        "nearest_other_eval_cosine": pct(nn_cos),
        "nearest_other_eval_ngram_jaccard": pct(nn_jac),
        "most_similar_eval_pair": [qs[i], qs[j], round(float(sims[i, j]), 4)],
        "question_vs_own_teacher_rewrite_cosine": pct(own),
        "eval_questions_flagged_vs_other_eval": int(((nn_cos >= datagen.COSINE_MAX)
                                                     | (np.array(nn_jac) >= datagen.NGRAM_MAX)).sum()),
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
