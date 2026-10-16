"""Student rewrites from a GGUF model with llama.cpp on the local GPU (docs/ROADMAP.md, gate 7). No LLM call.

Run from the repo root: python scripts/gguf_rewrites.py models/qlora_v2_llama3.1-8b-Q6_K.gguf [--set dev|test_v3]

Starts tools/llama.cpp-b11696/llama-server.exe (prebuilt CUDA 13.4 release; the Colab conversion used commit
2a667032dc, a few hours newer, which changes only Mistral conversion) with every layer on the GPU, and rewrites
each question of the set with the gate 4a decoding: the prompt is the HF chat template of the base model,
tokenized by its HF tokenizer and sent to the server as token ids (so the model reads exactly the tokens the bf16
run read; the server must not add a BOS of its own, which is checked), greedy (temperature 0), at most 256 new
tokens, stop at an end-of-generation token, whitespace normalization only. One warm-up question first (the first
training pair, as on Colab); prompt caching is off so every question is computed from scratch.

Writes results/rewrites_<gguf stem>_<set>_v1.jsonl with the fields of the Colab rewrites, plus the GPU memory in
use after loading (results/<same stem>_meta.json). Never overwrites.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import kvkk_rag
import qlora
import student
from golden import load_golden

SERVER = kvkk_rag.ROOT / "tools/llama.cpp-b11696/llama-server.exe"
BUILD = "b11696"
PORT = 8089
NAME = "llama3.1-8b"
SETS = {"dev": kvkk_rag.DEV, "test_v3": kvkk_rag.TEST_V3}


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def wait_ready(proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except OSError:
            pass
        time.sleep(1)
    sys.exit("llama-server did not become ready")


def gpu_used_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--set", choices=SETS, default="dev")
    args = parser.parse_args()
    stem = args.gguf.stem
    out = kvkk_rag.RESULTS / f"rewrites_{stem}_{args.set}_v1.jsonl"
    out_meta = kvkk_rag.RESULTS / f"{stem}_{args.set}_v1_meta.json"
    if out.exists() or out_meta.exists():
        sys.exit(f"{out.name} or {out_meta.name} already exists")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(student.MODELS[NAME])
    questions = load_golden(SETS[args.set])
    warmup = qlora.load_pairs("train", kvkk_rag.ROOT / "data/pairs_v2.jsonl")[0]["question"]

    gpu_before = gpu_used_mb()
    proc = subprocess.Popen([str(SERVER), "-m", str(args.gguf), "-ngl", "99", "-c", "2048", "-np", "1",
                             "--port", str(PORT), "--host", "127.0.0.1"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(proc)
        gpu_loaded = gpu_used_mb()

        def rewrite(question: str) -> dict:
            ids = tok.apply_chat_template(student.messages(question), add_generation_prompt=True,
                                          return_dict=True)["input_ids"]
            t0 = time.perf_counter()
            r = post("/completion", {"prompt": ids, "n_predict": student.MAX_NEW_TOKENS, "temperature": 0,
                                     "cache_prompt": False, "stream": False})
            latency = time.perf_counter() - t0
            assert r["tokens_evaluated"] == len(ids), f"server read {r['tokens_evaluated']} tokens, sent {len(ids)}"
            raw = r["content"]
            return {"rewrite": student.normalize(raw), "raw": raw, "new_tokens": r["tokens_predicted"],
                    "truncated": r["stop_type"] == "limit", "stop_type": r["stop_type"], "latency_s": round(latency, 3)}

        rewrite(warmup)
        rows = []
        for i, q in enumerate(questions, 1):
            r = rewrite(q["question"])
            rows.append({"id": q["id"], "question": q["question"], **r, "model": NAME, "gguf": args.gguf.name,
                         "llama_cpp": BUILD})
            print(f"[{i:2}/{len(questions)}] {r['latency_s']:.2f}s {r['new_tokens']:3} tok {r['stop_type']}  {r['rewrite'][:110]}")
    finally:
        proc.terminate()
        proc.wait()

    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    lat = sorted(r["latency_s"] for r in rows)
    meta = {"gguf": args.gguf.name, "gguf_bytes": args.gguf.stat().st_size, "llama_cpp": BUILD,
            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip(),
            "gpu_mb_before_load": gpu_before, "gpu_mb_after_load": gpu_loaded,
            "latency_p50_s": lat[len(lat) // 2], "truncated": sum(r["truncated"] for r in rows),
            "stop_types": sorted({r["stop_type"] for r in rows})}
    out_meta.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print(f"wrote {out.name}, {out_meta.name}")


if __name__ == "__main__":
    main()
