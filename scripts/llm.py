"""This project's LLM interface, separate from kvkk-rag's: own usage log, own budget.

Backend: Gemini via the google-genai SDK. The key is read only from GEMINI_API_KEY in this repo's .env
(git-ignored); a key set elsewhere in the environment is never picked up. Every call is appended to
results/llm_usage.csv, and no call is made once the logged total reaches BUDGET_USD.

generate() with its defaults sends exactly the request kvkk-rag's scripts/llm.py sends (same model,
temperature 0, no thinking config, AFC disabled), so the teacher is the same; tests/test_teacher.py checks it.
"""
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
USAGE_CSV = ROOT / "results/llm_usage.csv"
USAGE_FIELDS = ["date", "script", "purpose", "model", "thinking_level", "input_tokens", "output_tokens",
                "cost_usd", "latency_s"]

MODEL = "gemini-3.8-flash"
# USD per 1M tokens, paid tier, valid through 2026-12-31 (both double on 2027-01-01); same as kvkk-rag.
# Thinking tokens are billed as output.
PRICES = {"gemini-3.8-flash": {"input": 0.75, "output": 3.75}}
BUDGET_USD = 6.0


class BudgetExceeded(RuntimeError):
    pass


def api_key() -> str:
    """The key from .env, or exit with a message saying it is missing."""
    key = dotenv_values(ROOT / ".env").get("GEMINI_API_KEY") if (ROOT / ".env").exists() else None
    if not key:
        sys.exit(f"GEMINI_API_KEY is missing in {ROOT / '.env'}; add it there (the file is git-ignored). Stopping.")
    return key


def spent_usd(purpose: str | None = None) -> float:
    return sum(float(r["cost_usd"]) for r in usage_rows(purpose))


def usage_rows(purpose: str | None = None) -> list[dict]:
    if not USAGE_CSV.exists():
        return []
    with USAGE_CSV.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if purpose is None or r["purpose"] == purpose]


def cost_usd(input_tokens: int, output_tokens: int, model: str = MODEL) -> float:
    p = PRICES[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def confirm_spend(what: str, n_calls: int, mean_usd: float, max_usd: float) -> float:
    """Print the estimate and wait for "yes". Returns the approved amount (the upper estimate)."""
    expected, upper = n_calls * mean_usd, n_calls * max_usd
    spent = spent_usd()
    print(f"\n{what}: {n_calls} LLM calls ({MODEL})")
    print(f"  estimated cost: ${expected:.4f} (upper ${upper:.4f}, from ${mean_usd:.5f} mean / ${max_usd:.5f} max per call)")
    print(f"  spent so far in this project: ${spent:.4f} of ${BUDGET_USD:.2f} (results/llm_usage.csv)")
    if spent + upper > BUDGET_USD:
        sys.exit(f"refused: the upper estimate would exceed the ${BUDGET_USD:.2f} budget; use a smaller --limit")
    try:
        answer = input(f"Type 'yes' to spend up to ${upper:.4f}: ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "yes":
        sys.exit("not confirmed, nothing was sent")
    return upper


class RunCap:
    """Stops a run once its own spending passes the approved amount (with 10% slack for estimate noise)."""

    def __init__(self, approved_usd: float):
        self.limit = approved_usd * 1.1
        self.spent = 0.0

    def add(self, cost: float):
        self.spent += cost
        if self.spent > self.limit:
            raise BudgetExceeded(f"run spent ${self.spent:.4f}, more than the approved ${self.limit:.4f}; stopping")


def _log_usage(row: dict):
    new = not USAGE_CSV.exists()
    USAGE_CSV.parent.mkdir(exist_ok=True)
    with USAGE_CSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=USAGE_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=api_key())
    return _client


def request_config(system: str, temperature: float = 0, thinking_level: str | None = None, json_output: bool = False):
    """GenerateContentConfig; with the defaults it equals the one kvkk-rag's llm.generate builds."""
    from google.genai import types
    kw = {}
    if thinking_level is not None:
        kw["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    if json_output:
        kw["response_mime_type"] = "application/json"
    return types.GenerateContentConfig(
        system_instruction=system, temperature=temperature,
        # no tools are passed; this only silences the SDK's AFC warning
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), **kw)


def generate(prompt: str, system: str, *, purpose: str, temperature: float = 0,
             thinking_level: str | None = None, json_output: bool = False) -> dict:
    """{"text", "input_tokens", "output_tokens", "cost_usd", "latency_s"}."""
    spent = spent_usd()
    if spent >= BUDGET_USD:
        raise BudgetExceeded(f"LLM budget reached: ${spent:.4f} of ${BUDGET_USD:.2f} spent, see {USAGE_CSV.name}")

    config = request_config(system, temperature, thinking_level, json_output)
    t0 = time.perf_counter()
    resp = _get_client().models.generate_content(model=MODEL, contents=prompt, config=config)
    latency = time.perf_counter() - t0
    u = resp.usage_metadata
    input_tokens = u.prompt_token_count or 0
    output_tokens = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
    cost = cost_usd(input_tokens, output_tokens)

    _log_usage({"date": datetime.now().isoformat(timespec="seconds"), "script": Path(sys.argv[0]).name,
                "purpose": purpose, "model": MODEL, "thinking_level": thinking_level or "default",
                "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": f"{cost:.6f}",
                "latency_s": f"{latency:.2f}"})
    return {"text": resp.text or "", "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cost_usd": cost, "latency_s": latency}


def per_call_estimate(purpose: str, default_mean: float, default_max: float, min_rows: int = 2) -> tuple[float, float]:
    """(mean, max) cost per call from this project's own log once it has min_rows calls of this purpose."""
    rows = usage_rows(purpose)
    if len(rows) < min_rows:
        return default_mean, default_max
    costs = [float(r["cost_usd"]) for r in rows]
    return sum(costs) / len(costs), max(costs)


def n_calls(n_items: int, per_call: int) -> int:
    return math.ceil(n_items / per_call)
