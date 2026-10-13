"""Small-model (student) rewrites: the prompt, the models and greedy generation. Used by the Colab notebooks.

The prompt is kvkk-rag's rewrite prompt, read from the pinned submodule as text (importing rewrite.py would
pull in google-genai); tests/test_student.py checks that it equals rewrite.SYSTEM. torch and transformers are
imported only inside load() and rewrite(), so the prompt part runs without them.
Gate 4a rules (decoding, limits, what is recorded): docs/ROADMAP.md.
"""
import ast
import time
from pathlib import Path

RAG = Path(__file__).resolve().parents[1] / "external/kvkk-rag"
MAX_NEW_TOKENS = 256

# short name (used in file names) -> Hugging Face id
MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "kumru-2b": "vngrs-ai/Kumru-2B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}
# end-of-turn tokens besides the tokenizer's eos (Kumru's template closes turns with <|eot_id|>, its eos is <EOS>)
EXTRA_STOP = ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>")


def system_prompt() -> str:
    tree = ast.parse((RAG / "scripts/rewrite.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets] == ["SYSTEM"]:
            return ast.literal_eval(node.value)
    raise RuntimeError("SYSTEM not found in kvkk-rag rewrite.py")


SYSTEM = system_prompt()


def messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def normalize(text: str) -> str:
    # as kvkk-rag's rewrite(): whitespace only
    return " ".join(text.split())


def load(hf_id: str, token=None, device: str = "cuda"):
    """(model, tokenizer, stop token ids) in bf16 on the GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(hf_id, token=token, dtype=torch.bfloat16, device_map=device)
    model.eval()
    vocab = tok.get_vocab()
    stop = sorted({tok.eos_token_id, *(vocab[t] for t in EXTRA_STOP if t in vocab)} - {None})
    return model, tok, stop


def rewrite(model, tok, stop: list[int], question: str) -> dict:
    """Greedy rewrite of one question: {"rewrite", "raw", "new_tokens", "truncated", "latency_s"}."""
    import torch

    ids = tok.apply_chat_template(messages(question), add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)["input_ids"].to(model.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else stop[0]
    sync = torch.cuda.synchronize if model.device.type == "cuda" else (lambda: None)
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             eos_token_id=stop, pad_token_id=pad)
    sync()
    latency = time.perf_counter() - t0
    new = out[0, ids.shape[1]:].tolist()
    ended = bool(new) and new[-1] in stop
    raw = tok.decode(new, skip_special_tokens=True)
    return {"rewrite": normalize(raw), "raw": raw, "new_tokens": len(new),
            "truncated": not ended and len(new) >= MAX_NEW_TOKENS, "latency_s": round(latency, 3)}
