"""The teacher request must equal kvkk-rag's rewrite request. No LLM call: both clients are fakes."""
import importlib.util
from types import SimpleNamespace

import kvkk_rag
import llm


class FakeClient:
    def __init__(self):
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=1, thoughts_token_count=0)
        return SimpleNamespace(text=" yeniden   yazılmış  sorgu ", usage_metadata=usage)


def load_rag_module(name):
    spec = importlib.util.spec_from_file_location(f"rag_{name}", kvkk_rag.RAG / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_teacher_request_equals_kvkk_rag(monkeypatch, tmp_path):
    rag_llm = load_rag_module("llm")
    rag_fake, our_fake = FakeClient(), FakeClient()
    monkeypatch.setattr(rag_llm, "_client", rag_fake)
    monkeypatch.setattr(rag_llm, "_log_usage", lambda row: None)
    monkeypatch.setattr(rag_llm, "spent_usd", lambda: 0.0)
    monkeypatch.setattr(llm, "_client", our_fake)
    monkeypatch.setattr(llm, "USAGE_CSV", tmp_path / "usage.csv")

    import gen_rewrites
    question = "Müşterimiz verilerini silmemizi istedi, ne yapmalıyız?"
    rag_out = rag_llm.generate(question, gen_rewrites.SYSTEM)
    ours = gen_rewrites.teacher_rewrite(question)

    (a,), (b,) = rag_fake.calls, our_fake.calls
    assert a["model"] == b["model"] == rag_llm.MODEL == llm.MODEL
    assert a["contents"] == b["contents"] == question
    assert a["config"].model_dump() == b["config"].model_dump()
    assert ours["text"] == " ".join(rag_out["text"].split()) == "yeniden yazılmış sorgu"
    assert llm.PRICES == rag_llm.PRICES


def test_teacher_prompt_is_kvkk_rag_prompt():
    import gen_rewrites
    src = (kvkk_rag.RAG / "scripts/rewrite.py").read_text(encoding="utf-8")
    assert gen_rewrites.SYSTEM in src
    assert gen_rewrites.SYSTEM.startswith("Görevin, kullanıcının kişisel verilerin korunmasıyla ilgili sorusunu")
