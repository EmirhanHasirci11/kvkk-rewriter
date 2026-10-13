"""The student prompt must be kvkk-rag's rewrite prompt, unchanged. No model is loaded."""
import rewrite as rag_rewrite    # kvkk-rag's module (its `from llm import generate` resolves to ours)
import student


def test_system_prompt_equals_kvkk_rag():
    assert student.SYSTEM == rag_rewrite.SYSTEM


def test_messages_and_normalize():
    assert student.messages("soru?") == [{"role": "system", "content": rag_rewrite.SYSTEM},
                                         {"role": "user", "content": "soru?"}]
    assert student.normalize("  yeniden\n yazılmış   sorgu ") == "yeniden yazılmış sorgu"
