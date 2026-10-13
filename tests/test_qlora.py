"""Training examples must be the inference prompt plus the teacher rewrite; loss on the rewrite only.

Uses the students' tokenizers from the local Hugging Face cache (no download); skipped when one is not cached.
"""
import pytest

import qlora
import student


@pytest.mark.parametrize("name", list(student.MODELS))
def test_encode_learns_only_the_rewrite(name):
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(student.MODELS[name], local_files_only=True)
    except OSError:
        pytest.skip(f"{name} tokenizer not in the local cache")
    pair = qlora.load_pairs("train")[0]
    ex = qlora.encode(tok, pair["question"], pair["rewrite"], qlora.CONFIG["max_len"])
    prompt = tok.apply_chat_template(student.messages(pair["question"]), add_generation_prompt=True,
                                     return_dict=True)["input_ids"]
    assert ex["input_ids"][:len(prompt)] == prompt
    assert ex["labels"][:len(prompt)] == [-100] * len(prompt)
    learned = ex["input_ids"][len(prompt):]
    assert ex["labels"][len(prompt):] == learned
    assert learned[-1] in student.stop_ids(tok)
    assert tok.decode(learned[:-1]).strip() == pair["rewrite"]


def test_splits():
    assert (len(qlora.load_pairs("train")), len(qlora.load_pairs("val"))) == (966, 52)
