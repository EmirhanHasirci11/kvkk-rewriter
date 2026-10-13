"""QLoRA training of a student on the teacher rewrites, and loading the merged model. Used by the Colab notebooks.

Rules (data, hyperparameters, which adapter is used): docs/ROADMAP.md, gate 4b. Each training example is the
inference prompt exactly as student.rewrite() builds it, followed by the teacher rewrite and the chat template's
end-of-turn token; the loss is on that continuation only. torch / transformers / peft / bitsandbytes are
imported inside the functions, so encode() can be tested with a tokenizer alone.
"""
import json
import math
import random
import time
from pathlib import Path

import student

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data/pairs_v1.jsonl"
CONFIG = {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "lr": 2e-4, "warmup_ratio": 0.03, "epochs": 3,
          "batch_size": 8, "grad_accum": 2, "max_len": 512, "seed": 42}


def load_pairs(split: str) -> list[dict]:
    return [r for r in map(json.loads, PAIRS.read_text(encoding="utf-8").splitlines()) if r and r["split"] == split]


def encode(tok, question: str, rewrite: str, max_len: int) -> dict:
    """{"input_ids", "labels"}: prompt tokens get label -100, the rewrite and end-of-turn tokens are learned."""
    prompt = tok.apply_chat_template(student.messages(question), add_generation_prompt=True, return_dict=True)["input_ids"]
    full = tok.apply_chat_template(student.messages(question) + [{"role": "assistant", "content": rewrite}],
                                   return_dict=True)["input_ids"]
    if full[:len(prompt)] != prompt:
        raise ValueError("chat template: the training text does not start with the inference prompt")
    stops = set(student.stop_ids(tok))
    end = max(i for i, t in enumerate(full) if t in stops) + 1    # drop what the template adds after end-of-turn
    full = full[:end][:max_len]
    return {"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt):]}


def collate(batch: list[dict], pad_id: int):
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
        labels[i, :len(b["labels"])] = torch.tensor(b["labels"])
        mask[i, :len(b["input_ids"])] = 1
    return ids, labels, mask


def train(hf_id: str, out_dir: Path, token=None, cfg: dict = CONFIG, max_steps: int | None = None, log=print) -> dict:
    """Train a LoRA adapter on the 4-bit base, save it to out_dir, return the training record."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(hf_id, token=token)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else student.stop_ids(tok)[0]
    train_set = [encode(tok, r["question"], r["rewrite"], cfg["max_len"]) for r in load_pairs("train")]
    val_set = [encode(tok, r["question"], r["rewrite"], cfg["max_len"]) for r in load_pairs("val")]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(hf_id, token=token, quantization_config=bnb, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                             lora_dropout=cfg["lora_dropout"], target_modules="all-linear",
                                             task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])
    steps_per_epoch = math.ceil(len(train_set) / (cfg["batch_size"] * cfg["grad_accum"]))
    total = steps_per_epoch * cfg["epochs"] if max_steps is None else max_steps
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(cfg["warmup_ratio"] * total), total)

    def loss_on(ids, labels, mask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda()).loss

    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_set), cfg["batch_size"]):
                batch = val_set[i:i + cfg["batch_size"]]
                ids, labels, mask = collate(batch, pad_id)
                k = int((labels[:, 1:] != -100).sum())    # tokens that carry loss (labels are shifted by one)
                tot += loss_on(ids, labels, mask).item() * k
                n += k
        model.train()
        return tot / n

    torch.cuda.reset_peak_memory_stats()
    record = {"hf_id": hf_id, "config": cfg, "trainable_params": trainable, "train_n": len(train_set),
              "val_n": len(val_set), "steps": total, "val_loss": [round(val_loss(), 4)], "train_loss": []}
    log(f"trainable {trainable:,}; {len(train_set)} train / {len(val_set)} val; {total} steps; "
        f"val loss before training {record['val_loss'][0]}")
    t0 = time.time()
    model.train()
    step = 0
    for epoch in range(cfg["epochs"]):
        order = list(range(len(train_set)))
        random.shuffle(order)
        micro = [order[i:i + cfg["batch_size"]] for i in range(0, len(order), cfg["batch_size"])]
        for j in range(0, len(micro), cfg["grad_accum"]):
            group = micro[j:j + cfg["grad_accum"]]
            k_all = sum(sum(len(train_set[i]["input_ids"]) - train_set[i]["labels"].count(-100) for i in mb)
                        for mb in group)
            loss_sum = 0.0
            for mb in group:    # token-weighted average over the accumulated micro-batches
                ids, labels, mask = collate([train_set[i] for i in mb], pad_id)
                k = int((labels[:, 1:] != -100).sum())
                loss = loss_on(ids, labels, mask) * k / k_all
                loss.backward()
                loss_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            record["train_loss"].append(round(loss_sum, 4))
            if step % 10 == 0 or step == total:
                log(f"epoch {epoch + 1} step {step}/{total} loss {loss_sum:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{(time.time() - t0) / 60:.1f} min")
            if step >= total:
                break
        record["val_loss"].append(round(val_loss(), 4))
        log(f"epoch {epoch + 1} val loss {record['val_loss'][-1]}")
        if step >= total:
            break
    record["train_min"] = round((time.time() - t0) / 60, 2)
    record["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    (out_dir / "train_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    del model, opt
    torch.cuda.empty_cache()
    return record


def load_merged(hf_id: str, adapter_dir: Path, token=None, device: str = "cuda"):
    """(model, tokenizer, stop ids): the bf16 base with the adapter merged in, ready for student.rewrite()."""
    from peft import PeftModel

    model, tok, stop = student.load(hf_id, token=token, device=device)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.eval()
    return model, tok, stop
