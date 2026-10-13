# Small-model shortlist

Candidates for the student, 1-4B parameter instruct models on Hugging Face as of 2026-10-10. Facts come from
the model cards, `config.json` and the Hub API (parameter counts from the safetensors metadata); no weights
were downloaded.

**Chosen 2026-10-13 (user):** Qwen3-4B-Instruct-2507, Kumru-2B, and Llama-3.1-8B-Instruct (8B, outside the
shortlist range, the user has gated access) for the zero-shot baseline, on a Colab L4. Rules: `docs/ROADMAP.md`, gate 4a.

| | Qwen3.5-2B | Qwen3-4B-Instruct-2507 | Gemma 4 E2B-it | Kumru-2B |
|---|---|---|---|---|
| Hub id | `Qwen/Qwen3.5-2B` | `Qwen/Qwen3-4B-Instruct-2507` | `google/gemma-4-E2B-it` | `vngrs/Kumru-2B` (card code: `vngrs-ai/Kumru-2B`) |
| Released | 2026-02 | 2025-08 | 2026-03 | 2025-09 |
| Parameters | 2.27B total, incl. vision encoder (card: 2B LM) | 4.02B | 2.3B effective; 5.1B with per-layer embeddings, + ~150M vision, ~300M audio | 2.38B |
| Architecture | hybrid: Gated DeltaNet + gated attention, 24 layers, vocab 248K | dense transformer, 36 layers, vocab 152K | dense, sliding + global attention, 35 layers, vocab 262K, per-layer embeddings | Mistral, 18 layers, vocab 50K, Turkish tokenizer |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| Turkish | multilingual, "201 languages" (card); Turkish not named on the card | multilingual, 119 languages; Turkish named (Qwen3 blog) | multilingual, "35+ languages, pre-trained on 140+" (card); Turkish not named on the card | Turkish only, trained from scratch (300B tokens, 1M SFT examples) |
| Thinking | off by default (2B) | none (instruct-only) | off unless `<\|think\|>` is in the system prompt | none |
| Context | 262K | 262K | 128K | 8K (enough: prompt + question + rewrite is ~0.5K) |
| 4-bit weights, est. | ~2.2 GB | ~2.8 GB | ~7-8 GB | ~1.8 GB |
| QLoRA on T4 (16 GB) | fits by memory; see risk | fits | fits, least headroom | fits |
| QLoRA on L4 (24 GB) | fits | fits | fits | fits |
| Main risk | the DeltaNet layers want `flash-linear-attention` / `causal-conv1d` kernels; without them training falls back to slow PyTorch code, and T4 (sm_75) support is unverified | oldest candidate; Turkish quality at 4B unmeasured | 5.1B real parameters; per-layer embedding tables stay in 16-bit; fp16 on T4 (no bf16) unverified for Gemma 4 | small community; one benchmark (Cetvel) on the card; no instruction-following numbers |

Family requirements: Qwen and Gemma (multilingual) and Kumru-2B (Turkish-specific) are all in.

## How the memory was estimated

bitsandbytes NF4 quantizes the linear layers (~0.56 bytes per parameter with the quantization constants);
token embeddings, norms, the untied LM head and the vision / audio encoders stay in 16-bit (2 bytes).
From `config.json`:

- Qwen3.5-2B: embedding 248,320 x 2,048 = 0.51B (tied) -> 1.0 GB; rest ~1.8B -> ~1.0-1.3 GB.
- Qwen3-4B-Instruct-2507: embedding 151,936 x 2,560 = 0.39B (tied) -> 0.8 GB; rest 3.6B -> 2.0 GB.
- Gemma 4 E2B: per-layer embeddings 35 x 256 x 262,144 = 2.35B -> 4.7 GB; main embedding 262,144 x 1,536 =
  0.40B -> 0.8 GB; encoders ~0.45B -> 0.9 GB; text linears ~1.9B -> 1.1 GB.
- Kumru-2B: embedding + untied head 2 x 50,176 x 3,072 = 0.31B -> 0.6 GB; rest 2.07B -> 1.2 GB.

Training adds LoRA weights and optimizer state (tens of MB at r = 16) and activations; a training example
is ~0.5K tokens, so with gradient checkpointing and batch 8 activations stay around 1-3 GB. All four fit on
a T4 by this estimate. These are estimates: the first Colab run of each model records its peak GPU memory.

T4 has no bf16, so training there runs in fp16; L4 has bf16.

## Looked at, not shortlisted

- `Qwen/Qwen3.5-4B`: 4.66B with the vision encoder, thinking on by default; the 2B covers the newest Qwen.
- `google/gemma-4-E4B-it`: 4.5B effective, 8B with embeddings; over the 1-4B range in practice.
- `google/gemma-3-4b-it`: older generation, gated, Gemma license (not Apache).
- `meta-llama/Llama-3.2-3B-Instruct`: Turkish is not one of its 8 listed languages; gated.
- `AlicanKiraz0/Kara-Kumru-v1.0-2B`: a community fine-tune of Kumru-2B; its benchmark is self-reported.
  A possible second Turkish candidate if Kumru-2B's zero-shot output is unusable.
- `meta-llama/Llama-3.1-8B-Instruct`: added by the user on 2026-10-13 as a larger reference (they have gated
  access); 8.03B, Llama 3.1 license (not Apache), Turkish not among its 8 listed languages; bf16 ~16 GB, fits an L4.
- Other Turkish models found on the Hub are 7B+ (Turkish-Llama-8b, Turkish-Gemma-9b, Trendyol, Turkcell)
  or small GPT-2 models without instruction tuning.
