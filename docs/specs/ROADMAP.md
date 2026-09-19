# Program Roadmap — Inference-Infra

Five labs, increasing difficulty, each building the harness the next one uses.

## Purposes

1. Run rented GPUs online competently
2. Understand what vLLM buys under concurrency (TTFT, throughput, batching)
3. Distributed **inference** — tensor parallelism across GPUs
4. Distributed **training** — QLoRA and gradient sync

## The ladder

| Lab | GPUs | $/hr | Teaches | Purpose |
|---|---|---|---|---|
| **00 hello-gpu** | 1× L4 24 GB | $0.49 | the full loop end to end, timed | 1 |
| **01 batching** | 1× L4 | $0.49 | concurrency sweep: TTFT, throughput, preemption | 2 |
| **02 tensor-parallel** | 2× L4 | $0.98 | TP=1 vs TP=2 on a model that fits on one card | 3 |
| **03 serving-27b** | 1–2× PRO 6000 96 GB | $2.09–4.18 | FP8/BF16 × TP=1/TP=2, a clean 2×2 | 2+3 |
| **04 distributed-training** | 2× PRO 6000 | $4.18 | QLoRA, FSDP, gradient all-reduce over PCIe | 4 |

All in **US-MO-2**, all mounting the same 150 GB network volume. A `cpu3g`
workspace pod ($0.08/hr) mounts it too — that is where you edit.

## Two design decisions that make the ladder work

**Same architecture at every rung.** Labs 00–02 use `Qwen/Qwen3.5-4B`; lab 03
uses `Qwen/Qwen3.8-27B`. Both are `Qwen3_5ForConditionalGeneration` with
`full_attention_interval: 4`, `head_dim: 256`, `kv_heads: 4`, 262K context.
The 4B is the 27B at 1/6 scale. Nothing learned cheaply has to be relearned
expensively.

The GPUs differ by tier, deliberately: the L4 is Ada (native FP8), the RTX
PRO 6000 is Blackwell (native FP8 + NVFP4). Both do FP8 in hardware, unlike
the Ampere cards originally planned, where vLLM falls back to `fp8_marlin`
weight-only W8A16 — memory win, no compute win.

**`common/` is built *by* the labs, not before them.** Lab 00 writes
`runner.py` at its simplest: launch a server, curl it. Lab 01 adds the
concurrency sweep. Lab 02 adds multi-GPU and NCCL. Lab 03 adds the config
matrix. Each rung's difficulty step is also a harness feature. Nothing is
throwaway.

## Layout

```
common/          shared harness  (runner, vram, parse, pod)
labs/NN-name/    config.yaml  results/  findings.md
docs/specs/      one spec per lab
```

Each `findings.md` is a self-contained writeup. Five stacked is the portfolio
artifact.

## Note on lab 04

Distributed training is a different beast and 2× A40 has hard limits:

- **No NVLink.** Gradient all-reduce over PCIe costs far more in training than
  tensor-parallel activations cost in inference — you sync every step.
- **Full fine-tuning a 27B is out.** weights + grads + Adam states ≈ 8 B/param
  → ~220 GB. Two PRO 6000s give 192 GB — closer than the A40 plan, still short.
- **QLoRA fits easily.** 4-bit frozen base ≈ 15 GB against 96 GB per card.

Lab 04 is therefore specced as *QLoRA + FSDP mechanics*, not "train a 27B".
It needs its own design pass before starting.
