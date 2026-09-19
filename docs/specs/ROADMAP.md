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
| **00 hello-gpu** | 1× A5000 24 GB | $0.27 | pod → ssh → rsync → download → serve → curl → terminate | 1 |
| **01 batching** | 1× A5000 | $0.27 | concurrency sweep: TTFT, throughput, preemption | 2 |
| **02 tensor-parallel** | 2× A5000 | $0.54 | TP=1 vs TP=2 on a model that fits on one card | 3 |
| **03 serving-27b** | 2× A40 96 GB | $0.98 | five-config matrix: TP, FP8, KV dtype, MTP | 2+3 |
| **04 distributed-training** | 2× A40 | $0.98 | QLoRA, FSDP, gradient all-reduce over PCIe | 4 |

Whole ladder ≈ **$28**.

## Two design decisions that make the ladder work

**Same architecture at every rung.** Labs 00–02 use `Qwen/Qwen3.5-4B`; lab 03
uses `Qwen/Qwen3.8-27B`. Both are `Qwen3_5ForConditionalGeneration` with
`full_attention_interval: 4`, `head_dim: 256`, `kv_heads: 4`, 262K context.
The 4B is the 27B at 1/6 scale. Nothing learned cheaply has to be relearned
expensively.

Likewise the GPUs: the **A5000 is GA102**, the same silicon family as the A40
and A6000. Ampere behaviour — Marlin FP8, memory characteristics, NCCL over
PCIe — carries from the $0.27/hr card to the $0.98/hr one unchanged.

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
  → ~220 GB. You have 96.
- **QLoRA fits comfortably.** 4-bit frozen base ≈ 15 GB, adapters are tiny,
  activations manageable with gradient checkpointing.

Lab 04 is therefore specced as *QLoRA + FSDP mechanics*, not "train a 27B".
It needs its own design pass before starting.
