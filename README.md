# Inference-Infra

A hands-on lab for multi-GPU LLM inference with vLLM.

**Goal:** serve `Qwen/Qwen3.8-27B` (27.8B params, vision-language, 262K context)
tensor-parallel across 2× RTX A6000 on RunPod, and measure what tensor
parallelism and quantization actually cost and buy.

## The four layers

| Layer | Lives in | Rebuilt by |
|---|---|---|
| This code | git (Mac → GitHub) | `git clone` |
| Deps | `uv.lock` | `uv sync` |
| CUDA + torch + vLLM | container image | RunPod pulls it |
| Model weights (~52 GB) | pod network volume | `hf download` |

## Topology

    MAC  ──── make sync  ───▶  POD      code out (rsync)
    MAC  ◀─── make fetch ────  POD      results in (rsync)
    MAC  ──── make save  ───▶  GITHUB   git push

The pod holds no `.git` and no credentials. It is a disposable executor;
GitHub has exactly one writer, this Mac.

## Status

Design in progress — see `docs/specs/`.
