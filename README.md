# Inference-Infra

A five-lab ladder for learning GPU inference infrastructure, from renting a
single card to distributed training.

See **[docs/specs/ROADMAP.md](docs/specs/ROADMAP.md)** for the full program.

| Lab | GPUs | $/hr | Teaches |
|---|---|---|---|
| 00 hello-gpu | 1× A5000 | $0.27 | the full loop: pod → serve → curl → terminate |
| 01 batching | 1× A5000 | $0.27 | concurrency, TTFT, continuous batching |
| 02 tensor-parallel | 2× A5000 | $0.54 | TP=1 vs TP=2, isolated |
| 03 serving-27b | 2× A40 | $0.98 | five-config matrix on Qwen3.8-27B |
| 04 distributed-training | 2× A40 | $0.98 | QLoRA, FSDP, gradient sync |

Whole ladder ≈ $28.

## The four layers

| Layer | Lives in | Rebuilt by |
|---|---|---|
| This code | git (Mac → GitHub) | `git clone` |
| Deps | `uv.lock` | `uv sync` |
| CUDA + torch + vLLM | container image | RunPod pulls it |
| Model weights | pod volume disk | `hf download` |

## Topology

    MAC  ──── make sync  ───▶  POD      code out (rsync)
    MAC  ◀─── make fetch ────  POD      results in (rsync)
    MAC  ──── make save  ───▶  GITHUB   git push

The pod holds no `.git` and no credentials. It is a disposable executor;
GitHub has exactly one writer, this Mac.

## Layout

    common/          shared harness (runner, vram, parse, pod)
    labs/NN-name/    config.yaml  results/  findings.md
    docs/specs/      one spec per lab
