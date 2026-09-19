# Inference-Infra

A five-lab ladder for learning GPU inference infrastructure, from renting a
single card to distributed training.

See **[docs/specs/ROADMAP.md](docs/specs/ROADMAP.md)** for the full program.

| Lab | GPUs | $/hr | Teaches |
|---|---|---|---|
| 00 hello-gpu | 1× L4 | $0.49 | the full loop: pod → serve → curl → terminate |
| 01 batching | 1× L4 | $0.49 | concurrency, TTFT, continuous batching |
| 02 tensor-parallel | 2× L4 | $0.98 | TP=1 vs TP=2, isolated |
| 03 serving-27b | 1–2× PRO 6000 | $2.09–4.18 | FP8/BF16 × TP1/TP2 on Qwen3.8-27B |
| 04 distributed-training | 2× PRO 6000 | $4.18 | QLoRA, FSDP, gradient sync |

## The four layers

| Layer | Lives in | Rebuilt by |
|---|---|---|
| This code | git (pod → GitHub) | `git clone` |
| Deps | `uv.lock` | `uv sync` |
| CUDA + torch + vLLM | container image | RunPod pulls it |
| Model weights | network volume | `hf download` |

## Topology — everything lives on the pod

    YOUR MAC                POD (US-MO-2)              NETWORK VOLUME
    ssh key only  ──ssh──▶  ephemeral compute ──mounts──▶ inference-infra 150 GB
                            AI CLIs, git, deps           /workspace/Inference-Infra
                                    │                    /workspace/hf-cache
                                    └──── git ────▶ GITHUB

The Mac holds **only the SSH private key**. All editing, git, and execution
happen on the pod. The volume persists across pod terminations; the pod does
not. Create a pod when you sit down, terminate when you stand up.

| Tier | Spec | $/hr |
|---|---|---|
| Workspace | `cpu3g` 2 vCPU / 8 GB | $0.08 |
| Small labs | 1× L4 24 GB | $0.49 |
| Big labs | 1–2× RTX PRO 6000 96 GB | $2.09 / $4.18 |

Volume: $10.50/mo, always. Compute: only while a pod runs.

## Layout

    common/          shared harness (runner, vram, parse, pod)
    labs/NN-name/    config.yaml  results/  findings.md
    docs/specs/      one spec per lab
