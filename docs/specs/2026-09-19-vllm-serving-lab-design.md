# vLLM Multi-GPU Serving Lab — Design

**Date:** 2026-09-19 (rev. 2 — hardware re-chosen against live catalog)
**Status:** Draft, awaiting review
**Scope:** Phases 0–4. Gateway (phase 5) and LoRA loop (phase 6) are out of scope
and will get their own specs.

## 1. Goal

Learn multi-GPU LLM inference by serving a 27B vision-language model
tensor-parallel across two GPUs, and **measure** what tensor parallelism and
quantization cost and buy.

The deliverable is not "a model is running." It is a reproducible benchmark
table with an explanation of every number in it.

### Why this, why now

The skills line `Inference and Serving: vLLM, model quantization fundamentals`
is the only hedged claim on a resume that otherwise carries nine years, five
peer-reviewed papers, a Kaggle gold, a production RecSys, and a LoRA-tuned VLM
that beats zero-shot GPT-4o. The gap is ownership of the serving layer, which
is the load-bearing skill for AI Infrastructure roles.

The chosen model is a Qwen VLM, the same family as the EMBC 2026
`Qwen2.5-VL-3B-Instruct` work at ~9× the parameters. That makes phase 6
(fine-tune → serve → eval) a continuation rather than a new project.

## 2. Decisions

### 2.1 Hardware: 2× NVIDIA A40, Secure Cloud

**Chosen against the live catalog on 2026-09-19, not from published price lists.**

| GPU | 2-GPU availability | $/hr (pair) | Datacenters (count=2) | Network volume? |
|---|---|---|---|---|
| RTX A6000 | **NONE** | — | none | — |
| **A40** | **HIGH** | **$0.98** | CA-MTL-1 (low), EU-SE-1 (med) | no |
| L40S | LOW | $2.18 | OC-AU-1, US-MO-1 | no |
| RTX 4090 | HIGH | $1.48 | EU-RO-1 (high) | yes |

**A6000 is out of stock** — `availability: NONE` at 1 and 2 GPUs, every CUDA
version. The earlier choice rested on a 10% memory-bandwidth edge (768 vs
696 GB/s) over the A40. That edge is unpurchasable.

**The 4090 is rejected because it breaks the experiment, not on price.**
Config 1 is the TP=1 baseline: the 30.9 GB FP8 checkpoint on one GPU. A 4090
has 24 GB — it cannot load. Config 2, the centerpiece, is meaningless without
config 1 on identical weights. Config 3 (BF16, 52 GB) needs 96 GB. The 4090
would buy native FP8 tensor cores and network-volume support at the cost of
the matrix being incoherent.

**A40 is the only in-stock configuration where all five configs are viable:**
FP8 (31 GB) fits one 48 GB card; BF16 (52 GB) spans two.

Region **EU-SE-1** (MEDIUM) over CA-MTL-1 (LOW). Latency from the US east
coast is ~110 ms, which is irrelevant: the load generator runs on-pod by
design, so no measurement crosses the Atlantic.

Cost: **$0.98/hr** for the pair.

A40 is Ampere: no FP8 tensor cores, vLLM dequantizes via Marlin. See §2.3.

### 2.2 Storage — volume disk, not network volume

| Tier | Mount | Survives stop | Survives terminate | Price |
|---|---|---|---|---|
| Container disk | `/` | no | no | $0.10/GB/mo running |
| **Volume disk** | `/workspace` | **yes** | **no** | $0.10 running / $0.20 stopped |
| Network volume | `/workspace` | yes | yes | $0.07/GB/mo — **unavailable here** |

**Network volumes are impossible for this hardware.** Cross-referencing the
volume-capable datacenters against where A40 and L40S have 2-GPU capacity
yields zero overlap. Only the 4090 in EU-RO-1 has both, and the 4090 is
rejected above.

So `/workspace` is a **120 GB volume disk**: FP8 (31) + BF16 (52) +
smoke-test model (18) + `.venv` and caches (5) ≈ 106 GB, with headroom.

**This reverses the "terminate is free" property.** The arithmetic:

- leave the pod stopped: 120 GB × $0.20/GB/mo = **$24/mo**
- terminate and re-download: 83 GB of weights ≈ 15–25 min ≈ **$0.40** of GPU time

Re-downloading wins decisively for weekend-burst usage. **Terminate between
sessions**; `make bootstrap` re-pulls weights as one command.

**Container disk: 50 GB, not the 20 GB default.** The image unpacks to
~15–20 GB; the rest is `/tmp`, torch compile cache, CUDA graph cache. The
default fills silently and fails hours in.

Revised total for the lab: **~$20** (20 GPU-hours at $0.98 plus disk).

### 2.3 Model: `Qwen/Qwen3.8-27B`

Verified against the HF API on 2026-09-19, not from secondary sources.

```
params            27.8 B          license   Apache 2.0
architecture      Qwen3_5ForConditionalGeneration
image_token_id    248056          language_model_only: false   → vision-language
layers            64              full_attention_interval: 4
                                  → 16 full attention, 48 linear (Gated DeltaNet)
attn heads        24              kv heads   4      head_dim 256
linear k heads    16              linear v heads 48  hidden 5120
max_position      262144          (~1M with YaRN)
extras            mtp.safetensors → MTP speculative decoding
```

Two checkpoints:

| | On disk | Notes |
|---|---|---|
| `Qwen3.8-27B-FP8` | **30.9 GB** | 24.7 B params F8_E4M3 + 3.1 B BF16; vision blocks excluded from quant |
| `Qwen3.8-27B` | ~52 GB | BF16 |

**TP=2 divisibility** (all must be even): attn heads 24→12, kv heads 4→2,
linear k 16→8, linear v 48→24. All clean. TP=8 would break on kv_heads.

**Hybrid attention matters for this lab.** Only 16 of 64 layers keep a real KV
cache, so long-context KV is far cheaper than a dense model of equal size. The
official vLLM recipe reports FP8/TP2 on 2×5090 at 14.28 GiB/GPU weights and
377,456 KV tokens at 262K context.

**On Ampere, FP8 is a memory optimization only.** The A6000 has no FP8 tensor
cores; vLLM dequantizes via Marlin kernels. Expect the memory win with no
compute win. This is a finding to report, not a problem to fix.

### 2.4 Software

- Region: EU-SE-1.
- Image: official `vllm/vllm-openai`, pinned. vLLM ≥ 0.17 required for this
  architecture; recipe verified on 0.26.x. Exact tag confirmed against the
  `runpod-templates` skill before first pod.
- CUDA + torch + vLLM ship together in the image because they are
  version-coupled. Analysis deps (pandas, pyyaml, requests, matplotlib) come
  from `uv` because they are not.
- **The runner shells out to `vllm`; it never imports it.** This keeps the
  runner's dependency set tiny and fully decoupled from the CUDA stack.

## 3. Topology

```
   MAC                        GITHUB                POD (2× A40)
   Claude Code + git          AI-architecture       /workspace (netvol)
   ├── make sync  ──rsync───────────────────────▶   Inference-Infra/  (no .git)
   ├── make fetch ◀─rsync───────────────────────    results/
   └── make save  ──push───▶  [one writer]          hf-cache/  .venv/
                                    ✗
                              POD never talks to GitHub
```

Three one-directional flows, one writer per destination, nothing can conflict.

The pod holds **no `.git` and no credentials**. It is a rented machine someone
else owns; the only secret it needs is `HF_TOKEN` for weight downloads. Code
is recoverable from the Mac and GitHub, so the pod is genuinely disposable.

Corollary: everything left of the pod is buildable and testable on the Mac for
$0, against a fake `vllm` stub. The first pod should test *inference*, not our
own orchestration bugs.

## 4. Repo layout

```
Inference-Infra/                  (remote: qianli-hu/AI-architecture)
  configs/matrix.yaml             the five configs
  src/
    vram.py                       predicted VRAM budget
    runner.py                     launch → health → bench → scrape → teardown
    parse.py                      vllm-bench stdout + Prometheus → dict
  tests/                          pytest, runs on the Mac against a vllm stub
  docs/specs/                     this file
  results/results.jsonl           one JSON object per (config, concurrency)
  notebooks/analysis.ipynb
  Makefile
  pyproject.toml  uv.lock
```

## 5. The VRAM model (phase 0)

`src/vram.py` predicts, per GPU:

```
  total            48.0 GB
  − CUDA context    0.5 GB   ← present before any weight loads; easy to forget
  − weights         (checkpoint bytes) / tp_size
  − activations + CUDA graph capture
  ─────────────────────────
  = KV cache budget
```

KV per token is computed over the **16 full-attention layers only**, not 64:

```
kv_bytes_per_token = 2 (K,V) × 16 layers × 4 kv_heads × 256 head_dim × dtype_bytes
```

**The method is the point.** The predicted budget is logged *before* vLLM
launches; vLLM then reports `# GPU blocks: N`. Every run records both, and the
writeup explains each gap. Predict → measure → explain is what turns flag-pasting
into understanding.

## 6. The experiment matrix

| # | Config | Isolates |
|---|---|---|
| 1 | FP8, TP=1 | baseline — 31 GB fits one A6000 |
| 2 | FP8, TP=2 | **pure TP cost/benefit** — identical weights, only TP changes |
| 3 | BF16, TP=2 | what the extra 48 GB bought; impossible at TP=1 |
| 4 | FP8 + `--kv-cache-dtype fp8`, TP=2 | KV quantization as an axis separate from weight quantization |
| 5 | FP8 + MTP spec decode, TP=2 | `{"method":"mtp","num_speculative_tokens":3}` |

Config 2 is the centerpiece. Running TP=2 on a model that *fits on one GPU* is
the only way to measure tensor-parallel overhead with no confound. Published
comparisons almost never separate "TP is slower" from "the model is bigger."

Expect config 2 to show **lower** single-stream tok/s than config 1 — two NCCL
all-reduces per layer × 64 layers = 128 syncs per token, over PCIe, no NVLink —
while supporting a much larger KV cache and higher concurrency. Naming that
trade correctly is the primary learning objective.

Swept at concurrency **1, 4, 16, 64**. 5 × 4 = 20 cells.

### Metrics per cell

From `vllm bench serve`: TTFT p50/p99, ITL p50/p99, TPOT, E2E, output tok/s.
From `/metrics`: `gpu_cache_usage_perc`, **`num_preemptions_total`**,
`num_requests_waiting`.
From `src/vram.py` and vLLM startup logs: predicted vs measured VRAM and KV blocks.

`num_preemptions_total` is the one most benchmarks omit. It is the direct signal
that KV ran out and vLLM began evicting — the mechanical explanation for the
latency cliff at high concurrency.

## 7. Build order

1. **`vllm` stub + runner + tests on the Mac.** $0. Prove orchestration works.
2. **Makefile targets, driven by hand** on the first live pod. One target per
   config. You watch every flag take effect and can poke the server between steps.
3. **Runner calls those same targets** once the shape is understood. Nothing
   is thrown away.

Do not script what you have never run by hand.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Hybrid linear-attention kernels are new; a dual-consumer-GPU field report hit scratch-tensor allocation failures and OOMs presenting as "compatibility" errors | Pin a known-good vLLM tag. **Smoke-test the whole harness on a ~9B model on one GPU first.** Never debug the runner and a novel attention kernel at once. |
| 2× A40 Secure unavailable in the volume's datacenter | **Check GPU availability by region *before* creating the network volume** — the volume pins you to one datacenter permanently |
| Container disk fills mid-run | 50 GB, not 20 |
| Benchmark client competes with the server for CPU | Run on-pod by default; do one Mac-side run to quantify and name the difference |
| vLLM holds ~90% VRAM until process exit | Runner waits for VRAM to drop between configs |

## 9. Out of scope

Phase 5 (Model Gateway, off-pod, + Grafana) and phase 6 (LoRA fine-tune →
hot-swap serve → eval harness) get their own specs.

They are cheap to add later because `vllm bench serve` already occupies the
exact socket the gateway will fill: an HTTP client against an OpenAI-compatible
endpoint. The gateway moves that client off-pod; `--enable-lora` adds adapters
behind the identical endpoint. Neither is a rewrite.

## 10. Success criteria

1. A committed `results.jsonl` with all 20 cells.
2. Predicted vs measured VRAM for all five configs, **with the gaps explained**.
3. A stated, defensible answer to: *what did tensor parallelism cost, and what
   did it buy?*
4. Evidence that FP8 on Ampere is memory-only, with the mechanism named.
5. Total spend under $50.
