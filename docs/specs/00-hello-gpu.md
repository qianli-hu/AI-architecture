# Lab 00 — hello-gpu

**Goal:** prove the entire loop end to end for about a dollar, and *time every
step*. This is not a benchmark. It is a smoke test with a stopwatch.

## Model: `Qwen/Qwen3.5-4B`

```
params        4.66 B BF16        repo size   9.34 GB
architecture  Qwen3_5ForConditionalGeneration
layers        32                 full_attention_interval: 4  → 8 full, 24 linear
attn heads    16    kv heads 4   head_dim 256   hidden 2560
max_position  262144
```

Chosen because it is **the lab-03 model at 1/6 scale** — same architecture
class, same hybrid attention, same head geometry. Also: 9.34 GB downloads in
~2 minutes, and TP=2 divides cleanly (16→8 heads, 4→2 kv heads) so lab 02 can
reuse it unchanged.

## Hardware: 1× L4, 24 GB, $0.49/hr, US-MO-2

Ada, so FP8 is native — the same property the Blackwell cards in lab 03 have.
It is also the only cheap card in the one datacenter that holds the volume
(see `docs/HANDOFF.md`); the RTX A5000 this spec first named is not reachable
from there.

VRAM budget to predict before launching (`make plan`):

```
  24.0 GB  card
 − 0.5     CUDA context
 − 9.34    weights
 − 2.0     activation reserve
 ────────
  12.2 GB  for KV
```

KV costs `2 × 8 full-attn layers × 4 kv_heads × 256 head_dim × 2 B` =
**32 KB/token** → roughly 400K tokens of cache. Enormous headroom, which is
what makes lab 01's concurrency sweep possible on one cheap card.

The prediction is deliberately naive: it takes the card at its nominal 24 GB
and knows nothing of `--gpu-memory-utilization` (default 0.9) or the linear
layers' recurrent state. Expect it to overshoot. Saying by how much, and why,
is step 5.

## Pod configuration

| Setting | Value | Why |
|---|---|---|
| GPU | 1× NVIDIA L4 | $0.49/hr |
| Datacenter | US-MO-2 | where the volume lives |
| Network volume | `59jaue9ud8` at `/workspace` | repo and weights arrive mounted — nothing is copied |
| Container disk | 30 GB | image unpacks ~15–20 GB |
| Ports | **TCP 22** only | vLLM is reached on localhost over SSH, never exposed |
| Image | `vllm/vllm-openai:v0.29.0-cu129` | pinned 2026-09-19 |
| Entrypoint | replaced: install sshd, authorize key, `sleep infinity` | the image's own entrypoint *is* the server; idling instead lets `make serve` be re-run with new flags on one billed pod |

The workspace pod reaches the GPU pod with its own key,
`/workspace/.home/.ssh/id_ed25519`, generated on first `make gpu-up` and
passed to the pod as `LAB_PUBLIC_KEY`. The Mac's private key stays on the Mac.

## The loop

```
workspace pod ($0.08/hr)              GPU pod ($0.49/hr)
─────────────────────────             ──────────────────
make plan          predict
make boot          weights → volume
make gpu-up        ── BILLING ──▶     boots, mounts the same volume
make gpu-ssh       ─────────────▶     make serve     launch, /health, predicted vs measured
                                      make smoke     one request
                                      make stop
make gpu-down      ── stops ────▶     ✗
make timings save
```

Everything above `make gpu-up` is free to get wrong. Results land in
`labs/00-hello-gpu/results/<run>/` (`RUN=base` unless named) on the shared volume, so there is nothing to
fetch back.

## Components built

All reusable, all landing in `common/`:

| File | Job | Reused by |
|---|---|---|
| `common/vram.py` | VRAM arithmetic | all labs |
| `common/plan.py` | config → prediction → `results/plan.json` | all labs |
| `common/pod.py` | `up` / `ssh` / `down` / `status` / `prefetch` via the RunPod REST API | all labs |
| `common/serve.py` | launch vLLM, wait for `/health`, print predicted vs measured | all labs |
| `common/parse.py` | KV cache size out of a vLLM log | all labs |
| `common/smoke.py` | one streamed chat completion: TTFT, decode tok/s, total | all labs |
| `common/timing.py` | the stopwatch: `results/timings.jsonl` | all labs |
| `labs/00-hello-gpu/config.yaml` | model, gpu, image, volume | — |

`common/runner.py` (the concurrency sweep behind `make bench`) is lab 01's
job and does not exist yet.

## Success criteria — the stopwatch table

The deliverable is `labs/00-hello-gpu/findings.md` containing measured
wall-clock for each step. This table sets the real cost of a cold start and
calibrates every later lab. Steps 1, 2, 3 and 4 record themselves
(`make timings`).

| # | Step | Where | Target | Measured |
|---|---|---|---|---|
| 1 | `make boot` — `hf download` 9.34 GB | workspace | < 4 min | |
| 2 | `make gpu-up` — pod create → SSH accepts | workspace | < 5 min (10.7 GB image pull) | |
| 3 | `make serve` — `vllm serve` → `/health` 200 | GPU pod | < 90 s | |
| 4 | `make smoke` — returns tokens; TTFT and decode tok/s recorded | GPU pod | works | |
| 5 | predicted vs measured KV cache | GPU pod | gap explained | |
| 6 | `make gpu-down` — pod gone from `make status` | workspace | clean | |
| 7 | total pod lifetime and cost | | < $1 | |

Step 5 is the one that matters intellectually — everything else is plumbing.

Lab 00 also settles an open question from the handoff: whether one network
volume mounts on two pods at once. The loop above assumes it does. If
`make gpu-up` is refused for that reason, the fallback is to terminate the
workspace pod and run every target on the GPU pod directly.

## Scaling path

- **Lab 01** reuses this pod spec and model verbatim; adds a concurrency sweep
  to `common/runner.py`.
- **Lab 02** changes one line — `count: 2` — and adds `--tensor-parallel-size 2`.
  Same model, so TP overhead is measured with no confound.
- **Lab 03** swaps the model and GPU; `pod.py`, `vram.py` and the Makefile are
  unchanged.

## Out of scope

No benchmarking, no concurrency, no tensor parallelism. One request, one
response, eight timings. If lab 00 takes more than two hours, something is
wrong with the plumbing and that is exactly what it exists to find.
