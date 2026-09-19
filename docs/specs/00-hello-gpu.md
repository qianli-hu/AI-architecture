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

## Hardware: 1× RTX A5000, 24 GB, $0.27/hr

GA102 — the same silicon family as the A40 in lab 03.

VRAM budget to predict before launching:

```
  24.0 GB  card
 − 0.5     CUDA context
 − 9.34    weights
 ────────
  14.2 GB  for KV + activations
```

KV costs `2 × 8 full-attn layers × 4 kv_heads × 256 head_dim × 2 B` =
**32 KB/token** → roughly 400K tokens of cache. Enormous headroom, which is
what makes lab 01's concurrency sweep possible on one cheap card.

## Pod configuration

| Setting | Value | Why |
|---|---|---|
| GPU | 1× RTX A5000 | $0.27/hr |
| Container disk | 30 GB | image unpacks ~15–20 GB |
| Volume disk | 40 GB | 9.34 GB weights + `.venv` + room |
| Ports | **TCP 22** | direct SSH, required for rsync |
| Env | `HF_TOKEN` | weight download |
| Image | `vllm/vllm-openai` (tag confirmed at creation) | |

## Components built

All reusable, all landing in `common/`:

| File | Job | Reused by |
|---|---|---|
| `common/pod.py` | create / list / terminate pods | all labs |
| `common/vram.py` | predicted VRAM budget from config | all labs |
| `Makefile` | `bootstrap sync fetch serve smoke save status` | all labs |
| `labs/00-hello-gpu/config.yaml` | model, gpu, disk sizes | — |

## Success criteria — the stopwatch table

The deliverable is `labs/00-hello-gpu/findings.md` containing measured
wall-clock for each step. This table sets the real cost of a cold start and
calibrates every later lab.

| # | Step | Target | Measured |
|---|---|---|---|
| 1 | pod create → SSH accepts | < 3 min | |
| 2 | `make sync` (rsync ~13 KB) | < 2 s | |
| 3 | `uv sync` | < 60 s | |
| 4 | `hf download` 9.34 GB | < 4 min | |
| 5 | `vllm serve` → `/health` 200 | < 90 s | |
| 6 | `curl /v1/chat/completions` returns tokens | works | |
| 7 | predicted vs measured KV blocks | gap explained | |
| 8 | `make fetch` + terminate | clean | |

Step 7 is the one that matters intellectually — everything else is plumbing.

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
