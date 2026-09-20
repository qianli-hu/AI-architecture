# Lab 00 — hello-gpu: findings

Run on 2026-09-20. 1× NVIDIA L4 (US-MO-2), `vllm/vllm-openai:v0.29.0-cu129`,
`Qwen/Qwen3.5-4B`, BF16, TP=1, `max_model_len` 32768. Raw evidence is in
`results/base/`.

**The loop works, end to end, for $0.09.** The pod lived 11 minutes and 28
seconds, of which 9 minutes 41 seconds were two cold starts stacked on top of
each other.

## Stopwatch

| # | Step | Where | Target | Measured | |
|---|---|---|---|---|---|
| 1 | `make boot` — download 9.34 GB | workspace | < 4 min | **12.5 s** | ✓ |
| 2 | `make gpu-up` — create → SSH accepts | workspace | < 5 min | **285.9 s** | ✓ barely |
| 3 | `make serve` — launch → `/health` 200 | GPU pod | < 90 s | **295.4 s** | ✗ 3.3× over |
| 4 | `make smoke` — returns tokens | GPU pod | works | **works**, 10/10 | ✓ |
| 5 | predicted vs measured KV cache | GPU pod | gap explained | **−3.06 GiB**, explained below | ✓ |
| 6 | `make gpu-down` — gone from `make status` | workspace | clean | **clean** | ✓ |
| 7 | pod lifetime and cost | | < $1 | **688 s, $0.094** | ✓ |

The rule of thumb this buys: **a cold start costs about ten minutes and eight
cents before the first token**, and the model download is not part of it.

## Step 5 — predicted 12.16 GiB of KV cache, measured 9.10

The planner's arithmetic was `24 − 0.5 − 9.34 − 2.0 = 12.16`. vLLM's own
accounting, from `vllm.log`:

```
Free memory on device (21.84/22.03 GiB) on startup.
Desired GPU memory utilization is (0.92, 20.27 GiB).
Actual usage is 9.01 GiB for consumed memory (weights + non-torch),
2.17 GiB for peak activation, and 0.24 GiB for CUDAGraph memory.
Available KV cache memory: 9.1 GiB
```

Walking from one to the other:

| | GiB | Running | Why |
|---|---|---|---|
| planner's KV budget | | 12.16 | |
| a "24 GB" L4 is 22.03 GiB to CUDA | −1.97 | 10.19 | ECC and reserved memory come out of the nameplate. `nvidia-smi` says 23034 MiB; vLLM sees less again |
| vLLM only takes 92% of the card | −1.76 | 8.43 | `--gpu-memory-utilization` defaults to 0.92 in v0.29 (not the 0.90 of older docs). The planner assumed 100% |
| weights + context came in *under* | +0.83 | 9.26 | predicted 0.5 + 9.34 = 9.84, measured 9.01. "9.34 GB" is decimal; the checkpoint is **8.68 GiB**. The planner subtracted GB from GiB |
| activations a little over | −0.17 | 9.09 | reserved 2.0, peak was 2.17 — this includes profiling the vision encoder with a max-size image |
| **measured** | | **9.10** | |

Three of the four terms are the planner being naive about *the card*, not
about the model: nameplate vs usable, the utilization cap, and a unit error.
The two biggest (−1.97, −1.76) are the same on every model this card will
ever run, so they are cheap to fix once.

**Tokens: predicted 398,458, measured 267,842.** Two-thirds of that shortfall
is the GiB gap above. The rest is per-token cost: 9.10 GiB ÷ 267,842 =
**35.6 KiB/token against a predicted 32.0**, 11% more. The log says why:

```
Setting attention block size to 528 tokens to ensure that attention page
size is >= mamba page size.
Padding mamba page size by 0.76% ...
```

The 24 linear-attention layers do not keep a per-token cache, but their
recurrent state is not free: vLLM carves it out of the *same pool*, in pages
sized to match a 528-token attention block. `vram.py` prices those layers at
zero. *This reading is an inference from two log lines, not yet tested* —
lab 01 can test it: state cost should scale with the number of running
sequences, not with their length.

Even at 267K tokens the log reports `Maximum concurrency for 32,768 tokens
per request: 8.17x`, so lab 01's sweep has the room it was promised.

## Step 3 — why `serve` took 295 s, and what it was not

The obvious suspect was loading 9 GB of weights over a network filesystem.
It was innocent:

| Phase | Seconds | |
|---|---|---|
| process start → vLLM banner | ~48 | first import of a 10 GB image's Python stack, cold |
| config, tokenizer, image processor | ~46 | |
| **load weights from the network volume** | **16.7** | 8.68 GiB off FUSE = 0.52 GiB/s |
| torch.compile | 32.2 | cache is in `/root/.cache` — dies with the pod, so every pod recompiles |
| profiling + warmup run | 76.1 | includes the vision encoder |
| CUDA graph capture (two passes) | 37 | 51 piecewise + 35 full graphs |
| remainder of engine init | ~39 | |

Weights are 6% of startup. **The network volume is fast enough; the 27B can
be served straight off it** (~60 s for the FP8 checkpoint at this rate), and
the copy-to-local-disk fallback is not needed. What is slow is compilation
and warmup, which is the same work repeated on every fresh pod. Pointing
`VLLM_CACHE_ROOT` at the volume should remove most of the 32 s compile on the
second pod onward; `--enforce-eager` would remove compile *and* graph capture
at a cost in decode speed. Both are one-line experiments for lab 01.

## Step 4 — ten identical requests

Prompt: *"In one sentence, what does tensor parallelism do?"* (23 prompt
tokens, thinking disabled, temperature 0, sent one after another).

> Tensor parallelism distributes the computation of a single large tensor
> across multiple GPUs to enable the training and inference of models that
> exceed the memory capacity of a single device.

All ten replies were byte-identical, 33 tokens each.

| | mean | median | stdev | min | max | cv |
|---|---|---|---|---|---|---|
| TTFT (s) | 0.1081 | 0.1018 | 0.0225 | 0.0973 | 0.1718 | 20.8% |
| decode (tok/s) | 29.79 | 29.79 | 0.015 | 29.75 | 29.80 | **0.05%** |
| total (s) | 1.218 | 1.212 | 0.0225 | 1.207 | 1.282 | 1.8% |

- **Decode is as stable as a clock**: 29.75–29.80 tok/s, a spread of 0.05%.
  It is also almost exactly what the hardware allows. Generating one token
  means reading every weight once; the L4 moves 300 GB/s and the weights are
  9.25 GB, so the ceiling is ~32 tok/s. We got 92% of it. **An idle L4
  decoding this model is memory-bandwidth-bound**, which is the fact that
  makes batching worth a whole lab: more sequences per weight-read is nearly
  free throughput.
- **TTFT's 20.8% is one outlier, not noise.** Request 1 took 172 ms; requests
  2–10 sit in 97–105 ms (cv ≈ 2%). The tempting explanation — the prefix
  cache served the repeated prompt — is wrong: the log reports `Prefix cache
  hit rate: 0.0%` throughout, because only *full* blocks are cached and a
  23-token prompt never fills a 528-token block. Every request computed its
  own prefill. The extra 70 ms on request 1 is first-request warmup. So
  **~100 ms is an honest TTFT for a short prompt on an idle server.** It will
  stop being honest once prompts exceed 528 tokens and repeat — lab 01 must
  vary its prompts or pass `--no-enable-prefix-caching` by then.

## What the plumbing found

- **One network volume does mount on two pods at once.** The workspace stayed
  up and watched results appear. This was an open question in the handoff and
  the whole workspace-drives-GPU design rested on it.
- **RunPod's REST API rejects `US-MO-2`** in `dataCenterIds` — its enum
  predates the datacenter. Omitting the field works: the volume pins the
  location by itself. Cost: one refused request, $0.
- **`make gpu-ssh c="a && b"` ran `b` on the workspace.** The command was
  expanded unquoted into the recipe. It made the GPU pod look like it had no
  vLLM for two minutes of billed time. Now passed through an env var.
- **The vLLM image ships `uv`**, so the Makefile's "use uv if present" rebuilt
  the shared `.venv` from the GPU pod. Harmless, 2 s, fixed: where `vllm` is
  installed, use the system python.
- **`requests.iter_lines()` buffers 512 bytes**, which delivers a short reply
  in one burst and turns TTFT into total latency. Caught by the fake-server
  dry run before any GPU was rented — the reason that dry run exists.

## Carry into lab 01

1. `vram.py` v2: usable GiB per card (22.03 for L4) instead of nameplate,
   `gpu_memory_utilization` as an input, weights in GiB, and a term for
   linear-attention state. Re-predict, then see if the gap closes to < 0.3 GiB.
2. Put vLLM's compile cache on the volume; measure the second cold start.
3. Prefix caching did nothing here (prompt < one 528-token block) but will
   once prompts are long and repeated: vary them, or disable it.
4. The idle baseline to beat: **~100 ms TTFT, 29.8 tok/s per sequence.**
