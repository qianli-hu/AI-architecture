# The pipeline: how a lab runs

**In one sentence:** two machines share one disk, one config file says what to
do, and seven small programs pass results to each other as files.

## The machines

```
  WORKSPACE POD  ($0.08/hr, no GPU)             GPU POD  ($0.49/hr, 1× L4)
  editing, git, driving the experiment          exists for minutes, only runs vLLM
            │                                            │
            └───────────────┬────────────────────────────┘
                            ▼
              NETWORK VOLUME  /workspace   (mounted on BOTH at once)
              ├── Inference-Infra/    the code, the config, the results
              └── hf-cache/           the model weights
```

**Nothing is ever copied between the pods.** Both see the same `/workspace`.
The GPU pod boots with the code and 9 GB of weights already there, and a
result file it writes is already on the workspace. The only thing that crosses
the network is SSH commands.

Why two machines: a GPU pod bills the whole card by wall-clock whether it is
busy or not. Everything that does not need a GPU — writing code, downloading
weights, predicting memory, reading results — happens on the machine that
costs six times less. Lab 00 held a GPU for 11 minutes and cost $0.09.

## One input: `config.yaml`

`labs/00-hello-gpu/config.yaml` holds every decision: which GPU, which
container image, which volume to attach, which model, the server flags, and
the question to ask. Every program reads it; nothing is hardcoded anywhere
else. **A lab is a config, not a program** — lab 01 is a different YAML file
run through the same code.

## What runs, in order, and what each step writes

Outputs land in `labs/<lab>/results/<run>/` (`base` unless a run is named).

| # | Command | Runs on | What it does | Writes |
|---|---|---|---|---|
| 1 | `make plan` | workspace | `plan.py` reads the config and calls the arithmetic in `vram.py`: card − weights − overheads = predicted KV cache | `plan.json` — the **prediction** |
| 2 | `make boot` | workspace | `pod.py prefetch` downloads the weights from Hugging Face onto the volume, at CPU prices | `/workspace/hf-cache/…` |
| 3 | `make gpu-up` | workspace | `pod.py up` asks RunPod's API for a GPU with this image and this volume, then polls until SSH answers. **Billing starts here** | `.pod` — the pod's id, IP, port |
| 4 | `make gpu-ssh c="…"` | workspace → GPU pod | `pod.py ssh` reads `.pod`, connects, runs the command inside the repo on the GPU pod | — |
| 5 | `make serve` | GPU pod | `serve.py` builds `vllm serve …` from the config, launches it, polls `/health`; `parse.py` pulls the KV cache size out of vLLM's log; reads `plan.json` and prints predicted vs measured | `vllm.log`, `vllm.pid`, `measured.json` — the **measurement** |
| 6 | `make smoke N=10` | GPU pod | `smoke.py` sends the question to `localhost:8000`, streaming, N times in sequence; times the first token and the rest | `smoke.json` — every request + statistics |
| 7 | `make stop` | GPU pod | `serve.py --stop` reads `vllm.pid`, stops the server | — |
| 8 | `make gpu-down` | workspace | `pod.py down` reads `.pod`, tells RunPod to terminate *that pod id*, deletes `.pod`. **Billing stops here** | — |

Every timed step also appends one line to `timings.jsonl` (via `timing.py`);
`make timings` prints it as the stopwatch table.

Afterwards a person reads those files and writes `findings.md`. That is the
only part with no code behind it, on purpose: explaining the gap between
prediction and measurement is the point of the project.

## How the programs are linked

They never call each other and share no memory. Three kinds of file connect
them:

1. **`config.yaml`** — what to do. Read by all, written by none.
2. **`results/<run>/*.json`** — data handed forward. `plan.py` writes
   `plan.json`; minutes later, on a *different machine*, `serve.py` reads it to
   print the comparison. The shared volume is what makes that work.
3. **`.pod`** — which GPU pod is ours. `pod.py up` writes it; `ssh` and `down`
   read it. `make gpu-down` can only ever terminate the id in that file, so it
   cannot take the workspace pod with it. It is written *before* anything can
   fail, because a pod you lose track of bills forever.

The consequence: **every intermediate value is a file you can open.** When a
number looks wrong, there is no debugger to attach — you read the JSON that
step wrote.

## The modules

| Module | Lines | Job |
|---|---|---|
| `common/vram.py` | ~135 | the memory arithmetic; pure functions, no I/O |
| `common/plan.py` | ~50 | config → prediction → `plan.json` |
| `common/pod.py` | ~215 | rent, reach and terminate the GPU pod (RunPod REST API + SSH) |
| `common/serve.py` | ~120 | launch vLLM, wait for health, compare predicted vs measured |
| `common/parse.py` | ~45 | regex the KV cache size out of a vLLM log |
| `common/smoke.py` | ~140 | streamed chat request; TTFT, decode tok/s, statistics over N |
| `common/timing.py` | ~60 | the stopwatch: append and render `timings.jsonl` |
| `common/config.py` | ~25 | load YAML, locate the results directory |

Everything that can be tested without a GPU is (`tests/`, 35 tests): the
request RunPod receives, the vLLM command line, the log parser, the streaming
timer. Before the first GPU was rented, the whole serve → smoke → stop path
was run against a 20-line fake `vllm` — which caught a real bug (HTTP
buffering made TTFT equal total latency) for $0.

## Reading order

1. `labs/00-hello-gpu/config.yaml` — the input
2. `Makefile` — which module each command runs → see [02-makefile.md](02-makefile.md)
3. `common/vram.py` — the prediction
4. `common/pod.py` — renting and reaching the GPU
5. `common/serve.py` — launching vLLM and measuring
6. `common/smoke.py` — the request and its timing
7. `labs/00-hello-gpu/results/base/` — the outputs
8. `labs/00-hello-gpu/findings.md` — what they mean

## If asked "why build it this way?"

- **Files, not function calls, between steps** — the steps run on different
  machines at different times; a file on a shared volume is the simplest thing
  that survives both. It also makes every run auditable after the pod is gone.
- **Config-driven** — scaling from a 4B model on an L4 to a 27B on two
  96 GB cards is a YAML edit, so what is learned cheaply is reused expensively.
- **Predict, then measure** — a prediction written down *before* the launch
  cannot be quietly adjusted to fit. Lab 00 predicted 12.16 GiB of KV cache and
  measured 9.10; every GiB of the gap is accounted for in `findings.md`.
- **Cost as a design constraint** — the architecture is shaped by the billing
  model: keep the expensive machine alive only for the steps that need it.
