# Handoff — session of 2026-09-19

Everything decided in the design session, for picking up inside the remote
workspace. The repo is the transfer mechanism; this file holds the *why*.

## Infrastructure as it exists now

| Thing | Value |
|---|---|
| Network volume | `inference-infra`, **150 GB**, **US-MO-2**, STANDARD, id `59jaue9ud8` |
| Volume cost | $10.50/mo, billing from 2026-09-19 |
| SSH key | `ed25519`, registered, fp `SHA256:OhQsxYZAgMpIJlGD6xAzKuDgCmX5R6QNXUdLTMVO/EI` |
| Repo | `github.com/qianli-hu/AI-architecture` → `/workspace/Inference-Infra` |
| Pods | none yet |

## Three compute tiers, one volume

| Tier | Spec | $/hr | Use |
|---|---|---|---|
| **Workspace** | `cpu3g` CPU pod, 4 vCPU / 16 GB | **$0.16** | editing, git, AI CLIs |
| Small labs | 1× L4, 24 GB | $0.49 | labs 00–02, Qwen3.5-4B |
| Big labs | 1–2× RTX PRO 6000, 96 GB | $2.09 / $4.18 | labs 03–04, Qwen3.8-27B |

A GPU pod bills the whole reserved card by wall-clock, not utilization — so
never idle on one. Terminate and recreate on the CPU tier.

## Why US-MO-2

Network volumes exist only in certain datacenters, and a pod must run in the
same building as its volume. Cross-referencing volume-capable datacenters
against GPU stock, **US-MO-2 is the only one carrying both a cheap tier (L4,
$0.49) and a 96 GB tier (RTX PRO 6000)** — which is exactly what the
workspace pattern needs.

The pin costs you the 48 GB tier: US-MO-2 jumps 24 → 96 GB with nothing
between. A40 ($0.49/48 GB), L40S, RTX 6000 Ada and A100 are all unreachable
from this volume. Our ladder doesn't need 48 GB, so it doesn't bite.

Escape routes, cheapest first: make a second volume elsewhere ($10.50/mo);
copy across with the S3 API or `runpodctl send/receive`; or rebuild — code
from GitHub, weights from HF, `.venv` from `uv.lock`, ~30 min of waiting.

## Why 96 GB per card changes the experiment

The original plan (2× A40, 48 GB) could not run the 27B in BF16 at TP=1, so
the quantization axis and the parallelism axis were tangled. With 96 GB both
fit on one card, and lab 03 becomes a clean 2×2:

|  | TP=1 | TP=2 |
|---|---|---|
| **FP8** (30.9 GB) | ✓ | ✓ |
| **BF16** (52 GB) | ✓ | ✓ |

RTX PRO 6000 is Blackwell, so FP8 runs on **native tensor cores** — a real
compute win, unlike the A40's `fp8_marlin` path which is weight-only W8A16
(memory win, no compute win). `maxCount: 9` also allows TP=4 and TP=8.

## Models, verified against the HF API on 2026-09-19

| Model | Size | Role |
|---|---|---|
| `Qwen/Qwen3.5-4B` | 9.34 GB BF16 | labs 00–02 |
| `Qwen/Qwen3.8-27B-FP8` | **30.9 GB** | lab 03 |
| `Qwen/Qwen3.8-27B` | ~52 GB | lab 03 |

Both are `Qwen3_5ForConditionalGeneration` with `full_attention_interval: 4`,
`head_dim: 256`, `kv_heads: 4`, 262K context. **The 4B is the 27B at 1/6
scale** — nothing learned cheaply has to be relearned expensively.

Only 1 layer in 4 keeps a KV cache; the rest are linear attention (Gated
DeltaNet) with constant per-sequence state. That makes KV **4× cheaper** than
a dense model of equal size. `common/vram.py` encodes this, with tests.

The 27B is a **vision-language model** (`image_token_id`, `language_model_only:
false`) — same family as the EMBC 2026 `Qwen2.5-VL-3B` LoRA work, which is
what makes lab 04 a continuation rather than a new project.

## Decisions that changed under contact, and why

1. **A6000 → A40 → PRO 6000.** A6000 showed `availability: NONE` at every
   count. A40 had stock but no network volume anywhere it lives.
2. **4090 rejected twice.** Not on price — 30.9 GB will not load on a 24 GB
   card, which kills the TP=1 baseline and with it the TP comparison's meaning.
3. **Flat repo → five-lab ladder.** The original spec jumped straight to the
   hardest rung.
4. **Mac-as-workspace → pod-as-workspace.** User preference, matching their
   team. Cost: credentials now live on rented hardware, and the environment
   must be reproducible — hence `scripts/bootstrap-workspace.sh`.

## The discipline that keeps this cheap

**If you install something by hand, add it to
`scripts/bootstrap-workspace.sh` and push.** The volume is a cache, not a
vault. A hand-built environment nobody can reproduce is the one thing that
would make the US-MO-2 pin genuinely expensive.

## Open questions

- `--kv-cache-dtype fp8` on Blackwell — verify in the smoke test
- Can one network volume mount to two pods at once? Affects whether the CPU
  workspace can stay up while a GPU pod runs
- `common/{plan,pod,serve,smoke}.py` are referenced by the Makefile but not
  yet written (~250 lines)
- Account balance unverified — the MCP surface exposes spend, not balance

---

# Your setup, as of now

## Nothing lives on the Mac

Exactly one thing stays local, because it cannot move:

| On the Mac | Why |
|---|---|
| `~/.ssh/id_ed25519` (private key) | it is how you reach the pod; putting it on the pod is circular |

That is the whole list. `~/Documents/projects/Inference-Infra` is now a
**stale mirror** — everything in it is pushed. Leave it as a cold backup or
delete it, but **do not edit it.** Two writers to one GitHub repo is how you
lose an afternoon.

## Where everything actually lives

| | Holds | Lifetime |
|---|---|---|
| **Network volume** `inference-infra` | `/workspace/Inference-Infra` (the repo), `hf-cache` (weights), `.venv` | permanent, $10.50/mo |
| **Pod** | OS, CUDA, the CLIs `bootstrap-workspace.sh` installs | **disposable** — terminate freely |
| **GitHub** | source of truth for code | permanent |
| **Mac** | one SSH private key | — |

## Daily rhythm

```
sit down   →  create pod (cpu3g 2 vCPU, $0.08/hr) with volume attached
           →  ssh in; /workspace is already populated, ~1 min
           →  claude     # work
stand up   →  git push
           →  TERMINATE the pod        (not stop — nothing on it is worth keeping)
```

A three-hour session costs **24 cents**. Terminating loses nothing, because
the volume holds everything and re-mounting is instant. This is precisely
what the $10.50/mo bought.

## What changed by moving onto the pod

The pod now talks to GitHub, which the earlier Mac-centric design deliberately
avoided. That means **a credential with write access to your repos now lives
on rented hardware you do not own.** Two mitigations:

1. Use `gh auth login` over HTTPS — a scoped, revocable token — never an SSH
   deploy key.
2. Terminate pods when you are done. A terminated pod's container disk is
   gone, token included.

## The rule that keeps the US-MO-2 pin cheap

**Anything you install by hand goes into `scripts/bootstrap-workspace.sh`
and gets pushed.** The volume is a cache, not a vault. A hand-built
environment nobody can reproduce is the single thing that would turn a
half-hour migration into a lost weekend.
