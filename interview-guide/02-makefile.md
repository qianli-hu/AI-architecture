# The Makefile: where the `make` commands live

**In one sentence:** `make` is a menu of shortcuts; all of them are defined in
one file, `Makefile`, at the root of the repo, and each one just runs a Python
module with the lab's config as its argument.

## Where

`/workspace/Inference-Infra/Makefile` — one file, tracked in git, about 3 KB.
There is no other Makefile anywhere on the volume.

When you type `make <name>`, the `make` program looks for a file called
`Makefile` **in the current directory** and runs the block with that name. So
the commands only work from the repo root; anywhere else you get
`No rule to make target`.

The GPU pod uses the *same file* — not a copy. It mounts the same volume, so
`make serve` over there reads the very same `Makefile`.

## What one entry looks like

```make
plan:  ## predicted VRAM budget for this lab
	@$(PY) -m common.plan $(CFG)
```

- `plan:` — the name you type after `make`
- `## …` — a description. `make help` greps these out of the file, so the
  documentation cannot drift from the code
- the indented line — the actual command. The `@` stops make from echoing it

So `make plan` is shorthand for:

```
uv run python -m common.plan labs/00-hello-gpu/config.yaml
```

Nothing clever lives in the Makefile. The logic is in `common/`; this file only
saves typing and keeps every lab's commands identical.

## The variables at the top

| Variable | Default | Meaning | Example |
|---|---|---|---|
| `LAB` | `00-hello-gpu` | which lab folder | `LAB=01-batching make plan` |
| `CFG` | `labs/$(LAB)/config.yaml` | derived from `LAB` | — |
| `RUN` | `base` | results subfolder, so a re-run or a variant never overwrites the first | `make serve RUN=fp8-tp2` |
| `N` | `1` | how many times `smoke` repeats its request | `make smoke N=10` |
| `PY` | auto | how to run Python: `uv run python` on the workspace, plain `python3` where vLLM is installed (the GPU pod) | — |
| `c` | empty | a command for `gpu-ssh` to run on the GPU pod | `make gpu-ssh c="make serve"` |

`LAB` is what lets one Makefile drive every lab: same targets, different
config file.

## The targets, grouped by where they run

```
workspace pod:   make plan boot gpu-up          then   make gpu-ssh
GPU pod:         make serve smoke stop
workspace pod:   make gpu-down timings save m="..."
```

| Target | Costs money? | Does |
|---|---|---|
| `help` | | list every target |
| `plan` | | predict the VRAM budget → `plan.json` |
| `test` | | run the test suite |
| `status` | | list live pods and their $/hr |
| `boot` | | download the lab's weights onto the volume |
| `gpu-up` | **starts billing** | create the lab's GPU pod, wait for SSH |
| `gpu-ssh` | | shell on the GPU pod (or run `c="…"` there) |
| `serve` | | launch vLLM, wait for `/health`, print predicted vs measured |
| `smoke` | | one streamed request (`N=10` for a distribution) |
| `stop` | | stop vLLM |
| `logs` | | follow the vLLM log |
| `gpu-down` | **stops billing** | terminate the GPU pod |
| `timings` | | print the stopwatch table |
| `ssh-config` | | print the `~/.ssh/config` block for reaching this pod from a Mac |
| `save` | | `git add -A`, commit with `m="…"`, push |

## Two bugs this file had, both found by lab 00

Worth knowing because they are the kind of thing that only shows up on real
infrastructure:

- **`make gpu-ssh c="a && b"` ran `b` on the wrong machine.** `$(c)` was pasted
  unquoted into the command, so the local shell split it at `&&`: `a` went to
  the GPU pod, `b` ran on the workspace. For two billed minutes the GPU pod
  appeared to have no vLLM installed. Fix: pass the command through an
  environment variable so its `&& ; |` and quotes arrive untouched.
- **"Use `uv` if it is installed" was the wrong test.** The vLLM image also
  ships `uv`, so the GPU pod rebuilt the shared `.venv` against its own Python.
  Fix: test for what actually distinguishes the machines — if `vllm` is on the
  PATH, use the system Python and leave `.venv` alone.

Both are one-line fixes with a comment above them in the Makefile explaining
why the line looks the way it does.
