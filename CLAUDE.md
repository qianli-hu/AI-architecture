# CLAUDE.md

A five-lab ladder for learning GPU inference infrastructure with vLLM, run on
RunPod. **You are most likely running on the workspace pod itself**, not on a
laptop.

## Read these first

1. `docs/HANDOFF.md` — infrastructure as it exists, and the reasoning behind
   every hardware choice. Start here.
2. `docs/specs/ROADMAP.md` — the five labs and why they are ordered this way.
3. `docs/specs/<NN>-<lab>.md` — the spec for whichever lab is active.

## Where you are

```
/                          container disk · DIES when the pod terminates
├── root/          (= ~)   container disk · DIES
│   ├── .config     -> /workspace/.home/.config      (symlinked, survives)
│   ├── .claude     -> /workspace/.home/.claude      (symlinked, survives)
│   └── .bashrc, .gitconfig, apt packages, node, npm  ← all ephemeral
└── workspace/             NETWORK VOLUME · SURVIVES
    ├── Inference-Infra/   this repo, and .venv
    ├── hf-cache/          model weights (HF_HOME)
    └── .home/             persisted credentials
```

A pod is disposable; the volume is not. Anything that must outlive the pod
goes under `/workspace`, and anything installed by hand goes into
`scripts/bootstrap-workspace.sh` so the next pod gets it too. **That script is
the environment's only memory.**

## Gotchas that have already cost time

- **`nproc` and `free` report the host, not the pod.** A 2 vCPU pod shows 64
  cores and 755 GB. Never size anything off them.
- **`/workspace` is absolute.** `cd ~` walks off the volume onto the
  ephemeral disk; `cd workspace` from `~` fails.
- **After editing `bootstrap-workspace.sh`, re-run it from the repo**
  (`bash scripts/bootstrap-workspace.sh`), not via `curl | bash` —
  raw.githubusercontent.com caches for minutes and will serve the old copy.
- **A GPU pod bills the whole reserved card by wall-clock, not utilization.**
  0% busy costs the same as 100%. Terminate, do not idle.
- **Terminate, never stop.** Stop still wipes the container disk, so it buys
  almost nothing and keeps charging for held disk.

## Cost discipline

| Tier | Spec | $/hr |
|---|---|---|
| Workspace | `cpu3g` 2 vCPU | $0.08 |
| Small labs | 1–2× L4 | $0.49 / $0.98 |
| Big labs | 1–2× RTX PRO 6000 96 GB | $2.09 / $4.18 |

Volume: $10.50/mo regardless. Everything up to the first `vllm serve` is CPU
work — write and test the harness on the cheap tier.

## Conventions

- `uv run` for everything Python; deps in `pyproject.toml`, never bare `pip`.
- `common/` is shared harness, built up *by* the labs rather than before them.
  `common/vram.py` and `tests/test_vram.py` are the pattern to follow: small,
  pure, tested.
- Each lab owns `config.yaml`, `results/`, and `findings.md`.
- **Predict, then measure.** Log the predicted VRAM budget before launching
  vLLM, compare against its `# GPU blocks: N`, and explain the gap in
  `findings.md`. That comparison is the point of the whole project.
- `make help` lists every target.

## State as of 2026-09-19

Done: repo, specs, `common/vram.py` (8 tests green), Makefile, bootstrap
script, 150 GB volume in US-MO-2, workspace pod verified.

Next: `common/{plan,pod,serve,smoke,runner}.py` — roughly 250 lines, no GPU
needed — then lab 00 can run. See `docs/specs/00-hello-gpu.md` for the
stopwatch table it has to fill in.
