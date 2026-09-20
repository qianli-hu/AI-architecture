"""Load a lab's config.yaml and locate its results directory."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def load(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def results_dir(path: str | Path, run: str | None = None) -> Path:
    """labs/NN-name/config.yaml -> labs/NN-name/results/<run>, created if
    missing. One directory per run, so a re-run or a second variant (lab 03's
    FP8/BF16 x TP1/TP2) never overwrites the evidence of the first. The run
    name comes from $RUN, which the Makefile exports: `make serve RUN=fp8-tp2`."""
    d = Path(path).resolve().parent / "results" / (run or os.environ.get("RUN") or "base")
    d.mkdir(parents=True, exist_ok=True)
    return d
