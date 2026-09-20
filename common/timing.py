"""The stopwatch. Every harness step appends one line to results/timings.jsonl,
so a lab's cold-start cost is a file rather than a memory.

    python -m common.timing labs/00-hello-gpu/config.yaml
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

from common import config

FILE = "timings.jsonl"


def record(results: Path, step: str, seconds: float, **extra) -> dict:
    row = {"step": step, "seconds": round(seconds, 2), "at": int(time.time()), **extra}
    with open(results / FILE, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  [stopwatch] {step}: {seconds:.1f}s")
    return row


@contextmanager
def stopwatch(results: Path, step: str, **extra):
    t0 = time.monotonic()
    yield
    record(results, step, time.monotonic() - t0, **extra)


def read(results: Path) -> list[dict]:
    p = results / FILE
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def render(rows: list[dict]) -> str:
    """Latest measurement per step, in first-seen order."""
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r["step"]] = r
    out = ["| Step | Measured |", "|---|---|"]
    out += [f"| {s} | {r['seconds']:.1f} s |" for s, r in latest.items()]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Print the lab's stopwatch table.")
    p.add_argument("config")
    a = p.parse_args(argv)
    rows = read(config.results_dir(a.config))
    print(render(rows) if rows else "no timings recorded yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
