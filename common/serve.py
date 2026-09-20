"""Launch vLLM for the lab, wait for /health, then set the measured KV cache
beside the prediction. Run this ON the GPU pod.

    python -m common.serve labs/00-hello-gpu/config.yaml --log .../vllm.log
    python -m common.serve labs/00-hello-gpu/config.yaml --stop
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from common import config, parse, timing


def command(cfg: dict) -> list[str]:
    s = cfg["serve"]
    cmd = ["vllm", "serve", cfg["model"]["repo"],
           "--port", str(s.get("port", 8000)),
           "--tensor-parallel-size", str(s.get("tensor_parallel_size", 1)),
           "--max-model-len", str(s["max_model_len"])]
    if "gpu_memory_utilization" in s:
        cmd += ["--gpu-memory-utilization", str(s["gpu_memory_utilization"])]
    if "kv_cache_dtype" in s:
        cmd += ["--kv-cache-dtype", s["kv_cache_dtype"]]
    return cmd + [str(x) for x in s.get("extra_args", [])]


def compare(predicted: dict, m: parse.Measured) -> str:
    """The point of the project, as four lines."""
    lines = [f"  predicted  {predicted['kv_per_gpu_gb']:>7.2f} GB KV/GPU   "
             f"{predicted['max_kv_tokens']:>10,} tokens"]
    if not m.found:
        return "\n".join(lines + ["  measured   not found in the log -- read it by hand"])
    gib = f"{m.kv_gib:>7.2f} GB KV/GPU" if m.kv_gib is not None else " " * 17
    tok = f"{m.kv_tokens:>10,} tokens" if m.kv_tokens is not None else ""
    lines.append(f"  measured   {gib}   {tok}")
    if m.kv_gib is not None:
        gap = m.kv_gib - predicted["kv_per_gpu_gb"]
        lines.append(f"  gap        {gap:>+7.2f} GB  -> explain this in findings.md")
    return "\n".join(lines)


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def stop(results: Path) -> int:
    pidfile = results / "vllm.pid"
    if not pidfile.exists():
        print("no vllm.pid -- nothing to stop")
        return 0
    try:
        os.killpg(int(pidfile.read_text()), signal.SIGTERM)
        print("vLLM stopped")
    except ProcessLookupError:
        print("vLLM was already gone")
    pidfile.unlink()
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Launch vLLM and wait for /health.")
    p.add_argument("config")
    p.add_argument("--log")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args(argv)

    cfg = config.load(a.config)
    results = config.results_dir(a.config)
    if a.stop:
        return stop(results)
    port = cfg["serve"].get("port", 8000)
    if healthy(port):
        sys.exit(f"something already answers on :{port} -- `make stop` first")

    log = Path(a.log) if a.log else results / "vllm.log"
    cmd = command(cfg)
    print("  $ " + " ".join(cmd))
    env = {**os.environ, "HF_HOME": os.environ.get("HF_HOME", "/workspace/hf-cache")}
    t0 = time.monotonic()
    with open(log, "w") as f:
        # own session: survives this process and a dropped SSH connection
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    (results / "vllm.pid").write_text(str(proc.pid))

    while not healthy(port):
        if proc.poll() is not None:
            print(f"vLLM exited with {proc.returncode}. Last lines of {log}:\n")
            print("\n".join(log.read_text().splitlines()[-25:]))
            return 1
        if time.monotonic() - t0 > a.timeout:
            print(f"no /health after {a.timeout}s -- still running, see {log}")
            return 1
        time.sleep(1)
    timing.record(results, "vllm serve -> /health 200", time.monotonic() - t0)

    measured = parse.kv_cache(log.read_text())
    (results / "measured.json").write_text(json.dumps({
        "gpu_blocks": measured.gpu_blocks, "kv_tokens": measured.kv_tokens,
        "kv_gib": measured.kv_gib}, indent=2) + "\n")
    plan_file = results / "plan.json"
    if plan_file.exists():
        print("\nPREDICTED vs MEASURED\n")
        print(compare(json.loads(plan_file.read_text()), measured) + "\n")
    else:
        print("no plan.json -- run `make plan` first to get a prediction to compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
