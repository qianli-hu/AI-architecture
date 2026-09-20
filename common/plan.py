"""Predict the lab's VRAM budget from its config, print it, and save it to
results/plan.json so `serve` can set the measurement beside it.

    python -m common.plan labs/00-hello-gpu/config.yaml
"""
from __future__ import annotations

import json

from common import config
from common.vram import CARDS, MODELS, Budget, plan, render


def from_config(cfg: dict) -> tuple:
    model = MODELS[cfg["model"]["key"]]
    card_gb = CARDS[cfg["pod"]["card"]]
    tp = cfg["serve"].get("tensor_parallel_size", 1)
    kv_bytes = 1 if cfg["serve"].get("kv_cache_dtype") == "fp8" else 2
    return model, plan(model, card_gb, tp, kv_bytes)


def as_dict(b: Budget) -> dict:
    return {
        "card_gb": b.card_gb,
        "tp_size": b.tp_size,
        "weights_per_gpu_gb": b.weights_per_gpu_gb,
        "kv_per_gpu_gb": round(b.kv_per_gpu_gb, 3),
        "kv_total_gb": round(b.kv_total_gb, 3),
        "kv_bytes_per_token": b.kv_bytes_per_token,
        "max_kv_tokens": b.max_kv_tokens,
        "fits": b.fits,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Predict the lab's VRAM budget.")
    p.add_argument("config")
    a = p.parse_args(argv)
    cfg = config.load(a.config)
    model, b = from_config(cfg)
    print(f"\nPREDICTED  ({cfg['pod']['card']} x{b.tp_size})\n")
    print(render(model, b))
    print("\n  -> compare against the KV cache size vLLM logs at startup\n")
    out = config.results_dir(a.config) / "plan.json"
    out.write_text(json.dumps(as_dict(b), indent=2) + "\n")
    return 0 if b.fits else 1


if __name__ == "__main__":
    raise SystemExit(main())
