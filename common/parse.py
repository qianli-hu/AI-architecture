"""Pull the KV-cache measurement out of a vLLM startup log.

This is the "measure" half of predict-then-measure. vLLM has changed how it
reports the number across versions, so every known spelling is tried:

    V0:  # GPU blocks: 25600, # CPU blocks: 2048
    V1:  Available KV cache memory: 11.87 GiB
         GPU KV cache size: 388,912 tokens
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_GPU_BLOCKS = re.compile(r"# GPU blocks:\s*([\d,]+)")
_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_KV_GIB = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")


@dataclass(frozen=True)
class Measured:
    gpu_blocks: int | None
    kv_tokens: int | None
    kv_gib: float | None

    @property
    def found(self) -> bool:
        return any(v is not None for v in (self.gpu_blocks, self.kv_tokens, self.kv_gib))


def _last_int(rx: re.Pattern, text: str) -> int | None:
    m = rx.findall(text)
    return int(m[-1].replace(",", "")) if m else None


def kv_cache(log: str, block_size: int = 16) -> Measured:
    """Last occurrence wins -- a log may hold several launches."""
    blocks = _last_int(_GPU_BLOCKS, log)
    tokens = _last_int(_KV_TOKENS, log)
    gib = _KV_GIB.findall(log)
    if tokens is None and blocks is not None:
        tokens = blocks * block_size
    return Measured(blocks, tokens, float(gib[-1]) if gib else None)
