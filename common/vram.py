"""Predict a vLLM VRAM budget before launching, so the launch can be checked
against a number rather than a hope.

The method matters more than the arithmetic: predict, measure, explain the gap.
vLLM reports `# GPU blocks: N` at startup; that is the measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

GB = 1024**3

# Present on every CUDA device before a single weight loads: kernels, driver
# structures, workspace buffers. Forgetting this is the classic off-by-500MB.
CUDA_CONTEXT_GB = 0.5


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights_gb: float          # checkpoint size on disk == VRAM footprint
    n_layers: int
    full_attention_interval: int   # 1 = dense; 4 = every 4th layer caches
    kv_heads: int
    head_dim: int

    @property
    def n_full_attn_layers(self) -> int:
        """Only full-attention layers hold a KV cache. Linear-attention layers
        carry a constant per-sequence recurrent state instead, not per-token."""
        return self.n_layers // self.full_attention_interval

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        # 2 = one K tensor and one V tensor
        return 2 * self.n_full_attn_layers * self.kv_heads * self.head_dim * dtype_bytes


@dataclass(frozen=True)
class Budget:
    card_gb: float
    tp_size: int
    cuda_context_gb: float
    weights_per_gpu_gb: float
    activation_reserve_gb: float
    kv_per_gpu_gb: float
    kv_bytes_per_token: int

    @property
    def kv_total_gb(self) -> float:
        return self.kv_per_gpu_gb * self.tp_size

    @property
    def max_kv_tokens(self) -> int:
        return int(self.kv_total_gb * GB / self.kv_bytes_per_token)

    @property
    def fits(self) -> bool:
        return self.kv_per_gpu_gb > 0


def plan(
    model: ModelSpec,
    card_gb: float,
    tp_size: int = 1,
    kv_dtype_bytes: int = 2,
    activation_reserve_gb: float = 2.0,
    cuda_context_gb: float = CUDA_CONTEXT_GB,
) -> Budget:
    """Per-GPU VRAM budget. Tensor parallelism shards weights across GPUs, so
    each card holds weights/tp_size -- but each card pays the CUDA context and
    activation reserve in full."""
    weights_per_gpu = model.weights_gb / tp_size
    kv_per_gpu = card_gb - cuda_context_gb - weights_per_gpu - activation_reserve_gb
    return Budget(
        card_gb=card_gb,
        tp_size=tp_size,
        cuda_context_gb=cuda_context_gb,
        weights_per_gpu_gb=weights_per_gpu,
        activation_reserve_gb=activation_reserve_gb,
        kv_per_gpu_gb=kv_per_gpu,
        kv_bytes_per_token=model.kv_bytes_per_token(kv_dtype_bytes),
    )


def render(model: ModelSpec, b: Budget) -> str:
    v = "fits" if b.fits else "DOES NOT FIT"
    return "\n".join([
        f"  model            {model.name}",
        f"  layers           {model.n_layers} ({model.n_full_attn_layers} full-attn, "
        f"{model.n_layers - model.n_full_attn_layers} linear)",
        f"  tensor parallel  {b.tp_size}",
        "",
        f"  card             {b.card_gb:>8.2f} GB  (per GPU)",
        f"  - cuda context   {b.cuda_context_gb:>8.2f} GB",
        f"  - weights        {b.weights_per_gpu_gb:>8.2f} GB",
        f"  - activations    {b.activation_reserve_gb:>8.2f} GB",
        f"  {'':17}{'-' * 11}",
        f"  = KV per GPU     {b.kv_per_gpu_gb:>8.2f} GB   [{v}]",
        f"    KV total       {b.kv_total_gb:>8.2f} GB",
        "",
        f"  KV per token     {b.kv_bytes_per_token / 1024:>8.1f} KB",
        f"  max KV tokens    {b.max_kv_tokens:>8,}",
    ])


# Verified against the HF API on 2026-09-19.
MODELS = {
    "qwen3.5-4b": ModelSpec("Qwen/Qwen3.5-4B", 9.34, 32, 4, 4, 256),
    "qwen3.5-9b": ModelSpec("Qwen/Qwen3.5-9B", 19.33, 32, 4, 4, 256),
    "qwen3.8-27b-fp8": ModelSpec("Qwen/Qwen3.8-27B-FP8", 30.9, 64, 4, 4, 256),
    "qwen3.8-27b": ModelSpec("Qwen/Qwen3.8-27B", 52.0, 64, 4, 4, 256),
}

# Nominal GB, as sold. What the driver actually exposes is less -- that
# difference is part of the gap findings.md has to explain.
CARDS = {"l4": 24.0, "pro6000": 96.0,
         "a5000": 24.0, "a40": 48.0, "a6000": 48.0, "4090": 24.0, "l40s": 48.0}


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Predict a vLLM VRAM budget.")
    p.add_argument("--model", default="qwen3.5-4b", choices=sorted(MODELS))
    p.add_argument("--card", default="l4", choices=sorted(CARDS))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--kv-dtype-bytes", type=int, default=2, choices=[1, 2])
    a = p.parse_args(argv)
    m = MODELS[a.model]
    b = plan(m, CARDS[a.card], a.tp, a.kv_dtype_bytes)
    print(f"\nPREDICTED  ({a.card} x{a.tp})\n")
    print(render(m, b))
    print("\n  -> compare against vLLM's '# GPU blocks: N' at startup\n")
    return 0 if b.fits else 1


if __name__ == "__main__":
    raise SystemExit(main())
