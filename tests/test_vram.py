import pytest
from common.vram import MODELS, CARDS, plan


def test_hybrid_attention_layer_count():
    """32 layers, interval 4 -> 8 cache, 24 do not."""
    m = MODELS["qwen3.5-4b"]
    assert m.n_full_attn_layers == 8
    assert m.n_layers - m.n_full_attn_layers == 24


def test_kv_per_token_counts_only_full_attention_layers():
    m = MODELS["qwen3.5-4b"]
    # 2 * 8 layers * 4 kv_heads * 256 head_dim * 2 bytes
    assert m.kv_bytes_per_token(2) == 32 * 1024


def test_hybrid_is_four_times_cheaper_than_dense():
    from common.vram import ModelSpec
    hybrid = MODELS["qwen3.5-4b"]
    dense = ModelSpec("dense", 9.34, 32, 1, 4, 256)
    assert dense.kv_bytes_per_token() == 4 * hybrid.kv_bytes_per_token()


def test_lab00_fits_on_one_a5000():
    b = plan(MODELS["qwen3.5-4b"], CARDS["a5000"], tp_size=1)
    assert b.fits
    assert b.kv_per_gpu_gb == pytest.approx(24.0 - 0.5 - 9.34 - 2.0, abs=0.01)
    assert b.max_kv_tokens > 300_000


def test_tp_shards_weights_but_not_overhead():
    m = MODELS["qwen3.8-27b"]
    one = plan(m, CARDS["a40"], tp_size=1)
    two = plan(m, CARDS["a40"], tp_size=2)
    assert two.weights_per_gpu_gb == pytest.approx(one.weights_per_gpu_gb / 2)
    # overhead is paid per-card, not shared
    assert two.cuda_context_gb == one.cuda_context_gb
    assert two.activation_reserve_gb == one.activation_reserve_gb


def test_27b_bf16_needs_two_a40s():
    m = MODELS["qwen3.8-27b"]
    assert not plan(m, CARDS["a40"], tp_size=1).fits   # 52 GB > 48 GB card
    assert plan(m, CARDS["a40"], tp_size=2).fits


def test_27b_fp8_cannot_baseline_on_a_4090():
    """The reason lab 03 uses A40 and not 4090: 30.9 GB will not load on 24 GB,
    which kills the TP=1 baseline and with it config 2's meaning."""
    m = MODELS["qwen3.8-27b-fp8"]
    assert not plan(m, CARDS["4090"], tp_size=1).fits
    assert plan(m, CARDS["a40"], tp_size=1).fits


def test_fp8_kv_cache_halves_per_token_cost():
    m = MODELS["qwen3.8-27b-fp8"]
    assert m.kv_bytes_per_token(1) * 2 == m.kv_bytes_per_token(2)
