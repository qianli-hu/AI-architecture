"""The harness's pure parts. Nothing here touches RunPod, a GPU, or the network."""
from pathlib import Path

import pytest

from common import config, parse, plan, pod, serve, smoke, timing

LAB00 = Path(__file__).resolve().parent.parent / "labs/00-hello-gpu/config.yaml"


@pytest.fixture
def cfg():
    return config.load(LAB00)


# ---- plan -------------------------------------------------------------------
def test_lab00_config_resolves_to_a_known_model_and_card(cfg):
    model, budget = plan.from_config(cfg)
    assert model.name == cfg["model"]["repo"]
    assert budget.fits


def test_fp8_kv_cache_halves_bytes_per_token(cfg):
    _, bf16 = plan.from_config(cfg)
    cfg["serve"]["kv_cache_dtype"] = "fp8"
    _, fp8 = plan.from_config(cfg)
    assert fp8.kv_bytes_per_token * 2 == bf16.kv_bytes_per_token


# ---- pod --------------------------------------------------------------------
def test_create_body_pins_the_pod_to_the_volume(cfg):
    body = pod.create_body(cfg, "ssh-ed25519 AAAA test")
    # the volume is what pins the datacenter; the REST enum rejects US-MO-2
    assert "dataCenterIds" not in body
    assert body["networkVolumeId"] == "59jaue9ud8"
    assert body["gpuTypeIds"] == ["NVIDIA L4"] and body["gpuCount"] == 1
    assert body["env"]["LAB_PUBLIC_KEY"] == "ssh-ed25519 AAAA test"


def test_create_body_never_exposes_the_model_server(cfg):
    assert pod.create_body(cfg, "k")["ports"] == ["22/tcp"]


def test_create_body_replaces_the_vllm_entrypoint_with_sshd(cfg):
    body = pod.create_body(cfg, "k")
    assert body["dockerEntrypoint"] == ["bash", "-c"]
    assert "sshd" in body["dockerStartCmd"][0]
    assert body["dockerStartCmd"][0].rstrip().endswith("sleep infinity")


def test_hf_token_is_forwarded_only_when_set(cfg, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert "HF_TOKEN" not in pod.create_body(cfg, "k")["env"]
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert pod.create_body(cfg, "k")["env"]["HF_TOKEN"] == "hf_x"


def test_ssh_address_waits_for_both_ip_and_port():
    assert pod.ssh_address({"publicIp": "", "portMappings": {}}) is None
    assert pod.ssh_address({"publicIp": "1.2.3.4", "portMappings": None}) is None
    assert pod.ssh_address({"publicIp": "1.2.3.4", "portMappings": {"22": 10341}}) == ("1.2.3.4", 10341)


def test_ssh_remote_command_runs_inside_the_repo():
    argv = pod.ssh_argv("1.2.3.4", 10341, ["make", "serve"])
    assert argv[-2] == "root@1.2.3.4"
    assert argv[-1] == f"cd {pod.REPO} && make serve"


def test_status_totals_only_running_pods():
    table = pod.status_table([
        {"id": "a", "name": "workspace", "desiredStatus": "RUNNING", "costPerHr": 0.08},
        {"id": "b", "name": "lab", "desiredStatus": "RUNNING", "costPerHr": 0.49},
        {"id": "c", "name": "old", "desiredStatus": "EXITED", "costPerHr": 2.09},
        {"id": "d", "name": "gone", "desiredStatus": "TERMINATED", "costPerHr": 4.18},
    ])
    assert "running total: $0.57/hr" in table
    assert "gone" not in table


def test_status_says_so_when_nothing_is_billing():
    assert "nothing is billing" in pod.status_table([])


def test_down_without_a_pod_file_calls_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(pod, "POD_FILE", tmp_path / ".pod")
    monkeypatch.setattr(pod, "api", lambda *a, **k: pytest.fail("API was called"))
    assert pod.down(str(LAB00)) == 0


# ---- serve ------------------------------------------------------------------
def test_serve_command_from_config(cfg):
    assert serve.command(cfg) == [
        "vllm", "serve", "Qwen/Qwen3.5-4B", "--port", "8000",
        "--tensor-parallel-size", "1", "--max-model-len", "32768"]


def test_serve_command_optional_flags(cfg):
    cfg["serve"].update(kv_cache_dtype="fp8", gpu_memory_utilization=0.85,
                        extra_args=["--enforce-eager"])
    cmd = serve.command(cfg)
    assert cmd[-5:] == ["--gpu-memory-utilization", "0.85",
                        "--kv-cache-dtype", "fp8", "--enforce-eager"]


def test_compare_reports_the_gap():
    predicted = {"kv_per_gpu_gb": 12.16, "max_kv_tokens": 398458}
    out = serve.compare(predicted, parse.Measured(None, 350000, 10.66))
    assert "-1.50 GB" in out


def test_compare_survives_an_unparseable_log():
    predicted = {"kv_per_gpu_gb": 12.16, "max_kv_tokens": 398458}
    assert "by hand" in serve.compare(predicted, parse.Measured(None, None, None))


# ---- parse ------------------------------------------------------------------
def test_parse_v1_log():
    log = ("INFO gpu_worker.py: Available KV cache memory: 11.87 GiB\n"
           "INFO kv_cache_utils.py: GPU KV cache size: 388,912 tokens\n")
    m = parse.kv_cache(log)
    assert (m.kv_gib, m.kv_tokens, m.gpu_blocks) == (11.87, 388912, None)


def test_parse_v0_log_derives_tokens_from_blocks():
    m = parse.kv_cache("INFO # GPU blocks: 25600, # CPU blocks: 2048\n")
    assert m.gpu_blocks == 25600 and m.kv_tokens == 25600 * 16


def test_parse_takes_the_last_launch_in_a_log():
    log = "GPU KV cache size: 100 tokens\n...\nGPU KV cache size: 200 tokens\n"
    assert parse.kv_cache(log).kv_tokens == 200


def test_parse_finds_nothing_in_a_crash_log():
    assert not parse.kv_cache("CUDA out of memory").found


# ---- smoke ------------------------------------------------------------------
def test_smoke_request_streams_and_disables_thinking(cfg):
    body = smoke.request_body(cfg)
    assert body["stream"] and body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 64 and body["model"] == "Qwen/Qwen3.5-4B"


def _sse(delta=None, usage=None):
    import json
    chunk = {"choices": [{"delta": delta}] if delta is not None else [], "usage": usage}
    return f"data: {json.dumps(chunk)}".encode()


def test_consume_measures_ttft_and_decode_rate():
    # the clock is read at send, then once per content chunk, then at the end
    ticks = iter([0.0, 0.5, 1.0, 1.5, 1.6])
    lines = [_sse({"role": "assistant", "content": ""}),      # no content: not a token
             b"", b": keep-alive",
             _sse({"content": "It "}), _sse({"content": "splits "}), _sse({"content": "layers."}),
             _sse(usage={"prompt_tokens": 20, "completion_tokens": 3}),
             b"data: [DONE]"]
    m = smoke.consume(lines, clock=lambda: next(ticks))
    assert m["text"] == "It splits layers."
    assert m["ttft_s"] == 0.5
    assert m["decode_tokens_per_s"] == 2.0      # 2 tokens after the first, over 1.0 s
    assert m["total_s"] == 1.6
    assert (m["prompt_tokens"], m["completion_tokens"]) == (20, 3)


def test_consume_with_an_empty_reply_reports_no_ttft():
    m = smoke.consume([b"data: [DONE]"], clock=lambda: 0.0)
    assert m["text"] == "" and m["ttft_s"] is None and m["decode_tokens_per_s"] is None


def test_summarize_reports_spread_and_skips_missing_values():
    rows = [{"ttft_s": 0.10, "decode_tokens_per_s": 50.0, "total_s": 1.0},
            {"ttft_s": 0.20, "decode_tokens_per_s": None, "total_s": 1.0},
            {"ttft_s": 0.30, "decode_tokens_per_s": 54.0, "total_s": 1.0}]
    s = smoke.summarize(rows)
    assert s["ttft_s"]["mean"] == 0.2 and s["ttft_s"]["median"] == 0.2
    assert s["ttft_s"]["stdev"] == 0.1 and s["ttft_s"]["cv_pct"] == 50.0
    assert (s["ttft_s"]["min"], s["ttft_s"]["max"]) == (0.10, 0.30)
    assert s["decode_tokens_per_s"]["n"] == 2          # the None is not a zero
    assert s["total_s"]["stdev"] == 0 and s["total_s"]["cv_pct"] == 0


def test_summarize_single_request_has_no_spread():
    s = smoke.summarize([{"ttft_s": 0.1, "decode_tokens_per_s": 50.0, "total_s": 1.0}])
    assert s["ttft_s"]["stdev"] == 0.0


# ---- timing -----------------------------------------------------------------
def test_timing_round_trip_keeps_the_latest_per_step(tmp_path, capsys):
    timing.record(tmp_path, "uv sync", 41.0)
    timing.record(tmp_path, "vllm serve", 80.0)
    timing.record(tmp_path, "uv sync", 3.5)
    table = timing.render(timing.read(tmp_path))
    assert "| uv sync | 3.5 s |" in table and "41.0" not in table
    assert table.index("uv sync") < table.index("vllm serve")


def test_results_are_separated_by_run(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    monkeypatch.delenv("RUN", raising=False)
    assert config.results_dir(cfg_path) == tmp_path / "results" / "base"
    monkeypatch.setenv("RUN", "fp8-tp2")
    assert config.results_dir(cfg_path) == tmp_path / "results" / "fp8-tp2"
    assert config.results_dir(cfg_path, run="explicit").name == "explicit"


def test_timing_read_with_no_file(tmp_path):
    assert timing.read(tmp_path) == []
