"""One streamed request, one reply, three timings: TTFT, decode rate, total.
Proves the server actually generates -- and shows what a single uncontended
request costs, the baseline lab 01's concurrency sweep is read against.

    python -m common.smoke labs/00-hello-gpu/config.yaml          # once
    python -m common.smoke labs/00-hello-gpu/config.yaml --n 10   # a distribution

Repeats are sent one after another, never overlapping: this measures how
stable an idle server is, not how it behaves under load. That is lab 01.
"""
from __future__ import annotations

import json
import statistics
import time
from typing import Callable, Iterable

import requests

from common import config, timing


def request_body(cfg: dict) -> dict:
    return {
        "model": cfg["model"]["repo"],
        "messages": [{"role": "user", "content": cfg["smoke"]["prompt"]}],
        "max_tokens": cfg["smoke"].get("max_tokens", 64),
        "temperature": 0,
        # streaming is the only way to see the first token arrive
        "stream": True,
        "stream_options": {"include_usage": True},
        # Qwen3.5 thinks by default; 64 tokens of reasoning leaves an empty
        # answer, which looks like a broken server when it is not
        "chat_template_kwargs": {"enable_thinking": False},
    }


def consume(lines: Iterable[bytes | str], clock: Callable[[], float] = time.monotonic,
            t0: float | None = None) -> dict:
    """Read an OpenAI-style SSE stream. Pass `t0` from just before the request
    was sent, so TTFT includes connection, queueing and prefill -- what a user
    actually waits for -- not just the time since the headers came back."""
    t0 = clock() if t0 is None else t0
    first = last = None
    text, usage = [], {}
    for raw in lines:
        line = raw.decode() if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices", []):
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                last = clock()
                first = first or last
                text.append(piece)
    total = clock() - t0
    n = usage.get("completion_tokens")
    # the first token is prefill's; the rate of the rest is decode
    decode = (n - 1) / (last - first) if n and n > 1 and last > first else None
    return {
        "text": "".join(text),
        "ttft_s": round(first - t0, 4) if first else None,
        "decode_tokens_per_s": round(decode, 2) if decode else None,
        "total_s": round(total, 4),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": n,
    }


METRICS = ("ttft_s", "decode_tokens_per_s", "total_s")


def summarize(rows: list[dict]) -> dict:
    """Spread of each metric across repeats. cv = stdev/mean: the one number
    that says "how stable" without needing to know the unit."""
    out = {}
    for k in METRICS:
        xs = [r[k] for r in rows if r.get(k) is not None]
        if not xs:
            continue
        mean = statistics.fmean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        out[k] = {"n": len(xs), "mean": round(mean, 4), "median": round(statistics.median(xs), 4),
                  "stdev": round(sd, 4), "min": min(xs), "max": max(xs),
                  "cv_pct": round(100 * sd / mean, 2) if mean else None}
    return out


def render(rows: list[dict], summary: dict) -> str:
    out = ["    #    TTFT s   decode tok/s   total s   tokens"]
    out += [f"  {i:>3}  {r['ttft_s']:>8.4f}  {r['decode_tokens_per_s'] or 0:>13.2f}  "
            f"{r['total_s']:>8.4f}  {r['completion_tokens']:>7}" for i, r in enumerate(rows, 1)]
    out.append("")
    for k, s in summary.items():
        out.append(f"  {k:<20} mean {s['mean']:<9} median {s['median']:<9} stdev {s['stdev']:<8} "
                   f"min {s['min']:<8} max {s['max']:<8} cv {s['cv_pct']}%")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Send one streamed chat completion.")
    p.add_argument("config")
    p.add_argument("--base-url")
    p.add_argument("--n", type=int, default=1, help="repeat the request, sequentially")
    a = p.parse_args(argv)

    cfg = config.load(a.config)
    results = config.results_dir(a.config)
    base = a.base_url or f"http://localhost:{cfg['serve'].get('port', 8000)}"
    body = request_body(cfg)
    rows = []
    for _ in range(a.n):
        t0 = time.monotonic()
        with requests.post(f"{base}/v1/chat/completions", json=body, stream=True, timeout=300) as r:
            r.raise_for_status()
            # chunk_size=1: the default 512 holds lines back until the buffer
            # fills, which delivers a short reply in one burst and makes TTFT a lie
            rows.append(consume(r.iter_lines(chunk_size=1), t0=t0))
    m, summary = rows[0], summarize(rows)

    timing.record(results, "chat completion", m["total_s"], ttft_s=m["ttft_s"],
                  decode_tokens_per_s=m["decode_tokens_per_s"])
    (results / "smoke.json").write_text(json.dumps(
        {"prompt": cfg["smoke"]["prompt"], "n": a.n, "summary": summary, "requests": rows},
        indent=2) + "\n")

    print(f"\n  > {cfg['smoke']['prompt']}\n  < {m['text'].strip()}\n")
    print(f"  prompt tokens {m['prompt_tokens']}, temperature 0, "
          f"{len({r['text'] for r in rows})} distinct reply text(s) in {a.n} request(s)\n")
    print(render(rows, summary) + "\n")
    return 0 if all(r["text"].strip() for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
