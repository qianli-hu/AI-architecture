"""Create, list and terminate the lab's GPU pod from the workspace pod.

    python -m common.pod up       labs/00-hello-gpu/config.yaml   # STARTS BILLING
    python -m common.pod ssh      labs/00-hello-gpu/config.yaml [cmd ...]
    python -m common.pod down     labs/00-hello-gpu/config.yaml   # stops billing
    python -m common.pod status
    python -m common.pod prefetch labs/00-hello-gpu/config.yaml

The GPU pod mounts the same network volume as the workspace, so the repo and
the weights are already there when it boots -- nothing is copied. `up` writes
the pod's id and SSH address to .pod; `down` only ever terminates that id, so
it cannot take the workspace pod with it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from common import config, timing

API = "https://rest.runpod.io/v1"
REPO = Path(__file__).resolve().parent.parent
POD_FILE = REPO / ".pod"
# On the volume: the workspace pod's private key for reaching GPU pods. The
# Mac's key never comes here; this one only ever opens pods we created.
SSH_KEY = Path("/workspace/.home/.ssh/id_ed25519")

# vllm/vllm-openai ships no sshd and its entrypoint is the server itself.
# Replace it: install sshd, authorize our key, then idle so `make serve` can
# be run -- and re-run with different flags -- over SSH on one billed pod.
START_SCRIPT = """\
apt-get update -qq && apt-get install -y -qq openssh-server make tmux >/dev/null
mkdir -p /run/sshd ~/.ssh && chmod 700 ~/.ssh
printf '%s\\n%s\\n' "$LAB_PUBLIC_KEY" "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
/usr/sbin/sshd
sleep infinity
"""


# ---- pure: tested without the network ---------------------------------------
def create_body(cfg: dict, public_key: str) -> dict:
    pod = cfg["pod"]
    env = {"LAB_PUBLIC_KEY": public_key, "HF_HOME": "/workspace/hf-cache"}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    return {
        "name": cfg["lab"],
        "computeType": "GPU",
        "cloudType": pod.get("cloud", "SECURE"),
        "gpuTypeIds": [pod["gpu"]],
        "gpuCount": pod.get("count", 1),
        # No dataCenterIds: the REST schema's enum predates US-MO-2 and rejects
        # it with a 400. The volume pins the datacenter anyway -- a pod can
        # only mount a volume in its own building.
        "networkVolumeId": pod["network_volume_id"],
        "volumeMountPath": "/workspace",
        "containerDiskInGb": pod["container_disk_gb"],
        "imageName": pod["image"],
        "ports": pod.get("ports", ["22/tcp"]),
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [START_SCRIPT],
        "env": env,
    }


def ssh_address(pod: dict) -> tuple[str, int] | None:
    """(ip, port) once RunPod has mapped port 22, else None."""
    ip, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
    return (ip, int(port)) if ip and port else None


def ssh_argv(ip: str, port: int, remote: list[str] | None = None) -> list[str]:
    argv = [
        "ssh", "-i", str(SSH_KEY), "-p", str(port),
        # pods are disposable and IPs recycle: a known_hosts entry is a
        # guaranteed false alarm on the next pod
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30",
        f"root@{ip}",
    ]
    if remote:
        argv.append(f"cd {REPO} && {' '.join(remote)}")
    return argv


def status_table(pods: list[dict]) -> str:
    live = [p for p in pods if p.get("desiredStatus") != "TERMINATED"]
    if not live:
        return "no pods -- nothing is billing"
    rows = [f"  {p['id']:<16}{p.get('name', ''):<22}{p.get('desiredStatus', ''):<10}"
            f"${p.get('costPerHr', 0):.2f}/hr" for p in live]
    total = sum(p.get("costPerHr", 0) for p in live if p.get("desiredStatus") == "RUNNING")
    return "\n".join(rows + [f"\n  running total: ${total:.2f}/hr"])


# ---- network ----------------------------------------------------------------
def api(method: str, path: str, body: dict | None = None):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set -- add it to /workspace/.home/.bash_env")
    r = requests.request(method, API + path, json=body, timeout=60,
                         headers={"Authorization": f"Bearer {key}"})
    if r.status_code in (401, 403):
        sys.exit(f"RunPod API refused the key ({r.status_code}). The key RunPod "
                 "injects into a pod is scoped to that pod; create a read/write "
                 "key in the console and export it from /workspace/.home/.bash_env")
    if not r.ok:
        sys.exit(f"RunPod API {method} {path} -> {r.status_code}: {r.text}")
    return r.json() if r.content else None


def ensure_key() -> str:
    if not SSH_KEY.exists():
        SSH_KEY.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-q",
                        "-C", "inference-infra-workspace", "-f", str(SSH_KEY)], check=True)
    return SSH_KEY.with_suffix(".pub").read_text().strip()


def saved() -> dict | None:
    return json.loads(POD_FILE.read_text()) if POD_FILE.exists() else None


def up(cfg_path: str, timeout_s: int = 1200) -> int:
    if saved():
        sys.exit(f"{POD_FILE} exists: a GPU pod is already up. `make gpu-down` first.")
    cfg = config.load(cfg_path)
    results = config.results_dir(cfg_path)
    t0 = time.monotonic()
    pod = api("POST", "/pods", create_body(cfg, ensure_key()))
    state = {"id": pod["id"], "lab": cfg["lab"], "created_at": time.time(),
             "cost_per_hr": pod.get("costPerHr")}
    # written before anything can fail: a pod we lose track of bills forever
    POD_FILE.write_text(json.dumps(state, indent=2) + "\n")
    print(f"created {pod['id']}  ${pod.get('costPerHr', '?')}/hr  -- BILLING HAS STARTED")

    print("waiting for SSH (image pull dominates a cold start) ...")
    while time.monotonic() - t0 < timeout_s:
        addr = ssh_address(api("GET", f"/pods/{pod['id']}"))
        if addr and subprocess.run(
                ssh_argv(*addr)[:-1] + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                                        f"root@{addr[0]}", "true"],
                capture_output=True).returncode == 0:
            state["ip"], state["port"] = addr
            POD_FILE.write_text(json.dumps(state, indent=2) + "\n")
            timing.record(results, "pod create -> SSH accepts", time.monotonic() - t0)
            print(f"ready: make gpu-ssh   ({addr[0]}:{addr[1]})")
            return 0
        time.sleep(5)
    print(f"SSH not up after {timeout_s}s. The pod is STILL BILLING -- "
          "`make gpu-down`, or inspect it in the console.", file=sys.stderr)
    return 1


def down(cfg_path: str) -> int:
    state = saved()
    if not state:
        print("no .pod file -- no GPU pod of ours is up (check `make status`)")
        return 0
    api("DELETE", f"/pods/{state['id']}")
    POD_FILE.unlink()
    hours = (time.time() - state["created_at"]) / 3600
    cost = hours * (state.get("cost_per_hr") or 0)
    timing.record(config.results_dir(cfg_path), "pod lifetime", hours * 3600,
                  cost_usd=round(cost, 3))
    print(f"terminated {state['id']} after {hours * 60:.0f} min, ~${cost:.2f}")
    return 0


def ssh(remote: list[str]) -> int:
    state = saved()
    if not state or "ip" not in state:
        sys.exit("no reachable GPU pod -- `make gpu-up` first")
    os.execvp("ssh", ssh_argv(state["ip"], state["port"], [r for r in remote if r.strip()]))


def prefetch(cfg_path: str) -> int:
    """Download weights into the volume's HF cache. Runs anywhere the volume is
    mounted -- so do it from the workspace pod, not at GPU prices."""
    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
    from huggingface_hub import snapshot_download
    cfg = config.load(cfg_path)
    repo = cfg["model"]["repo"]
    with timing.stopwatch(config.results_dir(cfg_path), f"hf download {repo}"):
        path = snapshot_download(repo)
    print(f"weights at {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="GPU pod lifecycle.")
    p.add_argument("action", choices=["up", "down", "status", "ssh", "prefetch"])
    p.add_argument("config", nargs="?")
    p.add_argument("remote", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if a.action == "status":
        print(status_table(api("GET", "/pods")))
        return 0
    if a.action == "ssh":
        return ssh(a.remote)
    if not a.config:
        p.error(f"{a.action} needs a config path")
    return {"up": up, "down": down, "prefetch": prefetch}[a.action](a.config)


if __name__ == "__main__":
    raise SystemExit(main())
