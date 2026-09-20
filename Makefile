# Inference-Infra — run these ON the workspace pod, in /workspace/Inference-Infra.
#
#   workspace pod:  make plan boot gpu-up      then  make gpu-ssh
#   GPU pod:        make serve smoke stop
#   workspace pod:  make gpu-down timings save m="..."

LAB ?= 00-hello-gpu
# every output lands in labs/$(LAB)/results/$(RUN)/ -- name a run to keep it
RUN ?= base
N ?= 1
export RUN
CFG := labs/$(LAB)/config.yaml
OUT := labs/$(LAB)/results/$(RUN)
# The workspace runs from the repo's .venv through uv. The GPU pod runs the
# vLLM image, whose system python already carries everything common/ imports
# -- and which also ships uv, so "use uv if present" rebuilt the shared .venv
# against the wrong interpreter from there. Where vllm is installed, use the
# system python and leave .venv alone.
PY := $(shell command -v vllm >/dev/null 2>&1 && echo python3 || echo "uv run python")

.PHONY: ssh-config help plan test boot serve stop smoke bench logs timings save status gpu-up gpu-ssh gpu-down clean
.DEFAULT_GOAL := help

help:  ## show this help
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t14

# ---- free: no GPU needed ---------------------------------------------------
plan:  ## predicted VRAM budget for this lab
	@$(PY) -m common.plan $(CFG)

test:  ## run the test suite
	@uv run --extra dev pytest -q

status:  ## live pods and what they cost
	@$(PY) -m common.pod status

ssh-config:  ## ~/.ssh/config block for reaching THIS pod from a Mac
	@printf 'Host pod\n  HostName %s\n  Port %s\n  User root\n  IdentityFile ~/.ssh/id_ed25519\n  ServerAliveInterval 30\n  ServerAliveCountMax 6\n  TCPKeepAlive yes\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n' "$$RUNPOD_PUBLIC_IP" "$$RUNPOD_TCP_PORT_22"

save:  ## commit and push
	@git add -A && git commit -q -m "$(LAB): $(m)" && git push -q && echo pushed

# ---- GPU pod lifecycle, driven from the workspace pod ----------------------
gpu-up:  ## create the lab's GPU pod  [STARTS BILLING]
	@$(PY) -m common.pod up $(CFG)

# c travels as an env var, so its quotes && ; | reach the GPU pod untouched
# instead of being parsed -- and partly run -- by the shell here
gpu-ssh: export CMD := $(c)
gpu-ssh:  ## shell on the GPU pod, in this repo  (c="make serve RUN=x" runs one command)
	@$(PY) -m common.pod ssh $(CFG) "$$CMD"

gpu-down:  ## terminate it  [STOPS BILLING]
	@$(PY) -m common.pod down $(CFG)

boot:  ## download weights onto the volume -- from the workspace, at CPU prices
	@$(PY) -m common.pod prefetch $(CFG)

# ---- run ON the GPU pod ----------------------------------------------------
serve:  ## launch vLLM, wait for /health
	@$(PY) -m common.serve $(CFG)

stop:  ## stop vLLM
	@$(PY) -m common.serve $(CFG) --stop

smoke:  ## one streamed request: reply, TTFT, tok/s  (N=10 repeats it for a distribution)
	@$(PY) -m common.smoke $(CFG) --n $(N)

bench:  ## concurrency sweep -> results.jsonl
	@$(PY) -m common.runner $(CFG) --out $(OUT)/results.jsonl

logs:  ## tail the vLLM log
	@tail -f $(OUT)/vllm.log

timings:  ## the stopwatch table so far
	@$(PY) -m common.timing $(CFG)

clean:
	@rm -rf .pytest_cache **/__pycache__
