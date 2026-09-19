# Inference-Infra — run these ON the workspace pod, in /workspace/Inference-Infra.
#
#   LAB=00-hello-gpu make plan boot serve smoke save

LAB ?= 00-hello-gpu
CFG := labs/$(LAB)/config.yaml

.PHONY: help plan test boot serve smoke bench logs save status gpu-up gpu-down clean
.DEFAULT_GOAL := help

help:  ## show this help
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t14

# ---- free: no GPU needed ---------------------------------------------------
plan:  ## predicted VRAM budget for this lab
	@uv run python -m common.plan $(CFG)

test:  ## run the test suite
	@uv run --extra dev pytest -q

status:  ## live pods and what they cost
	@uv run python -m common.pod status

save:  ## commit and push
	@git add -A && git commit -q -m "$(LAB): $(m)" && git push -q && echo pushed

# ---- GPU pod lifecycle, driven from the workspace pod ----------------------
gpu-up:  ## create the lab's GPU pod  [STARTS BILLING]
	@uv run python -m common.pod up $(CFG)

gpu-down:  ## terminate it  [STOPS BILLING]
	@uv run python -m common.pod down $(CFG)

# ---- run on whichever pod has the GPUs -------------------------------------
boot:  ## download weights for this lab
	@uv run python -m common.pod prefetch $(CFG)

serve:  ## launch vLLM, wait for /health
	@uv run python -m common.serve $(CFG) --log labs/$(LAB)/results/vllm.log

smoke:  ## one request, print the reply
	@uv run python -m common.smoke $(CFG) --out labs/$(LAB)/results/smoke.json

bench:  ## concurrency sweep -> results.jsonl
	@uv run python -m common.runner $(CFG) --out labs/$(LAB)/results/results.jsonl

logs:  ## tail the vLLM log
	@tail -f labs/$(LAB)/results/vllm.log

clean:
	@rm -rf .pytest_cache **/__pycache__
