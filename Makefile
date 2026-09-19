# Inference-Infra — one entrypoint for every lab.
#
# Run every target from your Mac. Targets that need the GPU wrap themselves in
# ssh; you never have to remember which is which.
#
#   LAB=00-hello-gpu make plan up sync boot serve smoke fetch save down

LAB     ?= 00-hello-gpu
CFG     := labs/$(LAB)/config.yaml
REMOTE  := /workspace/Inference-Infra
POD     := .pod            # written by `make up`, gitignored
SSHOPTS := -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10

# host/port for the live pod, if there is one
-include $(POD)
SSH := ssh $(SSHOPTS) -p $(PORT) root@$(HOST)
ONPOD = $(SSH) "cd $(REMOTE) && LAB=$(LAB) $(1)"

.PHONY: help plan test up sync boot serve smoke logs fetch save status down ssh clean
.DEFAULT_GOAL := help

help:  ## show this help
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

# ---- local, no pod, no cost ------------------------------------------------
plan:  ## predicted VRAM budget for this lab
	@uv run python -m common.plan $(CFG)

test:  ## run the test suite
	@uv run --extra dev pytest -q

# ---- pod lifecycle ---------------------------------------------------------
up:  ## create the pod and wait for SSH  [STARTS BILLING]
	@uv run python -m common.pod up $(CFG) --write $(POD)

status:  ## live pods and what they cost
	@uv run python -m common.pod status

down:  ## terminate the pod  [STOPS BILLING]
	@uv run python -m common.pod down --pod-file $(POD) && rm -f $(POD)

ssh:  ## interactive shell on the pod (debugging escape hatch)
	@$(SSH)

# ---- code and data movement ------------------------------------------------
sync:  ## rsync code Mac -> pod
	@rsync -az --delete --info=stats1 \
	  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
	  --exclude 'results/*' --exclude '$(POD)' \
	  -e "ssh $(SSHOPTS) -p $(PORT)" ./ root@$(HOST):$(REMOTE)/

fetch:  ## rsync results pod -> Mac
	@rsync -az --info=stats1 -e "ssh $(SSHOPTS) -p $(PORT)" \
	  root@$(HOST):$(REMOTE)/labs/$(LAB)/results/ labs/$(LAB)/results/

save:  ## commit and push results
	@git add -A && git commit -q -m "$(LAB): results" && git push -q && echo "pushed"

# ---- work that happens on the GPU -----------------------------------------
boot:  ## uv sync + download weights  (on pod)
	@$(call ONPOD,make _boot)

serve:  ## launch vLLM, wait for /health  (on pod)
	@$(call ONPOD,make _serve)

smoke:  ## one request, print the reply  (on pod)
	@$(call ONPOD,make _smoke)

logs:  ## tail the vLLM log  (on pod)
	@$(SSH) "tail -f $(REMOTE)/labs/$(LAB)/results/vllm.log"

# ---- inner targets: these run ON the pod, invoked by the wrappers above ----
_boot:
	@export HF_HOME=/workspace/hf-cache && \
	 uv sync --quiet && \
	 uv run python -m common.pod prefetch $(CFG)

_serve:
	@export HF_HOME=/workspace/hf-cache && \
	 uv run python -m common.serve $(CFG) --log labs/$(LAB)/results/vllm.log

_smoke:
	@uv run python -m common.smoke $(CFG) --out labs/$(LAB)/results/smoke.json

clean:
	@rm -rf .pytest_cache **/__pycache__
