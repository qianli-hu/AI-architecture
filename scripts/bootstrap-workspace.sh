#!/usr/bin/env bash
# Build the remote workspace from nothing. Idempotent -- safe to re-run.
#
#   curl -fsSL https://raw.githubusercontent.com/qianli-hu/AI-architecture/main/scripts/bootstrap-workspace.sh | bash
#
# This script is the reason pinning to US-MO-2 is cheap: the workspace is
# reproducible from git, so the network volume stays a cache and never
# becomes a vault. If you install something by hand, ADD IT HERE.
set -euo pipefail

WORKSPACE=/workspace
REPO_URL=https://github.com/qianli-hu/AI-architecture.git
REPO_DIR=$WORKSPACE/Inference-Infra
export HF_HOME=$WORKSPACE/hf-cache

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }

say "persist home config onto the volume"
# $HOME lives on the container disk and dies with the pod. These paths hold
# credentials and state worth keeping, so they live on the network volume and
# are symlinked back into $HOME. Idempotent: re-running relinks, and an
# existing real directory is migrated rather than clobbered.
PERSIST=$WORKSPACE/.home
mkdir -p "$PERSIST"

persist() {                       # persist <path-under-$HOME> <dir|file>
  local name=$1 kind=$2 src=$HOME/$1 dst=$PERSIST/$1
  [ -L "$src" ] && return 0                      # already linked
  if [ -e "$src" ]; then
    if [ -e "$dst" ]; then rm -rf "$src"         # volume copy wins
    else mv "$src" "$dst"; fi                    # migrate into the volume
  fi
  # a dir must exist before something mkdirs inside it; a file may dangle,
  # since writing through the link creates the target
  [ "$kind" = dir ] && mkdir -p "$dst"
  ln -s "$dst" "$src"
  printf '  %-16s -> %s\n' "~/$name" "$dst"
}

persist .config       dir    # gh auth token
persist .claude       dir    # Claude Code config
persist .claude.json  file   # Claude Code state
persist .bash_history file

say "system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl rsync tmux jq build-essential ca-certificates >/dev/null

say "uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

say "node + claude code"
if ! command -v node >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
  apt-get install -y -qq nodejs >/dev/null
fi
command -v claude >/dev/null || npm install -g @anthropic-ai/claude-code >/dev/null

say "gh cli"
if ! command -v gh >/dev/null; then
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg status=none
  echo "deb [signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
  apt-get update -qq && apt-get install -y -qq gh >/dev/null
fi

say "git identity"
git config --global user.name  "Qianli Hu"
git config --global user.email "qianlihuwork@gmail.com"
git config --global init.defaultBranch main
git config --global --add safe.directory "$REPO_DIR"

say "repo"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

say "python deps"
cd "$REPO_DIR" && uv sync --extra dev --quiet

say "shell env"
grep -q 'HF_HOME' ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'RC'

# --- inference-infra ---
export HF_HOME=/workspace/hf-cache
export PATH="$HOME/.local/bin:$PATH"
# secrets and per-machine env live on the volume, not in this ephemeral file
[ -f /workspace/.home/.bash_env ] && . /workspace/.home/.bash_env
cd /workspace/Inference-Infra 2>/dev/null || true
RC

mkdir -p "$HF_HOME"
touch "$PERSIST/.bash_env"

say "verify"
printf '  uv      %s\n' "$(uv --version 2>/dev/null || echo MISSING)"
printf '  node    %s\n' "$(node --version 2>/dev/null || echo MISSING)"
printf '  claude  %s\n' "$(claude --version 2>/dev/null || echo MISSING)"
printf '  gh      %s\n' "$(gh --version 2>/dev/null | head -1 || echo MISSING)"
printf '  gpus    %s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd', ' || echo 'none (CPU pod)')"
cd "$REPO_DIR" && uv run --extra dev pytest -q 2>&1 | tail -1

cat <<'NEXT'

  Next, by hand (interactive, can't be scripted):
    gh auth login                 # HTTPS
    claude                        # then /login
    echo 'export HF_TOKEN=...' >> /workspace/.home/.bash_env

  You only do these ONCE. ~/.config and ~/.claude are symlinked onto the
  network volume, so both logins survive pod termination.

NEXT
