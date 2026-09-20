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

say "sshd keepalive"
# the base image ships ClientAliveInterval 0, so an idle session (Claude Code
# thinking, a long build) gets reaped by NAT and you see "broken pipe"
if ! grep -q '^ClientAliveInterval' /etc/ssh/sshd_config 2>/dev/null; then
  printf '\nClientAliveInterval 30\nClientAliveCountMax 20\nTCPKeepAlive yes\n' >> /etc/ssh/sshd_config
  # HUP only the LISTENER via its pidfile -- `pkill -HUP sshd` would also hit
  # the per-connection children and kill the very session running this script
  [ -f /run/sshd.pid ] && kill -HUP "$(cat /run/sshd.pid)" 2>/dev/null || true
fi

say "system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl rsync tmux jq build-essential ca-certificates openssh-client >/dev/null

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

say "herdr"
command -v herdr >/dev/null || curl -fsSL https://herdr.dev/install.sh | sh
# herdr keeps config, session.json and its sockets in ~/.config/herdr, which
# is already on the volume (persist .config above). Only the binary is
# ephemeral. Seed the config once; never clobber an edited one.
if [ ! -e "$HOME/.config/herdr/config.toml" ]; then
  mkdir -p "$HOME/.config/herdr"
  cat > "$HOME/.config/herdr/config.toml" <<'EOF'
# Full reference: herdr --default-config

[session]
# after a server restart (= every new pod), reopen agent panes into their
# previous conversation. Needs: herdr integration install claude
resume_agents_on_restore = true

[experimental]
# replay recent pane screen contents after a full server restart
pane_history = true
EOF
fi

# herdr plugins install under ~/.config/herdr, so they are on the volume and
# survive. But herdr-sidebar opens a preview by TYPING `herdr-sidebar --preview`
# into a fresh shell, and RunPod's /etc/rp_environment (sourced by ~/.bashrc)
# resets PATH in every shell -- so the command is not found and the sidebar
# reports a misleading "preview switch blocked". Link it somewhere PATH keeps.
# Only links what was installed by hand; this script never fetches a plugin.
for bin in "$HOME"/.config/herdr/plugins/github/herdr-sidebar-*/plugins/herdr-sidebar/target/release/herdr-sidebar; do
  [ -x "$bin" ] && ln -sfn "$bin" /usr/local/bin/herdr-sidebar
done

say "git identity"
git config --global user.name  "Qianli Hu"
git config --global user.email "qianlihuwork@gmail.com"
git config --global init.defaultBranch main
git config --global --replace-all safe.directory "$REPO_DIR"
# ~/.gitconfig is ephemeral but the gh token is on the volume: `gh auth login`
# wired the credential helper once, and a new pod never logs in again, so
# re-wire it here or `git push` fails on every pod after the first.
gh auth status >/dev/null 2>&1 && gh auth setup-git

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
