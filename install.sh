#!/usr/bin/env bash
# One-command installer for the Linux + NVIDIA path of this stack
# (docker/compose.linux.yaml). Clone the repo, run this, open the UI:
#
#   ./install.sh           full install — safe to re-run, every step is idempotent
#   ./install.sh --check   report what would happen — no changes, no downloads
#
# Orchestrates, in order:
#   setup-nvidia.sh        NVIDIA driver >= 525, Docker CE, nvidia-container-toolkit
#   hf CLI                 pipx first, pip --user fallback (for the weight download)
#   fetch-models.sh        ~11 GB of GGUF weights into models/ (resumable)
#   .env                   BONSAI_API_KEY (generated once, never overwritten)
#   compose up --build -d  CUDA build of the fork's llama-server + Open WebUI
#   readiness              waits on :8080/health and :9090, then opens the UI
#
# The heavy lifting is deliberately delegated: GPU/Docker detection and install
# logic lives in setup-nvidia.sh (already proven on the target machine), weight
# download in fetch-models.sh. This script only sequences them and fills the
# gaps in between. Re-running after any failure picks up where things left off.
set -euo pipefail

CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

# Colored step output, disabled when stdout is not a terminal.
if [[ -t 1 ]]; then
  C_INFO=$'\033[0;36m'; C_OK=$'\033[0;32m'; C_WARN=$'\033[0;33m'; C_ERR=$'\033[0;31m'; C_OFF=$'\033[0m'
else
  C_INFO=''; C_OK=''; C_WARN=''; C_ERR=''; C_OFF=''
fi
info() { printf '%s==>%s %s\n' "$C_INFO" "$C_OFF" "$*"; }
ok()   { printf '%s ok %s %s\n' "$C_OK" "$C_OFF" "$*"; }
warn() { printf '%swarn%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }
die()  { printf '%serr %s %s\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }

# In --check mode, print the command instead of running it.
would() { printf '  would: %s\n' "$*"; }

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# --- 0. Platform gate ---------------------------------------------------------
# This is the Linux path only. macOS runs inference natively on Metal with a
# completely different setup — point those users at the right instructions.
[[ "$(uname -s)" == "Linux" ]] || \
  die "this installer is Linux-only. On macOS, see docs/macos.md 'First-time setup (macOS)'."

# Installation needs root. Run through sudo rather than re-execing the whole
# script, so the invoking user stays known — same approach as setup-nvidia.sh.
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  if [[ $CHECK -eq 0 ]]; then
    command -v sudo >/dev/null 2>&1 || die "not root and sudo not found — re-run as root"
    sudo -n true 2>/dev/null || { info "installation needs sudo"; sudo -v || die "sudo access refused"; }
  fi
  SUDO="sudo"
fi

[[ $CHECK -eq 1 ]] && info "check mode — reporting only, no changes"

# --- 1. GPU, driver, Docker, nvidia-container-toolkit -------------------------
# setup-nvidia.sh detects what's missing and installs ONLY that, so on an
# already-set-up machine this is a no-op that prints a row of OKs. A non-zero
# exit aborts this script too (set -e), which is the propagation we want.
info "host setup (GPU driver / Docker / container toolkit)"
if [[ $CHECK -eq 1 ]]; then
  ./setup-nvidia.sh --check
else
  ./setup-nvidia.sh
  # After a FRESH driver install, setup-nvidia.sh exits 0 with a "REBOOT"
  # message because the new driver only loads on boot. Detect that state here:
  # if nvidia-smi still fails there is no point continuing — the weight
  # download would work but the compose stack could never see the GPU.
  if ! nvidia-smi >/dev/null 2>&1; then
    echo
    warn "NVIDIA driver installed but not loaded yet — it loads on boot."
    warn "REBOOT, then re-run ./install.sh — it will resume from the weight download."
    exit 0
  fi
fi

# In --check mode, say what the remaining steps WOULD do and stop: no
# downloads, no builds, no writes.
if [[ $CHECK -eq 1 ]]; then
  echo
  info "later steps would:"
  would "install the hf CLI (pipx install huggingface_hub[cli], pip --user fallback) — if missing"
  would "./fetch-models.sh — ~11 GB of GGUF weights into models/ (idempotent, resumable)"
  would "generate .env with a random BONSAI_API_KEY (skipped if .env already exists)"
  would "docker compose --env-file .env -f docker/compose.linux.yaml up --build -d"
  would "  (first build takes 10-20 minutes — CUDA compile of the llama.cpp fork)"
  would "wait for http://127.0.0.1:8080/health and http://127.0.0.1:9090, then open the UI"
  exit 0
fi

# --- 2. hf CLI ----------------------------------------------------------------
# fetch-models.sh needs the Hugging Face CLI. Ubuntu >= 23.04 refuses a bare
# pip install (PEP 668 externally-managed), so go through pipx first and fall
# back to pip --user on older releases.
info "checking the hf CLI (for the weight download)"
if command -v hf >/dev/null 2>&1; then
  ok "hf CLI present: $(command -v hf)"
else
  info "installing the hf CLI"
  # shellcheck disable=SC2086  # $SUDO is intentionally word-split ("" or "sudo")
  if $SUDO apt-get update && $SUDO apt-get install -y pipx && pipx install "huggingface_hub[cli]"; then
    ok "hf installed via pipx"
  else
    warn "pipx route failed — falling back to pip --user"
    python3 -m pip install --user "huggingface_hub[cli]" \
      || die "could not install the hf CLI via pipx or pip — install huggingface_hub[cli] manually"
  fi
  # Both routes land in ~/.local/bin, which is not always on PATH in the
  # invoking shell (pipx only edits shell rc files for FUTURE logins).
  export PATH="$HOME/.local/bin:$PATH"
  command -v hf >/dev/null 2>&1 \
    || die "hf still not on PATH after install — add $HOME/.local/bin to PATH and re-run"
  ok "hf CLI ready: $(command -v hf)"
fi

# --- 3. Weights (~11 GB) ------------------------------------------------------
# BEFORE any compose command, on purpose: fetch-models.sh creates models/ with
# the invoking user's ownership. If docker creates the bind-mount source first
# (because it is missing at `compose up`), it is root-owned and the download
# fails. Idempotent — files with the expected size are skipped.
info "fetching model weights (~11 GB — skipped for files already present)"
./fetch-models.sh

# --- 4. .env with BONSAI_API_KEY ----------------------------------------------
# The API key is enforced by the llama container and passed to Open WebUI for
# its upstream connection. Generate once, NEVER overwrite — an existing .env
# may hold a key other services already use.
if [[ -f .env ]]; then
  ok ".env already exists — keeping it"
else
  printf 'BONSAI_API_KEY=bonsai-%s\n' "$(openssl rand -hex 20)" > .env
  chmod 600 .env
  ok ".env created with a fresh BONSAI_API_KEY (mode 600)"
fi

# --- 5. Docker access prefix ---------------------------------------------------
# setup-nvidia.sh may have just added this user to the docker group, which
# only takes effect after re-login. Rather than forcing that, fall back to
# sudo for this run and warn.
DOCKER=""
if docker info >/dev/null 2>&1; then
  ok "docker daemon reachable as $(id -un)"
elif $SUDO docker info >/dev/null 2>&1; then
  DOCKER="sudo"
  warn "docker needs sudo for this user — if setup-nvidia.sh just added you to"
  warn "the docker group, LOG OUT AND BACK IN to drop the sudo requirement."
else
  die "docker daemon unreachable even via sudo — re-run ./setup-nvidia.sh"
fi

# --- 6. Compose plugin ---------------------------------------------------------
# compose.linux.yaml is driven with `docker compose`, not the legacy
# docker-compose binary. setup-nvidia.sh installs docker-compose-plugin with
# Docker CE, so this is only a fallback for distro-provided Docker.
if $DOCKER docker compose version >/dev/null 2>&1; then
  ok "compose plugin present ($($DOCKER docker compose version --short 2>/dev/null || echo present))"
else
  info "installing the compose plugin"
  # docker-compose-v2 is Ubuntu's own package; docker-compose-plugin is the
  # name in the official Docker apt repo. Try both.
  $SUDO apt-get update
  $SUDO apt-get install -y docker-compose-v2 || $SUDO apt-get install -y docker-compose-plugin \
    || die "could not install a compose plugin — install docker-compose-plugin manually"
  $DOCKER docker compose version >/dev/null 2>&1 \
    || die "compose plugin still not working after install"
  ok "compose plugin installed"
fi

# --- 7. Build and start the stack ----------------------------------------------
# --env-file .env is REQUIRED (not a nicety): compose variable interpolation
# reads the shell env and the project-dir .env (docker/), not the repo-root
# one — without this flag the OPENAI_API_KEY line in compose.linux.yaml fails
# with "required variable is missing a value".
COMPOSE=(compose --env-file .env -f docker/compose.linux.yaml)
warn "the first build takes 10-20 minutes — it compiles the llama.cpp fork with CUDA"
info "building and starting the stack (llama-server + Open WebUI)"
$DOCKER docker "${COMPOSE[@]}" up --build -d

# --- 8. Readiness --------------------------------------------------------------
# llama-server maps GBs of weights on first request, and Open WebUI runs DB
# migrations — both answer TCP well before they can serve. Poll the HTTP
# endpoints, and abort early if a container dies so the user sees the logs
# instead of watching dots for 15 minutes.
stack_exited() {
  [[ -n "$($DOCKER docker "${COMPOSE[@]}" ps --status exited --quiet 2>/dev/null)" ]]
}

wait_for() {
  local url="$1" limit="$2" label="$3" waited=0
  printf '%s==>%s waiting for %s ' "$C_INFO" "$C_OFF" "$label"
  while (( waited < limit )); do
    if curl -sf -o /dev/null "$url" 2>/dev/null; then
      printf '\n'
      ok "$label is up (${waited}s)"
      return 0
    fi
    if stack_exited; then
      printf '\n' >&2
      warn "a container exited while waiting for $label — last 30 log lines:"
      $DOCKER docker "${COMPOSE[@]}" logs --tail=30 >&2 || true
      die "stack failed to start — see the logs above"
    fi
    printf '.'
    sleep 5
    waited=$((waited + 5))
  done
  printf '\n' >&2
  die "timed out after $((limit / 60)) minutes waiting for $label ($url) — check: $DOCKER docker ${COMPOSE[*]} logs"
}

# /health is public by upstream design (no API key needed) — see
# docs/architecture.md "Ports & security". 15 minutes: model load on first
# request is genuinely slow.
wait_for "http://127.0.0.1:8080/health" 900 "llama-server (:8080)"
wait_for "http://127.0.0.1:9090/" 300 "Open WebUI (:9090)"

# --- Done ----------------------------------------------------------------------
echo
info "install complete — the stack is running"
cat <<'EOF'

    chat UI : http://127.0.0.1:9090
    API     : http://127.0.0.1:8080  (OpenAI-compatible, needs the key in .env)

    In the UI's model dropdown, pick:
      bonsai-27b-1bit      <- start here on 8 GB VRAM (RTX 3070 Ti), comfortable
      bonsai-27b-ternary   higher quality, tight on 8 GB at the default ctx

    Both ports are loopback-only — see docs/architecture.md "Ports & security".
EOF

# Open the UI when there is a desktop session. Backgrounded and detached so
# xdg-open's own lifetime does not hold this script (or the terminal) open.
if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
  (xdg-open "http://127.0.0.1:9090" >/dev/null 2>&1 &)
fi
