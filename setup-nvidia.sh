#!/usr/bin/env bash
# Host setup for the Linux + NVIDIA path of this stack (docker/compose.linux.yaml).
#
#   ./setup-nvidia.sh           detect what's missing and install it
#   ./setup-nvidia.sh --check   report only — no changes
#
# Brings a fresh Ubuntu box with an NVIDIA RTX card to the point where
# `docker compose --env-file .env -f docker/compose.linux.yaml up --build -d`
# works: NVIDIA driver >= 525 (required by the CUDA 12.4 images), Docker CE,
# and nvidia-container-toolkit wired into the Docker daemon.
#
# Every step detects first and installs ONLY what is missing, so re-running on
# a fully set-up machine is a no-op that prints a row of OKs. Targets Ubuntu
# 20.04 / 22.04 / 24.04; Debian-with-apt mostly works too but is untested.
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

# --- 0. Platform and privilege gate ------------------------------------------

# This script is apt-based. Refuse anything else up front rather than failing
# halfway through with a confusing error.
[[ -r /etc/os-release ]] || die "no /etc/os-release — this script targets Ubuntu/Debian with apt"
# shellcheck source=/dev/null
. /etc/os-release
case "${ID:-}" in
  ubuntu) DOCKER_DISTRO="ubuntu" ;;
  debian) DOCKER_DISTRO="debian" ;;
  *) die "unsupported OS '${ID:-unknown}' — targets Ubuntu 20.04/22.04/24.04 (apt-based)" ;;
esac
command -v apt-get >/dev/null 2>&1 || die "apt-get not found — this script only works on apt-based systems"
CODENAME="${VERSION_CODENAME:-}"
[[ -n "$CODENAME" ]] || die "no VERSION_CODENAME in /etc/os-release — cannot select the Docker apt repo"

# Installation needs root. Run through sudo rather than re-execing the whole
# script, so the invoking user stays known (for the docker group below).
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  if [[ $CHECK -eq 0 ]]; then
    command -v sudo >/dev/null 2>&1 || die "not root and sudo not found — re-run as root"
    sudo -n true 2>/dev/null || { info "installation needs sudo"; sudo -v || die "sudo access refused"; }
  fi
  SUDO="sudo"
fi
# Who to add to the docker group: the real user even under `sudo ./setup-nvidia.sh`.
TARGET_USER="${SUDO_USER:-${USER}}"

[[ $CHECK -eq 1 ]] && info "check mode — reporting only, no changes"

# --- 1. NVIDIA GPU present ----------------------------------------------------
# This script is only useful on NVIDIA boxes; if lspci sees no NVIDIA device
# there is nothing to set up here.
info "checking for an NVIDIA GPU"
if ! command -v lspci >/dev/null 2>&1; then
  if [[ $CHECK -eq 1 ]]; then
    would "apt-get install -y pciutils (provides lspci)"
    warn "lspci missing — cannot detect the GPU in check mode"
  else
    $SUDO apt-get update -qq
    $SUDO apt-get install -y pciutils
  fi
fi
if command -v lspci >/dev/null 2>&1; then
  if lspci | grep -qi nvidia; then
    ok "NVIDIA GPU present: $(lspci | grep -i nvidia | head -1 | cut -d: -f3- | sed 's/^ //')"
  else
    die "no NVIDIA device in lspci output — this script is only for NVIDIA boxes"
  fi
fi

# --- 2. NVIDIA driver >= 525 --------------------------------------------------
# CUDA 12.4 (what docker/Dockerfile builds against) needs driver >= 525.
# nvidia-smi both proves the driver is loaded and reports its version.
info "checking NVIDIA driver (need >= 525 for CUDA 12.4)"
driver_version=""
if command -v nvidia-smi >/dev/null 2>&1; then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
fi
driver_major="${driver_version%%.*}"
if [[ "$driver_major" =~ ^[0-9]+$ ]] && (( driver_major >= 525 )); then
  ok "driver $driver_version (>= 525)"
else
  if [[ -n "$driver_version" ]]; then
    warn "driver $driver_version is too old (< 525)"
  else
    warn "no working NVIDIA driver found"
  fi
  if [[ $CHECK -eq 1 ]]; then
    would "apt-get install -y ubuntu-drivers-common && ubuntu-drivers install"
    would "reboot if the freshly installed driver is not loaded yet, then re-run"
  else
    info "installing the recommended driver via ubuntu-drivers"
    $SUDO apt-get update
    $SUDO apt-get install -y ubuntu-drivers-common
    $SUDO ubuntu-drivers install
    # A freshly installed driver is not loaded until reboot, and nvidia-smi
    # will keep failing until then. Detect that and stop — continuing would
    # only fail at the container-toolkit verification below.
    if ! nvidia-smi >/dev/null 2>&1; then
      echo
      warn "driver packages installed, but nvidia-smi still fails — the new driver loads on boot."
      warn "REBOOT, then re-run ./setup-nvidia.sh to finish (Docker + container toolkit)."
      exit 0
    fi
    ok "driver installed and loaded"
  fi
fi

# --- 3. Docker CE + compose plugin -------------------------------------------
# docker-compose-plugin is required: compose.linux.yaml is driven with
# `docker compose`, not the legacy docker-compose binary.
info "checking Docker"
docker_ok=0
if command -v docker >/dev/null 2>&1 && docker --version >/dev/null 2>&1; then
  # `docker info` proves the daemon is reachable, not just the CLI installed.
  if docker info >/dev/null 2>&1; then
    ok "$(docker --version)"
    docker_ok=1
  elif [[ -n "$SUDO" ]] && $SUDO docker info >/dev/null 2>&1; then
    # Daemon is up but this user cannot reach it — almost always the
    # docker-group membership below, which needs a re-login to take effect.
    ok "$(docker --version) (daemon reachable via sudo)"
    docker_ok=1
    warn "docker daemon is not reachable as $TARGET_USER — group membership pending re-login?"
  fi
fi
if [[ $docker_ok -eq 0 ]]; then
  if [[ $CHECK -eq 1 ]]; then
    would "install Docker CE from the official apt repo for $DOCKER_DISTRO/$CODENAME"
    would "packages: docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin"
  else
    info "installing Docker CE from the official $DOCKER_DISTRO apt repo ($CODENAME)"
    # Official Docker apt repo setup — https://docs.docker.com/engine/install/ubuntu/
    $SUDO apt-get update
    $SUDO apt-get install -y ca-certificates curl gnupg
    $SUDO install -m 0755 -d /etc/apt/keyrings
    curl -fsSL "https://download.docker.com/linux/$DOCKER_DISTRO/gpg" | $SUDO tee /etc/apt/keyrings/docker.asc >/dev/null
    $SUDO chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$DOCKER_DISTRO $CODENAME stable" \
      | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
    $SUDO apt-get update
    $SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ok "Docker installed: $(docker --version)"
  fi
fi
# Let the invoking user run docker without sudo. Idempotent: usermod -aG is a
# no-op when already a member, but check first to keep --check output honest.
if id -nG "$TARGET_USER" 2>/dev/null | grep -qw docker; then
  ok "user $TARGET_USER is in the docker group"
elif getent group docker >/dev/null 2>&1; then
  if [[ $CHECK -eq 1 ]]; then
    would "usermod -aG docker $TARGET_USER"
  else
    $SUDO usermod -aG docker "$TARGET_USER"
    warn "added $TARGET_USER to the docker group — LOG OUT AND BACK IN before running docker without sudo"
  fi
fi

# --- 4. nvidia-container-toolkit ----------------------------------------------
# This is what makes `--gpus all` (and compose's `gpus: all`) work: it injects
# the driver libraries into containers at run time.
info "checking nvidia-container-toolkit"
if dpkg -l nvidia-container-toolkit >/dev/null 2>&1 || command -v nvidia-ctk >/dev/null 2>&1; then
  ok "nvidia-container-toolkit installed"
else
  if [[ $CHECK -eq 1 ]]; then
    would "add the NVIDIA libnvidia-container apt repo and install nvidia-container-toolkit"
    would "nvidia-ctk runtime configure --runtime=docker && systemctl restart docker"
  else
    info "installing nvidia-container-toolkit from the NVIDIA apt repo"
    # Official repo setup — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    $SUDO apt-get update
    $SUDO apt-get install -y nvidia-container-toolkit
    # Register the NVIDIA runtime with Docker, then bounce the daemon so the
    # new /etc/docker/daemon.json takes effect.
    $SUDO nvidia-ctk runtime configure --runtime=docker
    $SUDO systemctl restart docker
    ok "nvidia-container-toolkit installed and docker restarted"
  fi
fi

# --- 5. End-to-end verification ----------------------------------------------
# The real test: can a container actually see the GPU? Uses the same CUDA 12.4
# base line as docker/Dockerfile. Skipped in --check mode (it pulls an image
# and runs a container — not a read-only probe).
if [[ $CHECK -eq 1 ]]; then
  info "skipping GPU-in-Docker verification in check mode"
else
  info "verifying GPU access from a container"
  if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi; then
    ok "GPU is visible inside Docker"
  elif [[ -n "$SUDO" ]] && $SUDO docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi; then
    ok "GPU is visible inside Docker (via sudo — re-login for group membership)"
  else
    die "container could not see the GPU — check driver >= 525 and nvidia-container-toolkit setup above"
  fi
fi

# --- Done ---------------------------------------------------------------------
echo
info "setup complete — next steps:"
cat <<'EOF'

    ./fetch-models.sh        # ~11 GB, both models — lands in models/
    printf 'BONSAI_API_KEY=bonsai-%s\n' "$(openssl rand -hex 20)" > .env && chmod 600 .env
    # --env-file .env is REQUIRED (compose interpolation reads docker/.env otherwise)
    docker compose --env-file .env -f docker/compose.linux.yaml up --build -d

  UI on http://127.0.0.1:9090, API on :8080. See README "Linux + NVIDIA (Docker)".
EOF
