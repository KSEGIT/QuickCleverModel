#!/usr/bin/env bash
# Update the Linux/Docker Bonsai stack, prove it still serves, roll back if not.
#
#   ./update.sh              update, smoke-test every model, roll back on failure
#   ./update.sh --check      report drift only; changes nothing; exit 1 if behind
#   ./update.sh --rollback   return to the last recorded good state
#
# Linux/Docker only. macOS runs everything natively — see stack.sh.
#
# Why this smoke-tests instead of trusting a health check: on 2026-09-10 the
# box reported "Up 29 hours (healthy)" and /health returned {"status":"ok"}
# while every single model failed to load. The repo had been moved with
# `mv QuickCleverModel/ ../firesandbrain1/` under a running stack, Docker
# recreated the missing bind-mount source as an empty directory, and
# /app/models held no weights. Container liveness says nothing about whether
# the thing can answer. Only a real generation does.
set -uo pipefail

# BONSAI_UPDATE_ROOT is set by the re-exec below, where this file has been
# copied to a temp dir and dirname no longer points at the repo.
ROOT="${BONSAI_UPDATE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
COMPOSE_FILE="$ROOT/docker/compose.linux.yaml"
ENV_FILE="$ROOT/.env"
STATE="$ROOT/run/update-state"
LLAMA_IMAGE="bonsai-llama:cuda"
ROLLBACK_IMAGE="bonsai-llama:rollback"
WEBUI_IMAGE="ghcr.io/open-webui/open-webui:main"
LLAMA_CONTAINER="bonsai-llama-1"

log()  { printf '%s  %s\n' "$(date -u +%H:%M:%S)" "$*"; }
warn() { printf '%s  WARN %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die()  { printf '%s  FAIL %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; exit 1; }

# --- pure helpers, unit-tested in tests/test_update.py -----------------------

models_present() {
  # True when at least one weight file is visible under $1. Guards the exact
  # 2026-09-10 failure: a bind-mount source that exists but is empty.
  local dir="${1:-}"
  [[ -n "$dir" && -d "$dir" ]] || return 1
  # -print -quit stops at the first hit; the tree is ~11 GB and the only
  # question is whether it holds any weights at all.
  [[ -n "$(find "$dir" -name '*.gguf' -type f -print -quit 2>/dev/null)" ]]
}

parse_model_ids() {
  # /v1/models on stdin -> one alias per line. The smoke test loads all of
  # them, because the router loads each alias as its own child process and
  # one working model proves nothing about the others.
  python3 -c 'import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
for m in d.get("data", []):
    if m.get("id"):
        print(m["id"])'
}

pinned_bad() {
  # The last commit that failed its smoke test here, if any. do_rollback
  # appends, so the newest pin is the one that counts.
  sed -n 's/^bad=//p' "$STATE" 2>/dev/null | tail -1
}

smoke_ok() {
  # A /v1/chat/completions reply on stdin. Exit 0 only for a real generation.
  python3 -c 'import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)          # curl writes nothing when the connection fails
if not isinstance(d, dict) or "error" in d:
    sys.exit(1)
choices = d.get("choices") or []
if not choices:
    sys.exit(1)
msg = choices[0].get("message") or {}
# Bonsai is a thinking model: a short reply can hit max_tokens while still
# inside its reasoning block, arriving as content="" with the text in
# reasoning_content. That is a healthy generation, not a failure — scoring it
# as one would roll back a stack that works.
text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
sys.exit(0 if text.strip() else 1)'
}

# ask <url> [curl args...] -> prints "<http_code>\n<body>", returns curl's status
#
# Deliberately no -f: curl with -f prints NOTHING on an HTTP error and just
# returns 22. The body of a 500 is the entire diagnostic here — it carries
# `{"error":{"message":"model ... failed to load"}}`, the exact text this
# script exists to surface. With -f the operator gets a rollback and an empty
# reason in the journal.
ask() {
  local url="$1"; shift
  local out rc
  out="$(curl -sS -w $'\n%{http_code}' -H "Authorization: Bearer $BONSAI_API_KEY" \
        "$@" "$url" 2>/dev/null)"; rc=$?
  printf '%s\n%s' "${out##*$'\n'}" "${out%$'\n'*}"
  return "$rc"
}

# Exit status, and it has three meanings, not two:
#   0  every model answered
#   1  a model could not answer            -> a real outage, roll back
#   2  we could not ASK                    -> our problem, do NOT roll back
#
# The third is the point. A rotated BONSAI_API_KEY, a connection reset or a
# timeout would otherwise look exactly like a dead model, and tear down a
# stack that was serving perfectly.
smoke() {
  local resp code body ids id failed=0

  resp="$(ask "$API/v1/models" -m 15)" || {
    warn "could not reach $API/v1/models (curl failed)"; return 2; }
  code="${resp%%$'\n'*}"; body="${resp#*$'\n'}"
  if [[ "$code" != "200" ]]; then
    warn "/v1/models returned HTTP $code — cannot judge the stack: ${body:0:200}"
    return 2
  fi
  ids="$(printf '%s' "$body" | parse_model_ids)"
  [[ -n "$ids" ]] || { warn "/v1/models listed no models"; return 2; }

  while read -r id; do
    [[ -n "$id" ]] || continue
    log "  smoke: $id"
    resp="$(ask "$API/v1/chat/completions" -m 600 \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$id\",\"messages\":[{\"role\":\"user\",\"content\":\"say OK\"}],\"max_tokens\":8}")" || {
        warn "  smoke: $id — curl failed; cannot judge the stack"; return 2; }
    code="${resp%%$'\n'*}"; body="${resp#*$'\n'}"
    case "$code" in
      401|403|404)
        # Our credentials or our URL, not the model.
        warn "  smoke: $id returned HTTP $code — cannot judge the stack"
        return 2;;
    esac
    if printf '%s' "$body" | smoke_ok; then
      log "  smoke: $id OK"
    else
      warn "  smoke: $id FAILED (HTTP $code) — ${body:0:200}"
      failed=1
    fi
  done <<< "$ids"
  return "$failed"
}

# Sourced by the test suite to exercise the helpers above without running an
# update. Executing the script normally leaves BONSAI_UPDATE_LIB unset, the
# && short-circuits, and the `return` is never reached.
[[ -n "${BONSAI_UPDATE_LIB:-}" ]] && return 0

# --- run from a copy, never from the file we are about to rewrite -----------
#
# `git pull --ff-only` and `git reset --hard` both rewrite update.sh whenever
# an update touches this script. Bash reads a script lazily, by byte offset,
# so a file that changes underneath a running shell makes it resume at a stale
# offset and execute whatever bytes now sit there. That is worst during a
# rollback: the one code path whose whole job is to bring a broken stack back.
#
# Copying to a temp file and exec-ing that leaves bash reading a file nothing
# will touch. The copy deletes itself on exit.
if [[ -z "${BONSAI_UPDATE_REEXEC:-}" ]]; then
  copy="$(mktemp "${TMPDIR:-/tmp}/bonsai-update.XXXXXX")" || die "mktemp failed"
  # Only reached if a step below fails: a successful exec replaces this shell
  # and the child cleans up after itself.
  trap 'rm -f -- "$copy"' EXIT
  cat "${BASH_SOURCE[0]}" > "$copy" || die "could not copy myself to $copy"
  chmod +x "$copy"
  export BONSAI_UPDATE_REEXEC=1
  export BONSAI_UPDATE_ROOT="$ROOT"
  # execfail keeps the shell alive if the exec cannot happen, so the fallback
  # below is reachable. /tmp mounted noexec is ordinary hardening, and bash
  # can still READ a script it is not allowed to execute.
  shopt -s execfail
  exec "$copy" "$@"
  exec bash "$copy" "$@"
fi

# Captured now: inside a function BASH_SOURCE[0] is the function's source, so
# reading it at trap time would delete the wrong path.
SELF="${BASH_SOURCE[0]}"
# Two traps, deliberately. bash does not run an EXIT trap on an untrapped
# signal, so the copy would leak on systemd's SIGTERM at TimeoutStartSec —
# but bash also RESUMES after a non-EXIT handler returns, so a handler that
# only cleans up would let the run carry on pulling and building until
# SIGKILL arrives. The signal handler has to exit.
trap 'rm -f -- "$SELF"' EXIT
trap 'warn "interrupted — stopping"; rm -f -- "$SELF"; exit 143' INT TERM

# --- everything below needs a real box ---------------------------------------

# .env carries BONSAI_API_KEY and the bind addresses. Exported because compose
# reads them for interpolation too.
set -a
if [[ -f "$ENV_FILE" ]]; then
  # .env is per-machine and never in the repo, so there is nothing to follow.
  # shellcheck source=/dev/null
  . "$ENV_FILE"
fi
set +a
API="http://${LLAMA_BIND:-127.0.0.1}:8080"

compose() { docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"; }

local_image_digest() {
  # RepoDigests holds the manifest digest the image was pulled by, which is
  # the only local value comparable with a registry digest. .Id is a local
  # config hash and matches nothing upstream.
  docker image inspect "$1" \
    --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' 2>/dev/null \
    | sed -n 's/.*@//p'
}

remote_image_digest() {
  # The manifest-INDEX digest, which is the value a pull records in
  # RepoDigests. Two tempting shortcuts do not work:
  #   docker manifest inspect --verbose  returns one entry per platform, each
  #     with its own digest; none of them is the index digest.
  #   docker buildx imagetools inspect   needs the buildx CLI plugin, which
  #     the distro docker.io package does not ship (verified on the box).
  # So ask the registry, which needs nothing but python3 and the network.
  python3 - "$1" <<'PY' 2>/dev/null
import json, sys, urllib.error, urllib.request

ref = sys.argv[1]
name, _, tag = ref.rpartition(":")
first = name.split("/")[0]
# A first segment with a dot or a port is a registry host; otherwise this is
# a Docker Hub short name, and a bare one lives under library/.
if "/" not in name or ("." not in first and ":" not in first and first != "localhost"):
    registry, repo = "registry-1.docker.io", (name if "/" in name else "library/" + name)
else:
    registry, _, repo = name.partition("/")
url = "https://%s/v2/%s/manifests/%s" % (registry, repo, tag)
accept = ("application/vnd.oci.image.index.v1+json,"
          "application/vnd.docker.distribution.manifest.list.v2+json,"
          "application/vnd.oci.image.manifest.v1+json,"
          "application/vnd.docker.distribution.manifest.v2+json")


def head(token=None):
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("Accept", accept)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return urllib.request.urlopen(req, timeout=15)


try:
    r = head()
except urllib.error.HTTPError as e:
    if e.code != 401:
        sys.exit(1)
    # Anonymous pull: follow the Bearer challenge for a throwaway token.
    chal = e.headers.get("WWW-Authenticate", "")
    body = chal.split(" ", 1)[1] if " " in chal else ""
    parts = dict(p.split("=", 1) for p in body.split(",") if "=" in p)
    realm = parts.pop("realm", "").strip('"')
    if not realm:
        sys.exit(1)
    q = "&".join("%s=%s" % (k, v.strip('"')) for k, v in parts.items())
    try:
        with urllib.request.urlopen(realm + ("?" + q if q else ""), timeout=15) as t:
            tok = json.load(t)
        r = head(tok.get("token") or tok.get("access_token"))
    except Exception:
        sys.exit(1)
except Exception:
    sys.exit(1)

d = r.headers.get("Docker-Content-Digest")
if not d:
    sys.exit(1)
print(d)
PY
}

running_working_dir() {
  docker inspect "$LLAMA_CONTAINER" \
    --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}' \
    2>/dev/null
}

preflight() {
  # parse_model_ids and smoke_ok are python3, and curl carries every request.
  # A missing interpreter exits 127, which reads exactly like a dead model —
  # so it would tear down and roll back a perfectly healthy stack. A missing
  # tool must never be able to manufacture a fake outage.
  local missing=()
  local tool
  for tool in docker git curl python3; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done
  if (( ${#missing[@]} )); then
    die "missing required tool(s): ${missing[*]}
This is the Linux/Docker path; macOS uses ./stack.sh"
  fi
  [[ -f "$COMPOSE_FILE" ]] || die "no compose file at $COMPOSE_FILE"
  [[ -f "$ENV_FILE" ]]     || die "no .env at $ENV_FILE"
  [[ -n "${BONSAI_API_KEY:-}" ]] || die "BONSAI_API_KEY is not set in $ENV_FILE"

  models_present "$ROOT/models" || die \
"no *.gguf under $ROOT/models

Refusing to restart into an empty bind mount. That is the 2026-09-10 failure
mode: the container comes up healthy and every model load fails. Either run
./fetch-models.sh, or check the repo was not moved while the stack was up."

  # A running stack pinned to a different directory means compose would mount
  # that old path, not this one. Restarting would recreate the outage.
  local live; live="$(running_working_dir)"
  if [[ -n "$live" && "$live" != "$ROOT/docker" ]]; then
    die \
"the running stack was started from a different checkout

  running:  $live
  this one: $ROOT/docker

Stop it from the checkout that owns it, then start it from here:
  (cd ${live%/docker} && docker compose --env-file .env -f docker/compose.linux.yaml down)
  docker compose --env-file .env -f docker/compose.linux.yaml up -d"
  fi
}

wait_for_health() {
  # Condition polling, not a fixed sleep: a cold 27B load off a spinning disk
  # is minutes, a warm restart is seconds.
  local deadline=$((SECONDS + 180))
  while (( SECONDS < deadline )); do
    if curl -fsS -m 5 "$API/health" 2>/dev/null | grep -q '"ok"'; then
      return 0
    fi
    sleep 3
  done
  return 1
}

record_rollback_point() {
  local verified="${1:-0}"
  mkdir -p "$(dirname "$STATE")" \
    || die "cannot create $(dirname "$STATE") — refusing to update with no way back"
  local sha webui llama pins
  sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"
  webui="$(docker image inspect "$WEBUI_IMAGE" --format '{{.Id}}' 2>/dev/null)"
  llama="$(docker image inspect "$LLAMA_IMAGE" --format '{{.Id}}' 2>/dev/null)"
  # Retag rather than copy: the old layers stay on disk under a second name,
  # so rollback is a tag move and needs no network. Recording the image id
  # alongside it matters — if this tag fails, bonsai-llama:rollback still
  # points at whatever an earlier run left there, and restoring it would pair
  # one commit's source with another commit's binary.
  if [[ -n "$llama" ]]; then
    docker tag "$LLAMA_IMAGE" "$ROLLBACK_IMAGE" \
      || warn "could not tag $ROLLBACK_IMAGE; rollback may have no image to restore"
  else
    warn "no $LLAMA_IMAGE present to save"
  fi
  # do_rollback appends bad= pins here; a plain truncate would drop them.
  pins="$(grep '^bad=' "$STATE" 2>/dev/null)"
  # An unwritten rollback point is only discovered when it is needed, which
  # is after the stack is already broken. Fail now instead.
  printf 'sha=%s\nwebui=%s\nllama=%s\nverified=%s\n' \
    "$sha" "$webui" "$llama" "$verified" > "$STATE" \
    || die "cannot write the rollback point to $STATE — refusing to update with no way back"
  [[ -n "$pins" ]] && printf '%s\n' "$pins" >> "$STATE"
  log "rollback point: ${sha:0:12} / ${webui:0:19} (verified=$verified)"
}

# do_rollback [pin]
#
# "pin" is passed ONLY by the smoke-failure path. A build that failed on a
# network blip, a compose up that lost a race with tailscaled bringing up
# LLAMA_BIND2 (compose.linux.yaml warns about exactly this), or a human
# running --rollback deliberately must not blacklist a commit — the refusal
# message says the commit "already failed its smoke test", and that has to
# stay true.
do_rollback() {
  [[ -f "$STATE" ]] || die "no rollback point recorded at $STATE"
  local pin="${1:-}"
  local sha webui llama verified bad restored=1
  sha="$(sed -n 's/^sha=//p' "$STATE")"
  webui="$(sed -n 's/^webui=//p' "$STATE")"
  llama="$(sed -n 's/^llama=//p' "$STATE")"
  verified="$(sed -n 's/^verified=//p' "$STATE")"
  [[ "$verified" == "1" ]] || warn \
    "the recorded rollback point was never proven to serve — restoring it anyway"

  # The commit being left behind is the one that failed. Remember it before
  # the reset moves HEAD, or the next timer run fast-forwards straight back
  # onto it and the box goes dark again on the same commit, every week.
  bad="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"

  log "rolling back to ${sha:0:12}"
  if [[ -z "$sha" ]]; then
    warn "no commit recorded; leaving the checkout where it is"
    restored=0
  elif ! git -C "$ROOT" reset --hard "$sha"; then
    # Silence here is how a rollback reports success with the broken code
    # still checked out.
    warn "git reset to $sha FAILED — the broken commit is STILL checked out"
    restored=0
  fi

  local have
  have="$(docker image inspect "$ROLLBACK_IMAGE" --format '{{.Id}}' 2>/dev/null)"
  if [[ -z "$have" ]]; then
    warn "no $ROLLBACK_IMAGE to restore; keeping the current llama image"
    restored=0
  elif [[ -n "$llama" && "$have" != "$llama" ]]; then
    # A stale tag would pair this commit's source with another commit's binary.
    warn "$ROLLBACK_IMAGE is ${have:0:19}, not the recorded ${llama:0:19} — not restoring it"
    restored=0
  else
    docker tag "$ROLLBACK_IMAGE" "$LLAMA_IMAGE" \
      || { warn "could not restore $LLAMA_IMAGE"; restored=0; }
  fi
  if [[ -n "$webui" ]] && docker image inspect "$webui" >/dev/null 2>&1; then
    docker tag "$webui" "$WEBUI_IMAGE" || warn "could not restore $WEBUI_IMAGE"
  fi

  # Appended after the reset so the pin survives even when the reset failed.
  if [[ "$pin" == "pin" && -n "$bad" && "$bad" != "$sha" ]]; then
    printf 'bad=%s\n' "$bad" >> "$STATE" \
      || warn "could not pin the failed commit; it may be retried next week"
  fi

  compose up -d --force-recreate || warn "compose failed during rollback"

  if wait_for_health && smoke; then
    if (( restored )); then
      log "rollback verified — stack is serving again"
      return 0
    fi
    # Answering is not the same as being restored: a transient registry
    # failure can leave the bad code checked out and still serving.
    warn "the stack answers, but the previous state was NOT fully restored"
    return 1
  fi
  warn "ROLLBACK DID NOT RECOVER THE STACK — needs a human"
  return 1
}

cmd_check() {
  preflight
  local behind=0 rc=0
  if git -C "$ROOT" fetch --quiet 2>/dev/null; then
    behind="$(git -C "$ROOT" rev-list --count 'HEAD..@{u}' 2>/dev/null || echo 0)"
  else
    warn "could not fetch; commit drift unknown"
  fi
  if (( behind > 0 )); then
    log "behind origin by $behind commit(s)"
    rc=1
  else
    log "code up to date"
  fi

  local have want
  have="$(local_image_digest "$WEBUI_IMAGE")"
  want="$(remote_image_digest "$WEBUI_IMAGE")"
  if [[ -z "$have" ]]; then
    log "open-webui image not present locally"
    rc=1
  elif [[ -z "$want" ]]; then
    # Never report "up to date" from a comparison that did not happen.
    warn "could not read the registry digest; image drift unknown"
    rc=1
  elif [[ "$have" != "$want" ]]; then
    log "open-webui image is behind"
    log "  local  $have"
    log "  remote $want"
    rc=1
  else
    log "open-webui image up to date"
  fi
  return "$rc"
}

cmd_update() {
  preflight

  # A commit that already failed here must not be walked onto again. Without
  # this the rolled-back branch is a plain ancestor of origin, so next week's
  # run fast-forwards right back onto the bad commit, fails, rolls back, and
  # repeats — the box goes down every week with only a journal line to say so.
  local target bad
  git -C "$ROOT" fetch --quiet 2>/dev/null || warn "fetch failed; using what is already here"
  target="$(git -C "$ROOT" rev-parse '@{u}' 2>/dev/null)"
  bad="$(pinned_bad)"
  if [[ -n "$bad" && "$target" == "$bad" ]]; then
    die "upstream is still at ${target:0:12}, which already failed its smoke test
here and was rolled back. Nothing to do until a newer commit lands.

To try it again anyway:
  sed -i '/^bad=/d' $STATE"
  fi

  # --rollback promises the last version that WORKED. A point recorded on a
  # stack that was already dark would make that a lie, so prove it first.
  local verified=0
  log "checking the current stack before changing anything"
  if wait_for_health && smoke; then
    verified=1
  else
    warn "the stack is not serving BEFORE this update — recording the rollback point as unverified"
  fi
  record_rollback_point "$verified"

  log "pulling code"
  if ! git -C "$ROOT" pull --ff-only; then
    die "git pull --ff-only failed — the checkout has diverged, fix it by hand"
  fi

  log "building llama image"
  compose build llama || { warn "build failed"; do_rollback; exit 1; }

  log "pulling open-webui"
  compose pull open-webui || warn "pull failed; keeping the current image"

  log "restarting stack"
  compose up -d || { warn "compose up failed"; do_rollback; exit 1; }

  log "waiting for health"
  wait_for_health || { warn "never became healthy"; do_rollback; exit 1; }

  log "smoke testing every model"
  smoke; local verdict=$?
  if (( verdict == 2 )); then
    # We could not ask. Tearing down a stack we failed to interrogate would
    # turn our own broken credentials or network into a real outage.
    die "could not determine whether the stack works, so this run will not
judge it. The update is applied and NOT rolled back — check the warnings
above, then run ./update.sh --rollback yourself if you want the old version."
  elif (( verdict != 0 )); then
    warn "smoke test failed after update — rolling back"
    do_rollback pin
    exit 1
  fi

  log "update complete and verified"
}

# A here-doc, not a line range out of the header: `sed -n 2,10p` stopped in
# the middle of a sentence, and any edit to the comment block silently
# changed what --help printed.
usage() {
  cat <<'USAGE'
Update the Linux/Docker Bonsai stack, prove it still serves, roll back if not.

  ./update.sh              update, smoke-test every model, roll back on failure
  ./update.sh --check      report drift only; changes nothing; exit 1 if behind
  ./update.sh --rollback   return to the last recorded good state

Linux/Docker only. macOS runs everything natively — see stack.sh.
USAGE
}

case "${1:-}" in
  "")         cmd_update;;
  --check)    cmd_check;;
  --rollback) preflight; do_rollback;;
  -h|--help)  usage;;
  *)          usage >&2; exit 2;;
esac
