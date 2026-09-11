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

# The file actually running. Captured here because inside a function
# BASH_SOURCE[0] names that function's source, and several guards below
# compare against it.
SELF="${BASH_SOURCE[0]}"

# True only when this process is one of our own temp copies.
#
# Two conditions, and the second is what makes it safe. Matching the variable
# alone is not: it is ordinary environment, so a systemd Environment= line, an
# inherited export or a hand-typed value can name anything — including the
# repo's own update.sh, which then got the delete-on-exit trap and was removed
# from disk. mktemp names every copy bonsai-update.XXXXXX and the repo's file
# is update.sh, so the pattern can never match the real one.
looks_like_temp_copy() {
  [[ "${BONSAI_UPDATE_REEXEC:-}" == "$SELF" ]] || return 1
  case "${SELF##*/}" in
    bonsai-update.*) return 0;;
    *) return 1;;
  esac
}

# BONSAI_UPDATE_ROOT carries the repo across the re-exec, where this file sits
# in a temp dir and dirname no longer finds it. Trusted only from a real copy,
# so a leaked value cannot aim the script at someone else's checkout.
if looks_like_temp_copy && [[ -n "${BONSAI_UPDATE_ROOT:-}" ]]; then
  ROOT="$BONSAI_UPDATE_ROOT"
else
  ROOT="$(cd "$(dirname "$SELF")" && pwd)"
fi
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

cleanup() {
  # Remove the DIRECTORY, not the individual paths. The curl config holds the
  # API key, and a path assigned lazily inside a subshell — `ask` reached
  # through a pipeline, say — is invisible to this trap in the parent, so the
  # key file survived the run. The directory is known before any subshell can
  # exist, so removing it catches whatever was put inside.
  [[ -n "${RUN_TMP:-}" ]] && rm -rf -- "$RUN_TMP"
  [[ -n "${CURL_CONF:-}" ]] && rm -f -- "$CURL_CONF"
  [[ -n "${CURL_ERR:-}" ]] && rm -f -- "$CURL_ERR"
  return 0
}

# Installed unconditionally, including library mode, because the file it
# removes carries the API key and library mode installs no other traps — a
# sourced smoke() left one behind in /tmp for good. The re-exec block below
# replaces this with a fuller handler that also removes the temp copy; a later
# trap on the same signal supersedes an earlier one.
trap 'cleanup' EXIT
RUN_TMP=""

# --- pure helpers, unit-tested in tests/test_update.py -----------------------

models_present() {
  # True when at least one weight file is visible under $1. Guards the exact
  # 2026-09-10 failure: a bind-mount source that exists but is empty.
  local dir="${1:-}"
  [[ -n "$dir" && -d "$dir" ]] || return 1
  # -L because weights routinely live on a second drive with models/ (or one
  # model dir) symlinked in. Docker resolves that bind mount fine, so a plain
  # -P find would report "no weights" for a stack that works, and preflight
  # would block update, --check and --rollback alike.
  # -print -quit stops at the first hit; the tree is ~11 GB and the only
  # question is whether it holds any weights at all.
  [[ -n "$(find -L "$dir" -name '*.gguf' -type f -print -quit 2>/dev/null)" ]]
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
if not isinstance(d, dict):
    sys.exit(1)          # a bare scalar would raise AttributeError into the journal
for m in d.get("data", []):
    if m.get("id"):
        print(m["id"])'
}

is_pinned_bad() {
  # Whether THIS commit is pinned — matched against every recorded pin, not
  # just the newest. Pins accumulate (do_rollback appends, and a rewrite
  # preserves them), so checking only the last one let a maintainer rewinding
  # origin to an older already-failed commit walk the box straight back onto
  # it.
  [[ -n "${1:-}" ]] || return 1
  grep -qxF "bad=$1" "$STATE" 2>/dev/null
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

# ask <url> [curl args...] -> sets ASK_CODE and ASK_BODY, returns curl's status
#
# Deliberately no -f: curl with -f prints NOTHING on an HTTP error and just
# returns 22. The body of a 500 is the entire diagnostic here — it carries
# `{"error":{"message":"model ... failed to load"}}`, the exact text this
# script exists to surface. With -f the operator gets a rollback and an empty
# reason in the journal.
#
# Globals, not a printed "code\nbody": re-printing in that order loses the
# separator when the body is empty. Command substitution strips the trailing
# newline, "500\n" becomes "500", the caller's ${resp#*\n} finds nothing to
# cut, and the BODY silently became the status code — so a 500 with no body
# logged "FAILED (HTTP 500) — 500", discarding the diagnostic this -f-free
# design exists to preserve. curl puts the code last, which parses correctly
# for an empty body and a multi-line one alike.
ASK_CODE=""
ASK_BODY=""
ASK_ERR=""
CURL_CONF=""
CURL_ERR=""


# The Authorization header goes in a 0600 config file, not in argv.
# `curl -H "Authorization: Bearer $KEY"` is readable by every local user with
# `ps` for as long as the request runs — up to CHAT_TIMEOUT per model, once
# per alias, every week. This box has several human accounts on it.
curl_conf() {
  # -s, not -f: mktemp creates the file first, so a chmod or write that fails
  # afterwards used to leave CURL_CONF naming an existing but EMPTY file. The
  # cache check then short-circuited on every later call and curl sent
  # unauthenticated requests — which come back 401, read as "cannot judge",
  # and send the operator chasing a model failure that was really a full disk.
  [[ -n "$CURL_CONF" && -s "$CURL_CONF" ]] && return 0
  if [[ -n "${RUN_TMP:-}" ]]; then
    CURL_CONF="$RUN_TMP/curl.conf"
    : > "$CURL_CONF" || { CURL_CONF=""; return 1; }
  else
    CURL_CONF="$(mktemp "${TMPDIR:-/tmp}/bonsai-curlcfg.XXXXXX")" || { CURL_CONF=""; return 1; }
  fi
  # Quoted value, with the key escaped for it.
  #
  # curl un-escapes \\ and \" inside a double-quoted config value and ends the
  # value at the first UNescaped " — so a key containing either character
  # produced a truncated Authorization header, every request came back 401,
  # and smoke reported "cannot judge" forever while blaming the credentials
  # rather than the encoding. Measured against a listener: key ab"cd\ef went
  # out as `Authorization: Bearer ab`.
  #
  # The obvious remedy — dropping the quotes — is WRONG and was briefly
  # shipped here: with `header = Authorization: Bearer KEY` curl sends no
  # Authorization header at all. Measured against the live API: unquoted 401,
  # quoted 200, and a --trace-ascii shows zero authorization lines for the
  # unquoted form. So keep the quotes and escape the value. Backslashes first,
  # or the escapes introduced for quotes would be escaped again.
  local esc="${BONSAI_API_KEY//\\/\\\\}"
  esc="${esc//\"/\\\"}"
  if ! chmod 600 "$CURL_CONF" \
     || ! printf 'header = "Authorization: Bearer %s"\n' "$esc" > "$CURL_CONF"; then
    rm -f -- "$CURL_CONF"
    CURL_CONF=""
    return 1
  fi
}

ask() {
  local url="$1"; shift
  local out rc
  curl_conf || { warn "could not write the curl config"; return 1; }
  # Keep curl's own message. -S exists precisely so curl explains the failure
  # despite -s, and sending stderr to /dev/null threw that away — every
  # transport failure logged a bare "curl failed (exit 7)" and nothing about
  # why. One reused file rather than one per request.
  if [[ -z "$CURL_ERR" || ! -f "$CURL_ERR" ]]; then
    if [[ -n "${RUN_TMP:-}" ]]; then
      CURL_ERR="$RUN_TMP/curl.err"
      : > "$CURL_ERR" || CURL_ERR=""
    else
      CURL_ERR="$(mktemp "${TMPDIR:-/tmp}/bonsai-curlerr.XXXXXX")" || CURL_ERR=""
    fi
  fi
  if [[ -n "$CURL_ERR" ]]; then
    out="$(curl -sS -w $'\n%{http_code}' --config "$CURL_CONF" \
          "$@" "$url" 2>"$CURL_ERR")"; rc=$?
    ASK_ERR="$(tr '\n' ' ' < "$CURL_ERR" | head -c 300)"
  else
    out="$(curl -sS -w $'\n%{http_code}' --config "$CURL_CONF" \
          "$@" "$url" 2>/dev/null)"; rc=$?
    ASK_ERR=""
  fi
  ASK_CODE="${out##*$'\n'}"
  ASK_BODY="${out%$'\n'*}"
  return "$rc"
}

smoke_body() {
  # Aliases come from /v1/models, and per models.ini.in they are ini section
  # names — free-form operator-editable text. Interpolating one straight into
  # a JSON string meant an alias containing " or \ produced malformed JSON,
  # the server answered 400, and smoke_ok scored that as a dead model: roll
  # back AND pin the commit. Exactly the fake outage the 401/403 and
  # transient_reply arms exist to prevent.
  python3 -c 'import json, sys
print(json.dumps({"model": sys.argv[1],
                  "messages": [{"role": "user", "content": "say OK"}],
                  "max_tokens": 8}))' "$1"
}

transient_reply() {
  # Momentary router answers that say nothing about the commit.
  #
  # start-server.sh runs with --models-max ${BONSAI_MODELS_MAX:-1}, so the
  # router holds ONE child at a time and a request for a second alias while
  # the first is busy is refused. Observed on the box, firing two aliases at
  # once:
  #   HTTP 500 {"error":{"code":500,"message":"model limit reached, try again later"}}
  # Note it is 500, not 503 — a status-code allowlist alone does not catch it.
  # llama.cpp also answers 503 while a model is still loading.
  #
  # Scoring either as a dead model rolls the stack back AND pins the commit,
  # which blocks every future update until something newer lands upstream. A
  # person chatting while the weekly timer runs is enough to trigger it.
  case "$1" in
    *"model limit reached"*|*"try again later"*|*"Loading model"*|*unavailable_error*)
      return 0;;
  esac
  return 1
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
  local ids id rc failed=0 unknown=0

  ask "$API/v1/models" -m 15 || {
    warn "could not reach $API/v1/models: ${ASK_ERR:-curl failed}"; return 2; }
  if [[ "$ASK_CODE" != "200" ]]; then
    warn "/v1/models returned HTTP $ASK_CODE — cannot judge the stack: ${ASK_BODY:0:200}"
    return 2
  fi
  ids="$(printf '%s' "$ASK_BODY" | parse_model_ids)"
  [[ -n "$ids" ]] || { warn "/v1/models listed no models"; return 2; }

  # A whole-sweep deadline, so the systemd budget can be believed. Without it
  # the retry loop multiplies the worst case by three per alias, and a unit
  # budgeted for one attempt each gets SIGTERM mid-rollback.
  local sweep_deadline=$(( SECONDS + SMOKE_SWEEP_MAX ))
  local attempt busy this_unknown
  while read -r id; do
    [[ -n "$id" ]] || continue
    if (( SECONDS > sweep_deadline )); then
      warn "  smoke: sweep exceeded ${SMOKE_SWEEP_MAX}s; not asking $id"
      unknown=1
      continue
    fi
    log "  smoke: $id"
    busy=0
    this_unknown=0
    for attempt in 1 2 3; do
      busy=0
      # Checked per ATTEMPT, not just per alias: the retry loop could
      # otherwise carry one alias up to 3 x CHAT_TIMEOUT past the ceiling,
      # and the systemd budget is written against the ceiling plus a single
      # request in flight.
      if (( attempt > 1 && SECONDS > sweep_deadline )); then
        warn "  smoke: $id — sweep deadline passed; not retrying"
        this_unknown=1
        break
      fi
      ask "$API/v1/chat/completions" -m "$CHAT_TIMEOUT" \
        -H "Content-Type: application/json" -d "$(smoke_body "$id")"; rc=$?
      if (( rc != 0 )); then
        # /health answering does not mean a model is loaded: in router mode
        # the first request carries the whole cold load. A timeout is
        # ambiguous — a wedged model looks the same as a very slow first read
        # — so it stays "cannot judge" and never rolls back on its own.
        if (( rc == 28 )); then
          warn "  smoke: $id timed out after ${CHAT_TIMEOUT}s — a cold load can exceed this; raise BONSAI_SMOKE_TIMEOUT if the box is slow"
        else
          warn "  smoke: $id — ${ASK_ERR:-curl failed} (exit $rc); cannot judge it"
        fi
        this_unknown=1
        break
      fi
      case "$ASK_CODE" in
        401|403|404)
          # Our credentials or our URL, not the model.
          warn "  smoke: $id returned HTTP $ASK_CODE — cannot judge it"
          this_unknown=1
          break;;
        429|502|503|504) busy=1;;
        2*) ;;   # a real reply; transient wording inside it means nothing
        *) transient_reply "$ASK_BODY" && busy=1;;
      esac
      (( busy )) || break
      if (( attempt < 3 )); then
        warn "  smoke: $id is busy (HTTP $ASK_CODE) — retrying in $(( attempt * RETRY_SLEEP ))s"
        sleep $(( attempt * RETRY_SLEEP ))
      fi
    done
    # Per-alias, not sweep-level: one alias we could not ask must not stop
    # the rest from being judged.
    if (( this_unknown )); then
      unknown=1
      continue
    fi
    if (( busy )); then
      # Still busy after three tries. Someone is using the box, or a model is
      # still loading; either way this alias cannot be judged, and must not
      # be allowed to pin a commit.
      warn "  smoke: $id still busy after 3 tries — cannot judge it"
      unknown=1
      continue
    fi
    if printf '%s' "$ASK_BODY" | smoke_ok; then
      log "  smoke: $id OK"
    else
      warn "  smoke: $id FAILED (HTTP $ASK_CODE) — ${ASK_BODY:-<empty body>}"
      failed=1
    fi
  done <<< "$ids"

  # A model proven dead outranks one that could not be asked. Returning 2
  # here because a LATER alias hit a rotated key would throw away the
  # evidence, and cmd_update would leave the outage in production.
  (( failed )) && return 1
  (( unknown )) && return 2
  return 0
}

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
#
# The test is "am I a copy", not "is the variable set". Keying on presence
# meant any leaked value skipped the copy entirely, and `git pull` and `git
# reset --hard` then rewrote the very file bash was reading by byte offset.
if [[ -z "${BONSAI_UPDATE_LIB:-}" ]] && ! looks_like_temp_copy; then
  # Backstop against an exec loop if TMPDIR ever lands where the name pattern
  # cannot match. Two attempts is one more than a healthy run needs.
  (( ${BONSAI_UPDATE_DEPTH:-0} >= 2 )) \
    && die "re-exec loop: not running from a temp copy after ${BONSAI_UPDATE_DEPTH} attempts"
  copy="$(mktemp "${TMPDIR:-/tmp}/bonsai-update.XXXXXX")" || die "mktemp failed"
  # Only reached if a step below fails: a successful exec replaces this shell
  # and the child cleans up after itself.
  trap 'rm -f -- "$copy"' EXIT
  cat "$SELF" > "$copy" || die "could not copy myself to $copy"
  chmod +x "$copy"
  # Carries the copy's PATH, not a flag — see looks_like_temp_copy above.
  export BONSAI_UPDATE_REEXEC="$copy"
  export BONSAI_UPDATE_ROOT="$ROOT"
  export BONSAI_UPDATE_DEPTH=$(( ${BONSAI_UPDATE_DEPTH:-0} + 1 ))
  # execfail keeps the shell alive if the exec cannot happen, so the fallback
  # below is reachable. /tmp mounted noexec is ordinary hardening, and bash
  # can still READ a script it is not allowed to execute.
  shopt -s execfail
  exec "$copy" "$@"
  exec bash "$copy" "$@"
  # Never fall through to the code below from here: it would run on from the
  # very file git is about to rewrite, which is what the copy exists to avoid.
  die "could not run the copy at $copy (is ${TMPDIR:-/tmp} noexec, and is bash missing?)"
fi

# Delete-on-exit applies to a temp copy and NOTHING else — looks_like_temp_copy
# above explains why the name pattern, not just the variable, is what makes
# that safe.
#
# Two traps, deliberately. bash does not run an EXIT trap on an untrapped
# signal, so the copy would leak on systemd's SIGTERM at TimeoutStartSec —
# but bash also RESUMES after a non-EXIT handler returns, so a handler that
# only cleans up would let the run carry on pulling and building until
# SIGKILL arrives. The signal handler has to exit.
if [[ -z "${BONSAI_UPDATE_LIB:-}" ]]; then
  if looks_like_temp_copy; then
    trap 'cleanup; rm -f -- "$SELF"' EXIT
    trap 'warn "interrupted — stopping"; cleanup; rm -f -- "$SELF"; exit 143' INT TERM
  else
    trap 'cleanup' EXIT
    trap 'warn "interrupted — stopping"; cleanup; exit 143' INT TERM
  fi
fi

# Created HERE, after the re-exec above, and not before it: `exec` replaces
# the process without running its EXIT trap, so a directory made by the parent
# leaked every single run. Still early enough to be known before anything
# calls ask, which is the point — a path assigned lazily inside a subshell is
# invisible to the trap in this shell, and the file holds the API key.
# 0700 from mktemp -d is what makes the fixed filenames inside it safe.
RUN_TMP="$(mktemp -d "${TMPDIR:-/tmp}/bonsai-run.XXXXXX" 2>/dev/null)" || RUN_TMP=""

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

# Per-model smoke request. Generous: /health answering says nothing about
# whether a model is loaded, and in router mode the first request pays the
# whole cold load. Override on a slow box with BONSAI_SMOKE_TIMEOUT.
#
# Set HERE, after .env is sourced. It used to sit with the other constants
# 200 lines above, so a BONSAI_SMOKE_TIMEOUT in .env — which .env.example
# calls "the one source of truth for the whole stack", and which the timeout
# warning tells the operator to raise — was read before .env existed and
# silently ignored. LLAMA_BIND worked only because it is read below this.
CHAT_TIMEOUT="${BONSAI_SMOKE_TIMEOUT:-900}"

# Backoff between retries of a busy model, multiplied by the attempt number.
# Set to 0 to retry immediately (the test suite does this).
RETRY_SLEEP="${BONSAI_SMOKE_RETRY_SLEEP:-15}"

# How long to wait for /health after a restart. Tunable for the same reason
# BONSAI_SMOKE_TIMEOUT is: blowing this deadline rolls the stack back AND pins
# the commit, so a slow disk plus nvidia-runtime init on a cold recreate must
# not be able to blacklist good code.
HEALTH_TIMEOUT="${BONSAI_HEALTH_TIMEOUT:-180}"

# Ceiling on one whole smoke sweep. The per-alias retry loop would otherwise
# multiply the worst case by three, and the systemd unit's time budget is
# written against a sweep, not against an attempt.
# 4x, not 3x: with three aliases at the default per-model timeout, a 3x
# ceiling leaves nothing for the /v1/models call or any backoff, so the last
# alias is skipped and a healthy box degrades to "cannot judge".
SMOKE_SWEEP_MAX="${BONSAI_SMOKE_SWEEP_MAX:-$(( CHAT_TIMEOUT * 4 ))}"

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

code_already_current() {
  # True when HEAD already CONTAINS $1 — equal to it, or ahead of it with
  # local commits on top.
  #
  # Comparing shas said "not current" whenever the checkout carried a local
  # commit, so the run skipped its nothing-to-do exit, ran `git merge
  # --ff-only` against an ancestor (git prints "Already up to date", HEAD does
  # not move), then rebuilt, pulled, force-restarted and swept twice — every
  # Monday, to change nothing — while logging "moving to <upstream-sha>" and
  # arming the pin on a run that moved no code.
  [[ -n "${1:-}" ]] || return 1
  git -C "$ROOT" merge-base --is-ancestor "$1" HEAD 2>/dev/null
}

running_working_dir() {
  # Found by IMAGE, not by container name. COMPOSE_PROJECT_NAME in .env
  # overrides the compose file's `name: bonsai` and is passed through by
  # --env-file, which renames the container — a name-based lookup then finds
  # nothing, reads as "nothing is running", and lets the update restart into
  # the stale bind mount this whole script exists to prevent.
  local dir
  dir="$(docker ps --filter "ancestor=$LLAMA_IMAGE" \
        --format '{{.Label "com.docker.compose.project.working_dir"}}' 2>/dev/null \
        | grep -v '^$' | sort -u | head -1)"
  # Fall back to the conventional name for a stack started before a retag —
  # but only while it is RUNNING. docker inspect resolves stopped containers
  # too, and a stopped one left behind by the move this script exists to
  # detect would make preflight refuse both update and rollback, pointing the
  # operator at a directory that no longer exists.
  [[ -n "$dir" ]] || dir="$(docker inspect "$LLAMA_CONTAINER" \
    --format '{{if .State.Running}}{{index .Config.Labels "com.docker.compose.project.working_dir"}}{{end}}' \
    2>/dev/null)"
  printf '%s' "$dir"
}

take_lock() {
  # One writer at a time. The weekly timer and a human running --rollback can
  # otherwise interleave `git reset --hard`, `docker tag` and the state
  # rewrite on the same checkout, and the loser silently corrupts the winner.
  # Held on fd 9 for the life of the process; the kernel drops it on exit, so
  # a killed run leaves nothing to clean up.
  command -v flock >/dev/null 2>&1 || {
    warn "flock not found; running without a lock"; return 0; }
  # Derived from $STATE so the lock always lands beside the state it guards.
  local dir lock
  dir="$(dirname "$STATE")"
  lock="$dir/update.lock"
  mkdir -p "$dir" || {
    warn "could not create $dir; running without a lock"; return 0; }
  exec 9>"$lock" || {
    warn "could not open the lock file; running without a lock"; return 0; }
  flock -n 9 || die "another update or rollback is already running
(lock: $lock)"
}

# preflight <mode>   update | rollback | check
preflight() {
  local mode="${1:-update}"
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
  # --check only reads: git fetch, rev-list and a registry HEAD. It never
  # calls compose(), never touches $API and never needs a key, so none of the
  # gates below apply to it — including the file checks. A fresh clone with no
  # .env yet should still be able to ask what it is behind.
  [[ "$mode" == "check" ]] && return 0

  [[ -f "$COMPOSE_FILE" ]] || die "no compose file at $COMPOSE_FILE"
  [[ -f "$ENV_FILE" ]]     || die "no .env at $ENV_FILE"
  [[ -n "${BONSAI_API_KEY:-}" ]] || die "BONSAI_API_KEY is not set in $ENV_FILE"

  # A rollback restores code and images, not weights. Refusing to run it
  # because the weights are missing takes away the one tool that helps when
  # the stack is already down.
  [[ "$mode" == "rollback" ]] || models_present "$ROOT/models" || die \
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
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
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

  # Never trade a PROVEN rollback point for an unproven one.
  #
  # A run whose smoke returned "could not ask" leaves the update applied and
  # deliberately does not roll back, so HEAD can sit on a broken commit while
  # the state still names the last good one. Recording unconditionally on the
  # next run would overwrite that good sha and retag :rollback to the broken
  # image — and --rollback would then faithfully restore the breakage. The
  # one way back would be gone.
  if [[ "$verified" != "1" ]] \
     && [[ "$(sed -n 's/^verified=//p' "$STATE" 2>/dev/null)" == "1" ]]; then
    local kept; kept="$(sed -n 's/^sha=//p' "$STATE" 2>/dev/null)"
    log "keeping the proven rollback point ${kept:0:12} — this run could not prove the current state"
    return 0
  fi
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
  # Write beside it and rename. `> "$STATE"` truncates when the redirect is
  # set up, i.e. BEFORE printf can fail — so a full or read-only filesystem
  # destroyed the previous sha, images and pins, and only then announced it
  # was "refusing to update with no way back", by which point there was none.
  local new="$STATE.new"
  # An explicit flag, because every concise spelling of this is wrong. A
  # compound command exits with the status of its LAST command, so:
  #   { printf; [[ -n "$pins" ]] && printf; }   returns 1 when pins is empty,
  #     which aborted every update on every healthy box;
  #   { printf; if [[ -n "$pins" ]]; then printf; fi; }   returns 0 from the
  #     false if, masking a printf that failed on ENOSPC/EDQUOT/EIO;
  #   ( set -e; printf; [[ -z "$pins" ]] || printf ) > f || die
  #     ALSO masks it: bash disables errexit inside a subshell that is the
  #     left operand of ||, so the trailing true [[ ]] becomes the status.
  #     Verified on Linux against /dev/full — this one looked right and was
  #     not, which is why the test below writes to a device that fails.
  # A group command is not a subshell, so the flag survives into this scope.
  local ok=1
  {
    printf 'sha=%s\nwebui=%s\nllama=%s\nverified=%s\n' \
      "$sha" "$webui" "$llama" "$verified" || ok=0
    if [[ -n "$pins" ]]; then
      printf '%s\n' "$pins" || ok=0
    fi
  } > "$new" || ok=0
  (( ok )) || die "cannot write $new — refusing to update with no way back"
  mv -f "$new" "$STATE" \
    || die "cannot replace $STATE — refusing to update with no way back"
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
stash_local_edits() {
  # A rewind is about to discard whatever is in the tree. `git merge --ff-only`
  # refuses only when local edits COLLIDE with the merged files, so edits to
  # untouched files reach this point alive — and a hard reset would erase
  # them without a word. Park them instead; `git stash list` gets them back.
  if git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet \
     && [[ -z "$(git -C "$ROOT" ls-files --others --exclude-standard)" ]]; then
    return 0
  fi
  warn "uncommitted changes in the checkout — stashing them before the rewind
(recover with: git -C $ROOT stash list)"
  git -C "$ROOT" stash push -u -m "update.sh rollback" >/dev/null 2>&1 \
    || warn "could not stash; the rewind may discard local edits"
  return 0
}

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
  local cur
  cur="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"
  if [[ -z "$sha" ]]; then
    warn "no commit recorded; leaving the checkout where it is"
    restored=0
  elif [[ "$sha" == "$cur" ]]; then
    # Nothing to rewind. Without this, an image-only run whose new open-webui
    # image failed would `git reset --hard` onto the commit already checked
    # out — undoing nothing and silently deleting the operator's uncommitted
    # work for a failure the code never caused. The build path was guarded
    # for exactly this; the guard had not been carried here.
    log "checkout is already at ${sha:0:12}; leaving it alone"
  else
    stash_local_edits
    if ! git -C "$ROOT" reset --hard "$sha"; then
      # Silence here is how a rollback reports success with the broken code
      # still checked out.
      warn "git reset to $sha FAILED — the broken commit is STILL checked out"
      restored=0
    fi
  fi

  local have
  have="$(docker image inspect "$ROLLBACK_IMAGE" --format '{{.Id}}' 2>/dev/null)"
  if [[ -z "$have" ]]; then
    warn "no $ROLLBACK_IMAGE to restore; keeping the current llama image"
    restored=0
  elif [[ -z "$llama" ]]; then
    # Nothing was recorded, so :rollback is whatever an earlier run left
    # there. Restoring it would pair this commit's source with that run's
    # binary, which is the very thing the recorded id exists to prevent.
    warn "no llama image id recorded — cannot prove $ROLLBACK_IMAGE belongs to ${sha:0:12}; not restoring it"
    restored=0
  elif [[ "$have" != "$llama" ]]; then
    # A stale tag would pair this commit's source with another commit's binary.
    warn "$ROLLBACK_IMAGE is ${have:0:19}, not the recorded ${llama:0:19} — not restoring it"
    restored=0
  else
    docker tag "$ROLLBACK_IMAGE" "$LLAMA_IMAGE" \
      || { warn "could not restore $LLAMA_IMAGE"; restored=0; }
  fi
  if [[ -z "$webui" ]]; then
    # Nothing was recorded because there WAS nothing to record — true on a box
    # that has not pulled open-webui yet. That is not a failed restore, and
    # treating it as one made every later rollback report "NOT fully restored"
    # and exit 1 even when the commit and the llama image both came back.
    log "no open-webui image was recorded; leaving the current one in place"
  elif ! docker image inspect "$webui" >/dev/null 2>&1; then
    # `docker system prune -a` on a timer is common and takes the old layers.
    warn "recorded open-webui image ${webui:0:19} is gone (pruned?); leaving the current one"
    restored=0
  elif ! docker tag "$webui" "$WEBUI_IMAGE"; then
    warn "could not restore $WEBUI_IMAGE"
    restored=0
  fi

  # Appended after the reset so the pin survives even when the reset failed.
  if [[ "$pin" == "pin" && -n "$bad" && "$bad" != "$sha" ]]; then
    printf 'bad=%s\n' "$bad" >> "$STATE" \
      || warn "could not pin the failed commit; it may be retried next week"
  fi

  compose up -d --force-recreate \
    || { warn "compose failed during rollback"; restored=0; }

  local verdict=1
  if wait_for_health; then smoke; verdict=$?; fi

  if (( ! restored )); then
    # Answering is not the same as being restored: a pruned image or a failed
    # reset can leave the bad code in place and still serve.
    warn "the previous state was NOT fully restored — see the warnings above"
    return 1
  fi
  case "$verdict" in
    0) log "rollback verified — stack is serving again"; return 0;;
    2) # smoke's "could not ask" — a rotated key, a network fault. Every
       # restore step succeeded, so the rollback itself did work; calling it a
       # failure would send someone chasing a stack that is fine.
       warn "rolled back, but could not verify it serves (see above) — check by hand"
       return 0;;
    *) warn "ROLLBACK DID NOT RECOVER THE STACK — needs a human"; return 1;;
  esac
}

cmd_check() {
  preflight check
  local behind rc=0
  git -C "$ROOT" fetch --quiet 2>/dev/null || warn "could not fetch; drift may be stale"
  # A detached HEAD or a branch with no upstream makes @{u} fail. Folding that
  # into 0 reported "code up to date" on a box arbitrarily far behind — the
  # same mistake the image half of this function refuses to make below.
  if ! behind="$(git -C "$ROOT" rev-list --count 'HEAD..@{u}' 2>/dev/null)"; then
    warn "no upstream for HEAD (detached, or no tracking branch) — commit drift unknown"
    rc=1
  elif (( behind > 0 )); then
    log "behind origin by $behind commit(s)"
    rc=1
  else
    log "code up to date"
  fi

  local have want
  if ! docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    # preflight only checked that the docker BINARY exists. With the daemon
    # stopped, or the user not in the docker group, local_image_digest returns
    # empty and the branch below claimed the image was missing — sending the
    # operator to pull something already on disk. The neighbouring branch
    # refuses to report from a comparison that did not happen; so does this.
    warn "cannot reach the docker daemon; image drift unknown"
    return 1
  fi
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
  preflight update
  take_lock

  # A commit that already failed here must not be walked onto again. Without
  # this the rolled-back branch is a plain ancestor of origin, so next week's
  # run fast-forwards right back onto the bad commit, fails, rolls back, and
  # repeats — the box goes down every week with only a journal line to say so.
  local target
  git -C "$ROOT" fetch --quiet 2>/dev/null || warn "fetch failed; using what is already here"
  target="$(git -C "$ROOT" rev-parse '@{u}' 2>/dev/null)" \
    || die "no upstream for HEAD (detached, or no tracking branch) — nothing to update to"
  [[ -n "$target" ]] || die "could not resolve the upstream commit"
  # Prove the stack BEFORE anything else, and before deciding whether there
  # is work to do.
  #
  # --rollback promises the last version that WORKED, so a point recorded on a
  # dark stack would make that a lie. This is also the only weekly check a box
  # that never drifts ever gets: returning early without it meant a driver
  # upgrade or an OOM could leave every model dead while the timer logged
  # "nothing to do" and exited 0, week after week — and run/update-state was
  # never written, so --rollback had nothing to restore either.
  local head_sha have want verified=0 pre_verdict
  head_sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"
  log "checking the current stack before changing anything"
  if wait_for_health; then smoke; pre_verdict=$?; else pre_verdict=1; fi
  case "$pre_verdict" in
    0) verified=1;;
    2) # Could not ASK — a busy router, a rotated key. Not proof of anything,
       # and in particular NOT proof the stack was broken.
       warn "could not judge the stack before this update — recording the
rollback point as unverified";;
    *) warn "the stack is NOT serving before this update — recording the
rollback point as unverified";;
  esac
  record_rollback_point "$verified"

  # The pin check sits HERE, after the proof above, not before it. Placed
  # earlier it meant that once a commit was pinned the box was never
  # smoke-tested again and run/update-state was never refreshed — possibly
  # for months, until a newer commit happened to land upstream.
  if is_pinned_bad "$target"; then
    die "upstream is still at ${target:0:12}, which already failed here and was
rolled back. Nothing to update until a newer commit lands.

To try it again anyway:
  sed -i '/^bad=/d' $STATE"
  fi

  # Nothing to do? Stop before the EXPENSIVE part — the rebuild, the restart
  # and the second smoke sweep. The check above has already run, so a dark
  # stack is still reported and the rollback point still exists.
  #
  # This matters because a failed fetch only warns: @{u} then resolves to the
  # stale ref and the merge is a no-op, but without this the run would still
  # rebuild, restart and sweep again. With --models-max 1 and three aliases
  # that is six cold 27B loads, every Monday, to prove nothing changed.
  if code_already_current "$target"; then
    have="$(local_image_digest "$WEBUI_IMAGE")"
    want="$(remote_image_digest "$WEBUI_IMAGE")"
    # An unreadable registry digest is NOT a reason to rebuild and restart.
    # cmd_check refuses to guess in this situation and so must this: the code
    # is provably current, and guessing "behind" tears the stack down and
    # evicts the resident model for a rebuild that changes nothing — usually
    # during the very outage (DNS, GHCR, no network) that hid the digest.
    if [[ -n "$have" && -z "$want" ]]; then
      warn "could not read the registry digest; code is current, so assuming nothing to do"
      want="$have"
    fi
    if [[ -n "$have" && -n "$want" && "$have" == "$want" ]]; then
      if (( verified )); then
        log "already up to date (code ${head_sha:0:12} contains ${target:0:12}, images current) — nothing to do"
        return 0
      fi
      warn "already up to date (code ${head_sha:0:12}, images current), but the
stack did NOT pass its check above — there is nothing to update that would fix
it. See the warnings above."
      return 1
    fi
  fi

  # merge, not pull: `git pull` runs its OWN fetch, which can land on a commit
  # newer than the $target the pin was just checked against — including the
  # pinned bad one, which is the single thing the pin exists to prevent.
  # Only a run that actually MOVED the checkout may pin. record_rollback_point
  # deliberately keeps an older proven commit A when a run could not prove
  # itself, so HEAD can already be B when this one starts. If the work below
  # then fails because of a new open-webui image rather than the code, pinning
  # would blacklist B — a commit that was already running and that this update
  # never introduced — and every later run would refuse it. The docs promise
  # only the code gets blamed; this keeps that true.
  local pin_arg=""
  if ! code_already_current "$target"; then
    pin_arg="pin"
    if (( pre_verdict == 1 )); then
      # Only POSITIVE proof that the stack was already broken disarms the pin.
      # A driver upgrade, an OOM, or weights that moved would otherwise
      # blacklist an innocent commit, and is_pinned_bad would refuse every
      # later run while the real outage went untouched.
      #
      # "Could not judge" (2) must NOT land here. Folding it in meant the
      # timer firing at 04:00 while someone was chatting — the router answers
      # "model limit reached" to every alias with --models-max 1 — disarmed
      # the pin, so a genuinely crash-looping commit was rolled back unpinned
      # and the box walked onto it again the next week, and the next. That is
      # the exact loop the pin exists to break.
      warn "the stack was already failing before this update, so this run will
not blame the new commit if it fails"
      pin_arg=""
    fi
    log "moving to ${target:0:12}"
    if ! git -C "$ROOT" merge --ff-only "$target"; then
      die "git merge --ff-only $target failed — the checkout has diverged, fix it by hand"
    fi
  else
    log "code already at ${head_sha:0:12}; updating images only"
  fi

  log "building llama image"
  if ! compose build llama; then
    # Undo only what this run did. Nothing has been replaced — :cuda still
    # points at the running image and the containers were never touched — so
    # do_rollback's `compose up -d --force-recreate` would evict a resident
    # 27B model for minutes over a transient build error (a dropped network
    # while the Dockerfile fetches CUDA packages is the usual one). It would
    # also reset to the recorded sha, which after a "kept the proven point"
    # run can be older than where this run started, rewinding past commits
    # the update never introduced.
    if [[ -n "$pin_arg" ]]; then
      warn "build failed — undoing the code move; the running stack was not touched"
      # do_rollback stashes before its reset for exactly this reason; the
      # merge path had been left out. `git merge --ff-only` refuses only on
      # COLLIDING paths, so an uncommitted edit to a file the commit does not
      # touch reaches here alive and would be erased without a word.
      stash_local_edits
      # Guarded by pin_arg, which is set only when the merge above actually
      # ran. Unconditional, this reset fired on image-only runs too — where
      # it reset to the commit already checked out and silently deleted any
      # uncommitted work in the tree. `git merge --ff-only` refuses only when
      # local edits collide with the merged files, so edits to untouched
      # files survive the merge and were then destroyed here. That is the
      # opposite of what docs/updating.md promises.
      git -C "$ROOT" reset --hard "$head_sha" >/dev/null 2>&1 \
        || warn "could not undo the merge; the checkout is left on ${target:0:12}"
    else
      warn "build failed — nothing to undo: this run moved no code and did not
touch the running stack"
    fi
    exit 1
  fi

  log "pulling open-webui"
  compose pull open-webui || warn "pull failed; keeping the current image"

  log "restarting stack"
  compose up -d || { warn "compose up failed"; do_rollback; exit 1; }

  log "waiting for health"
  # This one DOES pin. Build and compose up both succeeded, so the container
  # started; llama-server binds /health before it loads any model, so failing
  # to answer for 180s after a clean start is the code, not a transient bind
  # race (that would have failed compose up). Without a pin, a commit that
  # crash-loops the container is re-applied every week, forever.
  wait_for_health || {
    warn "never became healthy within ${HEALTH_TIMEOUT}s — if this box is slow to
recreate containers, raise BONSAI_HEALTH_TIMEOUT in .env before the next run"
    do_rollback "$pin_arg"; exit 1; }

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
    do_rollback "$pin_arg"
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

# Sourced by the test suite with BONSAI_UPDATE_LIB=1 to drive the functions
# above directly. It sits HERE, below every definition, rather than above
# them: when it sat higher, sourcing returned before preflight,
# record_rollback_point, do_rollback, cmd_check and cmd_update existed, so
# their tests could only grep the source text. A group-command exit-status
# bug that aborted every update passed a green suite that way.
[[ -n "${BONSAI_UPDATE_LIB:-}" ]] && return 0

case "${1:-}" in
  "")         cmd_update;;
  --check)    cmd_check;;
  --rollback) preflight rollback; take_lock; do_rollback;;
  -h|--help)  usage;;
  *)          usage >&2; exit 2;;
esac
