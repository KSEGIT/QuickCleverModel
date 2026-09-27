#!/usr/bin/env bash
# Offline identity: configured build pins and the installed binary are distinct.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

configured() {
  local value
  value="$(awk -F= -v key="$1" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$ROOT/runtime-versions.env")"
  printf '%s' "${value:-unknown}"
}

qcm_revision=unknown
if [[ -f "$ROOT/build-revision" ]]; then
  qcm_revision="$(head -n 1 "$ROOT/build-revision")"
elif command -v git >/dev/null && git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  qcm_revision="$(git -C "$ROOT" rev-parse HEAD)"
  if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]]; then
    qcm_revision="$qcm_revision (modified checkout)"
  fi
fi
printf 'QuickCleverModel: %s\n' "$qcm_revision"
if [[ -f "$ROOT/runtime-versions.env" ]]; then
  printf 'Configured Prism repository: %s\n' "$(configured PRISM_REPOSITORY)"
  printf 'Configured Prism SHA: %s\n' "$(configured PRISM_SHA)"
  printf 'Configured Prism commit date: %s\n' "$(configured PRISM_COMMIT_DATE)"
  printf 'Configured upstream llama.cpp baseline: %s\n' "$(configured LLAMA_UPSTREAM_SHA)"
  printf 'Configured CUDA: %s\n' "$(configured CUDA_VERSION)"
  printf 'Configured Docker build base: %s\n' "$(configured CUDA_BUILD_IMAGE)"
  printf 'Configured Docker runtime base: %s\n' "$(configured CUDA_RUNTIME_IMAGE)"
else
  echo 'Configured runtime pins: unavailable (runtime-versions.env missing)'
fi

binary="${LLAMA_SERVER:-}"
if [[ -z "$binary" ]]; then
  for candidate in "$ROOT/src/llama.cpp-prism/build/bin/llama-server" /opt/llama/bin/llama-server; do
    if [[ -x "$candidate" ]]; then binary="$candidate"; break; fi
  done
fi
[[ -n "$binary" ]] || binary="$(command -v llama-server || true)"
if [[ -n "$binary" && -x "$binary" ]]; then
  printf 'Installed llama-server (%s; not a running-service check):\n' "$binary"
  if ! "$binary" --version 2>&1; then echo '  version command failed'; fi
else
  echo 'Installed llama-server: not found (no running-service check performed)'
fi

echo 'Configured model artifacts (presence does not certify loadability or hash):'
if [[ -f "$ROOT/models.lock.tsv" ]]; then
  while IFS=$'\t' read -r group repo revision file bytes sha; do
    [[ -z "$group" || "$group" == \#* ]] && continue
    presence=absent
    [[ ! -f "$ROOT/models/${repo#*/}/$file" ]] || presence=present
    printf '  %s/%s @ %s [%s; %s bytes]\n    SHA256: %s\n' "$repo" "$file" "$revision" "$presence" "$bytes" "$sha"
  done < "$ROOT/models.lock.tsv"
else
  echo '  unavailable (models.lock.tsv missing)'
fi
