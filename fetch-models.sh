#!/usr/bin/env bash
# Download revision-pinned GGUFs and verify their full SHA256 before installation.
# Default remains Bonsai. A model selector downloads its default quant only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
usage() { echo "Usage: $0 [bonsai|agents|browser|all|--all|--model ID [--quant QUANT]]" >&2; exit 2; }
selection="${1:-bonsai}"
model_id=""
quant=""
case "$selection" in
  bonsai|agents|browser|all|--all) [[ $# -eq 0 || $# -eq 1 ]] || usage ;;
  --model)
    [[ $# -ge 2 && $# -le 4 ]] || usage
    model_id="$2"
    if [[ $# -gt 2 ]]; then
      [[ $# -eq 4 && "$3" == --quant ]] || usage
      quant="$4"
    fi
    [[ "$model_id" =~ ^[a-z0-9][a-z0-9._-]*$ && ( -z "$quant" || "$quant" =~ ^[A-Z0-9_]+$ ) ]] || usage
    ;;
  *) usage ;;
esac
[[ "$selection" != --all ]] || selection=all

selected() {
  local group="$1" id="$2" variant="$3" primary="$4"
  if [[ "$selection" == --model ]]; then
    [[ "$id" == "$model_id" ]] || return 1
    if [[ -n "$quant" ]]; then [[ "$variant" == "$quant" ]]; else [[ "$primary" == yes ]]; fi
  else
    [[ "$selection" == all || "$selection" == "$group" ]]
  fi
}

size_of() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null; }
sha256_of() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
verify() {
  local path="$1" want_size="$2" want_sha="$3"
  [[ "$(size_of "$path")" == "$want_size" ]] || {
    echo "ERROR: size mismatch for $path (expected $want_size bytes)" >&2; return 1;
  }
  [[ "$(sha256_of "$path")" == "$want_sha" ]] || {
    echo "ERROR: SHA256 mismatch for $path (expected $want_sha)" >&2; return 1;
  }
}

# Validate the entire catalogue and selector before downloading any large file.
selected_count=0
selected_bytes=0
while IFS=$'\t' read -r group id variant primary repo revision file bytes sha extra; do
  [[ -z "$group" || "$group" == \#* ]] && continue
  [[ "$group" =~ ^(bonsai|agents|browser)$ && "$id" =~ ^[a-z0-9][a-z0-9._-]*$ \
     && "$variant" =~ ^[A-Z0-9_]+$ && "$primary" =~ ^(yes|no)$ ]] || {
    echo "ERROR: invalid catalogue identity for $id" >&2; exit 1;
  }
  [[ "$revision" =~ ^[a-f0-9]{40}$ ]] || { echo "ERROR: unpinned revision for $repo" >&2; exit 1; }
  [[ "$sha" =~ ^[a-f0-9]{64}$ && "$bytes" =~ ^[1-9][0-9]*$ && -z "$extra" ]] || {
    echo "ERROR: invalid hash/size in models.lock.tsv for $repo" >&2; exit 1;
  }
  [[ "$repo" =~ ^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$ && "$file" =~ ^[A-Za-z0-9_-][A-Za-z0-9_.-]*\.gguf$ ]] || {
    echo "ERROR: invalid artifact path in models.lock.tsv" >&2; exit 1;
  }
  if selected "$group" "$id" "$variant" "$primary"; then
    selected_count=$((selected_count + 1))
    selected_bytes=$((selected_bytes + bytes))
  fi
done < "$ROOT/models.lock.tsv"
[[ "$selected_count" -gt 0 ]] || { echo "ERROR: no catalogue entry for model/quant selection" >&2; exit 1; }
printf 'Selected %d artifact(s), %.2f GiB total before existing-file checks.\n' \
  "$selected_count" "$(awk -v bytes="$selected_bytes" 'BEGIN { print bytes / 1073741824 }')"

# Staging is on the destination filesystem so successful installation is atomic.
# Existing files are never removed or replaced, including mismatched old weights.
stage=""
# Keep failed downloads for hf's range/partial-download resume support.
trap '[[ -z "$stage" ]] || printf "Partial download retained for retry: %s\n" "$stage" >&2' EXIT

while IFS=$'\t' read -r group id variant primary repo revision file bytes sha extra; do
  [[ -z "$group" || "$group" == \#* ]] && continue
  selected "$group" "$id" "$variant" "$primary" || continue
  dest_dir="$ROOT/models/${repo#*/}"
  dest="$dest_dir/$file"
  if [[ -e "$dest" || -L "$dest" ]]; then
    if ! verify "$dest" "$bytes" "$sha"; then
      echo "Existing file preserved. Move it aside explicitly before retrying: $dest" >&2
      exit 1
    fi
    printf '  %s verified (%s bytes)\n' "$file" "$bytes"
    continue
  fi
  command -v hf >/dev/null || {
    echo "hf CLI not found. Install with: pipx install huggingface_hub==1.32.0" >&2
    exit 1
  }
  mkdir -p "$dest_dir"
  stage="$dest_dir/.download-$sha"
  mkdir -p "$stage"
  printf '  %s downloading at %s...\n' "$file" "$revision"
  hf download "$repo" "$file" --revision "$revision" --local-dir "$stage"
  verify "$stage/$file" "$bytes" "$sha"
  # A hard link also prevents a concurrent downloader from replacing a file.
  ln "$stage/$file" "$dest"
  rm -rf -- "$stage"
  stage=""
  printf '  %s verified and installed (%s bytes)\n' "$file" "$bytes"
done < "$ROOT/models.lock.tsv"

echo "Selected weights verified in $ROOT/models"
