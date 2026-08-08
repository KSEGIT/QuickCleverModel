#!/usr/bin/env bash
# Download the GGUF weights this stack needs from Hugging Face — both models,
# ~11 GB total. Portable: macOS (native Metal stack) and Linux (Docker stack).
#
# Not wired into `make up` — 11 GB should be an explicit decision, not a
# side effect of starting the stack.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

command -v hf >/dev/null || {
  echo "hf CLI not found. Install it with: pip install -U \"huggingface_hub[cli]\"" >&2
  echo "(on macOS, brew install huggingface-cli also works)" >&2
  exit 1
}

# repo <TAB> file <TAB> expected bytes
FILES=$(cat <<'EOF'
prism-ml/Ternary-Bonsai-27B-gguf	Ternary-Bonsai-27B-Q2_0.gguf	7165121600
prism-ml/Ternary-Bonsai-27B-gguf	Ternary-Bonsai-27B-mmproj-Q8_0.gguf	629246880
prism-ml/Bonsai-27B-gguf	Bonsai-27B-Q1_0.gguf	3803452480
prism-ml/Bonsai-27B-gguf	Bonsai-27B-mmproj-Q8_0.gguf	629246880
EOF
)

# GNU stat (-c%s) on Linux, BSD stat (-f%z) on macOS — try both.
size_of() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0; }

while IFS=$'\t' read -r repo file want; do
  [[ -z "$repo" ]] && continue
  dest_dir="$ROOT/models/${repo#*/}"
  dest="$dest_dir/$file"
  have="$(size_of "$dest")"
  if [[ "$have" == "$want" ]]; then
    printf "  %-32s present (%s bytes)\n" "$file" "$have"
    continue
  fi
  [[ "$have" != 0 ]] && printf "  %-32s size %s != expected %s, re-downloading\n" "$file" "$have" "$want"
  printf "  %-32s downloading...\n" "$file"
  mkdir -p "$dest_dir"
  hf download "$repo" "$file" --local-dir "$dest_dir"
  got="$(size_of "$dest")"
  [[ "$got" == "$want" ]] || { echo "ERROR: $file is $got bytes, expected $want" >&2; exit 1; }
  printf "  %-32s ok (%s bytes)\n" "$file" "$got"
done <<< "$FILES"

echo "weights ready in $ROOT/models"
