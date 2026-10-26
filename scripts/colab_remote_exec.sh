#!/usr/bin/env bash
set -euo pipefail

target="${1:-}"
session="${2:-${COLAB_SESSION:-}}"

if [[ -z "$target" ]]; then
  echo "usage: bash scripts/colab_remote_exec.sh <file.py|file.ipynb> [session]" >&2
  exit 64
fi
if [[ -z "$session" ]]; then
  echo "COLAB_REMOTE_REQUIRED: no COLAB_SESSION supplied." >&2
  echo "HARD STOP: local execution is forbidden." >&2
  exit 65
fi
case "$target" in
  *.py|*.ipynb) ;;
  *) echo "REFUSED: only .py or .ipynb source may be forwarded to Colab." >&2; exit 66 ;;
esac
if [[ ! -f "$target" ]]; then
  echo "REFUSED: source file not found: $target" >&2
  exit 67
fi
echo "[colab-only] validating existing remote session: $session" >&2
if ! colab status -s "$session"; then
  echo "COLAB_REMOTE_REQUIRED: session '$session' is unavailable." >&2
  echo "HARD STOP: no local fallback is permitted." >&2
  exit 68
fi
echo "[colab-only] REMOTE EXECUTION: $target -> Colab session '$session'" >&2
exec colab exec -s "$session" -f "$target"