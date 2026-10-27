#!/usr/bin/env bash
# Launch Cortex AGI (Linux / macOS).
# Interpreter: 2PY2, else optional python.env, else python3/python.
set -euo pipefail
cd "$(dirname "$0")"

die() { echo "run.sh: $*" >&2; exit 1; }

read_python_env() {
  local file="$1" line val
  [ -f "$file" ] || return 1
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in
      ''|\#*) continue ;;
    esac
    case "$line" in
      PYTHON=*|python=*) val="${line#*=}" ;;
      *) val="$line" ;;
    esac
    val="${val#\"}"
    val="${val%\"}"
    val="${val#\'}"
    val="${val%\'}"
    val="${val#"${val%%[![:space:]]*}"}"
    val="${val%"${val##*[![:space:]]}"}"
    if [ -n "$val" ]; then
      PY="$val"
      return 0
    fi
  done < "$file"
  return 1
}

# 2PY2 starts with a digit — $2PY2 would be positional $2. Use printenv.
PY="$(printenv 2PY2 2>/dev/null || true)"
SRC="2PY2"
if [ -z "$PY" ]; then
  SRC="python.env"
  read_python_env python.env || true
fi
if [ -z "${PY:-}" ]; then
  SRC="default"
  if [ -x .venv/bin/python ]; then
    PY=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  elif command -v python >/dev/null 2>&1; then
    PY="python"
  else
    die "no Python found — set 2PY2 or add python.env"
  fi
fi

echo "python: $PY  ($SRC)"
exec "$PY" run.py "$@"
