#!/usr/bin/env bash
# Fast-forward Cortex AGI source from GitHub. data/ and trained weights stay.
# Interpreter: 2PY2, else optional python.env, else python3/python. (Linux / macOS)
set -euo pipefail
cd "$(dirname "$0")"
export GIT_TERMINAL_PROMPT=0 GIT_OPTIONAL_LOCKS=0
REPO="${AGI_UPDATE_REPO:-codero-sus/agi}"

die() { echo "updater: $*" >&2; exit 1; }

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
      PYTHON="$val"
      return 0
    fi
  done < "$file"
  return 1
}

# 2PY2 starts with a digit — $2PY2 would be positional $2. Use printenv.
PYTHON="$(printenv 2PY2 2>/dev/null || true)"
SRC="2PY2"
if [ -z "$PYTHON" ]; then
  SRC="python.env"
  read_python_env python.env || true
fi
if [ -z "${PYTHON:-}" ]; then
  SRC="default"
  if [ -x .venv/bin/python ]; then
    PYTHON=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    die "no Python found — set 2PY2 or add python.env"
  fi
fi

command -v git >/dev/null 2>&1 || die "git not installed"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not a git checkout"

origin="$(git remote get-url origin 2>/dev/null || true)"
[ -n "$origin" ] || die "no origin remote"
case "$origin" in
  *github.com[:/]"$REPO".git|*github.com[:/]"$REPO"|*github.com[:/]"$REPO"/) ;;
  *) die "origin is not github.com/${REPO} — refusing" ;;
esac

if [ -n "$(git status --porcelain)" ]; then
  die "working tree dirty — commit or stash first (data/ and weights are already ignored)"
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$branch" != "HEAD" ] || die "detached HEAD — checkout a branch"

echo "python: $PYTHON  ($SRC)"
echo "fetching origin/${branch}…"
git fetch --depth 50 origin "$branch"
before="$(git rev-parse --short HEAD)"
if git merge-base --is-ancestor "origin/${branch}" HEAD 2>/dev/null && [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/${branch}")" ]; then
  echo "already current (${before})"
  exit 0
fi
git merge --ff-only "origin/${branch}"
after="$(git rev-parse --short HEAD)"
echo "source ${before} → ${after}"

if [ -f requirements.txt ]; then
  echo "syncing requirements with $PYTHON …"
  "$PYTHON" -m pip install -r requirements.txt -q --disable-pip-version-check \
    || echo "updater: pip failed (source is updated; install deps yourself)" >&2
fi

echo "Cortex AGI updated. Restart: $PYTHON -m agi   or   ./run.sh"
