#!/usr/bin/env bash
# Run (or resume) one conventions-eval seed. Safe to re-run after any interruption:
# every finished episode is cached, so it picks up exactly where it stopped.
#
#   ./run_seed.sh 5
#
# Needs a keys file with comma-separated Groq keys (see KEYS_FILE below).
set -u
SEED="${1:?usage: ./run_seed.sh <seed>}"
KEYS_FILE="${SKILLLOOP_KEYS_FILE:-$HOME/.groq_keys}"

if [ ! -f "$KEYS_FILE" ]; then
  echo "No keys file at $KEYS_FILE"
  echo "Create it with your comma-separated Groq keys, or set SKILLLOOP_KEYS_FILE."
  exit 1
fi

export OPENAI_API_KEYS="$(tr -d '[:space:]' < "$KEYS_FILE")"
unset OPENAI_API_KEY
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
export SKILLLOOP_MODEL="${SKILLLOOP_MODEL:-openai/gpt-oss-120b}"
export SKILLLOOP_SPREAD_KEYS=1          # spread load across the key pool
export SKILLLOOP_TIMEOUT="${SKILLLOOP_TIMEOUT:-60}"

LOG="seed${SEED}.log"
echo "seed $SEED -> $LOG   (keys: $(echo "$OPENAI_API_KEYS" | tr ',' '\n' | grep -c .))"
python -u -m qa.conv_eval.resume "$SEED" 2>&1 | tee -a "$LOG"
echo
echo "status:  python -m qa.conv_eval.status $SEED"
echo "resume:  ./run_seed.sh $SEED     (cached episodes are reused)"
