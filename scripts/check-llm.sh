#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 || ( "$1" != codex && "$1" != local ) ]]; then
    echo 'Usage: scripts/check-llm.sh codex|local [pytest options]' >&2
    echo 'Codex example:' >&2
    echo '  scripts/check-llm.sh codex --llm-model gpt-5.6-luna \' >&2
    echo '    --llm-timeout 1800' >&2
    echo 'Local example (verified with Ollama):' >&2
    echo '  scripts/check-llm.sh local --llm-model qwen3.8:27b \' >&2
    echo '    --llm-reasoning-effort none --llm-temperature 0.7 --llm-timeout 900' >&2
    exit 2
fi
provider=$1
shift
exec "${TRADINGDEV_CHECK_PYTHON:-.venv/bin/python}" -u -m pytest \
    tests/e2e/test_llm_workflows.py tests/e2e/test_llm_cleanup.py \
    tests/e2e/test_llm_execution_plans.py tests/e2e/test_llm_ml.py \
    --llm-provider "$provider" -m live_llm \
    -x -vv --capture=tee-sys --tb=short --durations=3 "$@"
