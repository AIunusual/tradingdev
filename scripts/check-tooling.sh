#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/check-fast.sh
exec "${TRADINGDEV_CHECK_PYTHON:-.venv/bin/python}" -m pytest tests/tooling
