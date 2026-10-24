#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${TRADINGDEV_CHECK_PYTHON:-.venv/bin/python}" -m tooling.documentation
