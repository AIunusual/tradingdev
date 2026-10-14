#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec "${TRADINGDEV_CHECK_PYTHON:-.venv/bin/python}" -m tooling.git.check_pr "$@"
