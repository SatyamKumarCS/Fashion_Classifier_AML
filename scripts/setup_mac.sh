#!/bin/bash
# Native (non-Docker) setup with uv. Run from the project root: bash scripts/setup_mac.sh
# uv installs Python 3.11 itself if needed and reproduces the exact uv.lock environment in .venv.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh'
  echo 'Then rerun: bash scripts/setup_mac.sh'
  exit 1
fi
uv sync --locked
mkdir -p reports
uv run --locked python scripts/check_environment.py | tee reports/environment-local.txt
uv run --locked python -m ipykernel install --user --name aml-fashion --display-name "Python (AML Fashion)"
echo 'Setup complete. Run commands with uv run, e.g.: uv run python app.py'
echo 'Then: uv run python -m jupyterlab notebooks/exploration.ipynb'
