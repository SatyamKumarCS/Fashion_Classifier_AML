#!/bin/bash
# Run from the project root: bash scripts/setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v python3.11 >/dev/null 2>&1; then
  echo 'Python 3.11 is required. If Homebrew is installed, run: brew install python@3.11'
  echo 'Then rerun: bash scripts/setup_mac.sh'
  exit 1
fi
if [ ! -d .venv ]; then
  python3.11 -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,11), "Existing .venv must use Python 3.11; preserve it and create a new project copy."'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
mkdir -p reports
.venv/bin/python scripts/check_environment.py | tee reports/environment-local.txt
.venv/bin/python -m pip freeze > requirements-lock-local.txt
.venv/bin/python -m ipykernel install --user --name aml-fashion --display-name "Python (AML Fashion)"
echo 'Setup complete. Run: source .venv/bin/activate'
echo 'Then: python -m jupyterlab notebooks/exploration.ipynb'
