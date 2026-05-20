#!/usr/bin/env bash
# Regime Trader — first-time setup script
# Creates a venv, installs deps, prepares .env. Idempotent.

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { printf "${GREEN}[setup]${NC} %s\n" "$1"; }
warn()  { printf "${YELLOW}[warn]${NC}  %s\n" "$1"; }
error() { printf "${RED}[error]${NC} %s\n" "$1" >&2; }

# 1. Check Python 3.10+
info "Checking Python version..."
if ! command -v python3 >/dev/null 2>&1; then
    error "python3 not found. Install Python 3.10 or newer from https://python.org"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required (found $PY_VERSION). Upgrade your Python."
    exit 1
fi
info "Python $PY_VERSION ✓"

# 2. Create venv at .venv if missing
if [ ! -d ".venv" ]; then
    info "Creating virtual environment at .venv ..."
    python3 -m venv .venv
else
    info "Virtual environment already exists at .venv ✓"
fi

# 3. Activate and install requirements
info "Activating venv and installing dependencies..."
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt
info "Dependencies installed ✓"

# 4. Copy .env.example to .env if .env doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        info "Created .env from .env.example ✓"
    else
        warn ".env.example not found — you'll need to create .env manually"
    fi
else
    info ".env already exists ✓ (not overwriting)"
fi

# 5. Print next steps
cat <<EOF

${GREEN}Setup complete.${NC}

Next steps:

  1. Edit .env and add your Alpaca paper trading API keys.
     Get them from: https://alpaca.markets

  2. Activate the virtual environment for any new terminal:
       source .venv/bin/activate

  3. Train the HMM model:
       python main.py train-only

  4. Run a backtest:
       python main.py backtest --symbols SPY --start 2020-01-01 --end 2024-12-31 --compare

  5. (When market is open) Run paper trading dry-run:
       python main.py live --dry-run

EOF
