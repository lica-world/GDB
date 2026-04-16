#!/usr/bin/env bash
# Install the Claude Code CLI so scripts/run_benchmarks.py --provider claude_code works.
#
# Mirrors the installation flow used by the Harbor adapter's claude-code agent,
# so parity runs on both sides start from the same tooling.
set -euo pipefail

if command -v claude >/dev/null 2>&1; then
  echo "claude already installed: $(claude --version 2>/dev/null || true)"
  exit 0
fi

if ! command -v node >/dev/null 2>&1; then
  echo "Installing Node.js via nvm..."
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [ ! -s "$NVM_DIR/nvm.sh" ]; then
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
  fi
  # shellcheck disable=SC1091
  . "$NVM_DIR/nvm.sh"
  nvm install 22
fi

echo "Installing @anthropic-ai/claude-code..."
npm install -g @anthropic-ai/claude-code

claude --version
echo
echo "Next:"
echo "  export ANTHROPIC_API_KEY=..."
echo "  python scripts/run_benchmarks.py --provider claude_code \\"
echo "      --benchmarks svg-1 --dataset-root data/gdb-dataset --n 2"
