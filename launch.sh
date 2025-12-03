#!/usr/bin/env bash
set -e

########################################
# 1. Install deps
########################################

# Create venv
uv venv
source .venv/bin/activate

# Install vLLM (editable + precompiled)
if [ -d "vllm" ]; then
  cd vllm
  VLLM_USE_PRECOMPILED=1 uv pip install --editable . --prerelease=allow
  cd ..
else
  echo "❌ ERROR: vllm directory not found"
  exit 1
fi

# Install backend package (editable)
uv pip install -e .

########################################
# 2. Start tmux sessions
########################################

# FRONTEND (backend FastAPI)
if ! tmux has-session -t backend 2>/dev/null; then
  tmux new-session -d -s backend "cd backend && python main_fastapi.py"
  echo "✅ Started backend (FastAPI) in tmux session: backend"
else
  echo "ℹ️ Tmux session 'backend' already exists"
fi

# DASHBOARD (Next.js frontend)
if ! tmux has-session -t frontend 2>/dev/null; then
  tmux new-session -d -s frontend "cd dashboard && npm install && npm run dev"
  echo "✅ Started frontend (Next.js) in tmux session: frontend"
else
  echo "ℹ️ Tmux session 'frontend' already exists"
fi

########################################
# Done
########################################
echo "🎉 All setup complete!"
echo "👉 Attach to backend:  tmux attach -t backend"
echo "👉 Attach to frontend: tmux attach -t frontend"
