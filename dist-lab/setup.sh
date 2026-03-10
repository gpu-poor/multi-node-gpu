#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# dist-lab/setup.sh
# Bootstrap a fresh VM for running the PyTorch distributed lab.
# Assumes: non-root user with sudo, Debian/Ubuntu-based OS.
# Usage:   bash setup.sh [--gpu]
#   --gpu   Install CUDA-enabled PyTorch + NCCL (default: CPU-only)
# ─────────────────────────────────────────────────────────────────────

GPU=false
for arg in "$@"; do
    case "$arg" in
        --gpu) GPU=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# ── 1. System packages ──────────────────────────────────────────────
echo ">>> Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    curl \
    git \
    net-tools \
    iproute2 \
    htop \
    tmux \
    python3-dev

# ── 2. Install uv ───────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | SHELL=/bin/bash sh
    # Make uv available in current shell
    export PATH="$HOME/.local/bin:$PATH"
else
    echo ">>> uv already installed: $(uv --version)"
fi

# ── 3. Create venv ──────────────────────────────────────────────────
echo ">>> Creating virtual environment at $VENV_DIR ..."
uv venv "$VENV_DIR" --python 3.11

# ── 4. Install Python packages ──────────────────────────────────────
echo ">>> Installing Python packages..."

if [ "$GPU" = true ]; then
    echo "    (GPU mode — installing CUDA-enabled PyTorch)"
    uv pip install --python "$VENV_DIR/bin/python" \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
else
    echo "    (CPU mode — installing CPU-only PyTorch)"
    uv pip install --python "$VENV_DIR/bin/python" \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
fi

# Useful extras for distributed work
uv pip install --python "$VENV_DIR/bin/python" \
    tensorboard \
    torch-tb-profiler \
    pynvml \
    psutil \
    py-spy \
    rich

# ── 5. Verify install ───────────────────────────────────────────────
echo ""
echo ">>> Verifying installation..."
"$VENV_DIR/bin/python" -c "
import torch
print(f'  PyTorch       : {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count     : {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    [{i}] {torch.cuda.get_device_name(i)}')
import torch.distributed as dist
print(f'  Gloo available: {dist.is_gloo_available()}')
print(f'  NCCL available: {dist.is_nccl_available()}')
"

# ── 6. Print activation instructions ────────────────────────────────
echo ""
echo "========================================="
echo " Setup complete!"
echo "========================================="
echo ""
echo "Activate the venv:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then run (example single-node, 4 processes):"
echo "  torchrun --nnodes=1 --nproc-per-node=4 \\"
echo "    --rdzv-id=local --rdzv-backend=c10d \\"
echo "    --rdzv-endpoint=127.0.0.1:29400 \\"
echo "    01_basic_collectives.py --backend gloo"
echo ""
if [ "$GPU" = true ]; then
    echo "GPU mode was selected. For NCCL runs, set:"
    echo "  export NCCL_SOCKET_IFNAME=<your-interface>"
else
    echo "CPU mode was selected. Re-run with --gpu for CUDA/NCCL support."
fi
echo ""
