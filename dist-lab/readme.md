# PyTorch Distributed Lab

A hands-on lab covering PyTorch `torch.distributed` primitives across 4 files.

## Quick Start (Fresh VM)

Run this on **every node** to bootstrap the environment:

```bash
# CPU-only (gloo backend)
bash setup.sh

# With GPU/CUDA support (nccl backend)
bash setup.sh --gpu
```

Then activate the venv:

```bash
source .venv/bin/activate
```

The script handles: system deps, uv installation, Python 3.11 venv, PyTorch, and useful extras (tensorboard, pynvml, psutil, py-spy, rich).

## Setup

- **2 nodes**, **2 processes per node**, **world size = 4**
- Replace `10.0.0.10` with your node0 IP
- Replace `eth0` with your actual network interface (`ip addr` to check)
- Put the same files on both nodes

## Backend Guide

| Backend | Device | Use for |
|---------|--------|---------|
| `gloo`  | CPU    | Files 01, 03, 04 — and most of 02 |
| `nccl`  | GPU    | File 02 when you want `all_to_all` / `all_to_all_single` |

## Run Commands

Run each command **on both nodes**. Only `--node-rank` differs (node0=implicit, node1=implicit — torchrun with rdzv handles it automatically).

### 1) Basic Collectives (gloo/CPU)

Covers: `barrier`, `broadcast`, `all_reduce`, `reduce`, `all_gather`, `gather`, `scatter`

```bash
export GLOO_SOCKET_IFNAME=eth0

torchrun \
  --nnodes=2 \
  --nproc-per-node=2 \
  --rdzv-id=toy-dist-01 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=10.0.0.10:29400 \
  01_basic_collectives.py \
  --backend gloo
```

### 2) Point-to-Point (gloo/CPU)

Covers: `send`, `recv`, `isend`, `irecv`, `batch_isend_irecv`

```bash
export GLOO_SOCKET_IFNAME=eth0

torchrun \
  --nnodes=2 \
  --nproc-per-node=2 \
  --rdzv-id=toy-dist-01 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=10.0.0.10:29400 \
  04_p2p.py \
  --backend gloo
```

### 3) Objects & Debug (gloo only)

Covers: `broadcast_object_list`, `all_gather_object`, `gather_object`, `scatter_object_list`, `monitored_barrier`

```bash
export GLOO_SOCKET_IFNAME=eth0

torchrun \
  --nnodes=2 \
  --nproc-per-node=2 \
  --rdzv-id=toy-dist-01 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=10.0.0.10:29400 \
  03_objects_and_debug.py \
  --backend gloo
```

### 4) Tensor Collectives — gloo (skips all_to_all)

Covers: `all_gather_into_tensor`, `reduce_scatter`, `reduce_scatter_tensor`

```bash
export GLOO_SOCKET_IFNAME=eth0

torchrun \
  --nnodes=2 \
  --nproc-per-node=2 \
  --rdzv-id=toy-dist-01 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=10.0.0.10:29400 \
  02_tensor_collectives.py \
  --backend gloo
```

### 5) Tensor Collectives — nccl (includes all_to_all)

Covers: everything in step 4 **plus** `all_to_all_single`, `all_to_all`

```bash
export NCCL_SOCKET_IFNAME=eth0

torchrun \
  --nnodes=2 \
  --nproc-per-node=2 \
  --rdzv-id=toy-dist-01 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=10.0.0.10:29400 \
  02_tensor_collectives.py \
  --backend nccl
```

## Recommended Order

| Step | File | Backend | Why |
|------|------|---------|-----|
| 1 | `01_basic_collectives.py` | gloo | "Who talks to everyone" — the fundamentals |
| 2 | `04_p2p.py` | gloo | "Who talks to one peer" — send/recv patterns |
| 3 | `03_objects_and_debug.py` | gloo | Object collectives + `monitored_barrier` for debugging |
| 4 | `02_tensor_collectives.py` | gloo | Shape-sensitive tensor collectives (skips all_to_all) |
| 5 | `02_tensor_collectives.py` | nccl | Re-run with GPUs to unlock `all_to_all*` |

## Single-Node Quick Test

To test locally on one machine with no multi-node setup:

```bash
torchrun \
  --nnodes=1 \
  --nproc-per-node=4 \
  --rdzv-id=local-test \
  --rdzv-backend=c10d \
  --rdzv-endpoint=127.0.0.1:29400 \
  01_basic_collectives.py \
  --backend gloo
```

## Notes

- `monitored_barrier()` is Gloo-only — useful for debugging desynchronized ranks
- `all_to_all` / `all_to_all_single` require NCCL (not supported on Gloo)
- Object collectives use pickle under the hood — learning only, not for production perf
- If interface auto-detection fails, set `GLOO_SOCKET_IFNAME` or `NCCL_SOCKET_IFNAME`
- The rendezvous endpoint default port is 29400 if not specified
