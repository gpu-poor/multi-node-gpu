import os
import argparse
import torch
import torch.distributed as dist


def setup(backend: str):
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return rank, world, device


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    args = parser.parse_args()

    rank, world, device = setup(args.backend)

    dist.barrier()
    log(rank, "barrier passed")

    # 1) broadcast: rank 0 sends one value to everybody
    x = torch.tensor([111 if rank == 0 else -1], device=device)
    dist.broadcast(x, src=0)
    log(rank, f"broadcast -> {x.tolist()}")

    dist.barrier()

    # 2) all_reduce: everybody contributes, everybody gets the sum
    x = torch.tensor([rank + 1], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    log(rank, f"all_reduce SUM -> {x.tolist()}")

    dist.barrier()

    # 3) reduce: everybody contributes, only rank 0 gets the sum
    x = torch.tensor([rank + 1], device=device)
    dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)
    log(rank, f"reduce to rank 0 -> {x.tolist()}")

    dist.barrier()

    # 4) all_gather: everybody receives one tensor from each rank
    x = torch.tensor([rank], device=device)
    out = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(out, x)
    log(rank, f"all_gather -> {[t.item() for t in out]}")

    dist.barrier()

    # 5) gather: only rank 0 receives from everybody
    x = torch.tensor([rank * 10], device=device)
    if rank == 0:
        gather_list = [torch.zeros_like(x) for _ in range(world)]
    else:
        gather_list = None
    dist.gather(x, gather_list=gather_list, dst=0)
    if rank == 0:
        log(rank, f"gather -> {[t.item() for t in gather_list]}")

    dist.barrier()

    # 6) scatter: rank 0 sends one tensor to each rank
    y = torch.empty(1, device=device, dtype=torch.int64)
    if rank == 0:
        scatter_list = [torch.tensor([100 + r], device=device) for r in range(world)]
    else:
        scatter_list = None
    dist.scatter(y, scatter_list=scatter_list, src=0)
    log(rank, f"scatter <- {y.tolist()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
