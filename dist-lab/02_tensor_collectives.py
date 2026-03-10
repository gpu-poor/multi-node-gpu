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

    # 1) all_gather_into_tensor
    x = torch.tensor([rank, rank + 10], device=device, dtype=torch.int64)
    out = torch.empty(world * x.numel(), device=device, dtype=torch.int64)
    dist.all_gather_into_tensor(out, x)
    log(rank, f"all_gather_into_tensor -> {out.tolist()}")

    dist.barrier()

    # 2) reduce_scatter
    # Every rank gives a list of length=world.
    # Slot i across all ranks is reduced, rank i receives that result.
    input_list = [
        torch.tensor([rank * 100 + i], device=device, dtype=torch.int64)
        for i in range(world)
    ]
    out = torch.empty(1, device=device, dtype=torch.int64)
    dist.reduce_scatter(out, input_list, op=dist.ReduceOp.SUM)
    log(rank, f"reduce_scatter -> {out.tolist()}")

    dist.barrier()

    # 3) reduce_scatter_tensor
    chunk = 2
    x = torch.arange(world * chunk, device=device, dtype=torch.int64) + rank * 1000
    out = torch.empty(chunk, device=device, dtype=torch.int64)
    dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM)
    log(rank, f"reduce_scatter_tensor -> {out.tolist()}")

    dist.barrier()

    # 4) all_to_all_single
    if args.backend == "nccl":
        chunk = 2
        x = torch.arange(world * chunk, device=device, dtype=torch.int64) + rank * 10000
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x)
        log(rank, f"all_to_all_single -> {out.tolist()}")
    else:
        log(rank, "skip all_to_all_single on gloo")

    dist.barrier()

    # 5) all_to_all
    if args.backend == "nccl":
        input_list = [
            torch.tensor([rank, i], device=device, dtype=torch.int64)
            for i in range(world)
        ]
        output_list = [torch.empty(2, device=device, dtype=torch.int64) for _ in range(world)]
        dist.all_to_all(output_list, input_list)
        log(rank, f"all_to_all -> {[t.tolist() for t in output_list]}")
    else:
        log(rank, "skip all_to_all on gloo")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
