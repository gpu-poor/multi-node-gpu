import argparse
import torch.distributed as dist


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gloo"], default="gloo")
    args = parser.parse_args()

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world = dist.get_world_size()

    # 1) broadcast_object_list
    objs = [{"from": 0, "msg": "hello"}] if rank == 0 else [None]
    dist.broadcast_object_list(objs, src=0)
    log(rank, f"broadcast_object_list -> {objs}")

    dist.barrier()

    # 2) all_gather_object
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, {"rank": rank, "square": rank * rank})
    log(rank, f"all_gather_object -> {gathered}")

    dist.barrier()

    # 3) gather_object
    if rank == 0:
        gathered_objs = [None for _ in range(world)]
    else:
        gathered_objs = None
    dist.gather_object({"rank": rank, "cube": rank ** 3}, gathered_objs, dst=0)
    if rank == 0:
        log(rank, f"gather_object -> {gathered_objs}")

    dist.barrier()

    # 4) scatter_object_list
    out_obj = [None]
    if rank == 0:
        in_objs = [f"obj_for_rank_{r}" for r in range(world)]
    else:
        in_objs = None
    dist.scatter_object_list(out_obj, in_objs, src=0)
    log(rank, f"scatter_object_list <- {out_obj}")

    dist.barrier()

    # 5) monitored_barrier
    dist.monitored_barrier()
    log(rank, "monitored_barrier passed")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
