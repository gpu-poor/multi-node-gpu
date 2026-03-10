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

    if world % 2 != 0:
        if rank == 0:
            print("This demo expects an even world size.", flush=True)
        dist.destroy_process_group()
        return

    peer = rank + 1 if rank % 2 == 0 else rank - 1

    # 0) sync so everybody starts together
    dist.barrier()

    # 1) blocking send / recv
    send_tensor = torch.tensor([rank], device=device, dtype=torch.int64)
    recv_tensor = torch.empty(1, device=device, dtype=torch.int64)

    if rank % 2 == 0:
        dist.send(send_tensor, dst=peer)
        dist.recv(recv_tensor, src=peer)
    else:
        dist.recv(recv_tensor, src=peer)
        dist.send(send_tensor, dst=peer)

    log(rank, f"send/recv with peer {peer} -> received {recv_tensor.tolist()}")

    dist.barrier()

    # 2) non-blocking isend / irecv
    send_tensor = torch.tensor([rank + 100], device=device, dtype=torch.int64)
    recv_tensor = torch.empty(1, device=device, dtype=torch.int64)

    if rank % 2 == 0:
        req1 = dist.isend(send_tensor, dst=peer)
        req2 = dist.irecv(recv_tensor, src=peer)
    else:
        req2 = dist.irecv(recv_tensor, src=peer)
        req1 = dist.isend(send_tensor, dst=peer)

    req1.wait()
    req2.wait()

    log(rank, f"isend/irecv with peer {peer} -> received {recv_tensor.tolist()}")

    dist.barrier()

    # 3) batch_isend_irecv in a ring: every rank sends to next and receives from prev
    send_to = (rank + 1) % world
    recv_from = (rank - 1 + world) % world

    send_tensor = torch.tensor([rank, rank + 1000], device=device, dtype=torch.int64)
    recv_tensor = torch.empty(2, device=device, dtype=torch.int64)

    ops = [
        dist.P2POp(dist.isend, send_tensor, send_to),
        dist.P2POp(dist.irecv, recv_tensor, recv_from),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    log(rank, f"batch_isend_irecv: recv from {recv_from} -> {recv_tensor.tolist()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
