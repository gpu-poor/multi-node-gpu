import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

ip = ""
port = ""

def setup(rank, world_size, ip, port):
    os.environ['MASTER_ADDR'] = ip
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=world_size,
        rank=rank
    )

setup(0, 4, ip, port)

tensor = torch.tensor([rank], dtype=torch.float32).cuda(rank)
    
# Perform all_reduce operation
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

print(f"Rank {rank}, Tensor: {tensor.item()}")
    
dist.destroy_process_group()