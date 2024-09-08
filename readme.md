# How to test if multi-node GPU setup is working using pytorch

> List of the cloud GPU services which offers seamless docker-in-docker
> * https://tensordock.com/
> * https://valdi.ai/
> * https://console.paperspace.com/
> * https://www.genesiscloud.com/

Make sure to use ip which are visible in `ifconfig`

command to launch distributed processes

> ```torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=192.168.9.215 --master_port=1234  torch-distributed-gpu-test.py```

> std.out of rank - 0  process : 
> ```
    outside try
    [gc-nervous-cartwright:0] Reduction op=sum result: 2.0
    [gc-nervous-cartwright:0] is OK (global rank: 0/2)
    pt=2.4.0+cu121, cuda=12.1, nccl=(2, 20, 5)
    device compute capabilities=(8, 6)
    pytorch compute capabilities=['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
    [rank0]:[W827 13:08:51.068884356 ProcessGroupNCCL.cpp:1168] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
    ```

> std.out of rank - 1 process
> ```
    [gc-relaxed-jang:0] Reduction op=sum result: 2.0
    [gc-relaxed-jang:0] is OK (global rank: 1/2)
    [rank1]:[W827 13:02:45.426541970 ProcessGroupNCCL.cpp:1168] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
    ```