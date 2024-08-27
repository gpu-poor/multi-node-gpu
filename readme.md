# How to test if multi-node GPU setup is working using pytorch

> List of the cloud GPU services which offers seamless docker-in-docker
> * https://tensordock.com/
> * https://valdi.ai/
> * https://console.paperspace.com/
> * https://www.genesiscloud.com/

Make sure to use ip which are visible in `ifconfig`

command to launch distributed processes
```torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=192.168.9.215 --master_port=1234  torch-distributed-gpu-test.py```
