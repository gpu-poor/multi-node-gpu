#!/bin/bash

# Configuration
TOTAL_NODES=2
GPUS_PER_NODE=1
CPUS_PER_TASK=96
MASTER_ADDR="147.189.195.2"  # EDIT: IP of the first VM
MASTER_PORT=6000
NODE_RANK=1  # EDIT: Set to 0 for first VM, 1 for second VM

# EDIT: Path to your Python script
PROGRAM_PATH="/home/ubuntu/multi-node-gpu/torch-distributed-gpu-test.py"

echo "START TIME: $(date)"

# Logging
LOG_PATH="main_log_node_${NODE_RANK}.txt"

# Function to run the distributed training
run_distributed_training() {
    python -u -m torch.distributed.run \
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $TOTAL_NODES \
        --node_rank $NODE_RANK \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv_backend c10d \
        --max_restarts 0 \
        --role $(hostname -s|tr -dc '0-9'): \
        --tee 3 \
        $PROGRAM_PATH
}

# Run the training and log output
run_distributed_training 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
