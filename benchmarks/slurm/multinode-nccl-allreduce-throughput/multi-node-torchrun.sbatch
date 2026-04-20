#!/bin/bash
#SBATCH --job-name=torch-bench
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --output=bench_%j.log
#SBATCH --exclusive

# ── These now run on the HOST, so Slurm commands work ──
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500

# ── srun runs on host; --container-image wraps only the torchrun command ──
srun --mpi=pmix \
  --container-image=nvcr.io/nvidia/pytorch:26.03-py3 \
  --container-mounts="/dev/infiniband:/dev/infiniband,${HOME}/torchrun_benchmark.py:/workspace/benchmark.py" \
  --container-workdir=/workspace \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    /workspace/benchmark.py 

