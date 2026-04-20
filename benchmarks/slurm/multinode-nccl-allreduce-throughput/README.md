# Multi-Node NCCL All-Reduce Throughput Benchmark

Multi-node GPU benchmark that measures NCCL all-reduce bandwidth and synthetic DDP training throughput across a Slurm cluster using Pyxis/Enroot containers.

## Files

- `multi-node-torchrun.sbatch` — Slurm batch script that launches `torchrun` inside an NVIDIA PyTorch container via Pyxis.
- `torchrun_benchmark.py` — PyTorch benchmark that runs NCCL all-reduce at multiple message sizes and a synthetic DDP training loop.

## Requirements

- Slurm with Pyxis/Enroot (`--container-image` support on `srun`)
- PMIx (`--mpi=pmix`)
- InfiniBand fabric mounted at `/dev/infiniband`
- `torchrun_benchmark.py` present at `$HOME/torchrun_benchmark.py` on the submission host (mounted into the container at `/workspace/benchmark.py`)

## Usage

From the submission host:

```bash
cp torchrun_benchmark.py ~/
sbatch multi-node-torchrun.sbatch
```

Logs are written to `bench_<jobid>.log` in the submission directory.

## Default job shape

Set in `multi-node-torchrun.sbatch`:

- 4 nodes × 8 GPUs = 32 ranks
- 1 task per node, 64 CPUs per task, exclusive allocation
- Partition `all`, wall time 1 hour
- Container: `nvcr.io/nvidia/pytorch:26.03-py3`

Adjust `#SBATCH` directives to match your cluster.

## NCCL environment

The sbatch script exports:

- `NCCL_IB_DISABLE=0` — enable InfiniBand
- `NCCL_NET_GDR_LEVEL=PHB` — GPUDirect RDMA across PCIe host bridge
- `MASTER_ADDR` / `MASTER_PORT` — rendezvous endpoint derived from `SLURM_NODELIST`

## Benchmark arguments

`torchrun_benchmark.py` accepts:

| Flag | Default | Description |
|---|---|---|
| `--nccl-iters` | 100 | All-reduce iterations per size |
| `--sizes-mb` | 1 8 64 256 512 1024 2048 4096 | Message sizes in MB |
| `--train-iters` | 50 | Synthetic training iterations |
| `--batch-size` | 64 | Per-GPU batch size |
| `--hidden-dim` | 4096 | Hidden dim of the synthetic MLP |
| `--num-layers` | 8 | Number of transformer-like blocks |

To override, append flags after the script path in the `srun` line of the sbatch file.

## Output

Rank 0 prints three sections:

1. **Cluster info** — PyTorch/CUDA/NCCL versions, world size, GPU model and memory.
2. **NCCL all-reduce** — per-size avg latency (ms), algorithmic bandwidth, and bus bandwidth (GB/s).
3. **Synthetic DDP training** — per-GPU and global samples/sec over the timed iterations.
