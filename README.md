# gpuone_benchmarking

Scripts for benchmarking and validating GPU computing platforms — aimed at HGX-class
nodes (default target: 8× NVIDIA B300 with 8× ConnectX-8 HCAs, 256 CPUs, Ubuntu) and
SLURM-based multi-node clusters.

## Contents

### `healthchecks/`

- **`gpu-healthcheck.sh`** — single-node diagnostic script. Run as root on a host to
  audit OS/kernel, CPU, memory/NUMA, GPUs (driver, XID errors, retired pages, NVLink,
  fabric manager), InfiniBand/ConnectX HCAs, CUDA/NCCL/cuDNN/GDRCopy, container
  runtime, SLURM, storage, and HPC-relevant sysctls. Human-readable output with
  PASS / WARN / FAIL / INFO tallies; `--json` flag reserved for machine-readable mode.

  ```bash
  sudo bash healthchecks/gpu-healthcheck.sh 2>&1 | tee healthcheck_$(date +%Y%m%d_%H%M%S).log
  ```

### `benchmarks/slurm/`

SLURM + Pyxis/Enroot benchmarks for NVIDIA GPU clusters.

- **`gpu-cluster-benchmarks/`** — ClusterMAX-equivalent benchmark suite covering the
  four main evaluation axes:

  | # | Benchmark | Tool | Measures |
  |---|-----------|------|----------|
  | 01 | GEMM TFLOPS | PyTorch | Per-GPU FP32/FP16/BF16/FP8 compute + HBM bandwidth |
  | 02 | NCCL Bandwidth | nccl-tests | Intra-node (NVLink/NVSwitch) + inter-node (RoCE/IB) collectives |
  | 03 | Training MFU | torchtitan | Multi-node Llama 3 8B pretraining — MFU, step time, tokens/sec |
  | 04 | Inference | vLLM | Offline throughput (tok/s), latency (TTFT, ITL) |

  Orchestrated by `run_all.sh`; cluster-specific settings and pass/fail thresholds
  live in `configs/cluster.conf`. Results land in timestamped `results/<run>/` dirs
  with a combined JSON summary. See `gpu-cluster-benchmarks/README.md` for the full
  quick-start, configuration reference, and per-benchmark details.

- **`multinode-nccl-allreduce-throughput/`** — standalone multi-node NCCL + synthetic
  DDP training benchmark run via `torchrun` under SLURM with Pyxis.
  - `multi-node-torchrun.sbatch` — 4-node / 8-GPU-per-node sbatch job that launches
    `torchrun` inside the NGC PyTorch container (`nvcr.io/nvidia/pytorch:26.03-py3`)
    with `--mpi=pmix` and `/dev/infiniband` mounted in.
  - `torchrun_benchmark.py` — sweeps NCCL all-reduce from 1 MB to 4 GB (reporting
    AlgBW / BusBW) and then runs a transformer-like DDP synthetic training loop for
    samples/sec throughput.

  ```bash
  sbatch benchmarks/slurm/multinode-nccl-allreduce-throughput/multi-node-torchrun.sbatch
  ```

## Typical workflow

1. **Per-node health** — run `healthchecks/gpu-healthcheck.sh` on each compute node
   and resolve any FAIL items before running distributed workloads.
2. **Cluster validation** — run `gpu-cluster-benchmarks/run_all.sh` for a full
   ClusterMAX-style evaluation, or the standalone `multi-node-torchrun.sbatch` for
   a quick NCCL + DDP sanity check.

## Prerequisites

- SLURM with Pyxis and Enroot for containerized execution
- NGC access (or pre-cached PyTorch / nccl-tests container images)
- `jq` on the submit node for JSON parsing
- Optional: HuggingFace token for gated models in the inference benchmark

## License

See [`LICENSE`](LICENSE).
