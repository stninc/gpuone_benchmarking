# GPU Cluster Benchmark Suite

ClusterMAX-equivalent benchmark suite for GPU clusters using open-source tooling.
Covers the same evaluation axes as [SemiAnalysis ClusterMAX](https://www.clustermax.ai/):

| # | Benchmark | Tool | What It Measures |
|---|-----------|------|------------------|
| 01 | GEMM TFLOPS | PyTorch | Per-GPU compute at FP32/FP16/BF16/FP8 + HBM bandwidth |
| 02 | NCCL Bandwidth | nccl-tests | Intra-node (NVLink/NVSwitch) + inter-node (RoCE/IB) collectives |
| 03 | Training MFU | torchtitan | Multi-node Llama pretraining — MFU, step time, tokens/sec |
| 04 | Inference | vLLM | Offline throughput (tok/s), latency (TTFT, ITL) |

Designed for **SLURM + Pyxis/Enroot** container execution on NVIDIA GPU clusters.
Default configuration targets **HGX B300 8-GPU** nodes (Supermicro AS-8126GS-NB3RT or similar).

## Directory Structure

```
gpu-cluster-benchmarks/
├── run_all.sh                  # Main orchestrator
├── configs/
│   └── cluster.conf            # ← EDIT THIS — cluster-specific settings & thresholds
├── benchmarks/
│   ├── 01_gemm.sh              # Single-GPU GEMM TFLOPS
│   ├── 02_nccl.sh              # NCCL collective bandwidth
│   ├── 03_training.sh          # Multi-node training MFU
│   └── 04_inference.sh         # Inference throughput + latency
├── scripts/
│   └── common.sh               # Shared functions (logging, thresholds, SLURM helpers)
└── results/                    # Auto-created per run with timestamp
    └── 20260417_143000/
        ├── combined_results.json
        ├── 01_gemm/
        ├── 02_nccl/
        ├── 03_training/
        └── 04_inference/
```

## Quick Start

```bash
# 1. Clone/copy to your SLURM login node
scp -r gpu-cluster-benchmarks/ login-node:~/

# 2. Edit cluster config
vi configs/cluster.conf    # Set PARTITION, GPU_TYPE, container images, thresholds

# 3. Run all benchmarks
chmod +x run_all.sh benchmarks/*.sh
./run_all.sh

# 4. Run specific benchmarks only
./run_all.sh --benchmarks gemm,nccl

# 5. Dry run (see what would be submitted)
./run_all.sh --dry-run

# 6. Fire and forget (submit jobs, don't wait)
./run_all.sh --no-wait
```

## Configuration

All tunable parameters live in `configs/cluster.conf`. Key settings:

### Cluster
- `PARTITION` — SLURM partition name
- `GPUS_PER_NODE` — GPUs per node (8 for HGX)
- `GPU_TYPE` — Used for labeling (e.g., "b300", "h100", "h200")

### Multi-Node
- `NCCL_NODES` — Nodes for NCCL inter-node tests (default: 2)
- `TRAIN_NODES` — Nodes for training benchmark (default: 2)

### Container Images
- `PYTORCH_IMAGE` — NGC PyTorch image (used for GEMM, training, inference)
- `NCCL_TEST_IMAGE` — Image for building/running nccl-tests

### Thresholds (Pass/Fail)
Conservative minimums. Adjust per your hardware:

| Metric | B300 Default | H100 Suggestion | H200 Suggestion |
|--------|-------------|-----------------|-----------------|
| FP16 GEMM | 1800 TFLOPS | 900 TFLOPS | 900 TFLOPS |
| FP8 GEMM | 3500 TFLOPS | 1800 TFLOPS | 1800 TFLOPS |
| FP32 GEMM | 150 TFLOPS | 60 TFLOPS | 60 TFLOPS |
| HBM BW | 6000 GB/s | 2500 GB/s | 3500 GB/s |
| Intra-node BW | 800 GB/s | 400 GB/s | 400 GB/s |
| Inter-node BW | 80 GB/s | 40 GB/s | 40 GB/s |
| Training MFU | 35% | 35% | 35% |

### NCCL Tuning
The `NCCL_*` variables in the config control network transport. Defaults assume InfiniBand/RoCE.
Adjust `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `NCCL_NET` for your fabric.

## Benchmark Details

### 01 — GEMM TFLOPS
Runs a PyTorch matmul benchmark on each GPU independently (8 tasks on 1 node).
Tests FP32 (8192³), FP16/BF16/FP8 (16384³), and HBM bandwidth via tensor copy.
Outputs per-GPU JSON with TFLOPS and bandwidth numbers.

**Why it matters:** A GPU that can't hit expected GEMM TFLOPS has a hardware or thermal issue.
This is the first thing to check before investing time in multi-node tests.

### 02 — NCCL Bandwidth
Builds nccl-tests inside the container and runs all_reduce, all_gather, and
reduce_scatter from 8MB to 8GB message sizes.

- **Intra-node:** Validates NVLink/NVSwitch fabric (1 node, 8 GPUs)
- **Inter-node:** Validates RoCE/IB network (N nodes, 8 GPUs each)

Reports bus bandwidth at the largest message size. This is the metric that matters
for distributed training performance.

### 03 — Training MFU
Runs Llama 3 8B pretraining using Meta's torchtitan framework on the C4 dataset.
Measures Model FLOPS Utilization (MFU) and tokens/sec across multiple nodes.

MFU is the ratio of observed compute throughput to theoretical peak — the single
most important number for evaluating a training cluster. An MFU of 40%+ on Llama 8B
across 16 GPUs indicates a well-configured cluster.

**Note:** torchtitan requires network access to clone the repo on first run.
For air-gapped clusters, pre-build a container image with torchtitan included.

### 04 — Inference Throughput
Uses vLLM for offline throughput measurement and single-request latency profiling.

- **Throughput:** 500 prompts at 512 input / 512 output tokens → tok/s
- **Latency:** 20 sequential single-prompt runs → avg, P50, P99 latency + inter-token latency

For gated models (Llama 3.1), set `HF_TOKEN` in your environment or config.

## Output

Each run creates a timestamped directory in `results/`. The combined results JSON
(`combined_results.json`) contains all pass/fail results for downstream processing.

```json
{
  "cluster": "b300-cluster",
  "timestamp": "2026-04-17T14:30:00Z",
  "gpu_type": "b300",
  "summary": { "passed": 42, "failed": 0, "skipped": 2 },
  "results": [
    { "test": "GPU0 FP16 GEMM", "status": "PASS", "value": "2100.3 TFLOPS", "threshold": ">= 1800 TFLOPS" },
    ...
  ]
}
```

## Prerequisites

- **SLURM** with Pyxis and Enroot for container execution
- **NGC access** for pulling container images (or pre-cached images)
- **jq** on the submit node for JSON parsing
- **Network access** from compute nodes for cloning repos (torchtitan, nccl-tests)
  — or pre-build containers with these included
- **HuggingFace token** (optional) for gated models in inference benchmark

## Adapting for Other GPU Types

1. Copy `configs/cluster.conf` to `configs/h100.conf`
2. Adjust thresholds for your GPU (see table above)
3. Run: `./run_all.sh --config configs/h100.conf`

## Relation to ClusterMAX

This suite covers the **benchmarking** dimensions of ClusterMAX. The full ClusterMAX
rating also evaluates security posture, orchestration maturity, storage performance,
networking configuration, reliability/SLAs, monitoring, pricing, and support — which
are operational evaluations rather than benchmark scripts.

For storage benchmarking, see the companion C.2 acceptance script (fio-based NVMe tests).
For security evaluation, see the OS hardening and remote attestation documentation.
