#!/usr/bin/env python3
"""
Multi-node GPU benchmark: NCCL all-reduce + synthetic training throughput.
Designed to run via torchrun on Slurm with Pyxis.
"""

import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-node GPU Benchmark")
    parser.add_argument("--nccl-iters", type=int, default=100, help="NCCL all-reduce iterations")
    parser.add_argument("--sizes-mb", type=float, nargs="+", default=[1, 8, 64, 256, 512, 1024, 2048, 4096],
                        help="Message sizes in MB for all-reduce benchmark")
    parser.add_argument("--train-iters", type=int, default=50, help="Synthetic training iterations")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension for synthetic model")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of transformer-like layers")
    return parser.parse_args()


def log_rank0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def nccl_allreduce_benchmark(args):
    """Benchmark NCCL all-reduce at various message sizes."""
    log_rank0("\n" + "=" * 70)
    log_rank0("NCCL ALL-REDUCE BENCHMARK")
    log_rank0("=" * 70)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    for size_mb in args.sizes_mb:
        numel = int(size_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
        tensor = torch.randn(numel, device=device)

        # Warmup
        for _ in range(10):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # Timed runs
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.nccl_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        avg_ms = (elapsed / args.nccl_iters) * 1000
        # bus bandwidth: 2*(N-1)/N * size / time
        algbw = (size_mb / 1000) / (avg_ms / 1000)  # GB/s
        busbw = algbw * 2 * (world_size - 1) / world_size

        log_rank0(f"  Size: {size_mb:>8.1f} MB | Avg: {avg_ms:>8.2f} ms | "
                  f"AlgBW: {algbw:>8.2f} GB/s | BusBW: {busbw:>8.2f} GB/s")

        del tensor
        torch.cuda.empty_cache()


def synthetic_training_benchmark(args):
    """Benchmark synthetic DDP training throughput."""
    log_rank0("\n" + "=" * 70)
    log_rank0("SYNTHETIC DDP TRAINING BENCHMARK")
    log_rank0("=" * 70)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")

    # Simple transformer-like model
    layers = []
    for _ in range(args.num_layers):
        layers.extend([
            nn.Linear(args.hidden_dim, args.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(args.hidden_dim * 4, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
        ])
    layers.append(nn.Linear(args.hidden_dim, 1000))
    model = nn.Sequential(*layers).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    param_count = sum(p.numel() for p in model.parameters())
    log_rank0(f"  Model params: {param_count / 1e6:.1f}M | Layers: {args.num_layers} | "
              f"Hidden: {args.hidden_dim} | Batch/GPU: {args.batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Warmup
    for _ in range(5):
        x = torch.randn(args.batch_size, args.hidden_dim, device=device)
        y = torch.randint(0, 1000, (args.batch_size,), device=device)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Timed runs
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.train_iters):
        x = torch.randn(args.batch_size, args.hidden_dim, device=device)
        y = torch.randint(0, 1000, (args.batch_size,), device=device)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    samples_per_sec_local = (args.train_iters * args.batch_size) / elapsed
    samples_per_sec_global = samples_per_sec_local * world_size

    log_rank0(f"  Time: {elapsed:.2f}s | "
              f"Throughput/GPU: {samples_per_sec_local:.1f} samples/s | "
              f"Global: {samples_per_sec_global:.1f} samples/s")


def gpu_info():
    """Print GPU and NCCL info."""
    log_rank0("\n" + "=" * 70)
    log_rank0("CLUSTER INFO")
    log_rank0("=" * 70)
    log_rank0(f"  PyTorch:    {torch.__version__}")
    log_rank0(f"  CUDA:       {torch.version.cuda}")
    log_rank0(f"  NCCL:       {torch.cuda.nccl.version()}")
    log_rank0(f"  World size: {dist.get_world_size()}")
    log_rank0(f"  GPUs/node:  {torch.cuda.device_count()}")
    log_rank0(f"  GPU model:  {torch.cuda.get_device_name(0)}")
    log_rank0(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def main():
    args = parse_args()
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    gpu_info()
    nccl_allreduce_benchmark(args)
    synthetic_training_benchmark(args)

    log_rank0("\n" + "=" * 70)
    log_rank0("BENCHMARK COMPLETE")
    log_rank0("=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

