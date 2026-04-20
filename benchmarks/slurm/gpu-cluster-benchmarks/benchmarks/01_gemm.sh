#!/usr/bin/env bash
# ==============================================================================
# Benchmark 01 — Single-GPU GEMM TFLOPS
#
# Verifies that individual GPUs achieve expected TFLOPS for General Matrix
# Multiply operations across precision levels (FP32, FP16, BF16, FP8).
# This is the ClusterMAX-equivalent "GEMM benchmark" — a quick way to verify
# each GPU is performing to spec before moving to multi-node tests.
#
# Runs one job per GPU on one node (8 tasks for 8 GPUs).
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/common.sh"
source "${SCRIPT_DIR}/../configs/cluster.conf"

header "Benchmark 01 — Single-GPU GEMM TFLOPS"

mkdir -p "${RESULTS_DIR}/01_gemm"

# ── Embedded Python benchmark ─────────────────────────────────────────────────
GEMM_SCRIPT=$(cat <<'PYEOF'
import torch
import torch.utils.benchmark as benchmark
import json
import os
import sys
import time

def gemm_benchmark(dtype, m, n, k, num_iters=200, warmup=50):
    """Run GEMM and return achieved TFLOPS."""
    if dtype == "fp8" and hasattr(torch, "float8_e4m3fn"):
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        # FP8 matmul via torch._scaled_mm
        scale_a = torch.ones(1, device="cuda", dtype=torch.float32)
        scale_b = torch.ones(1, device="cuda", dtype=torch.float32)
        def run():
            torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    elif dtype == "fp8":
        print(f"  FP8 not available in this PyTorch build, skipping", flush=True)
        return None
    else:
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        a = torch.randn(m, k, device="cuda", dtype=torch_dtype)
        b = torch.randn(k, n, device="cuda", dtype=torch_dtype)
        def run():
            torch.mm(a, b)

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(num_iters):
        run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops = 2 * m * n * k * num_iters
    tflops = flops / elapsed / 1e12
    return tflops

def hbm_bandwidth(size_mb=256, iters=100):
    """Measure HBM bandwidth via large tensor copy."""
    size = size_mb * 1024 * 1024 // 4  # float32 elements
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    # warmup
    for _ in range(10):
        b = a.clone()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        b = a.clone()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    bytes_moved = 2 * size * 4 * iters  # read + write
    bw_gbs = bytes_moved / elapsed / 1e9
    return bw_gbs

def main():
    gpu_id = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(gpu_id)
    gpu_name = torch.cuda.get_device_name(gpu_id)
    gpu_uuid = None
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits",
             f"--id={gpu_id}"],
            capture_output=True, text=True
        )
        gpu_uuid = result.stdout.strip()
    except Exception:
        gpu_uuid = f"gpu-{gpu_id}"

    print(f"[GPU {gpu_id}] {gpu_name} ({gpu_uuid})", flush=True)

    # Matrix sizes — large enough to saturate GPU
    M, N, K = 8192, 8192, 8192
    M_large, N_large, K_large = 16384, 16384, 16384

    results = {
        "gpu_id": gpu_id,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "benchmarks": {}
    }

    # FP32
    print(f"  [GPU {gpu_id}] FP32 GEMM ({M}x{N}x{K})...", flush=True)
    fp32 = gemm_benchmark("fp32", M, N, K, num_iters=100)
    results["benchmarks"]["fp32_tflops"] = round(fp32, 1) if fp32 else None
    print(f"  [GPU {gpu_id}] FP32: {fp32:.1f} TFLOPS" if fp32 else f"  [GPU {gpu_id}] FP32: SKIPPED", flush=True)

    # FP16
    print(f"  [GPU {gpu_id}] FP16 GEMM ({M_large}x{N_large}x{K_large})...", flush=True)
    fp16 = gemm_benchmark("fp16", M_large, N_large, K_large, num_iters=200)
    results["benchmarks"]["fp16_tflops"] = round(fp16, 1) if fp16 else None
    print(f"  [GPU {gpu_id}] FP16: {fp16:.1f} TFLOPS" if fp16 else f"  [GPU {gpu_id}] FP16: SKIPPED", flush=True)

    # BF16
    print(f"  [GPU {gpu_id}] BF16 GEMM ({M_large}x{N_large}x{K_large})...", flush=True)
    bf16 = gemm_benchmark("bf16", M_large, N_large, K_large, num_iters=200)
    results["benchmarks"]["bf16_tflops"] = round(bf16, 1) if bf16 else None
    print(f"  [GPU {gpu_id}] BF16: {bf16:.1f} TFLOPS" if bf16 else f"  [GPU {gpu_id}] BF16: SKIPPED", flush=True)

    # FP8
    print(f"  [GPU {gpu_id}] FP8 GEMM ({M_large}x{N_large}x{K_large})...", flush=True)
    fp8 = gemm_benchmark("fp8", M_large, N_large, K_large, num_iters=200)
    results["benchmarks"]["fp8_tflops"] = round(fp8, 1) if fp8 else None
    print(f"  [GPU {gpu_id}] FP8: {fp8:.1f} TFLOPS" if fp8 else f"  [GPU {gpu_id}] FP8: SKIPPED", flush=True)

    # HBM Bandwidth
    print(f"  [GPU {gpu_id}] HBM bandwidth...", flush=True)
    hbm_bw = hbm_bandwidth(size_mb=512, iters=100)
    results["benchmarks"]["hbm_bandwidth_gbs"] = round(hbm_bw, 1)
    print(f"  [GPU {gpu_id}] HBM BW: {hbm_bw:.1f} GB/s", flush=True)

    # Write per-GPU JSON
    outdir = os.environ.get("RESULTS_DIR", "/tmp")
    outfile = os.path.join(outdir, f"gemm_gpu{gpu_id}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [GPU {gpu_id}] Results written to {outfile}", flush=True)

if __name__ == "__main__":
    main()
PYEOF
)

# ── Write Python script to temp location ──────────────────────────────────────
GEMM_PY="${RESULTS_DIR}/01_gemm/gemm_bench.py"
echo "$GEMM_SCRIPT" > "$GEMM_PY"

# ── Submit SLURM job ──────────────────────────────────────────────────────────
info "Submitting GEMM benchmark (${GPUS_PER_NODE} GPUs on 1 node)..."

GEMM_DIR="${RESULTS_DIR}/01_gemm"

JOB_ID=$(sbatch --parsable \
    $(build_slurm_args) \
    --job-name="bench-01-gemm" \
    --nodes=1 \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --time=00:30:00 \
    --output="${GEMM_DIR}/slurm-%j.out" \
    --export=ALL,RESULTS_DIR="${GEMM_DIR}" \
    --wrap="srun ${PYXIS_COMMON} \
        --container-image=${PYTORCH_IMAGE} \
        --container-mounts=${GEMM_DIR}:${GEMM_DIR} \
        bash -c 'python3 ${GEMM_DIR}/gemm_bench.py' " \
    2>&1) || { fail "Failed to submit GEMM job"; return 1; }

info "GEMM job submitted: ${JOB_ID}"
echo "$JOB_ID" > "${RESULTS_DIR}/01_gemm/job_id"

# ── Parse results (called after job completes) ────────────────────────────────
parse_gemm_results() {
    header "Benchmark 01 — GEMM Results"

    local all_pass=true
    for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
        local jf="${RESULTS_DIR}/01_gemm/gemm_gpu${gpu}.json"
        if [[ ! -f "$jf" ]]; then
            record_skip "GPU ${gpu} GEMM" "results file not found"
            continue
        fi

        local gpu_name fp32 fp16 bf16 fp8 hbm
        gpu_name=$(jq -r '.gpu_name' "$jf")
        fp32=$(jq -r '.benchmarks.fp32_tflops // empty' "$jf")
        fp16=$(jq -r '.benchmarks.fp16_tflops // empty' "$jf")
        bf16=$(jq -r '.benchmarks.bf16_tflops // empty' "$jf")
        fp8=$(jq -r '.benchmarks.fp8_tflops // empty' "$jf")
        hbm=$(jq -r '.benchmarks.hbm_bandwidth_gbs // empty' "$jf")

        info "GPU ${gpu} (${gpu_name}):"

        [[ -n "$fp32" ]] && check_threshold "GPU${gpu} FP32 GEMM" "$fp32" "$GEMM_FP32_TFLOPS_MIN" "TFLOPS"
        [[ -n "$fp16" ]] && check_threshold "GPU${gpu} FP16 GEMM" "$fp16" "$GEMM_FP16_TFLOPS_MIN" "TFLOPS"
        [[ -n "$bf16" ]] && check_threshold "GPU${gpu} BF16 GEMM" "$bf16" "$GEMM_FP16_TFLOPS_MIN" "TFLOPS"
        [[ -n "$fp8" ]]  && check_threshold "GPU${gpu} FP8 GEMM"  "$fp8"  "$GEMM_FP8_TFLOPS_MIN"  "TFLOPS"
        [[ -n "$hbm" ]]  && check_threshold "GPU${gpu} HBM BW"    "$hbm"  "$HBM_BW_MIN_GBS"       "GB/s"
    done
}

# If called directly (not sourced), run and wait
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if wait_for_job "$JOB_ID" 1800; then
        parse_gemm_results
    fi
    print_summary "01 — GEMM TFLOPS"
    write_results_json "${RESULTS_DIR}/01_gemm/results.json"
fi
