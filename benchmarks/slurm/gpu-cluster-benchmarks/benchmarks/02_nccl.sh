#!/usr/bin/env bash
# ==============================================================================
# Benchmark 02 — NCCL Collective Bandwidth
#
# Tests GPU interconnect performance using nccl-tests. Runs:
#   - Intra-node all_reduce  (NVLink/NVSwitch — 1 node, 8 GPUs)
#   - Inter-node all_reduce  (RoCE/IB — N nodes, 8 GPUs each)
#   - All-gather, reduce-scatter (single + multi node)
#
# This validates that NVLink/NVSwitch fabric and inter-node network are
# delivering expected bandwidth — a critical ClusterMAX evaluation axis.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/common.sh"
source "${SCRIPT_DIR}/../configs/cluster.conf"

header "Benchmark 02 — NCCL Collective Bandwidth"

mkdir -p "${RESULTS_DIR}/02_nccl"
NCCL_DIR="${RESULTS_DIR}/02_nccl"

# ── Generate NCCL env file from cluster.conf ─────────────────────────────────
# Captures every NCCL_*/UCX_*/etc var currently set, writes them to a file
# the runner sources AFTER nuking image-baked vars. This is the single source
# of truth for NCCL config — edit cluster.conf, not the runner.
BENCH_NCCL_ENV_FILE="${NCCL_DIR}/nccl_env.sh"
write_nccl_env_file "${BENCH_NCCL_ENV_FILE}"

# ── Build & run nccl-tests inside the container ──────────────────────────────
# INTRA-NODE: nccl-tests built without MPI, single-process-multi-thread (-g 8).
# INTER-NODE: single-process-per-rank across 2 nodes via torchrun bootstrap,
#             running a Python wrapper that calls torch.distributed collectives
#             (same measurement the test does, but actually spans nodes).
NCCL_BUILD_AND_RUN=$(cat <<'RUNEOF'
#!/bin/bash
set -e

# ── Step 1: Nuke image-baked NCCL/UCX env vars ────────────────────────────────
# NGC containers ship with NCCL_IB_SL=1 / UCX_IB_TRAFFIC_CLASS=0xE0 baked in,
# which can hit non-PFC priorities on some clusters and break QPs.
for var in $(env | grep -oP '^(NCCL_|UCX_|NVSHMEM_|HPCX_|OMPI_MCA_)[^=]*'); do
    unset "$var"
done

# ── Step 2: Re-apply user's intended config from cluster.conf ────────────────
if [[ -z "${BENCH_NCCL_ENV_FILE:-}" ]]; then
    echo "WARNING: BENCH_NCCL_ENV_FILE not set in env — NCCL config from cluster.conf will NOT be applied" >&2
elif [[ ! -f "${BENCH_NCCL_ENV_FILE}" ]]; then
    echo "WARNING: BENCH_NCCL_ENV_FILE=${BENCH_NCCL_ENV_FILE} does not exist on this host — check container mounts" >&2
else
    echo "Sourcing NCCL env file: ${BENCH_NCCL_ENV_FILE}"
    # shellcheck disable=SC1090
    source "${BENCH_NCCL_ENV_FILE}"
fi

# ── Step 3: Defensive defaults if cluster.conf left them blank ───────────────
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}

# torchrun rendezvous uses GLOO; mirror NCCL_SOCKET_IFNAME so it binds correctly.
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
fi

echo "=== NCCL env (post-source) ==="
env | grep -E '^(NCCL_|UCX_|GLOO_)' | sort
echo ""

echo "=== RDMA visibility check ==="
echo "/dev/infiniband contents:"
ls /dev/infiniband/ 2>&1 | head -10 | sed 's/^/  /'
echo "ibv_devices output:"
(command -v ibv_devices >/dev/null && ibv_devices 2>&1 || echo "  ibv_devices not installed") | head -10 | sed 's/^/  /'
echo ""

OUTDIR="${RESULTS_DIR:-/tmp}"
MINBYTES="${NCCL_MIN_BYTES:-8M}"
MAXBYTES="${NCCL_MAX_BYTES:-8G}"
STEPFACTOR="${NCCL_STEP_FACTOR:-2}"
ITERS="${NCCL_ITERS:-100}"
WARMUP="${NCCL_WARMUP:-50}"
NGPUS="${NCCL_GPUS_PER_PROC:-8}"
NNODES="${SLURM_NNODES:-1}"

echo "=== NCCL Test Configuration ==="
echo "Hostname: $(hostname)"
echo "NNODES: ${NNODES}, NODE_RANK: ${SLURM_NODEID:-0}, GPUs/node: ${NGPUS}"
echo "Bytes: ${MINBYTES} → ${MAXBYTES} (step ×${STEPFACTOR})"
echo "Iters: ${ITERS}, Warmup: ${WARMUP}"
echo ""

if [[ "${NNODES}" -eq 1 ]]; then
    # ── INTRA-NODE path: nccl-tests binary, single-proc-multi-thread ──
    if [[ ! -x /tmp/nccl-tests/build/all_reduce_perf ]]; then
        echo "[$(hostname)] Building nccl-tests (no MPI)..."
        cd /tmp
        git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git 2>/dev/null || true
        cd nccl-tests
        make CUDA_HOME=/usr/local/cuda -j$(nproc)
        echo "[$(hostname)] nccl-tests built successfully"
    fi
    TESTBIN="/tmp/nccl-tests/build"
    for TEST in all_reduce all_gather reduce_scatter; do
        echo "────────────────────────────────────────────────────────────"
        echo "Running: ${TEST}_perf (intra-node, -g ${NGPUS})"
        echo "────────────────────────────────────────────────────────────"
        "${TESTBIN}/${TEST}_perf" \
            -b "$MINBYTES" -e "$MAXBYTES" -f "$STEPFACTOR" \
            -g "$NGPUS" -n "$ITERS" -w "$WARMUP" \
            2>&1 | tee "${OUTDIR}/nccl_intra_${TEST}.log"
        echo ""
    done
else
    # ── INTER-NODE path: torchrun launches one process per GPU across nodes ──
    # Writes a tiny Python script that measures the three collectives. This is
    # what nccl-tests would do with MPI, but without requiring MPI infra.
    PYFILE="${OUTDIR}/nccl_torch_bench.py"
    cat > "${PYFILE}" <<'PYEOF'
import os, time, datetime, torch, torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
rank, world = dist.get_rank(), dist.get_world_size()

def fmt_bytes(b):
    if b >= 1<<30: return f"{b/(1<<30):.2f} GB"
    if b >= 1<<20: return f"{b/(1<<20):.0f} MB"
    return f"{b/(1<<10):.0f} KB"

def log(msg):
    if rank == 0: print(msg, flush=True)

minb = int(os.environ.get("NCCL_MIN_BYTES_PARSED", 8 * (1<<20)))
maxb = int(os.environ.get("NCCL_MAX_BYTES_PARSED", 8 * (1<<30)))
stepf = int(os.environ.get("NCCL_STEP_FACTOR", 2))
iters = int(os.environ.get("NCCL_ITERS", 50))
warmup = int(os.environ.get("NCCL_WARMUP", 20))

def bench(op, size_bytes):
    elems = size_bytes // 4  # float32
    if op == "all_reduce":
        buf = torch.zeros(elems, dtype=torch.float32, device=f"cuda:{local_rank}")
        fn = lambda: dist.all_reduce(buf)
        # algbw = bytes / time, busbw = algbw * 2*(n-1)/n
        busbw_factor = 2 * (world - 1) / world
    elif op == "all_gather":
        shard = size_bytes // world
        elems = shard // 4
        inp = torch.zeros(elems, dtype=torch.float32, device=f"cuda:{local_rank}")
        out = torch.empty(elems * world, dtype=torch.float32, device=f"cuda:{local_rank}")
        fn = lambda: dist.all_gather_into_tensor(out, inp)
        busbw_factor = (world - 1) / world
    elif op == "reduce_scatter":
        shard = size_bytes // world
        elems = shard // 4
        inp = torch.zeros(elems * world, dtype=torch.float32, device=f"cuda:{local_rank}")
        out = torch.empty(elems, dtype=torch.float32, device=f"cuda:{local_rank}")
        fn = lambda: dist.reduce_scatter_tensor(out, inp)
        busbw_factor = (world - 1) / world
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    algbw = size_bytes / dt / 1e9
    busbw = algbw * busbw_factor
    return dt*1e6, algbw, busbw

for op in ["all_reduce", "all_gather", "reduce_scatter"]:
    log(f"\n{'─'*60}\nRunning: {op}_perf (inter-node, world={world})\n{'─'*60}")
    log(f"#       size       count   type   time(us)   algbw(GB/s)   busbw(GB/s)")
    s = minb
    while s <= maxb:
        dt_us, algbw, busbw = bench(op, s)
        log(f"  {s:>12d}  {s//4:>10d}  float  {dt_us:>10.1f}  {algbw:>11.2f}  {busbw:>11.2f}  0")
        s *= stepf

dist.destroy_process_group()
PYEOF

    # Parse sizes like "8M" "8G" to bytes
    parse_size() {
        local s="$1"
        case "$s" in
            *G|*g) echo $(( ${s%[Gg]} * 1024 * 1024 * 1024 )) ;;
            *M|*m) echo $(( ${s%[Mm]} * 1024 * 1024 )) ;;
            *K|*k) echo $(( ${s%[Kk]} * 1024 )) ;;
            *) echo "$s" ;;
        esac
    }
    export NCCL_MIN_BYTES_PARSED=$(parse_size "$MINBYTES")
    export NCCL_MAX_BYTES_PARSED=$(parse_size "$MAXBYTES")
    export NCCL_STEP_FACTOR="$STEPFACTOR"
    export NCCL_ITERS="$ITERS"
    export NCCL_WARMUP="$WARMUP"

    # torchrun expects MASTER_ADDR to be set by the sbatch wrapper
    export MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be set in env}"
    export MASTER_PORT="${MASTER_PORT:-29500}"

    echo "torchrun: NNODES=${NNODES} NODE_RANK=${SLURM_NODEID:-0} NGPUS=${NGPUS} MASTER=${MASTER_ADDR}:${MASTER_PORT}"

    # Combined log file for parser; tee lets us also see it live.
    torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NGPUS}" \
        --node_rank="${SLURM_NODEID:-0}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "${PYFILE}" \
        2>&1 | tee "${OUTDIR}/nccl_inter.log"

    # Split the combined log into per-op logs so the existing parser still works.
    awk '
        /^Running: all_reduce_perf/     { out="'"${OUTDIR}"'/nccl_inter_all_reduce.log" }
        /^Running: all_gather_perf/     { out="'"${OUTDIR}"'/nccl_inter_all_gather.log" }
        /^Running: reduce_scatter_perf/ { out="'"${OUTDIR}"'/nccl_inter_reduce_scatter.log" }
        out { print > out }
    ' "${OUTDIR}/nccl_inter.log"
fi
RUNEOF
)

NCCL_RUNNER="${NCCL_DIR}/nccl_runner.sh"
echo "$NCCL_BUILD_AND_RUN" > "$NCCL_RUNNER"
chmod +x "$NCCL_RUNNER"

# ── Phase A: Intra-node (1 node, 8 GPUs) ─────────────────────────────────────
info "Submitting intra-node NCCL test (1 node, ${GPUS_PER_NODE} GPUs, single process)..."

INTRA_JOB=$(sbatch --parsable \
    $(build_slurm_args) \
    --job-name="bench-02-nccl-intra" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --time=00:30:00 \
    --output="${NCCL_DIR}/slurm-intra-%j.out" \
    --export=ALL,RESULTS_DIR="${NCCL_DIR}",BENCH_NCCL_ENV_FILE="${BENCH_NCCL_ENV_FILE}",NCCL_GPUS_PER_PROC="${GPUS_PER_NODE}",NCCL_MIN_BYTES=8M,NCCL_MAX_BYTES=8G,NCCL_ITERS=100,NCCL_WARMUP=50 \
    --wrap="srun ${PYXIS_COMMON} \
        --container-image=${NCCL_TEST_IMAGE} \
        --container-mounts=${PYXIS_MOUNTS},${NCCL_DIR}:${NCCL_DIR} \
        bash ${NCCL_RUNNER}" \
    2>&1) || { fail "Failed to submit intra-node NCCL job"; INTRA_JOB=""; }

[[ -n "$INTRA_JOB" ]] && info "Intra-node NCCL job: ${INTRA_JOB}"

# ── Phase B: Inter-node (N nodes, 8 GPUs each) ───────────────────────────────
if [[ "${NCCL_NODES}" -gt 1 ]]; then
    TOTAL_GPUS=$((NCCL_NODES * GPUS_PER_NODE))
    info "Submitting inter-node NCCL test (${NCCL_NODES} nodes, ${TOTAL_GPUS} GPUs)..."

    INTER_JOB=$(sbatch --parsable \
        $(build_slurm_args) \
        --job-name="bench-02-nccl-inter" \
        --nodes="${NCCL_NODES}" \
        --ntasks-per-node=1 \
        --gpus-per-node="${GPUS_PER_NODE}" \
        --time=00:45:00 \
        --output="${NCCL_DIR}/slurm-inter-%j.out" \
        --export=ALL,RESULTS_DIR="${NCCL_DIR}",BENCH_NCCL_ENV_FILE="${BENCH_NCCL_ENV_FILE}",NCCL_GPUS_PER_PROC="${GPUS_PER_NODE}",NCCL_MIN_BYTES=8M,NCCL_MAX_BYTES=8G,NCCL_ITERS=50,NCCL_WARMUP=20 \
        --wrap="
            export MASTER_ADDR=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | head -1)
            export MASTER_PORT=29500
            srun ${PYXIS_COMMON} \
                --container-image=${NCCL_TEST_IMAGE} \
                --container-mounts=${PYXIS_MOUNTS},${NCCL_DIR}:${NCCL_DIR} \
                bash ${NCCL_RUNNER}" \
        2>&1) || { fail "Failed to submit inter-node NCCL job"; INTER_JOB=""; }

    [[ -n "$INTER_JOB" ]] && info "Inter-node NCCL job: ${INTER_JOB}"
else
    warn "NCCL_NODES=${NCCL_NODES} — skipping inter-node test"
    INTER_JOB=""
fi

echo "$INTRA_JOB" > "${NCCL_DIR}/intra_job_id"
echo "${INTER_JOB:-none}" > "${NCCL_DIR}/inter_job_id"

# ── Parse nccl-tests output ───────────────────────────────────────────────────
# Extracts the bus bandwidth at the largest message size from each test log.
extract_nccl_busbw() {
    local logfile="$1"
    if [[ ! -f "$logfile" ]]; then
        echo "0"
        return
    fi
    # nccl-tests output: last data line has the largest message size result
    # Columns: size(B)  count  type  redop  root  time(us)  algbw(GB/s)  busbw(GB/s)  error
    # We want busbw (column 12 in newer nccl-tests, or the second-to-last numeric column)
    # Force --text in case log has binary control bytes from container output;
    # strip ANSI escapes before parsing.
    grep -a -E '^\s+[0-9]' "$logfile" \
        | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' \
        | tail -1 | awk '{print $(NF-1)}' 2>/dev/null || echo "0"
}

parse_nccl_results() {
    header "Benchmark 02 — NCCL Results"

    # Intra-node — reads nccl_intra_*.log (written by phase A inside container)
    for test in all_reduce all_gather reduce_scatter; do
        local logfile="${RESULTS_DIR}/02_nccl/nccl_intra_${test}.log"
        local bw threshold
        bw=$(extract_nccl_busbw "$logfile")

        # Per-collective thresholds
        case "$test" in
            all_reduce)      threshold="${NCCL_INTRA_BW_MIN_GBS}" ;;
            all_gather)      threshold="${NCCL_INTRA_GATHER_BW_MIN_GBS:-${NCCL_INTRA_BW_MIN_GBS}}" ;;
            reduce_scatter)  threshold="${NCCL_INTRA_RS_BW_MIN_GBS:-${NCCL_INTRA_BW_MIN_GBS}}" ;;
        esac

        if [[ "$bw" != "0" && -n "$bw" ]]; then
            check_threshold "Intra-node ${test} bus BW" "$bw" "$threshold" "GB/s"
        else
            record_skip "Intra-node ${test}" "no results"
        fi
    done

    # Inter-node — reads nccl_inter_*.log (written by phase B after torchrun split)
    if [[ "${NCCL_NODES}" -gt 1 ]]; then
        for test in all_reduce all_gather reduce_scatter; do
            local inter_logfile="${RESULTS_DIR}/02_nccl/nccl_inter_${test}.log"
            local inter_bw
            inter_bw=$(extract_nccl_busbw "$inter_logfile")

            if [[ "$inter_bw" != "0" && -n "$inter_bw" ]]; then
                check_threshold "Inter-node ${test} bus BW (${NCCL_NODES}N)" "$inter_bw" "$NCCL_INTER_BW_MIN_GBS" "GB/s"
            else
                record_skip "Inter-node ${test} (${NCCL_NODES}N)" "no results"
            fi
        done
    fi
}

# ── Direct execution ──────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    [[ -n "$INTRA_JOB" ]] && wait_for_job "$INTRA_JOB" 2400
    [[ -n "$INTER_JOB" ]] && wait_for_job "$INTER_JOB" 3600
    parse_nccl_results
    print_summary "02 — NCCL Bandwidth"
    write_results_json "${RESULTS_DIR}/02_nccl/results.json"
fi
