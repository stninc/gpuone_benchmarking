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

# ── NCCL environment block ────────────────────────────────────────────────────
NCCL_ENV=""
NCCL_ENV+="NCCL_IB_DISABLE=${NCCL_IB_DISABLE},"
NCCL_ENV+="NCCL_CROSS_NIC=${NCCL_CROSS_NIC},"
NCCL_ENV+="NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX},"
NCCL_ENV+="NCCL_DEBUG=INFO,"
NCCL_ENV+="NCCL_DEBUG_SUBSYS=INIT,NET"
[[ -n "${NCCL_NET:-}" ]]           && NCCL_ENV+=",NCCL_NET=${NCCL_NET}"
[[ -n "${NCCL_IB_HCA:-}" ]]       && NCCL_ENV+=",NCCL_IB_HCA=${NCCL_IB_HCA}"
[[ -n "${NCCL_SOCKET_IFNAME:-}" ]] && NCCL_ENV+=",NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"

# ── Build & run nccl-tests inside the container ──────────────────────────────
# We compile nccl-tests at job start inside the container (takes ~60s).
# If your cluster caches a pre-built image, skip the build step.
NCCL_BUILD_AND_RUN=$(cat <<'RUNEOF'
#!/bin/bash
set -e

# ── Nuke ALL NCCL and UCX env vars from SLURM prolog / container image ────────
# Then set ONLY the minimal known-good values. This matches the manual srun
# command that works: just NCCL_IB_DISABLE=0, NCCL_SOCKET_IFNAME, LD_LIBRARY_PATH.
for var in $(env | grep -oP '^(NCCL_|UCX_|NVSHMEM_|HPCX_|OMPI_MCA_)[^=]*'); do
    unset "$var"
done

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}

echo "=== NCCL env (cleaned) ==="
env | grep -E '^NCCL_' | sort
echo ""

# Build nccl-tests WITHOUT MPI — avoids UCX wireup issues entirely
if [[ ! -x /tmp/nccl-tests/build/all_reduce_perf ]]; then
    echo "[$(hostname)] Building nccl-tests (no MPI)..."
    cd /tmp
    git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git 2>/dev/null || true
    cd nccl-tests
    make CUDA_HOME=/usr/local/cuda -j$(nproc)
    echo "[$(hostname)] nccl-tests built successfully"
fi

TESTBIN="/tmp/nccl-tests/build"
OUTDIR="${RESULTS_DIR:-/tmp}"
MINBYTES="${NCCL_MIN_BYTES:-8M}"
MAXBYTES="${NCCL_MAX_BYTES:-8G}"
STEPFACTOR="${NCCL_STEP_FACTOR:-2}"
ITERS="${NCCL_ITERS:-100}"
WARMUP="${NCCL_WARMUP:-50}"

echo "=== NCCL Test Configuration ==="
echo "Hosts: $(hostname), Tasks: ${SLURM_NTASKS:-1}, GPUs/proc: ${NCCL_GPUS_PER_PROC:-8}"
echo "Bytes: ${MINBYTES} → ${MAXBYTES} (step ×${STEPFACTOR})"
echo "Iters: ${ITERS}, Warmup: ${WARMUP}"
echo ""

NGPUS="${NCCL_GPUS_PER_PROC:-8}"

for TEST in all_reduce all_gather reduce_scatter; do
    echo "────────────────────────────────────────────────────────────"
    echo "Running: ${TEST}_perf"
    echo "────────────────────────────────────────────────────────────"
    "${TESTBIN}/${TEST}_perf" \
        -b "$MINBYTES" -e "$MAXBYTES" -f "$STEPFACTOR" \
        -g "$NGPUS" -n "$ITERS" -w "$WARMUP" \
        2>&1 | tee "${OUTDIR}/nccl_${TEST}.log"
    echo ""
done
RUNEOF
)

NCCL_RUNNER="${RESULTS_DIR}/02_nccl/nccl_runner.sh"
echo "$NCCL_BUILD_AND_RUN" > "$NCCL_RUNNER"
chmod +x "$NCCL_RUNNER"

# ── Phase A: Intra-node (1 node, 8 GPUs) ─────────────────────────────────────
info "Submitting intra-node NCCL test (1 node, ${GPUS_PER_NODE} GPUs, single process)..."

NCCL_DIR="${RESULTS_DIR}/02_nccl"

INTRA_JOB=$(sbatch --parsable \
    $(build_slurm_args) \
    --job-name="bench-02-nccl-intra" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --time=00:30:00 \
    --output="${NCCL_DIR}/slurm-intra-%j.out" \
    --export=ALL,RESULTS_DIR="${NCCL_DIR}",NCCL_GPUS_PER_PROC="${GPUS_PER_NODE}",NCCL_MIN_BYTES=8M,NCCL_MAX_BYTES=8G,NCCL_ITERS=100,NCCL_WARMUP=50,${NCCL_ENV} \
    --wrap="srun ${PYXIS_COMMON} \
        --container-image=${NCCL_TEST_IMAGE} \
        --container-mounts=${NCCL_DIR}:${NCCL_DIR} \
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
        --export=ALL,RESULTS_DIR="${NCCL_DIR}",NCCL_GPUS_PER_PROC="${GPUS_PER_NODE}",NCCL_MIN_BYTES=8M,NCCL_MAX_BYTES=8G,NCCL_ITERS=50,NCCL_WARMUP=20,${NCCL_ENV} \
        --wrap="srun ${PYXIS_COMMON} \
            --container-image=${NCCL_TEST_IMAGE} \
            --container-mounts=${NCCL_DIR}:${NCCL_DIR} \
            bash ${NCCL_RUNNER}" \
        2>&1) || { fail "Failed to submit inter-node NCCL job"; INTER_JOB=""; }

    [[ -n "$INTER_JOB" ]] && info "Inter-node NCCL job: ${INTER_JOB}"
else
    warn "NCCL_NODES=${NCCL_NODES} — skipping inter-node test"
    INTER_JOB=""
fi

echo "$INTRA_JOB" > "${RESULTS_DIR}/02_nccl/intra_job_id"
echo "${INTER_JOB:-none}" > "${RESULTS_DIR}/02_nccl/inter_job_id"

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
    grep -E '^\s+[0-9]' "$logfile" | tail -1 | awk '{print $(NF-1)}' 2>/dev/null || echo "0"
}

parse_nccl_results() {
    header "Benchmark 02 — NCCL Results"

    # Intra-node
    for test in all_reduce all_gather reduce_scatter; do
        local logfile="${RESULTS_DIR}/02_nccl/nccl_${test}.log"
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

    # Inter-node (results will be in same dir if multi-node job overwrites)
    if [[ "${NCCL_NODES}" -gt 1 ]]; then
        # Inter-node logs may have a different prefix or be in slurm output
        local inter_log="${RESULTS_DIR}/02_nccl/slurm-inter-${INTER_JOB}.out"
        if [[ -f "$inter_log" ]]; then
            # Extract last all_reduce bus bandwidth from the combined output
            local inter_bw
            inter_bw=$(grep -A 200 "all_reduce_perf" "$inter_log" | grep -E '^\s+[0-9]' | tail -1 | awk '{print $(NF-1)}' 2>/dev/null || echo "0")
            [[ "$inter_bw" != "0" ]] \
                && check_threshold "Inter-node all_reduce bus BW (${NCCL_NODES}N)" "$inter_bw" "$NCCL_INTER_BW_MIN_GBS" "GB/s" \
                || record_skip "Inter-node all_reduce" "could not parse results"
        else
            record_skip "Inter-node NCCL" "output file not found"
        fi
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
