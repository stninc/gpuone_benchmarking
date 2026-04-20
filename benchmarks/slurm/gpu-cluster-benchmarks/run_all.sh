#!/usr/bin/env bash
# ==============================================================================
# GPU Cluster Benchmark Suite — Main Orchestrator
#
# ClusterMAX-equivalent benchmark suite for GPU clusters using open-source tools.
# Covers the same evaluation axes as SemiAnalysis ClusterMAX:
#
#   01. GEMM TFLOPS      — Per-GPU compute verification (FP32/FP16/BF16/FP8)
#   02. NCCL Bandwidth   — Intra-node (NVLink) + inter-node (RoCE/IB) collectives
#   03. Training MFU     — Multi-node Llama pretraining via torchtitan
#   04. Inference         — vLLM throughput + latency benchmark
#
# Usage:
#   ./run_all.sh [OPTIONS]
#
# Options:
#   --config FILE        Path to cluster config (default: configs/cluster.conf)
#   --benchmarks LIST    Comma-separated list: gemm,nccl,training,inference,all
#                        (default: all)
#   --dry-run            Print what would be submitted without actually submitting
#   --no-wait            Submit all jobs and exit without waiting for results
#   --skip-parse         Submit and wait, but skip result parsing
#   -h, --help           Show this help
#
# Prerequisites:
#   - SLURM cluster with Pyxis + Enroot (for --container-image)
#   - NGC container access (or pre-pulled images)
#   - jq (for JSON parsing)
#
# Examples:
#   ./run_all.sh                                  # Run all benchmarks
#   ./run_all.sh --benchmarks gemm,nccl           # GEMM + NCCL only
#   ./run_all.sh --config configs/h100.conf       # Use alternate config
#   ./run_all.sh --no-wait                        # Fire and forget
# ==============================================================================

set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG_FILE="${SUITE_DIR}/configs/cluster.conf"
BENCHMARKS="all"
DRY_RUN=false
NO_WAIT=false
SKIP_PARSE=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG_FILE="$2"; shift 2 ;;
        --benchmarks)   BENCHMARKS="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --no-wait)      NO_WAIT=true; shift ;;
        --skip-parse)   SKIP_PARSE=true; shift ;;
        -h|--help)
            sed -n '2,/^# =====/p' "$0" | grep '^#' | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Load config and common functions ──────────────────────────────────────────
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

source "${SUITE_DIR}/scripts/common.sh"
source "$CONFIG_FILE"

# ── Validate environment ──────────────────────────────────────────────────────
header "GPU Cluster Benchmark Suite"

info "Cluster:    ${CLUSTER_NAME}"
info "GPU type:   ${GPU_TYPE}"
info "Partition:  ${PARTITION}"
info "Tag:        ${BENCHMARK_TAG}"
info "Config:     ${CONFIG_FILE}"
info "Benchmarks: ${BENCHMARKS}"
info "Results:    ${RESULTS_DIR}"
echo

check_slurm

if ! command -v jq &>/dev/null; then
    warn "jq not found — result parsing may fail. Install with: sudo apt install jq"
fi

mkdir -p "${RESULTS_DIR}"

# Save config snapshot
cp "$CONFIG_FILE" "${RESULTS_DIR}/cluster.conf.snapshot"
info "Config snapshot saved to ${RESULTS_DIR}/cluster.conf.snapshot"

# ── Determine which benchmarks to run ─────────────────────────────────────────
run_gemm=false; run_nccl=false; run_training=false; run_inference=false

if [[ "$BENCHMARKS" == "all" ]]; then
    run_gemm=true; run_nccl=true; run_training=true; run_inference=true
else
    IFS=',' read -ra BENCH_LIST <<< "$BENCHMARKS"
    for b in "${BENCH_LIST[@]}"; do
        case "$(echo "$b" | tr '[:upper:]' '[:lower:]' | xargs)" in
            gemm|01)       run_gemm=true ;;
            nccl|02)       run_nccl=true ;;
            training|train|03) run_training=true ;;
            inference|infer|04) run_inference=true ;;
            *) warn "Unknown benchmark: ${b}" ;;
        esac
    done
fi

# ── Dry run mode ──────────────────────────────────────────────────────────────
if $DRY_RUN; then
    header "Dry Run — Would Submit:"
    $run_gemm      && info "  01. GEMM TFLOPS (1 node, ${GPUS_PER_NODE} GPUs)"
    $run_nccl      && info "  02. NCCL Bandwidth (intra: 1 node, inter: ${NCCL_NODES} nodes)"
    $run_training   && info "  03. Training MFU (${TRAIN_NODES} nodes, model: ${TRAIN_MODEL:-llama3_8b})"
    $run_inference  && info "  04. Inference (${INFERENCE_NODES} node, model: ${INFER_MODEL:-meta-llama/Llama-3.1-8B-Instruct})"
    echo
    info "Total GPU-hours estimate: ~$((1 + NCCL_NODES + TRAIN_NODES * 2 + INFERENCE_NODES * 2)) node-hours"
    exit 0
fi

# ── Submit benchmarks ─────────────────────────────────────────────────────────
declare -A JOB_IDS

if $run_gemm; then
    source "${SUITE_DIR}/benchmarks/01_gemm.sh"
    JOB_IDS[gemm]="${JOB_ID:-}"
fi

if $run_nccl; then
    source "${SUITE_DIR}/benchmarks/02_nccl.sh"
    JOB_IDS[nccl_intra]="${INTRA_JOB:-}"
    JOB_IDS[nccl_inter]="${INTER_JOB:-}"
fi

if $run_training; then
    source "${SUITE_DIR}/benchmarks/03_training.sh"
    JOB_IDS[training]="${TRAIN_JOB:-}"
fi

if $run_inference; then
    source "${SUITE_DIR}/benchmarks/04_inference.sh"
    JOB_IDS[inference]="${INFER_JOB:-}"
fi

# ── Print submitted jobs ──────────────────────────────────────────────────────
header "Submitted Jobs"
for name in "${!JOB_IDS[@]}"; do
    local_id="${JOB_IDS[$name]}"
    [[ -n "$local_id" && "$local_id" != "none" ]] \
        && info "  ${name}: ${local_id}" \
        || warn "  ${name}: not submitted"
done
echo

if $NO_WAIT; then
    info "All jobs submitted. Use 'squeue -u \$USER' to monitor progress."
    info "Results will be written to: ${RESULTS_DIR}"
    info "To parse results later, re-run with: --benchmarks <name> --skip-parse=false"
    exit 0
fi

# ── Wait for all jobs ─────────────────────────────────────────────────────────
header "Waiting for Jobs to Complete"

for name in "${!JOB_IDS[@]}"; do
    local_id="${JOB_IDS[$name]}"
    if [[ -n "$local_id" && "$local_id" != "none" ]]; then
        wait_for_job "$local_id" 7200 || warn "Job ${name} (${local_id}) did not complete successfully"
    fi
done

if $SKIP_PARSE; then
    info "Skipping result parsing (--skip-parse). Raw output in: ${RESULTS_DIR}"
    exit 0
fi

# ── Parse results ─────────────────────────────────────────────────────────────
# Reset counters for combined summary
PASS_COUNT=0; FAIL_COUNT=0; SKIP_COUNT=0; RESULTS=()

$run_gemm      && parse_gemm_results
$run_nccl      && parse_nccl_results
$run_training   && parse_training_results
$run_inference  && parse_inference_results

# ── Final summary ─────────────────────────────────────────────────────────────
print_summary "GPU Cluster Benchmark Suite — ${CLUSTER_NAME}"
write_results_json "${RESULTS_DIR}/combined_results.json"

# ── Verdict ───────────────────────────────────────────────────────────────────
echo
if [[ $FAIL_COUNT -eq 0 && $SKIP_COUNT -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}  ✓ ALL BENCHMARKS PASSED${NC}"
elif [[ $FAIL_COUNT -eq 0 ]]; then
    echo -e "${YELLOW}${BOLD}  ⚠ ALL RUN BENCHMARKS PASSED (${SKIP_COUNT} skipped)${NC}"
else
    echo -e "${RED}${BOLD}  ✗ ${FAIL_COUNT} BENCHMARK(S) FAILED${NC}"
fi
echo
echo "Full results: ${RESULTS_DIR}"
echo "Combined JSON: ${RESULTS_DIR}/combined_results.json"
echo
