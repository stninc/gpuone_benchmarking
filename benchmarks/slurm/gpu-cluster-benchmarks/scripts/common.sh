#!/usr/bin/env bash
# ==============================================================================
# GPU Cluster Benchmark Suite — Common Functions
# ==============================================================================

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ── Logging ───────────────────────────────────────────────────────────────────
_ts() { date '+%Y-%m-%d %H:%M:%S'; }
info()    { echo -e "$(_ts) ${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "$(_ts) ${GREEN}[PASS]${NC}  $*"; }
warn()    { echo -e "$(_ts) ${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "$(_ts) ${RED}[FAIL]${NC}  $*"; }
header()  { echo -e "\n${BOLD}════════════════════════════════════════════════════════════════${NC}"; echo -e "${BOLD}  $*${NC}"; echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}\n"; }

# ── Result Recording ──────────────────────────────────────────────────────────
PASS_COUNT=0; FAIL_COUNT=0; SKIP_COUNT=0
declare -a RESULTS=()

record_pass() {
    ((PASS_COUNT++)) || true
    RESULTS+=("PASS|$1|$2|$3")    # status|test_name|value|threshold
    success "$1: $2 (threshold: $3)"
}

record_fail() {
    ((FAIL_COUNT++)) || true
    RESULTS+=("FAIL|$1|$2|$3")
    fail "$1: $2 (threshold: $3)"
}

record_skip() {
    ((SKIP_COUNT++)) || true
    RESULTS+=("SKIP|$1|${2:-N/A}|${3:-N/A}")
    warn "$1: SKIPPED — ${2:-no reason given}"
}

# ── Threshold Check ───────────────────────────────────────────────────────────
# Usage: check_threshold "test name" measured_value minimum_threshold [unit]
check_threshold() {
    local name="$1" value="$2" threshold="$3" unit="${4:-}"
    local display_val="${value} ${unit}"
    local display_thr="${threshold} ${unit}"

    if awk "BEGIN{exit !($value >= $threshold)}"; then
        record_pass "$name" "$display_val" ">= $display_thr"
    else
        record_fail "$name" "$display_val" ">= $display_thr"
    fi
}

# ── SLURM Helpers ─────────────────────────────────────────────────────────────
build_slurm_args() {
    local args="--partition=${PARTITION}"
    [[ -n "${ACCOUNT:-}" ]] && args+=" --account=${ACCOUNT}"
    echo "$args"
}

# Wait for SLURM job to complete and return exit code
wait_for_job() {
    local job_id="$1" timeout="${2:-3600}"
    local elapsed=0
    info "Waiting for job ${job_id} (timeout: ${timeout}s)..."
    while [[ $elapsed -lt $timeout ]]; do
        local state
        state=$(sacct -j "$job_id" --format=State --noheader -P | head -1 | tr -d ' ')
        case "$state" in
            COMPLETED)  return 0 ;;
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                fail "Job ${job_id} ended with state: ${state}"
                return 1 ;;
        esac
        sleep 10
        ((elapsed += 10))
    done
    fail "Job ${job_id} timed out after ${timeout}s"
    return 1
}

# ── Results Summary ───────────────────────────────────────────────────────────
print_summary() {
    local benchmark_name="${1:-Benchmark}"
    header "${benchmark_name} — Results Summary"

    printf "%-50s %-8s %-25s %-25s\n" "Test" "Status" "Value" "Threshold"
    printf '%.0s─' {1..110}; echo

    for r in "${RESULTS[@]}"; do
        IFS='|' read -r status name value threshold <<< "$r"
        local color
        case "$status" in
            PASS) color="$GREEN" ;;
            FAIL) color="$RED" ;;
            *)    color="$YELLOW" ;;
        esac
        printf "${color}%-50s %-8s${NC} %-25s %-25s\n" "$name" "$status" "$value" "$threshold"
    done

    echo
    printf '%.0s─' {1..110}; echo
    echo -e "Total: ${GREEN}${PASS_COUNT} passed${NC}, ${RED}${FAIL_COUNT} failed${NC}, ${YELLOW}${SKIP_COUNT} skipped${NC}"
    echo
}

# Write results to JSON for downstream processing
write_results_json() {
    local outfile="$1"
    cat > "$outfile" <<JSONEOF
{
  "cluster": "${CLUSTER_NAME}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "gpu_type": "${GPU_TYPE}",
  "summary": {
    "passed": ${PASS_COUNT},
    "failed": ${FAIL_COUNT},
    "skipped": ${SKIP_COUNT}
  },
  "results": [
$(
    local first=true
    for r in "${RESULTS[@]}"; do
        IFS='|' read -r status name value threshold <<< "$r"
        $first || echo ","
        first=false
        printf '    {"test": "%s", "status": "%s", "value": "%s", "threshold": "%s"}' \
            "$name" "$status" "$value" "$threshold"
    done
    echo
)
  ]
}
JSONEOF
    info "Results written to ${outfile}"
}

# ── Prereqs ───────────────────────────────────────────────────────────────────
check_slurm() {
    if ! command -v srun &>/dev/null; then
        fail "srun not found — this script must be run from a SLURM login/submit node"
        exit 1
    fi
    if ! command -v sbatch &>/dev/null; then
        fail "sbatch not found"
        exit 1
    fi
    info "SLURM client tools found"
}
