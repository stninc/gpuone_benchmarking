#!/usr/bin/env bash
# ==============================================================================
# Benchmark 04 — Inference Throughput (vLLM)
#
# Runs inference benchmarks using vLLM. Tests:
#   - Offline throughput (tokens/sec at configured batch)
#   - Latency (single-request, TTFT + inter-token latency)
#
# Default: Qwen/Qwen2.5-14B-Instruct (non-gated, 40 heads, TP=8 compatible)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/common.sh"
source "${SCRIPT_DIR}/../configs/cluster.conf"

header "Benchmark 04 — Inference Throughput (vLLM)"

INFER_DIR="${RESULTS_DIR}/04_inference"
mkdir -p "${INFER_DIR}"

# Generate sourceable NCCL env file from cluster.conf vars currently in env.
# The container runner sources this AFTER nuking image-baked NCCL_*/UCX_* vars.
BENCH_NCCL_ENV_FILE="${INFER_DIR}/nccl_env.sh"
write_nccl_env_file "${BENCH_NCCL_ENV_FILE}"

# ── Configurable parameters ───────────────────────────────────────────────────
INFER_MODEL="${INFER_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
INFER_TP="${INFER_TP:-${GPUS_PER_NODE}}"
INFER_INPUT_LEN="${INFER_INPUT_LEN:-512}"
INFER_OUTPUT_LEN="${INFER_OUTPUT_LEN:-512}"
INFER_NUM_PROMPTS="${INFER_NUM_PROMPTS:-500}"
HF_TOKEN="${HF_TOKEN:-}"

# ── Write Python benchmark scripts to files ───────────────────────────────────
# vLLM with TP>1 uses multiprocessing.spawn which needs a real .py file,
# not stdin. Writing to files that get mounted into the container.

cat > "${INFER_DIR}/bench_throughput.py" <<'PYEOF'
#!/usr/bin/env python3
"""vLLM offline throughput benchmark — runs as one replica per node.

When launched as a multi-node SLURM job (ntasks-per-node=1, N nodes), each
node runs an independent vLLM instance with local tensor parallelism. This
measures replica-level throughput; the aggregate across replicas is computed
by the surrounding shell script.

Each replica writes its own JSON file keyed by SLURM_NODEID so the aggregator
can find them all.
"""
import json, time, os, socket

def main():
    from vllm import LLM, SamplingParams

    model = os.environ.get("INFER_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    tp = int(os.environ.get("INFER_TP", "8"))
    input_len = int(os.environ.get("INFER_INPUT_LEN", "512"))
    output_len = int(os.environ.get("INFER_OUTPUT_LEN", "512"))
    num_prompts = int(os.environ.get("INFER_NUM_PROMPTS", "500"))
    outdir = os.environ.get("INFER_DIR", "/tmp")
    node_id = int(os.environ.get("SLURM_NODEID", "0"))
    nnodes = int(os.environ.get("SLURM_NNODES", "1"))
    hostname = socket.gethostname()

    print(f"[replica {node_id}/{nnodes} on {hostname}] Loading {model} with TP={tp}...", flush=True)
    llm = LLM(model=model, tensor_parallel_size=tp, dtype="auto")

    # Generate dummy prompts of fixed input length
    tokenizer = llm.get_tokenizer()
    dummy_ids = [1] * input_len
    dummy_text = tokenizer.decode(dummy_ids, skip_special_tokens=True)
    prompts = [dummy_text] * num_prompts

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True,
    )

    # Warmup
    print(f"[replica {node_id}] Warmup (5 prompts)...", flush=True)
    _ = llm.generate(prompts[:5], sampling_params)

    # Timed run
    print(f"[replica {node_id}] Running {num_prompts} prompts...", flush=True)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_tokens = total_input_tokens + total_output_tokens

    results = {
        "replica_id": node_id,
        "num_replicas": nnodes,
        "hostname": hostname,
        "model": model,
        "tensor_parallel": tp,
        "num_prompts": num_prompts,
        "input_len": input_len,
        "output_len": output_len,
        "elapsed_s": round(elapsed, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "throughput_tok_per_s": round(total_output_tokens / elapsed, 1),
        "total_tok_per_s": round(total_tokens / elapsed, 1),
        "avg_latency_per_prompt_ms": round(elapsed / num_prompts * 1000, 1),
    }

    print(f"\n[replica {node_id}] Results:", flush=True)
    print(f"  Elapsed: {results['elapsed_s']}s", flush=True)
    print(f"  Output throughput: {results['throughput_tok_per_s']} tok/s", flush=True)
    print(f"  Total throughput: {results['total_tok_per_s']} tok/s", flush=True)
    print(f"  Avg latency/prompt: {results['avg_latency_per_prompt_ms']} ms", flush=True)

    # Per-replica result file so aggregator can find all N of them
    outfile = os.path.join(outdir, f"throughput_results_replica{node_id}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[replica {node_id}] Results written to {outfile}", flush=True)

if __name__ == "__main__":
    main()
PYEOF

cat > "${INFER_DIR}/bench_latency.py" <<'PYEOF'
#!/usr/bin/env python3
"""vLLM single-request latency benchmark.

Latency is a per-replica property — running the same measurement on N replicas
gives N nearly-identical numbers. We only run this on replica 0; other ranks
exit immediately.
"""
import json, time, os, socket, sys

def main():
    node_id = int(os.environ.get("SLURM_NODEID", "0"))
    if node_id != 0:
        print(f"[replica {node_id}] skipping latency benchmark (only runs on replica 0)", flush=True)
        return

    from vllm import LLM, SamplingParams

    model = os.environ.get("INFER_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    tp = int(os.environ.get("INFER_TP", "8"))
    input_len = int(os.environ.get("INFER_INPUT_LEN", "512"))
    output_len = int(os.environ.get("INFER_OUTPUT_LEN", "512"))
    outdir = os.environ.get("INFER_DIR", "/tmp")
    hostname = socket.gethostname()

    print(f"[replica 0 on {hostname}] Loading {model} with TP={tp}...", flush=True)
    llm = LLM(model=model, tensor_parallel_size=tp, dtype="auto")

    tokenizer = llm.get_tokenizer()
    dummy_ids = [1] * input_len
    dummy_text = tokenizer.decode(dummy_ids, skip_special_tokens=True)

    params = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)

    # Warmup
    _ = llm.generate([dummy_text], params)

    # Measure latency over multiple runs
    latencies = []
    num_runs = 20
    for i in range(num_runs):
        start = time.perf_counter()
        out = llm.generate([dummy_text], params)
        elapsed = time.perf_counter() - start
        output_tokens = len(out[0].outputs[0].token_ids)
        latencies.append({
            "elapsed_ms": round(elapsed * 1000, 2),
            "output_tokens": output_tokens,
            "itl_ms": round(elapsed / output_tokens * 1000, 3) if output_tokens > 0 else None,
        })

    avg_latency = sum(l["elapsed_ms"] for l in latencies) / len(latencies)
    avg_itl = sum(l["itl_ms"] for l in latencies if l["itl_ms"]) / len([l for l in latencies if l["itl_ms"]])
    p50_lat = sorted(l["elapsed_ms"] for l in latencies)[len(latencies) // 2]
    p99_lat = sorted(l["elapsed_ms"] for l in latencies)[int(len(latencies) * 0.99)]

    results = {
        "model": model,
        "tensor_parallel": tp,
        "input_len": input_len,
        "output_len": output_len,
        "num_runs": num_runs,
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p99_latency_ms": round(p99_lat, 2),
        "avg_inter_token_latency_ms": round(avg_itl, 3),
    }

    print(f"\nLatency Results:", flush=True)
    print(f"  Avg: {results['avg_latency_ms']} ms", flush=True)
    print(f"  P50: {results['p50_latency_ms']} ms", flush=True)
    print(f"  P99: {results['p99_latency_ms']} ms", flush=True)
    print(f"  Avg ITL: {results['avg_inter_token_latency_ms']} ms", flush=True)

    outfile = os.path.join(outdir, "latency_results.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {outfile}", flush=True)

if __name__ == "__main__":
    main()
PYEOF

# ── Bash runner (runs inside container) ───────────────────────────────────────
cat > "${INFER_DIR}/infer_runner.sh" <<'RUNEOF'
#!/bin/bash
set -e

# ── Step 1: Nuke image-baked NCCL/UCX env vars ────────────────────────────────
for var in $(env | grep -oP '^(NCCL_|UCX_|NVSHMEM_|HPCX_|OMPI_MCA_)[^=]*'); do
    unset "$var"
done

# ── Step 2: Re-apply user's NCCL config from cluster.conf ────────────────────
if [[ -z "${BENCH_NCCL_ENV_FILE:-}" ]]; then
    echo "WARNING: BENCH_NCCL_ENV_FILE not set in env — NCCL config from cluster.conf will NOT be applied" >&2
elif [[ ! -f "${BENCH_NCCL_ENV_FILE}" ]]; then
    echo "WARNING: BENCH_NCCL_ENV_FILE=${BENCH_NCCL_ENV_FILE} does not exist on this host — check container mounts" >&2
else
    echo "Sourcing NCCL env file: ${BENCH_NCCL_ENV_FILE}"
    # shellcheck disable=SC1090
    source "${BENCH_NCCL_ENV_FILE}"
fi

# ── Step 3: Defensive defaults ───────────────────────────────────────────────
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}

echo "=== Inference Benchmark ==="
echo "Hostname: $(hostname)"
echo "Model: ${INFER_MODEL}"
echo "TP: ${INFER_TP}"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo ""
echo "=== Effective NCCL env ==="
env | grep -E '^(NCCL_|UCX_)' | sort
echo ""

echo "=== RDMA visibility check ==="
echo "/dev/infiniband contents:"
ls /dev/infiniband/ 2>&1 | head -10 | sed 's/^/  /'
echo "ibv_devices output:"
(command -v ibv_devices >/dev/null && ibv_devices 2>&1 || echo "  ibv_devices not installed") | head -10 | sed 's/^/  /'
echo ""

# Install vLLM if not present
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install --quiet vllm 2>&1 | tail -5
fi
echo "vLLM version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"

# HF token
[[ -n "${HF_TOKEN:-}" ]] && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# ── Test 1: Throughput ────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Test 1: Offline Throughput"
echo "════════════════════════════════════════════════════════════════"
python3 "${INFER_DIR}/bench_throughput.py" 2>&1 | tee "${INFER_DIR}/throughput.log"

# ── Test 2: Latency ──────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Test 2: Single-Request Latency"
echo "════════════════════════════════════════════════════════════════"
python3 "${INFER_DIR}/bench_latency.py" 2>&1 | tee "${INFER_DIR}/latency.log"

echo ""
echo "Inference benchmarks complete."
RUNEOF

chmod +x "${INFER_DIR}/infer_runner.sh"

# ── Submit job ────────────────────────────────────────────────────────────────
TOTAL_INFER_GPUS=$((INFERENCE_NODES * INFER_TP))
if [[ "${INFERENCE_NODES}" -gt 1 ]]; then
    info "Submitting inference benchmark (${INFERENCE_NODES} replicas × TP=${INFER_TP} = ${TOTAL_INFER_GPUS} GPUs, model=${INFER_MODEL})..."
    info "  NOTE: Replicas run independently — this measures aggregate fleet throughput, not inter-node NCCL."
else
    info "Submitting inference benchmark (1 replica, TP=${INFER_TP}, model=${INFER_MODEL})..."
fi

HF_EXPORT=""
[[ -n "${HF_TOKEN:-}" ]] && HF_EXPORT=",HF_TOKEN=${HF_TOKEN}"

INFER_JOB=$(sbatch --parsable \
    $(build_slurm_args) \
    --job-name="bench-04-inference" \
    --nodes="${INFERENCE_NODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --time=02:00:00 \
    --output="${INFER_DIR}/slurm-%j.out" \
    --export=ALL,INFER_DIR="${INFER_DIR}",BENCH_NCCL_ENV_FILE="${BENCH_NCCL_ENV_FILE}",INFER_MODEL="${INFER_MODEL}",INFER_TP="${INFER_TP}",INFER_INPUT_LEN="${INFER_INPUT_LEN}",INFER_OUTPUT_LEN="${INFER_OUTPUT_LEN}",INFER_NUM_PROMPTS="${INFER_NUM_PROMPTS}"${HF_EXPORT} \
    --wrap="srun ${PYXIS_COMMON} \
        --container-image=${PYTORCH_IMAGE} \
        --container-mounts=${PYXIS_MOUNTS},${INFER_DIR}:${INFER_DIR} \
        bash ${INFER_DIR}/infer_runner.sh" \
    2>&1) || { fail "Failed to submit inference job"; INFER_JOB=""; }

[[ -n "$INFER_JOB" ]] && info "Inference job submitted: ${INFER_JOB}"
echo "${INFER_JOB:-none}" > "${INFER_DIR}/job_id"

# ── Aggregate per-replica results into a single summary JSON ─────────────────
aggregate_replica_results() {
    local indir="$1" nnodes="$2"
    local replica_files=()
    local i
    for (( i=0; i<nnodes; i++ )); do
        local f="${indir}/throughput_results_replica${i}.json"
        if [[ -f "$f" ]]; then
            replica_files+=("$f")
        else
            warn "Missing per-replica result file: $f"
        fi
    done

    if [[ ${#replica_files[@]} -eq 0 ]]; then
        warn "No replica result files found — nothing to aggregate"
        return 1
    fi

    # Sum output-token throughput across replicas, take first replica for
    # scalar metadata. All replicas run the same prompts, so we sum prompts
    # and tokens. Elapsed is max (slowest replica gates aggregate throughput).
    local agg_file="${indir}/throughput_results.json"
    jq -s '
        {
            model:                 .[0].model,
            num_replicas:          length,
            tensor_parallel_per_replica: .[0].tensor_parallel,
            total_gpus:            (length * .[0].tensor_parallel),
            num_prompts_per_replica: .[0].num_prompts,
            total_num_prompts:     (length * .[0].num_prompts),
            input_len:             .[0].input_len,
            output_len:            .[0].output_len,
            elapsed_s_max:         (map(.elapsed_s) | max),
            elapsed_s_min:         (map(.elapsed_s) | min),
            per_replica_throughput_tok_per_s: map(.throughput_tok_per_s),
            per_replica_total_tok_per_s:      map(.total_tok_per_s),
            throughput_tok_per_s:  (map(.throughput_tok_per_s) | add),
            total_tok_per_s:       (map(.total_tok_per_s) | add),
            note: "Multi-replica aggregate: N independent vLLM instances, each with local TP. No inter-node NCCL; sum is fleet throughput."
        }
    ' "${replica_files[@]}" > "$agg_file"

    info "Aggregated ${#replica_files[@]} replica(s) → $agg_file"
}

# ── Parse results ─────────────────────────────────────────────────────────────
parse_inference_results() {
    header "Benchmark 04 — Inference Results"

    # Aggregate per-replica results if this was a multi-replica run
    aggregate_replica_results "${INFER_DIR}" "${INFERENCE_NODES}" || true

    # Throughput
    local tp_file="${INFER_DIR}/throughput_results.json"
    if [[ -f "$tp_file" ]]; then
        local tps model elapsed num_prompts num_replicas tp_per_replica total_gpus
        tps=$(jq -r '.throughput_tok_per_s // .total_tok_per_s // empty' "$tp_file")
        model=$(jq -r '.model // "unknown"' "$tp_file")
        elapsed=$(jq -r '.elapsed_s_max // .elapsed_s // empty' "$tp_file")
        num_prompts=$(jq -r '.total_num_prompts // .num_prompts // empty' "$tp_file")
        num_replicas=$(jq -r '.num_replicas // 1' "$tp_file")
        tp_per_replica=$(jq -r '.tensor_parallel_per_replica // .tensor_parallel // empty' "$tp_file")
        total_gpus=$(jq -r '.total_gpus // empty' "$tp_file")

        if [[ "$num_replicas" -gt 1 ]]; then
            info "Model: ${model}, Replicas: ${num_replicas} × TP=${tp_per_replica} (${total_gpus} GPUs total)"
            info "Prompts: ${num_prompts} total across replicas, Slowest replica elapsed: ${elapsed}s"
            info "NOTE: Replicas run independently (no inter-node NCCL). Throughput is aggregate fleet capacity."
        else
            info "Model: ${model}, Prompts: ${num_prompts}, Elapsed: ${elapsed}s (single replica, TP=${tp_per_replica})"
        fi

        if [[ -n "$tps" ]]; then
            local label="Inference throughput (${model}"
            if [[ "$num_replicas" -gt 1 ]]; then
                label="${label}, ${num_replicas}× TP=${tp_per_replica} replicas"
            else
                label="${label}, TP=${tp_per_replica}"
            fi
            label="${label})"
            check_threshold "$label" "$tps" "$INFERENCE_THROUGHPUT_MIN_TOKS" "tok/s"
        else
            record_skip "Inference throughput" "not reported"
        fi
    else
        record_skip "Inference throughput" "results file not found"
    fi

    # Latency (measured on replica 0 only)
    local lat_file="${INFER_DIR}/latency_results.json"
    if [[ -f "$lat_file" ]]; then
        local avg_lat p99_lat avg_itl
        avg_lat=$(jq -r '.avg_latency_ms // empty' "$lat_file")
        p99_lat=$(jq -r '.p99_latency_ms // empty' "$lat_file")
        avg_itl=$(jq -r '.avg_inter_token_latency_ms // empty' "$lat_file")

        [[ -n "$avg_lat" ]] && info "  Avg latency: ${avg_lat} ms (single replica, representative)"
        [[ -n "$p99_lat" ]] && info "  P99 latency: ${p99_lat} ms"
        [[ -n "$avg_itl" ]] && info "  Avg ITL: ${avg_itl} ms"
    fi
}

# ── Direct execution ──────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ -n "${INFER_JOB:-}" ]] && wait_for_job "$INFER_JOB" 7200; then
        parse_inference_results
    fi
    print_summary "04 — Inference Throughput"
    write_results_json "${INFER_DIR}/results.json"
fi
