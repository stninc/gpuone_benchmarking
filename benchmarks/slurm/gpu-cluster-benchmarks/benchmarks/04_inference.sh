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
"""vLLM offline throughput benchmark."""
import json, time, os

def main():
    from vllm import LLM, SamplingParams

    model = os.environ.get("INFER_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    tp = int(os.environ.get("INFER_TP", "8"))
    input_len = int(os.environ.get("INFER_INPUT_LEN", "512"))
    output_len = int(os.environ.get("INFER_OUTPUT_LEN", "512"))
    num_prompts = int(os.environ.get("INFER_NUM_PROMPTS", "500"))
    outdir = os.environ.get("INFER_DIR", "/tmp")

    print(f"Loading model {model} with TP={tp}...")
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
    print("Warmup (5 prompts)...")
    _ = llm.generate(prompts[:5], sampling_params)

    # Timed run
    print(f"Running {num_prompts} prompts...")
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_tokens = total_input_tokens + total_output_tokens

    results = {
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

    print(f"\nResults:")
    print(f"  Elapsed: {results['elapsed_s']}s")
    print(f"  Output throughput: {results['throughput_tok_per_s']} tok/s")
    print(f"  Total throughput: {results['total_tok_per_s']} tok/s")
    print(f"  Avg latency/prompt: {results['avg_latency_per_prompt_ms']} ms")

    outfile = os.path.join(outdir, "throughput_results.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {outfile}")

if __name__ == "__main__":
    main()
PYEOF

cat > "${INFER_DIR}/bench_latency.py" <<'PYEOF'
#!/usr/bin/env python3
"""vLLM single-request latency benchmark."""
import json, time, os

def main():
    from vllm import LLM, SamplingParams

    model = os.environ.get("INFER_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    tp = int(os.environ.get("INFER_TP", "8"))
    input_len = int(os.environ.get("INFER_INPUT_LEN", "512"))
    output_len = int(os.environ.get("INFER_OUTPUT_LEN", "512"))
    outdir = os.environ.get("INFER_DIR", "/tmp")

    print(f"Loading model {model} with TP={tp}...")
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

    print(f"\nLatency Results:")
    print(f"  Avg: {results['avg_latency_ms']} ms")
    print(f"  P50: {results['p50_latency_ms']} ms")
    print(f"  P99: {results['p99_latency_ms']} ms")
    print(f"  Avg ITL: {results['avg_inter_token_latency_ms']} ms")

    outfile = os.path.join(outdir, "latency_results.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {outfile}")

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
info "Submitting inference benchmark (${INFERENCE_NODES} node, TP=${INFER_TP}, model=${INFER_MODEL})..."

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

# ── Parse results ─────────────────────────────────────────────────────────────
parse_inference_results() {
    header "Benchmark 04 — Inference Results"

    # Throughput
    local tp_file="${INFER_DIR}/throughput_results.json"
    if [[ -f "$tp_file" ]]; then
        local tps model elapsed num_prompts
        tps=$(jq -r '.throughput_tok_per_s // .total_tok_per_s // empty' "$tp_file")
        model=$(jq -r '.model // "unknown"' "$tp_file")
        elapsed=$(jq -r '.elapsed_s // empty' "$tp_file")
        num_prompts=$(jq -r '.num_prompts // empty' "$tp_file")

        info "Model: ${model}, Prompts: ${num_prompts}, Elapsed: ${elapsed}s"

        if [[ -n "$tps" ]]; then
            check_threshold "Inference throughput (${model})" "$tps" "$INFERENCE_THROUGHPUT_MIN_TOKS" "tok/s"
        else
            record_skip "Inference throughput" "not reported"
        fi
    else
        record_skip "Inference throughput" "results file not found"
    fi

    # Latency
    local lat_file="${INFER_DIR}/latency_results.json"
    if [[ -f "$lat_file" ]]; then
        local avg_lat p99_lat avg_itl
        avg_lat=$(jq -r '.avg_latency_ms // empty' "$lat_file")
        p99_lat=$(jq -r '.p99_latency_ms // empty' "$lat_file")
        avg_itl=$(jq -r '.avg_inter_token_latency_ms // empty' "$lat_file")

        [[ -n "$avg_lat" ]] && info "  Avg latency: ${avg_lat} ms"
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
