#!/usr/bin/env bash
# ==============================================================================
# Benchmark 03 — Multi-Node Training MFU (FSDP)
#
# Runs a training benchmark using PyTorch FSDP with any HuggingFace model.
# Measures Model FLOPS Utilization (MFU), step time, and tokens/sec.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/common.sh"

# Only source default config if this script is executed directly.
# When run via run_all.sh, config is already loaded and should not be overridden.
if [[ -z "${CLUSTER_NAME:-}" ]]; then
    source "${SCRIPT_DIR}/../configs/cluster.conf"
fi

header "Benchmark 03 — Multi-Node Training MFU"

TRAIN_DIR="${RESULTS_DIR}/03_training"
mkdir -p "${TRAIN_DIR}"

# ── Configurable training parameters ──────────────────────────────────────────
TRAIN_MODEL="${TRAIN_MODEL:-Qwen/Qwen2.5-7B}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
TRAIN_WARMUP="${TRAIN_WARMUP:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-1}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-4096}"
TRAIN_TRUST_REMOTE_CODE="${TRAIN_TRUST_REMOTE_CODE:-0}"
TOTAL_GPUS=$((TRAIN_NODES * GPUS_PER_NODE))
MASTER_PORT="${MASTER_PORT:-29500}"

# ── Embedded Python training script ──────────────────────────────────────────
cat > "${TRAIN_DIR}/fsdp_train.py" <<'PYEOF'
#!/usr/bin/env python3
"""
FSDP Training MFU Benchmark

Runs training steps on synthetic data using PyTorch FSDP with any HuggingFace
causal LM model. Reports MFU, step time, and tokens/sec.
"""
import json
import os
import time
import datetime

def main():
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))

    print(f"[rank {global_rank}] Starting, LOCAL_RANK={local_rank}, PID={os.getpid()}", flush=True)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[rank {global_rank}] Calling dist.init_process_group('nccl')...", flush=True)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=5),
        device_id=device,
    )
    world_size = dist.get_world_size()
    print(f"[rank {global_rank}] NCCL init complete. World size: {world_size}", flush=True)

    model_name = os.environ.get("TRAIN_MODEL", "Qwen/Qwen2.5-7B")
    batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "2"))
    grad_accum = int(os.environ.get("TRAIN_GRAD_ACCUM", "1"))
    seq_len = int(os.environ.get("TRAIN_SEQ_LEN", "4096"))
    num_steps = int(os.environ.get("TRAIN_STEPS", "100"))
    warmup_steps = int(os.environ.get("TRAIN_WARMUP", "10"))
    hf_token = os.environ.get("HF_TOKEN", None) or None
    train_dir = os.environ.get("TRAIN_DIR", "/tmp")
    trust_remote_code = os.environ.get("TRAIN_TRUST_REMOTE_CODE", "0") == "1"

    if global_rank == 0:
        print("\n=== FSDP Training Benchmark ===", flush=True)
        print(f"Model:          {model_name}", flush=True)
        print(f"World size:     {world_size} GPUs", flush=True)
        print(f"Batch size/GPU: {batch_size}", flush=True)
        print(f"Grad accum:     {grad_accum}", flush=True)
        print(f"Effective batch:{batch_size * grad_accum}", flush=True)
        print(f"Seq len:        {seq_len}", flush=True)
        print(f"Steps:          {num_steps} (warmup: {warmup_steps})", flush=True)
        print("Precision:      BF16 mixed\n", flush=True)

    from transformers import AutoConfig, AutoModelForCausalLM, CONFIG_MAPPING
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision

    # ── Load config on rank 0 only, write to shared file, then barrier ───────
    config_path = os.path.join(train_dir, "hf_config.json")

    if global_rank == 0:
        print(f"[rank {global_rank}] before AutoConfig.from_pretrained()", flush=True)
        config_dict = AutoConfig.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        ).to_dict()
        with open(config_path, "w") as f:
            json.dump(config_dict, f)
        print(f"[rank {global_rank}] wrote config to {config_path}", flush=True)

    print(f"[rank {global_rank}] before config-file barrier", flush=True)
    dist.barrier()
    print(f"[rank {global_rank}] after config-file barrier", flush=True)

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    model_type = config_dict.get("model_type")
    if not model_type:
        raise RuntimeError(f"model_type missing from config file: {config_path}")

    config_cls = CONFIG_MAPPING[model_type]
    config = config_cls.from_dict(config_dict)
    vocab_size = config.vocab_size

    # ── Symmetric model construction on all ranks ─────────────────────────────
    print(f"[rank {global_rank}] building model on CPU", flush=True)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    num_params = sum(p.numel() for p in model.parameters())

    if global_rank == 0:
        print(f"Model params:   {num_params / 1e9:.2f}B", flush=True)
        print(f"Vocab size:     {vocab_size}", flush=True)
        print("Gradient checkpointing: enabled", flush=True)

    print(f"[rank {global_rank}] moving model to {device}", flush=True)
    model = model.to(device)

    print(f"[rank {global_rank}] before post-model barrier", flush=True)
    dist.barrier()
    print(f"[rank {global_rank}] after post-model barrier", flush=True)

    # ── Wrap in FSDP ──────────────────────────────────────────────────────────
    if global_rank == 0:
        print("Initializing FSDP...", flush=True)

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    print(f"[rank {global_rank}] wrapping with FSDP", flush=True)
    model = FSDP(
        model,
        mixed_precision=bf16_policy,
        use_orig_params=True,
        sync_module_states=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    if global_rank == 0:
        print("FSDP initialized\n", flush=True)

    # ── Quick NCCL smoke test ─────────────────────────────────────────────────
    test_tensor = torch.ones(1, device=device)
    dist.all_reduce(test_tensor)
    if global_rank == 0:
        print(f"NCCL smoke test passed (sum={test_tensor.item()})", flush=True)

    # ── Synthetic data ────────────────────────────────────────────────────────
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # ── Training loop ─────────────────────────────────────────────────────────
    step_times = []
    mfu_values = []

    if global_rank == 0:
        print("Starting training loop...", flush=True)

    for step in range(num_steps + warmup_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        loss_for_log = None
        for micro in range(grad_accum):
            output = model(input_ids=input_ids, labels=input_ids)
            loss = output.loss
            loss_for_log = loss.detach()
            (loss / grad_accum).backward()

        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if step >= warmup_steps:
            step_times.append(elapsed * 1000)

            tokens_per_step = batch_size * seq_len * world_size * grad_accum
            flops_per_step = 6 * num_params * batch_size * seq_len * grad_accum
            tflops_per_gpu = flops_per_step / elapsed / 1e12

            gpu_name = torch.cuda.get_device_name(0).lower()
            if "b300" in gpu_name or "b200" in gpu_name:
                peak_tflops = 2250
            elif "h200" in gpu_name:
                peak_tflops = 990
            elif "h100" in gpu_name:
                peak_tflops = 990
            elif "a100" in gpu_name:
                peak_tflops = 312
            else:
                peak_tflops = float(os.environ.get("GPU_PEAK_TFLOPS", "1000"))

            mfu = tflops_per_gpu / peak_tflops * 100
            mfu_values.append(mfu)

            if global_rank == 0 and (step - warmup_steps) % 10 == 0:
                tps = tokens_per_step / elapsed
                print(
                    f"  step {step - warmup_steps:4d}/{num_steps} | "
                    f"loss {loss_for_log.item():.4f} | "
                    f"step_time {elapsed*1000:.1f}ms | "
                    f"MFU {mfu:.1f}% | "
                    f"tflops/gpu {tflops_per_gpu:.1f} | "
                    f"tokens/sec {tps:.0f}",
                    flush=True,
                )

    # ── Results ───────────────────────────────────────────────────────────────
    if global_rank == 0:
        skip = max(1, len(step_times) // 10)
        steady_times = step_times[skip:]
        steady_mfu = mfu_values[skip:]

        avg_step_ms = sum(steady_times) / len(steady_times) if steady_times else 0
        avg_mfu = sum(steady_mfu) / len(steady_mfu) if steady_mfu else 0
        tokens_per_step = batch_size * seq_len * world_size * grad_accum
        avg_tps = tokens_per_step / (avg_step_ms / 1000) if avg_step_ms > 0 else 0

        print(f"\n=== Results (steady state, {len(steady_times)} steps) ===", flush=True)
        print(f"  Avg step time:  {avg_step_ms:.1f} ms", flush=True)
        print(f"  Avg MFU:        {avg_mfu:.1f}%", flush=True)
        print(f"  Avg tokens/sec: {avg_tps:.0f}", flush=True)
        print(f"  Model:          {model_name} ({num_params/1e9:.2f}B params)", flush=True)
        print(f"  GPUs:           {world_size}\n", flush=True)

        results = {
            "model": model_name,
            "num_params_B": round(num_params / 1e9, 2),
            "nodes": world_size // int(os.environ.get("GPUS_PER_NODE", 8)),
            "gpus_per_node": int(os.environ.get("GPUS_PER_NODE", 8)),
            "total_gpus": world_size,
            "batch_size_per_gpu": batch_size,
            "grad_accum": grad_accum,
            "effective_batch_size_per_gpu": batch_size * grad_accum,
            "seq_len": seq_len,
            "steps": num_steps,
            "warmup_steps": warmup_steps,
            "precision": "bf16",
            "mfu_pct": round(avg_mfu, 2),
            "avg_step_time_ms": round(avg_step_ms, 2),
            "tokens_per_second": round(avg_tps, 1),
            "step_times_ms": [round(t, 2) for t in step_times],
        }

        outfile = os.path.join(train_dir, "training_metrics.json")
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {outfile}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PYEOF

chmod +x "${TRAIN_DIR}/fsdp_train.py"

# ── Training runner script (runs inside container) ───────────────────────────
cat > "${TRAIN_DIR}/train_runner.sh" <<'TRAINEOF'
#!/bin/bash
set -euo pipefail

for var in $(env | grep -oP '^(NCCL_|UCX_|NVSHMEM_|HPCX_|OMPI_MCA_)[^=]*'); do
    unset "$var"
done
unset GLOO_SOCKET_IFNAME 2>/dev/null || true

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    export NCCL_SOCKET_IFNAME
    export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "=== Training Benchmark ==="
echo "Hostname:  $(hostname)"
echo "Model:     ${TRAIN_MODEL}"
echo "Nodes:     ${SLURM_NNODES:-1}"
echo "GPUs/node: ${GPUS_PER_NODE:-8}"
echo ""
env | grep -E '^(NCCL_|GLOO_|PYTORCH_CUDA_ALLOC_CONF)' | sort || true
echo ""

if ! python3 -c "import transformers" >/dev/null 2>&1; then
    echo "transformers not found; installing..." >&2
    python3 -m pip install --no-input --quiet "transformers>=4.40" "huggingface_hub>=0.23"
fi

python3 -c "import transformers" >/dev/null 2>&1

export MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be set by host-side batch script}"
export MASTER_PORT="${MASTER_PORT:-29500}"

NNODES="${SLURM_NNODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"
NPROC="${GPUS_PER_NODE:-8}"

echo "Distributed: MASTER=${MASTER_ADDR}:${MASTER_PORT}, NNODES=${NNODES}, NODE_RANK=${NODE_RANK}"
echo ""

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_DIR}/fsdp_train.py" \
    2>&1 | tee "${TRAIN_DIR}/train_node${NODE_RANK}.log"

echo "Training complete on node ${NODE_RANK}"
TRAINEOF

chmod +x "${TRAIN_DIR}/train_runner.sh"

# ── Host-side batch script (runs inside Slurm allocation, outside container) ─
cat > "${TRAIN_DIR}/train_job.sh" <<'JOBEOF'
#!/bin/bash
set -euo pipefail

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "Host-side resolved MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"

srun ${PYXIS_COMMON} \
    --container-image="${PYTORCH_IMAGE}" \
    --container-mounts="${TRAIN_DIR}:${TRAIN_DIR}" \
    --export=ALL,MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
    bash "${TRAIN_DIR}/train_runner.sh"
JOBEOF

chmod +x "${TRAIN_DIR}/train_job.sh"

# ── Submit training job ───────────────────────────────────────────────────────
info "Submitting training benchmark (${TRAIN_NODES} nodes, ${TOTAL_GPUS} GPUs, model=${TRAIN_MODEL})..."

TRAIN_JOB=$(sbatch --parsable \
    $(build_slurm_args) \
    --job-name="bench-03-train" \
    --nodes="${TRAIN_NODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --time=02:00:00 \
    --output="${TRAIN_DIR}/slurm-%j.out" \
    --export=ALL,\
TRAIN_DIR="${TRAIN_DIR}",\
TRAIN_MODEL="${TRAIN_MODEL}",\
TRAIN_STEPS="${TRAIN_STEPS}",\
TRAIN_WARMUP="${TRAIN_WARMUP}",\
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}",\
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM}",\
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN}",\
TRAIN_TRUST_REMOTE_CODE="${TRAIN_TRUST_REMOTE_CODE}",\
GPUS_PER_NODE="${GPUS_PER_NODE}",\
HF_TOKEN="${HF_TOKEN:-}",\
MASTER_PORT="${MASTER_PORT}",\
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}",\
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}",\
NCCL_DEBUG="${NCCL_DEBUG:-WARN}",\
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}",\
PYXIS_COMMON="${PYXIS_COMMON}",\
PYTORCH_IMAGE="${PYTORCH_IMAGE}" \
    "${TRAIN_DIR}/train_job.sh" \
    2>&1) || { fail "Failed to submit training job"; TRAIN_JOB=""; }

[[ -n "$TRAIN_JOB" ]] && info "Training job submitted: ${TRAIN_JOB}"
echo "${TRAIN_JOB:-none}" > "${TRAIN_DIR}/job_id"

# ── Parse results ─────────────────────────────────────────────────────────────
parse_training_results() {
    header "Benchmark 03 — Training MFU Results"

    local metrics_file="${TRAIN_DIR}/training_metrics.json"
    if [[ ! -f "$metrics_file" ]]; then
        record_skip "Training MFU" "metrics file not found — check SLURM output"
        return
    fi

    local mfu avg_step tps model num_params
    model=$(jq -r '.model' "$metrics_file")
    num_params=$(jq -r '.num_params_B' "$metrics_file")
    mfu=$(jq -r '.mfu_pct // empty' "$metrics_file")
    avg_step=$(jq -r '.avg_step_time_ms // empty' "$metrics_file")
    tps=$(jq -r '.tokens_per_second // empty' "$metrics_file")

    info "Model: ${model} (${num_params}B params), Nodes: ${TRAIN_NODES}, Total GPUs: ${TOTAL_GPUS}"

    if [[ -n "$mfu" && "$mfu" != "null" ]]; then
        check_threshold "Training MFU (${model}, ${TOTAL_GPUS}GPU)" "$mfu" "$TRAIN_MFU_MIN_PCT" "%"
    else
        record_skip "Training MFU" "not reported — check training logs"
    fi

    [[ -n "$avg_step" && "$avg_step" != "null" ]] && info "  Avg step time: ${avg_step} ms"
    [[ -n "$tps" && "$tps" != "null" ]] && info "  Tokens/sec: ${tps}"
}

# ── Direct execution ──────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ -n "${TRAIN_JOB:-}" ]] && wait_for_job "$TRAIN_JOB" 7200; then
        parse_training_results
    fi
    print_summary "03 — Training MFU"
    write_results_json "${TRAIN_DIR}/results.json"
fi
