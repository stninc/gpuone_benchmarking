# gpu-healthcheck.sh

Single-node diagnostic script for GPU HPC servers. Audits the OS, CPUs, memory,
NVIDIA GPUs, InfiniBand / ConnectX HCAs, GPU software stack, storage, container
runtime, and workload manager in one pass, and emits both a colorised terminal
report and a Prometheus textfile-collector `.prom` file.

Default target profile: **8× NVIDIA B300 GPUs · 8× ConnectX-8 HCAs · 256 CPUs ·
Ubuntu**. Edit `EXPECTED_GPUS` / `EXPECTED_NICS` / `EXPECTED_CPUS` at the top
of the script to match other topologies.

## Usage

```bash
sudo bash gpu-healthcheck.sh
```

Tee to a timestamped log for later review:

```bash
sudo bash gpu-healthcheck.sh 2>&1 | tee healthcheck_$(date +%Y%m%d_%H%M%S).log
```

Root is strongly recommended — without it, `dmesg` (XID errors), `dmidecode`,
and some `ibstat` paths are unavailable and those sections are skipped with a
warning.

### Flags

| Flag | Status | Effect |
|------|--------|--------|
| `--json` | reserved | Toggles an internal JSON-mode flag; machine-readable output is not yet wired up. Prometheus output is always produced. |

## What it checks

Each section prints a mix of `[ OK ]`, `[ WARN ]`, `[ FAIL ]`, and `[ INFO ]`
lines, tallied into the summary at the end.

1. **Operating System & Kernel** — distribution, kernel, architecture, server
   model / serial, BIOS, uptime, notable kernel cmdline params (`iommu`,
   `hugepages`, `numa`).
2. **CPU** — model, sockets / cores / threads, NUMA node count, online vs.
   expected CPU count, offline CPU list, `scaling_governor`, hyper-threading.
3. **Memory** — total / available memory, hugepage pool, per-node NUMA memory
   distribution.
4. **NVIDIA GPUs** — driver responsiveness (with `timeout` so a hung driver
   can't stall the script), driver / CUDA version, GPU count, per-GPU
   temperature / power / memory / utilisation / ECC, persistence mode, compute
   mode, XID errors in `dmesg`, retired pages (SBE / DBE / pending), NVLink
   link state and error counters, NVLink topology, clock settings, and
   `nvidia-fabricmanager` service state.
5. **InfiniBand / ConnectX-8 HCAs** — MOFED version, HCA count, per-port state
   and speed (via `ibstat` or `/sys/class/infiniband`), RDMA-capable Ethernet
   links, `mlx*` driver / firmware versions, GPUDirect RDMA module
   (`nv_peer_mem` / `nvidia_peermem`).
6. **Software & Packages** — `nvcc`, NCCL, cuDNN, GDRCopy, apt-held packages,
   `unattended-upgrades` and `apt-daily*` timers (flagged as WARN when active
   — they can cause unexpected reboots on HPC nodes).
7. **System Configuration** — swap, `vm.swappiness`, Transparent Hugepages,
   networking sysctls (`rmem` / `wmem` / `netdev_max_backlog` / `tcp_rmem` /
   `tcp_wmem`), firewall, NTP synchronisation.
8. **Storage** — filesystem usage (flagged `WARN` at ≥70%, `FAIL` at ≥90%),
   NVMe drives with per-device model / firmware / SMART `critical_warning` /
   `percentage_used` / `available_spare` / temperature. Also enforces the
   Supermicro `AS -4126GS-NBR-LCC` 8-NVMe requirement when that chassis is
   detected.
9. **Container Runtime** — Docker, NVIDIA Container Toolkit, default runtime,
   Enroot, Apptainer / Singularity.
10. **Workload Manager** — SLURM (`slurmctld`, `slurmd`) or PBS/Torque
    detection, plus RKE2 / Kubernetes: RKE2 binary + role, `rke2-agent`
    service, kubeconfig, API reachability, node readiness, GPU Operator /
    NVIDIA device plugin / DCGM Exporter / Network Operator pod health,
    allocatable `nvidia.com/gpu` on this node, and non-Running pods on this
    node.

## Output

### Terminal

Banner-delimited sections with tallied counters at the end:

```
  PASS: 42   WARN: 3   FAIL: 0   INFO: 57
  ⚡ All critical checks passed, but 3 warning(s) to review.
```

Exit is informational only — the script always completes; inspect the tally
and the Prometheus `gpuone_healthcheck_health` gauge (`2`=healthy,
`1`=warnings, `0`=failures) to drive automation.

### Prometheus (node_exporter textfile collector)

Metrics are written to:

```
/var/lib/node_exporter/textfile_collector/gpu_health_check.prom
```

All metrics carry a `host="<short-hostname>"` label. Per-GPU, per-HCA, per-NIC,
per-mountpoint, per-NVMe, and per-service metrics add their own labels
(`gpu`, `gpu_name`, `device`, `port`, `interface`, `mountpoint`, `service`,
etc.). Summary gauges:

| Metric | Meaning |
|--------|---------|
| `gpuone_healthcheck_pass_total` | Count of PASS checks |
| `gpuone_healthcheck_warn_total` | Count of WARN checks |
| `gpuone_healthcheck_fail_total` | Count of FAIL checks |
| `gpuone_healthcheck_info_total` | Count of INFO entries |
| `gpuone_healthcheck_health` | `2`=healthy, `1`=warnings, `0`=failures |
| `gpuone_healthcheck_timestamp_seconds` | Unix timestamp of the run |

Duplicate `# HELP` / `# TYPE` lines are stripped before writing the final
file, so the output is safe to scrape directly.

## Scheduling

For continuous visibility, run from cron or a systemd timer. Example cron
entry (every 5 minutes):

```cron
*/5 * * * * root /usr/local/bin/gpu-healthcheck.sh >/var/log/gpu-healthcheck.log 2>&1
```

`node_exporter` picks up the `.prom` file on its next scrape — no service
restart needed.

## Requirements

The script degrades gracefully when tools are missing, but for full coverage:

- `nvidia-smi` (NVIDIA driver)
- `ibstat` / `ofed_info` (MOFED) and `ethtool`
- `nvme-cli`
- `numactl`, `lscpu`, `lsb_release`, `dmidecode`
- `docker` / `nvidia-container-cli` (if containers are in use)
- `sinfo` (SLURM) or `rke2` + `kubectl` (Kubernetes)

## Design notes

- `set -u` only — the script intentionally does **not** `set -e`. A
  diagnostic tool must survive failing sub-commands and still print a full
  report.
- Every `nvidia-smi` call is wrapped in `timeout ${NSMI_TIMEOUT}` (default
  15 s) so a hung driver can't stall the run.
- `GPU_SECTION_SKIP` short-circuits all GPU sub-checks when `nvidia-smi` is
  unresponsive, rather than letting each check hit its own timeout.
