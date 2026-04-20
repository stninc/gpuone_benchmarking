#!/usr/bin/env bash
###############################################################################
#  GPU HPC Server Health Check
#  Target: 8× NVIDIA B300  ·  8× ConnectX-8 HCA  ·  256 CPUs  ·  Ubuntu
#
#  Run as root (or with sudo) for full visibility.
#  Usage:  sudo bash gpu_health_check.sh [--json]
###############################################################################

set -u  # Only nounset — a diagnostic script must NEVER abort on errors

# ── Globals ──────────────────────────────────────────────────────────────────
EXPECTED_GPUS=8
EXPECTED_NICS=8
EXPECTED_CPUS=256
PASS=0; WARN=0; FAIL=0; INFO=0
JSON_MODE=false
[[ "${1:-}" == "--json" ]] && JSON_MODE=true
JSON_ENTRIES=()

# ── Colors & formatting ─────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[0;33m'
CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'

banner()  { echo -e "\n${BLD}${CYN}═══════════════════════════════════════════════════════════════${RST}"; echo -e "${BLD}${CYN}  $1${RST}"; echo -e "${BLD}${CYN}═══════════════════════════════════════════════════════════════${RST}"; }
section() { echo -e "\n${BLD}── $1 ──${RST}"; }
ok()      { echo -e "  [${GRN}  OK  ${RST}] $1"; (( PASS++ )) || true; }
warn()    { echo -e "  [${YEL} WARN ${RST}] $1"; (( WARN++ )) || true; }
fail()    { echo -e "  [${RED} FAIL ${RST}] $1"; (( FAIL++ )) || true; }
info()    { echo -e "  [${CYN} INFO ${RST}] $1"; (( INFO++ )) || true; }
json_add(){ JSON_ENTRIES+=("$(printf '{"section":"%s","status":"%s","detail":"%s"}' "$1" "$2" "$3")"); }

divider() { echo -e "  ${CYN}─────────────────────────────────────────────────────────────${RST}"; }

# ── Preflight ────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    echo -e "${YEL}WARNING: Not running as root. Some checks will be incomplete.${RST}"
    echo -e "${YEL}Re-run with: sudo bash $0${RST}\n"
fi

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
HOSTNAME=$(hostname -f 2>/dev/null || hostname)

banner "GPU HPC Server Health Check"
echo -e "  Host:       ${BLD}${HOSTNAME}${RST}"
echo -e "  Timestamp:  ${BLD}${TIMESTAMP}${RST}"
echo -e "  Expected:   ${EXPECTED_GPUS} GPUs · ${EXPECTED_NICS} NICs · ${EXPECTED_CPUS} CPUs"

###############################################################################
#  1. OPERATING SYSTEM & KERNEL
###############################################################################
banner "1. Operating System & Kernel"

# Ubuntu release
if command -v lsb_release &>/dev/null; then
    DISTRO=$(lsb_release -ds 2>/dev/null)
    RELEASE=$(lsb_release -rs 2>/dev/null)
    info "Distribution:      ${DISTRO}"
else
    DISTRO=$(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2)
    RELEASE=""
    info "Distribution:      ${DISTRO}"
fi

info "Kernel:            $(uname -r)"
info "Architecture:      $(uname -m)"
info "Boot time:         $(who -b 2>/dev/null | awk '{print $3, $4}' || uptime -s 2>/dev/null)"
UPTIME=$(uptime -p 2>/dev/null || uptime | sed 's/.*up /up /' | sed 's/,.*//');
info "Uptime:            ${UPTIME}"

# Kernel command-line notable params
section "Kernel Command Line (notable params)"
KCMD=$(cat /proc/cmdline 2>/dev/null || echo "N/A")
for kw in iommu hugepages numa default_hugepagesz; do
    match=$(echo "$KCMD" | grep -oE "[^ ]*${kw}[^ ]*" || true)
    [[ -n "$match" ]] && info "  $match" || true
done

###############################################################################
#  2. CPU
###############################################################################
banner "2. CPU"

CPU_ONLINE=$(nproc --all 2>/dev/null || grep -c ^processor /proc/cpuinfo)
CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs)
NUMA_NODES=$(lscpu 2>/dev/null | grep "NUMA node(s)" | awk '{print $NF}')
SOCKETS=$(lscpu 2>/dev/null | grep "Socket(s)" | awk '{print $NF}')
CORES_PER_SOCKET=$(lscpu 2>/dev/null | grep "Core(s) per socket" | awk '{print $NF}')
THREADS_PER_CORE=$(lscpu 2>/dev/null | grep "Thread(s) per core" | awk '{print $NF}')

info "CPU Model:         ${CPU_MODEL:-N/A}"
info "Sockets:           ${SOCKETS:-N/A}"
info "Cores/Socket:      ${CORES_PER_SOCKET:-N/A}"
info "Threads/Core:      ${THREADS_PER_CORE:-N/A}"
info "NUMA Nodes:        ${NUMA_NODES:-N/A}"

if [[ "$CPU_ONLINE" -eq "$EXPECTED_CPUS" ]]; then
    ok "Online CPUs: ${CPU_ONLINE}/${EXPECTED_CPUS}"
elif [[ "$CPU_ONLINE" -gt 0 ]]; then
    fail "Online CPUs: ${CPU_ONLINE}/${EXPECTED_CPUS} — some CPUs offline"
else
    fail "Cannot determine CPU count"
fi

# Offline CPUs
OFFLINE_CPUS=$(cat /sys/devices/system/cpu/offline 2>/dev/null)
if [[ -n "$OFFLINE_CPUS" && "$OFFLINE_CPUS" != "" ]]; then
    fail "Offline CPUs detected: ${OFFLINE_CPUS}"
else
    ok "No offline CPUs"
fi

# Governor
section "CPU Frequency Governor"
if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    if [[ "$GOV" == "performance" ]]; then
        ok "CPU governor: performance"
    else
        warn "CPU governor: ${GOV} (recommend 'performance' for HPC)"
    fi
else
    info "CPU frequency governor: not available (may be firmware-managed)"
fi

# Hyper-threading
section "Hyper-Threading"
if [[ "${THREADS_PER_CORE:-1}" -gt 1 ]]; then
    info "Hyper-Threading: ENABLED (threads/core=${THREADS_PER_CORE})"
else
    info "Hyper-Threading: DISABLED or single-threaded"
fi

###############################################################################
#  3. MEMORY
###############################################################################
banner "3. Memory"

TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$(awk "BEGIN {printf \"%.0f\", ${TOTAL_MEM_KB}/1048576}")
FREE_MEM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
FREE_MEM_GB=$(awk "BEGIN {printf \"%.1f\", ${FREE_MEM_KB}/1048576}")

info "Total Memory:      ${TOTAL_MEM_GB} GB"
info "Available Memory:  ${FREE_MEM_GB} GB"

# Hugepages
section "Hugepages"
HP_TOTAL=$(grep -i HugePages_Total /proc/meminfo 2>/dev/null | awk '{print $2}')
HP_FREE=$(grep -i HugePages_Free /proc/meminfo 2>/dev/null | awk '{print $2}')
HP_SIZE=$(grep -i Hugepagesize /proc/meminfo 2>/dev/null | awk '{print $2, $3}')
info "HugePages Total:   ${HP_TOTAL:-0}"
info "HugePages Free:    ${HP_FREE:-0}"
info "Hugepage Size:     ${HP_SIZE:-N/A}"

# NUMA memory balance
section "NUMA Memory Distribution"
if command -v numactl &>/dev/null; then
    numactl --hardware 2>/dev/null | grep -E "^node [0-9]+ size" | while read -r line; do
        info "  $line"
    done
elif [[ -d /sys/devices/system/node ]]; then
    for node_dir in /sys/devices/system/node/node*; do
        node=$(basename "$node_dir")
        meminfo="${node_dir}/meminfo"
        if [[ -f "$meminfo" ]]; then
            total=$(grep MemTotal "$meminfo" | awk '{print $4}')
            total_gb=$(awk "BEGIN {printf \"%.0f\", ${total}/1048576}")
            info "  ${node}: ${total_gb} GB"
        fi
    done
else
    info "NUMA info not available"
fi

###############################################################################
#  4. NVIDIA GPUs
###############################################################################
banner "4. NVIDIA GPUs"

NSMI_TIMEOUT=15  # seconds — per nvidia-smi invocation

# Wrapper: runs nvidia-smi with a timeout so a hung driver can't stall the script
nsmi() {
    timeout "${NSMI_TIMEOUT}" nvidia-smi "$@" 2>/dev/null
}

if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found — NVIDIA driver not installed or not in PATH"
else
    # ── Probe: can nvidia-smi respond at all? ──
    section "Driver Responsiveness"
    if ! nsmi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
        fail "nvidia-smi is not responding (hung or driver fault) — skipping GPU checks"
        GPU_SECTION_SKIP=1
    else
        ok "nvidia-smi responding within ${NSMI_TIMEOUT}s"
        GPU_SECTION_SKIP=0
    fi

    if [[ "${GPU_SECTION_SKIP}" -eq 0 ]]; then

    # Driver & CUDA
    DRIVER_VER=$(nsmi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    CUDA_VER=$(nsmi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' || echo "N/A")
    info "Driver Version:    ${DRIVER_VER:-N/A}"
    info "CUDA Version:      ${CUDA_VER}"

    # GPU count
    GPU_COUNT=$(nsmi --query-gpu=name --format=csv,noheader | wc -l)
    if [[ "$GPU_COUNT" -eq "$EXPECTED_GPUS" ]]; then
        ok "GPU count: ${GPU_COUNT}/${EXPECTED_GPUS}"
    else
        fail "GPU count: ${GPU_COUNT}/${EXPECTED_GPUS}"
    fi

    # Per-GPU details
    section "Per-GPU Status"
    divider
    printf "  ${BLD}%-4s %-22s %-8s %-8s %-10s %-8s %-6s${RST}\n" \
        "GPU" "Name" "Temp" "Power" "Mem Used" "Util" "ECC"
    divider

    nsmi --query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu,ecc.errors.uncorrected.volatile.total \
        --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r idx name temp power mem_used mem_total util ecc_err; do
        name=$(echo "$name" | xargs)
        temp=$(echo "$temp" | xargs)
        power=$(echo "$power" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        util=$(echo "$util" | xargs)
        ecc_err=$(echo "$ecc_err" | xargs)

        ecc_status="OK"
        if [[ "$ecc_err" =~ ^[0-9]+$ && "$ecc_err" -gt 0 ]]; then
            ecc_status="${RED}${ecc_err} errs${RST}"
        elif [[ "$ecc_err" == "N/A" || "$ecc_err" == "[N/A]" ]]; then
            ecc_status="N/A"
        fi

        printf "  %-4s %-22s %-8s %-8s %-10s %-8s %-6b\n" \
            "$idx" "$name" "${temp}°C" "${power}W" "${mem_used}/${mem_total}M" "${util}%" "$ecc_status"
    done
    divider

    # Persistence mode
    section "Persistence Mode"
    PERSIST=$(nsmi --query-gpu=persistence_mode --format=csv,noheader | sort -u | xargs)
    if [[ "$PERSIST" == "Enabled" ]]; then
        ok "Persistence mode: Enabled (all GPUs)"
    else
        warn "Persistence mode: ${PERSIST} (recommend Enabled for HPC)"
    fi

    # Compute mode
    section "Compute Mode"
    CMODE=$(nsmi --query-gpu=compute_mode --format=csv,noheader | sort -u | xargs)
    info "Compute mode: ${CMODE}"
    if [[ "$CMODE" == *"Exclusive"* ]]; then
        ok "Exclusive compute mode set"
    elif [[ "$CMODE" == *"Default"* ]]; then
        info "Compute mode is Default (consider Exclusive_Process for dedicated workloads)"
    fi

    # XID errors
    section "XID Errors (dmesg)"
    if [[ $EUID -eq 0 ]]; then
        XID_COUNT=$(dmesg 2>/dev/null | grep -ci "NVRM.*Xid" || true)
        if [[ "$XID_COUNT" -eq 0 ]]; then
            ok "No XID errors in dmesg"
        else
            fail "Found ${XID_COUNT} XID error(s) in dmesg"
            dmesg 2>/dev/null | grep -i "NVRM.*Xid" | tail -10 | while read -r line; do
                echo -e "       ${RED}${line}${RST}"
            done
        fi
    else
        warn "Cannot read dmesg without root — skipping XID check"
    fi

    # Retired pages / Row remapper
    section "GPU Memory Health (Retired Pages)"
    nsmi --query-gpu=index,retired_pages.pending,retired_pages.sbe,retired_pages.dbe \
        --format=csv,noheader 2>/dev/null | while IFS=', ' read -r idx pending sbe dbe; do
        pending=$(echo "$pending" | xargs)
        sbe=$(echo "$sbe" | xargs)
        dbe=$(echo "$dbe" | xargs)
        if [[ "$dbe" =~ ^[0-9]+$ && "$dbe" -gt 0 ]]; then
            fail "GPU ${idx}: ${dbe} double-bit retired pages (needs attention)"
        elif [[ "$pending" == "Yes" ]]; then
            warn "GPU ${idx}: Retired page pending (reboot recommended)"
        else
            ok "GPU ${idx}: Retired pages SBE=${sbe:-0} DBE=${dbe:-0} Pending=${pending:-No}"
        fi
    done

    # NVLink status
    section "NVLink Status"
    NVLINK_ERR=0
    for gpu_id in $(seq 0 $((GPU_COUNT - 1))); do
        LINKS=$(timeout "${NSMI_TIMEOUT}" nvidia-smi nvlink -s -i "$gpu_id" 2>&1 || true)
        if [[ -z "$LINKS" || "$LINKS" == *"not supported"* || "$LINKS" == *"error"* || "$LINKS" == *"timed out"* ]]; then
            info "GPU ${gpu_id}: NVLink query not supported or timed out"
            continue
        fi
        TOTAL_LINKS=$(echo "$LINKS" | grep -cE "Link [0-9]+" || true)
        INACTIVE=$(echo "$LINKS" | grep -ciE "inactiv|down|disabled|false" || true)
        if [[ "$TOTAL_LINKS" -eq 0 ]]; then
            info "GPU ${gpu_id}: No NVLink connections reported (NVSwitch topology may use fabric manager)"
        elif [[ "$INACTIVE" -gt 0 ]]; then
            fail "GPU ${gpu_id}: ${INACTIVE}/${TOTAL_LINKS} NVLink(s) inactive/down"
            (( NVLINK_ERR++ )) || true
        else
            ok "GPU ${gpu_id}: ${TOTAL_LINKS} NVLink(s) detected"
        fi
    done

    # NVLink error counters
    section "NVLink Error Counters"
    NVL_REPLAY_TOTAL=0
    for gpu_id in $(seq 0 $((GPU_COUNT - 1))); do
        ERRS=$(timeout "${NSMI_TIMEOUT}" nvidia-smi nvlink -e -i "$gpu_id" 2>/dev/null || true)
        if [[ -n "$ERRS" ]]; then
            REPLAY=$(echo "$ERRS" | grep -i "replay" | awk '{sum += $NF} END {print sum+0}' || true)
            RECOVERY=$(echo "$ERRS" | grep -i "recovery" | awk '{sum += $NF} END {print sum+0}' || true)
            CRC_FLIT=$(echo "$ERRS" | grep -i "crc.*flit" | awk '{sum += $NF} END {print sum+0}' || true)
            if [[ "${REPLAY:-0}" -gt 0 || "${RECOVERY:-0}" -gt 0 ]]; then
                warn "GPU ${gpu_id}: NVLink errors — replay=${REPLAY} recovery=${RECOVERY} crc_flit=${CRC_FLIT}"
            else
                ok "GPU ${gpu_id}: NVLink error counters clean"
            fi
        fi
    done

    # GPU topology
    section "GPU Topology (NVLink/NVSwitch)"
    timeout "${NSMI_TIMEOUT}" nvidia-smi topo -m 2>/dev/null | head -12 || info "nvidia-smi topo not available or timed out"

    # Clocks
    section "GPU Clock Settings"
    nsmi --query-gpu=index,clocks.current.graphics,clocks.max.graphics,clocks.current.memory,clocks.max.memory \
        --format=csv,noheader,nounits 2>/dev/null | head -2 | while IFS=', ' read -r idx gc gm mc mm; do
        gc=$(echo "$gc" | xargs); gm=$(echo "$gm" | xargs)
        mc=$(echo "$mc" | xargs); mm=$(echo "$mm" | xargs)
        info "GPU ${idx}: Graphics ${gc}/${gm} MHz  Memory ${mc}/${mm} MHz"
    done || true
    [[ "$GPU_COUNT" -gt 2 ]] && info "  ... (showing first 2 of ${GPU_COUNT})" || true

    # Fabric Manager (needed for NVSwitch / multi-GPU)
    section "Fabric Manager"
    if systemctl is-active --quiet nvidia-fabricmanager 2>/dev/null; then
        ok "nvidia-fabricmanager service: active"
    elif systemctl list-unit-files 2>/dev/null | grep -q nvidia-fabricmanager; then
        fail "nvidia-fabricmanager service: installed but not active"
    else
        info "nvidia-fabricmanager: not found (may not be required for this topology)"
    fi

    fi  # end GPU_SECTION_SKIP
fi

###############################################################################
#  5. NETWORKING — ConnectX-8 HCAs
###############################################################################
banner "5. InfiniBand / ConnectX-8 HCAs"

# OFED / MOFED version
section "OFED / MOFED Version"
if command -v ofed_info &>/dev/null; then
    OFED_VER=$(ofed_info -s 2>/dev/null | xargs)
    info "OFED Version:      ${OFED_VER}"
else
    warn "ofed_info not found — Mellanox OFED may not be installed"
fi

# Detect HCAs
section "HCA Detection"
if command -v ibstat &>/dev/null; then
    HCA_COUNT=$(ibstat -l 2>/dev/null | wc -l)
elif [[ -d /sys/class/infiniband ]]; then
    HCA_COUNT=$(ls /sys/class/infiniband 2>/dev/null | wc -l)
else
    HCA_COUNT=0
fi

if [[ "$HCA_COUNT" -eq "$EXPECTED_NICS" ]]; then
    ok "HCA count: ${HCA_COUNT}/${EXPECTED_NICS}"
elif [[ "$HCA_COUNT" -gt 0 ]]; then
    warn "HCA count: ${HCA_COUNT}/${EXPECTED_NICS}"
else
    # Fall back to lspci for Mellanox/NVIDIA NICs
    MLNX_PCI=$(lspci 2>/dev/null | grep -i -E "mellanox|connectx|infiniband|BlueField" | wc -l)
    if [[ "$MLNX_PCI" -gt 0 ]]; then
        warn "Found ${MLNX_PCI} Mellanox PCI devices but IB subsystem not loaded"
    else
        fail "No ConnectX HCAs detected"
    fi
fi

# Per-port link status
section "HCA Port Status"
if command -v ibstat &>/dev/null; then
    divider
    printf "  ${BLD}%-16s %-8s %-14s %-16s %-12s${RST}\n" \
        "HCA" "Port" "State" "Speed" "PhysState"
    divider
    ibstat 2>/dev/null | awk '
    /CA '\''/{hca=$2; gsub(/'\''/, "", hca)}
    /Port [0-9]+/{port=$2}
    /State:/{state=$2}
    /Physical state:/{phys=$3}
    /Rate:/{rate=$2" "$3; printf "  %-16s %-8s %-14s %-16s %-12s\n", hca, port, state, rate, phys}
    '
    divider
elif [[ -d /sys/class/infiniband ]]; then
    for dev in /sys/class/infiniband/*; do
        dev_name=$(basename "$dev")
        for port_dir in "$dev"/ports/*; do
            port=$(basename "$port_dir")
            state=$(cat "$port_dir/state" 2>/dev/null | awk '{print $2}')
            rate=$(cat "$port_dir/rate" 2>/dev/null)
            if [[ "$state" == "ACTIVE" ]]; then
                ok "${dev_name} port ${port}: ${state} @ ${rate}"
            else
                fail "${dev_name} port ${port}: ${state}"
            fi
        done
    done
fi

# Also check standard Ethernet NICs for link
section "Ethernet Link Status (RDMA-capable)"
for netdev in /sys/class/net/*/device/infiniband*; do
    [[ -e "$netdev" ]] || continue
    parent=$(echo "$netdev" | cut -d'/' -f5)
    carrier=$(cat /sys/class/net/"$parent"/carrier 2>/dev/null || echo "0")
    speed=$(cat /sys/class/net/"$parent"/speed 2>/dev/null || echo "?")
    operstate=$(cat /sys/class/net/"$parent"/operstate 2>/dev/null || echo "unknown")
    mtu=$(cat /sys/class/net/"$parent"/mtu 2>/dev/null || echo "?")
    if [[ "$operstate" == "up" ]]; then
        ok "${parent}: UP @ ${speed} Mbps  MTU=${mtu}"
    else
        warn "${parent}: ${operstate}"
    fi
done

# Firmware versions
section "NIC Firmware Versions"
if command -v ethtool &>/dev/null; then
    for netdev in $(ls /sys/class/net/ 2>/dev/null | grep -vE "^(lo|docker|br|veth|virbr)"); do
        fw=$(ethtool -i "$netdev" 2>/dev/null | grep "^firmware-version" | awk '{print $2}')
        drv=$(ethtool -i "$netdev" 2>/dev/null | grep "^driver" | awk '{print $2}')
        if [[ -n "$fw" && "$drv" =~ ^mlx ]]; then
            info "${netdev}: driver=${drv}  firmware=${fw}"
        fi
    done
fi

# GPUDirect RDMA / PeerDirect
section "GPUDirect RDMA"
if lsmod 2>/dev/null | grep -q nv_peer_mem; then
    ok "nv_peer_mem module loaded (GPUDirect RDMA)"
elif lsmod 2>/dev/null | grep -q nvidia_peermem; then
    ok "nvidia_peermem module loaded (GPUDirect RDMA)"
else
    warn "No GPUDirect RDMA module loaded (nv_peer_mem / nvidia_peermem)"
fi

###############################################################################
#  6. SOFTWARE & PACKAGES
###############################################################################
banner "6. Software & Package Management"

# CUDA toolkit
section "CUDA Toolkit"
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $NF}' | tr -d ',')
    info "nvcc version:      ${NVCC_VER}"
else
    info "nvcc not in PATH (CUDA toolkit may not be installed or not in PATH)"
fi

# NCCL
section "NCCL"
NCCL_LIB=$(ldconfig -p 2>/dev/null | grep libnccl.so | head -1 | awk '{print $NF}')
if [[ -n "$NCCL_LIB" ]]; then
    NCCL_VER=$(strings "$NCCL_LIB" 2>/dev/null | grep -oP 'NCCL version \K[0-9.]+' | head -1 || echo "detected")
    ok "NCCL library found: ${NCCL_LIB} (${NCCL_VER})"
else
    NCCL_PKG=$(dpkg -l 2>/dev/null | grep -i libnccl | head -1 | awk '{print $2, $3}')
    if [[ -n "$NCCL_PKG" ]]; then
        info "NCCL package: ${NCCL_PKG}"
    else
        warn "NCCL library not detected"
    fi
fi

# cuDNN
section "cuDNN"
CUDNN_LIB=$(ldconfig -p 2>/dev/null | grep libcudnn.so | head -1 | awk '{print $NF}')
if [[ -n "$CUDNN_LIB" ]]; then
    info "cuDNN library: ${CUDNN_LIB}"
else
    CUDNN_PKG=$(dpkg -l 2>/dev/null | grep -i libcudnn | head -1 | awk '{print $2, $3}')
    if [[ -n "$CUDNN_PKG" ]]; then
        info "cuDNN package: ${CUDNN_PKG}"
    else
        info "cuDNN not detected"
    fi
fi

# GDRCopy
section "GDRCopy"
if lsmod 2>/dev/null | grep -q gdrdrv; then
    ok "GDRCopy kernel module (gdrdrv) loaded"
else
    info "GDRCopy kernel module not loaded"
fi

# Pinned / held packages
section "Pinned / Held Packages"
HELD_PKGS=$(dpkg --get-selections 2>/dev/null | grep "hold$" || true)
APT_PINNED=$(apt-mark showhold 2>/dev/null || true)
if [[ -n "$HELD_PKGS" || -n "$APT_PINNED" ]]; then
    info "Held packages:"
    if [[ -n "$APT_PINNED" ]]; then
        echo "$APT_PINNED" | while read -r pkg; do
            ver=$(dpkg -l "$pkg" 2>/dev/null | awk '/^ii/{print $3}')
            info "  ${pkg}  ${ver}"
        done
    fi
else
    info "No held/pinned packages"
fi

# Auto-updates / unattended-upgrades
section "Automatic Updates"
if systemctl is-active --quiet unattended-upgrades 2>/dev/null; then
    warn "unattended-upgrades service is ACTIVE (may cause unexpected reboots/updates)"
elif dpkg -l unattended-upgrades 2>/dev/null | grep -q "^ii"; then
    ENABLED=$(grep -r "Unattended-Upgrade" /etc/apt/apt.conf.d/ 2>/dev/null | grep -v "^#" | grep -ci '"1"' || true)
    if [[ "$ENABLED" -gt 0 ]]; then
        warn "unattended-upgrades installed and appears enabled"
    else
        ok "unattended-upgrades installed but appears disabled"
    fi
else
    ok "unattended-upgrades not installed"
fi

# Check apt-daily timers
for timer in apt-daily.timer apt-daily-upgrade.timer; do
    if systemctl is-active --quiet "$timer" 2>/dev/null; then
        warn "${timer} is ACTIVE (recommend disabling for HPC stability)"
    else
        ok "${timer} is inactive/disabled"
    fi
done

###############################################################################
#  7. SYSTEM SERVICES & CONFIGURATION
###############################################################################
banner "7. System Configuration"

# Swap
section "Swap"
SWAP_TOTAL=$(free -g 2>/dev/null | awk '/Swap/{print $2}')
if [[ "${SWAP_TOTAL:-0}" -eq 0 ]]; then
    ok "Swap: disabled (recommended for GPU HPC)"
else
    warn "Swap: ${SWAP_TOTAL} GB enabled (consider disabling for HPC workloads)"
fi

# vm.swappiness
SWAPPINESS=$(sysctl -n vm.swappiness 2>/dev/null || echo "?")
info "vm.swappiness = ${SWAPPINESS}"

# Transparent Hugepages
section "Transparent Hugepages"
THP=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo "N/A")
info "THP: ${THP}"
if echo "$THP" | grep -q "\[never\]"; then
    ok "Transparent Hugepages disabled (good for predictable latency)"
elif echo "$THP" | grep -q "\[always\]"; then
    info "THP set to 'always' (may cause latency jitter — consider 'madvise' or 'never')"
fi

# Kernel params for networking
section "Network Tuning (selected sysctl)"
for param in \
    net.core.rmem_max \
    net.core.wmem_max \
    net.core.rmem_default \
    net.core.wmem_default \
    net.core.netdev_max_backlog \
    net.ipv4.tcp_rmem \
    net.ipv4.tcp_wmem; do
    val=$(sysctl -n "$param" 2>/dev/null || echo "N/A")
    info "${param} = ${val}"
done

# Firewall
section "Firewall"
if command -v ufw &>/dev/null; then
    UFW_STATUS=$(ufw status 2>/dev/null | head -1)
    info "UFW: ${UFW_STATUS}"
elif command -v iptables &>/dev/null; then
    IPT_RULES=$(iptables -L 2>/dev/null | grep -c "^Chain" || echo "?")
    info "iptables chains: ${IPT_RULES}"
fi

# NTP / time sync
section "Time Synchronization"
if command -v timedatectl &>/dev/null; then
    NTP_SYNC=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || timedatectl 2>/dev/null | grep "synchronized" | awk '{print $NF}')
    if [[ "$NTP_SYNC" == "yes" ]]; then
        ok "NTP synchronized: yes"
    else
        warn "NTP not synchronized (important for distributed training)"
    fi
fi

###############################################################################
#  8. STORAGE
###############################################################################
banner "8. Storage"

section "Filesystem Usage (>70% flagged)"
df -h --type=ext4 --type=xfs --type=btrfs --type=lustre --type=gpfs --type=nfs --type=nfs4 --type=tmpfs 2>/dev/null | \
    awk 'NR==1 || $5+0 > 0' | while read -r line; do
    pct=$(echo "$line" | awk '{print $5}' | tr -d '%')
    if [[ "$pct" =~ ^[0-9]+$ ]]; then
        if [[ "$pct" -ge 90 ]]; then
            fail "  ${line}"
        elif [[ "$pct" -ge 70 ]]; then
            warn "  ${line}"
        else
            info "  ${line}"
        fi
    else
        echo -e "  ${BLD}${line}${RST}"
    fi
done

###############################################################################
#  9. DOCKER / CONTAINER RUNTIME
###############################################################################
banner "9. Container Runtime"

if command -v docker &>/dev/null; then
    DOCKER_VER=$(docker --version 2>/dev/null)
    info "Docker: ${DOCKER_VER}"

    # NVIDIA Container Toolkit
    if command -v nvidia-container-cli &>/dev/null; then
        ok "NVIDIA Container Toolkit: installed"
    elif dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
        ok "NVIDIA Container Toolkit: installed (package)"
    else
        warn "NVIDIA Container Toolkit not detected"
    fi

    # Default runtime
    DEFAULT_RT=$(docker info 2>/dev/null | grep "Default Runtime" | awk '{print $NF}')
    if [[ "$DEFAULT_RT" == "nvidia" ]]; then
        ok "Default Docker runtime: nvidia"
    else
        info "Default Docker runtime: ${DEFAULT_RT:-unknown} (set to 'nvidia' if needed)"
    fi
else
    info "Docker not installed"
fi

if command -v enroot &>/dev/null; then
    info "Enroot: $(enroot version 2>/dev/null)"
fi

if command -v apptainer &>/dev/null; then
    info "Apptainer: $(apptainer --version 2>/dev/null)"
elif command -v singularity &>/dev/null; then
    info "Singularity: $(singularity --version 2>/dev/null)"
fi

###############################################################################
#  10. SCHEDULER / WORKLOAD MANAGER
###############################################################################
banner "10. Workload Manager"

if command -v sinfo &>/dev/null; then
    info "SLURM detected"
    info "  slurmctld: $(systemctl is-active slurmctld 2>/dev/null || echo 'N/A')"
    info "  slurmd:    $(systemctl is-active slurmd 2>/dev/null || echo 'N/A')"
    sinfo --version 2>/dev/null | while read -r line; do info "  $line"; done
elif command -v pbsnodes &>/dev/null; then
    info "PBS/Torque detected"
else
    info "No workload manager detected"
fi

###############################################################################
#  SUMMARY
###############################################################################
banner "SUMMARY"

TOTAL=$((PASS + WARN + FAIL))
echo -e "  ${GRN}PASS: ${PASS}${RST}   ${YEL}WARN: ${WARN}${RST}   ${RED}FAIL: ${FAIL}${RST}   ${CYN}INFO: ${INFO}${RST}"
echo ""

if [[ "$FAIL" -gt 0 ]]; then
    echo -e "  ${RED}${BLD}⚠  ${FAIL} check(s) FAILED — review items above.${RST}"
elif [[ "$WARN" -gt 0 ]]; then
    echo -e "  ${YEL}${BLD}⚡ All critical checks passed, but ${WARN} warning(s) to review.${RST}"
else
    echo -e "  ${GRN}${BLD}✓  All checks passed. Server appears healthy.${RST}"
fi

echo ""
echo -e "  Report generated: ${TIMESTAMP}"
echo -e "  Log this output:  sudo bash $0 2>&1 | tee healthcheck_\$(date +%Y%m%d_%H%M%S).log"
echo ""
