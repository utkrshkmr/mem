#!/usr/bin/env bash
# Node check (PLAN.md §5.1). Prints the checks and writes reports/node.json.
# Exits non-zero if any hard requirement fails; CPU/RAM and token-file mode only warn.
set -u
cd "$(dirname "$0")/.."
mkdir -p reports

fail=0
warn=0
say()  { printf '%-10s %s\n' "$1" "$2"; }
bad()  { say "FAIL" "$1"; fail=1; }
soft() { say "WARN" "$1"; warn=1; }

gpus_csv=$(nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader,nounits 2>/dev/null)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
topo=$(nvidia-smi topo -m 2>/dev/null)
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
ncpu=$(nproc)
ram_gb=$(free -g | awk '/^Mem:/{print $2}')
disk_free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
py311=$(python3.11 --version 2>/dev/null || true)

n_gpu=$(printf '%s\n' "$gpus_csv" | grep -c . || true)
n_h100=$(printf '%s\n' "$gpus_csv" | awk -F', ' '$2 ~ /H100/ && $3 >= 79*1024 {c++} END {print c+0}')

say "GPUs" "$n_gpu found, $n_h100 are H100 with >= 79 GB"
[ "$n_gpu" -eq 4 ] && [ "$n_h100" -eq 4 ] || bad "need 4 H100 GPUs with >= 79 GB each"

say "Driver" "$driver"
driver_major=${driver%%.*}
if [ -n "$driver_major" ] && [ "$driver_major" -ge 580 ]; then install_path=A; else install_path=B; fi
say "Install" "path $install_path (§5.2)"

if [ -n "$apps" ]; then bad "GPUs are not idle: compute PIDs $(echo $apps | tr '\n' ' ')"; else say "Idle" "no compute processes"; fi

say "CPU/RAM" "$ncpu cores, $ram_gb GB"
[ "$ncpu" -ge 32 ] || soft "fewer than 32 cores"
[ "$ram_gb" -ge 256 ] || soft "less than 256 GB RAM"

say "Disk" "$disk_free_gb GB free at $(pwd)"
[ "$disk_free_gb" -ge 400 ] || bad "need >= 400 GB free where runs/ lives"

if [ -n "$py311" ]; then say "Python" "$py311"; else bad "python3.11 not found (try: uv python install 3.11)"; fi

tok_mode=""
if [ -s hf-tok ]; then
  tok_mode=$(stat -c '%a' hf-tok)
  say "Token" "hf-tok present, mode $tok_mode"
  [ "$tok_mode" = "600" ] || soft "hf-tok mode is $tok_mode; run: chmod 600 hf-tok"
else
  bad "hf-tok missing or empty"
fi

GPUS_CSV="$gpus_csv" DRIVER="$driver" TOPO="$topo" APPS="$apps" NCPU="$ncpu" RAM_GB="$ram_gb" \
DISK_FREE_GB="$disk_free_gb" PY311="$py311" TOK_MODE="$tok_mode" INSTALL_PATH="$install_path" \
FAIL="$fail" WARN="$warn" python3 - <<'EOF'
import datetime, json, os, socket
e = os.environ
gpus = []
for line in e["GPUS_CSV"].strip().splitlines():
    idx, name, mem, uuid = [x.strip() for x in line.split(",")]
    gpus.append({"index": int(idx), "name": name, "memory_mib": int(mem), "uuid": uuid})
out = {
    "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "gpus": gpus,
    "driver_version": e["DRIVER"],
    "install_path": e["INSTALL_PATH"],
    "topology": e["TOPO"],
    "gpu_compute_pids": [p for p in e["APPS"].split() if p],
    "cpu_cores": int(e["NCPU"]),
    "ram_gb": int(e["RAM_GB"]),
    "disk_free_gb": int(e["DISK_FREE_GB"]),
    "python311": e["PY311"],
    "hf_tok_mode": e["TOK_MODE"],
    "passed": e["FAIL"] == "0",
    "warnings": e["WARN"] == "1",
}
with open("reports/node.json", "w") as f:
    json.dump(out, f, indent=2)
EOF

if [ "$fail" -ne 0 ]; then say "RESULT" "FAIL (see above); reports/node.json written"; exit 1; fi
say "RESULT" "PASS$([ "$warn" -ne 0 ] && echo ' with warnings'); reports/node.json written"
