#!/usr/bin/env bash
# ncu.sh -- one profiled launch per rung, a fixed set of Nsight Compute
# metrics, one table.  PROVIDED.
#
#   bash ncu.sh ./sgemm N RUNG...   (make ncu and make ncu-sweep do this)
#
# Each rung runs as `./sgemm RUNG N --once`, so the profiler sees one
# launch of one kernel.  For cublas the filter `regex:gemm` picks the
# GEMM kernel out of whatever else cuBLAS launches.
#
# Columns (metric names in brackets, so you can ask ncu for them by hand):
#   ms        kernel duration         [gpu__time_duration.sum]
#             ncu locks the GPU clock to its BASE frequency while it
#             profiles, so these times are longer than the ladder's.
#             Compare rungs with each other here, not with ladder.txt.
#   regs      registers per thread    [launch__registers_per_thread]
#   occ%      achieved occupancy: resident warps / maximum, averaged
#             over the run  [sm__warps_active.avg.pct_of_peak_sustained_active]
#   SM%       SM throughput, % of peak: the busiest of the SM's units
#             [sm__throughput.avg.pct_of_peak_sustained_elapsed]
#   DRAM%     DRAM throughput, % of peak
#             [dram__throughput.avg.pct_of_peak_sustained_elapsed]
#   DRAM_MB   megabytes read from DRAM            [dram__bytes_read.sum]
#   gld_Msec  global-load sectors (32 B), millions
#             [l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum]
#   sec/req   global-load sectors per load request: 4 when a warp reads
#             128 consecutive bytes, 32 when every lane hits its own
#             sector, 1 when every lane reads the same word
#             [... / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum]
#   wf/shld   shared-memory wavefronts per shared-load instruction: 1
#             when a warp's shared load is conflict-free or a broadcast,
#             n for an n-way bank conflict (a 64-bit or 128-bit load
#             needs 2 or 4 even without conflicts)
#             [l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum /
#              smsp__inst_executed_op_shared_ld.sum]
#
# Needs ncu on the PATH (it comes with the cuda module) and permission
# to read the GPU's counters.  If a run prints ERR_NVGPUCTRPERM, stop
# and tell the TAs.

prog=${1:?usage: $0 PROG N RUNG...}
n=${2:?usage: $0 PROG N RUNG...}
shift 2
rungs=("$@")
[ ${#rungs[@]} -eq 0 ] && { echo "no rungs given" >&2; exit 1; }
command -v ncu > /dev/null || { echo "ncu not found: module load cuda" >&2; exit 1; }

metrics=gpu__time_duration.sum,launch__registers_per_thread,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
smsp__inst_executed_op_shared_ld.sum

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "# ncu $(ncu --version | tail -1)"
echo "# N=$n  one launch per rung, clocks locked to base by ncu"
printf '%-15s %9s %5s %6s %6s %6s %9s %9s %8s %8s\n' \
    rung ms regs 'occ%' 'SM%' 'DRAM%' DRAM_MB gld_Msec sec/req wf/shld

for r in "${rungs[@]}"; do
    filter=()
    [ "$r" = cublas ] && filter=(--kernel-name regex:gemm)
    echo "profiling $r ..." >&2
    ncu --csv --log-file "$tmp/$r.csv" --metrics "$metrics" \
        "${filter[@]}" --launch-count 1 "$prog" "$r" "$n" --once \
        > "$tmp/$r.out" 2>&1
    python3 - "$tmp/$r.csv" "$r" <<'EOF'
import csv, sys
path, rung = sys.argv[1], sys.argv[2]
lines = open(path, errors="replace").read().splitlines()
start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
if start is None:
    err = [l for l in lines if "ERR" in l or "error" in l.lower()]
    print(f"{rung:<15} no data -- {err[0] if err else 'see ncu output'}")
    sys.exit(0)
m = {}
for row in csv.DictReader(l for l in lines[start:] if l.startswith('"')):
    v = row["Metric Value"].replace(",", "")
    try:
        m[row["Metric Name"]] = float(v)
    except ValueError:
        pass
def g(k): return m.get(k, float("nan"))
def ratio(a, b):
    a, b = g(a), g(b)
    return a / b if b else float("nan")
print(f"{rung:<15} {g('gpu__time_duration.sum') * 1e-6:9.3f}"
      f" {g('launch__registers_per_thread'):5.0f}"
      f" {g('sm__warps_active.avg.pct_of_peak_sustained_active'):6.1f}"
      f" {g('sm__throughput.avg.pct_of_peak_sustained_elapsed'):6.1f}"
      f" {g('dram__throughput.avg.pct_of_peak_sustained_elapsed'):6.1f}"
      f" {g('dram__bytes_read.sum') * 1e-6:9.1f}"
      f" {g('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum') * 1e-6:9.2f}"
      f" {ratio('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', 'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'):8.2f}"
      f" {ratio('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum', 'smsp__inst_executed_op_shared_ld.sum'):8.2f}")
EOF
done
