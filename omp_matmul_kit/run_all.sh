#!/usr/bin/env bash
# run_all.sh - benchmark OpenMP block matrix multiplication
# Workload 1x/2x/4x  x  threads 1..4  x  N runs  -> avg time + speedup tables
#
# Usage: ./run_all.sh [options]
#   -t SEC   target time of 1 thread @ 1x workload (default 20)
#   -r RUNS  runs per setting (default 10)
#   -T MAX   max threads (default 4)
#   -b B     block size (default 16)
#   -n N     use this N for 1x directly (skip auto-calibration)
#   -h       help
# Quick test:  ./run_all.sh -t 2 -r 2
set -euo pipefail
cd "$(dirname "$0")"

TARGET=20; RUNS=10; MAXT=4; B=16; N1=""
while getopts "t:r:T:b:n:h" o; do
  case $o in
    t) TARGET=$OPTARG;; r) RUNS=$OPTARG;; T) MAXT=$OPTARG;;
    b) B=$OPTARG;; n) N1=$OPTARG;;
    h|*) sed -n '2,13p' "$0"; exit 0;;
  esac
done

echo "== 1) Compile with Makefile =="
make clean >/dev/null 2>&1 || true
make 2>&1 | sed 's/^/   /'
[ -x ./main ] || { echo "compile failed"; exit 1; }

CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "?")
echo "== Machine: $CORES logical cores (threads > cores will not speed up) =="

# round N to a multiple of B
roundN() { awk -v n="$1" -v b="$B" 'BEGIN{printf "%d", int(n/b+0.5)*b}'; }

if [ -z "$N1" ]; then
  echo "== 2) Calibrate N so that 1 thread @ 1x ~ ${TARGET}s =="
  N0=768
  T0=$(./main $N0 1 1 $B | awk -F, '{print $5}')
  N1=$(roundN "$(awk -v n=$N0 -v t=$T0 -v T=$TARGET 'BEGIN{print n*exp(log(T/t)/3)}')")
  # refine once with a larger probe (cache effects make small N look faster)
  NP=$(roundN "$(awk -v n=$N1 'BEGIN{print n/2}')")
  TP=$(./main $NP 1 1 $B | awk -F, '{print $5}')
  N1=$(roundN "$(awk -v n=$NP -v t=$TP -v T=$TARGET 'BEGIN{print n*exp(log(T/t)/3)}')")
fi
N2=$(roundN "$(awk -v n=$N1 'BEGIN{print n*exp(log(2)/3)}')")
N4=$(roundN "$(awk -v n=$N1 'BEGIN{print n*exp(log(4)/3)}')")
echo "   N: 1x=$N1  2x=$N2  4x=$N4  (block=$B, runs=$RUNS)"

echo "== 3) Run benchmark (this can take a long time) =="
CSV=results.csv
echo "workload,N,threads,b,run,time,status" > $CSV
for W in 1 2 4; do
  eval N=\$N$W
  for T in $(seq 1 $MAXT); do
    echo "   workload ${W}x  N=$N  threads=$T ..."
    ./main $N $T $RUNS $B | sed "s/^/${W}x,/" | tee -a $CSV | awk -F, '{printf "      run %s: %ss %s\n",$5,$6,$7}'
  done
done

echo "== 4) Summary =="
awk -F, -v maxt=$MAXT '
NR>1 { k=$1" "$3; s[k]+=$6; c[k]++; if($7!="Passed") bad++ }
END {
  split("1x 2x 4x", W, " ")
  print "\nAvg Exe Time (seconds)"
  printf "%-10s %12s %12s %12s\n","","Workload=1x","Workload=2x","Workload=4x"
  for(t=1;t<=maxt;t++){ printf "Threads=%-2d", t
    for(i=1;i<=3;i++){ k=W[i]" "t; a[k]=s[k]/c[k]; printf " %12.3f", a[k] }
    printf "\n" }
  print "\nSpeedup (T1/Tn)"
  printf "%-10s %12s %12s %12s\n","","Workload=1x","Workload=2x","Workload=4x"
  for(t=1;t<=maxt;t++){ printf "Threads=%-2d", t
    for(i=1;i<=3;i++){ printf " %12.3f", a[W[i]" 1"]/a[W[i]" "t] }
    printf "\n" }
  print "\nEfficiency (Speedup/Threads)"
  for(t=1;t<=maxt;t++){ printf "Threads=%-2d", t
    for(i=1;i<=3;i++){ printf " %12.3f", a[W[i]" 1"]/a[W[i]" "t]/t }
    printf "\n" }
  printf "\nValidation: %s\n", (bad>0 ? bad" run(s) FAILED" : "all runs Passed")
}' $CSV | tee summary.txt
echo; echo "Raw times: $CSV   Summary: summary.txt"
