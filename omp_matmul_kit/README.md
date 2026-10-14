# OpenMP block matrix multiplication - run kit

Files: `MatMul_omp.c` (fixed OpenMP kernel), `main.c` (timer with omp_get_wtime + validation),
`matrix.h`, `Makefile` (original, gcc -Wall -fopenmp), `run_all.sh` (one-command benchmark).

## Run
    ./run_all.sh            # full: auto-calibrate N (1 thread ~20 s), 1x/2x/4x, 1-4 threads, 10 runs
    ./run_all.sh -t 2 -r 2  # quick test (about 1-2 minutes)
    ./run_all.sh -n 1744    # fixed N for 1x instead of calibration
Options: -t target sec, -r runs, -T max threads, -b block size, -n N.

## Run a single case manually
    make && ./main <N> <threads> <runs> [b]     # e.g. ./main 1744 2 10 16

Output: results.csv (every run), summary.txt (avg time, speedup, efficiency tables).
Needs: gcc with OpenMP, make, bash, awk (Linux / macOS with gcc / WSL).
Note: threads beyond your physical core count will not give more speedup.
