# Source me: limits BLAS/rayon threads (shared login nodes cap threads per user).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4
