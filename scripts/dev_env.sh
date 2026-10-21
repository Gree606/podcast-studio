# Source me: limits BLAS/rayon threads (shared login nodes cap threads per user).
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4
# Where the API keys live: the PODCAST_SECRETS_FILE line of the repo's git-ignored .env (a path, not a key).
if [ -z "${PODCAST_SECRETS_FILE:-}" ] && [ -f "$(dirname "${BASH_SOURCE[0]:-$0}")/../.env" ]; then
  _psf=$(grep -m1 '^PODCAST_SECRETS_FILE=' "$(dirname "${BASH_SOURCE[0]:-$0}")/../.env" | cut -d= -f2- | tr -d "\"'")
  [ -n "$_psf" ] && export PODCAST_SECRETS_FILE="$_psf"; unset _psf
fi
