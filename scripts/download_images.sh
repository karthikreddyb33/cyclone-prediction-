# Download all GridSat IR crops -- no training.
# Resumable: crops already in data/images are skipped, so just re-run if it stops.
#   bash scripts/download_images.sh
set -e
cd "$(dirname "$0")/.."
rm -rf data/tmp*   # partial files left by an interrupted run
echo "== 00/12 UTC $(date +%T)"
python -u scripts/fetch_gridsat.py --workers 10 --hours 0 12 >> outputs/fetch_gridsat_0012.log 2>&1
echo "== 06/18 UTC $(date +%T)"
python -u scripts/fetch_gridsat.py --workers 10 --hours 6 18 >> outputs/fetch_gridsat_0618.log 2>&1
rm -rf data/tmp*
echo "== done $(date +%T): $(ls data/images | wc -l) crops in data/images"
