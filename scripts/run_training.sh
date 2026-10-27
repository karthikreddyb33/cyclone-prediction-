# Train all four models and evaluate on 2025; every line of output is logged.
#   bash scripts/run_training.sh
set -eo pipefail
cd "$(dirname "$0")/.."
LOG=outputs/training_logs
mkdir -p "$LOG"
{
  echo "started   $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "host      $(hostname)"
  python -c "import sys,torch,torchvision,platform;print('python   ',sys.version.split()[0]);print('torch    ',torch.__version__,'| torchvision',torchvision.__version__);print('device   ','cuda' if torch.cuda.is_available() else 'cpu',platform.processor(),'threads',torch.get_num_threads())"
  echo "crops     $(ls data/images | wc -l) in data/images"
} | tee "$LOG/00_environment.txt"

run() {  # run <logname> <train.py args...>
  local name=$1; shift
  echo "== $name  $(date +%T)" | tee -a "$LOG/run_all.log"
  python -u train.py "$@" 2>&1 | tee "$LOG/$name.log"
}
run 01_tabular_physical          --model tabular    --features physical
run 02_tabular_no_pressure       --model tabular    --features no_pressure
run 03_multimodal_physical       --model multimodal --features physical    --backbone efficientnet_b0
run 04_multimodal_no_pressure    --model multimodal --features no_pressure --backbone efficientnet_b0
echo "== evaluate  $(date +%T)" | tee -a "$LOG/run_all.log"
python -u evaluate.py 2>&1 | tee "$LOG/05_evaluate.log"
echo "finished  $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG/00_environment.txt"
