#!/usr/bin/env bash
# Run the whole MoE speed experiment suite from the repo root:
#   bash experiments/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== 1/4 micro benchmark (all expert implementations) ====="
python experiments/bench_micro.py

echo
echo "===== 2/4 end-to-end benchmark (real trainer settings, 2 GPUs) ====="
python experiments/bench_e2e.py --iters 10

echo
echo "===== 3/4 non-expert ablation ====="
python experiments/bench_ablation.py

echo
echo "===== 4/4 tiny training (all variants, synthetic task) ====="
python experiments/train_tiny.py --steps 150 --log-every 50
