#!/bin/bash

# Test 2:4 activation sparsity on A100.
# Runs two back-to-back: dense baseline, then sparse24, both d26.
# Compare wall clock time and CORE metric between the two.
#
# Usage (single A100, full d26 run):
#   bash runs/sparse24.sh
# Usage (quick d12 test, 500 steps):
#   DEPTH=12 NITERS=500 bash runs/sparse24.sh
# Usage (8x A100):
#   NGPU=8 bash runs/sparse24.sh
# Usage with wandb:
#   WANDB_RUN=sparse24-test bash runs/sparse24.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# --- Python venv setup ---
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# --- Config ---
NGPU=${NGPU:-1}
WANDB_RUN=${WANDB_RUN:-"dummy"}
DEPTH=${DEPTH:-26}
DBS=${DBS:-16}  # device-batch-size (A100 80GB can handle 16)
RATIO=${RATIO:-8.25}
NITERS=${NITERS:-"-1"}  # -1 = use target-param-data-ratio, else explicit step count

if [ "$NGPU" -gt 1 ]; then
    RUN_CMD="torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train --"
else
    RUN_CMD="python -m scripts.base_train"
fi

# Single-GPU: no FA3, must use full attention
WINDOW_ARGS=""
if [ "$NGPU" -eq 1 ]; then
    WINDOW_ARGS="--window-pattern=L"
fi

# Quick test mode: override eval/sample/core settings
ITER_ARGS="--target-param-data-ratio=$RATIO"
EVAL_ARGS="--eval-every=250 --sample-every=2000"
if [ "$NITERS" -ne -1 ] 2>/dev/null; then
    ITER_ARGS="--num-iterations=$NITERS"
    EVAL_ARGS="--eval-every=100 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1"
fi

# --- Data & Tokenizer (skip if cached) ---
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "=== Downloading data and training tokenizer ==="
    python -m nanochat.dataset -n 8
    python -m nanochat.dataset -n 370 &
    DATASET_PID=$!
    python -m scripts.tok_train
    python -m scripts.tok_eval
    echo "Waiting for full dataset download..."
    wait $DATASET_PID
else
    # Still need enough data shards for d26 training
    python -m nanochat.dataset -n 370
fi

# --- Run 1: Dense baseline ---
echo ""
echo "============================================"
echo "  RUN 1: Dense baseline (d${DEPTH})"
echo "============================================"
$RUN_CMD \
    --depth=$DEPTH \
    $ITER_ARGS \
    --device-batch-size=$DBS \
    $WINDOW_ARGS \
    $EVAL_ARGS \
    --run=${WANDB_RUN}-dense \
    --model-tag=d${DEPTH}_dense

# --- Run 2: Sparse24 ---
echo ""
echo "============================================"
echo "  RUN 2: Sparse24 (d${DEPTH})"
echo "============================================"
$RUN_CMD \
    --depth=$DEPTH \
    $ITER_ARGS \
    --device-batch-size=$DBS \
    --sparse24 \
    $WINDOW_ARGS \
    $EVAL_ARGS \
    --run=${WANDB_RUN}-sparse24 \
    --model-tag=d${DEPTH}_sparse24

echo ""
echo "============================================"
echo "  Both runs complete. Compare in wandb."
echo "============================================"
