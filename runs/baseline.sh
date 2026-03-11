#!/bin/bash

# Baseline training run for RTX 3060 Laptop (6GB VRAM)
# This mirrors the leaderboard methodology (same optimizer, schedulers, etc.)
# but with a smaller model (d6) sized for single-GPU consumer hardware.
#
# The goal is to establish a reproducible baseline for tracking improvements.
# Any code change that improves this baseline will also improve 8xH100 runs.
#
# Run as:
#   PYTHONIOENCODING=utf-8 uv run --extra gpu bash runs/baseline.sh
# or step by step manually.

export PYTHONIOENCODING=utf-8
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# wandb run name - change this for each new experiment
WANDB_RUN=${WANDB_RUN:-"baseline-d6-v1"}
MODEL_TAG=${MODEL_TAG:-"d6_baseline_v1"}

# --- Phase 1: Data & Tokenizer (skip if already cached) ---
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Training tokenizer..."
    python -m nanochat.dataset -n 8
    python -m scripts.tok_train --max-chars=2000000000
    python -m scripts.tok_eval
fi

# --- Phase 2: Base model pretraining ---
# RTX 3060 6GB constraints:
#   depth=6         -> ~23M params (vs d24 ~800M on leaderboard)
#   head-dim=64     -> smaller heads (vs 128 default)
#   window-pattern=L -> full attention (no FA3, SDPA doesn't support sliding window well)
#   max-seq-len=512 -> shorter context (vs 2048 default)
#   device-batch-size=4 -> VRAM limited (vs 16-32 on H100)
#   total-batch-size=16384 -> grad accum compensates (4*512=2048 per fwd, so 8 accum steps)
#
# We use target-param-data-ratio=10.5 (compute-optimal) matching the leaderboard methodology.
# eval-every=100 gives nice dense curves for wandb graphs.
# core-metric runs at the end for final score.

python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=4 \
    --total-batch-size=16384 \
    --target-param-data-ratio=10.5 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=999999 \
    --core-metric-max-per-task=500 \
    --sample-every=500 \
    --run=$WANDB_RUN \
    --model-tag=$MODEL_TAG

# --- Phase 3: Base model evaluation ---
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

echo "Done! Check wandb for graphs: https://wandb.ai"
