#!/bin/bash
# USAGE: source scripts/setup_env.sh

# ---- default paths (override by exporting before sourcing) ----
export MIRA_DATA_DIR="${MIRA_DATA_DIR:-$HOME/data/csgo}"
export RS_DINO_WEIGHTS_DIR="${RS_DINO_WEIGHTS_DIR:-$HOME/data/dino_weights}"

mkdir -p "$MIRA_DATA_DIR/raw" "$MIRA_DATA_DIR/mira/train" "$MIRA_DATA_DIR/mira/test" "$RS_DINO_WEIGHTS_DIR"

# ---- wandb ----
if ! uv run wandb login --verify >/dev/null 2>&1; then
    read -r -s -p "wandb API key (from wandb.ai/authorize, empty to skip): " WANDB_KEY; echo
    if [[ -n "$WANDB_KEY" ]]; then
        uv run wandb login "$WANDB_KEY"
    else
        echo "wandb skipped, use wandb.mode=offline or disabled"
    fi
fi

echo "--- environment ready ---"
echo "MIRA_DATA_DIR       = $MIRA_DATA_DIR"
echo "RS_DINO_WEIGHTS_DIR = $RS_DINO_WEIGHTS_DIR"
echo "dino weights found  : $(ls "$RS_DINO_WEIGHTS_DIR" 2>/dev/null | grep -c '\.pth$') file(s)"

