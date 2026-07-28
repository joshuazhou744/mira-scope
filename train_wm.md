# World model training on RunPod

```bash
cd /workspace/mira-scope
```

## Environment
```bash
export MIRA_DATA_DIR=/workspace/data/csgo
export RS_DINO_WEIGHTS_DIR=/workspace/data/dino_weights
source scripts/setup_env.sh
```

## Verify the codec checkpoint is where we think
```bash
ls -lh path_to_codec_checkpoint
```

## Train WM
```bash
uv run python scripts/train_world_model.py \
  model.architecture.config.codec_checkpoint=path_to_codec_checkpoint \
  dataset.train_index=$MIRA_DATA_DIR/mira/train \
  dataset.test_index=$MIRA_DATA_DIR/mira/test \
  other.configs=set_here
```