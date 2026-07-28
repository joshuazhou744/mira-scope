# Codec training on RunPod

## Clone + install
```bash
cd /workspace/mira-scope && git pull     # or clone if fresh volume
uv sync --extra train --extra decode --extra hf --extra eval
uv run python -c "import torch; print(torch.cuda.is_available())"
apt-get install -y ffmpeg                # converter needs it if template lacks it
```

## Environment
```bash
export MIRA_DATA_DIR=/workspace/data/csgo
export RS_DINO_WEIGHTS_DIR=/workspace/data/dino_weights
source scripts/setup_env.sh
```

## Feature extractor (DINO weights)
Ensure sure you have a DINOv3 checkpoint `.pth` file at `RS_DINO_WEIGHTS_DIR`.

Copy to pod:
```bash
scp -P <pod_port> /path/to/checkpoint root@<pod_IP>:/path/to/dino_weights
```

## Build the dataset
Split by source tarball: hdf5_dm_july2021_5401_to_5500 (100 files, ~100 min) = TEST ONLY,
14 other tarballs (~2800 files, ~47 h) = TRAIN. One output shard per source tar.

### CLEAN START ONLY
```bash
rm -rf $MIRA_DATA_DIR/mira/train $MIRA_DATA_DIR/mira/test
mkdir -p $MIRA_DATA_DIR/mira/train $MIRA_DATA_DIR/mira/test
```

Test split (one tar, all 100 files):
```bash
T=hdf5_dm_july2021_5401_to_5500
uv run python scripts/download_assets.py tarball --name $T.tar --out $MIRA_DATA_DIR/raw
uv run python scripts/convert_csgo_data.py --from-tar $MIRA_DATA_DIR/raw/$T.tar \
  --out $MIRA_DATA_DIR/mira/test --shard-name dataset_$T.tar
rm $MIRA_DATA_DIR/raw/$T.tar
```

Train split (loop: download -> convert straight from tar -> delete -> next):
```bash
TARS=(hdf5_dm_july2021_1_to_200 hdf5_dm_july2021_201_to_400 hdf5_dm_july2021_401_to_600 \
      hdf5_dm_july2021_601_to_800 hdf5_dm_july2021_801_to_1000 hdf5_dm_july2021_1001_to_1200 \
      hdf5_dm_july2021_1201_to_1400 hdf5_dm_july2021_1401_to_1600 hdf5_dm_july2021_1601_to_1800 \
      hdf5_dm_july2021_1801_to_2000 hdf5_dm_july2021_2001_to_2200 hdf5_dm_july2021_2201_to_2400 \
      hdf5_dm_july2021_2401_to_2600 hdf5_dm_july2021_2601_to_2800)

TARS=(hdf5_dm_july2021_2801_to_3000 hdf5_dm_july2021_3001_to_3200 hdf5_dm_july2021_3201_to_3400 \
        hdf5_dm_july2021_3401_to_3600 hdf5_dm_july2021_3601_to_3800 hdf5_dm_july2021_3801_to_4000 \
        hdf5_dm_july2021_4001_to_4200 hdf5_dm_july2021_4201_to_4400 hdf5_dm_july2021_4401_to_4600 \
        hdf5_dm_july2021_4601_to_4800 hdf5_dm_july2021_4801_to_5000 hdf5_dm_july2021_5001_to_5200 \
        hdf5_dm_july2021_5201_to_5400)


for T in "${TARS[@]}"; do
  uv run python scripts/download_assets.py tarball --name $T.tar --out $MIRA_DATA_DIR/raw
  uv run python scripts/convert_csgo_data.py --from-tar $MIRA_DATA_DIR/raw/$T.tar \
    --out $MIRA_DATA_DIR/mira/train --shard-name dataset_$T.tar --append
  rm $MIRA_DATA_DIR/raw/$T.tar
done
```

## Train Codec
```bash
uv run python scripts/train_codec.py \
  dataset.train_index=$MIRA_DATA_DIR/mira/train \
  dataset.test_index=$MIRA_DATA_DIR/mira/test
  other.configs=set_here
```