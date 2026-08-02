> This project is a fork of MIRA (refer to its [technical report](https://arxiv.org/abs/2607.05352) for full technical details). The original project's README is preserved at [README_ORIGINAL.md](README_ORIGINAL.md).

## MIRA-SCOPE for FPS World Modelling

https://github.com/user-attachments/assets/951460cb-6257-4d45-a5c1-e7d9770abf47

[Demo](https://www.youtube.com/watch?v=P0m3G-bCoRA) with more samples and raw gameplay.

MIRA-SCOPE is a proof of concept CS:GO world model that adapts MIRA's autoencoder and diffusion architecture to an FPS world, integrating action decoupling inspired by SCOPE. The world model is built on a codec that uses a frozen **DINOv3** encoder for feature extraction.

The idea borrowed from SCOPE is separating in-scope effects local to specific areas in generation like weapon 
recoil, muzzle flash, from out-of-scope world generation, stable background scene that updates globally with actions.
In MIRA-SCOPE, all actions stay on MIRA's Adaptive LayerNorm (AdaLN) action conditioning path while certain actions are routed
through per-block cross-attention modules allowing the model to learn localized effects in the generated latent.


Tong et al., *SCOPE: Simulating Cross-game Operations in Playable Environments for FPS World Models*, 2026. <br>
[Paper](https://arxiv.org/abs/2605.23345) · [Code](https://github.com/z2tong/SCOPE)

### SCOPE Action Conditioning

![transformer-diagram](/public/transformer-diagram.png)

SCOPE adds a step before AdaLN using `ActionModule`.
AdaLN is always present for global action conditioning. SCOPE is an additive route for specifically chosen actions.

![adaln-diagram](/public/adaln-diagram.png)

AdaLN is a global modulation of all positions of the latent grid (one gamma and beta, per channel scale and bias).

![scope-diagram](/public/scope-diagram.png)

The `ActionModule` block takes scoped actions (fire, reload, weapon change) from a separate encoder 
and attends latent grid tokens to them (latent positions are queries, scoped actions are keys and values).
This allows each region of the frame to learn action effects independently.
The output projection of `ActionModule` is zero-initialized so it's gradually learned.

### Dataset

MIRA-SCOPE trains on single-player, deathmatch-style CS:GO data. We use the CS:GO Deathmatch dataset from Pearce & Zhu: ~5,500 matches scraped 
from online CS:GO deathmatches on the Dust2 map.
Each match is ~1000 frames at 16 FPS, ~62.5 seconds of data. Each frame is matched with recorded keyboard/mouse action inputs.
The dataset provides raw keyboard inputs and mouse movements one-hot encoded over 23 x-axis and 15 y-axis bins.
The [convert_csgo_data.py](scripts/convert_csgo_data.py) script decodes the bins into scalar floats, then aggregates all actions over each pair 
of frames to match the time-downsampled latents. This is the same dataset used to train the DIAMOND CS:GO diffusion world model.

The dataset is fine to use for world model training under the [Valve Video Policy](https://store.steampowered.com/video_policy).

Pearce & Zhu, *Counter-Strike Deathmatch with Large-Scale Behavioural Cloning*, IEEE CoG 2022 (Best Paper). <br>
[Paper](https://arxiv.org/abs/2104.04258) · [Code](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning) · [Data](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch)

Alonso et al., *Diffusion for World Modeling: Visual Details Matter in Atari*, NeurIPS 2024. <br>
[Paper](https://arxiv.org/abs/2405.12399) · [Code](https://github.com/eloialonso/diamond)

### Training

We trained in two stages. First the codec, a frozen DINOv3 encoder paired with a learned decoder, trained to
compress CS:GO frames into a compact latent grid and reconstruct them back to game frames.

On top of the frozen codec we trained the world model, a flow-matching diffusion transformer that predicts the next
latent frames from past frames and player actions. To ablate SCOPE we trained two arms that are 
identical in every way (same codec, same data, same hyperparameters) except how combat actions are conditioned. A baseline
routes all actions through MIRA's AdaLN path, and a SCOPE arm that additionally routes localized combat actions through per-block
cross-attention. Any difference in generation can be attributed to the SCOPE mechanism alone.

For the one of SCOPE action conditioned checkpoints I implemented forward filling for weapon changes to improve weapon consistency in generated
frames. This means that weapon change actions behave like a continuous state that defines the weapon type for a given frame.

### Training Specifications

#### Codec
| | Codec v0 | Codec v1 |
| --- | --- | --- |
| Feature extractor (encoder) | DINOv3-B/16 | DINOv3-L/16
| Aggregation layers | [2, 5, 8, 11] | [11, 13, 15, 17, 19, 21, 23]
| ViT decoder | 1024 width, 24 depth | 1152 width, 28 depth |
| Latent grid | 5×9×32, 2× temporal | 5×9×32, 2× temporal |
| Training | 16 batch size, 2×4090 | 8 batch size (global batch 16), 2×5090  |
| Steps | 100k | 100k |
| Final Validation LPIPS | 0.184 | 0.193 |
| Final Validation Total Loss | 0.235 | 0.244 |
| Total params | 497M | 900M |


#### World Model 800M (trained on Codec v0)
| Action conditioning | AdaLN | AdaLN | SCOPE | SCOPE
| --- | --- | --- | --- | --- |
| Diffusion transformer hyperparams | 16 layers, 1024 hidden dim, 16 heads | 16 layers, 1024 hidden dim, 16 heads | 16 layers, 1024 hidden dim, 16 heads | 16 layers, 1024 hidden dim, 16 heads |
| Training | 16 batch size, 2x5090 | 16 batch size, 2x5090 | 16 batch size, 2x5090 | 16 batch size, 2x5090 |
| Context Window (seconds) | 2 | 3 | 2 | 3 |
| Steps | 100k | 100k | 200k | 200k |
| Forward fill weapon changes | No | No | No | Yes |
| Final Validation LPIPS | 0.269 | 0.282 | 0.267 | 0.278 |
| Final Validation Total Loss | 0.283 | 0.284 | 0.276 | 0.273 |
| Total params | 796M | 796M | 862M | 862M |


#### World Model 1B (trained on Codec v1)
| | AdaLN | SCOPE |
| --- | --- | --- |
| Diffusion transformer hyperparams | --- | 16 layers, 1024 hidden dim, 16 heads |
| Training | --- | 16 batch size, 2x5090 |
| Context Window (seconds) | --- | 2 |
| Steps | --- | 100k |
| Final Validation LPIPS | --- | 0.278 |
| Final Validation Total Loss | --- | 0.365 |
| Total params | --- | 1.27B |

> not necessarily worse, but not better than the 800M checkpoints

### Reproducing the Training (on RunPod)

1. **Install the environment** and download the gated DINOv3 encoder weights, see the `Installation` and `Training` sections of [README_ORIGINAL.md](README_ORIGINAL.md). You need `RS_DINO_WEIGHTS_DIR` pointing at the DINOv3 `.pth` (world-model training/inference don't need it, codec training does).
    - 1.2B checkpoint uses `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`, 800M checkpoint uses `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
2. **Train the codec** using steps (clone/install, environment, dataset build from the TeaPearce HDF5 files, and the `torchrun` launch) from: [train_codec](train_codec.md).
3. **Train the world model** on a finished codec checkpoint, for either the AdaLN baseline or the SCOPE arm: [train_wm.md](train_wm.md).

### Validations

<table>
<tr>
<td align="center">SCOPE 862M Final Validation</td>
<td align="center">AdaLN 796M Final Validation</td>
</tr>
<tr>
<td width="50%">

https://github.com/user-attachments/assets/8da2f192-4070-4346-bada-5d4bd2eaf225

</td>
<td width="50%">

https://github.com/user-attachments/assets/0842c520-8a09-4eed-9919-2f842a361074

</td>
</tr>
</table>

<table>
<tr>
<td>DINO-L Codec Reconstruction Final Validation</td>
</tr>
<tr>
<td>

https://github.com/user-attachments/assets/f5a68707-1229-4698-a437-e4a7c4d6d5ed

</td>
</tr>

<tr>
<td>DINO-B Codec Reconstruction Final Validation</td>
</tr>
<tr>
<td>

https://github.com/user-attachments/assets/188bda26-ec99-47bf-9438-eaf693b62ae3

</td>
</tr>
</table>

> Batch size 8 for DINO-L, 16 for DINO-B (validation batch size matches training)

### Live Demo

This project includes a script [stream_harness.py](scripts/demo/stream_harness.py) to test world model checkpoints at specific configs (diffusion steps, generation length, generation initial frames).

It also includes [server.py](scripts/demo/server.py) and [client.py](scripts/demo/client.py) scripts to set up a websocket for bidirectional 
communication between a world model generating frames (server) and a pygame client that sends user inputs (actions) and displays rollouts 
accordingly from the server.

```bash
# start server
# --compile for slow start but faster generation
uv run python scripts/demo/server.py \
    --checkpoint path/to/checkpoint.pth \
    --data $MIRA_DATA_DIR/path/to/index \
    --match <match_name_in_index> \
    --clip <clip_index_in_match> \
    --host 0.0.0.0 \
    --port 8765 \
    --steps <n_diffusion_steps> \
    --compile

# if the server runs on a remote pod, tunnel its port to your local machine
ssh -N -L 8765:localhost:8765 root@<POD_IP> -p <POD_PORT> -i ~/.ssh/id_ed25519

# start client (on your local machine)
uv run python scripts/demo/client.py \
    --server ws://localhost:8765 \
    --record <out_file.mp4> \
    --sens-x 6.0 \
    --sens-y 3.0
```

### Limitations

| Observation | Cause | Fix |
| --- | --- | --- |
| Weapon actions unstable, changing weapons back and forth isn't preserved | Weapons are tracked by change, not a continuous state | Forward-filling weapon change actions allows the model to learn from continuous state rather than re-generating from previous frames |
| Player teleports locations after staring at wall | Featureless views beyond context window loses positional cues and the infers an arbitrary location simulating teleportation | Increase training context window length |
| Inconsistent enemy interactions, player deaths are random | Enemies have no generation from their perspective nor conditioning signal so interactions are learned visually | Requires game state conditioning or joint multi-generation, MIRA's multiplayer Rocket League handles opponent interactions better because it conditions on all 4 players' actions in a contained map |

Some limitations are shared with [DIAMOND](https://youtu.be/fOF0By6fOWw?si=3oMBtEmQe0VSG1fn&t=1738).

### License

Apache License 2.0, see [LICENSE](LICENSE). 

Meta's DINOv3 License, see [DINOv3 License](DINOv3_LICENSE.md).

