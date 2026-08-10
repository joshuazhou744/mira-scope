> This project is a fork of MIRA (refer to its [technical report](https://arxiv.org/abs/2607.05352) for full technical details). The original project's README is preserved at [README_ORIGINAL.md](README_ORIGINAL.md).

## MIRA-SCOPE for FPS World Modelling

MIRA is a real-time, latent diffusion world model that uses flow matching to generate gameplay frame-by-frame from player actions and gameplay footage.
The 5B parameter world model was trained on full 2v2 matches with perspectives and actions from all 4 players.
The model is built on a codec that uses a frozen **DINOv3** encoder for feature extraction.
Read more and play it live at [mira-wm.com](https://mira-wm.com/).

SCOPE is an FPS world model built on Wan2.2-5B.
It inserts a module into a pretrained model that learns video game actions to condition video generation.
The zero-initialized action modules learn to turn a general video generator into a dynamic game that reacts to user inputs.
SCOPE does not generate in real-time and thus cannot be considered a video game experience.

MIRA-SCOPE adapts MIRA's autoencoder and diffusion architecture with SCOPE's localized action conditioning to generate FPS worlds.
Action decoupling separates local effects, like weapon recoil and muzzle flash, from global effects, like the stable background.
In MIRA-SCOPE, all actions stay on MIRA's Adaptive LayerNorm (AdaLN) action conditioning path while scoped actions are routed
through cross-attention modules that learn to localize effects within the generated latent.
Preserving MIRA's compact latent diffusion architecture with few-step flow matching allows us to generate in real-time, resulting in a playable video game.

Hu et al., *MIRA: Multiplayer Interactive World Models with Representation Autoencoders*, 2026. <br>
[Paper](https://arxiv.org/abs/2607.05352) · [Code](https://github.com/mira-wm/mira)

Tong et al., *SCOPE: Simulating Cross-game Operations in Playable Environments for FPS World Models*, 2026. <br>
[Paper](https://arxiv.org/abs/2605.23345) · [Code](https://github.com/z2tong/SCOPE)

### Demo Video

Samples of generated gameplay at 16 FPS recorded using the [Live Demo](#live-demo) with different checkpoints. \
See [Training](#training) for specifications of checkpoints.

https://github.com/user-attachments/assets/951460cb-6257-4d45-a5c1-e7d9770abf47

Click [here](https://www.youtube.com/watch?v=P0m3G-bCoRA) for a demo with more samples and raw gameplay.

### Action Conditioning

Action conditioning is how the world model translates player actions to in-game effects in the generated frame.
At each step it predicts the next latent from three things: past latents (context), flow matching timestep ($\tau$), and player actions (keyboard and mouse inputs) for that frame.
Actions are encoded into an embedding, `a`, then the model denoises the next latent, which the codec decodes back to video frame(s).

Conditioning happens inside every transformer block. The question is where and how `a` is injected.
**AdaLN** applies a per-channel scale and shift identically at every spatial position.
**SCOPE** additionally routes scoped actions through per-position (per-pixel) cross-attention, where each latent position attends to scoped action tokens independently.

The table below compares the two routes step by step; the diagrams that follow break down each route.

![algorithms-diagram](/public/adaln_vs_scope.png)

Left: AdaLN (baseline) derives a scale and bias vector for each sublayer and applies Adaptive LayerNorm before each sublayer operates. \
Right: SCOPE is essentially the same block with cross-attention of latent positions over scoped actions (latent positions are queries, scoped actions are keys and values).

![transformer-diagram](/public/transformer-diagram.png)

> In the actual trained model, I set `ada_attn_ln=true` in the config, a small tweak that applies AdaLN modulation to all three sublayers in a transformer block (space attention, time attention, and the MLP), each with its own $\gamma$, $\beta$.

The diagram shows that SCOPE inserts an `ActionModule` block between Time Attention and the FFN (AdaLN-Modulated MLP).

![adaln-diagram](/public/adaln-diagram.png)

The diagram shows AdaLN applied before the MLP (it's also applied before the attention sublayers, see note above). The same $\gamma$, $\beta$ pair is broadcast across the latent grid, so every position gets the same modulation.

![scope-diagram](/public/scope-diagram.png)

The diagram shows that `ActionModule` takes scoped action tokens from a separate action encoder and has the latent positions attend to them.
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
identical in every way (same codec, same data, same hyperparameters) except how scoped actions are conditioned.
Any difference in generation can be attributed to the SCOPE mechanism alone.

For one of the SCOPE action conditioned checkpoints I implemented forward filling for weapon changes to improve weapon consistency in generated
frames. This means that weapon change actions behave like a continuous state that defines the weapon type for a given frame.

### Training Specifications

#### Codec
| Feature extractor (encoder) | DINOv3-B/16 | DINOv3-L/16
| --- | --- | --- |
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
| Steps | 100k | 200k | 100k | 200k |
| Forward fill weapon changes | No | No | No | Yes |
| Final Validation LPIPS | 0.269 | 0.282 | 0.267 | 0.278 |
| Final Validation Total Loss | 0.283 | 0.284 | 0.276 | 0.273 |
| Final FDD vs. Recon at 8 frames | 6.70 | -- | 6.04 | -- |
| Final Latent Drift | 0.2403 | -- | 0.2305 | -- |
| Total params | 796M | 796M | 862M | 862M |

> Columns 2 and 4 are the ablated SCOPE and AdaLN checkpoints. Focus on these columns for metric comparison and analysis. \
> FDD vs. Recon measures how close DINO features of the world model's generated frame are to the codec's reconstructed frame. \
> A value of zero means the DINO feature distributions match exactly with the codec reconstruction, this can be thought of as the generation quality ceiling.


#### World Model 1B (trained on Codec v1)
| Action conditioning | AdaLN | SCOPE |
| --- | --- | --- |
| Diffusion transformer hyperparams | --- | 16 layers, 1024 hidden dim, 16 heads |
| Training | --- | 16 batch size, 2x5090 |
| Context Window (seconds) | --- | 2 |
| Steps | --- | 100k |
| Final Validation LPIPS | --- | 0.278 |
| Final Validation Total Loss | --- | 0.365 |
| Total params | --- | 1.27B |

> not necessarily worse, but not better than the 800M checkpoints so I didn't bother training an AdaLN arm

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
| Player teleports after staring at something featureless | Featureless views beyond context window lose positional cues and the model infers an arbitrary location, simulating teleportation | Increase training context window length |
| Inconsistent enemy interactions, player deaths are random | Enemies are never generated from their perspective and carry no conditioning signal so interactions are learned only visually | Requires game state conditioning or joint multi-view generation, MIRA's multiplayer Rocket League handles opponent interactions better because it conditions on all 4 players' actions in a contained map |

Some limitations are shared with [DIAMOND](https://youtu.be/fOF0By6fOWw?si=3oMBtEmQe0VSG1fn&t=1738).

### License

Apache License 2.0, see [LICENSE](LICENSE). 

Meta's DINOv3 License, see [DINOv3 License](DINOv3_LICENSE.md).

