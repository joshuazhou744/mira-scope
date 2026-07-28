"""Streaming generation loop benchmark/test script

Bootstraps context from a real clip, then generates latents one at a time with
`streaming_inference_step` driven by scripted action stream (hold W + sinusoidal mouse pan).
Reports s/latent and effective video fps, which answers whether `--steps` diffusion steps is real-time on this GPU.

Usage:
    uv run python scripts/demo/stream_harness.py \
        --checkpoint .../checkpoint.pth --data ~/data/csgo/mira/test \
        --latents 48 --steps 10 --out stream_harness.mp4
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from mira.data import RocketScienceDataset
from mira.data.actions import KeyVocab
from mira.data.batch import VideoActionBatch
from mira.data.viz import _encode_mp4
from mira.inference.loading import load_world_model
from mira.world_model.actions_config import ActionConfig, ActionTensors
from mira.world_model.config import WorldModelInferenceConfig

CSGO_KEYS = ("W", "A", "S", "D", "Space", "Ctrl", "Shift", "1", "2", "3", "R", "Fire", "RClick")
FPS = 16
CTX_FRAMES = 30  # 15 context latents + 1 generated = 16-latent window (the trained window)

torch.set_float32_matmul_precision("high")

def scripted_action(step: int) -> tuple[list[int], list[float]]:
    """Action for one video frame: hold W, pan the mouse in a slow sine sweep to test this stream harness before creating pygame interface for actual actions"""
    keys = [0] * len(CSGO_KEYS)
    keys[CSGO_KEYS.index("W")] = 1
    dx = 60.0 * math.sin(step / 8.0)
    return keys, [dx, 0.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="converted dataset dir (bootstrap context source)")
    ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--latents", type=int, default=48, help="latents to generate (2 frames each)")
    ap.add_argument("--steps", type=int, default=10, help="n_diffusion_steps")
    ap.add_argument("--out", default="stream_harness.mp4")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_world_model(args.checkpoint, device=device)
    model.eval()
    model.set_inference_context(CTX_FRAMES)
    if args.compile:
        model.world_model = torch.compile(model.world_model) # type: ignore
        model.codec.decoder = torch.compile(model.codec.decoder) # type: ignore
    window = model.n_context_latents + 1  # latents per streaming window
    td = model.action_temporal_downsampling  # action steps per latent (2)

    # Bootstrap: encode window*td frames of a real clip into the initial latent window.
    boot_frames = window * td
    ds = RocketScienceDataset.from_local(args.data, vocab=KeyVocab(CSGO_KEYS))
    clips = ds.iter_clips(clip_len=boot_frames, target_fps=FPS, perspective=0, seed=0)
    for _ in range(args.clip):
        next(clips)
    clip = next(clips)
    assert clip.mouse is not None and clip.frames is not None

    cfg = ActionConfig(valid_keys=list(CSGO_KEYS), source_fps=FPS, target_fps=FPS)
    actions = ActionTensors(config=cfg, batch_size=1)
    actions.key_presses = clip.actions[0].unsqueeze(0).to(torch.int32)
    actions.mouse_movements = clip.mouse[0].unsqueeze(0).to(torch.float32)
    actions.game_mouse_sensitivity = torch.full((1,), float("nan"))

    with torch.no_grad():
        batch = VideoActionBatch(video=clip.frames.clone(), actions=actions)
        z = model.init_streaming_inference(batch)  # (1, window, h, w, c)
    assert z.shape[1] == window, f"bootstrap gave {z.shape[1]} latents, expected {window}"

    inf_cfg = WorldModelInferenceConfig(
        n_diffusion_steps=args.steps, noise_level=0.0, schedule_type="linear"
    )

    frames_out = []
    kv_cache = None
    times = []
    with torch.no_grad():
        for i in range(args.latents):
            for j in range(td):
                keys, mouse = scripted_action(i * td + j)
                actions.key_presses = torch.cat(
                    [actions.key_presses, torch.tensor(keys, dtype=torch.int32).view(1, 1, -1)], dim=1
                )
                actions.mouse_movements = torch.cat(
                    [actions.mouse_movements, torch.tensor(mouse).view(1, 1, 2)], dim=1
                )

            t0 = time.perf_counter()
            z, kv_cache = model.streaming_inference_step(z, actions, kv_cache, config=inf_cfg)
            new_frames = model.decode_to_video(z[:, -1:])  # (1, td, 3, H, W) in [0, 1]
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

            frames_out.append(
                (new_frames[0].permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            )

    # Skip the first few timings (compile/warmup) when reporting steady state.
    steady = times[3:] if len(times) > 6 else times
    s_per_latent = sum(steady) / len(steady)
    video_fps = td / s_per_latent
    print(
        f"{args.latents} latents @ {args.steps} diffusion steps on {device}: "
        f"{s_per_latent * 1000:.0f} ms/latent -> {video_fps:.1f} video fps "
        f"({'REAL-TIME' if video_fps >= FPS else 'below real-time'} vs {FPS} fps target)"
    )

    Path(args.out).write_bytes(_encode_mp4(np.concatenate(frames_out), fps=FPS))
    print(f"wrote {args.out} ({args.latents * td} generated frames)")


if __name__ == "__main__":
    main()
