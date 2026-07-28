"""Live demo websocket server

Loads the WM, bootstraps context from real clip, serves a websocket to take actions:
each incoming {"key": [...], "mouse": [dx, dy]} message advances the world by one
latent (2 video frames) and returns frames as JPEG bytes to be displayed elsewhere.

Usage:
    uv run python scripts/demo/server.py \
        --checkpoint /path/to/checkpoint --data /path/to/data/index \
        --steps 10 --compile --port some_port
"""

import argparse
import asyncio
import io
import json

import torch
import websockets
import copy
from PIL import Image

from mira.data import RocketScienceDataset
from mira.data.actions import KeyVocab
from mira.data.batch import VideoActionBatch
from mira.inference.loading import load_world_model
from mira.world_model.actions_config import ActionConfig, ActionTensors
from mira.world_model.config import WorldModelInferenceConfig


CSGO_KEYS = ("W", "A", "S", "D", "Space", "Ctrl", "Shift", "1", "2", "3", "R", "Fire", "RClick")
FPS = 16
CTX_FRAMES = 8
JPEG_QUALITY = 80
SEED = 0  # reseeded per connection so the rollout is identical from the same bootstrap context

torch.set_float32_matmul_precision("high")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="converted dataset dir (bootstrap context source)")
    ap.add_argument("--match", default=None,
                    help="match_id from index.json; with it, only that match is read (no full-dataset "
                         "scan) and --clip is the window index WITHIN it. Without it, --clip indexes "
                         "the flat list of all matches' windows.")
    ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10, help="n_diffusion_steps")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    return ap.parse_args()

def setup(args):
    """Load model, bootstrap context"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_world_model(args.checkpoint, device=device)
    model.eval()
    model.set_inference_context(CTX_FRAMES)
    if args.compile:
        model.world_model = torch.compile(model.world_model) # type: ignore
        model.codec.decoder = torch.compile(model.codec.decoder) # type: ignore
    window = model.n_context_latents + 1  # latents per streaming window
    td = model.action_temporal_downsampling  #  2 action steps per latent

    # Bootstrap: encode window*td frames of a real clip into the initial latent window.
    boot_frames = window * td
    ds = RocketScienceDataset.from_local(args.data, vocab=KeyVocab(CSGO_KEYS))
    if args.match is not None:
        # Random access: read only this match's chunk, pick window `args.clip` within it.
        if args.match not in ds.matches:
            raise SystemExit(f"--match {args.match!r} not in index.json ({len(ds.matches)} matches)")
        match_clips = ds.load_match(
            args.match, clip_len=boot_frames, target_fps=FPS, perspective=0, seed=0
        )
        if args.clip >= len(match_clips):
            raise SystemExit(
                f"--clip {args.clip} out of range for match {args.match}: only {len(match_clips)} "
                f"windows of {boot_frames} frames (valid 0-{len(match_clips) - 1})"
            )
        clip = match_clips[args.clip]
        print(f"bootstrap from {args.match} window {args.clip}")
    else:
        # No match given: stream the dataset and advance to the args.clip-th window.
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
    return model, z, actions, td, inf_cfg

def step_world(model, z, actions, inf_cfg, kv_cache, keys_mouse):
    """Advance one latent from client input.
    Returns (z, kv_cache, jpeg bytes)"""
    for keys, mouse in keys_mouse:
        actions.key_presses = torch.cat(
            [actions.key_presses, torch.tensor(keys, dtype=torch.int32).view(1, 1, -1)], dim=1
        )
        actions.mouse_movements = torch.cat(
            [actions.mouse_movements, torch.tensor(mouse).view(1, 1, 2)], dim=1
        )

    with torch.no_grad():
        z, kv_cache = model.streaming_inference_step(z, actions, kv_cache, config=inf_cfg)
        new_frames = model.decode_to_video(z[:, -1:]) # get two video frames from the last latent frame
    
    jpegs = []
    frames = (new_frames[0].permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy() # image tensor of shape (2, H, W, 3)
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=JPEG_QUALITY)
        jpegs.append(buf.getvalue())
    return z, kv_cache, jpegs

async def serve(args) -> None:
    model, z0, actions, _, inf_cfg = setup(args)
    actions0 = copy.deepcopy(actions)
    z, kv_cache = z0.clone(), None
    active = False
    print(f"Listening on ws://{args.host}:{args.port}")

    # loop over client messages (user actions) and send corresponding frames back
    async def handler(ws):
        nonlocal z, kv_cache, active
        if active:
            await ws.close(code=1013, reason="server busy")
            return
        active = True
        try:
            z = z0.clone()
            actions.key_presses = actions0.key_presses.clone()
            actions.mouse_movements = actions0.mouse_movements.clone()
            kv_cache = None
            # Reseed so the diffusion noise (torch.randn in streaming_inference_step) is identical
            # Thus each session gets same rollout from the same bootstrap context
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            print("client connected")
            async for message in ws:
                msg = json.loads(message)
                keys_mouse = list(zip(msg["keys"], msg["mouse"]))
                z, kv_cache, jpegs = step_world(model, z, actions, inf_cfg, kv_cache, keys_mouse)
                for jpeg in jpegs:
                    await ws.send(jpeg)
        finally:
            active = False
            print("client disconnected")

    async with websockets.serve(handler, args.host, args.port, max_size=None, ping_timeout=None): # listen for connections and send connections to handler
        await asyncio.Future() # park forever so websocket keeps running, killing server process kills websocket

def main() -> None:
    asyncio.run(serve(parse_args()))

if __name__ == "__main__":
    main()