"""Live demo pygame client

Polls keboard/mouse at 16Hz, sends action pairs (2 per latent tick) to the websocket server,
displays the JPEG frames it returns. User controls world model generation.

Usage:
    uv run python scripts/demo/client.py --server ws://<pod_ip>:8765
"""

import argparse
import time
import io
import json

import pygame
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from websockets.sync.client import connect

CSGO_KEYS = ("W", "A", "S", "D", "Space", "Ctrl", "Shift", "1", "2", "3", "R", "Fire", "RClick")
# pygame keymap
KEYMAP = {
    "W": pygame.K_w, "A": pygame.K_a, "S": pygame.K_s, "D": pygame.K_d,
    "Space": pygame.K_SPACE, "Ctrl": pygame.K_LCTRL, "Shift": pygame.K_LSHIFT,
    "1": pygame.K_1, "2": pygame.K_2, "3": pygame.K_3, "R": pygame.K_r,
}
FPS = 16
TD = 2 # action samples per server message (one latent frame)
FRAME_W, FRAME_H = 288, 160 # native model resolution
SCALE = 1 # display upscale
SENS_X, SENS_Y = 6.0, 3.0 # px -> mouse-dot multipliers, per axis (see bin ranges below)

MOUSE_X_BINS = [-1000.0, -500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, 0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0, 1000.0]
MOUSE_Y_BINS = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, 0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]

KEYS = ["W", "A", "S", "D", "1", "2", "3", "R", "Fire"]
KEYS_H = 64
_FONT = None

def draw_keys(screen, keys, y0, h):
    global _FONT
    if _FONT is None:
        _FONT = pygame.font.SysFont("consolas", 16, bold=True)
    n = len(KEYS)
    w = screen.get_width()
    box = min(h - 12, w // n - 4)
    pad = (w - n * box) // (n + 1)
    screen.fill((20, 20, 24), (0, y0, w, h))
    x = pad
    for name in KEYS:
        on = keys[CSGO_KEYS.index(name)]
        rect = pygame.Rect(x, y0 + (h - box) // 2, box, box)
        pygame.draw.rect(screen, (60, 220, 90) if on else (48, 48, 54), rect, border_radius=6)
        label = _FONT.render(name, True, (10, 10, 10) if on else (140, 140, 140))
        screen.blit(label, label.get_rect(center=rect.center))
        x += box + pad

def snap(value: float, bins: list[float]) -> float:
    """Nearest bin center, so every delta we send is a value the model trained on."""
    return min(bins, key=lambda b: abs(b - value))

def sample_input(sens_x: float, sens_y: float) -> tuple[list[int], list[float]]:
    """Poll one action sample: 13-key multi-hot list + bin-snapped mouse deltas [dx, dy]"""
    pygame.event.pump() # refresh input state
    pressed = pygame.key.get_pressed()
    buttons = pygame.mouse.get_pressed()

    keys = [0] * len(CSGO_KEYS)
    for name, code in KEYMAP.items():
        keys[CSGO_KEYS.index(name)] = int(pressed[code])
    keys[CSGO_KEYS.index("Fire")] = int(buttons[0]) # left click
    keys[CSGO_KEYS.index("RClick")] = int(buttons[2]) # right click

    cx, cy = FRAME_W * SCALE // 2, FRAME_H * SCALE // 2
    mx, my = pygame.mouse.get_pos()
    dx, dy = mx - cx, my - cy
    pygame.mouse.set_pos((cx, cy))
    return keys, [snap(dx * sens_x, MOUSE_X_BINS), snap(dy * sens_y, MOUSE_Y_BINS)]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", nargs="?", const=f"recording_{int(time.time())}.mp4", default=None, help="save session recording as mp4, can specify output file or fallback to default, no arg means no recording")
    ap.add_argument("--server", default="ws://localhost:8765")
    ap.add_argument("--sens-x", type=float, default=SENS_X)
    ap.add_argument("--sens-y", type=float, default=SENS_Y)
    ap.add_argument("--show_keys", action="store_true", help="draw a key-press overlay below the frame")
    args = ap.parse_args()

    keys_h = KEYS_H if args.show_keys else 0

    pygame.init()
    screen = pygame.display.set_mode((FRAME_W * SCALE, FRAME_H * SCALE + keys_h))
    pygame.display.set_caption("MIRA-SCOPE live demo")
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    recorded = []
    try:
        with connect(args.server, max_size=None, ping_timeout=None) as ws:
            print("connected to server")
            keys_buf, mouse_buf, frame_queue = [], [], []
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                keys, mouse = sample_input(args.sens_x, args.sens_y)
                keys_buf.append(keys)
                mouse_buf.append(mouse)

                if len(keys_buf) == TD:
                    ws.send(json.dumps({"keys": keys_buf, "mouse": mouse_buf}))
                    keys_buf, mouse_buf = [], []
                    for _ in range(TD):
                        frame_queue.append(ws.recv())

                if frame_queue:
                    jpeg = frame_queue.pop(0)
                    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                    surface = pygame.image.frombytes(img.tobytes(), img.size, "RGB")
                    frame_surf = pygame.transform.scale(surface, (FRAME_W * SCALE, FRAME_H * SCALE))
                    screen.blit(frame_surf, (0, 0))
                    if args.show_keys:
                        draw_keys(screen, keys, FRAME_H * SCALE, keys_h)
                    pygame.display.flip()
                    if args.record is not None:
                        recorded.append(pygame.surfarray.array3d(screen).swapaxes(0, 1))

                clock.tick(FPS)
    finally:
        if args.record is not None and recorded:
            imageio.mimwrite(args.record, recorded, fps=FPS, quality=8)
            print(f"saved {len(recorded)} frames ({len(recorded)/FPS:.1f} seconds) to {args.record}")
        pygame.quit()

if __name__ == "__main__":
    main()

