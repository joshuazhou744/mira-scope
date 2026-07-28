"""Download project assets: TeaPearce CSGO tarballs from HF hub and DINOv3 weights from Meta

example usage:
    uv run run python scripts/download_assets.py tarball --name tarball_name
    uv run python scripts/download_assets.py dino --url 'url_to_dino_weights'
"""

import argparse
import sys
import urllib.request
from pathlib import Path

HF_REPO = "TeaPearce/CounterStrike_Deathmatch"
DEFAULT_RAW_DIR = Path.home() / "data/csgo/raw"
DEFAULT_DINO_DIR = Path.home() / "data/dino_weights"

def download_tarball(name: str, out_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(HF_REPO, name, repo_type="dataset", local_dir=out_dir)
    print(f"Downloaded {name} to {path}")

def download_dino(url: str, out_dir: Path) -> None:
    name = url.split("?")[0].rsplit("/", 1)[-1]
    if not name.endswith(".pth"):
        sys.exit(f"Error: DINO weights file must end with .pth, got {name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / name

    def hook(blocks: int, block_size: int, total: int) -> None:
        done = blocks * block_size
        pct = f"{100 * done / total:5.1f}%" if total > 0 else f"{done / 1e6:.0f}MB"
        print(f"\r{name}: {pct}", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print(f"\ndone: {dest}  (set RS_DINO_WEIGHTS_DIR={out_dir})")

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tarball", help="download a tarball from HF hub")
    t.add_argument("--name", required=True, help="name of the tarball to download")
    t.add_argument("--out", type=Path, default=DEFAULT_RAW_DIR)

    d = sub.add_parser("dino", help="download DINO weights from a URL")
    d.add_argument("--url", required=True, help="URL of the DINO weights to download")
    d.add_argument("--out", type=Path, default=DEFAULT_DINO_DIR)

    args = ap.parse_args()
    if args.cmd == "tarball":
        download_tarball(args.name, args.out)
    else:
        download_dino(args.url, args.out)

if __name__ == "__main__":
    main()