"""Convert TeaPearce CSGO deathmatch HDF5 files into dataset format supported by RocketScienceDataset (mp4 + jsonl + index.json)

TeaPearce format (per frame i in 0..999):
    frame_i_x: (150, 280, 3) image
    frame_i_y: (51,) vector of actions
        [0:11] keys WASD, Space, Ctrl, Shift, 1, 2, 3, R
        [11] fire
        [12] ADS
        [13:36] mouse_x one-hot encoded over 23 bins
        [36:51] mouse_y one-hot encoded over 15 bins

Files are converted in parallel (one process per file); tar members are written serially
as results arrive, so the output shard is deterministic.
"""

import argparse
import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from mira.data.dataset import chunk_key
from mira.data.schema import Index, MatchEntry, Perspective
from mira.data.viz import _encode_mp4

# key names in _y index order
KEYS = ["W", "A", "S", "D", "Space", "Ctrl", "Shift", "1", "2", "3", "R", "Fire", "RClick"]

# mouse bin center values from TeaPearce config
MOUSE_X_POSSIBLES = [-1000.0, -500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, 0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0, 1000.0]
MOUSE_Y_POSSIBLES = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, 0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]

FPS = 16  # TeaPearce capture rate


def decode_y(y: np.ndarray) -> dict:
    """Decode one frame's 51-dim action vector into {"keys": [...], "mouse": [dx, dy]}"""
    assert y.shape == (51,)
    keys = [KEYS[k] for k in range(13) if y[k] == 1]
    dx = MOUSE_X_POSSIBLES[int(np.argmax(y[13:36]))]
    dy = MOUSE_Y_POSSIBLES[int(np.argmax(y[36:51]))]
    return {"keys": keys, "mouse": [dx, dy]}


def convert_file(hdf5_source) -> tuple[bytes, bytes, int]:
    """One TeaPearce hdf5 (path or file-like) -> (mp4 bytes, jsonl bytes, n_frames)"""
    with h5py.File(hdf5_source, "r") as f:
        n_frames = len(f.keys()) // 4
        frames = np.stack([np.array(f[f"frame_{i}_x"]) for i in range(n_frames)])
        actions = [decode_y(np.array(f[f"frame_{i}_y"])) for i in range(n_frames)]
    video = _encode_mp4(frames, fps=FPS)
    jsonl = ("\n".join(json.dumps(a) for a in actions) + "\n").encode()
    return video, jsonl, n_frames


def convert_source(spec: tuple[str, str, str | None]) -> tuple[str, bytes, bytes, int]:
    """Worker: convert one source. spec = (match_id, path, tar_member_or_None).

    When tar_member is set, `path` is a source tarball and the worker opens it itself
    (tar-member file objects cannot be shared across processes).
    """
    match_id, path, tar_member = spec
    try:
        if tar_member is not None:
            with tarfile.open(path) as src:
                video, jsonl, n_frames = convert_file(src.extractfile(tar_member))
        else:
            video, jsonl, n_frames = convert_file(path)
    except Exception as e:
        # A corrupt/truncated hdf5 (or ffmpeg failure) shouldn't kill the whole batch: signal the
        # failure with video=None and carry the error string in place of the jsonl; main skips it.
        return match_id, None, str(e).splitlines()[0], 0
    return match_id, video, jsonl, n_frames


def list_sources(args) -> list[tuple[str, str, str | None]]:
    """(match_id, path, tar_member_or_None) for every hdf5 in --from-tar and/or file paths."""
    sources = []
    if args.from_tar:
        with tarfile.open(args.from_tar) as src:
            for m in sorted(src.getmembers(), key=lambda m: m.name):
                if m.name.endswith(".hdf5"):
                    sources.append((Path(m.name).stem.replace(".", "_"), args.from_tar, m.name))
    for path in args.hdf5_files:
        sources.append((Path(path).stem.replace(".", "_"), path, None))
    return sources


def build_entry(match_id: str, n_frames: int, shard_name: str) -> MatchEntry:
    """Index entry for one converted hdf5 file: 1 player and 1 chunk of n_frames"""
    return MatchEntry(
        match_id=match_id,
        shard=shard_name,
        n_players=1,
        chunk_frames=[n_frames],
        perspectives=[
            Perspective(
                player_id=0,
                team=0,
                frames=n_frames,
                duration=n_frames / FPS,
            )
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5_files", nargs="*", help="extracted .hdf5 files (optional if --from-tar)")
    ap.add_argument("--from-tar", help="read .hdf5 members directly from a source tarball (no extraction)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-name", default="dataset_00.tar")
    ap.add_argument("--append", action="store_true", help="append to existing index.json instead of overwriting")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel conversion processes")
    args = ap.parse_args()
    if not args.from_tar and not args.hdf5_files:
        ap.error("provide hdf5 files and/or --from-tar")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_path = out / args.shard_name
    if shard_path.exists():
        raise SystemExit(
            f"shard {shard_path} already exists — refusing to overwrite. "
            f"Pick a distinct --shard-name per conversion batch (or delete the partial shard)."
        )

    sources = list_sources(args)
    print(f"converting {len(sources)} files with {args.workers} workers")
    entries = []
    with tarfile.open(shard_path, "w") as tar, ProcessPoolExecutor(max_workers=args.workers) as pool:
        skipped = []
        for match_id, video, jsonl, n_frames in pool.map(convert_source, sources, chunksize=1):
            if video is None:
                skipped.append(match_id)
                print(f"SKIPPING {match_id}: conversion failed ({jsonl})")
                continue
            key = chunk_key(match_id, 0)
            for name, data in [(f"{key}.p0.mp4", video), (f"{key}.p0.jsonl", jsonl)]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            entries.append(build_entry(match_id, n_frames, args.shard_name))
            print(f"converted {match_id}: {n_frames} frames, {len(video)/1e6:.1f} MB mp4")

    index_path = out / "index.json"
    if args.append and index_path.exists():
        existing = Index.load(index_path)
        old_ids = {e.match_id for e in existing.entries}
        dups = [e.match_id for e in entries if e.match_id in old_ids]
        if dups:
            print(f"WARNING: {len(dups)} matches already in index, keeping existing entries: {dups[:5]}...")
        entries = existing.entries + [e for e in entries if e.match_id not in old_ids]
    index = Index(total_samples=len(entries), entries=entries)
    index_path.write_text(index.model_dump_json(indent=2))

    print(f"wrote {out}/index.json with {len(entries)} entries")
    if skipped:
        print(f"skipped {len(skipped)} corrupt/failed files: {skipped}")


if __name__ == "__main__":
    main()
