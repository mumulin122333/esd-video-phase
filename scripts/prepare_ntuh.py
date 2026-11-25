#!/usr/bin/env python
"""NTUH (in-house): 1 fps cropped frames, video windows and the case-level 5-fold split.

    python scripts/prepare_ntuh.py --workers 64

Split: configs/splits/ntuh_5fold.tsv (the Experiment 1 split; cases are pseudonyms NTUH_01-13).
  * every hand-cut clip of classes 2-5, 7, 8-9, 13-14 is extracted at 1 fps, cropped to the endoscopic
    view and letterboxed to 512x448; each clip has one phase, so each window has a single-segment target;
  * train windows: 32 s, stride 8 s (clips shorter than 32 s: one window of the clip length), then at
    most --cap windows per phase (seeded); val / test windows tile each clip (windows.tile);
  * external/ntuh_all.jsonl is the union of the five test folds (all 13 cases), used to test models
    that were not trained on NTUH.
Outputs under <work_root>/windows/: ntuh_5fold/fold<k>/{train,val,test}{,_clips}.jsonl,
external/ntuh_all{,_clips}.jsonl, clips/ntuh/... (MP4), and summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esdvideo.index import read_split  # noqa: E402
from esdvideo.ntuh import enumerate_clips, extract_clip_1fps  # noqa: E402
from esdvideo.paths import CONFIG_DIR, load_paths  # noqa: E402
from esdvideo.windows import dataset_info, encode_window, make_row, tile, train_windows, write_split  # noqa: E402


def merge_dataset_info(path: Path, entries: dict) -> None:
    info = json.loads(path.read_text()) if path.is_file() else {}
    info.update(entries)
    path.write_text(json.dumps(info, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=int, default=200, help="max train windows per phase")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    a = ap.parse_args()
    paths = load_paths()
    out = paths.work_root / "windows"
    frames_root = paths.work_root / "frames" / "ntuh"
    clips_root = out / "clips" / "ntuh"

    clips = enumerate_clips(paths.require("ntuh_phase_root"))
    n_sec = {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(extract_clip_1fps, str(src), str(frames_root / case / cid), a.ffmpeg): (case, cid)
                for case, cid, _, src in clips}
        for f in as_completed(futs):
            n_sec[futs[f]] = f.result()
    print(f"NTUH: {len(clips)} clips, {sum(n_sec.values())} s at 1 fps")

    split = read_split(CONFIG_DIR / "splits" / "ntuh_5fold.tsv")
    jobs, summary = {}, {}
    test_all = []
    for fold in sorted(split):
        roles = split[fold]
        fold_dir = out / f"ntuh_5fold/fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        rows = {"train": [], "val": [], "test": []}
        dropped = Counter()
        for case, cid, phase, _ in clips:
            role = roles[case]
            n = n_sec[(case, cid)]
            fdir = str(frames_root / case / cid)
            if role == "train":
                wins = [(s, L, s, s + L) for s, L in train_windows(n)]
            else:
                wins = tile(n)
                if not wins:
                    dropped[phase] += n
            for s, L, x0, x1 in wins:
                labels = [phase] * L if role == "train" else [phase if x0 <= s + i < x1 else None for i in range(L)]
                clip = str(clips_root / case / f"{cid}_w{s:05d}_{L:02d}.mp4")
                jobs[clip] = (fdir, s, L)
                rows[role].append(make_row("ntuh", case, cid, s, L, labels, (x0, x1), clip))
        rng = random.Random(a.seed + fold)
        by_phase = defaultdict(list)
        for r in rows["train"]:
            by_phase[r["labels"][0]].append(r)
        train = []
        for ph in sorted(by_phase):
            g = by_phase[ph]
            train += g if len(g) <= a.cap else rng.sample(g, a.cap)
        rows["train"] = sorted(train, key=lambda r: r["window_id"])
        for role, rs in rows.items():
            write_split(fold_dir / role, rs)
        test_all += rows["test"]
        (fold_dir / "dataset_info.json").write_text(json.dumps(dataset_info(
            {"ntuh_phase_train": "train", "ntuh_phase_val": "val", "ntuh_phase_test": "test"}), indent=2) + "\n")
        summary[fold] = {role: {"windows": len(rs), "cases": sorted({r["case"] for r in rs}),
                                "windows_by_phase": dict(Counter(next(x for x in r["labels"] if x) for r in rs)),
                                "scored_seconds_by_phase": dict(Counter(x for r in rs for x in r["labels"] if x))}
                         for role, rs in rows.items()}
        summary[fold]["unscored_short_clip_seconds"] = dict(dropped)

    ext = out / "external"
    ext.mkdir(parents=True, exist_ok=True)
    write_split(ext / "ntuh_all", test_all)
    merge_dataset_info(ext / "dataset_info.json", dataset_info({"ntuh_all": "ntuh_all"}))

    needed = set()
    for fold in split:
        for role in ("train", "val", "test"):
            for line in (out / f"ntuh_5fold/fold{fold}/{role}_clips.jsonl").open():
                needed.add(json.loads(line)["videos"][0])
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(encode_window, *jobs[c], c, False, a.ffmpeg) for c in needed]):
            f.result()
    (out / "ntuh_5fold/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({f: {r: summary[f][r]["windows_by_phase"] for r in ("train", "val", "test")} for f in summary}, indent=1))


if __name__ == "__main__":
    main()
