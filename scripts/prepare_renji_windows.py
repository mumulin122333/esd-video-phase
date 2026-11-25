#!/usr/bin/env python
"""Renji evaluation windows (Experiment 1 protocol), for every case of a release.

    python scripts/prepare_renji_windows.py --release cnls --workers 64
    python scripts/prepare_renji_windows.py --release egc  --workers 64

Needs scripts/prepare_renji_frames.py. Each video's annotated timeline is tiled into 32 s windows
(windows.tile) from the 1 fps frames; seconds are scored with the Experiment 1 mapping (preparation,
marking and clips excluded); windows without any scored second are skipped.

Writes <work_root>/windows/external/renji_<release>{,_clips}.jsonl (all cases: external test of models
not trained on this release) and windows/renji_oof/<release>/fold<k>/renji_<release>{,_clips}.jsonl,
the subset of cases that are test cases of fold k in the Renji 5-fold split (prepare_renji_clips.py,
same seed), for out-of-fold scoring of models trained on this release.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esdvideo.paths import load_paths  # noqa: E402
from esdvideo.renji_clips import load_cases, split_cases  # noqa: E402
from esdvideo.windows import dataset_info, encode_window, make_row, tile, write_split  # noqa: E402


def merge_dataset_info(path: Path, entries: dict) -> None:
    info = json.loads(path.read_text()) if path.is_file() else {}
    info.update(entries)
    path.write_text(json.dumps(info, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, choices=["cnls", "egc"])
    ap.add_argument("--seed", type=int, default=20261022, help="Renji 5-fold split seed (as prepare_renji_clips.py)")
    ap.add_argument("--val-cases", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    a = ap.parse_args()
    paths = load_paths()
    ds = f"renji_{a.release}"
    frames = paths.work_root / "frames"
    out = paths.work_root / "windows"

    scored: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    with open(frames / "index" / f"{ds}.tsv", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            scored[(r["case"], r["video"])][int(r["t"])] = r["phase"]
    rows, jobs = [], {}
    for (case, video) in sorted(scored, key=lambda k: (int(k[0]), k[1])):
        fdir = frames / ds / case / video
        _, got = (int(x) for x in (fdir / ".extracted").read_text().split())
        for s, L, x0, x1 in tile(got):
            labels = [scored[(case, video)].get(s + i) if x0 <= s + i < x1 else None for i in range(L)]
            if not any(labels):
                continue
            clip = str(out / f"clips/{ds}/{case}/{video}_w{s:05d}_{L:02d}.mp4")
            jobs[clip] = (str(fdir), s, L)
            rows.append(make_row(ds, case, video, s, L, labels, (x0, x1), clip))
    ext = out / "external"
    ext.mkdir(parents=True, exist_ok=True)
    write_split(ext / ds, rows)
    merge_dataset_info(ext / "dataset_info.json", dataset_info({ds: ds}))
    n_scored = Counter(x for r in rows for x in r["labels"] if x)
    print(f"{ds}: {len({r['case'] for r in rows})} cases, {len(rows)} windows, scored seconds {dict(n_scored)}")

    case_ids = [c.case_id for c in load_cases(paths.renji_root(a.release))]
    for fold in range(5):
        test = set(split_cases(case_ids, seed=a.seed, test_cases=0, val_cases=a.val_cases, fold=fold)["test"])
        sub = [r for r in rows if r["case"] in test]
        d = out / "renji_oof" / a.release / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        write_split(d / ds, sub)
        (d / "dataset_info.json").write_text(json.dumps(dataset_info({ds: ds}), indent=2) + "\n")
        print(f"  fold {fold}: {len(test)} test cases, {len(sub)} windows")

    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(encode_window, fdir, s, L, clip, True, a.ffmpeg) for clip, (fdir, s, L) in jobs.items()]):
            f.result()
    print(f"encoded {len(jobs)} windows")


if __name__ == "__main__":
    main()
