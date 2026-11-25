#!/usr/bin/env python
"""Renji (public): first frame of every annotated second, and the per-second label index.

    python scripts/prepare_renji_frames.py --release cnls --workers 16
    python scripts/prepare_renji_frames.py --release egc  --workers 16

Writes <work_root>/frames/renji_<release>/<case>/<video>/sNNNNNN.jpg and
<work_root>/frames/index/renji_<release>.tsv (Experiment 1 mapping: estimation -> Diagnosis,
injection -> Local injection, incision/esd -> Incision/Dissection, vessel_treatment -> Hemostasis;
preparation, marking and clips are not indexed, i.e. not scored). Annotations may run up to ~1 s past
the end of a video; such seconds (at most MAX_SHORT per video) are dropped from the index.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esdvideo import renji_frames as renji  # noqa: E402
from esdvideo.index import write_index  # noqa: E402
from esdvideo.labels import PHASES  # noqa: E402
from esdvideo.paths import load_paths  # noqa: E402

MAX_SHORT = 2
# annotated seconds per phase in the public releases (Diagnosis, LocalInjection, IncisionDissection, Hemostasis)
EXPECTED = {"renji_egc": (7381, 5385, 29520, 7012), "renji_cnls": (8966, 14540, 62008, 1411)}


def _marker(out_dir: Path) -> Path:
    return out_dir / ".extracted"


def _extract_one(video: str, out_dir: str, n_seconds: int, ffmpeg: str) -> tuple[str, int, int]:
    out = Path(out_dir)
    m = _marker(out)
    if m.is_file():
        want, got = (int(x) for x in m.read_text().split())
        if want == n_seconds:
            return video, want, got
    wrote = renji.extract_first_frame_per_second(Path(video), out, n_seconds, ffmpeg=ffmpeg)
    if n_seconds - wrote <= MAX_SHORT:
        m.write_text(f"{n_seconds} {wrote}\n")
    return video, n_seconds, wrote


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, choices=["cnls", "egc"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    a = ap.parse_args()
    paths = load_paths()
    ds = f"renji_{a.release}"
    frames_root = paths.work_root / "frames"
    videos = renji.load_videos(paths.renji_root(a.release))

    jobs = [(str(v.path), str(frames_root / ds / v.case / v.video_id), len(v.labels), a.ffmpeg) for v in videos]
    decodable, too_short = {}, []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(_extract_one, *j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            video, want, got = fut.result()
            if want - got > MAX_SHORT:
                too_short.append((Path(video).name, want, got))
            print(f"[{i}/{len(jobs)}] {Path(video).name}: {got}/{want}", flush=True)
    if too_short:
        sys.exit(f"{len(too_short)} videos end more than {MAX_SHORT} s before their annotation: {too_short[:3]}")
    for v in videos:
        decodable[v.video_id] = int(_marker(frames_root / ds / v.case / v.video_id).read_text().split()[1])

    frames = renji.build_frames(ds, videos)
    got = tuple(Counter(f.phase for f in frames).get(p, 0) for p in PHASES)
    print(f"{ds}: {len({v.case for v in videos})} cases, {len(videos)} videos, annotated seconds by phase {got}")
    if got != EXPECTED[ds]:
        print(f"warning: phase counts differ from the public release counts {EXPECTED[ds]}")
    kept = [f for f in frames if f.t < decodable[f.video]]
    n = write_index(frames_root / "index" / f"{ds}.tsv", kept)
    print(f"indexed {n} seconds; dropped {len(frames) - n} annotated seconds past the video end")


if __name__ == "__main__":
    main()
