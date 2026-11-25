"""NTUH (in-house) hand-cut ESD clips.

Layout: ``<ntuh_phase_root>/cases/NTUH_xx/clips/<NN>_<class>/<clip>.MP4``; every clip holds one fine
action class and therefore one phase. Original clip file names can contain medical-record numbers, so
clips are referred to only by pseudonymous ids ``c<class>_k<index>`` (index in sorted file-name order).

The recordings are 1920x1080 captures of the endoscopy monitor whose left third is an information
panel; frames are cropped to the endoscopic view (right 1280x1080, x 640-1919) as in Experiment 1.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .labels import NTUH_FINE_CLASSES, ntuh_clip_dirname
from .windows import LETTERBOX

CROP = "crop=1280:1080:640:0"
NTUH_CLASSES = [c for c, (_, ph) in sorted(NTUH_FINE_CLASSES.items()) if ph is not None]  # 2-5, 7, 8-9, 13-14


def enumerate_clips(root: Path) -> list[tuple[str, str, str, Path]]:
    """(case, pseudonymous clip id, phase, source path) for every clip of a phase class."""
    clips = []
    for case_dir in sorted((root / "cases").iterdir()):
        for cls in NTUH_CLASSES:
            d = case_dir / "clips" / ntuh_clip_dirname(cls)
            if not d.is_dir():
                continue
            files = sorted(p for p in d.iterdir() if p.suffix.lower() == ".mp4")
            for k, src in enumerate(files):
                clips.append((case_dir.name, f"c{cls:02d}_k{k:02d}", NTUH_FINE_CLASSES[cls][1], src))
    return clips


def extract_clip_1fps(src: str, out_dir: str, ffmpeg: str = "ffmpeg") -> int:
    """1 fps frames s000000.jpg ... cropped to the endoscopic view and letterboxed; returns the count."""
    out = Path(out_dir)
    marker = out / ".done"
    if marker.is_file():
        return int(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-an",
                    "-vf", f"fps=1,{CROP},{LETTERBOX}", "-q:v", "2", "-start_number", "0",
                    str(out / "s%06d.jpg")], check=True)
    n = len(list(out.glob("s*.jpg")))
    marker.write_text(str(n))
    return n
