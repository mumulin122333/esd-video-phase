"""Renji ESD-for-CNLs / ESD-for-EGC public releases.

Layouts
- CNLs: ``<root>/<case>/<case>-<k>.MP4`` + ``<case>-<k>.txt``; one label per line (1 Hz) from t=0.
- EGC:  ``<root>/<case>/M_<stem>.MP4`` + ``<stem>.txt``; ``Frame<TAB>Phase`` header, one row per
  video frame (50 fps). Video order comes from ``_<kkkk>-<nn>`` at the end of the stem.

Each annotated second uses the first frame of that second, matching the
release's 1 fps annotation protocol. The unannotated tail of a video is dropped.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .labels import RENJI_FINE_TO_PHASE, normalize_renji_label, renji_phase
from .index import Frame

_CNLS_RE = re.compile(r"^(?P<case>[^/]+)-(?P<part>\d+)$")
_EGC_PART_RE = re.compile(r"_(?P<part>\d{4})-\d+$")


@dataclass
class Video:
    case: str
    part: int
    path: Path
    labels: list[str]  # normalised fine label per second

    @property
    def video_id(self) -> str:
        return f"{self.case}-{self.part}"


def read_labels(path: Path, fps: int = 50) -> list[str]:
    lines = [ln.rstrip("\r") for ln in path.read_text(encoding="utf-8").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[0].split("\t")[:2] == ["Frame", "Phase"]:
        rows = [ln.split("\t") for ln in lines[1:]]
        frames = [int(r[0]) for r in rows]
        if frames != list(range(len(frames))):
            raise ValueError(f"{path}: frame indices are not contiguous from 0")
        labels = [normalize_renji_label(r[1]) for r in rows[::fps]]
    else:
        labels = [normalize_renji_label(ln) for ln in lines]
    unknown = sorted({lab for lab in labels if lab not in RENJI_FINE_TO_PHASE})
    if unknown:
        raise ValueError(f"{path}: unknown labels {unknown}")
    return labels


def _video_for(txt: Path, case: str) -> tuple[int, Path]:
    m = _CNLS_RE.match(txt.stem)
    if m and m["case"] == case:
        part, cands = int(m["part"]), [txt.with_suffix(s) for s in (".MP4", ".mp4")]
    else:
        e = _EGC_PART_RE.search(txt.stem)
        if not e:
            raise ValueError(f"unexpected annotation name: {txt}")
        part = int(e["part"])
        cands = [txt.with_name(f"M_{txt.stem}{s}") for s in (".MP4", ".mp4")]
    video = next((c for c in cands if c.is_file()), None)
    if video is None:
        raise FileNotFoundError(f"missing video for {txt}")
    return part, video


def load_videos(root: Path, fps: int = 50) -> list[Video]:
    out: list[Video] = []
    case_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name))
    for cdir in case_dirs:
        vids = []
        for txt in cdir.glob("*.txt"):
            part, video = _video_for(txt, cdir.name)
            vids.append(Video(cdir.name, part, video, read_labels(txt, fps)))
        if len({v.part for v in vids}) != len(vids):
            raise ValueError(f"case {cdir.name}: duplicate video part numbers")
        out.extend(sorted(vids, key=lambda v: v.part))
    if not out:
        raise FileNotFoundError(f"no annotated videos under {root}")
    return out


def image_relpath(dataset: str, video: Video, second: int) -> str:
    return f"{dataset}/{video.case}/{video.video_id}/s{second:06d}.jpg"


def build_frames(dataset: str, videos: list[Video]) -> list[Frame]:
    """One Frame per annotated second whose label maps to a phase."""
    frames = []
    for v in videos:
        for s, lab in enumerate(v.labels):
            ph = renji_phase(lab)
            if ph is not None:
                frames.append(Frame(dataset, v.case, v.video_id, s, lab, ph, image_relpath(dataset, v, s)))
    return frames


def probe_fps(video: Path, ffprobe: str = "ffprobe") -> int:
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
                         check=True, capture_output=True, text=True).stdout.strip()
    num, den = (int(x) for x in out.split("/"))
    if num % den:
        raise ValueError(f"{video}: non-integer frame rate {out}")
    return num // den


def extract_first_frame_per_second(video: Path, out_dir: Path, n_seconds: int, *, ffmpeg: str = "ffmpeg",
                                   ffprobe: str = "ffprobe", quality: int = 2) -> int:
    """Write s000000.jpg ... for seconds [0, n_seconds) (frame index s*fps).

    ``quality`` is ffmpeg's -q:v (2 is visually lossless). Returns the number of files written.
    """
    fps = probe_fps(video, ffprobe)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp_%06d.jpg"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(video),
                    "-vf", f"select=not(mod(n\\,{fps}))", "-fps_mode", "vfr",
                    "-frames:v", str(n_seconds), "-q:v", str(quality), str(tmp)], check=True)
    written = 0
    for i in range(1, n_seconds + 1):
        src = out_dir / f"_tmp_{i:06d}.jpg"
        if not src.exists():
            break
        src.replace(out_dir / f"s{i - 1:06d}.jpg"); written += 1
    for leftover in out_dir.glob("_tmp_*.jpg"):
        leftover.unlink()
    return written


# ---------------------------------------------------------------- A6 sensitivity mapping
_MONO = {"preparation": None, "estimation": 0, "marking": 0, "injection": 1,
         "incision": 2, "esd": 2, "vessel_treatment": 3, "clips": 3}
_MONO_PHASES = ("Diagnosis", "LocalInjection", "IncisionDissection", "Hemostasis")


def monotonic_case_phases(case_labels: list[str]) -> list[str]:
    """Alternative (sensitivity-analysis) mapping: one monotonic phase per second of a case.

    Port of ``fit_macro_phases`` from the QwenVideoSegment project: Diagnosis until the first
    injection, Local injection until the first incision/esd, Incision/Dissection until the first
    vessel_treatment/clips after the last incision/esd, Hemostasis afterwards.
    """
    mapped = [_MONO[normalize_renji_label(x)] for x in case_labels]
    dissection = [i for i, p in enumerate(mapped) if p == 2]
    if not dissection:
        raise ValueError("case has no incision/esd labels")
    injection = next((i for i, p in enumerate(mapped) if p == 1), None)
    if injection is not None and injection > dissection[0]:
        injection = None
    start_x = dissection[0]
    start_i = injection if injection is not None else start_x
    start_h = next((i for i in range(dissection[-1] + 1, len(mapped)) if mapped[i] == 3), len(mapped))
    return [_MONO_PHASES[0 if i < start_i else 1 if i < start_x else 2 if i < start_h else 3]
            for i in range(len(mapped))]
