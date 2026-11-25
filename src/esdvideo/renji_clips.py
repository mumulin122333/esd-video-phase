"""Build LlamaFactory phase-segmentation training data from the raw Renji release.

The public Renji ESD-for-CNLs release stores, per case directory, one or more
videos (``<case>/<case>-<k>.MP4``) and a sibling ``.txt`` with one fine-grained
action label per second, starting at t=0. The annotation can stop before the
end of the video; the unannotated tail is never used.

Fine labels are collapsed into the four macro phases used by this project with
a monotonic, rule-based fit over the case timeline (all videos of a case in
order):

* Diagnosis: everything before the first ``injection``
* Local injection: first ``injection`` up to the first ``incision``/``esd``
* Incision / Dissection: up to the last ``incision``/``esd``, through to the
  first hemostasis label after it
* Hemostasis: first ``vessel_treatment``/``clips`` after the last dissection
  label, to the end of the annotation

Re-injections and short vessel treatments during dissection therefore belong to
"Incision / Dissection", which is the macro-phase semantics the inference
pipeline's monotonic decoder assumes.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .contracts import PHASES, prompt

FINE_TO_PHASE: dict[str, int | None] = {
    "preparation": None,
    "estimation": 0,
    "marking": 0,
    "injection": 1,
    "incision": 2,
    "esd": 2,
    "vessel_treatment": 3,
    "clips": 3,
}

_VIDEO_RE = re.compile(r"^(?P<case>[^/]+)-(?P<part>\d+)$")


@dataclass
class VideoAnnotation:
    case_id: str
    video_id: str
    video_path: Path
    fine_labels: list[str]
    phases: list[int] = field(default_factory=list)


@dataclass
class CaseTimeline:
    case_id: str
    videos: list[VideoAnnotation]
    boundaries: dict[str, int | None] = field(default_factory=dict)
    agreement: float = 0.0

    @property
    def fine_labels(self) -> list[str]:
        return [label for video in self.videos for label in video.fine_labels]


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    split: str
    kind: str  # "pure" or "boundary"
    case_id: str
    video_id: str
    source_video: str
    start_seconds: int
    duration_seconds: int
    segments: tuple[tuple[int, int, int], ...]  # (phase index, start, end), clip-relative

    @property
    def label_key(self) -> str:
        return "->".join(PHASES[phase] for phase, _, _ in self.segments)


_EGC_PART_RE = re.compile(r"_(?P<part>\d{4})-\d+$")


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_")


def _read_labels(path: Path, frames_per_second: int = 50) -> list[str]:
    """Return one fine label per second.

    Two release formats are supported:
    * ESD-for-CNLs: one label per line at 1 Hz, no header.
    * ESD-for-EGC: a ``Frame<TAB>Phase`` header and one row per video frame
      (50 fps) starting at frame 0; each second's label is taken from its
      first frame, matching the 1 fps annotation protocol of the release.
    """

    lines = [line.rstrip("\r") for line in path.read_text(encoding="utf-8").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[0].split("\t")[:2] == ["Frame", "Phase"]:
        rows = [line.split("\t") for line in lines[1:]]
        frames = [int(row[0]) for row in rows]
        if frames != list(range(len(frames))):
            raise ValueError(f"{path}: frame indices are not contiguous from 0")
        labels = [_normalize_label(row[1]) for row in rows[::frames_per_second]]
    else:
        labels = [_normalize_label(line) for line in lines]
    unknown = sorted({label for label in labels if label not in FINE_TO_PHASE})
    if unknown:
        raise ValueError(f"{path}: unknown labels {unknown}")
    return labels


def _video_for(txt: Path, case_id: str) -> tuple[int, Path]:
    """Locate the video and its order within the case for either release layout."""

    match = _VIDEO_RE.match(txt.stem)
    if match and match["case"] == case_id:  # CNLs: <case>/<case>-<k>.txt + .MP4
        part, candidates = int(match["part"]), [txt.with_suffix(s) for s in (".MP4", ".mp4")]
    else:  # EGC: <case>/<stem>.txt + M_<stem>.MP4, part from "_000k-01"
        egc = _EGC_PART_RE.search(txt.stem)
        if not egc:
            raise ValueError(f"unexpected annotation name: {txt}")
        part = int(egc["part"])
        candidates = [txt.with_name(f"M_{txt.stem}{s}") for s in (".MP4", ".mp4")]
    video = next((c for c in candidates if c.is_file()), None)
    if video is None:
        raise FileNotFoundError(f"missing video for {txt}")
    return part, video


def load_cases(root: Path, frames_per_second: int = 50) -> list[CaseTimeline]:
    cases: list[CaseTimeline] = []
    for case_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        videos: list[tuple[int, VideoAnnotation]] = []
        for txt in case_dir.glob("*.txt"):
            part, video = _video_for(txt, case_dir.name)
            # video_id is always <case>-<part>: stable and free of the release's record numbers
            videos.append(
                (
                    part,
                    VideoAnnotation(
                        case_dir.name,
                        f"{case_dir.name}-{part}",
                        video,
                        _read_labels(txt, frames_per_second),
                    ),
                )
            )
        if len({part for part, _ in videos}) != len(videos):
            raise ValueError(f"case {case_dir.name}: duplicate video part numbers")
        if videos:
            cases.append(CaseTimeline(case_dir.name, [v for _, v in sorted(videos, key=lambda x: x[0])]))
    if not cases:
        raise FileNotFoundError(f"no annotated cases under {root}")
    return cases


def fit_macro_phases(case: CaseTimeline) -> None:
    """Assign one monotonic macro phase to every annotated second of a case."""

    labels = case.fine_labels
    mapped = [FINE_TO_PHASE[label] for label in labels]
    dissection = [i for i, phase in enumerate(mapped) if phase == 2]
    if not dissection:
        raise ValueError(f"case {case.case_id} has no incision/esd labels")
    injection = next((i for i, phase in enumerate(mapped) if phase == 1), None)
    if injection is not None and injection > dissection[0]:
        injection = None  # dissection started before any labelled injection
    start_x = dissection[0]
    start_i = injection if injection is not None else start_x
    start_h = next((i for i in range(dissection[-1] + 1, len(mapped)) if mapped[i] == 3), len(mapped))

    phases = [
        0 if i < start_i else 1 if i < start_x else 2 if i < start_h else 3
        for i in range(len(labels))
    ]
    offset = 0
    for video in case.videos:
        video.phases = phases[offset : offset + len(video.fine_labels)]
        offset += len(video.fine_labels)

    informative = [(m, p) for m, p in zip(mapped, phases) if m is not None]
    case.agreement = sum(m == p for m, p in informative) / max(1, len(informative))
    case.boundaries = {
        "Diagnosis->Local injection": start_i if injection is not None and start_i > 0 else None,
        "Local injection->Incision / Dissection": start_x if injection is not None else None,
        "Incision / Dissection->Hemostasis": start_h if start_h < len(mapped) else None,
        "annotated_seconds": len(mapped),
    }


def _runs(values: list[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            runs.append((values[start], start, index))
            start = index
    return runs


def enumerate_clips(
    case: CaseTimeline,
    split: str,
    *,
    pure_seconds: int = 32,
    pure_stride: int = 8,
    boundary_seconds: int = 64,
    boundary_offsets: tuple[int, ...] = (16, 24, 32, 40, 48),
) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    for video in case.videos:
        for phase, run_start, run_end in _runs(video.phases):
            start = run_start
            while start + pure_seconds <= run_end:
                clips.append(
                    ClipSpec(
                        f"{video.video_id}_p{start:05d}",
                        split,
                        "pure",
                        case.case_id,
                        video.video_id,
                        str(video.video_path),
                        start,
                        pure_seconds,
                        ((phase, 0, pure_seconds),),
                    )
                )
                start += pure_stride
        for transition in range(1, len(video.phases)):
            left, right = video.phases[transition - 1], video.phases[transition]
            if left == right:
                continue
            for offset in boundary_offsets:
                start = transition - offset
                end = start + boundary_seconds
                if start < 0 or end > len(video.phases):
                    continue
                window = video.phases[start:end]
                if set(window) != {left, right} or window.index(right) != offset:
                    continue  # a third phase or a second transition inside the window
                clips.append(
                    ClipSpec(
                        f"{video.video_id}_b{start:05d}",
                        split,
                        "boundary",
                        case.case_id,
                        video.video_id,
                        str(video.video_path),
                        start,
                        boundary_seconds,
                        ((left, 0, offset), (right, offset, boundary_seconds)),
                    )
                )
    return clips


def split_cases(
    case_ids: list[str],
    *,
    seed: int,
    test_cases: int,
    val_cases: int,
    fold: int | None = None,
    num_folds: int = 5,
) -> dict[str, list[str]]:
    """Case-level split. With ``fold`` set, the shuffled cases are cut into
    ``num_folds`` contiguous test chunks (every case is tested exactly once
    across folds) and validation cases are drawn from the remaining cases."""

    shuffled = sorted(case_ids, key=lambda c: (len(c), c))
    random.Random(seed).shuffle(shuffled)
    if fold is None:
        test = shuffled[:test_cases]
        rest = shuffled[test_cases:]
    else:
        if not 0 <= fold < num_folds:
            raise ValueError(f"fold must be in [0, {num_folds})")
        chunks = [shuffled[i::num_folds] for i in range(num_folds)]
        test = chunks[fold]
        rest = [c for c in shuffled if c not in test]
        random.Random(seed + 1 + fold).shuffle(rest)
    split = {
        "test": test,
        "val": rest[:val_cases],
        "train": rest[val_cases:],
    }
    return {name: sorted(ids, key=lambda c: (len(c), c)) for name, ids in split.items()}


def balance(clips: list[ClipSpec], cap_per_label: int | None, rng: random.Random) -> list[ClipSpec]:
    """Cap every label key (pure phase or transition type) at ``cap_per_label``."""

    if cap_per_label is None:
        return list(clips)
    grouped: dict[str, list[ClipSpec]] = {}
    for clip in clips:
        grouped.setdefault(clip.label_key, []).append(clip)
    kept: list[ClipSpec] = []
    for key in sorted(grouped):
        group = grouped[key]
        kept.extend(group if len(group) <= cap_per_label else rng.sample(group, cap_per_label))
    return sorted(kept, key=lambda c: c.clip_id)


def target_json(clip: ClipSpec) -> str:
    return json.dumps(
        {
            "segments": [
                {"phase": PHASES[phase], "start_seconds": start, "end_seconds": end}
                for phase, start, end in clip.segments
            ]
        },
        separators=(",", ":"),
    )


def record(clip: ClipSpec, clip_path: Path) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt(clip.duration_seconds)},
            {"role": "assistant", "content": target_json(clip)},
        ],
        "videos": [str(clip_path)],
        "clip_id": clip.clip_id,
    }


def label_counts(clips: list[ClipSpec]) -> dict[str, int]:
    return dict(sorted(Counter(c.label_key for c in clips).items()))


def clip_metadata(clip: ClipSpec, clip_path: Path) -> dict:
    row = asdict(clip)
    row["segments"] = [list(s) for s in clip.segments]
    row["label_key"] = clip.label_key
    row["clip_path"] = str(clip_path)
    return row
