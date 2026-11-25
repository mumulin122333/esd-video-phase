"""Video windows shared by all four video VLMs, and per-second scoring of their JSON answers.

Every model reads the same 1 fps MP4 windows (<= 32 s, letterboxed to 512x448) with the same prompt
and JSON target (``contracts.prompt``). NTUH windows come from single-phase hand-cut clips; Renji
evaluation windows tile each video's annotated timeline (Experiment 1 protocol).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .contracts import parse_prediction, prompt
from .labels import PHASES, UNPARSED

# esd-vlm-bench phase ids <-> phase names in the prompt / JSON answers
PHASE_NAME = {"Diagnosis": "Diagnosis", "LocalInjection": "Local injection",
              "IncisionDissection": "Incision / Dissection", "Hemostasis": "Hemostasis"}
NAME_TO_ID = {v: k for k, v in PHASE_NAME.items()}

WINDOW, STRIDE, MIN_SECONDS = 32, 8, 4
W, H = 512, 448
LETTERBOX = f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=bicubic,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black"


def tile(n: int) -> list[tuple[int, int, int, int]]:
    """Evaluation windows over seconds [0, n): (start, length, score_from, score_to).

    32 s windows from 0; a remainder gets a window ending at n (overlapping the previous one when
    n >= 32) that scores only the remainder, so each second is scored exactly once. Timelines shorter
    than MIN_SECONDS are skipped."""
    if n < MIN_SECONDS:
        return []
    if n <= WINDOW:
        return [(0, n, 0, n)]
    out = [(s, WINDOW, s, s + WINDOW) for s in range(0, n - WINDOW + 1, WINDOW)]
    done = out[-1][3]
    if done < n:
        out.append((n - WINDOW, WINDOW, done, n))
    return out


def train_windows(n: int) -> list[tuple[int, int]]:
    """Training windows: 32 s with an 8 s stride; a clip shorter than 32 s is one window."""
    if n < MIN_SECONDS:
        return []
    if n <= WINDOW:
        return [(0, n)]
    return [(s, WINDOW) for s in range(0, n - WINDOW + 1, STRIDE)]


def target(labels: list[str | None]) -> str:
    """JSON target of a single-phase window (NTUH)."""
    phases = {x for x in labels if x is not None}
    if len(phases) != 1:
        raise ValueError("training windows must have exactly one phase")
    seg = [{"phase": PHASE_NAME[phases.pop()], "start_seconds": 0, "end_seconds": len(labels)}]
    return json.dumps({"segments": seg}, separators=(",", ":"))


def make_row(dataset, case, video, start, length, labels, score, clip_path) -> dict:
    """One window: LlamaFactory record fields (messages, videos) plus scoring metadata.

    labels: per-second phase ids of the window (None = not scored); score: absolute [from, to)."""
    answer = target(labels) if dataset == "ntuh" else json.dumps(
        {"segments": [{"phase": "Diagnosis", "start_seconds": 0, "end_seconds": length}]}, separators=(",", ":"))
    return {
        "window_id": f"{dataset}/{case}/{video}_w{start:05d}_{length:02d}",
        "dataset": dataset, "case": case, "video": video, "start": start, "length": length,
        "score_from": score[0], "score_to": score[1], "labels": labels,
        "messages": [{"role": "user", "content": prompt(length)}, {"role": "assistant", "content": answer}],
        "videos": [clip_path],
    }


def encode_window(frame_dir: str, start: int, length: int, dst: str, scale: bool, ffmpeg: str = "ffmpeg") -> str:
    """Encode frames s<start>..s<start+length-1>.jpg of frame_dir into a 1 fps MP4 (idempotent)."""
    d = Path(dst)
    if d.is_file() and d.stat().st_size > 0:
        return dst
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.with_name(f".{d.stem}.tmp.mp4")
    vf = ([LETTERBOX] if scale else []) + ["format=yuv420p"]
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", "1", "-start_number", str(start),
                    "-i", f"{frame_dir}/s%06d.jpg", "-frames:v", str(length), "-vf", ",".join(vf),
                    "-c:v", "libx264", "-crf", "18", "-r", "1", "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(d)
    return dst


def write_split(path_stem: Path, rows: list[dict]) -> None:
    """<stem>.jsonl (LlamaFactory records) and <stem>_clips.jsonl (scoring metadata)."""
    with path_stem.with_suffix(".jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({"messages": r["messages"], "videos": r["videos"]}, ensure_ascii=False) + "\n")
    with path_stem.with_name(path_stem.name + "_clips.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "messages"}) + "\n")


def dataset_info(names: dict[str, str]) -> dict:
    """LlamaFactory dataset_info.json entries: {dataset name: file stem}."""
    return {ds: {"file_name": f"{stem}.jsonl", "formatting": "sharegpt",
                 "columns": {"messages": "messages", "videos": "videos"},
                 "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
            for ds, stem in names.items()}


def per_second(meta_path: Path, pred_path: Path):
    """Per-second (case, truth, prediction) of one prediction file against window metadata.

    Each answer is parsed with the JSON contract (1-2 ordered segments covering the window); an
    invalid answer makes every second of its window UNPARSED (wrong). Only seconds inside the window's
    score range that carry a label are returned."""
    metas = [json.loads(l) for l in open(meta_path)]
    preds = {}
    for l in open(pred_path):
        r = json.loads(l)
        preds[r["index"]] = r["predict"]
    if len(preds) != len(metas):
        raise ValueError(f"{pred_path}: {len(preds)} predictions for {len(metas)} windows")
    cases, y_true, y_pred = [], [], []
    stats = {"windows": 0, "valid": 0, "two_segments": 0}
    for i, m in enumerate(metas):
        p = parse_prediction(preds[i], m["length"])
        stats["windows"] += 1
        stats["valid"] += p.valid
        if p.valid:
            stats["two_segments"] += len(p.segments) == 2
        for s, lab in enumerate(m["labels"]):
            if lab is None:
                continue
            if p.valid:
                t = s + 0.5
                seg = next((g for g in p.segments if g.start_seconds <= t < g.end_seconds), p.segments[-1])
                y = NAME_TO_ID[seg.phase]
            else:
                y = UNPARSED
            cases.append(m["case"])
            y_true.append(lab)
            y_pred.append(y)
    return cases, y_true, y_pred, stats


assert set(PHASE_NAME) == set(PHASES)
