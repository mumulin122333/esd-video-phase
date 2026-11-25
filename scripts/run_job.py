#!/usr/bin/env python
"""Train one model on one training set x fold (one GPU), then predict every test set.

    python scripts/run_job.py --set ntuh_cnls --model molmo2 --fold 0 --gpu 0

Test sets (written to <runs_root>/<set>/<model>/fold<k>/predictions/<name>.jsonl, one
{"index", "predict"} line per window / clip, in input order):
  ntuh_windows   NTUH windows: fold-k test cases if the set contains NTUH (out-of-fold), else all 13 cases
  cnls_windows   Renji CNLs Experiment 1 windows: fold-k test cases if the set contains CNLs, else all cases
  egc_windows    Renji EGC  Experiment 1 windows: same rule
  cnls_clips     Renji CNLs training-style test clips of fold k (only if the set contains CNLs)
  egc_clips      Renji EGC  training-style test clips of fold k (only if the set contains EGC)

Environments (Qwen3.8 and the other three need different transformers versions):
  QWEN_BIN    directory with llamafactory-cli of the Qwen environment (default: on PATH)
  VLM_PYTHON  python of the transformers 4.57.3 environment for esdvideo.vlm (default: this python)
Steps already done (TRAIN_COMPLETE, finished prediction files) are skipped, so a job can be rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esdvideo import qwen  # noqa: E402
from esdvideo.paths import CONFIG_DIR, load_paths  # noqa: E402

SETS = ["ntuh", "cnls", "egc", "cnls_egc", "ntuh_cnls", "ntuh_egc", "ntuh_cnls_egc"]


def sources(training_set: str) -> set[str]:
    return {"ntuh": {"ntuh"}, "cnls": {"cnls"}, "egc": {"egc"}, "cnls_egc": {"cnls", "egc"},
            "ntuh_cnls": {"ntuh", "cnls"}, "ntuh_egc": {"ntuh", "egc"}, "ntuh_cnls_egc": {"ntuh", "cnls", "egc"}}[training_set]


def test_sets(work: Path, training_set: str, fold: int) -> list[tuple[str, Path, str]]:
    """(name, jsonl, LlamaFactory dataset name) for every test set of this job."""
    src = sources(training_set)
    w = work / "windows"
    out = [("ntuh_windows", w / f"ntuh_5fold/fold{fold}/test.jsonl", "ntuh_phase_test") if "ntuh" in src
           else ("ntuh_windows", w / "external/ntuh_all.jsonl", "ntuh_all")]
    for rel in ("cnls", "egc"):
        if rel in src:
            out.append((f"{rel}_windows", w / f"renji_oof/{rel}/fold{fold}/renji_{rel}.jsonl", f"renji_{rel}"))
            out.append((f"{rel}_clips", work / f"renji_clips/{rel}/fold{fold}/test.jsonl", "renji_phase_test"))
        else:
            out.append((f"{rel}_windows", w / f"external/renji_{rel}.jsonl", f"renji_{rel}"))
    return out


def run(cmd: list[str], env: dict, log: Path) -> None:
    with log.open("a") as fh:
        fh.write(f"\n$ {' '.join(map(str, cmd))}\n")
        fh.flush()
        subprocess.run([str(c) for c in cmd], check=True, env=env, stdout=fh, stderr=subprocess.STDOUT)


def finalize(jsonl: Path, partial: Path, final: Path) -> None:
    n = sum(1 for _ in open(jsonl))
    got = {}
    for line in open(partial):
        r = json.loads(line)
        got[r["index"]] = r["predict"]
    if sorted(got) != list(range(n)):
        raise RuntimeError(f"{partial}: {len(got)} of {n} predictions")
    tmp = final.with_suffix(".tmp")
    with tmp.open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({"index": i, "predict": got[i]}, ensure_ascii=False) + "\n")
    tmp.replace(final)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument("--model", required=True, choices=["qwen38", "molmo2", "videochat3", "internvideo3"])
    ap.add_argument("--fold", type=int, required=True, choices=range(5))
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    paths = load_paths()
    spec = yaml.safe_load((CONFIG_DIR / "models.yaml").read_text())[a.model]
    weights = paths.pretrained_root / spec["dir"]
    data = paths.work_root / "training_sets" / a.set / f"fold{a.fold}"
    run_dir = paths.runs_root / a.set / a.model / f"fold{a.fold}"
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "job.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), FORCE_TORCHRUN="1")
    if os.environ.get("QWEN_BIN"):
        env["PATH"] = os.environ["QWEN_BIN"] + os.pathsep + env["PATH"]
    vlm_python = os.environ.get("VLM_PYTHON", sys.executable)
    vlm = [vlm_python, "-m", "esdvideo.vlm"]
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    # ---- train
    if not (run_dir / "TRAIN_COMPLETE").exists():
        if spec["engine"] == "llamafactory":
            y = run_dir / "train.yaml"
            y.write_text(qwen.train_yaml(weights, data, run_dir / "adapter"))
            run(["llamafactory-cli", "train", y], env, log)
            for ck in (run_dir / "adapter").glob("checkpoint-*"):
                shutil.rmtree(ck)
        else:
            run(vlm + ["train", "--model", a.model, "--model-path", weights, "--data", data, "--out", run_dir], env, log)
        (run_dir / "TRAIN_COMPLETE").touch()

    # ---- predict
    for name, jsonl, ds in test_sets(paths.work_root, a.set, a.fold):
        final = pred_dir / f"{name}.jsonl"
        if final.exists():
            continue
        if spec["engine"] == "llamafactory":
            out = run_dir / f"predict_{name}"
            y = run_dir / f"predict_{name}.yaml"
            y.write_text(qwen.predict_yaml(weights, run_dir / "adapter", ds, jsonl.parent, out, spec["predict_batch"]))
            run(["llamafactory-cli", "train", y], env, log)  # one GPU keeps the input order
            with open(out / "generated_predictions.jsonl") as src, (pred_dir / f"{name}.partial").open("w") as dst:
                for i, line in enumerate(src):
                    dst.write(json.dumps({"index": i, "predict": json.loads(line)["predict"]}, ensure_ascii=False) + "\n")
            finalize(jsonl, pred_dir / f"{name}.partial", final)
            shutil.rmtree(out)
        else:
            partial = pred_dir / f"{name}.partial"
            run(vlm + ["predict", "--model", a.model, "--model-path", weights, "--weights", run_dir / "best.safetensors",
                       "--data", jsonl, "--out", partial, "--batch", str(spec["predict_batch"])], env, log)
            finalize(jsonl, partial, final)
        (pred_dir / f"{name}.partial").unlink(missing_ok=True)
    (run_dir / "JOB_COMPLETE").touch()
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
