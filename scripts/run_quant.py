#!/usr/bin/env python
"""Predict the out-of-fold window test sets with a trained fold model at a given precision.

    python scripts/run_quant.py --set ntuh_cnls_egc --model molmo2 --fold 0 --quant nf4 --gpu 0

Precisions: bf16 (reference, also re-checks that predictions reproduce), int8 (bitsandbytes LLM.int8)
and nf4 (bitsandbytes 4-bit NF4 with double quantization, bf16 compute). Only the language model is
quantized; the vision encoder, projector, lm_head and LoRA weights stay in bf16 (LoRA is not merged).
Writes <runs_root>/<set>/<model>/fold<k>/predictions_<quant>/<test>_windows.jsonl and one
<test>_windows.stats.json per test set (rows/s, GPU memory). Score with
``scripts/score.py --pred-subdir predictions_<quant>``.
Needs bitsandbytes importable in both environments (e.g. via PYTHONPATH).
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
sys.path.insert(0, str(ROOT / "scripts"))

from esdvideo import qwen  # noqa: E402
from esdvideo.paths import CONFIG_DIR, load_paths  # noqa: E402
from run_job import SETS, finalize, run, test_sets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument("--model", required=True, choices=["qwen38", "molmo2", "videochat3", "internvideo3"])
    ap.add_argument("--fold", type=int, required=True, choices=range(5))
    ap.add_argument("--quant", required=True, choices=["bf16", "int8", "nf4"])
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    paths = load_paths()
    spec = yaml.safe_load((CONFIG_DIR / "models.yaml").read_text())[a.model]
    weights = paths.pretrained_root / spec["dir"]
    run_dir = paths.runs_root / a.set / a.model / f"fold{a.fold}"
    out_dir = run_dir / f"predictions_{a.quant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "quant.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), FORCE_TORCHRUN="0")
    if os.environ.get("QWEN_BIN"):
        env["PATH"] = os.environ["QWEN_BIN"] + os.pathsep + env["PATH"]
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    for name, jsonl, ds in test_sets(paths.work_root, a.set, a.fold):
        if not name.endswith("_windows"):
            continue  # window test sets only (out-of-fold for the sources of the training set)
        final = out_dir / f"{name}.jsonl"
        if final.exists():
            continue
        stats = out_dir / f"{name}.stats.json"
        if spec["engine"] == "llamafactory":
            pred = run_dir / f"predict_{a.quant}_{name}"
            cfg = yaml.safe_load(qwen.predict_yaml(weights, run_dir / "adapter", ds, jsonl.parent, pred, spec["predict_batch"]))
            if a.quant != "bf16":
                cfg.update({"quantization_bit": 8 if a.quant == "int8" else 4, "quantization_method": "bnb",
                            "quantization_type": "nf4", "double_quantization": True})
            y = out_dir / f"{name}.yaml"
            y.write_text(yaml.safe_dump(cfg, sort_keys=False))
            python = shutil.which("python", path=env["PATH"]) or sys.executable
            run([python, ROOT / "scripts" / "llamafactory_quant.py", y, stats], env, log)
            with open(pred / "generated_predictions.jsonl") as src, (out_dir / f"{name}.partial").open("w") as dst:
                for i, line in enumerate(src):
                    dst.write(json.dumps({"index": i, "predict": json.loads(line)["predict"]}, ensure_ascii=False) + "\n")
            finalize(jsonl, out_dir / f"{name}.partial", final)
            shutil.rmtree(pred)
        else:
            vlm_python = os.environ.get("VLM_PYTHON", sys.executable)
            partial = out_dir / f"{name}.partial"
            run([vlm_python, "-m", "esdvideo.vlm", "predict", "--model", a.model, "--model-path", weights,
                 "--weights", run_dir / "best.safetensors", "--data", jsonl, "--out", partial,
                 "--batch", str(spec["predict_batch"]), "--quant", a.quant], env, log)
            shutil.move(str(partial) + ".stats.json", stats)
            finalize(jsonl, partial, final)
        (out_dir / f"{name}.partial").unlink(missing_ok=True)
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
