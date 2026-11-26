#!/usr/bin/env python
"""Run a LlamaFactory prediction YAML with bitsandbytes quantization of the language model only.

    python scripts/llamafactory_quant.py PREDICT.yaml STATS.json

LlamaFactory's on-the-fly bitsandbytes quantization (``quantization_bit`` in the YAML) converts every
linear layer, including the vision encoder. For a like-for-like comparison with esdvideo.vlm this wrapper
keeps the vision encoder with its merger (``visual``) and ``lm_head`` in bf16 (``llm_int8_skip_modules``),
and writes peak GPU memory and throughput to STATS.json. Run on one GPU (FORCE_TORCHRUN=0).
"""

from __future__ import annotations

import json
import sys
import time

import torch
import transformers
import yaml

SKIP = ["visual", "lm_head"]


class _SkipVision(transformers.BitsAndBytesConfig):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("llm_int8_skip_modules", SKIP)
        super().__init__(*args, **kwargs)


def main() -> None:
    cfg_path, stats_path = sys.argv[1], sys.argv[2]
    transformers.BitsAndBytesConfig = _SkipVision
    import llamafactory.model.model_utils.quantization as q
    q.BitsAndBytesConfig = _SkipVision
    from llamafactory.train.tuner import run_exp

    cfg = yaml.safe_load(open(cfg_path))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    run_exp(cfg)
    total = time.time() - t0
    out = f"{cfg['output_dir']}/generated_predictions.jsonl"
    rows = sum(1 for _ in open(out))
    pred = json.load(open(f"{cfg['output_dir']}/predict_results.json"))
    stats = {"model": "qwen38", "quant": {8: "int8", 4: "nf4"}.get(cfg.get("quantization_bit"), "bf16"), "rows": rows,
             "seconds": round(pred["predict_runtime"], 1), "rows_per_s": round(rows / pred["predict_runtime"], 3),
             "batch": cfg.get("per_device_eval_batch_size"), "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
             "seconds_with_loading": round(total, 1)}
    open(stats_path, "w").write(json.dumps(stats) + "\n")
    print(stats, flush=True)


if __name__ == "__main__":
    main()
