"""LoRA fine-tuning and prediction for VideoChat3-4B, InternVideo3-8B and Molmo2-8B.

Data: LlamaFactory-format JSONL (messages + videos), the same windows, prompt and JSON target as the
Qwen3.8-27B LlamaFactory runs (esdvideo.qwen).

Recipe identical to the Qwen3.8 phase LoRA: LoRA r8 / alpha 16 / dropout 0 on every linear layer of the
language model, vision encoder frozen, vision-language projector fully trained, AdamW lr 1e-4, cosine
with 10 % warmup, max grad norm 1.0, 3 epochs, effective batch 6, bf16; validation loss after every epoch
and the lowest-loss epoch is kept. Video input: 1 fps, at most 32 frames (a 64 s clip is sampled to 32),
<= 50,176 pixels per frame for the Qwen-style processors (Molmo2 uses its fixed 378x378 frame size).

    python -m esdvideo.vlm train   --model videochat3 --model-path P --data DIR --out RUN_DIR
    python -m esdvideo.vlm predict --model videochat3 --model-path P --weights RUN_DIR/best.safetensors \
        --data FILE.jsonl --out preds.jsonl [--shard 0/2] [--batch 8]

Model-specific handling:
  * VideoChat3: no flash-attn -> vision attention via SDPA, with its dense whole-batch mask replaced by
    per-segment attention (numerically identical, far less memory).
  * InternVideo3: its remote code mis-handles left-padded batches -> prediction batches only windows of
    equal length (same frame count, no padding).
  * Molmo2: prediction under bf16 autocast as in training (its ViT otherwise runs float32 attention);
    frames capped with num_frames (molmo_utils' default samples a 1 fps clip twice per second).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

FPS = 1.0
MAX_FRAMES = 32
MAX_PIXELS = 50176
MIN_PIXELS = 4096
EOS = "<|im_end|>"

SPECS = {
    "videochat3": dict(auto="causal", utils="qwen", patch=14,
                       llm=r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.\w+",
                       projector=("model.multi_modal_projector.",)),
    "internvideo3": dict(auto="causal", utils="qwen", patch=16,
                         llm=r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.\w+",
                         projector=("model.visual.merger.",)),
    "molmo2": dict(auto="image_text", utils="molmo", patch=14,
                   llm=r"model\.transformer\.blocks\.\d+\.(self_attn|mlp)\.\w+",
                   projector=("model.vision_backbone.image_projector.", "model.vision_backbone.image_pooling_2d.")),
}


# ----------------------------------------------------------------------------- model / inputs
# Modules kept in bf16 when the language model is quantized: vision encoder, projector and output layer
# (plus InternVideo3's MLA kv_b_proj, whose weight is used directly in the absorbed attention einsum).
QUANT_SKIP = {
    "videochat3": ["vision_tower", "multi_modal_projector", "lm_head"],
    "internvideo3": ["visual", "lm_head", "kv_b_proj"],
    "molmo2": ["vision_backbone", "lm_head"],
}


def quantization_config(name: str, quant: str | None):
    """bitsandbytes config for the language model only (int8: LLM.int8; nf4: 4-bit NF4, double quantization)."""
    if not quant or quant == "bf16":
        return None
    from transformers import BitsAndBytesConfig

    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=QUANT_SKIP[name])
    if quant == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=QUANT_SKIP[name])
    raise ValueError(f"unknown quantization {quant!r}")


def load_model(name: str, path: str, device: str, quant: str | None = None):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

    spec = SPECS[name]
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    vc = getattr(cfg, "vision_config", None)
    if vc is not None and getattr(vc, "attn_impl", None) == "flash_attention_2":
        vc.attn_impl = "sdpa"  # VideoChat3 vision tower: no flash-attn installed
    cls = AutoModelForCausalLM if spec["auto"] == "causal" else AutoModelForImageTextToText
    qc = quantization_config(name, quant)
    if qc is None:
        model = cls.from_pretrained(path, config=cfg, dtype=torch.bfloat16, trust_remote_code=True,
                                    attn_implementation="sdpa").to(device)
    else:
        model = cls.from_pretrained(path, config=cfg, dtype=torch.bfloat16, trust_remote_code=True,
                                    attn_implementation="sdpa", quantization_config=qc, device_map={"": device})
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    _patch_block_sdpa(model)
    return model, processor


def _block_sdpa(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    """Exact replacement for VideoChat3's vision sdpa_attention: attention within each cu_seqlens
    segment, without the dense (tokens x tokens) mask the original builds for the whole batch."""
    import torch.nn.functional as F

    outs = []
    bounds = q_cu_seqlens.tolist()
    for a, b in zip(bounds[:-1], bounds[1:]):
        qi, ki, vi = (x[a:b].transpose(0, 1) for x in (q, k, v))
        outs.append(F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0).transpose(0, 1))
    out = torch.cat(outs, 0)
    return out.reshape(out.shape[0], -1)


def _patch_block_sdpa(model) -> None:
    import sys

    mod = sys.modules.get(type(model).__module__)
    table = getattr(mod, "VL_VISION_ATTENTION_FUNCTIONS", None)
    if table is not None and "sdpa" in table:
        table["sdpa"] = _block_sdpa


def _user_text(record: dict) -> str:
    return record["messages"][0]["content"].replace("<video>", "").strip()


def _messages(name: str, video: str, text: str) -> list[dict]:
    # At most MAX_FRAMES frames, as LlamaFactory video_maxlen 32 for Qwen3.8: a <=32 s 1 fps window
    # gives every frame; a 64 s Renji boundary clip gives 32 uniformly spaced frames.
    if SPECS[name]["utils"] == "molmo":
        # molmo_utils' default (384 frames, max_fps 2) would sample a 1 fps clip twice per second
        return [{"role": "user", "content": [{"type": "video", "video": video, "num_frames": MAX_FRAMES},
                                             {"type": "text", "text": text}]}]
    return [{"role": "user", "content": [
        {"type": "video", "video": video, "fps": FPS, "max_frames": MAX_FRAMES, "max_pixels": MAX_PIXELS,
         "min_pixels": MIN_PIXELS},
        {"type": "text", "text": text}]}]


def encode(name: str, processor, records: list[dict], with_answer: bool) -> dict:
    """Processor inputs for a batch; with_answer adds the target and -100-masked labels."""
    from transformers.video_utils import VideoMetadata

    spec = SPECS[name]
    texts, videos, metas, kwargs = [], [], [], {}
    for r in records:
        msgs = _messages(name, r["videos"][0], _user_text(r))
        if spec["utils"] == "molmo":
            from molmo_utils import process_vision_info
            _, vids, vk = process_vision_info(msgs)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            from qwen_vl_utils import process_vision_info
            _, vids, vk = process_vision_info(msgs, image_patch_size=spec["patch"], return_video_kwargs=True,
                                              return_video_metadata=True)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        v, m = vids[0]
        if isinstance(m, dict) and spec["utils"] == "qwen":
            fields = {f.name for f in dataclasses.fields(VideoMetadata)}
            m = VideoMetadata(**{k: x for k, x in m.items() if k in fields})
        if with_answer:
            text = text + r["messages"][1]["content"] + EOS
        texts.append(text)
        videos.append(v)
        metas.append(m)
        kwargs = vk or {}
    extra = {} if spec["utils"] == "molmo" else {"do_resize": False}
    inputs = processor(text=texts, videos=videos, video_metadata=metas, padding=True, return_tensors="pt",
                       **extra, **kwargs)
    inputs = dict(inputs)
    if with_answer:
        labels = torch.full_like(inputs["input_ids"], -100)
        for i, r in enumerate(records):
            n = len(processor.tokenizer(r["messages"][1]["content"] + EOS, add_special_tokens=False)["input_ids"])
            ids = inputs["input_ids"][i]
            mask = inputs["attention_mask"][i].bool()
            last = int(mask.nonzero().max()) + 1  # right edge of the real tokens
            labels[i, last - n:last] = ids[last - n:last]
        inputs["labels"] = labels
    return inputs


def to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(torch.bfloat16)
        out[k] = v
    return out


# ----------------------------------------------------------------------------- trainable state
def add_lora(name: str, model, rank: int, alpha: int):
    spec = SPECS[name]
    targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and re.fullmatch(spec["llm"], n)]
    if not targets:
        raise RuntimeError("no LoRA target modules found")
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets,
                                             bias="none"))
    for n, p in model.named_parameters():
        if any(f".{pref}" in n or n.startswith(pref) for pref in spec["projector"]) and "lora_" not in n:
            p.requires_grad_(True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().to(torch.bfloat16).cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, path: str) -> None:
    state = load_file(path)
    own = dict(model.named_parameters())
    missing = [k for k in state if k not in own]
    if missing:
        raise RuntimeError(f"{len(missing)} saved tensors not in model, e.g. {missing[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            own[k].copy_(v.to(own[k].dtype))


# ----------------------------------------------------------------------------- data
class Rows(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.open()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def loader(name, processor, rows, batch, shuffle, with_answer, seed=0, workers=4):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(rows, batch_size=batch, shuffle=shuffle, generator=g, num_workers=workers,
                      collate_fn=lambda rs: (rs, encode(name, processor, rs, with_answer)), persistent_workers=False)


# ----------------------------------------------------------------------------- train
def cmd_train(args) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    model, processor = load_model(args.model, args.model_path, args.device)
    model = add_lora(args.model, model, args.rank, args.alpha)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params {n_train / 1e6:.1f} M", flush=True)
    if args.grad_ckpt:
        # non-reentrant checkpointing needs no grad-requiring inputs (Molmo2 edits embeddings in place)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train, val = Rows(data / "train.jsonl"), Rows(data / "val.jsonl")
    steps_per_epoch = math.ceil(len(train) / args.grad_accum)
    total = steps_per_epoch * args.epochs
    warm = math.ceil(total * args.warmup)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    log = (out / "train_log.jsonl").open("a")
    history, step = [], 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        run, k = 0.0, 0
        for i, (_, batch) in enumerate(loader(args.model, processor, train, 1, True, True, args.seed + epoch, args.workers)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**to_device(batch, args.device)).loss
            (loss / args.grad_accum).backward()
            run += float(loss)
            k += 1
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0:
                    rec = {"epoch": epoch, "step": step, "loss": run / k, "lr": sched.get_last_lr()[0],
                           "elapsed_s": round(time.time() - t0), "max_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(rec, flush=True)
                    run, k = 0.0, 0
        val_loss = evaluate_loss(args, model, processor, val)
        save_file(trainable_state(model), str(out / f"epoch{epoch}.safetensors"))
        history.append({"epoch": epoch, "val_loss": val_loss})
        log.write(json.dumps({"epoch": epoch, "val_loss": val_loss}) + "\n")
        log.flush()
        print(f"epoch {epoch} val_loss {val_loss:.5f}", flush=True)
    best = min(history, key=lambda h: (h["val_loss"], h["epoch"]))
    os.replace(out / f"epoch{best['epoch']}.safetensors", out / "best.safetensors")
    for h in history:
        f = out / f"epoch{h['epoch']}.safetensors"
        if f.exists():
            f.unlink()
    (out / "train_summary.json").write_text(json.dumps({
        "model": args.model, "model_path": args.model_path, "data": str(data), "train_windows": len(train),
        "val_windows": len(val), "epochs": history, "best_epoch": best["epoch"], "optimizer_steps": step,
        "hparams": {k: v for k, v in vars(args).items() if k not in ("func",)},
        "train_seconds": round(time.time() - t0)}, indent=2) + "\n")
    (out / "TRAIN_COMPLETE").touch()


@torch.no_grad()
def evaluate_loss(args, model, processor, rows) -> float:
    model.eval()
    tot, n = 0.0, 0
    for recs, batch in loader(args.model, processor, rows, 1, False, True, 0, args.workers):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**to_device(batch, args.device)).loss
        tot += float(loss)
        n += 1
    model.train()
    return tot / max(1, n)


# ----------------------------------------------------------------------------- predict
@torch.no_grad()
def cmd_predict(args) -> None:
    rows = Rows(Path(args.data))
    idx = list(range(len(rows)))
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        idx = idx[k::n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["index"] for line in out.open()}
    todo = [i for i in idx if i not in done]
    print(f"{len(todo)} of {len(idx)} rows to predict", flush=True)
    if not todo:
        return
    torch.cuda.reset_peak_memory_stats()
    model, processor = load_model(args.model, args.model_path, args.device, args.quant)
    if args.weights:
        model = add_lora(args.model, model, args.rank, args.alpha)
        load_trainable(model, args.weights)
    if args.quant in (None, "bf16"):
        model.to(torch.bfloat16)
    else:  # quantized weights cannot be cast; LoRA and projector weights go to bf16
        for p_ in model.parameters():
            if p_.dtype == torch.float32:
                p_.data = p_.data.to(torch.bfloat16)
    model.eval()
    loaded_gb = torch.cuda.memory_allocated() / 1e9
    # Batch only windows of the same length: same frame count and size -> same token count, so no
    # padding (InternVideo3's remote code mis-handles left-padded rows).
    groups: dict[int, list[dict]] = {}
    for i in todo:
        n = int(re.search(r"Segment this (\d+)-second", rows[i]["messages"][0]["content"]).group(1))
        groups.setdefault(n, []).append(dict(rows[i], _index=i))
    batches = [g[s:s + args.batch] for _, g in sorted(groups.items()) for s in range(0, len(g), args.batch)]
    sub = [r for b in batches for r in b]
    t0 = time.time()
    dl = DataLoader(batches, batch_size=None, shuffle=False, num_workers=args.workers,
                    collate_fn=lambda rs: (rs, encode(args.model, processor, rs, False)))
    with out.open("a") as fh:
        eos = processor.tokenizer.convert_tokens_to_ids(EOS)

        # Molmo2 runs its ViT attention in float32 unless autocast is on (as in training); without it
        # a 16-video batch needs > 95 GB. VideoChat3 / InternVideo3 keep the plain bf16 path they used.
        use_autocast = args.model == "molmo2"

        def try_generate(batch) -> list[str] | None:
            try:
                batch = to_device(batch, args.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    gen = model.generate(**batch, max_new_tokens=128, do_sample=False, eos_token_id=eos,
                                         pad_token_id=processor.tokenizer.pad_token_id)
                return processor.tokenizer.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            except torch.OutOfMemoryError:
                return None  # retry happens outside the handler so the failed attempt's memory is freed

        def generate(recs, batch) -> list[str]:
            """Greedy decode; on CUDA OOM split the batch in half and retry (same outputs)."""
            texts = try_generate(batch)
            if texts is not None:
                return texts
            del batch
            torch.cuda.empty_cache()
            if len(recs) == 1:
                raise RuntimeError("CUDA OOM even at batch size 1")
            h = len(recs) // 2
            print(f"OOM at batch {len(recs)}, splitting", flush=True)
            return (generate(recs[:h], encode(args.model, processor, recs[:h], False))
                    + generate(recs[h:], encode(args.model, processor, recs[h:], False)))

        done_n = 0
        for j, (recs, batch) in enumerate(dl):
            if batch["attention_mask"].min() == 0:
                raise RuntimeError("padded batch")
            texts = generate(recs, batch)
            for r, t in zip(recs, texts):
                fh.write(json.dumps({"index": r["_index"], "predict": t}, ensure_ascii=False) + "\n")
            fh.flush()
            done_n += len(recs)
            if j % 20 == 0:
                print(f"{done_n}/{len(sub)} {done_n / (time.time() - t0):.2f} rows/s", flush=True)
    if args.quant is None:
        return
    stats = {"model": args.model, "quant": args.quant or "bf16", "rows": done_n, "seconds": round(time.time() - t0, 1),
             "rows_per_s": round(done_n / max(1e-9, time.time() - t0), 3), "batch": args.batch,
             "weights_loaded_gb": round(loaded_gb, 2), "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    Path(str(out) + ".stats.json").write_text(json.dumps(stats) + "\n")
    print(stats, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("train", "predict"):
        p = sub.add_parser(c)
        p.add_argument("--model", choices=sorted(SPECS), required=True)
        p.add_argument("--model-path", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--rank", type=int, default=8)
        p.add_argument("--alpha", type=int, default=16)
        p.add_argument("--workers", type=int, default=4)
    t = sub.choices["train"]
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=float, default=0.1)
    t.add_argument("--grad-accum", type=int, default=6)
    t.add_argument("--grad-ckpt", action="store_true")
    t.add_argument("--seed", type=int, default=20261022)
    p = sub.choices["predict"]
    p.add_argument("--weights")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--quant", choices=["bf16", "int8", "nf4"], default=None,
                   help="bitsandbytes quantization of the language model (vision encoder, projector, lm_head stay bf16)")
    p.add_argument("--shard")
    args = ap.parse_args()
    cmd_train(args) if args.cmd == "train" else cmd_predict(args)


if __name__ == "__main__":
    main()
