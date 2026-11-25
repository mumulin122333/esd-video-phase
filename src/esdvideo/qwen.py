"""LlamaFactory configs for Qwen3.8-27B (LoRA SFT and prediction), same recipe as the other models.

LoRA r 8 / alpha 16 / dropout 0 on all language-model linear layers, frozen vision tower, trainable
projector, lr 1e-4 cosine with 10 % warmup, 3 epochs, per-device batch 3 x accumulation 2 (global
batch 6 on one GPU), bf16, gradient checkpointing, best epoch by validation loss; video at 1 fps,
at most 32 frames, at most 50,176 pixels per frame.
"""

from __future__ import annotations

from pathlib import Path

VIDEO = """video_max_pixels: 50176
video_min_pixels: 4096
video_fps: 1.0
video_maxlen: 32"""


def train_yaml(base: Path, data_dir: Path, out_dir: Path, seed: int = 20261022) -> str:
    return f"""### model
model_name_or_path: {base}
trust_remote_code: true
{VIDEO}
flash_attn: sdpa

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0
lora_target: all
freeze_vision_tower: true
freeze_multi_modal_projector: false

### dataset
dataset: train
eval_dataset: val
dataset_dir: {data_dir}
template: qwen3_5_nothink
cutoff_len: 8192
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {out_dir}
logging_steps: 5
save_strategy: epoch
save_total_limit: 3
overwrite_output_dir: true
plot_loss: true
report_to: none

### train
per_device_train_batch_size: 3
gradient_accumulation_steps: 2
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
max_grad_norm: 1.0
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
ddp_find_unused_parameters: false
seed: {seed}

### eval
per_device_eval_batch_size: 2
eval_strategy: epoch
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
"""


def predict_yaml(base: Path, adapter: Path, dataset: str, dataset_dir: Path, out_dir: Path, batch: int = 8) -> str:
    return f"""### model
model_name_or_path: {base}
adapter_name_or_path: {adapter}
trust_remote_code: true
{VIDEO}
flash_attn: sdpa

### method
stage: sft
do_predict: true
predict_with_generate: true
finetuning_type: lora

### dataset
eval_dataset: {dataset}
dataset_dir: {dataset_dir}
template: qwen3_5_nothink
cutoff_len: 8192
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {out_dir}
overwrite_output_dir: true
report_to: none

### eval
per_device_eval_batch_size: {batch}
bf16: true
ddp_timeout: 180000000

### generate
max_new_tokens: 128
do_sample: false
"""
