"""Stage-0 seg_dir LoRA finetune config — Sa2VA-Qwen2.5-VL-7B on DAVIS direct masks.

Our copy of the vendored `sa2va_qwen_finetune.py` (B4: vendored upstream is never
edited in place; our run config lives in our tree). Differences from upstream:
  - 7B base instead of 3B (`path`, `pretrained_pth`)
  - data root points at the adapter output (e2w_localization.training.data)
  - dataset `repeats`/epochs/save tuned for ~1k real frames (not the 100x tiny-set
    default)

Run from the vendored Sa2VA dir so `from projects.sa2va...` resolves:
  bash third_party/sa2va/tools/dist.sh train <this file> <NGPU>
See training/run_stage0.md for env + weight-conversion (convert_to_pth) steps.
Loss / LoRA / SAM2 injection are all upstream (2*BCE + 0.5*Dice; LoRA r=128).
"""
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen2_5_VLProcessor

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.sa2va.models import Sa2VAModel, SAM2TrainRunner, DirectResize
from projects.sa2va.datasets import sa2va_collect_fn, Sa2VAFinetuneDataset
from projects.sa2va.datasets.data_utils import ConcatDatasetSa2VA
from projects.sa2va.models.mllm.qwenvl import Qwen2_5_VL

# ---- paths (set by run_stage0.md weight-prep) -----------------------------
# Base Qwen2.5-VL-7B (architecture + tokenizer). Fetched via hf-mirror if absent.
path = 'Qwen/Qwen2.5-VL-7B-Instruct'
# Sa2VA-Qwen2.5-VL-7B converted to .pth via tools/convert_to_pth.py --arch-type qwen
pretrained_pth = '/data/cwx/e2w-data/sa2va_ckpt/Sa2VA-Qwen2_5-VL-7B.pth'
# Adapter output (images/ + annotations.json) from e2w_localization.training.data
RES_ROOT = '/data/cwx/e2w-data/sa2va_stage0_segdir/'

# ---- schedule (tuned for ~1k frames; override at launch as needed) --------
template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192
batch_size = 1
accumulative_counts = 4
dataloader_num_workers = 8
max_epochs = 5
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.05
save_steps = 500
save_total_limit = 3
special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

tokenizer = dict(type=AutoTokenizer.from_pretrained, pretrained_model_name_or_path=path,
                 trust_remote_code=True, padding_side='right')
extra_image_processor = dict(type=DirectResize, target_length=1024)

# freeze_llm + freeze_visual, LoRA on the LLM, SAM2 mask decoder trainable, seg
# projection trainable → exactly the Stage-0 trainable set (LoRA + projection +
# mask decoder). Loss is per-mask Dice+BCE.
model = dict(
    type=Sa2VAModel, training_bs=batch_size, special_tokens=special_tokens,
    pretrained_pth=pretrained_pth, loss_sample_points=True, frozen_sam2_decoder=False,
    arch_type='qwen',
    mllm=dict(type=Qwen2_5_VL, model_path=path, freeze_llm=True, freeze_visual_encoder=True,
              llm_lora=dict(type=LoraConfig, r=128, lora_alpha=256, lora_dropout=0.05,
                            bias='none', task_type='CAUSAL_LM',
                            modules_to_save=["embed_tokens", "lm_head"])),
    tokenizer=tokenizer,
    grounding_encoder=dict(type=SAM2TrainRunner),
    loss_mask=dict(type=CrossEntropyLoss, use_sigmoid=True, reduction='mean', loss_weight=2.0),
    loss_dice=dict(type=DiceLoss, use_sigmoid=True, activate=True, reduction='mean',
                   naive_dice=True, eps=1.0, loss_weight=0.5),
)

sa2va_default_dataset_configs = dict(
    tokenizer=tokenizer, special_tokens=special_tokens,
    extra_image_processor=extra_image_processor, prompt_template=prompt_template,
    max_length=max_length)

train_dataset = dict(type=ConcatDatasetSa2VA, datasets=[
    dict(type=Sa2VAFinetuneDataset, name='Stage0SegDir', data_root=RES_ROOT,
         data_prefix=dict(img_path='images/'), ann_file='annotations.json',
         serialize_data=False, repeats=1, arch_type='qwen',
         preprocessor=dict(type=Qwen2_5_VLProcessor.from_pretrained,
                           pretrained_model_name_or_path=path, trust_remote_code=True),
         **sa2va_default_dataset_configs),
])
train_dataloader = dict(
    batch_size=batch_size, num_workers=dataloader_num_workers, dataset=train_dataset,
    sampler=dict(type=LengthGroupedSampler, length_property='modality_length',
                 per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=sa2va_collect_fn))

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=AdamW, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts, loss_scale='dynamic', dtype='bfloat16')

param_scheduler = [
    dict(type=LinearLR, start_factor=1e-5, by_epoch=True, begin=0,
         end=warmup_ratio * max_epochs, convert_to_iter_based=True),
    dict(type=CosineAnnealingLR, eta_min=0.0, by_epoch=True,
         begin=warmup_ratio * max_epochs, end=max_epochs, convert_to_iter_based=True),
]
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

custom_hooks = []
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, save_optimizer=False, by_epoch=False,
                    interval=save_steps, max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook))
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))
visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=0, deterministic=False)
log_processor = dict(by_epoch=False)
