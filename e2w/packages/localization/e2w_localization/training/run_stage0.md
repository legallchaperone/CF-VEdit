# Stage-0 seg_dir finetune — runbook

Trains `seg_dir` as Sa2VA referring segmentation on the DAVIS direct masks, using
the **vendored** Sa2VA XTuner trainer (`third_party/sa2va`). All paths below are
examples; the code is complete, the **env is the one piece not yet stood up on
this box** (no conda env has `xtuner` — see step 1).

Conventions: China network → `unset` proxies for HF and use the mirror
(`HF_ENDPOINT=https://hf-mirror.com`); big artifacts under `/data/cwx`.

## 0. Build the training data (done, no GPU)
```bash
cd e2w/packages/localization
PYTHONPATH=$(pwd) python -m e2w_localization.training.data \
  --out-root /data/cwx/e2w-data/davis2017_remove \
  --dst     /data/cwx/e2w-data/sa2va_stage0_segdir \
  --layer direct --frame-stride 5
# -> images/ + annotations.json (~974 frames) + val.jsonl (11 held-out clips)
```

## 1. Environment (PENDING — the gap)
No env here has `xtuner`+`deepspeed`. `edit2world-phase1-real` is closest
(torch 2.7 + mmengine + flash-attn + transformers 4.56). Clone it and add the two:
```bash
conda create -n sa2va-train --clone edit2world-phase1-real
conda activate sa2va-train
HF_ENDPOINT=https://hf-mirror.com pip install xtuner deepspeed
# xtuner pins an older transformers for InternVL; Sa2VA-Qwen loads on 4.51+ (the
# planner already handles the key-mapping). If xtuner downgrades transformers and
# Qwen2.5-VL fails to import, pin transformers back to 4.51-4.56 and re-test load.
python -c "from projects.sa2va.models import Sa2VAModel"   # smoke: run from third_party/sa2va
```

## 2. Weights (one-time)
```bash
# base Qwen2.5-VL-7B (arch + tokenizer for the config `path`)
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /data/cwx/Qwen2.5-VL-7B-Instruct
# Sa2VA-7B HF checkpoint -> .pth for xtuner `pretrained_pth`
cd e2w/packages/localization/third_party/sa2va
python tools/convert_to_pth.py /data/cwx/Sa2VA-Qwen2_5-VL-7B \
  --save-path /data/cwx/e2w-data/sa2va_ckpt/Sa2VA-Qwen2_5-VL-7B.pth --arch-type qwen
```
Then set `path`, `pretrained_pth`, `RES_ROOT` in
`e2w_localization/training/configs/sa2va_qwen7b_stage0_segdir.py`.

## 3. Train
```bash
cd e2w/packages/localization/third_party/sa2va        # so `from projects.sa2va` resolves
CFG=../../e2w_localization/training/configs/sa2va_qwen7b_stage0_segdir.py
HF_ENDPOINT=https://hf-mirror.com bash tools/dist.sh train "$CFG" <NGPU>
# -> work_dirs/<cfg>/iter_*.pth ; adjust accumulative_counts so bs*accum*NGPU≈16
```

## 4. Convert back + eval
```bash
python tools/convert_to_hf.py "$CFG" --pth-model work_dirs/.../iter_XXXX.pth \
  --save-path /data/cwx/e2w-data/sa2va_ckpt/stage0_segdir_hf --arch-type qwen
# held-out IoU/Dice — SAME command for the zero-shot baseline (point --weights at
# the stock Sa2VA-7B) and the finetuned model (point at the converted dir):
cd e2w/packages/localization
PYTHONPATH=$(pwd) python -m e2w_localization.training.eval \
  --val-jsonl /data/cwx/e2w-data/sa2va_stage0_segdir/val.jsonl \
  --out-root  /data/cwx/e2w-data/davis2017_remove \
  --weights   /data/cwx/Sa2VA-Qwen2_5-VL-7B \
  --report /data/cwx/e2w-data/sa2va_stage0_segdir/eval_zeroshot.json
```

## Calibrate, don't guess (TRAINING_NOTES)
Run the eval on the **stock** model first (zero-shot floor). seg_dir is
in-distribution for Sa2VA, so the floor is likely already high — training is a
polish, and the interesting signal is the learning curve (train on 25/50/100% of
`annotations.json`, plot held-out IoU) to size the data appetite before scaling
sources. Watch the empty-prediction rate in the eval summary.
