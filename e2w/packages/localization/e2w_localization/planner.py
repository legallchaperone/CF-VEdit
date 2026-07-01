"""Vanilla Sa2VA planner for E2W V0.

Boundary: emits only e2w_core seam types. It does not import generation.

Verified upstream API (pinned Sa2VA/HF remote code):
- load with transformers AutoModelForCausalLM + AutoProcessor, trust_remote_code=True;
- call model.predict_forward(video=<PIL frames>, text=..., processor=processor);
- result contains {'prediction', 'prediction_masks'};
- stock [SEG] masks live in result['prediction_masks'][0] when the decoded text
  contains [SEG].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from e2w_core.masks import ThreeLayerMask
from e2w_core.plan import EditPlan, Intervention, Operation


@dataclass(frozen=True)
class PlannerConfig:
    weights_path: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    frame_interval: int = 1
    max_frames_for_segmentation: int = 21
    initialize_bypassed_heads: bool = True
    edit_token_slots: int = 4
    renderer_condition_dim: int = 4096
    segmentation_prompt_template: str = "<image>Please segment {target_ref}. [SEG]"
    # Non-vanilla (full A.1) query-token path: how many frames enter the MLLM
    # context. SAM2 still propagates the mask across all loaded frames.
    max_content_frames: int = 5
    e2w_head_seed: int = 0


class CausalPlanner:
    """Wrap stock Sa2VA [SEG] for V0 vanilla direct-mask planning.

    Normal trained E2W will use [SEG_DIR]/[SEG_IND]/[EDIT]. In V0 these heads are
    initialized on the loaded model/tokenizer for shape compatibility, but they
    are intentionally bypassed: direct = stock [SEG], indirect = empty,
    edit_tokens = empty placeholder.
    """

    def __init__(self, config: PlannerConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._heads_initialized = False
        self.last_position_ids_mode = None

    def plan(self, video: str | Path | Iterable[Any], instruction: str, *, target_ref: str,
             operation: str | Operation, vanilla: bool = False) -> tuple[ThreeLayerMask, EditPlan]:
        frames = self._load_frames(video)
        if not frames:
            raise ValueError("planner received an empty video")

        intervention = Intervention(
            operation=Operation(operation) if isinstance(operation, str) else operation,
            target_ref=target_ref,
            instruction=instruction,
        )

        if vanilla:
            direct = self._predict_stock_seg_mask(frames, target_ref)
            indirect = self._zeros_like(direct)
            plan = EditPlan(
                intervention=intervention,
                region_query=None,
                edit_tokens=None,  # bypassed in vanilla; renderer uses native text condition.
            )
            return ThreeLayerMask(direct=direct, indirect=indirect), plan

        # Full A.1 (untrained): query-token [SEG_DIR]/[SEG_IND]/[EDIT] path. Emits a
        # real three-layer mask (indirect no longer zeros) + region_query + edit_tokens.
        from .overlay import attach_e2w_heads
        from .query_tokens import localize_three_layer

        model, processor = self._load_model()
        attach_e2w_heads(
            model,
            num_edit_slots=self.config.edit_token_slots,
            renderer_condition_dim=self.config.renderer_condition_dim,
            seed=self.config.e2w_head_seed,
        )
        out = localize_three_layer(
            model, processor, frames, instruction,
            num_edit_slots=self.config.edit_token_slots,
            max_content_frames=self.config.max_content_frames,
        )
        # Surfaced so the adapter can record it in run_meta — a silent M-RoPE
        # arange-fallback must not be mistaken for the validated path (ADR-0004).
        self.last_position_ids_mode = out.get("position_ids_mode")
        plan = EditPlan(
            intervention=intervention,
            region_query=out["region_query"],
            edit_tokens=out["edit_tokens"],
        )
        return ThreeLayerMask(direct=out["direct"], indirect=out["indirect"]), plan

    def unload(self) -> None:
        """Release Sa2VA weights before generation loads Wan/VACE."""
        import gc
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_model(self):
        if self._model is not None:
            return self._model, self._processor

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        from transformers.configuration_utils import PretrainedConfig

        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16
        model_config = AutoConfig.from_pretrained(self.config.weights_path, trust_remote_code=True)
        if isinstance(getattr(model_config, "text_config", None), dict):
            text_config = dict(model_config.text_config)
            # Sa2VA checkpoint/tokenizer uses text_config vocab_size; newer transformers
            # keeps Qwen2.5-VL's larger top-level default unless normalized here.
            if "vocab_size" in text_config:
                model_config.vocab_size = text_config["vocab_size"]
            # GenerationConfig.from_model_config in transformers>=4.51 expects
            # get_text_config(...).to_dict(), not a raw dict.
            model_config.text_config = PretrainedConfig.from_dict(text_config)

        model = AutoModelForCausalLM.from_pretrained(
            self.config.weights_path,
            config=model_config,
            torch_dtype=dtype,
            device_map=self.config.device,
            trust_remote_code=True,
            attn_implementation="eager",
            key_mapping={
                # Current transformers removed the language_model wrapper and
                # moved the visual tower one level up relative to Sa2VA's saved keys.
                r"^model\.model\.language_model\.": "model.model.",
                r"^model\.model\.visual\.": "model.visual.",
            },
        ).eval()
        processor = AutoProcessor.from_pretrained(self.config.weights_path, trust_remote_code=True)

        if self.config.initialize_bypassed_heads:
            self._initialize_bypassed_heads(model, processor)

        self._model = model
        self._processor = processor
        return model, processor

    def _initialize_bypassed_heads(self, model: Any, processor: Any) -> None:
        """Initialize [SEG_DIR]/[SEG_IND]/[EDIT] scaffolding without routing to it.

        This mirrors Sa2VA's [SEG] token registration pattern but keeps V0 output
        on the pretrained [SEG] path. The edit projection width is the verified
        Wan/VACE text-condition width from the Wan2.2 config (4096 by default).
        """
        if self._heads_initialized:
            return
        import torch
        from torch import nn

        tokenizer = processor.tokenizer
        # V0 must preserve the stock [SEG] path exactly. Registering new tokens and
        # resizing embeddings perturbs generation on this Sa2VA checkpoint under
        # transformers>=4.51, so the bypass scaffolding is initialized without
        # mutating tokenizer/model vocab. Trained E2W will own token registration.
        model.seg_dir_idx = tokenizer.convert_tokens_to_ids("[SEG_DIR]")
        model.seg_ind_idx = tokenizer.convert_tokens_to_ids("[SEG_IND]")
        model.edit_token_idx = tokenizer.convert_tokens_to_ids("[EDIT]")

        hidden = int(model.config.text_config.hidden_size)
        slots = int(self.config.edit_token_slots)
        out_dim = int(self.config.renderer_condition_dim)
        model.edit_hidden_fcs = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, slots * out_dim), nn.Dropout(0.0)
        ).to(next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        self._heads_initialized = True

    def _predict_stock_seg_mask(self, frames: list[Any], target_ref: str):
        import numpy as np

        model, processor = self._load_model()
        prompt = self.config.segmentation_prompt_template.format(target_ref=target_ref)
        result = model.predict_forward(video=frames, text=prompt, processor=processor)
        prediction = result.get("prediction", "")
        masks = result.get("prediction_masks") or []
        if not masks:
            first = frames[0]
            width, height = first.size
            return np.zeros((len(frames), height, width), dtype=bool)

        direct = np.asarray(masks[0]).astype(bool)
        if direct.ndim != 3:
            raise ValueError(f"Sa2VA mask must have shape (T,H,W), got {direct.shape}")
        if direct.shape[0] != len(frames):
            direct = self._temporal_nearest(direct, len(frames))
        return direct

    def _load_frames(self, video: str | Path | Iterable[Any]) -> list[Any]:
        from PIL import Image
        if not isinstance(video, (str, Path)):
            frames = list(video)
            return frames[: self.config.max_frames_for_segmentation]

        import cv2
        frames: list[Any] = []
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"failed to open video: {video}")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, self.config.frame_interval) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                if len(frames) >= self.config.max_frames_for_segmentation:
                    break
            idx += 1
        cap.release()
        return frames

    @staticmethod
    def _temporal_nearest(mask, target_t: int):
        import numpy as np
        if mask.shape[0] == target_t:
            return mask
        if mask.shape[0] == 0:
            raise ValueError("cannot upsample empty temporal mask")
        src_idx = np.linspace(0, mask.shape[0] - 1, target_t).round().astype(int)
        return mask[src_idx]

    @staticmethod
    def _zeros_like(mask):
        import numpy as np
        return np.zeros_like(mask, dtype=bool)
