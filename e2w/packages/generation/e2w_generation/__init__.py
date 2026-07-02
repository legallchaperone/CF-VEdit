"""e2w_generation — 生成半: frozen CogVideoX-Fun/VOID pass1 renderer (ADR-0007).

Consumes the seam types from ``e2w_core`` (``SourceLatent``, ``ThreeLayerMask``,
``EditPlan``) and produces the edited video. v0 renderer = ``void_renderer.py``
(frozen CogVideoX-Fun-InP + ``void_pass1.safetensors``, VOID mask channel-concat);
source payload = ``void_abduction.py``. The earlier VACE/Wan implementation
(``renderer.py``/``abduction.py``) was removed per ADR-0007. Depends on ``e2w_core``
only; must never import ``e2w_localization`` (boundary B3).
"""
