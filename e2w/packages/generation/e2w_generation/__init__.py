"""e2w_generation — 生成半: Abduction inversion + gated Renderer (VACE/Wan-based).

Consumes the seam types from ``e2w_core`` (``SourceLatent``, ``ThreeLayerMask``,
``EditPlan``) and produces the edited video. Home of true novelties ① (source
inversion → U prior) and ③ (invariant-preservation loss). Skeleton only — see the
package README. Depends on ``e2w_core`` only; must never import
``e2w_localization`` (boundary B3).
"""
