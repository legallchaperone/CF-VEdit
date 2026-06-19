"""e2w_data_engine — Kubric-style sim: shared-seed factual/CF pairs + dep graph.

Emits the three-piece training set (V, V*, M) and object-level causal
dependency-graph labels that supervise the indirect mask layer (novelty ②).
Skeleton only — see the package README. Dev/val only, never in the report
(boundary B5). Depends on ``e2w_core`` only.
"""
