# integration — the only place that knows *both* halves

End-to-end pipelines + model adapters. This is where the localization half and
the generation half are wired together (through `e2w_core`, never by importing
each other) and where outputs are written to `predictions/` for the benchmark.

```
integration/
  pipelines/   source → abduction → planner → gated renderer
               (the three training stages: align / end-to-end / optional RL)
  adapters/    e2w_adapter / bernini_adapter / vace_adapter → write predictions/<run>/
```

- `pipelines/` implements the inference flow (architecture §A.1) and the
  three-stage training orchestration (proposal §2.7).
- `adapters/` turn any model's output into the `predictions/<run>/` shape defined
  in [`e2w_core.io_contract`](../packages/e2w_core/e2w_core/io_contract.py). The
  benchmark then consumes that directory — it never imports anything here (B1).
  A working reference for the adapter pattern already exists:
  [`physics_iq_for_simple_eval/tools/make_prediction_run.py`](../../physics_iq_for_simple_eval/tools/make_prediction_run.py).

May import `e2w_core` and both halves; the benchmark must not import `integration`.
