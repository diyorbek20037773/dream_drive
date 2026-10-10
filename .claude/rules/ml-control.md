---
paths:
  - "inference/**"
  - "backend/app/perception/**"
  - "backend/app/agents/**"
  - "backend/app/control/**"
  - "backend/app/navigation/**"
  - "backend/app/simulators/**"
  - "backend/app/evaluation/**"
  - "config/models*.yaml"
---
# Perception and simulator rules
Read docs/ML_PIPELINE.md and docs/SIMULATION.md. Pin real weights/revisions and verify classes.
Inference only consumes RGB/static calibration. Ego/route telemetry is separate; dynamic truth is evaluator-only.
Image normalization/unpadding/mask coordinates must be tested. Tracking has per-run state/timestamps.
Synthetic RGB CV and learned models have different source labels and evaluation domains.
No LLM steering, relative depth as meters, fake autonomy, predetermined success or hidden autopilot.
Pixel ablation and hidden-truth isolation are required behavioral tests.
