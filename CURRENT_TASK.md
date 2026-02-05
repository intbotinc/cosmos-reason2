# CURRENT_TASK

Active task: `devdocs/tasks/orin-quantize-env/`

## Context
- Port the uv-managed environment to Jetson AGX Orin (aarch64, Python 3.10)
- Align torch/torchvision/numpy constraints with `ORIN_VENV_SETUP.md`
- Ensure `scripts/quantize.py` dependencies resolve and the script can run on Orin
- Ensure the quantized model works with `scripts/inference_sample.py` (no tokenizer-regex warning)

## Task Files
- Plan: `devdocs/tasks/orin-quantize-env/task_plan.md`
- Worklog: `devdocs/tasks/orin-quantize-env/worklog.md`
- Deliverable: `devdocs/tasks/orin-quantize-env/deliverable.md`

## History Search
- `rg -n "<keywords>" devdocs/tasks docs/tasks -S`
