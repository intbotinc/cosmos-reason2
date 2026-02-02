# Task Plan: tensorrt-agx-feasibility

## Goal
Provide a practical assessment of whether Cosmos-Reason2 (2B/8B) can be converted to a TensorRT(-LLM) engine and run on a Jetson AGX device, including likely blockers and a recommended path.

## Phases
- [x] Phase 1: Understand + plan
- [x] Phase 2: Investigate / gather context
- [x] Phase 3: Implement (N/A — assessment only)
- [x] Phase 4: Validate
- [x] Phase 5: Wrap up (deliverable)

## Key Questions
1. Which target device is “AGX” here: Jetson AGX Orin (Ampere, CUDA 12.x) or Jetson AGX Thor (CUDA 13.x)?
2. Is the target workflow full multimodal (image/video) or text-only?
3. Is “TensorRT model” allowed to mean a TensorRT-LLM pipeline (multiple engines + runtime), or must it be a single TensorRT engine?
4. What precision/quantization is acceptable on-device (FP16 only vs INT8; FP8/NVFP4 are not viable on Orin)?
5. Are there existing/maintained TensorRT(-LLM) recipes for Qwen3-VL (Cosmos-Reason2 base) on aarch64?

## Decisions
- Focus on feasibility + required work (this repo does not ship TRT export); avoid speculative performance claims without device profiling.
- Treat Cosmos-Reason2 TensorRT(-LLM) conversion as unsupported/high-effort until Qwen3-VL + Jetson runtime support is explicitly available.
- Treat NVIDIA pip TensorRT/TRT-LLM wheels as non-viable on Jetson unless explicitly Tegra-supported (TensorRT pip wheels error out on Tegra).

## Errors Encountered
- None yet.

## Status
**CURRENT:** Done — updated assessment (Jetson AGX Orin, text-first) recorded in `devdocs/tasks/tensorrt-agx-feasibility/deliverable.md`.
