# Task Plan: orin-quantize-env

## Goal
Make the uv-managed dependencies work on Jetson AGX Orin (Python 3.10) such that `scripts/quantize.py` runs without import/runtime dependency failures.

## Phases
- [x] Phase 1: Understand + plan
- [x] Phase 2: Investigate / gather context
- [x] Phase 3: Implement
- [x] Phase 4: Validate
- [x] Phase 5: Wrap up (deliverable)
- [x] Phase 6: Inference warning fix

## Key Questions
1. Should Orin support be additive (keep x86_64 CUDA extras intact) or replace the current defaults?
2. Does `scripts/quantize.py` need to run via its uv-script shebang, or is “run inside `.venv`” acceptable?
3. Can we avoid rebuilding `torchvision` via uv by reusing the existing Jetson procedure, or must uv handle it end-to-end?

## Decisions
- Prefer additive Orin support (markers / new extras) to avoid breaking existing x86_64 flows.

## Errors Encountered
- 2026-02-05 (UTC) `llmcompressor==0.3.0` failed to import with newer `compressed-tensors` (missing `safe_permute`, `KVCacheScaleType`) → pinned `compressed-tensors==0.10.2` for `aarch64`.
- 2026-02-05 (UTC) `torchvision` built under build isolation linked against a different torch (undefined symbols / `has_ops=False`) → rebuild with `--no-build-isolation` against Jetson torch.
- 2026-02-05 (UTC) NVIDIA Jetson torch wheel filename/version mismatch caused uv lock parse errors → run uv commands with `UV_SKIP_WHEEL_FILENAME_CHECK=1` (also embedded in `scripts/quantize.py` shebang).
- 2026-02-05 (UTC) `llmcompressor==0.3.0` recipe creation from Modifier objects generated YAML with `!!python/tuple`, but was parsed via `yaml.safe_load` → generate recipe YAML via `yaml.safe_dump` and pass the string to `llmcompressor.apply()`.
- 2026-02-05 (UTC) Sample generation failed with tensors split across CPU/CUDA → move model to the selected device and collate calibration tensors onto the same device.
- 2026-02-05 (UTC) Jetson torch SDPA lacks `enable_gqa` support (`TypeError: scaled_dot_product_attention(... enable_gqa=...)`) → use `attn_implementation=\"eager\"` when `enable_gqa` isn’t supported.
- 2026-02-05 (UTC) `scripts/inference_sample.py` printed an incorrect-tokenizer-regex warning when loading a locally-quantized model dir → pass `fix_mistral_regex=True` to `Qwen3VLProcessor.from_pretrained()` (with backward-compatible fallback).

## Status
**CURRENT:** Done
