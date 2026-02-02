# Worklog: tensorrt-agx-feasibility

## 2026-02-01 (UTC) — Codex

- Initialized task docs.
- Reviewed `README.md` and inference entrypoints: repo provides Transformers (`scripts/inference_sample.py`) and vLLM (`cosmos-reason2-inference`) paths, not TensorRT export/inference.
- Found existing Jetson AGX Orin notes under `devdocs/tasks/orin-venv-setup/` indicating TensorRT is installed but TensorRT engine export/TRT-LLM is not validated in this repo.
- Checked current TensorRT-LLM support matrix / docs: Qwen3 (text) is supported, but Qwen3-VL is not listed; VLM coverage includes Qwen2-VL / Qwen2.5-VL.
- Checked Jetson availability notes: community Jetson builds exist for some TRT-LLM versions, but official guidance still treats TRT-LLM on Jetson/Thor as unsupported.
- Verified this host is Jetson (aarch64, L4T r36.4.0) with TensorRT Python `10.3.0` and `trtexec` available under `/usr/src/tensorrt/bin/trtexec`.
- Confirmed Cosmos-Reason2-2B weights are cached locally and inspected `config.json`: model is `qwen3_vl` with `text_config.model_type=qwen3_vl_text` (MRoPE, 262k context).
- Ran a minimal **text-only** generation using `transformers==4.57.3` + Jetson torch (attention implementation forced to `eager` due to SDPA `enable_gqa` incompatibility).
- Surveyed TRT-LLM install options on Jetson Orin:
  - No `tensorrt-llm`/`trtllm` apt packages.
  - NVIDIA pip index (`pypi.nvidia.com/tensorrt-llm`) shows only a small set of `linux_aarch64` wheels for Python 3.10: `0.15.0` and two `0.16.0.dev*` builds; newer `linux_aarch64` wheels are `cp312` (Python 3.12).
  - The `0.15.0` / `0.16.0.dev*` aarch64 wheels pin `tensorrt~=10.6.0` and `transformers<=4.45.1`, and include converters for Qwen/Qwen2 but not Qwen3/Qwen3-VL.
  - Attempting to fetch `tensorrt-cu12==10.6.0` from NVIDIA’s pip index fails on Jetson with “TensorRT does not currently build wheels for Tegra systems”, so the pip dependency stack is not Jetson-friendly.
- Downloaded the newest TRT-LLM `linux_aarch64` wheel (`tensorrt_llm==1.1.0`, cp312) and inspected metadata: it pins CUDA 13 (`cuda-python>=13`), TensorRT `~=10.13.3`, `torch>=2.9.0a0`, and `transformers==4.56.0`, which is far beyond this Orin’s CUDA 12.6 / TensorRT 10.3 stack.
- Did public web research: JetPack 6.1/6.2/6.2.1 are still CUDA 12.6 + TensorRT 10.3; NVIDIA forums say TRT-LLM is not officially supported on Jetson, but Jetson AI Lab provides a preview TRT-LLM stack for Orin (v0.12.0-jetson branch + `dustynv/tensorrt_llm:0.12-r36.4.0`); TRT-LLM upstream has an open feature request for Qwen3-VL.
