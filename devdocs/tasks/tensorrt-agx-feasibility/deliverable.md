# Deliverable: tensorrt-agx-feasibility

## Summary
- Cosmos-Reason2 in this repo is a Hugging Face **Qwen3-VL** based VLM (see `README.md`) and the provided inference paths are **Transformers** (`scripts/inference_sample.py`) and **vLLM** (`cosmos-reason2-inference`) — there is **no** ONNX/TensorRT export pipeline shipped here.
- Converting Cosmos-Reason2 to a “TensorRT model” is not a straight ONNX→TensorRT step: autoregressive generation + KV-cache typically requires **TensorRT-LLM** (or a custom multi-engine runtime).
- Text-only is feasible today on Jetson AGX Orin via **PyTorch/Transformers** (verified with a text-only prompt using `transformers==4.57.3` + Jetson torch; SDPA requires an `attn_implementation="eager"` fallback on this box).
- For TensorRT(-LLM): current TensorRT-LLM docs list **Qwen3 (text)** support, but do **not** list **Qwen3-VL**; multimodal coverage is listed for **Qwen2-VL / Qwen2.5-VL** instead. This makes Cosmos-Reason2 TensorRT(-LLM) conversion (especially image/video) **unsupported / high-effort** today.
- On Jetson AGX Orin specifically, TensorRT is present (`10.3.0`), but TensorRT-LLM is not installed by default; getting TRT-LLM working on Jetson typically requires a Jetson-specific build/container and may lag upstream model support.

## TRT-LLM availability on this Jetson AGX Orin (64GB)

Practical blockers for “just install TRT-LLM and convert” on this box:

- **No apt packages:** There is no `tensorrt-llm`/`trtllm` package in the Jetson apt repos (only TensorRT itself).
- **Pip wheels don’t match Tegra constraints:** NVIDIA’s pip index has `tensorrt_llm` `linux_aarch64` wheels, but:
  - The only **Python 3.10** `linux_aarch64` wheels are `tensorrt_llm==0.15.0` and a couple of `0.16.0.dev*` builds.
  - Those wheels pin **`tensorrt~=10.6.0`** and **`transformers<=4.45.1`** (Cosmos-Reason2’s Qwen3-VL requires newer Transformers).
  - Those wheels include **Qwen/Qwen2** converters, but not **Qwen3** / **Qwen3-VL**.
- **New TRT-LLM requires CUDA 13 + TensorRT 10.13:** `tensorrt_llm==1.1.0` (the newest `linux_aarch64` wheel) pins **`cuda-python>=13`** and **`tensorrt~=10.13.3`**, which is incompatible with this Orin’s CUDA `12.6` + TensorRT `10.3` stack.
- **Pip TensorRT is explicitly not supported on Jetson/Tegra:** attempting to fetch `tensorrt-cu12` from NVIDIA’s pip index errors with “TensorRT does not currently build wheels for Tegra systems”, so TRT-LLM’s pinned `tensorrt~=10.6.0` dependency stack is not Jetson-friendly.

## Public research (as of 2026-02-01)

### 1) JetPack on Orin is still CUDA 12.6 + TensorRT 10.3

Jetson AGX Orin is on JetPack 6.x today. JetPack 6.1, 6.2, and 6.2.1 all state they ship CUDA 12.6 and TensorRT 10.3:

- JetPack 6.1 release notes: https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-61/release-notes/index.html
- JetPack 6.2 release notes: https://docs.nvidia.com/jetson/jetpack/6.2/release-notes/index.html
- JetPack 6.2.1 release notes: https://docs.nvidia.com/jetson/jetpack/release-notes/index.html

Practical implication: mainline TRT-LLM releases that depend on newer TensorRT/CUDA stacks cannot be “just installed” on Orin without a matching JetPack upgrade.

### 2) NVIDIA’s public stance: TRT-LLM on Jetson is not “officially supported”

NVIDIA forum responses consistently warn that TensorRT-LLM is not officially supported on Jetson:

- 2024-07-15: “TensorRT-LLM does not officially support Jetson currently.” (Jetson AGX Orin forum thread)  
  https://forums.developer.nvidia.com/t/can-i-use-tensorrt-llm-in-jetson-agx-orin/296059

- 2025-11-12: “we don’t have a concrete plan to support TensorRT-LLM on Jetson.” and recommends alternative frameworks for Orin.  
  https://forums.developer.nvidia.com/t/orin-nano-building-tensorrt-llm-from-source/350532

### 3) “TRT-LLM for Jetson Orin” exists as a preview via Jetson AI Lab + containers (v0.12 jetson branch)

There is public NVIDIA/Jetson AI Lab guidance for running TRT-LLM on Jetson AGX Orin with JetPack 6.1, via a patched `v0.12.0-jetson` branch and prebuilt container images:

- Jetson AI Lab page: “TensorRT-LLM for Jetson” (includes `dustynv/tensorrt_llm:0.12-r36.4.0` usage)  
  https://www.jetson-ai-lab.com/tensorrt_llm
- Forum announcement by `dusty_nv`: “TensorRT-LLM for Jetson” (same branch + guides)  
  https://forums.developer.nvidia.com/t/tensorrt-llm-for-jetson/313228
- Docker Hub container tag referenced by the guide:  
  https://hub.docker.com/r/dustynv/tensorrt_llm

So, **yes**: “TensorRT-LLM / TensorRT engine export” can be achieved on AGX Orin in the sense that you can build and run TRT-LLM engines for supported model families using that Jetson-specific preview stack.

### 4) But Cosmos-Reason2 (Qwen3-VL) is not supported by TRT-LLM engine conversion today

Public upstream indicators of the model-support gap:

- Qwen3 “engine-flow” (TensorRT engine conversion) has historically lagged; users request Qwen3 engine conversion support.  
  https://github.com/NVIDIA/TensorRT-LLM/issues/5450
- Qwen3-VL support is explicitly requested as a feature and is not generally available in TRT-LLM.  
  https://github.com/NVIDIA/TensorRT-LLM/issues/8722

Combined with the Jetson constraint (JetPack 6.x / TRT 10.3), this means: **even if TRT-LLM runs on Orin, Cosmos-Reason2 (Qwen3-VL) is not a drop-in conversion target**.

### 5) NVIDIA direction: TensorRT Edge-LLM is positioned for next-gen Jetson platforms

NVIDIA’s newer “TensorRT Edge-LLM” runtime is documented as supporting Qwen3-VL families and lists Jetson Thor (JetPack 7.1) as the officially supported Jetson platform:

- TensorRT Edge-LLM overview (supported platforms + model families):  
  https://nvidia.github.io/TensorRT-Edge-LLM/0.4.0/developer_guide/01.1_Overview.html

This suggests future embedded TensorRT paths for Qwen3-VL are more likely to land via **Edge-LLM** than via backporting TRT-LLM onto Orin.

## Validation
- Command: `rg -n "tensorrt|trtexec|TRT-LLM|onnx|torch_tensorrt" -S`
  - Outcome: no TensorRT export/inference code found in this repo (only schema mentions and prior Jetson notes).
- Command: `find . -maxdepth 4 -type f \( -name '*.onnx' -o -name '*.engine' -o -name '*.plan' \)`
  - Outcome: no ONNX/TensorRT engine artifacts in-repo.
- Command: `python3 -c "import json, os; p=os.path.expanduser('~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/981d433a76b30f0692e3fc07ac7ed787d9e7a0da/config.json'); print(json.load(open(p))['architectures'])"`
  - Outcome: `['Qwen3VLForConditionalGeneration']` (model is Qwen3-VL; language model config is `qwen3_vl_text`).
- Command: `python3 -m pip download --no-deps --extra-index-url https://pypi.nvidia.com tensorrt-cu12==10.6.0 -d /tmp/...`
  - Outcome: fails on Jetson with “TensorRT does not currently build wheels for Tegra systems”.
- Command: `python3 -m pip download --no-deps --extra-index-url https://pypi.nvidia.com tensorrt_llm==0.15.0 -d /tmp/...`
  - Outcome: downloads `tensorrt_llm-0.15.0-cp310-cp310-linux_aarch64.whl`; wheel metadata pins `tensorrt~=10.6.0` and `transformers<=4.45.1`.

## Notes / Follow-ups
- For Orin 64GB, start with text-only requirements: if “TensorRT required” is non-negotiable, plan on using **TensorRT-LLM** and expect engineering work to (a) obtain a Jetson-compatible TRT-LLM build, and (b) adapt conversion for the `qwen3_vl_text` language model weights.
- For image/video inputs, expect additional work: Qwen3-VL is not listed as supported by TRT-LLM, so end-to-end VLM TensorRT inference is likely blocked until Qwen3-VL support lands or you switch architectures.
- Practical conclusion for Jetson AGX Orin **today**: end-to-end TensorRT(-LLM) engine export for Cosmos-Reason2 is **not achievable as an off-the-shelf workflow**. It would require (at minimum) a JetPack stack that provides newer TensorRT (10.6+ and likely 10.13+) and/or a substantial porting effort to add Qwen3-VL support and build TRT-LLM against Jetson’s TensorRT.
