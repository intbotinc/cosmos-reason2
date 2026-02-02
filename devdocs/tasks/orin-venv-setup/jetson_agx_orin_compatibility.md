# Cosmos-Reason2-2B on Jetson AGX Orin: compatibility summary

## Conclusion

**Partially supported** on Jetson AGX Orin (Ampere SM87) via **Transformers + Jetson PyTorch**.

- ✅ **Architecture:** Ampere **SM87 is supported** (when using the NVIDIA Jetson PyTorch build that includes `sm_87` kernels).
- ✅ **Software stack:** Works with the Jetson CUDA/cuDNN ecosystem on this box (details below).
- ⚠️ **Not officially validated in upstream docs:** The repo states Cosmos-Reason2 is tested on Hopper/Blackwell; other hardware may work but isn’t officially validated.
- ⚠️ **Acceleration stacks (vLLM / TRT-LLM / TensorRT engine export):** not validated on Orin in this repo; treat as unsupported/experimental.

## Validated environment (this AGX Orin box)

- **Jetson Linux / L4T:** `R36.4.0` (`/etc/nv_tegra_release`)
- **CUDA toolkit:** `12.6` (`nvcc 12.6.68`, `/usr/local/cuda/version.json` shows CUDA SDK `12.6.11`)
- **PyTorch:** `2.5.0a0+872d972e41.nv24.08` (NVIDIA Jetson wheel, CUDA `12.6`)
- **cuDNN (via torch):** `90300` (cuDNN 9.x)
- **TensorRT:** `10.3.0` (system packages, Python bindings importable)
- **GPU:** `Orin`, compute capability **(8, 7)** (Ampere **SM87**)
- **GPU memory (unified):** ~`61.37 GiB` visible to CUDA (`torch.cuda.get_device_properties(0).total_memory`)

## GPU architecture requirements

### Minimum memory

From the repo README:
- **Cosmos-Reason2-2B minimum GPU memory:** **24 GB**

On Jetson AGX Orin, GPU memory is **unified** with system memory, so headroom depends on overall system load and memory fragmentation. A 64GB Orin is strongly recommended for reliable runs.

### Compute capability / kernel support

- Jetson AGX Orin is **Ampere SM87**.
- The critical requirement is that the **PyTorch build must include SM87 kernels**.

Observed on this box:
- `torch.cuda.get_device_capability(0) == (8, 7)`
- `torch.cuda.get_arch_list()` includes `sm_87` and `compute_87`

Practical implication:
- ✅ **Supported** with the NVIDIA Jetson PyTorch wheel we installed.
- ❌ **Not supported** with many generic CUDA wheels that only include `sm_80/sm_90` (those fail on Orin with “no kernel image is available…”).

## Software stack compatibility (JetPack/CUDA/cuDNN/PyTorch/TensorRT)

### JetPack / Jetson Linux

- This box is on **L4T R36.4.0** (the r36.4 Jetson software stack with CUDA 12.6).
- The PyTorch wheel used in setup comes from the NVIDIA JetPack redistribution index and is compatible with this stack in practice (torch imports and CUDA works).

### CUDA + cuDNN + PyTorch

Validated working for Transformers inference prerequisites:
- `import torch` works
- `torch.cuda.is_available()` is `True`
- `torch.version.cuda == "12.6"`
- `torch.backends.cudnn.version() == 90300`

### TensorRT

Validated on this box:
- TensorRT is installed and the Python bindings import (`tensorrt 10.3.0`).

Important limitation:
- The provided `scripts/inference_sample.py` runs via **Transformers + PyTorch**, not TensorRT.
- This repo does not provide a supported TensorRT export/inference path for Cosmos-Reason2-2B on Jetson; treat TensorRT execution as **unsupported/experimental** unless a dedicated TRT-LLM workflow is provided and validated separately.

## Unsupported / Hopper-or-Blackwell-only (or “not validated on Orin”) items

These don’t prevent Transformers inference, but they affect what is “supported” vs “best effort”:

- **FP8 inference paths:** Orin (Ampere) does not provide FP8 Tensor Cores like Hopper/Blackwell. FP8-only acceleration is **not available**.
- **Hopper/Blackwell-optimized kernels (performance):** Any Hopper/Blackwell-only attention/quantization optimizations are **not available**; expect reduced throughput on Orin.
- **vLLM deployment path:** repo recommends vLLM for serving; vLLM on Jetson/Orin is **not validated** here.
- **TensorRT engine deployment (TRT-LLM):** not provided/validated for this model in this repo.

## Required vs incompatible features (summary)

### Required for “it runs”

- Jetson-compatible **PyTorch build including `sm_87`** kernels
- CUDA stack compatible with that wheel (this box: CUDA 12.6)
- cuSPARSELt runtime (`libcusparseLt.so.0`) discoverable by the loader (handled by setup + script auto-reexec)
- `transformers>=4.57` (this box: `4.57.3`)
- Sufficient memory (24GB+; 64GB Orin recommended)
- Hugging Face access + auth for the gated model repo (`HF_TOKEN` or `huggingface-cli login`)

### Incompatible / not supported

- Using a generic PyTorch CUDA wheel without `sm_87` support (will fail at runtime on Orin)
- Expecting FP8-only inference/quantization acceleration (Hopper/Blackwell-specific)
- Expecting an officially supported TensorRT/vLLM production path for Orin (not validated here)

## Recommended versions (based on what worked here)

- **Jetson Linux / L4T:** `R36.4.x` (CUDA 12.6 stack)
- **Python:** 3.10
- **PyTorch:** NVIDIA Jetson wheel `torch 2.5.0a0+872d972e41.nv24.08` (CUDA 12.6)
- **Torchvision:** build from source against that torch (v0.20.1) with `--no-build-isolation`
- **Transformers:** `4.57.3`
- **TensorRT:** `10.3.x` present (not used by the sample, but available)
