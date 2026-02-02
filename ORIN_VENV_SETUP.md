# Cosmos-Reason2 venv setup on Jetson AGX Orin (aarch64)

This document summarizes the exact steps and fixes used to get `scripts/inference_sample.py` running successfully on Jetson AGX Orin. It includes the setup procedure and the issues encountered with their resolutions so you can reproduce the environment on another Orin box.

## Summary of repo changes

Changes visible in git:
- Modified: `scripts/inference_sample.py`
- Modified: `scripts/setup_orin_venv.sh`

Note: There was also a necessary patch to a **venv-installed** file:
- Patched: `.venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py`
  - This is inside the venv and **not** tracked by git.

## Why the original venv failed

- The initial venv used Python 3.12 and a CUDA wheel of PyTorch `2.10.0+cu126` built only for `sm_80`/`sm_90`.
- Jetson Orin is `sm_87`, so kernels failed with:
  - `CUDA error: no kernel image is available for execution on the device`

## Working setup procedure (clean, reproducible)

### 1) Recreate venv with Python 3.10
The NVIDIA Jetson PyTorch wheels are built for Python 3.10 (cp310). Recreate the venv to match:

```bash
mv .venv .venv.bak-<timestamp>
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install Jetson-compatible PyTorch
Install the CUDA-enabled Jetson wheel directly (avoid version-metadata mismatch):

```bash
source .venv/bin/activate
python -m pip install --no-cache-dir --no-deps \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0%2B872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

### 3) Install cuSPARSELt library
PyTorch requires `libcusparseLt.so.0` on Jetson. Install via pip:

```bash
source .venv/bin/activate
python -m pip install --no-cache-dir nvidia-cusparselt-cu12
```

Ensure the dynamic loader can find cuSPARSELt (needed for `import torch`, and for building torchvision from source):

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
```

### 4) Install the runtime Python deps
```bash
source .venv/bin/activate
python -m pip install --no-cache-dir \
  accelerate==1.12.0 \
  av==16.1.0 \
  pillow==12.0.0 \
  transformers==4.57.3
```

### 5) Downgrade NumPy to 1.x
The Jetson torch wheel is not compatible with NumPy 2.x:

```bash
source .venv/bin/activate
python -m pip install --no-cache-dir "numpy<2"
```

### 6) Build torchvision from source (compatible with Jetson torch)
Binary wheels of torchvision try to pull a different PyTorch build and also fail at runtime. Build from source against the installed Jetson torch:

```bash
source .venv/bin/activate
python -m pip uninstall -y torchvision
python -m pip install --no-cache-dir --no-deps \
  --no-build-isolation \
  "git+https://github.com/pytorch/vision.git@v0.20.1"
```

### 7) Patch torchvision meta registrations
`torchvision` tries to register a fake `nms` op, which fails if ops are not available on Jetson. Patch the file in the venv:

File:
```
.venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py
```

Change:
- Wrap the `@torch.library.register_fake("torchvision::nms")` in a guard so it only registers when ops exist.

Applied patch logic:
```python
if torchvision.extension._has_ops():
    @torch.library.register_fake("torchvision::nms")
    def meta_nms(...):
        ...
else:
    def meta_nms(...):
        return None
```

### 8) Script changes in repo (permanent)
`scripts/inference_sample.py` was updated to:
- Restart itself if `LD_LIBRARY_PATH` needs to include the cuSPARSELt path (so `import torch` works on Jetson).
- Select attention implementation safely (use `eager` when SDPA does not support `enable_gqa`).
- Keep CUDA-only requirement (exit if CUDA is not available).
- Print a clear message if the Hugging Face model repo is gated (requires login/token).

## Issues encountered and fixes

1) **CUDA kernel image error (sm_87 unsupported)**
   - Error: `CUDA error: no kernel image is available for execution on the device`
   - Cause: PyTorch wheel built only for `sm_80/sm_90` (not Orin).
   - Fix: Install Jetson-specific PyTorch wheel built for Orin (sm_87).

2) **Missing `libcusparseLt.so.0`**
   - Error: `ImportError: libcusparseLt.so.0: cannot open shared object file`
   - Fix: Install `nvidia-cusparselt-cu12` and ensure `LD_LIBRARY_PATH` includes the wheel’s `lib` directory.

3) **NumPy 2.x incompatibility**
   - Error: runtime warning + failure on torch import about NumPy 1.x build expectations.
   - Fix: downgrade to `numpy<2`.

4) **torchvision mismatch / wrong build isolation / missing ops**
   - Errors:
     - `RuntimeError: operator torchvision::nms does not exist`
     - `Failed to load image Python extension: ... torchvision/image.so: undefined symbol ...`
   - Causes:
     - torchvision wheel mismatched to the Jetson torch build; missing compiled ops.
     - Building torchvision **with build isolation** can silently compile/link against a different (non-Jetson) torch.
   - Fix:
     - Build torchvision from source against the **installed Jetson torch** using `--no-build-isolation` (ensure `wheel` is installed, and `LD_LIBRARY_PATH` includes the cuSPARSELt `lib` dir so `import torch` works during the build).
     - Patch `_meta_registrations.py` to guard the `nms` fake registration.

5) **SDPA API mismatch**
   - Error: `scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'`
   - Cause: Jetson torch build doesn’t support the newer SDPA API.
   - Fix: fall back to `attn_implementation="eager"` when `enable_gqa` isn’t supported.

6) **Hugging Face gated model access**
   - Error: `You are trying to access a gated repo` / `401 Client Error: Unauthorized`
   - Cause: `nvidia/Cosmos-Reason2-2B` requires Hugging Face access + authentication.
   - Fix: request access on Hugging Face, then authenticate via `huggingface-cli login` or set `HF_TOKEN`.

7) **Bad torchvision patch (syntax error)**
   - Error: importing torchvision fails with a Python `SyntaxError` in `_meta_registrations.py`.
   - Cause: patching `_meta_registrations.py` with incorrect indentation around the `meta_nms` function.
   - Fix: ensure the `def meta_nms(...)` (and its body) are properly indented under the `if torchvision.extension._has_ops():` guard.

## Final run command

```bash
source .venv/bin/activate
python scripts/inference_sample.py
```

## Notes

- This setup is specific to Jetson AGX Orin (aarch64 / sm_87).
- The venv is Python 3.10, because NVIDIA’s Jetson wheels are built for cp310.
- The torchvision patch is a local venv change; if you recreate the venv, reapply that patch.
