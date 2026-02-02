# Orin venv setup: issues encountered + fixes

Context (this box)
- Arch/OS: Jetson-class `aarch64`, Ubuntu 22.04, Python 3.10
- Goal: get `python scripts/inference_sample.py` running in a local `.venv/`

## 1) `ImportError: libcusparseLt.so.0` when importing torch

**Symptom**
- `ImportError: libcusparseLt.so.0: cannot open shared object file: No such file or directory`
- Seen immediately when running `python scripts/inference_sample.py` (torch import).

**Root cause**
- The Jetson PyTorch wheel requires cuSPARSELt (`libcusparseLt.so.0`), installed via pip, but the dynamic loader does not automatically search the wheel’s `.../site-packages/nvidia/cusparselt/lib` directory.

**Fix**
- Installed `nvidia-cusparselt-cu12`.
- Ensured `LD_LIBRARY_PATH` includes the cuSPARSELt lib directory during setup (for build-time torch imports).
- Updated `scripts/inference_sample.py` to auto-`execve()` itself with the correct `LD_LIBRARY_PATH` before importing torch.

Files touched
- `scripts/setup_orin_venv.sh`
- `scripts/inference_sample.py`

## 2) Torchvision build produced `undefined symbol ...` in `torchvision/image.so`

**Symptom**
- Warning/error on import:
  - `Failed to load image Python extension: ... torchvision/image.so: undefined symbol: ...`
- `torchvision.extension._has_ops()` reported `False`.

**Root cause**
- Building torchvision from source with default build isolation can silently compile/link against a different torch than the Jetson wheel, producing ABI/C++ symbol mismatches at runtime.

**Fix**
- Rebuilt torchvision from source with `--no-build-isolation` so it builds against the already-installed Jetson torch.
- Ensured `setuptools` + `wheel` are present in the venv for non-isolated builds.
- Exported cuSPARSELt `LD_LIBRARY_PATH` so torch can import during the build step.

Files touched
- `scripts/setup_orin_venv.sh`

## 3) Torchvision `_meta_registrations.py` patch could break with bad indentation

**Symptom**
- Python `SyntaxError` / broken indentation around `meta_nms` after patching.

**Root cause**
- The original patching approach was a simple string replace; it could insert the `if torchvision.extension._has_ops():` guard without correctly indenting the following `def meta_nms(...)` block.

**Fix**
- Rewrote the patch step in `scripts/setup_orin_venv.sh` to replace the whole `nms` meta-registration block with a known-good guarded version.
- This patch is applied to the venv-installed file:
  - `.venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py`

Files touched
- `scripts/setup_orin_venv.sh`

## 4) SDPA API mismatch (`enable_gqa` unsupported on Jetson torch build)

**Symptom**
- `scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'`

**Root cause**
- The installed Jetson torch build does not support the newer SDPA `enable_gqa` argument.

**Fix**
- Updated `scripts/inference_sample.py` to probe SDPA support at runtime and fall back to `attn_implementation="eager"` when `enable_gqa` is not supported.

Files touched
- `scripts/inference_sample.py`

## 5) Hugging Face gated repo / 401 unauthorized for `nvidia/Cosmos-Reason2-2B`

**Symptom**
- `You are trying to access a gated repo` and/or `401 Client Error: Unauthorized`

**Root cause**
- The model repo requires Hugging Face access approval + authentication.

**Fix**
- Updated `scripts/inference_sample.py` to catch this case and exit with a clear instruction:
  - request access on Hugging Face
  - then run `huggingface-cli login` or set `HF_TOKEN`

Files touched
- `scripts/inference_sample.py`

## Where the final procedure is documented

- `ORIN_VENV_SETUP.md` (full step-by-step procedure + issues/fixes)
- `scripts/setup_orin_venv.sh` (automation)
