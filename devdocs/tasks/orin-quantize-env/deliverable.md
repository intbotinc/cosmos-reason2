# Deliverable: orin-quantize-env

## Summary
- Updated `scripts/quantize.py` + `scripts/quantize.py.lock` to resolve and run on Jetson AGX Orin (aarch64, Python 3.10).
- Pinned Orin-only `compressed-tensors==0.10.2` (fixes `llmcompressor==0.3.0` import errors on newer `compressed-tensors`).
- Hardened the uv shebang for Orin quirks:
  - `UV_SKIP_WHEEL_FILENAME_CHECK=1` for NVIDIA Jetson torch wheel filename/version mismatch.
  - `--no-build-isolation-package torchvision` to prevent ABI-mismatched torchvision builds.
- Fixed Orin runtime failures during quantization:
  - Generate a safe YAML recipe string for `llmcompressor.apply()` (avoid `!!python/tuple` parse errors).
  - Move model and calibration tensors to the same device (avoid CPU/CUDA mismatches).
  - Force `attn_implementation="eager"` when Jetson SDPA lacks `enable_gqa`.
- Fixed `scripts/inference_sample.py` to load the processor with `fix_mistral_regex=True` (eliminates incorrect-tokenizer-regex warning for locally-quantized model dirs).
- Updated Orin setup docs/scripts to install quantize deps in `.venv`: `ORIN_VENV_SETUP.md` and `scripts/setup_orin_venv.sh`.

## Validation
- Command: `UV_SKIP_WHEEL_FILENAME_CHECK=1 uv run --script scripts/quantize.py --help`
  - Outcome: ok (prints CLI help; no dependency import errors).
- Command: `PY_SITE=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') && LD_LIBRARY_PATH="$PY_SITE/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}" .venv/bin/python -c 'import torch, torchvision, torchvision.extension; print(torch.__version__, torch.cuda.is_available()); print(torchvision.__version__, torchvision.extension._has_ops())'`
  - Outcome: ok (`torch` imports; `torchvision.extension._has_ops()` is `True`).
- Command: `PY_SITE=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') && LD_LIBRARY_PATH="$PY_SITE/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}" .venv/bin/python scripts/quantize.py --help`
  - Outcome: ok (prints CLI help).
- Command: `./scripts/quantize.py -o /tmp/cosmos-reason2/checkpoints-test --model nvidia/Cosmos-Reason2-2B --precision nvfp4 --kv-precision bf16 --num-samples 1 --max-sequence-length 2048`
  - Outcome: ok (quantization ran, sample generation succeeded, model saved under `/tmp/cosmos-reason2/checkpoints-test/model_nvfp4`).
- Command: `source .venv/bin/activate && python scripts/inference_sample.py --model /tmp/cosmos-reason2/checkpoints/model_nvfp4 --max-new-tokens 16 --fps 1 --runs 1 --warmup-runs 0 --no-print-output`
  - Outcome: ok (loads quantized model and generates; no tokenizer-regex warning).

## Notes / Follow-ups
- Building `torchvision` from source on Orin can take several minutes; it must be built against the installed NVIDIA Jetson torch wheel (avoid build isolation).
