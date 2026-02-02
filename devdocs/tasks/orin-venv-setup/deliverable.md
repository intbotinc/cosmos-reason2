# Deliverable: orin-venv-setup

## Summary
- Ran the Orin venv setup flow and fixed remaining Jetson-specific issues so `python scripts/inference_sample.py` runs without torch/torchvision import errors.
- Updated `scripts/setup_orin_venv.sh` to correctly build torchvision against Jetson torch (`--no-build-isolation`), set cuSPARSELt `LD_LIBRARY_PATH` for build-time imports, and apply a safe `_meta_registrations.py` patch.
- Updated `scripts/inference_sample.py` to auto-reexec when cuSPARSELt is missing from `LD_LIBRARY_PATH`, fall back to `attn_implementation="eager"` when `enable_gqa` is unsupported, and print a clear message when the HF repo is gated.
- Updated `ORIN_VENV_SETUP.md` with the new issues/fixes and the corrected torchvision build instructions.
- Added a Jira-ready platform compatibility assessment for Jetson AGX Orin: `devdocs/tasks/orin-venv-setup/jetson_agx_orin_compatibility.md`.

## Validation
- Command: `source .venv/bin/activate && export LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}" && python -c "import torch, torchvision, torchvision.extension; print(torch.__version__, torch.cuda.is_available()); print(torchvision.__version__, torchvision.extension._has_ops())"`
  - Outcome: ok (torch imports, CUDA available, torchvision ops available)
- Command: `source .venv/bin/activate && python scripts/inference_sample.py`
  - Outcome: fails cleanly with a gated-repo message (expected until HF auth is configured)

## Notes / Follow-ups
- To fully run inference, request access to `nvidia/Cosmos-Reason2-2B` on Hugging Face and authenticate (`huggingface-cli login` or set `HF_TOKEN`).
- cuSPARSELt is installed via pip but is not automatically added to your shell environment; `scripts/inference_sample.py` handles it, but other scripts may need the `LD_LIBRARY_PATH` export shown above.
