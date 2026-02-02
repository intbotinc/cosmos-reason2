# Task Plan: orin-venv-setup

## Goal
Create a reproducible `.venv` in this repo on Jetson AGX Orin so `python scripts/inference_sample.py` runs successfully.

## Phases
- [x] Phase 1: Understand + plan
- [x] Phase 2: Read docs + code
- [x] Phase 3: Create venv + install deps
- [x] Phase 4: Run inference sample + fix issues
- [x] Phase 5: Update docs/scripts + deliverable

## Key Questions
1. Is this host aarch64 + Jetson (CUDA available) and using Python 3.10?
2. Can we fetch PyTorch/torchvision + model assets over the network from this box?
3. Do we need additional system packages (e.g., build deps) to build `torchvision` from source?

## Decisions
- Use Python 3.10 venv at `.venv`: required for NVIDIA Jetson PyTorch wheels (cp310).
- Install NVIDIA Jetson PyTorch wheel + cuSPARSELt via pip: avoids CUDA arch mismatch on Orin (sm_87).
- Build `torchvision` from source and patch `_meta_registrations.py` in venv: avoids missing-ops runtime crash.

## Errors Encountered
- 2026-01-30 `ImportError: libcusparseLt.so.0` → set `LD_LIBRARY_PATH` (and make `scripts/inference_sample.py` auto-reexec).
- 2026-01-30 `torchvision/_meta_registrations.py` bad indent from patch → fix patch logic to preserve valid Python indentation.
- 2026-01-30 `torchvision/image.so: undefined symbol ...` → rebuild torchvision with `--no-build-isolation` against Jetson torch.
- 2026-01-30 HF gated repo (`401 Unauthorized`) → require HF access + `huggingface-cli login` / `HF_TOKEN`.

## Status
**CURRENT:** Complete — venv ready; HF auth needed to fully run model
