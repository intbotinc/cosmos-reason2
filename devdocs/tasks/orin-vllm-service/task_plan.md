# Task Plan: orin-vllm-service

## Goal
Have a reproducible Jetson AGX Orin `.venv` setup (via `scripts/setup_orin_venv.sh`) that installs vLLM and provides a runnable vLLM OpenAI-compatible service serving `/tmp/cosmos-reason2/checkpoints/model_nvfp4`.

## Phases
- [x] Phase 1: Understand + plan
- [ ] Phase 2: Investigate / gather context
- [ ] Phase 3: Implement
- [ ] Phase 4: Validate
- [ ] Phase 5: Wrap up (deliverable)

## Key Questions
1. Which vLLM version/build path works on Jetson Orin (aarch64, CUDA 12.6, NVIDIA torch 2.5.x)?
2. Can vLLM load the local checkpoint dir `/tmp/cosmos-reason2/checkpoints/model_nvfp4` (Qwen3-VL) and start the OpenAI server?
3. What environment variables are required for build/runtime (`LD_LIBRARY_PATH` cuSPARSELt, `TRITON_PTXAS_PATH`, etc.)?

## Decisions
- Install vLLM into the Orin `.venv` created by `scripts/setup_orin_venv.sh`: keeps the Orin setup self-contained and reproducible.
- Provide a repo script + optional systemd unit template for the server: avoids writing into `/etc` from the repo while making service setup copy/pasteable.

## Errors Encountered
- (none yet)

## Status
**CURRENT:** Phase 2 — Gather context (existing Orin venv constraints, vLLM install feasibility on aarch64/CUDA 12.6).
