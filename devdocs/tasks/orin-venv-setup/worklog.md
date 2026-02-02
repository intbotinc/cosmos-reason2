# Worklog: orin-venv-setup

## 2026-01-30 (UTC) — codex

- Initialized task docs.
- Ran `scripts/setup_orin_venv.sh` to create `.venv` (Jetson torch wheel + cuSPARSELt + deps).
- Found issues: cuSPARSELt not on `LD_LIBRARY_PATH`, torchvision patch indentation bug, and torchvision build-isolation linking against wrong torch.
- Fixed `scripts/setup_orin_venv.sh`: install `setuptools/wheel`, export cuSPARSELt `LD_LIBRARY_PATH`, build torchvision with `--no-build-isolation`, robust `_meta_registrations.py` patch + import sanity check.
- Updated `scripts/inference_sample.py` to auto-reexec for cuSPARSELt, use `eager` attention when `enable_gqa` unsupported, and print a clear gated-model message.
- Updated `ORIN_VENV_SETUP.md` with new issues (HF gated repo, torchvision build isolation, patch pitfalls).
- Wrote Jira-ready platform compatibility summary in `devdocs/tasks/orin-venv-setup/jetson_agx_orin_compatibility.md`.
