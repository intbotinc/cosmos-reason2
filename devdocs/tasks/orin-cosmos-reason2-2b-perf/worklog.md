# Worklog: orin-cosmos-reason2-2b-perf

## 2026-02-01 (UTC) — codex

- Initialized task docs.
- Confirmed no existing benchmark harness; `scripts/inference_sample.py` already has per-step timing logs.
- Planned `inference_sample.py` extensions: argparse knobs, multi-run support, JSONL metrics, per-run `tegrastats` capture.
- Implemented benchmark mode in `scripts/inference_sample.py` (CLI args, multi-run, JSONL metrics, per-run tegrastats parsing).
- Ran 2-run benchmarks for two input configs (fps=2/maxV=4096 and fps=4/maxV=8192); archived JSONL + tegrastats logs under `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/`.
- Captured Orin environment snapshot; `torch` imports require cuSPARSELt in `LD_LIBRARY_PATH`, and `jetson_clocks --show` requires root.
- Wrote a consolidated usage + results summary: `devdocs/tasks/orin-cosmos-reason2-2b-perf/README.md`.

## 2026-02-02 (UTC) — codex

- Verified background GPU load was coming from `robot-perception` (DeepStream) and rechecked GPU idle (`GR3D_FREQ ~0%`) after it was stopped.
- Backed up prior raw data before rerun: `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw_backup_20260202T133411Z_pre_isolated/`.
- Re-ran both benchmark configs and updated the report + README with the isolated-GPU results.
