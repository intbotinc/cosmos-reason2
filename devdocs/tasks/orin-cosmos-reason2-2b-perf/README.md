# Cosmos-Reason2-2B Orin Benchmark (via `scripts/inference_sample.py`)

This benchmark measures inference performance of `nvidia/Cosmos-Reason2-2B` on Jetson AGX Orin under a representative
video-captioning workload, focusing on:

- Latency (cold start + steady state)
- Throughput (tokens/sec for `generate()`)
- System utilization (GPU/CPU/RAM via `tegrastats`)

## What was implemented

`scripts/inference_sample.py` was extended with benchmark-friendly options:

- Multi-run execution: `--warmup-runs`, `--runs`
- Structured output: `--jsonl-out` (one JSON object per run)
- Utilization capture: `--tegrastats-dir` (per-run logs) + parsed summaries embedded in JSONL
- Workload knobs: `--fps`, `--max-vision-tokens`, `--max-new-tokens`, `--video`
- Benchmark hygiene: `--no-print-output` to avoid stdout skewing timings

## Metrics captured (in JSONL)

Each JSONL line includes:

- `setup_timings_s`: seed + model load + setup steps (one-time per process)
- `run_timings_s`: per-run `preprocess inputs`, `generate`, `decode` timings (CUDA-synchronized)
- `generated_tokens`, `tokens_per_sec`, `ms_per_token`
- `inputs_summary`: key tensor shapes (e.g., `input_ids_shape`, `pixel_values_videos`)
- `tegrastats_log` and `tegrastats_summary` (avg/max GPU%, CPU%, RAM MB, etc.)
- CUDA peak memory: `cuda_peak_allocated_bytes`, `cuda_peak_reserved_bytes`

Notes:

- `process_elapsed_s` is time since process start at the end of each run (useful for cold-start; use `run_timings_s` for
  steady-state comparisons).

## Usage

### Activate the venv

```bash
cd /agibot/data/home/agi/cosmos/cosmos-reason2
source .venv/bin/activate
```

If you run ad-hoc `python -c "import torch"` snippets, you may also need cuSPARSELt in `LD_LIBRARY_PATH` on Jetson:

```bash
export LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
```

(`scripts/inference_sample.py` can auto-reexec itself to fix this; other Python entrypoints may not.)

### Show options

```bash
python scripts/inference_sample.py --help
```

### Run a benchmark (two representative configs)

All outputs below are written into the task’s raw folder (JSONL + tegrastats logs):

```bash
mkdir -p devdocs/tasks/orin-cosmos-reason2-2b-perf/raw

# Config A: smaller context (lower FPS + smaller maxV)
python scripts/inference_sample.py \
  --name orin_fps2_maxv4096_isolated \
  --fps 2 \
  --max-vision-tokens 4096 \
  --runs 2 \
  --no-print-output \
  --jsonl-out devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/bench_fps2_maxv4096.jsonl \
  --tegrastats-dir devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/tegrastats_fps2_maxv4096

# Config B: larger context (baseline: higher FPS + larger maxV)
python scripts/inference_sample.py \
  --name orin_fps4_maxv8192_isolated \
  --fps 4 \
  --max-vision-tokens 8192 \
  --runs 2 \
  --no-print-output \
  --jsonl-out devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/bench_fps4_maxv8192.jsonl \
  --tegrastats-dir devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/tegrastats_fps4_maxv8192
```

Guidance:

- Use `--warmup-runs 1` if you want to exclude the first run from steady-state stats without restarting the process.
- Keep `--no-print-output` on for benchmarking to reduce noise (printing very long generations can distort wall time).

## Benchmark results (2026-02-02)

Definitions:

- **Cold-start latency**: `process_elapsed_s` at end of run 1 (includes model load).
- **Steady-state latency**: `preprocess + generate + decode` time for run 2 (model already loaded).
- **Throughput**: `generated_tokens / generate_time_s` for run 2.

| config | input_ids | pv_tokens | cold_e2e_run1_s | steady_run2_s | steady_tok/s_run2 | gpu%avg_run2 | cpu%avg_run2 | ram_gb_avg_run2 | cuda_peak_gb_run2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| orin_fps2_maxv4096_isolated (fps=2, maxV=4096) | 1777 | 6912 | 48.93 | 42.39 | 7.26 | 88.10 | 29.51 | 17.48 | 4.81 |
| orin_fps4_maxv8192_isolated (fps=4, maxV=8192) | 2945 | 11520 | 60.28 | 53.76 | 5.35 | 95.72 | 28.13 | 18.35 | 5.52 |

Notes:

- `input_ids` is `inputs.input_ids.shape[-1]`.
- `pv_tokens` is the first dimension of `pixel_values_videos` (proxy for video visual tokens after preprocessing).
- Generated token counts were ~244–306 tokens in these runs; tok/s varies with output length.

## Primary bottlenecks (from the measurements)

- `generate` dominates runtime (36–53s) and GPU utilization is high (avg ~88–96%) → **GPU-bound generation** under these settings.
- Cold-start adds meaningful overhead (model+processor load ~8.5s) → first inference is dominated by load/init + generate.
- Preprocess is small (<1s) and decode is ~1ms → not primary bottlenecks for this workload.
- Note: these numbers were collected after stopping a background DeepStream workload (`robot-perception`), with idle
  `GR3D_FREQ` observed at ~0% before the benchmark. A backup of earlier raw data exists at
  `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw_backup_20260202T133411Z_pre_isolated/`.

## Raw data locations

- Environment snapshot: `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/env.txt`
- JSONL per-run metrics:
  - `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/bench_fps2_maxv4096.jsonl`
  - `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/bench_fps4_maxv8192.jsonl`
- tegrastats logs:
  - `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/tegrastats_fps2_maxv4096/`
  - `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/tegrastats_fps4_maxv8192/`
