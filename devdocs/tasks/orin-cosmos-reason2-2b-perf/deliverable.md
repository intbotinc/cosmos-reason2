# Deliverable: orin-cosmos-reason2-2b-perf

## Summary
- Extended `scripts/inference_sample.py` with benchmark-friendly CLI options (multi-run, JSONL metrics, per-run
  `tegrastats` capture + parsing) while keeping the default single-run behavior.
- Re-ran benchmarks after stopping a background DeepStream process (`robot-perception`) to reduce external GPU load;
  previous raw data is backed up (see **Raw Data**).
- Measured Cosmos-Reason2-2B inference performance on Jetson AGX Orin for two input/context-size configurations and
  archived raw results (JSONL + tegrastats logs) alongside an environment snapshot.

## Environment
- Device: Jetson AGX Orin (aarch64), kernel `5.15.148-tegra`
- OS / JetPack: Ubuntu `22.04.4`, L4T `R36.4.0` (see `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/env.txt`)
- Power mode: `nvpmodel` reports `MODE_30W`
- Software: torch `2.5.0a0+872d972e41.nv24.08`, transformers `4.57.3`, CUDA `12.6`
- Note: `jetson_clocks --show` requires root on this system, so clocks may vary between runs.

## Results
Definitions used in this report:
- **Cold-start latency**: `process_elapsed_s` at the end of **run 1** (process start → end of first inference run; includes
  model load).
- **Steady-state latency**: `preprocess + generate + decode` time for **run 2** (model already loaded in the same process).
- **Throughput**: `tokens_per_sec = generated_tokens / generate_time_s` (effective throughput for the full `generate()` call).

| config | input_ids | pv_tokens | cold_e2e_run1_s | steady_run2_s | steady_tok/s_run2 | gpu%avg_run2 | cpu%avg_run2 | ram_gb_avg_run2 | cuda_peak_gb_run2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| orin_fps2_maxv4096_isolated (fps=2, maxV=4096) | 1777 | 6912 | 48.93 | 42.39 | 7.26 | 88.10 | 29.51 | 17.48 | 4.81 |
| orin_fps4_maxv8192_isolated (fps=4, maxV=8192) | 2945 | 11520 | 60.28 | 53.76 | 5.35 | 95.72 | 28.13 | 18.35 | 5.52 |

Notes:
- `input_ids` is `inputs.input_ids.shape[-1]` from the processor output.
- `pv_tokens` is the first dimension of `pixel_values_videos` (a proxy for video visual tokens after preprocessing).
- Generated token counts were ~244–306 tokens in these runs, so tok/s is sensitive to output-length variance.

## Bottlenecks (Primary Findings)
- **Dominant cost is generation (`generate`)**: 42–53s per run (steady-state), while tegrastats shows high GPU utilization
  (avg ~88–96% in run 2) → inference is **primarily GPU-bound** under these settings.
- **Cold-start overhead is meaningful**: model+processor load is ~8.5s in this run, and remains a major contributor to the
  first-run latency.
- **Preprocess and decode are minor** in this workload: preprocess ~0.35–0.54s, decode ~1ms; CPU utilization is moderate
  (avg ~28–30% in run 2) and RAM usage is stable (~17–19GB used).

## Raw Data (Archived)
- Environment snapshot: `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/env.txt`
- JSONL (per-run metrics):
  - `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/bench_fps2_maxv4096.jsonl`
  - `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/bench_fps4_maxv8192.jsonl`
- tegrastats logs (per run):
  - `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/tegrastats_fps2_maxv4096/`
  - `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/tegrastats_fps4_maxv8192/`
- Backup of earlier raw data (before GPU isolation): `devdocs/tasks/orin-cosmos-reason2-2b-perf/raw_backup_20260202T133411Z_pre_isolated/`

## Validation
- Command: `python3 -m py_compile scripts/inference_sample.py`
  - Outcome: ok
- Command:
  - `source .venv/bin/activate && export LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}" && python scripts/inference_sample.py --name orin_fps2_maxv4096_isolated --fps 2 --max-vision-tokens 4096 --runs 2 --no-print-output --jsonl-out devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/bench_fps2_maxv4096.jsonl --tegrastats-dir devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/tegrastats_fps2_maxv4096`
  - Outcome: ok (2 runs completed; JSONL + tegrastats logs written)
- Command:
  - `source .venv/bin/activate && export LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}" && python scripts/inference_sample.py --name orin_fps4_maxv8192_isolated --fps 4 --max-vision-tokens 8192 --runs 2 --no-print-output --jsonl-out devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/bench_fps4_maxv8192.jsonl --tegrastats-dir devdocs/tasks/orin-cosmos-reason2-2b-perf/raw/tegrastats_fps4_maxv8192`
  - Outcome: ok (2 runs completed; JSONL + tegrastats logs written)

## Notes / Follow-ups
- For tighter cold-start measurement, repeat after a reboot (or after dropping FS cache) to avoid page-cache effects.
- If performance varies between runs, record/lock clocks (requires root for `jetson_clocks`) and keep the same `nvpmodel`
  mode, thermals, and background load.
