# Task Plan: orin-cosmos-reason2-2b-perf

## Goal
Measure and report Cosmos-Reason2-2B inference performance on Jetson AGX Orin (latency, throughput, and system
utilization) for at least two workload configurations, using `scripts/inference_sample.py`, with raw data archived.

## Phases
- [x] Phase 1: Understand + plan
- [x] Phase 2: Investigate / gather context
- [x] Phase 3: Implement
- [x] Phase 4: Validate
- [x] Phase 5: Wrap up (deliverable)

## Key Questions
1. Which workload knobs best approximate “representative” use on Orin (FPS, vision tokens, max_new_tokens)?
2. What should “throughput” mean for this script (effective tokens/sec for full `generate` vs decode-only)?
3. Is `tegrastats` available in the target environment, and what fields does it expose on this Jetson image?

## Decisions
- Use `scripts/inference_sample.py` as the benchmark entrypoint (add argparse + JSONL output): keeps workload identical
  to the minimal example and minimizes new tooling.
- Define cold-start latency as process start → end of first inference run; define steady-state as subsequent runs in the
  same process after model load.
- Use `tegrastats` (1s interval) to monitor GPU/CPU/RAM during each inference run and parse logs into avg/peak stats.

## Errors Encountered
- 2026-02-01 `ImportError: libcusparseLt.so.0` (torch import) → export cuSPARSELt into `LD_LIBRARY_PATH` when running
  ad-hoc `python -c ...` snippets (the benchmark script can auto-reexec, but env snapshots and other scripts may need it).

## Status
**CURRENT:** Done
