# CURRENT_TASK

Active task: `devdocs/tasks/orin-cosmos-reason2-2b-perf/`

## Context
- Measure inference performance of cosmos-reason2-2B on Jetson AGX Orin
- Use scripts/inference_sample.py as the benchmark entrypoint (add args + JSONL output)
- Capture cold/steady latency, throughput, and system utilization (tegrastats)
- Run >=2 context/input-size configurations and summarize results

## Task Files
- Plan: `devdocs/tasks/orin-cosmos-reason2-2b-perf/task_plan.md`
- Worklog: `devdocs/tasks/orin-cosmos-reason2-2b-perf/worklog.md`
- Deliverable: `devdocs/tasks/orin-cosmos-reason2-2b-perf/deliverable.md`

## History Search
- `rg -n "<keywords>" devdocs/tasks docs/tasks -S`
