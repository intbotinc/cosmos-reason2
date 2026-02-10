# Deliverable: tegrastats-fp4-vs-fp16

## Summary
- Parsed `tegra-fp4.log` and `tegra-fp16.log` (tegrastats) and generated a full-window + active-window utilization comparison report.
- Active window is defined as samples between the first and last line with `GR3D_FREQ > 0%` (proxy for inference load).
- Key takeaway: FP4 and FP16 runs look very similar under load (GPU-bound; ~83–86% GR3D, ~29.5GB RAM used, ~7.2–7.3W VIN_SYS_5V0). Deltas are small and likely within normal run variance.

## Validation
- Command: `python3 devdocs/tasks/tegrastats-fp4-vs-fp16/analyze_tegrastats.py --fp4 tegra-fp4.log --fp16 tegra-fp16.log --out devdocs/tasks/tegrastats-fp4-vs-fp16/report.md`
  - Outcome: OK; wrote `devdocs/tasks/tegrastats-fp4-vs-fp16/report.md`

## Notes / Follow-ups
- Report details: `devdocs/tasks/tegrastats-fp4-vs-fp16/report.md`
- Parser script: `devdocs/tasks/tegrastats-fp4-vs-fp16/analyze_tegrastats.py`
- Both logs show `NVENC/NVDEC/NVJPG/VIC/OFA/NVDLA*` as `off` throughout; workload primarily uses GR3D (GPU).
