# Task Plan: tegrastats-fp4-vs-fp16

## Goal
Produce a clear FP4 vs FP16 tegrastats comparison (GPU/EMC/CPU/RAM/SWAP/temps/power) with avg/peak metrics and a short interpretation.

## Phases
- [x] Phase 1: Understand + plan
- [x] Phase 2: Parse + summarize logs
- [x] Phase 3: Compare + interpret
- [x] Phase 4: Validate sanity
- [x] Phase 5: Wrap up (deliverable)

## Key Questions
1. Do we report utilization over the whole log, or only “active inference” windows (avoid idle leading/trailing samples)?
2. Which tegrastats fields are present in these logs (GPU/EMC/CPU/RAM/SWAP/temps/power rails), and do any differ by dtype?

## Decisions
- Report both “full window” and “active window” summaries: “active” is samples where `GR3D_FREQ > 0%` (proxy for inference).

## Errors Encountered
- 2026-02-09 <error> → <resolution>

## Status
**CURRENT:** Complete — report + summary ready
