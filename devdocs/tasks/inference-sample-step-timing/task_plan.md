# Task Plan: inference-sample-step-timing

## Goal
Add per-step timing logs to `scripts/inference_sample.py` so users can see how long model load, preprocessing, generation, and decoding take.

## Phases
- [x] Phase 1: Understand + plan
- [x] Phase 2: Investigate / gather context
- [x] Phase 3: Implement
- [x] Phase 4: Validate
- [x] Phase 5: Wrap up (deliverable)

## Key Questions
1. Should timing logs go to stderr (so stdout stays clean for the generated caption)?
2. Should we `torch.cuda.synchronize()` to get accurate GPU timings?

## Decisions
- Log timings to stderr via `print(..., file=sys.stderr)`: keeps generated caption output clean.
- Synchronize CUDA around GPU steps: avoids under-reporting asynchronous GPU work.

## Errors Encountered
- None so far.

## Status
**CURRENT:** Complete
