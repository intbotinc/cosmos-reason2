# tegrastats comparison (FP4 vs FP16)

## FP4 — `tegra-fp4.log`

- Samples: full=215, active=91
- Time range (full): 2026-02-09 15:07:03 → 2026-02-09 15:08:53
- Time range (active): 2026-02-09 15:07:13 → 2026-02-09 15:08:00

| Metric | Full (avg/max) | Active (avg/max) |
|---|---:|---:|
| GPU GR3D % | 36.2% / 99.0% | 85.5% / 99.0% |
| GPU GR3D MHz | 398MHz / 611MHz | 567MHz / 611MHz |
| EMC % | 8.5% / 27.0% | 19.3% / 27.0% |
| CPU avg % | 7.6% / 28.0% | 9.0% / 28.0% |
| CPU max core % | 39% / 100% | 47% / 100% |
| RAM used | 24388MB / 30761MB | 29464MB / 30761MB |
| SWAP used | 1527MB / 1527MB | 1527MB / 1527MB |
| LFB count | 41 / 64 | 12 / 47 |
| LFB size | 4MB / 4MB | 4MB / 4MB |

**Temps (full max / active max)**
- `tj`: 55.6C / 55.6C
- `cpu`: 55.6C / 55.6C
- `soc0`: 51.7C / 51.7C
- `soc1`: 51.0C / 50.9C
- `soc2`: 50.2C / 50.2C

**Power rails (instantaneous, full avg/max / active avg/max)**
- `VIN_SYS_5V0`: 5813mW/8353mW / 7313mW/8353mW
- `VDD_GPU_SOC`: 5646mW/10004mW / 8441mW/10004mW
- `VDD_CPU_CV`: 822mW/2000mW / 963mW/2000mW

**Other engines (non-`off` samples in full window)**
- `NVENC`: on 0/215 samples
- `NVDEC`: on 0/215 samples
- `NVJPG`: on 0/215 samples
- `NVJPG1`: on 0/215 samples
- `VIC`: on 0/215 samples
- `OFA`: on 0/215 samples
- `NVDLA0`: on 0/215 samples
- `NVDLA1`: on 0/215 samples
- `PVA0_FREQ`: on 0/215 samples

## FP16 — `tegra-fp16.log`

- Samples: full=137, active=94
- Time range (full): 2026-02-09 15:11:00 → 2026-02-09 15:12:10
- Time range (active): 2026-02-09 15:11:11 → 2026-02-09 15:11:59

| Metric | Full (avg/max) | Active (avg/max) |
|---|---:|---:|
| GPU GR3D % | 56.8% / 99.0% | 82.8% / 99.0% |
| GPU GR3D MHz | 434MHz / 1300MHz | 556MHz / 611MHz |
| EMC % | 13.3% / 27.0% | 18.7% / 27.0% |
| CPU avg % | 9.5% / 37.0% | 8.8% / 26.1% |
| CPU max core % | 49% / 100% | 47% / 100% |
| RAM used | 26942MB / 31159MB | 29473MB / 31159MB |
| SWAP used | 1527MB / 1527MB | 1527MB / 1527MB |
| LFB count | 79 / 112 | 95 / 112 |
| LFB size | 3MB / 4MB | 2MB / 4MB |

**Temps (full max / active max)**
- `tj`: 55.4C / 55.4C
- `cpu`: 55.4C / 55.4C
- `soc0`: 51.9C / 51.9C
- `soc1`: 51.2C / 51.1C
- `soc2`: 50.5C / 50.5C

**Power rails (instantaneous, full avg/max / active avg/max)**
- `VIN_SYS_5V0`: 6471mW/8353mW / 7218mW/8353mW
- `VDD_GPU_SOC`: 6784mW/10004mW / 8287mW/10004mW
- `VDD_CPU_CV`: 969mW/2400mW / 932mW/2000mW

**Other engines (non-`off` samples in full window)**
- `NVENC`: on 0/137 samples
- `NVDEC`: on 0/137 samples
- `NVJPG`: on 0/137 samples
- `NVJPG1`: on 0/137 samples
- `VIC`: on 0/137 samples
- `OFA`: on 0/137 samples
- `NVDLA0`: on 0/137 samples
- `NVDLA1`: on 0/137 samples
- `PVA0_FREQ`: on 0/137 samples

## Active-window deltas (FP4 - FP16)

| Metric (active avg) | FP4 | FP16 | Δ (FP4-FP16) |
|---|---:|---:|---:|
| GPU GR3D % | 85.5% | 82.8% | 2.7% |
| EMC % | 19.3% | 18.7% | 0.6% |
| CPU avg % | 9.0% | 8.8% | 0.3% |
| RAM used | 29464MB | 29473MB | -9MB |
| VIN_SYS_5V0 | 7313mW | 7218mW | 94mW |
