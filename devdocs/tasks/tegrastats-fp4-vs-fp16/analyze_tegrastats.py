#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


_TS_RE = re.compile(r"^(?P<ts>\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})\b")
_RAM_RE = re.compile(r"\bRAM (?P<used>\d+)/(?P<total>\d+)MB\b")
_LFB_RE = re.compile(r"\(lfb (?P<count>\d+)x(?P<mb>\d+)MB\)")
_SWAP_RE = re.compile(r"\bSWAP (?P<used>\d+)/(?P<total>\d+)MB\b")
_SWAP_CACHED_RE = re.compile(r"\(cached (?P<mb>\d+)MB\)")
_CPU_RE = re.compile(r"\bCPU \[(?P<body>.*?)\]")
_CPU_ENTRY_RE = re.compile(r"(?P<pct>\d+)%@(?P<mhz>\d+)")
_EMC_RE = re.compile(r"\bEMC_FREQ (?P<pct>\d+)%@(?P<mhz>\d+)\b")
_GR3D_BRACKET_RE = re.compile(r"\bGR3D_FREQ (?P<pct>\d+)%@\[(?P<mhz>\d+),(?P<aux>\d+)\](?=\s|$)")
_GR3D_PLAIN_RE = re.compile(r"\bGR3D_FREQ (?P<pct>\d+)%@(?P<mhz>\d+)\b")
_TEMP_RE = re.compile(r"(?P<name>[A-Za-z0-9_]+)@(?P<c>[0-9.]+)C\b")
_PWR_RE = re.compile(r"\b(?P<rail>[A-Za-z0-9_]+) (?P<inst>\d+)mW/(?P<avg>\d+)mW\b")


_ENGINES = (
    "NVENC",
    "NVDEC",
    "NVJPG",
    "NVJPG1",
    "VIC",
    "OFA",
    "NVDLA0",
    "NVDLA1",
    "PVA0_FREQ",
)


@dataclass(frozen=True)
class Sample:
    ts: datetime | None
    ram_used_mb: float | None
    ram_total_mb: float | None
    lfb_count: float | None
    lfb_mb: float | None
    swap_used_mb: float | None
    swap_total_mb: float | None
    swap_cached_mb: float | None
    cpu_avg_pct: float | None
    cpu_max_pct: float | None
    emc_pct: float | None
    emc_mhz: float | None
    gpu_pct: float | None
    gpu_mhz: float | None
    temps_c: dict[str, float] = field(default_factory=dict)
    power_mw: dict[str, tuple[float, float]] = field(default_factory=dict)  # rail -> (inst, avg)
    engines: dict[str, str] = field(default_factory=dict)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def _summarize(values: Iterable[float | None]) -> dict[str, float] | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return {
        "n": float(len(xs)),
        "avg": _mean(xs) or 0.0,
        "p50": _quantile(xs, 0.50) or 0.0,
        "p95": _quantile(xs, 0.95) or 0.0,
        "max": max(xs),
        "min": min(xs),
    }


def parse_tegrastats(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            ts_match = _TS_RE.search(line)
            ts = None
            if ts_match is not None:
                ts = datetime.strptime(ts_match.group("ts"), "%m-%d-%Y %H:%M:%S")

            ram_match = _RAM_RE.search(line)
            lfb_match = _LFB_RE.search(line)
            swap_match = _SWAP_RE.search(line)
            swap_cached_match = _SWAP_CACHED_RE.search(line)
            cpu_match = _CPU_RE.search(line)
            emc_match = _EMC_RE.search(line)
            gr3d_match = _GR3D_BRACKET_RE.search(line) or _GR3D_PLAIN_RE.search(line)

            cpu_pcts: list[float] = []
            if cpu_match is not None:
                for ent in cpu_match.group("body").split(","):
                    ent = ent.strip()
                    m = _CPU_ENTRY_RE.fullmatch(ent)
                    if m is not None:
                        cpu_pcts.append(float(m.group("pct")))

            temps: dict[str, float] = {}
            for m in _TEMP_RE.finditer(line):
                temps[m.group("name")] = float(m.group("c"))

            pwrs: dict[str, tuple[float, float]] = {}
            for m in _PWR_RE.finditer(line):
                pwrs[m.group("rail")] = (float(m.group("inst")), float(m.group("avg")))

            engines: dict[str, str] = {}
            for key in _ENGINES:
                m = re.search(rf"\b{re.escape(key)} (?P<v>\S+)\b", line)
                if m is not None:
                    engines[key] = m.group("v")

            samples.append(
                Sample(
                    ts=ts,
                    ram_used_mb=float(ram_match.group("used")) if ram_match else None,
                    ram_total_mb=float(ram_match.group("total")) if ram_match else None,
                    lfb_count=float(lfb_match.group("count")) if lfb_match else None,
                    lfb_mb=float(lfb_match.group("mb")) if lfb_match else None,
                    swap_used_mb=float(swap_match.group("used")) if swap_match else None,
                    swap_total_mb=float(swap_match.group("total")) if swap_match else None,
                    swap_cached_mb=float(swap_cached_match.group("mb")) if swap_cached_match else None,
                    cpu_avg_pct=(sum(cpu_pcts) / len(cpu_pcts)) if cpu_pcts else None,
                    cpu_max_pct=max(cpu_pcts) if cpu_pcts else None,
                    emc_pct=float(emc_match.group("pct")) if emc_match else None,
                    emc_mhz=float(emc_match.group("mhz")) if emc_match else None,
                    gpu_pct=float(gr3d_match.group("pct")) if gr3d_match else None,
                    gpu_mhz=float(gr3d_match.group("mhz")) if gr3d_match else None,
                    temps_c=temps,
                    power_mw=pwrs,
                    engines=engines,
                )
            )

    return samples


def active_slice(samples: list[Sample]) -> list[Sample]:
    idxs = [i for i, s in enumerate(samples) if (s.gpu_pct or 0.0) > 0.0]
    if not idxs:
        return []
    return samples[idxs[0] : idxs[-1] + 1]


def _fmt(v: float | None, *, unit: str = "", digits: int = 1) -> str:
    if v is None:
        return "n/a"
    if digits == 0:
        return f"{v:.0f}{unit}"
    return f"{v:.{digits}f}{unit}"


def summarize(samples: list[Sample]) -> dict:
    out: dict = {
        "samples": len(samples),
        "start": samples[0].ts if samples and samples[0].ts else None,
        "end": samples[-1].ts if samples and samples[-1].ts else None,
        "gpu_pct": _summarize(s.gpu_pct for s in samples),
        "gpu_mhz": _summarize(s.gpu_mhz for s in samples),
        "emc_pct": _summarize(s.emc_pct for s in samples),
        "cpu_avg_pct": _summarize(s.cpu_avg_pct for s in samples),
        "cpu_max_pct": _summarize(s.cpu_max_pct for s in samples),
        "ram_used_mb": _summarize(s.ram_used_mb for s in samples),
        "swap_used_mb": _summarize(s.swap_used_mb for s in samples),
        "lfb_count": _summarize(s.lfb_count for s in samples),
        "lfb_mb": _summarize(s.lfb_mb for s in samples),
    }

    # Temps and power rails: track only keys present.
    temp_keys: set[str] = set()
    pwr_keys: set[str] = set()
    engine_keys: set[str] = set()
    for s in samples:
        temp_keys.update(s.temps_c.keys())
        pwr_keys.update(s.power_mw.keys())
        engine_keys.update(s.engines.keys())

    out["temps"] = {k: _summarize(s.temps_c.get(k) for s in samples) for k in sorted(temp_keys)}

    out["power_inst_mw"] = {
        k: _summarize((s.power_mw.get(k) or (None, None))[0] for s in samples) for k in sorted(pwr_keys)
    }
    out["power_avg_mw"] = {
        k: _summarize((s.power_mw.get(k) or (None, None))[1] for s in samples) for k in sorted(pwr_keys)
    }

    engine_stats: dict[str, dict[str, float]] = {}
    for k in sorted(engine_keys):
        vals = [s.engines.get(k) for s in samples if k in s.engines]
        if not vals:
            continue
        off = sum(1 for v in vals if v == "off")
        on = len(vals) - off
        engine_stats[k] = {
            "present_samples": float(len(vals)),
            "on_samples": float(on),
            "off_samples": float(off),
        }
    out["engines"] = engine_stats

    return out


def render_markdown(label: str, full: dict, active: dict) -> str:
    def ts_str(ts: datetime | None) -> str:
        return ts.isoformat(sep=" ") if ts is not None else "n/a"

    def row(name: str, key: str, unit: str = "", digits: int = 1) -> str:
        f = full.get(key)
        a = active.get(key)
        f_avg = f.get("avg") if isinstance(f, dict) else None
        f_max = f.get("max") if isinstance(f, dict) else None
        a_avg = a.get("avg") if isinstance(a, dict) else None
        a_max = a.get("max") if isinstance(a, dict) else None
        return (
            f"| {name} | {_fmt(f_avg, unit=unit, digits=digits)} / {_fmt(f_max, unit=unit, digits=digits)} |"
            f" {_fmt(a_avg, unit=unit, digits=digits)} / {_fmt(a_max, unit=unit, digits=digits)} |"
        )

    lines: list[str] = []
    lines.append(f"## {label}")
    lines.append("")
    lines.append(f"- Samples: full={int(full.get('samples', 0))}, active={int(active.get('samples', 0))}")
    lines.append(f"- Time range (full): {ts_str(full.get('start'))} → {ts_str(full.get('end'))}")
    lines.append(f"- Time range (active): {ts_str(active.get('start'))} → {ts_str(active.get('end'))}")
    lines.append("")
    lines.append("| Metric | Full (avg/max) | Active (avg/max) |")
    lines.append("|---|---:|---:|")
    lines.append(row("GPU GR3D %", "gpu_pct", unit="%", digits=1))
    lines.append(row("GPU GR3D MHz", "gpu_mhz", unit="MHz", digits=0))
    lines.append(row("EMC %", "emc_pct", unit="%", digits=1))
    lines.append(row("CPU avg %", "cpu_avg_pct", unit="%", digits=1))
    lines.append(row("CPU max core %", "cpu_max_pct", unit="%", digits=0))
    lines.append(row("RAM used", "ram_used_mb", unit="MB", digits=0))
    lines.append(row("SWAP used", "swap_used_mb", unit="MB", digits=0))
    lines.append(row("LFB count", "lfb_count", unit="", digits=0))
    lines.append(row("LFB size", "lfb_mb", unit="MB", digits=0))

    # Common temps/power rails (if present)
    temps = full.get("temps", {})
    pwrs = full.get("power_inst_mw", {})
    if isinstance(temps, dict) and temps:
        lines.append("")
        lines.append("**Temps (full max / active max)**")
        for k in ("tj", "cpu", "soc0", "soc1", "soc2"):
            if k not in temps:
                continue
            f_max = temps[k]["max"] if temps.get(k) else None
            a_max = (active.get("temps", {}) or {}).get(k, {}).get("max")
            lines.append(f"- `{k}`: {_fmt(f_max, unit='C', digits=1)} / {_fmt(a_max, unit='C', digits=1)}")

    if isinstance(pwrs, dict) and pwrs:
        lines.append("")
        lines.append("**Power rails (instantaneous, full avg/max / active avg/max)**")
        for rail in ("VIN_SYS_5V0", "VDD_GPU_SOC", "VDD_CPU_CV"):
            if rail not in pwrs:
                continue
            f = pwrs[rail]
            a = (active.get("power_inst_mw", {}) or {}).get(rail)
            if not f:
                continue
            lines.append(
                f"- `{rail}`: {_fmt(f['avg'], unit='mW', digits=0)}/{_fmt(f['max'], unit='mW', digits=0)}"
                f" / {_fmt((a or {}).get('avg'), unit='mW', digits=0)}/{_fmt((a or {}).get('max'), unit='mW', digits=0)}"
            )

    engines = full.get("engines", {})
    if isinstance(engines, dict) and engines:
        lines.append("")
        lines.append("**Other engines (non-`off` samples in full window)**")
        for k in _ENGINES:
            if k not in engines:
                continue
            on = engines[k].get("on_samples", 0.0)
            present = engines[k].get("present_samples", 0.0)
            lines.append(f"- `{k}`: on {int(on)}/{int(present)} samples")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize tegrastats logs (Jetson).")
    parser.add_argument("--fp4", default="tegra-fp4.log", type=Path)
    parser.add_argument("--fp16", default="tegra-fp16.log", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Write Markdown report to this path (also prints).")
    args = parser.parse_args()

    logs = [("FP4", args.fp4), ("FP16", args.fp16)]
    rendered: list[str] = ["# tegrastats comparison (FP4 vs FP16)", ""]

    summaries: dict[str, tuple[dict, dict]] = {}
    for label, path in logs:
        samples = parse_tegrastats(path)
        active = active_slice(samples)
        full_sum = summarize(samples)
        act_sum = summarize(active)
        summaries[label] = (full_sum, act_sum)
        rendered.append(render_markdown(f"{label} — `{path}`", full_sum, act_sum))

    # Quick delta table for key metrics (active window).
    fp4_active = summaries["FP4"][1]
    fp16_active = summaries["FP16"][1]

    def get_avg(d: dict, key: str) -> float | None:
        v = d.get(key)
        if not isinstance(v, dict):
            return None
        return v.get("avg")

    rendered.append("## Active-window deltas (FP4 - FP16)")
    rendered.append("")
    rendered.append("| Metric (active avg) | FP4 | FP16 | Δ (FP4-FP16) |")
    rendered.append("|---|---:|---:|---:|")
    for key, name, unit, digits in (
        ("gpu_pct", "GPU GR3D %", "%", 1),
        ("emc_pct", "EMC %", "%", 1),
        ("cpu_avg_pct", "CPU avg %", "%", 1),
        ("ram_used_mb", "RAM used", "MB", 0),
        ("power_inst_mw", "VIN_SYS_5V0", "mW", 0),
    ):
        if key == "power_inst_mw":
            fp4 = (fp4_active.get("power_inst_mw", {}) or {}).get("VIN_SYS_5V0", {})
            fp16 = (fp16_active.get("power_inst_mw", {}) or {}).get("VIN_SYS_5V0", {})
            v4 = fp4.get("avg") if isinstance(fp4, dict) else None
            v16 = fp16.get("avg") if isinstance(fp16, dict) else None
        else:
            v4 = get_avg(fp4_active, key)
            v16 = get_avg(fp16_active, key)
        delta = (v4 - v16) if (v4 is not None and v16 is not None) else None
        rendered.append(
            f"| {name} | {_fmt(v4, unit=unit, digits=digits)} | {_fmt(v16, unit=unit, digits=digits)} | {_fmt(delta, unit=unit, digits=digits)} |"
        )

    text = "\n".join(rendered).rstrip() + "\n"
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
