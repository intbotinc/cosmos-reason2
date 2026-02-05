#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate==1.12.0",
#   "av==16.1.0",
#   "pillow==12.0.0",
#   "transformers==4.57.3",
#   "torch==2.9.0",
#   "torchvision",
#   "torchcodec==0.9.1; platform_machine != 'aarch64'",
# ]
# ///

"""Minimal example of inference with Cosmos-Reason2."""

# Source: https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#new-qwen-vl-utils-usage

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).parents[1]
SEPARATOR = "-" * 20

PIXELS_PER_TOKEN = 32**2
"""Number of pixels per visual token."""

_CUSPARSELT_REEXEC_ENV = "_COSMOS_REASON2_CUSPARSELT_REEXEC"
_PROCESS_START_NS_ENV = "_COSMOS_REASON2_PROCESS_START_NS"


def _log_timing(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@contextmanager
def _timed_step(
    step_name: str,
    *,
    timings: dict[str, float] | None = None,
    sync: Callable[[], None] | None = None,
) -> Iterator[None]:
    if sync is not None:
        sync()
    start_time = time.perf_counter()
    try:
        yield
    finally:
        if sync is not None:
            sync()
        elapsed_s = time.perf_counter() - start_time
        if timings is not None:
            timings[step_name] = elapsed_s
        _log_timing(f"[timing] {step_name}: {elapsed_s:.3f}s")


def _maybe_reexec_with_cusparselt_in_ld_library_path() -> None:
    """Torch on Jetson may require cuSPARSELt in LD_LIBRARY_PATH (needs restart)."""

    if os.environ.get(_CUSPARSELT_REEXEC_ENV) == "1":
        return

    prefix = Path(sys.prefix)
    lib_dir = (
        prefix
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cusparselt"
        / "lib"
    )
    if not (lib_dir / "libcusparseLt.so.0").exists():
        return

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_paths = [p for p in ld_library_path.split(":") if p]
    if str(lib_dir) in ld_paths:
        return

    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_library_path}" if ld_library_path else str(lib_dir)
    new_env[_CUSPARSELT_REEXEC_ENV] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], new_env)


def _sdpa_supports_enable_gqa(torch) -> bool:
    try:
        q = torch.empty((1, 1, 1, 1))
        torch.nn.functional.scaled_dot_product_attention(q, q, q, enable_gqa=True)
        return True
    except TypeError as exc:
        if "enable_gqa" in str(exc):
            return False
        return False
    except Exception:
        return True


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=None, help="Optional tag recorded in JSON output.")
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-2B", help="Hugging Face model name or path.")
    parser.add_argument(
        "--video",
        default=str(ROOT / "assets/sample.mp4"),
        help="Path to input video (default: assets/sample.mp4).",
    )
    parser.add_argument("--fps", type=int, default=4, help="Video FPS used during preprocessing.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="max_new_tokens passed to generate().")
    parser.add_argument("--min-vision-tokens", type=int, default=256, help="Minimum vision tokens (size constraint).")
    parser.add_argument("--max-vision-tokens", type=int, default=8192, help="Maximum vision tokens (size constraint).")
    parser.add_argument("--runs", type=int, default=1, help="Number of measured inference runs in this process.")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Number of warmup runs before measured runs.")
    parser.add_argument(
        "--no-print-output",
        action="store_true",
        help="Do not print generated text (recommended for benchmarking).",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=None,
        help="Append per-run metrics as JSON lines to this file.",
    )
    parser.add_argument(
        "--tegrastats-dir",
        type=Path,
        default=None,
        help="If set, capture tegrastats for each run into this directory.",
    )
    parser.add_argument(
        "--tegrastats-interval-ms",
        type=int,
        default=1000,
        help="Tegrastats sampling interval in milliseconds.",
    )
    return parser.parse_args(argv)


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False)
        handle.write("\n")


@contextmanager
def _maybe_tegrastats(log_path: Path | None, *, interval_ms: int) -> Iterator[Path | None]:
    if log_path is None:
        yield None
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("wb")
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        handle.close()
        _log_timing("[timing] warning: tegrastats not found; skipping utilization logging")
        yield None
        return

    try:
        yield log_path
    finally:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        finally:
            handle.close()


_RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
_SWAP_RE = re.compile(r"SWAP (\d+)/(\d+)MB")
_CPU_RE = re.compile(r"CPU \[(.*?)\]")
_PCT_RE = re.compile(r"(\d+)%")
_GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%@?(\d+)?")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _parse_tegrastats(log_path: Path | None) -> dict[str, Any] | None:
    if log_path is None or not log_path.exists():
        return None

    gpu_pcts: list[float] = []
    gpu_freqs_mhz: list[float] = []
    cpu_avg_pcts: list[float] = []
    cpu_max_pcts: list[float] = []
    ram_used_mb: list[float] = []
    ram_total_mb: list[float] = []
    swap_used_mb: list[float] = []
    swap_total_mb: list[float] = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            ram = _RAM_RE.search(line)
            if ram is not None:
                ram_used_mb.append(float(ram.group(1)))
                ram_total_mb.append(float(ram.group(2)))

            swap = _SWAP_RE.search(line)
            if swap is not None:
                swap_used_mb.append(float(swap.group(1)))
                swap_total_mb.append(float(swap.group(2)))

            cpu = _CPU_RE.search(line)
            if cpu is not None:
                core_pcts = [float(p) for p in _PCT_RE.findall(cpu.group(1))]
                if core_pcts:
                    cpu_avg_pcts.append(sum(core_pcts) / len(core_pcts))
                    cpu_max_pcts.append(max(core_pcts))

            gr3d = _GR3D_RE.search(line)
            if gr3d is not None:
                gpu_pcts.append(float(gr3d.group(1)))
                if gr3d.group(2) is not None:
                    gpu_freqs_mhz.append(float(gr3d.group(2)))

    samples = max(len(gpu_pcts), len(cpu_avg_pcts), len(ram_used_mb))
    if samples == 0:
        return {"samples": 0}

    return {
        "samples": samples,
        "gpu_percent_avg": _mean(gpu_pcts),
        "gpu_percent_max": max(gpu_pcts) if gpu_pcts else None,
        "gpu_freq_mhz_avg": _mean(gpu_freqs_mhz),
        "gpu_freq_mhz_max": max(gpu_freqs_mhz) if gpu_freqs_mhz else None,
        "cpu_percent_avg": _mean(cpu_avg_pcts),
        "cpu_percent_max": max(cpu_max_pcts) if cpu_max_pcts else None,
        "ram_used_mb_avg": _mean(ram_used_mb),
        "ram_used_mb_max": max(ram_used_mb) if ram_used_mb else None,
        "ram_total_mb": max(ram_total_mb) if ram_total_mb else None,
        "swap_used_mb_avg": _mean(swap_used_mb),
        "swap_used_mb_max": max(swap_used_mb) if swap_used_mb else None,
        "swap_total_mb": max(swap_total_mb) if swap_total_mb else None,
    }


def _tensor_shape(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def main(argv: list[str] | None = None) -> int:
    if os.environ.get(_PROCESS_START_NS_ENV) is None:
        os.environ[_PROCESS_START_NS_ENV] = str(time.perf_counter_ns())

    args = _parse_args(argv)

    _maybe_reexec_with_cusparselt_in_ld_library_path()

    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit(
            "error: CUDA is not available. On Jetson, ensure you installed the Jetson PyTorch wheel and cuSPARSELt."
        )

    if args.runs < 1:
        raise SystemExit("error: --runs must be >= 1")
    if args.warmup_runs < 0:
        raise SystemExit("error: --warmup-runs must be >= 0")

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"error: video not found: {video_path}")

    # Ensure reproducibility
    setup_timings: dict[str, float] = {}
    sync_cuda = torch.cuda.synchronize
    sync_cuda()
    overall_start_time = time.perf_counter()

    with _timed_step("set seed", timings=setup_timings):
        transformers.set_seed(0)

    # Load model
    model_name = args.model
    attn_implementation = "sdpa" if _sdpa_supports_enable_gqa(torch) else "eager"
    with _timed_step("load model + processor", timings=setup_timings, sync=sync_cuda):
        try:
            model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
                attn_implementation=attn_implementation,
            )
            try:
                processor = transformers.Qwen3VLProcessor.from_pretrained(
                    model_name,
                    fix_mistral_regex=True,
                )
            except TypeError as exc:
                if "fix_mistral_regex" not in str(exc):
                    raise
                processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)
        except OSError as exc:
            msg = str(exc)
            if "gated repo" in msg or "Cannot access gated repo" in msg or "401" in msg:
                raise SystemExit(
                    "error: Hugging Face model 'nvidia/Cosmos-Reason2-2B' is gated.\n"
                    "Request access on Hugging Face, then authenticate via `huggingface-cli login` or set HF_TOKEN."
                ) from None
            raise

    # Optional: Limit vision tokens
    with _timed_step("configure vision token limits", timings=setup_timings):
        min_vision_tokens = args.min_vision_tokens
        max_vision_tokens = args.max_vision_tokens
        processor.image_processor.size = {
            "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
            "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
        }
        processor.video_processor.size = {
            "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
            "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
        }

    # Create inputs
    # IMPORTANT: Media is listed before text to match training inputs
    with _timed_step("build conversation", timings=setup_timings):
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                    },
                    {"type": "text", "text": "Caption the video in detail."},
                ],
            },
        ]

    total_runs = args.warmup_runs + args.runs
    process_start_ns = int(os.environ.get(_PROCESS_START_NS_ENV, str(time.perf_counter_ns())))

    torch_meta: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
    }

    for run_idx in range(total_runs):
        is_warmup = run_idx < args.warmup_runs
        run_timings: dict[str, float] = {}
        torch.cuda.reset_peak_memory_stats()

        tegra_log: Path | None = None
        if args.tegrastats_dir is not None:
            args.tegrastats_dir.mkdir(parents=True, exist_ok=True)
            tegra_log = args.tegrastats_dir / f"tegrastats_run{run_idx + 1}.log"

        with _maybe_tegrastats(tegra_log, interval_ms=args.tegrastats_interval_ms) as tegra_log_path:
            # Process inputs
            with _timed_step("preprocess inputs", timings=run_timings, sync=sync_cuda):
                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    fps=args.fps,
                )
                inputs = inputs.to(model.device)

            # Run inference
            with _timed_step("generate", timings=run_timings, sync=sync_cuda):
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

            with _timed_step("decode", timings=run_timings, sync=sync_cuda):
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            if not args.no_print_output and not is_warmup:
                print(SEPARATOR)
                print(output_text[0])
                print(SEPARATOR)

        generated_tokens = int(generated_ids_trimmed[0].shape[-1])
        generate_s = float(run_timings.get("generate", 0.0))
        tokens_per_sec = (generated_tokens / generate_s) if generate_s > 0 else None
        ms_per_token = (1000.0 * generate_s / generated_tokens) if generated_tokens > 0 else None

        process_elapsed_s = (time.perf_counter_ns() - process_start_ns) / 1e9
        cuda_peak_alloc_bytes = int(torch.cuda.max_memory_allocated())
        cuda_peak_reserved_bytes = int(torch.cuda.max_memory_reserved())

        tegrastats_summary = _parse_tegrastats(tegra_log_path)

        inputs_summary: dict[str, Any] = {
            "input_ids_shape": _tensor_shape(getattr(inputs, "input_ids", None)),
            "attention_mask_shape": _tensor_shape(getattr(inputs, "attention_mask", None)),
        }
        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "pixel_values_images",
        ):
            if key in inputs:
                inputs_summary[key] = _tensor_shape(inputs[key])

        run_record: dict[str, Any] = {
            "name": args.name,
            "run_index": run_idx + 1,
            "warmup": is_warmup,
            "process_elapsed_s": process_elapsed_s,
            "model_name": model_name,
            "dtype": "float16",
            "attn_implementation": attn_implementation,
            "config": {
                "video": str(video_path),
                "fps": args.fps,
                "max_new_tokens": args.max_new_tokens,
                "min_vision_tokens": args.min_vision_tokens,
                "max_vision_tokens": args.max_vision_tokens,
            },
            "setup_timings_s": dict(setup_timings),
            "run_timings_s": dict(run_timings),
            "generated_tokens": generated_tokens,
            "tokens_per_sec": tokens_per_sec,
            "ms_per_token": ms_per_token,
            "cuda_peak_allocated_bytes": cuda_peak_alloc_bytes,
            "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
            "inputs_summary": inputs_summary,
            "tegrastats_log": str(tegra_log_path) if tegra_log_path is not None else None,
            "tegrastats_summary": tegrastats_summary,
            "torch_meta": torch_meta,
        }
        if args.jsonl_out is not None:
            _append_jsonl(args.jsonl_out, run_record)

        _log_timing(
            f"[timing] run {run_idx + 1}/{total_runs} warmup={is_warmup} "
            f"gen_tokens={generated_tokens} tok/s={tokens_per_sec:.2f}"
            if tokens_per_sec is not None
            else f"[timing] run {run_idx + 1}/{total_runs} warmup={is_warmup} gen_tokens={generated_tokens}"
        )

    sync_cuda()
    overall_elapsed_s = time.perf_counter() - overall_start_time
    _log_timing(f"[timing] total: {overall_elapsed_s:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
