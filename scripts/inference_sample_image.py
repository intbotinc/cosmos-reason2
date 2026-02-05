#!/usr/bin/env python
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

"""Minimal example of image inference with Cosmos-Reason2."""

import warnings

warnings.filterwarnings("ignore")

import inspect
import os
from pathlib import Path
import sys
import time

def _ensure_cusparselt_on_ld_library_path() -> None:
    # Ensure cuSPARSELt from the pip wheel is visible to the dynamic loader.
    lib_dir = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cusparselt"
        / "lib"
    )
    if not lib_dir.is_dir():
        return
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths = ld_library_path.split(":") if ld_library_path else []
    if str(lib_dir) not in paths:
        # Restart the process so the loader sees LD_LIBRARY_PATH early.
        if os.environ.get("COSMOS_CUSPARSELT_LD_PATH_SET") == "1":
            return
        os.environ["LD_LIBRARY_PATH"] = (
            f"{lib_dir}:{ld_library_path}" if ld_library_path else str(lib_dir)
        )
        os.environ["COSMOS_CUSPARSELT_LD_PATH_SET"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _count_input_tokens(inputs) -> int:
    if "attention_mask" in inputs:
        return int(inputs["attention_mask"][0].sum().item())
    return int(inputs["input_ids"].shape[-1])


def _count_visual_tokens(inputs) -> int | None:
    total = 0
    found = False
    for key in ("image_grid_thw", "video_grid_thw"):
        if key not in inputs or inputs[key] is None:
            continue
        grid = inputs[key]
        try:
            if hasattr(grid, "reshape"):
                flat = grid.reshape(-1, grid.shape[-1])
                if flat.shape[-1] == 3:
                    total += int((flat[:, 0] * flat[:, 1] * flat[:, 2]).sum().item())
                    found = True
            else:
                for g in grid:
                    if len(g) == 3:
                        total += int(g[0] * g[1] * g[2])
                        found = True
        except Exception:
            return None
    return total if found else None


ROOT = Path(__file__).parents[1]
SEPARATOR = "-" * 20

PIXELS_PER_TOKEN = 32**2
"""Number of pixels per visual token."""


def main():
    _ensure_cusparselt_on_ld_library_path()

    import torch
    import transformers

    # Ensure reproducibility
    t0 = time.time()
    transformers.set_seed(0)
    t1 = time.time()
    print("set_seed done", f"{t1 - t0:.3f}s")

    # Load model
    #model_name = "nvidia/Cosmos-Reason2-2B"
    model_name = "/tmp/cosmos-reason2/checkpoints/model_nvfp4"
    if not torch.cuda.is_available():
        raise SystemExit("error: CUDA is not available; aborting")

    device_map = {"": 0}
    dtype = torch.float16

    try:
        sdpa_sig = inspect.signature(torch.nn.functional.scaled_dot_product_attention)
        attn_impl = "sdpa" if "enable_gqa" in sdpa_sig.parameters else "eager"
    except (TypeError, ValueError):
        attn_impl = "eager"

    t = time.time()
    print("load model")
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    t2 = time.time()
    print("load model done", f"{t2 - t:.3f}s")

    t = time.time()
    print("load processor")
    processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)
    t3 = time.time()
    print("load processor done", f"{t3 - t:.3f}s")

    t = time.time()
    print("preprocess")
    # Optional: Limit vision tokens
    min_vision_tokens = 256
    max_vision_tokens = 8192
    processor.image_processor.size = {
        "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
    }

    # Create inputs
    # IMPORTANT: Media is listed before text to match training inputs
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{ROOT}/assets/sample.png",
                },
                {"type": "text", "text": "Describe this image in detail."},
            ],
        },
    ]

    # Process inputs
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_tokens = _count_input_tokens(inputs)
    visual_tokens = _count_visual_tokens(inputs)
    print("input tokens", input_tokens)
    if visual_tokens is None:
        print("visual tokens", "n/a")
    else:
        print("visual tokens", visual_tokens)
    inputs = inputs.to(model.device)
    t4 = time.time()
    print("preprocess done", f"{t4 - t:.3f}s")

    t = time.time()
    print("generate")
    # Run inference
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    t5 = time.time()
    print("generate done", f"{t5 - t:.3f}s")
    output_tokens = generated_ids.shape[-1] - input_tokens
    print("output tokens", output_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    t = time.time()
    print("decode")
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    t6 = time.time()
    print("decode done", f"{t6 - t:.3f}s")
    print("total", f"{t6 - t0:.3f}s")
    print(SEPARATOR)
    print(output_text[0])
    print(SEPARATOR)


if __name__ == "__main__":
    main()
