#!/usr/bin/env -S UV_SKIP_WHEEL_FILENAME_CHECK=1 uv run --script --no-build-isolation-package torchvision
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

# https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   # Core deps
#   "compressed-tensors==0.10.2; platform_machine == 'aarch64'",
#   "datasets==4.4.1",
#   "llmcompressor==0.3.0; platform_machine == 'aarch64'",
#   "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git@6e459ed; platform_machine != 'aarch64'",
#   "pillow==12.0.0",
#   "pydantic==2.12.4",
#   "qwen-vl-utils==0.0.14",
#   "tyro>=0.9.35",
#   "transformers==4.57.3",
#
#   # Jetson AGX Orin (aarch64, JetPack 6.x): use NVIDIA's torch wheel + NumPy 1.x
#   "numpy<2; platform_machine == 'aarch64'",
#   "nvidia-cusparselt-cu12; platform_machine == 'aarch64'",
#   "torch @ https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0%2B872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl; platform_machine == 'aarch64'",
#   "torchvision @ git+https://github.com/pytorch/vision.git@v0.20.1; platform_machine == 'aarch64'",
#
#   # x86_64 CUDA (keep existing flow)
#   "torch==2.8.0; platform_machine != 'aarch64'",
#   "torchvision; platform_machine != 'aarch64'",
#   "torchcodec>=0.8.1; platform_machine != 'aarch64'",
# ]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128", marker = "platform_machine != 'aarch64'" }]
# torchvision = [{ index = "pytorch-cu128", marker = "platform_machine != 'aarch64'" }]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

"""Quantize a Cosmos-Reason2 model.

Example:

```shell
./scripts/quantize.py
```
"""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
import platform
from typing import Annotated, Literal

_MINIMUM_HF_CLI_VERSION = "1.3.5"
_CUSPARSELT_REEXEC_ENV = "_COSMOS_REASON2_CUSPARSELT_REEXEC"


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


def _maybe_patch_torchvision_meta_registrations() -> None:
    """Patch torchvision's nms meta registration to avoid missing-op crashes on Jetson.

    On some Jetson builds, torchvision ops may be unavailable, and importing torchvision can fail with:
    `RuntimeError: operator torchvision::nms does not exist`.
    """

    if platform.machine() != "aarch64":
        return

    tv_meta_file = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "torchvision"
        / "_meta_registrations.py"
    )
    if not tv_meta_file.is_file():
        return

    lines = tv_meta_file.read_text().splitlines()
    needle = '@torch.library.register_fake("torchvision::nms")'
    try:
        dec_idx = next(i for i, line in enumerate(lines) if needle in line)
    except StopIteration:
        return

    start_idx = dec_idx
    if dec_idx > 0 and lines[dec_idx - 1].strip() == "if torchvision.extension._has_ops():":
        start_idx = dec_idx - 1

    try:
        end_idx = next(i for i in range(dec_idx + 1, len(lines)) if lines[i].startswith("@register_meta("))
    except StopIteration:
        return

    patched_block = [
        "if torchvision.extension._has_ops():",
        '    @torch.library.register_fake("torchvision::nms")',
        "    def meta_nms(dets, scores, iou_threshold):",
        '        torch._check(dets.dim() == 2, lambda: f"boxes should be a 2d tensor, got {dets.dim()}D")',
        "        torch._check(",
        '            dets.size(1) == 4, lambda: f"boxes should have 4 elements in dimension 1, got {dets.size(1)}"',
        "        )",
        '        torch._check(scores.dim() == 1, lambda: f"scores should be a 1d tensor, got {scores.dim()}")',
        "        torch._check(",
        "            dets.size(0) == scores.size(0),",
        '            lambda: f"boxes and scores should have same number of elements in dimension 0, got {dets.size(0)} and {scores.size(0)}",',
        "        )",
        "        ctx = torch._custom_ops.get_ctx()",
        "        num_to_keep = ctx.create_unbacked_symint()",
        "        return dets.new_empty(num_to_keep, dtype=torch.long)",
        "else:",
        "    def meta_nms(dets, scores, iou_threshold):",
        "        return None",
        "",
    ]

    new_lines = lines[:start_idx] + patched_block + lines[end_idx:]
    tv_meta_file.write_text("\n".join(new_lines) + "\n")


def _maybe_patch_compressed_tensors_kv_cache_scale_type() -> None:
    """Backfill KVCacheScaleType for compressed-tensors versions that removed it.

    `llmcompressor==0.3.0` imports `KVCacheScaleType` from
    `compressed_tensors.quantization.lifecycle`, but newer `compressed-tensors`
    versions may not export it.
    """

    try:
        from compressed_tensors.quantization import lifecycle as ct_lifecycle  # type: ignore
    except Exception:
        return

    if hasattr(ct_lifecycle, "KVCacheScaleType"):
        return

    try:
        from enum import Enum

        class KVCacheScaleType(str, Enum):
            KEY = "key"
            VALUE = "value"

        ct_lifecycle.KVCacheScaleType = KVCacheScaleType  # type: ignore[attr-defined]
    except Exception:
        return


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


def init():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCH_LOGS"] = "-dynamo"

    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.ERROR)


init()
_maybe_reexec_with_cusparselt_in_ld_library_path()
_maybe_patch_torchvision_meta_registrations()
_maybe_patch_compressed_tensors_kv_cache_scale_type()
print("Loading dependencies...")


import base64
import json
from io import BytesIO

import pydantic
import requests
import torch
import tyro
from datasets import load_dataset

# llmcompressor API compatibility:
# - Newer versions (used on x86_64) provide `oneshot` + `moe_calibration_context`.
# - Jetson Orin uses `llmcompressor==0.3.0`, which provides `apply()` + `Recipe`.
try:  # pragma: no cover
    from llmcompressor import oneshot as llmcompressor_oneshot
except Exception:  # pragma: no cover
    llmcompressor_oneshot = None
    from llmcompressor import apply as llmcompressor_apply

try:  # pragma: no cover
    from llmcompressor.modeling.moe_context import moe_calibration_context
except Exception:  # pragma: no cover
    from contextlib import contextmanager

    @contextmanager
    def moe_calibration_context(_model):
        yield

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
try:  # pragma: no cover
    from llmcompressor.utils import dispatch_for_generation
except Exception:  # pragma: no cover
    def dispatch_for_generation(_model):
        return None
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

Precision = Literal["nvfp4", "fp8", "fp8_dynamic"]
KvPrecision = Literal["bf16", "fp8"]


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Directory to save the quantized model. Model will be saved in {output_dir}/model_{precision}."""
    model: str = "nvidia/Cosmos-Reason2-2B"
    """Local path to a model or model name from https://huggingface.co/collections/nvidia/cosmos-reason2."""
    num_samples: int = 512
    """Number of samples to use for calibration."""
    precision: Precision = "nvfp4"
    """Precision to use for quantization."""
    kv_precision: KvPrecision = "bf16"
    """Precision to use for the KV cache quantization."""
    smoothing_strength: float = 0.8
    """Smoothing strength to use for SmoothQuant."""
    max_sequence_length: int = 262144
    """Maximum sequence length to use for quantization. (Defaults to CR2/Qwen3-VL max model len)"""
    seed: int = 42
    """Seed to use for random number generator."""


def _hf_download(cmd_args: list[str]) -> str:
    """Run Hugging Face CLI download command and return the local path.

    Uses a newer Hugging Face CLI version to download checkpoint. The dependency
    version is very old and not robust.
    """
    cmd = [
        "uvx",
        f"hf>={_MINIMUM_HF_CLI_VERSION}",
        "download",
        *cmd_args,
    ]
    print(f"{shlex.join(cmd)}")
    subprocess.check_call(cmd, text=True)
    return subprocess.check_output(
        [*cmd, "--quiet"], text=True, env=dict(os.environ) | {"HF_HUB_OFFLINE": "1"}
    ).strip()


def preprocess_and_tokenize(
    example: dict, processor: AutoProcessor, max_sequence_length: int
) -> dict:
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base64_qwen},
                {"type": "text", "text": "What does the image show?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=max_sequence_length,
        truncation=True,
    )


def data_collator(batch: list[dict]) -> dict:
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def get_quantization_recipe(
    precision: Precision, kv_precision: KvPrecision, smoothing_strength: float
) -> list[SmoothQuantModifier | QuantizationModifier]:
    kv_scheme = (
        None
        if kv_precision == "bf16"
        else {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False}
    )
    recipe = [
        SmoothQuantModifier(
            smoothing_strength=smoothing_strength,
            mappings=[
                [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
                [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
            ],
        ),
        QuantizationModifier(
            targets="Linear",
            scheme=precision.upper(),
            ignore=[
                "re:.*lm_head",
                "re:visual.*",
                "re:model.visual.*",
                "re:.*mlp.gate$",
            ],
            kv_cache_scheme=kv_scheme,
        ),
    ]
    return recipe


def run_sample_generation(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    max_sequence_length: int,
    device: str,
):
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    test_url = "http://images.cocodataset.org/train2017/000000231895.jpg"
    test_image = Image.open(BytesIO(requests.get(test_url).content))

    test_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_url},
                {"type": "text", "text": "Please describe the animal in this image\n"},
            ],
        }
    ]

    prompt = processor.apply_chat_template(test_messages, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=[test_image],
        padding=False,
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    print("Generating response...")
    output = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    print("==========================================")


def save_model(
    model: Qwen3VLForConditionalGeneration, processor: AutoProcessor, output_dir: Path
):
    model.save_pretrained(output_dir, save_compressed=True)


def postprocess_config(config_path: Path):
    def remove_keys(d, keys_to_remove):
        if isinstance(d, dict):
            return {
                k: remove_keys(v, keys_to_remove)
                for k, v in d.items()
                if k not in keys_to_remove
            }
        elif isinstance(d, list):
            return [remove_keys(i, keys_to_remove) for i in d]
        else:
            return d

    with open(config_path) as f:
        config = json.load(f)
    clean_config = remove_keys(config, keys_to_remove=["zp_dtype", "scale_dtype"])
    with open(config_path, "w") as f:
        json.dump(clean_config, f, indent=2)


def _recipe_yaml_from_modifiers(modifiers: list) -> str:
    """Create a llmcompressor recipe YAML string from modifier objects.

    `llmcompressor==0.3.0` has a bug when creating recipes from modifiers: it uses
    `yaml.dump`, which serializes tuples as `!!python/tuple`, but then parses with
    `yaml.safe_load`, which rejects those tags. We use `yaml.safe_dump` to generate
    a safe, portable recipe string.
    """

    import yaml

    return yaml.safe_dump(
        {
            "DEFAULT_stage": {
                "DEFAULT_modifiers": {
                    modifier.__class__.__name__: modifier.model_dump(exclude_unset=True)
                    for modifier in modifiers
                }
            }
        },
        sort_keys=False,
    )


def quantize(args: Args):
    print("Pre-downloading dataset: lmms-lab/flickr30k")
    _hf_download(["lmms-lab/flickr30k", "--repo-type", "dataset"])
    if os.path.exists(args.model):
        model_path = Path(args.model)
    else:
        print(f"Pre-downloading model: {args.model}")
        model_path = Path(_hf_download([args.model]))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    attn_implementation = (
        "sdpa" if _sdpa_supports_enable_gqa(torch) else "eager"
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        attn_implementation=attn_implementation,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model to device: {device}")
    model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    dataset_id = "lmms-lab/flickr30k"
    dataset_split = {"calibration": f"test[:{args.num_samples}]"}
    output_dir = Path(args.output_dir) / f"model_{args.precision}"
    sequential_targets = ["Qwen3VLTextDecoderLayer"]

    # workaround to register KV metadata to the VLM HF config
    model.config.num_attention_heads = model.config.text_config.num_attention_heads
    model.config.num_key_value_heads = model.config.text_config.num_key_value_heads
    model.config.head_dim = model.config.text_config.head_dim

    print(f"Loading calibration dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split=dataset_split)
    ds = ds.shuffle(seed=args.seed)
    print("Preprocessing dataset...")
    ds = ds.map(
        lambda x: preprocess_and_tokenize(x, processor, args.max_sequence_length),
        batched=False,
        remove_columns=ds["calibration"].column_names,
    )
    recipe_modifiers = get_quantization_recipe(
        args.precision, args.kv_precision, args.smoothing_strength
    )

    print(f"Starting {args.precision} quantization process...")
    def device_data_collator(batch: list[dict]) -> dict:
        assert len(batch) == 1
        return {
            key: torch.tensor(value, device=device) for key, value in batch[0].items()
        }

    with moe_calibration_context(model):
        if llmcompressor_oneshot is not None:
            llmcompressor_oneshot(
                model=model,
                recipe=recipe_modifiers,
                max_seq_length=args.max_sequence_length,
                num_calibration_samples=args.num_samples,
                dataset=ds,
                data_collator=device_data_collator,
                sequential_targets=sequential_targets,
            )
        else:
            from torch.utils.data import DataLoader

            calib_loader = DataLoader(
                ds["calibration"],
                batch_size=1,
                shuffle=False,
                collate_fn=device_data_collator,
            )
            llmcompressor_apply(
                recipe=_recipe_yaml_from_modifiers(recipe_modifiers),
                model=model,
                calib_data=calib_loader,
                copy_data=False,
            )
    print("Quantization complete!")
    print("Running sample generation...")
    run_sample_generation(model, processor, args.max_sequence_length, device=device)
    print(f"Saving quantized model to: {output_dir}...")
    save_model(model, processor, output_dir)
    config_path = output_dir / "config.json"
    print(f"Postprocessing config file {config_path}...")
    postprocess_config(config_path)
    shutil.copytree(
        model_path,
        output_dir,
        ignore=lambda dir, files: [
            f for f in files if f == "config.json" or "safetensors" in f
        ],
        dirs_exist_ok=True,
    )
    print(f"Quantization complete! Model saved to: {output_dir}")


def main():
    from loguru import logger as loguru_logger

    loguru_logger.remove()

    args = tyro.cli(Args, description=__doc__)
    quantize(args)


if __name__ == "__main__":
    main()
