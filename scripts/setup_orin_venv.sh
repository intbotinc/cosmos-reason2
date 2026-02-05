#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
TORCH_WHL_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0%2B872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

cd "$ROOT_DIR"

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "error: python3.10 not found; install Python 3.10 first" >&2
  exit 1
fi

if [[ -d "$VENV_DIR" ]]; then
  ts="$(date +%Y%m%d-%H%M%S)"
  mv "$VENV_DIR" "${VENV_DIR}.bak-${ts}"
  echo "info: moved existing .venv to .venv.bak-${ts}"
fi

python3.10 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir --upgrade setuptools wheel

# Jetson-compatible PyTorch (sm_87) from NVIDIA JetPack index
python -m pip install --no-cache-dir --no-deps "$TORCH_WHL_URL"

# cuSPARSELt runtime library required by torch
python -m pip install --no-cache-dir nvidia-cusparselt-cu12
PY_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUSPARSELT_LIB_DIR="$PY_SITE/nvidia/cusparselt/lib"
if [[ -d "$CUSPARSELT_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$CUSPARSELT_LIB_DIR:${LD_LIBRARY_PATH:-}"
else
  echo "warning: cuSPARSELt lib dir not found at $CUSPARSELT_LIB_DIR; torch may fail to import" >&2
fi

# Runtime deps for inference sample
python -m pip install --no-cache-dir \
  accelerate==1.12.0 \
  av==16.1.0 \
  pillow==12.0.0 \
  transformers==4.57.3

# NumPy 2.x is incompatible with this torch build
python -m pip install --no-cache-dir "numpy<2"

# Runtime deps for quantization (`scripts/quantize.py`)
python -m pip install --no-cache-dir \
  "compressed-tensors==0.10.2" \
  datasets==4.4.1 \
  llmcompressor==0.3.0 \
  pydantic==2.12.4 \
  qwen-vl-utils==0.0.14 \
  tyro==0.9.35

# Build torchvision from source against Jetson torch (disable build isolation to avoid linking against a different torch)
python -m pip uninstall -y torchvision || true
python -m pip install --no-cache-dir --no-deps --no-build-isolation "git+https://github.com/pytorch/vision.git@v0.20.1"

# Patch torchvision meta registrations to avoid missing op crash
TV_META_FILE="$VENV_DIR/lib/python3.10/site-packages/torchvision/_meta_registrations.py"
if [[ -f "$TV_META_FILE" ]]; then
  python - "$TV_META_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).resolve()
lines = path.read_text().splitlines()

needle = '@torch.library.register_fake("torchvision::nms")'
try:
    dec_idx = next(i for i, line in enumerate(lines) if needle in line)
except StopIteration:
    print(f"warning: did not find {needle} in {path}; patch skipped", file=sys.stderr)
    raise SystemExit(0)

start_idx = dec_idx
if dec_idx > 0 and lines[dec_idx - 1].strip() == "if torchvision.extension._has_ops():":
    start_idx = dec_idx - 1

try:
    end_idx = next(i for i in range(dec_idx + 1, len(lines)) if lines[i].startswith("@register_meta("))
except StopIteration:
    print(f"warning: did not find next @register_meta(...) after nms in {path}; patch skipped", file=sys.stderr)
    raise SystemExit(0)

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
path.write_text("\n".join(new_lines) + "\n")
PY
else
  echo "warning: torchvision meta registrations file not found; patch skipped" >&2
fi

python - <<'PY'
import torch
import torchvision
import torchvision.extension

print(f"info: torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"info: torchvision={torchvision.__version__} has_ops={torchvision.extension._has_ops()}")
PY

echo "done: venv setup complete"

echo "next: source .venv/bin/activate && python scripts/inference_sample.py"
echo "note: the model repo may be gated; use huggingface-cli login or set HF_TOKEN if needed"
