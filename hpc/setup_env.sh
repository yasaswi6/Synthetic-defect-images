#!/bin/bash
# One-time setup on the Vanda LOGIN node (needs internet). Takes ~30-40 minutes.
#     bash ~/urop_code/hpc/setup_env.sh 2>&1 | tee ~/setup_env.log
# Safe to re-run: steps that are already done are skipped.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/paths.sh"
mkdir -p "$UROP_RESULTS" "$UROP_WORK" "$HF_HOME" "$PIP_CACHE_DIR" "$(dirname "$UROP_ENV")" "$HERE/third_party"

echo "== 1/5 Python environment at $UROP_ENV"
if [ -d "$UROP_ENV/conda-meta" ] || [ -f "$UROP_ENV/bin/activate" ]; then
    echo "   already exists"
elif load_conda_module; then
    echo "   using conda ($(conda --version))"
    conda create -y -p "$UROP_ENV" -c conda-forge --override-channels python=3.11 || {
        echo "   conda create failed, falling back to a venv with $(python3 --version)"; rm -rf "$UROP_ENV"; python3 -m venv "$UROP_ENV"; }
else
    PY=$(command -v python3.11 || command -v python3.10 || command -v python3)
    echo "   conda module not found, using $($PY --version) venv"
    "$PY" -m venv "$UROP_ENV"
fi
activate_env
python -m pip install --upgrade pip

echo "== 2/5 PyTorch with CUDA 12.1"
python -c "import torch" 2>/dev/null && echo "   already installed" || \
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo "== 3/5 the other packages"
pip install -r "$HERE/requirements_vanda.txt"

echo "== 4/5 diffusers' SDXL LoRA training script (v0.31.0, the version installed above)"
SCRIPT="$HERE/third_party/train_text_to_image_lora_sdxl.py"
[ -f "$SCRIPT" ] || curl -fsSL -o "$SCRIPT" \
    https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/examples/text_to_image/train_text_to_image_lora_sdxl.py
python "$HERE/fix_lora_script.py"
echo "   $SCRIPT"

echo "== 5/5 model weights into $HF_HOME (~8 GB)"
python "$HERE/download_models.py"

python - <<'EOF'
import torch, diffusers, transformers, peft, accelerate
print("OK  torch", torch.__version__, "| diffusers", diffusers.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| accelerate", accelerate.__version__)
EOF
echo
echo "Setup finished. Next: get the TE photos and make the splits (README step 4)."
