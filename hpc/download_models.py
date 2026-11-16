"""Download the model weights the GPU jobs need into $HF_HOME. Run on the LOGIN node
(compute nodes usually have no internet). Safe to re-run: finished files are skipped.

  SDXL 1.0 base (fp16 weights only, ~7 GB)          stabilityai/stable-diffusion-xl-base-1.0
  SDXL VAE that is stable in fp16 (~0.3 GB)          madebyollin/sdxl-vae-fp16-fix
  ViT-base in21k, the previous team's detector       google/vit-base-patch16-224-in21k
"""
from huggingface_hub import snapshot_download

MODELS = {
    "stabilityai/stable-diffusion-xl-base-1.0": [
        "model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
        "text_encoder/config.json", "text_encoder_2/config.json", "unet/config.json", "vae/config.json",
        "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors",
    ],
    "madebyollin/sdxl-vae-fp16-fix": ["config.json", "diffusion_pytorch_model.safetensors"],
    "google/vit-base-patch16-224-in21k": ["*.json", "*.safetensors"],
}

for repo, patterns in MODELS.items():
    print("downloading", repo, flush=True)
    path = snapshot_download(repo, allow_patterns=patterns)
    print("  ->", path, flush=True)
print("All models downloaded.")
