"""Make fake defect photos with SDXL and the per-class LoRAs.

  txt2img (what last semester did): the LoRA draws a new photo from the class prompt
      python hpc/generate.py --lora_root $UROP_WORK/lora/f0 --out $UROP_WORK/gen/txt2img/f0
  img2img ("approach B"): start from each real training photo and ask for a variation,
  so the defect's shape survives; --strength 0.2-0.5 = how much SDXL may change it
      python hpc/generate.py --mode img2img --strength 0.35 --src_root $UROP_WORK/lora_data/f0 \
          --lora_root $UROP_WORK/lora/f0 --out $UROP_WORK/gen/img2img35/f0

Writes out/<class>/<class>_0000.png ..., the folder layout vit_detector.py --extra reads.
A class without a trained LoRA is generated from base SDXL (and says so).
"""
import argparse
import os
import sys

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from splits_lib import DEFECTS, PROMPTS  # noqa: E402

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
VAE = "madebyollin/sdxl-vae-fp16-fix"
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["txt2img", "img2img"], default="txt2img")
ap.add_argument("--lora_root", default="", help="folder with <class>/pytorch_lora_weights.safetensors")
ap.add_argument("--src_root", default="", help="img2img: folder with <class>/ real photos (lora_data.py output)")
ap.add_argument("--classes", nargs="+", default=DEFECTS)
ap.add_argument("--n", type=int, default=50, help="images per class")
ap.add_argument("--strength", type=float, default=0.35)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=7.5)
ap.add_argument("--resolution", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
if args.mode == "img2img" and not args.src_root:
    sys.exit("img2img needs --src_root")

from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline  # noqa: E402

vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(BASE, vae=vae, torch_dtype=torch.float16, variant="fp16")
if args.mode == "img2img":
    pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

for c in args.classes:
    lora = os.path.join(args.lora_root, c) if args.lora_root else ""
    has_lora = bool(lora) and os.path.isfile(os.path.join(lora, "pytorch_lora_weights.safetensors"))
    if has_lora:
        pipe.load_lora_weights(lora, weight_name="pytorch_lora_weights.safetensors", adapter_name=c)
    else:
        print(f"  {c}: no LoRA found, using base SDXL")
    sources = []
    if args.mode == "img2img":
        d = os.path.join(args.src_root, c)
        sources = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
        if not sources:
            sys.exit(f"no source photos in {d}")
    out = os.path.join(args.out, c)
    os.makedirs(out, exist_ok=True)
    gen = torch.Generator("cuda").manual_seed(args.seed)
    for i in range(args.n):
        kw = dict(prompt=PROMPTS[c], num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen)
        if args.mode == "txt2img":
            img = pipe(height=args.resolution, width=args.resolution, **kw).images[0]
        else:
            src = Image.open(sources[i % len(sources)]).convert("RGB").resize((args.resolution, args.resolution))
            img = pipe(image=src, strength=args.strength, **kw).images[0]
        img.save(os.path.join(out, f"{c}_{i:04d}.png"))
    print(f"  {c}: {args.n} images -> {out}", flush=True)
    if has_lora:
        pipe.unload_lora_weights()
print("done")
