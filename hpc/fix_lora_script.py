"""Patch two lines of diffusers' SDXL LoRA training script (v0.31.0) in hpc/third_party/.
Safe to run many times (lines already patched are left alone).

1. The script asks for the "fp16" variant of EVERY model, including the separate VAE given with
   --pretrained_vae_model_name_or_path. madebyollin/sdxl-vae-fp16-fix has no fp16 variant, so
   loading failed with "... does not appear to have a file named diffusion_pytorch_model.fp16.bin".
   Fix: use the variant only for the VAE inside the SDXL repo.
2. After saving the LoRA, the script reloads it to check it, without naming the file. With no
   internet (HF_HUB_OFFLINE=1 on the GPU nodes) that fails with "When using the offline mode,
   you must specify a `weight_name`." Fix: name the file it has just saved.

  python hpc/fix_lora_script.py
"""
import os
import sys

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "train_text_to_image_lora_sdxl.py")
PATCHES = [
    ("separate VAE loads without the fp16 variant",
     '''        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )''',
     '''        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant if args.pretrained_vae_model_name_or_path is None else None,  # patched by fix_lora_script.py
    )'''),
    ("final LoRA reload names its file (works offline)",
     '''        pipeline.load_lora_weights(args.output_dir)
''',
     '''        pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")  # patched
'''),
]

src = open(SCRIPT).read()
for what, old, new in PATCHES:
    if new in src:
        print(f"LoRA script already patched: {what}")
    elif src.count(old) == 1:
        src = src.replace(old, new)
        print(f"LoRA script patched: {what}")
    else:
        sys.exit(f"!! could not find the lines for '{what}' in {SCRIPT}; is it diffusers v0.31.0?")
open(SCRIPT, "w").write(src)
