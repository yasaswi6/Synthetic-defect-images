"""Step 2d - the two pictures used in the doc:
  te_outputs/te_real_vs_sdxl_v6.png : 4 real photos and 4 SDXL v6 images per class
  te_outputs/te_cheap_fakes.png      : 3 examples of each cheap fake per defect class
"""
import os
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.environ.get("TE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "te_outputs"))


def font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "arial.ttf"])
    for n in names:
        for d in ["", "/usr/share/fonts/truetype/dejavu/", r"C:\Windows\Fonts\\"]:
            try:
                return ImageFont.truetype(d + n, size)
            except OSError:
                pass
    return ImageFont.load_default()


NAMES = {"OK": "OK", "Misalign": "NG Misalign", "NoSolder": "NG No solder", "LowSolder": "NG Low solder",
         "Single": "NG Single"}
pool = pd.read_csv(os.path.join(OUT, "te_pool.csv"))
gen = pd.read_csv(os.path.join(OUT, "te_generated.csv"))
cheap = pd.read_csv(os.path.join(OUT, "te_cheap.csv"))
f, fb = font(18), font(19, True)

# real vs SDXL v6
T, gap, lab, head = 150, 16, 150, 40
can = Image.new("RGB", (lab + 8 * T + gap, head + 5 * T), "white")
d = ImageDraw.Draw(can)
d.text((lab, 10), "Real TE photos", fill="black", font=fb)
d.text((lab + 4 * T + gap, 10), "SDXL v6 (one LoRA per class)", fill="black", font=fb)
for r, c in enumerate(NAMES):
    rp = pool[(pool.label == c) & (~pool.conflict)].sample(4, random_state=1).path.tolist()
    gp = gen[(gen.version == "v6") & (gen.label == c)].sample(4, random_state=1).path.tolist()
    y = head + r * T
    d.text((8, y + T // 2 - 10), NAMES[c], fill="black", font=f)
    for k, p in enumerate(rp + gp):
        can.paste(Image.open(p).convert("RGB").resize((T - 4, T - 4)), (lab + k * T + (gap if k >= 4 else 0), y + 2))
can.save(os.path.join(OUT, "te_real_vs_sdxl_v6.png"))

# cheap fakes
heads = {"rich_aug": "Stronger augmentation", "cutmix": "Cut-and-mix", "transplant": "Joint transplant"}
T, per = 120, 3
ng = [c for c in NAMES if c != "OK"]
can = Image.new("RGB", (lab + 3 * per * T + 2 * gap, head + len(ng) * T), "white")
d = ImageDraw.Draw(can)
for m_i, m in enumerate(heads):
    d.text((lab + m_i * (per * T + gap), 10), heads[m], fill="black", font=fb)
for r, c in enumerate(ng):
    y = head + r * T
    d.text((8, y + T // 2 - 10), NAMES[c], fill="black", font=f)
    for m_i, m in enumerate(heads):
        for k, p in enumerate(cheap[(cheap.version == m) & (cheap.label == c)].path[:per]):
            can.paste(Image.open(p).convert("RGB").resize((T - 4, T - 4)), (lab + m_i * (per * T + gap) + k * T, y + 2))
can.save(os.path.join(OUT, "te_cheap_fakes.png"))
print("saved te_real_vs_sdxl_v6.png and te_cheap_fakes.png")
