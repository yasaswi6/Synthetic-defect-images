"""Step 2a - cheap synthetic defect images (no generative model), made from the SAME
9-10 training photos per defect class that the SDXL LoRAs were trained on.

  rich_aug   : small rotation (+-10 deg), zoom (0.9-1.1), shift, colour/brightness jitter,
               occasional blur, random flip - the kind of augmentation used in last
               semester's classifier training
  cutmix     : paste a random rectangle (25-50% of the image) from one photo into
               another photo of the SAME defect class
  transplant : cut the pin/joint strip out of a defect photo and blend it onto a random
               OK photo with Poisson (seamless) blending, so the defect sits on a new
               background and lighting

50 images per method and defect class -> <UROP_data>/te_work/cheap/<method>/<class>/*.png
plus te_outputs/te_cheap.csv listing them, and a preview picture.
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402
WORK = config.TE_WORK          # big files (features, generated images) live outside OneDrive
os.makedirs(WORK, exist_ok=True)
OUT = os.environ.get("TE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "te_outputs"))
N_PER_CLASS = 50
NG = ["Misalign", "NoSolder", "LowSolder", "Single"]
rng = np.random.default_rng(42)

pool = pd.read_csv(os.path.join(OUT, "te_pool.csv"))
train = pool[pool.lora_train & ~pool.conflict]
load = lambda p: Image.open(p).convert("RGB").resize((299, 299), Image.BICUBIC)


def rich_aug(img):
    w, h = img.size
    ang = rng.uniform(-10, 10)
    sc = rng.uniform(0.9, 1.1)
    tx, ty = rng.uniform(-0.05, 0.05, 2) * w
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += (tx, ty)
    # reflect the borders so rotation/zoom leaves no black corners
    a = cv2.warpAffine(np.asarray(img), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    img = Image.fromarray(a)
    for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enh(img).enhance(rng.uniform(0.8, 1.2))
    hsv = np.asarray(img.convert("HSV")).copy()
    hsv[..., 0] = (hsv[..., 0].astype(int) + int(rng.uniform(-0.05, 0.05) * 255)) % 256
    img = Image.fromarray(hsv, "HSV").convert("RGB")
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def cutmix(a, b):
    a, b = np.asarray(a).copy(), np.asarray(b)
    h, w = a.shape[:2]
    area = rng.uniform(0.25, 0.5) * h * w
    ratio = rng.uniform(0.5, 2)
    rh, rw = int(min(h, np.sqrt(area * ratio))), int(min(w, np.sqrt(area / ratio)))
    y, x = rng.integers(0, h - rh + 1), rng.integers(0, w - rw + 1)
    a[y:y + rh, x:x + rw] = b[y:y + rh, x:x + rw]
    return Image.fromarray(a)


def transplant(src, dst):
    s = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
    d = cv2.cvtColor(np.asarray(dst), cv2.COLOR_RGB2BGR)
    h, w = s.shape[:2]
    x0, x1 = int(0.15 * w), int(0.65 * w)        # vertical strip that holds the pin and joint
    y0, y1 = int(0.03 * h), int(0.97 * h)
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    centre = ((x0 + x1) // 2, (y0 + y1) // 2)
    out = cv2.seamlessClone(s, d, mask, centre, cv2.NORMAL_CLONE)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


ok_paths = train[train.label == "OK"].path.tolist()
rows = []
for method in ["rich_aug", "cutmix", "transplant"]:
    for c in NG:
        paths = train[train.label == c].path.tolist()
        d = os.path.join(WORK, "cheap", method, c)
        os.makedirs(d, exist_ok=True)
        for k in range(N_PER_CLASS):
            a = load(paths[rng.integers(len(paths))])
            if method == "rich_aug":
                img = rich_aug(a)
            elif method == "cutmix":
                b = load(paths[rng.integers(len(paths))])
                img = cutmix(a, b)
            else:
                img = transplant(a, load(ok_paths[rng.integers(len(ok_paths))]))
            p = os.path.join(d, f"{k}.png")
            img.save(p)
            rows.append(dict(version=method, label=c, path=p))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "te_cheap.csv"), index=False)

# preview: 4 per class for each method
T = 130
sheet = Image.new("RGB", (3 * 4 * T + 2 * 12, len(NG) * T), "white")
df = pd.DataFrame(rows)
for r, c in enumerate(NG):
    for m_i, m in enumerate(["rich_aug", "cutmix", "transplant"]):
        for k, p in enumerate(df[(df.version == m) & (df.label == c)].path[:4]):
            sheet.paste(Image.open(p).resize((T - 4, T - 4)), (m_i * (4 * T + 12) + k * T, r * T + 2))
sheet.save(os.path.join(OUT, "te_cheap_preview.png"))
print("made", len(rows), "images")
