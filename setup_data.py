"""Download everything the experiments need into config.DATA_DIR (outside OneDrive).

  python setup_data.py te       model weights + previous team's repo with TE photos (~1.5 GB)
  python setup_data.py public   model weights + RIAWELC and Magnetic Tile datasets (~0.7 GB)
  python setup_data.py mvtec    MVTec AD transistor + cable from Hugging Face (~0.9 GB)
  python setup_data.py all      everything

Safe to re-run: anything already downloaded is skipped.
Uses git if it is installed, otherwise GitHub's zip downloads.
RIAWELC comes as a multi-part RAR archive: on Windows install 7-Zip (https://www.7-zip.org)
and this script will use it.
"""
import glob
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import config


def download(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("  already have", dst)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    print("  downloading", url)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print(f"\r    {done / 2**20:.0f}" + (f" / {total / 2**20:.0f}" if total else "") + " MB", end="", flush=True)
    print()
    os.replace(tmp, dst)


def get_repo(owner_repo, branch, dst):
    if os.path.isdir(dst) and os.listdir(dst):
        print("  already have", dst)
        return
    git = shutil.which("git")
    if git:
        print("  git clone", owner_repo, "->", dst)
        if subprocess.run([git, "clone", "--depth", "1", "-b", branch,
                           f"https://github.com/{owner_repo}.git", dst]).returncode == 0:
            return
        shutil.rmtree(dst, ignore_errors=True)
        print("  git failed, trying the zip download instead")
    z = dst + ".zip"
    download(f"https://github.com/{owner_repo}/archive/refs/heads/{branch}.zip", z)
    print("  unzipping", z)
    tmp = dst + "_tmp"
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    inner = os.path.join(tmp, os.listdir(tmp)[0])
    shutil.move(inner, dst)
    shutil.rmtree(tmp, ignore_errors=True)
    os.remove(z)


def find_7zip():
    for c in [shutil.which("7z"), shutil.which("7za"), r"C:\Program Files\7-Zip\7z.exe",
              r"C:\Program Files (x86)\7-Zip\7z.exe"]:
        if c and os.path.exists(c):
            return c
    return None


def mvtec(categories=("transistor", "cable")):
    """MVTec AD categories from the Hugging Face copy (same folders as the original, incl. masks;
    CC BY-NC-SA 4.0, non-commercial). Used because mvtec.com / mydrive.ch may not load from Singapore."""
    from huggingface_hub import snapshot_download
    dst = os.path.join(config.DATA_DIR, "mvtec")
    todo = [c for c in categories if not os.path.isdir(os.path.join(dst, c, "test"))]
    if not todo:
        print("  already have", ", ".join(categories))
        return
    print("  downloading MVTec AD:", ", ".join(todo), "(about 0.9 GB, from huggingface.co/datasets/foersben/mvtec-ad)")
    snapshot_download("foersben/mvtec-ad", repo_type="dataset", local_dir=dst,
                      allow_patterns=[f"{c}/*" for c in todo])
    for c in todo:
        n = sum(len(f) for _, _, f in os.walk(os.path.join(dst, c)))
        print(f"  {c}: {n} files")


def weights():
    print("Model weights")
    download(config.INCEPTION_URL, config.INCEPTION_FILE)
    download(config.RESNET_URL, config.RESNET_FILE)


def te():
    print("Previous team's repository (TE photos + SDXL outputs), about 1.4 GB")
    get_repo("TPH777/NUSxTE_UROP", "main", config.TE_REPO)


def public():
    print("Magnetic Tile Defect dataset")
    get_repo("abin24/Magnetic-tile-defect-datasets.", "master", config.MTD_DIR)
    print("RIAWELC weld X-ray dataset")
    src = os.path.join(config.DATA_DIR, "RIAWELC_repo")
    get_repo("stefyste/RIAWELC", "main", src)
    if os.path.isdir(config.RIAWELC_DIR) and len(os.listdir(config.RIAWELC_DIR)) >= 4:
        print("  already extracted", config.RIAWELC_DIR)
        return
    parts = sorted(glob.glob(os.path.join(src, "Dataset_partitioned", "*.rar")))
    out = os.path.join(config.DATA_DIR, "riawelc")
    os.makedirs(out, exist_ok=True)
    seven = find_7zip()
    if seven:
        print("  extracting with 7-Zip")
        subprocess.run([seven, "x", parts[0], f"-o{out}", "-y"], check=False, stdout=subprocess.DEVNULL)
    elif sys.platform.startswith("linux"):
        print("  extracting with libarchive")
        subprocess.run([sys.executable, os.path.join(config.CODE_DIR, "tools", "unrar_multi.py"),
                        os.path.dirname(parts[0]), out], check=False)
    if not (os.path.isdir(config.RIAWELC_DIR) and len(os.listdir(config.RIAWELC_DIR)) >= 4):
        print("\n  !! Could not extract RIAWELC. Install 7-Zip from https://www.7-zip.org, then run\n"
              "     python setup_data.py public   again (or right-click RIAWELC_dataset.part01.rar in\n"
              f"     {os.path.dirname(parts[0])}\n"
              f"     -> 7-Zip -> Extract files... to {out}).")
    else:
        print("  extracted to", config.RIAWELC_DIR)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "te"
    print("Data folder:", config.DATA_DIR)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    config.keep_awake(True)          # Windows: no sleep while downloading
    try:
        weights()
        if what in ("te", "all"):
            te()
        if what in ("public", "all"):
            public()
        if what in ("mvtec", "bench", "all"):
            print("MVTec AD (transistor, cable)")
            mvtec()
    finally:
        config.keep_awake(False)
    print("Done.")
