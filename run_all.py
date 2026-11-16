"""Run the experiments in order and keep a log of everything printed.

  python run_all.py te       Steps 1-2: re-score last semester's SDXL images + cheap fakes on TE data
                             (about 20-30 min on a laptop CPU)
  python run_all.py public   Results 1-3: FID noise, 15-image tests, split leakage on RIAWELC and
                             Magnetic Tile (about 30-60 min)
  python run_all.py bench    Step 4: the same test on MVTec AD transistor and cable (about 20-30 min)
  python run_all.py all      everything

Run  python setup_data.py te  (or public / all) first.
Logs: outputs/logs/<part>_<time>.log   Results: te/te_outputs and outputs/results, outputs/figures
"""
import datetime
import os
import subprocess
import sys
import time

import config

HERE = config.CODE_DIR
TE_STEPS = [
    ("Step 1a  de-duplicate TE photos, audit labels and splits", ["te/te_build_pool.py"]),
    ("Step 1b  Inception features (for FID/KID)", ["te/te_features.py", "inception"]),
    ("Step 1b  ResNet-50 features (for the classifier)", ["te/te_features.py", "resnet"]),
    ("Step 1b  flip/rotate views of the training photos", ["te/te_features.py", "aug"]),
    ("Step 1b  pack features", ["te/te_features.py", "pack"]),
    ("Step 1c  quality of SDXL images + downstream classifier test", ["te/te_analysis.py"]),
    ("Step 1d  within-class variety", ["te/te_diversity.py"]),
    ("Step 2a  make cheap fakes", ["te/te_cheap_synth.py"]),
    ("Step 2b  features of cheap fakes", ["te/te_cheap_eval.py", "features"]),
    ("Step 2c  score everything side by side", ["te/te_cheap_eval.py", "eval"]),
    ("Step 2d  pictures for the doc", ["te/te_pictures.py"]),
]
PUBLIC_STEPS = [(f"Features {ds}: {st}", ["extract_features.py", ds, st])
                for ds in ("riawelc", "mtd") for st in ("index", "inception", "resnet", "aug", "pack")] + [
    ("Result 1  FID/KID noise floor", ["exp1_fid_reliability.py"]),
    ("Result 2  100 OK + 10 NG protocol, 15-image tests", ["exp2_fewshot_protocol.py"]),
    ("Result 3  random vs grouped split", ["exp3_split_leakage.py"]),
]
BENCH_STEPS = [
    ("Step 4a  unpack and index the benchmark datasets", ["bench/bench_prepare.py"]),
    ("Step 4b  MVTec AD transistor: same test as Result 4", ["bench/bench_run.py", "mvtec_transistor"]),
    ("Step 4b  MVTec AD cable: same test as Result 4", ["bench/bench_run.py", "mvtec_cable"]),
]


def check(part):
    missing = []
    for f in (config.INCEPTION_FILE, config.RESNET_FILE):
        if not os.path.exists(f):
            missing.append(f)
    if part == "te" and not os.path.isdir(os.path.join(config.TE_REPO, "datasets")):
        missing.append(config.TE_REPO)
    if part == "public":
        for d in (config.RIAWELC_DIR, config.MTD_DIR):
            if not os.path.isdir(d):
                missing.append(d)
    if part == "bench":
        for c in ("transistor", "cable"):
            d = os.path.join(config.DATA_DIR, "mvtec", c, "test")
            if not os.path.isdir(d):
                missing.append(d)
        part = "mvtec"
    if missing:
        print("Missing:\n  " + "\n  ".join(missing) + f"\nRun:  python setup_data.py {part}")
        sys.exit(1)


def run(part, steps):
    check(part)
    os.makedirs(os.path.join(HERE, "outputs", "logs"), exist_ok=True)
    log_path = os.path.join(HERE, "outputs", "logs", f"{part}_{datetime.datetime.now():%Y%m%d_%H%M}.log")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    t_all = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        for i, (title, args) in enumerate(steps, 1):
            head = f"\n[{i}/{len(steps)}] {title}"
            print(head, flush=True)
            log.write(head + "\n")
            t0 = time.time()
            p = subprocess.Popen([sys.executable, os.path.join(HERE, args[0])] + args[1:], cwd=HERE, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace")
            for line in p.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if p.wait() != 0:
                print(f"\n!! Step failed: {title}. See {log_path}")
                sys.exit(1)
            msg = f"   done in {(time.time() - t0) / 60:.1f} min"
            print(msg)
            log.write(msg + "\n")
    print(f"\nAll {part} steps finished in {(time.time() - t_all) / 60:.1f} min. Log: {log_path}")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "te"
    config.keep_awake(True)          # Windows: no sleep / screen-off while this runs
    print("Keeping the laptop awake until this finishes (the screen stays on). Don't close the lid.")
    try:
        if part in ("te", "all"):
            run("te", TE_STEPS)
        if part in ("public", "all"):
            run("public", PUBLIC_STEPS)
        if part in ("bench", "all"):
            run("bench", BENCH_STEPS)
    finally:
        config.keep_awake(False)
