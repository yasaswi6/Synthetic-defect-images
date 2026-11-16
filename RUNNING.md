# How to run the experiments

The report itself is in [README.md](README.md). This page explains how to reproduce it.

Everything below runs on a normal laptop CPU (no GPU needed). The GPU pipeline for the NUS Vanda cluster has its own guide in [hpc/README.md](hpc/README.md).

TE's photos are **not** in this repository (they belong to the industry partner). `setup_data.py te` downloads them from the previous UROP team's repository; you need access to it.

Three parts:

| Part | What it does | Time on a laptop |
| --- | --- | --- |
| `te` | Steps 1–2: re-scores last semester's SDXL images on TE's photos, makes cheap fakes and compares everything (results R4–R6 in the report) | about 20–40 min |
| `public` | Results R1–R3: FID noise, 15-image tests and split leakage on two public datasets | about 30–60 min |
| `bench` | Result R7: the same test on MVTec AD (transistor, cable); run `run_step4.bat` after downloading `transistor.tar.xz` and `cable.tar.xz` from mvtec.com into `UROP_data\downloads` | about 10 min |

Big files (datasets, model weights, cached features, ~4 GB) go to **`C:\Users\<you>\UROP_data`**, outside OneDrive,
so nothing large (and none of TE's photos) is uploaded to the cloud. To change that, edit `DATA_DIR` in `config.py`.

---

## 1. One-time setup (about 15 minutes)

1. **Install Python 3.11** from https://www.python.org/downloads/ – on the first installer screen tick
   **"Add python.exe to PATH"**. (Python 3.10–3.12 all work.)
2. **Open PowerShell in this folder**: in File Explorer open `UROP\urop_code`, click the address bar,
   type `powershell` and press Enter.
3. **Create a Python environment outside OneDrive and install the packages** (copy line by line):
   ```powershell
   python -m venv $HOME\UROP_data\venv
   & $HOME\UROP_data\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   If PowerShell says running scripts is disabled, run this once and try the second line again:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
4. **Download the data and model weights**:
   ```powershell
   python setup_data.py te        # TE photos + SDXL outputs (~1.5 GB)
   python setup_data.py public    # optional: public datasets for Results 1-3 (~0.7 GB)
   ```
   The `public` part needs **7-Zip** (https://www.7-zip.org) to unpack the RIAWELC archive.

## 2. Run

Every time you open a new PowerShell window, activate the environment first:
```powershell
& $HOME\UROP_data\venv\Scripts\Activate.ps1
python run_all.py te          # Steps 1-2 on TE data
python run_all.py public      # Results 1-3 on public data
```
Progress prints as it goes; a full log is saved in `outputs\logs`. Keep the charger plugged in
(laptops slow down on battery) and stop the laptop sleeping:
`powercfg /change standby-timeout-dc 0` and `powercfg /change standby-timeout-ac 0`.

## 3. Where the results are

| Part | Folder | Main files |
| --- | --- | --- |
| `te` | `te\te_outputs` | `te_downstream_all.csv` (classifier gains, 95% CIs), `te_quality_all.csv` (recognised / KID / variety), `te_pool.csv` (photo audit), `te_real_vs_sdxl_v6.png`, `te_cheap_fakes.png` |
| `public` | `outputs\results`, `outputs\figures` | `exp1_*` FID noise, `exp2_*` 15-image tests, `exp3_*` split leakage |

Your numbers should match the report to within about a point. Small differences can come from
different library versions or CPUs.

---

## What each script does

| Script | Purpose |
| --- | --- |
| `config.py` | All folder locations |
| `setup_data.py` | Downloads weights and datasets (git if installed, otherwise zip) |
| `run_all.py` | Runs every step in order and logs it |
| `te/te_build_pool.py` | De-duplicates the TE photos across the old dataset folders; flags conflicting labels and split leaks |
| `te/te_features.py` | Turns every image into numbers (Inception for FID/KID, ResNet-50 for the classifier) |
| `te/te_analysis.py` | Checks the SDXL images (recognised as their class, KID, near-copies) and runs the classifier test |
| `te/te_diversity.py` | How varied each image set is |
| `te/te_cheap_synth.py` | Makes the cheap fakes: stronger augmentation, cut-and-mix, joint transplant |
| `te/te_cheap_eval.py` | Scores the cheap fakes and compares every method side by side |
| `te/te_filter_test.py` | Result R6: keep only the fakes a judge accepts, then re-run the test |
| `te/te_pictures.py` | Side-by-side pictures of real and fake joints (not included here: they show TE's parts) |
| `bench/bench_prepare.py`, `bench/bench_run.py` | Result R7: index MVTec AD, then run the same test with mask paste added |
| `hpc/` | GPU pipeline for NUS Vanda (see `hpc/README.md`) |
| `extract_features.py` | Features for the public datasets |
| `exp1_fid_reliability.py` | How much FID moves by chance at 10–150 images |
| `exp2_fewshot_protocol.py` | Re-creates the 100 OK + 10 NG protocol; how noisy 15-image tests are |
| `exp3_split_leakage.py` | Random vs grouped-by-radiograph split on RIAWELC |
| `tools/unrar_multi.py` | Linux-only RAR unpacker (Windows uses 7-Zip) |

## How the test works (in short)

* **Same data as last semester:** the classifier trains on exactly the photos the SDXL LoRAs were trained on
  (100 OK + 9–10 per defect class), then each method adds 15 extra images per defect class.
* **Bigger test:** it is tested on every other usable TE photo (436), not 15 per class.
* **Different classifier:** a frozen ImageNet ResNet-50 turns each image into 2,048 numbers and a logistic
  regression learns from them. It is much faster than fine-tuning last semester's ViT, so each comparison can be
  repeated 50 times with 95% confidence intervals. All methods go through the same classifier, so the comparison is fair,
  but the absolute accuracies are not the same as last semester's.

## Troubleshooting

* `python` not found: reinstall Python with "Add python.exe to PATH" ticked, then open a new PowerShell window.
* A step fails: the log in `outputs\logs` shows where; re-running `run_all.py` repeats the steps from the start.
* Out of disk space: the data folder needs about 4 GB free.
