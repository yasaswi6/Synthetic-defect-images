# Running the UROP experiments on NUS Vanda

## What Week 1 does

1. **Honest baseline.** It trains last semester's detector (ViT) on TE's photos and tests it on photos it has never seen.
   - It uses 5 folds, grouped by capture session, so near-identical burst photos can't leak between training and test.
   - It runs twice: once with all real photos, and once in the AITK situation, where there are at most 10 real photos per defect type.
2. **Fakes.** For each fold, it re-trains one SDXL LoRA per defect class with last semester's v6 recipe, then makes fakes in two ways:
   - **txt2img:** what last semester did.
   - **img2img:** variations of the real photos ("approach B").
3. **The question.** Does adding those fakes make the 10-photo detector better? Each fake run is compared with the 10-photo baseline.

## Steps

Replace `<nusnet-id>` below with your NUSNET ID. If you're off campus, connect to the NUS VPN first.

**1. Copy the code from your laptop.** In PowerShell on the laptop:
```powershell
scp -r "<path to this repo>" <nusnet-id>@vanda.nus.edu.sg:~/urop_code
```

**2. Log in.**
```bash
ssh <nusnet-id>@vanda.nus.edu.sg
```

**3. Check how Vanda is set up.** The output is saved in `~/vanda_check.txt`. It shows the GPU queue name, the conda module, and whether there's internet access.
```bash
bash ~/urop_code/hpc/check_vanda.sh | tee ~/vanda_check.txt
```

**4. One-time setup** (about 30–40 minutes, on the login node).
- If `tmux` exists on Vanda, run this inside `tmux new -s setup`, so a dropped connection doesn't stop it. Detach with Ctrl+B then D, and reattach with `tmux attach -t setup`.
```bash
bash ~/urop_code/hpc/setup_env.sh 2>&1 | tee ~/setup_env.log
```

**5. Get TE's photos and make the splits.**
```bash
source ~/urop_code/hpc/paths.sh && activate_env
cd ~/urop_code
python setup_data.py te      # last team's repo, with TE's photos, into ~/UROP_data
python te/te_build_pool.py   # de-duplicated list of the 584 photos
python hpc/make_splits.py    # 5 session-grouped folds -> hpc/splits/te_splits.csv
```

**6. GPU test** (2 minutes, plus the wait for a free GPU). Don't name a queue: Vanda refuses `-q gpu` for student accounts and routes the job itself (to `batch_gpu`, one A40 with 36 cores and 240 GB per GPU).
```bash
cd ~/urop_code/hpc/jobs
qsub 00_gpu_test.pbs
qstat -u $USER                 # wait until it disappears from the list
cat urop_gpu_test.o*           # should say "CUDA available: True" and name the GPU
```

**7. Smoke test** (about 30–45 minutes). A tiny version of every step, to catch problems cheaply.
```bash
bash ~/urop_code/hpc/submit_week1.sh none smoke
```
- When the jobs finish, the logs are in `~/urop_results/logs/`.
- If anything failed, the end of that job's log shows the error (`tail -30 <log file>`).

**8. Full run** (about 12–15 GPU-hours, split across separate jobs).
```bash
bash ~/urop_code/hpc/submit_week1.sh none full
```

**9. Get the results back.** In PowerShell on the laptop:
```powershell
scp <nusnet-id>@vanda.nus.edu.sg:~/urop_results/vit_results.csv .
scp -r <nusnet-id>@vanda.nus.edu.sg:/scratch/<nusnet-id>/urop_work/gen ./gen   # the fake images, to look at
```

If `qsub` complains about a project, add `-P <project code>`; the code is in your Vanda approval email. With `submit_week1.sh`, pass it as the third argument.

## Reading `vit_results.csv`

There is one row per fold. The `tag` column says which experiment the row belongs to:

| tag | what it is |
|---|---|
| `real_all` | Detector trained on all real training photos. This is the honest version of last semester's accuracy. |
| `real_fewshot10` | At most 10 real photos per defect type. This is the baseline the fakes must beat. |
| `fewshot10+sdxl_txt2img` | The same, plus 50 SDXL fakes per defect type, made last semester's way. |
| `fewshot10+sdxl_img2img30` / `45` / `60` | The same, plus 50 variations of the real photos, changed a little (0.3) to a lot (0.6). The FYP team uses 0.6. |

`macro_acc` is the accuracy averaged over the 5 classes. `escape_rate` is the share of defects called OK, and `false_alarm` is the share of good joints called defective. A method wins if it beats `real_fewshot10` in most folds.

## Useful commands

| command | does |
|---|---|
| `bash ~/urop_code/hpc/status.sh` | everything on one screen: jobs, the live line of each running job, finished LoRAs, fake-image counts, results so far, errors in the logs (`watch -n 60 bash ~/urop_code/hpc/status.sh` to refresh every minute) |
| `qstat -u $USER` | your jobs (Q = waiting, R = running) |
| `qdel <job id>` | cancel a job |
| `qstat -f <job id>` | details of a job |
| `hpc s` | storage use: home 40 GB, /scratch 500 GB |

## Files

| file | what it does |
|---|---|
| `paths.sh` | where everything lives on Vanda; every script loads it |
| `check_vanda.sh` | first-login fact check |
| `setup_env.sh`, `requirements_vanda.txt`, `download_models.py` | Python environment and model downloads |
| `splits_lib.py`, `make_splits.py` | classes, prompts, session-grouped folds, few-shot picks |
| `vit_detector.py` | the detector test (ViT, 5 folds, optional fake images) |
| `lora_data.py` | the training folders for each class's LoRA |
| `generate.py` | txt2img and img2img generation with the LoRAs |
| `jobs/*.pbs` | the GPU jobs; settings go in with `qsub -v NAME=value,...` |
| `submit_week1.sh` | submits the smoke or full run in the right order (full: one LoRA job per fold and class, so they run side by side) |
| `status.sh` | one-screen progress check |
| `fix_lora_script.py` | patches two lines of diffusers' LoRA script so it works offline with the separate fp16-safe VAE |
| `third_party/` | diffusers' LoRA training script (downloaded by `setup_env.sh`) |

## Storage

| what | where |
|---|---|
| Code, Python environment, results | your home folder (`/home/svu/<nusnet-id>`, 40 GB) |
| Model weights, LoRAs, fake images | `/scratch/<nusnet-id>/` (500 GB) |

`/scratch` deletes files that haven't been touched for 60 days (from 13 Nov 2026). Copy anything you want to keep back to your laptop. If the model weights get deleted, re-run `setup_env.sh` to download them again.
