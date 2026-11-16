# TE data experiments (results R4–R6 in the report)

Run everything with `python run_all.py te` from the parent folder (see `..\RUNNING.md` for setup).
The data is the previous team's repository, https://github.com/TPH777/NUSxTE_UROP, which
`setup_data.py te` downloads to `C:\Users\<you>\UROP_data\NUSxTE_UROP`.

Step by step, if you prefer (from the parent folder):
```
python te/te_build_pool.py                 # de-duplicate photos, audit labels and splits
python te/te_features.py inception         # then: resnet, aug, pack
python te/te_analysis.py                   # SDXL quality + classifier test
python te/te_diversity.py                  # variety within each class
python te/te_cheap_synth.py                # cheap fakes
python te/te_cheap_eval.py features        # then: eval
python te/te_pictures.py                   # pictures for the doc
```

Small results land in `te_outputs/`; features and the cheap fake images go to `UROP_data\te_work`.

| File | Contents |
| --- | --- |
| `te_pool.csv` | Every unique real photo: label, conflicts, which folders/splits it sits in (made when you run it; not included here) |
| `te_class_fidelity*.csv`, `te_kid_fid.csv`, `te_memorisation*.csv`, `te_diversity.csv` | SDXL image quality (Step 1) |
| `te_downstream.csv` | Step 1 classifier test |
| `te_quality_all.csv`, `te_downstream_all.csv` | Step 2: cheap fakes and SDXL side by side |
| `te_filter_survivors.csv`, `te_filter_downstream.csv` | Judge-filtered SDXL fakes (`te_filter_test.py`) |
| `te_real_vs_sdxl_v6.png`, `te_cheap_fakes.png` | Pictures (made when you run it; not included here because they show TE's parts) |

Key numbers (436 held-out photos; gain in macro accuracy over the 62.4% baseline):
about 10 extra real photos +10.2 (95% CI +4.4 to +16.4), stronger augmentation +4.8, class weights +4.4,
flip/rotate +4.2, joint transplant +3.6, cut-and-mix +2.3, SDXL v4–v7 +1.8 to +3.1. Only the real photos'
interval is clearly above zero (class weights' lower bound is +0.2 in this run and -0.3 in a repeat run).
