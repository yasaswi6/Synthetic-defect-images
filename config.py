"""Where everything lives. Edit DATA_DIR here (or set the UROP_DATA environment variable)
if you want the big files somewhere else.

  CODE_DIR    this folder: scripts, small result tables and figures
  DATA_DIR    downloaded datasets, model weights, cached features, generated images
              (about 4 GB in total). Defaults to a folder OUTSIDE OneDrive so the
              big files and TE's photos are not uploaded to the cloud:
              C:\\Users\\<you>\\UROP_data on Windows, ~/UROP_data on Mac/Linux.
"""
import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("UROP_DATA", os.path.join(os.path.expanduser("~"), "UROP_data"))
WEIGHTS_DIR = os.environ.get("UROP_WEIGHTS", os.path.join(DATA_DIR, "weights"))

# datasets (created by setup_data.py)
TE_REPO = os.path.join(DATA_DIR, "NUSxTE_UROP")                      # previous team's repo (TE photos + SDXL outputs)
RIAWELC_DIR = os.path.join(DATA_DIR, "riawelc", "DB - Copy", "testing")
MTD_DIR = os.path.join(DATA_DIR, "mtd")

# model weights (downloaded by setup_data.py from GitHub releases)
INCEPTION_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"
RESNET_URL = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth"
INCEPTION_FILE = os.path.join(WEIGHTS_DIR, "hub", "checkpoints", "pt_inception-2015-12-05-6726825d.pth")
RESNET_FILE = os.path.join(WEIGHTS_DIR, "resnet50_a1_0-14fe96d1.pth")
TE_WORK = os.path.join(DATA_DIR, "te_work")                          # TE features + cheap fake images


def keep_awake(on=True):
    """Windows: stop the laptop sleeping (and the screen switching off) while our scripts run.
    On many laptops the screen turning off sends the whole machine to standby, so both are held.
    Undone automatically when the script ends. Does nothing on Mac/Linux."""
    import sys
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
    flags = ES_CONTINUOUS | ((ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)
