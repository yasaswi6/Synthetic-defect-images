#!/bin/bash
# First login: collect the facts the job scripts need. Run on the Vanda login node:
#     bash ~/urop_code/hpc/check_vanda.sh | tee ~/vanda_check.txt
# The output is saved in ~/vanda_check.txt.
source "$(dirname "$0")/paths.sh"

line() { echo; echo "===== $1"; }

line "who / where"
echo "user=$USER  host=$(hostname)  home=$HOME"
date

line "storage (home quota, free space)"
hpc s 2>/dev/null || echo "(hpc s not available)"
df -h "$HOME" 2>/dev/null | tail -1
df -h "/scratch/$USER" 2>/dev/null | tail -1 || echo "no /scratch/$USER"

line "job queues (look for one with 'gpu' in its name)"
qstat -q 2>/dev/null || echo "qstat not found - is the scheduler Slurm? trying sinfo:"
command -v sinfo >/dev/null && sinfo -s

line "modules for python / conda / cuda / containers"
module avail 2>&1 | tr -s ' ' '\n' | grep -iE "conda|forge|python|cuda|singularity|apptainer" | sort -u | head -40

line "python on the login node"
python3 --version 2>&1

line "internet from the login node (needed to install packages and download models)"
for u in https://pypi.org https://download.pytorch.org/whl/cu121 https://huggingface.co https://github.com; do
    code=$(curl -s -o /dev/null -m 15 -w "%{http_code}" "$u")
    echo "$u -> $code"
done

line "GPU visible here? (normally none on the login node)"
nvidia-smi -L 2>&1 | head -5

line "done"
