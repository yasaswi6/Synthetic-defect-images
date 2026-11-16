# Where everything lives on Vanda. Every job script and setup script sources this file.
# Change a line here (or export the variable before running) if you keep things elsewhere.
#
#   UROP_HOME     this code folder (copied from your laptop)
#   UROP_DATA     the previous team's repo with TE's photos (~1.4 GB)
#   UROP_ENV      the Python environment (~6 GB; in home so it is never purged)
#   UROP_RESULTS  small result tables and logs you want to keep (home: 40 GB)
#   UROP_WORK     big working files: LoRA weights, generated images (/scratch: 500 GB,
#                 but files untouched for 60 days are deleted automatically)
#   HF_HOME       downloaded model weights (SDXL ~7 GB, ViT ~0.4 GB); re-downloadable

export UROP_HOME="${UROP_HOME:-$HOME/urop_code}"
export UROP_DATA="${UROP_DATA:-$HOME/UROP_data}"
export UROP_ENV="${UROP_ENV:-$HOME/envs/urop}"
export UROP_RESULTS="${UROP_RESULTS:-$HOME/urop_results}"
export UROP_WORK="${UROP_WORK:-/scratch/$USER/urop_work}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/pip_cache}"
export PYTHONUNBUFFERED=1

# Make "module" available in batch jobs (non-interactive shells do not always have it).
for f in /etc/profile.d/rec_modules.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    [ -f "$f" ] && source "$f" >/dev/null 2>&1
done

load_conda_module() {
    for m in Anaconda3 Miniforge3 miniforge Miniconda3 miniconda anaconda; do
        module load "$m" >/dev/null 2>&1 && command -v conda >/dev/null 2>&1 && return 0
    done
    command -v conda >/dev/null 2>&1
}

activate_env() {
    if [ -d "$UROP_ENV/conda-meta" ]; then
        load_conda_module || { echo "!! conda not found (module load miniconda failed)"; return 1; }
        eval "$(conda shell.bash hook)"
        conda activate "$UROP_ENV"
    elif [ -f "$UROP_ENV/bin/activate" ]; then
        source "$UROP_ENV/bin/activate"
    else
        echo "!! No Python environment at $UROP_ENV. Run:  bash \$UROP_HOME/hpc/setup_env.sh"
        return 1
    fi
}
