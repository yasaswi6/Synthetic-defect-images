#!/bin/bash
# Submit Week 1's GPU runs in the right order; later jobs wait for the ones they need.
#
#   bash ~/urop_code/hpc/submit_week1.sh <queue> smoke [project]   tiny version of every step (~30-45 min)
#                                                                 -> catches setup problems cheaply
#   bash ~/urop_code/hpc/submit_week1.sh <queue> full  [project]   the real runs, all 5 folds
#                                                                 (~12-15 GPU-hours, spread over several jobs)
# <queue>: a queue name, or  none  to leave the queue out and let Vanda pick it from the project
#          and the GPU request (Vanda refuses some queues when you name them yourself).
# [project]: the code that  hpc project  shows, e.g. personal-<nusnet-id>.
set -e
Q=$1
MODE=${2:-smoke}
PROJ=${3:-}
[ -z "$Q" ] && { echo "usage: bash submit_week1.sh <queue|none> smoke|full [project]"; exit 1; }
cd "$(dirname "$0")/jobs"
QS=(qsub)
[ "$Q" != "none" ] && QS+=(-q "$Q")
[ -n "$PROJ" ] && QS+=(-P "$PROJ")
sub() { local id; id=$("${QS[@]}" "$@") || exit 1; echo "$id"; }

if [ "$MODE" = "smoke" ]; then
    # short time limits: short jobs get a free GPU sooner
    W="-l walltime=00:30:00"
    l=$(sub $W -v FOLD=0,CLASSES=Misalign,STEPS=50 20_lora_train.pbs)
    g=$(sub $W -W depend=afterok:$l -v FOLD=0,CLASSES=Misalign,N=4 21_generate.pbs)
    i=$(sub $W -W depend=afterok:$l -v FOLD=0,CLASSES=Misalign,N=4,MODE=img2img 21_generate.pbs)
    v=$(sub $W -v FEWSHOT=10,FOLDS=0,EPOCHS=2,TAG=smoke 10_vit.pbs)
    e=$(sub $W -W depend=afterok:$g -v FEWSHOT=10,FOLDS=0,EPOCHS=2,EXTRA=gen/txt2img/f{fold},TAG=smoke_extra 10_vit.pbs)
    echo "smoke jobs: lora $l | txt2img $g | img2img $i | vit $v | vit+fakes $e"
elif [ "$MODE" = "full" ]; then
    # Short time limits get a free GPU sooner; each limit is ~2x the job's expected time on an A40.
    WV="-l walltime=02:00:00"   # detector, 5 folds x 20 epochs (~30-60 min)
    WL="-l walltime=01:30:00"   # one LoRA, 1500 steps (~40 min)
    WG="-l walltime=01:00:00"   # 50 fakes x 4 classes (~15-20 min)
    b1=$(sub $WV -v FEWSHOT=0,TAG=real_all 10_vit.pbs)
    b2=$(sub $WV -v FEWSHOT=10,TAG=real_fewshot10 10_vit.pbs)
    # One LoRA job per fold AND class (20 jobs), so they can run side by side on different GPUs.
    # img2img at three strengths (0.3 = small changes ... 0.6 = what the FYP team uses), txt2img as the control
    T="" I30="" I45="" I60=""
    for k in 0 1 2 3 4; do
        L=""
        for c in Misalign NoSolder LowSolder Single; do
            L="$L:$(sub $WL -v FOLD=$k,CLASSES=$c 20_lora_train.pbs)"
        done
        D="-W depend=afterok$L"   # this fold's fakes wait for all 4 of its LoRAs
        I30="$I30:$(sub $WG $D -v FOLD=$k,MODE=img2img,STRENGTH=0.3 21_generate.pbs)"
        I45="$I45:$(sub $WG $D -v FOLD=$k,MODE=img2img,STRENGTH=0.45 21_generate.pbs)"
        I60="$I60:$(sub $WG $D -v FOLD=$k,MODE=img2img,STRENGTH=0.6 21_generate.pbs)"
        T="$T:$(sub $WG $D -v FOLD=$k 21_generate.pbs)"
        echo "fold $k: LoRA jobs ${L#:}"
    done
    for s in 30 45 60; do
        dep_var="I$s"
        sub $WV -W depend=afterok${!dep_var} -v FEWSHOT=10,EXTRA=gen/img2img$s/f{fold},TAG=fewshot10+sdxl_img2img$s 10_vit.pbs
    done
    v1=$(sub $WV -W depend=afterok$T -v FEWSHOT=10,EXTRA=gen/txt2img/f{fold},TAG=fewshot10+sdxl_txt2img 10_vit.pbs)
    echo "baselines: $b1 $b2 | txt2img: $v1 | img2img detector runs listed above"
else
    echo "second argument must be smoke or full"; exit 1
fi
echo "Watch them with:  qstat -u $USER"
