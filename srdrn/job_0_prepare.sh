#!/bin/bash
#PBS -N prep_srdrn
#PBS -q workq
#PBS -l select=1:ncpus=4:ngpus=0
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o log_0_prepare_srdrn.txt

echo "=== Data preparation :: started on $(hostname) at $(date) ==="

# `set -u` catches unset variables; `pipefail` stops a failing python from being
# masked by a succeeding grep/tee further down a pipeline.
set -u
set -o pipefail

# ------------------------------------------------------------
# Environment -- EDIT THESE TWO LINES FOR YOUR CLUSTER
# ------------------------------------------------------------
source /lustre/home/hpc/bipink/anaconda3/etc/profile.d/conda.sh
conda activate jannu

cd "${PBS_O_WORKDIR:-$PWD}"
echo "Workdir: $PWD"
echo "Python:  $(which python)"
echo "Torch:   $(python -c 'import torch; print(torch.__version__)')"

# ------------------------------------------------------------
# Threads / CUDA / debuggability
# ------------------------------------------------------------
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------------------------------------------------------
# Corrects the precip units at the source and runs every pre-flight check.
# Cheap, CPU-only, and catches the failure modes that would otherwise produce a
# trained model rather than an error message.
#
# THIS STEP IS SHARED INFRASTRUCTURE ACROSS EVERY PACKAGE UNDER
# /home/ylale/extras/h/ (corrdiff_fm's edm/fm arms, this srdrn package, and any
# sibling wgan_gp/srgan-style package). It only needs to be run ONCE, in
# whichever package you run it first -- every package points its Config.HR_FILES
# at the same corrected file on the same ROOT. Do NOT run this once per
# package "just in case"; it is harmless but wasteful, and if you ever see it
# produce a DIFFERENT corrected file across packages that is a sign ROOT has
# diverged between them, which is worth investigating rather than papering over.
# ------------------------------------------------------------
python PrepareData.py

RC=$?
echo "exit code: ${RC}"
echo "=== Data preparation :: finished at $(date) ==="
exit ${RC}
