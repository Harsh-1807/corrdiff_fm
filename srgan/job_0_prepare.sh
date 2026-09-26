#!/bin/bash
#PBS -N prep_srgan
#PBS -q workq
#PBS -l select=1:ncpus=4:ngpus=0
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o log_0_prepare_srgan.txt

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
# THIS STEP IS SHARED INFRASTRUCTURE ACROSS EVERY PACKAGE IN
# /home/ylale/extras/h/ (corrdiff_fm's edm/fm arms, this srgan package, and the
# wgan_gp/srdrn siblings). Run it ONCE, from any one package's directory --
# they all point Config.HR_FILES at the same corrected file on disk. Running
# it again here is harmless (it detects the corrected file already exists and
# only re-validates) but is not required if you have already run it in
# corrdiff_fm or a sibling package.
# ------------------------------------------------------------
python PrepareData.py

RC=$?
echo "exit code: ${RC}"
echo "=== Data preparation :: finished at $(date) ==="
exit ${RC}
