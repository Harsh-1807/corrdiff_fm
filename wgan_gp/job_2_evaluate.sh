#!/bin/bash
#PBS -N eval_wgan_gp
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o log_2_evaluate_wgan_gp.txt

echo "=== Evaluation (WGAN-GP) :: started on $(hostname) at $(date) ==="

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

nvidia-smi

echo ""
echo "=========================================================="
echo "Scoring WGAN-GP on held-out TEST years"
echo "=========================================================="
echo ""

# Unlike corrdiff_fm's Evaluate.py, there is no --steps flag to match between
# arms: this Generator is a single feed-forward pass per ensemble member
# (NFE=1), not an iterative sampler. Compare.py will correctly flag an "NFE
# MISMATCH" warning when this JSON is compared against a diffusion arm's --
# that is expected, see Evaluate.py's module docstring for why it isn't a bug.
python Evaluate.py --members 16 --batch 4 --out outputs/eval_wgan_gp.json

RC=$?
echo "exit code: ${RC}"
echo "=== Evaluation (WGAN-GP) :: finished at $(date) ==="
exit ${RC}
