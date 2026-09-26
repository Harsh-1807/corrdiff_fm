#!/bin/bash
#PBS -N eval_srgan
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o log_2_evaluate_srgan.txt

echo "=== Evaluation (SRGAN) :: started on $(hostname) at $(date) ==="

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
echo "Scoring SRGAN on held-out TEST years"
echo "Deterministic generator -> members=1 always; CRPS degenerates"
echo "to MAE and spread/spread_skill are always 0 (see Evaluate.py)."
echo "=========================================================="
echo ""

python Evaluate.py --batch 4 --out outputs/eval_srgan.json

RC=$?
echo "exit code: ${RC}"
echo "=== Evaluation (SRGAN) :: finished at $(date) ==="
echo ""
echo "Compare against corrdiff_fm once both have an eval JSON:"
echo "  python Compare.py ../corrdiff_fm/outputs/eval_edm.json outputs/eval_srgan.json"
exit ${RC}
