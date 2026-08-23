#!/bin/bash
#PBS -N eval_edm
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o log_3_evaluate_edm.txt

echo "=== Evaluation (EDM) :: started on $(hostname) at $(date) ==="

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
echo "Scoring EDM on held-out TEST years"
echo "=========================================================="
echo ""

# --steps MUST match what the other arm uses, or the NFE budgets differ and the
# comparison is meaningless. 18 steps = 35 NFE per member in both.
python Evaluate.py --members 16 --steps 18 --batch 4 --out outputs/eval_edm.json

# Sampler-stochasticity sweep: separates how much of the ensemble spread comes
# from the parameterization versus merely from the stochastic sampler. This is
# the most informative single result in the comparison.
python Evaluate.py --members 16 --steps 18 --batch 4 \
    --churn-sweep 0 10 40 --out outputs/eval_edm_sweep.json

RC=$?
echo "exit code: ${RC}"
echo "=== Evaluation (EDM) :: finished at $(date) ==="
exit ${RC}
