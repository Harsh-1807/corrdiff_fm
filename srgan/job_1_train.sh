#!/bin/bash
#PBS -N train_srgan
#PBS -q workq
#PBS -l select=1:ncpus=16:ngpus=2
#PBS -l walltime=240:00:00
#PBS -j oe
#PBS -o log_1_train_srgan.txt

echo "=== SRGAN training :: started on $(hostname) at $(date) ==="

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

# ------------------------------------------------------------
# Distributed
# ------------------------------------------------------------
# GPU count derived from the actual allocation rather than hard-coded: a
# hard-coded value that disagrees with what the scheduler gave you either
# oversubscribes devices or silently wastes half of them.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    N_GPUS=$(nvidia-smi -L | wc -l)
fi
echo "Detected ${N_GPUS} GPU(s)"

# Randomized so two concurrent jobs on one node do not collide on the port.
export MASTER_PORT=$((29000 + RANDOM % 2000))
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo ""
echo "=========================================================="
echo "SRGAN training (${N_GPUS} GPUs)"
echo "Single script, two phases per fold:"
echo "  Phase 1 -- MSE pretraining of the generator alone"
echo "  Phase 2 -- adversarial fine-tuning (generator + discriminator)"
echo "See Train.py's module docstring for why the phases cannot be skipped"
echo "or merged, and for the 1e-3 adversarial-weight rationale."
echo "=========================================================="
echo ""

torchrun --nproc_per_node="${N_GPUS}" --master_port="${MASTER_PORT}" Train.py

RC=$?
echo "exit code: ${RC}"
echo "=== SRGAN training :: finished at $(date) ==="
exit ${RC}
