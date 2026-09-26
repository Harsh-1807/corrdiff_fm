#!/bin/bash
#PBS -N infer_srdrn
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o log_3_inference_srdrn.txt

echo "=== 8km->2km cascade (SRDRN) :: started on $(hostname) at $(date) ==="

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
# Submit as a PBS ARRAY to shard by YEAR across GPUs:
#     qsub -J 0-7 job_3_inference.sh
# Sharding is by year rather than by fold so you can use more GPUs than there
# are folds. Without -J it runs every year serially in one job.
# ------------------------------------------------------------
SHARD="${PBS_ARRAY_INDEX:-}"
NUM_SHARDS=8

echo ""
echo "=========================================================="
echo "Cascade: real 8 km fields -> 2 km, SRDRN (deterministic, no ensemble)"
echo "=========================================================="
echo ""

ARGS="--batch 4 --out-dir outputs/downscaled_2km_srdrn"

# Supply the REAL 2 km orography if you have it. Without it the model loses the
# single most informative predictor it has at 2 km -- upsampling the 8 km field
# invents no new terrain detail.
ARGS="$ARGS --oro-2km $(python -c 'from Config import ORO_2KM; print(ORO_2KM)')"

if [ -n "${SHARD}" ]; then
    echo "Array shard ${SHARD} of ${NUM_SHARDS}"
    python Inference.py $ARGS --shard "${SHARD}" --num-shards "${NUM_SHARDS}"
else
    python Inference.py $ARGS
    python Inference.py --merge-only --out-dir outputs/downscaled_2km_srdrn
fi

RC=$?
echo "exit code: ${RC}"
echo "=== Cascade (SRDRN) :: finished at $(date) ==="
exit ${RC}
