#!/bin/bash
#SBATCH --job-name=ift-angle-map
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=qhn
#SBATCH --output=ift-angle-map-%j.out
#SBATCH --error=ift-angle-map-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Prevent every Python process from creating an additional BLAS thread team.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON=${PYTHON:-python3}
WORKERS_PER_NODE=${WORKERS_PER_NODE:-$SLURM_CPUS_PER_TASK}

echo "Preparing shared reference filling on one node"
srun --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --output="ift-prepare-${SLURM_JOB_ID}.out" --error="ift-prepare-${SLURM_JOB_ID}.err" \
    "$PYTHON" angleDependentSOTMapping.py --stage prepare \
    --workers-per-node "$WORKERS_PER_NODE"

echo "Computing $SLURM_NNODES shards with $WORKERS_PER_NODE local workers per node"
srun --kill-on-bad-exit=1 --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --output="ift-compute-${SLURM_JOB_ID}-%t.out" --error="ift-compute-${SLURM_JOB_ID}-%t.err" \
    "$PYTHON" angleDependentSOTMapping.py --stage compute \
    --workers-per-node "$WORKERS_PER_NODE" --bz-workers-per-orientation 1

echo "Aggregating all checkpoints and fitting angular harmonics on one node"
srun --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --output="ift-aggregate-${SLURM_JOB_ID}.out" --error="ift-aggregate-${SLURM_JOB_ID}.err" \
    "$PYTHON" angleDependentSOTMapping.py --stage aggregate \
    --workers-per-node "$WORKERS_PER_NODE" --shard-count "$SLURM_NNODES"

echo "Angle-dependent mapping complete"
