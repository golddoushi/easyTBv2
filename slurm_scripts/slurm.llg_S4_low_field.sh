#!/bin/bash
#SBATCH --job-name=llg-S4-low
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=qhn
#SBATCH --output=llg-S4-%j.out
#SBATCH --error=llg-S4-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=${PYTHON:-python3}
OUT=${OUT:-mcalc/MnBi2Te4/S4_low_field_${SLURM_JOB_ID}}
DATA=${DATA:-mcalc/MnBi2Te4}
mkdir -p "$OUT"
common=( S4 --output "$OUT" --data "$DATA" --workers "$SLURM_CPUS_PER_TASK"
    --samples "${SAMPLES:-300}" --fine-step-fs "${FINE_STEP_FS:-500}"
    --coarse-step-fs "${COARSE_STEP_FS:-1000}"
    --first-center-fs "${FIRST_CENTER_FS:-14400}" --relax-ps "${RELAX_PS:-2199.9}" )
srun --kill-on-bad-exit=1 --nodes=8 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=96 \
    --output="$OUT/node-%t-%j.log" --error="$OUT/node-%t-%j.err" \
    "$PYTHON" -u slurm_scripts/run_llg_tables.py "${common[@]}"
"$PYTHON" slurm_scripts/run_llg_tables.py "${common[@]}" --merge
echo "Completed S4: $OUT"
