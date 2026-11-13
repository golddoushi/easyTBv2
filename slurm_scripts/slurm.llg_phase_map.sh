#!/bin/bash
#SBATCH --job-name=llg-phase-map
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=qhn
#SBATCH --output=llg-phase-map-%j.out
#SBATCH --error=llg-phase-map-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=${PYTHON:-python3}
OUT=${OUT:-mcalc/MnBi2Te4/Q2_phase_map_${SLURM_JOB_ID}}
DATA=${DATA:-mcalc/MnBi2Te4}
mkdir -p "$OUT"
read -r -a fluences <<< "${FLUENCES:-0.25 0.5 0.75 1.0}"
read -r -a durations <<< "${DURATIONS_PS:-3.6 7.2 14.4}"
read -r -a energies <<< "${OMEGAS:-0.75 2.4}"
common=( --output "$OUT" --data "$DATA" --workers "$SLURM_CPUS_PER_TASK"
    --samples "${SAMPLES:-300}" --fluences "${fluences[@]}"
    --durations-ps "${durations[@]}" --omegas "${energies[@]}"
    --ref-field "${REF_FIELD:-0.005}" --ref-fwhm-ps "${REF_FWHM_PS:-7.2}"
    --fine-step-fs "${FINE_STEP_FS:-500}" --coarse-step-fs "${COARSE_STEP_FS:-1000}"
    --relax-ps "${RELAX_PS:-2199.9}" --separation-ps "${SEPARATION_PS:-100}"
    --matrix-entry "${MATRIX_ENTRY:-1}" )
srun --kill-on-bad-exit=1 --nodes=8 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=96 \
    --output="$OUT/node-%t-%j.log" --error="$OUT/node-%t-%j.err" \
    "$PYTHON" -u slurm_scripts/run_llg_phase_map.py "${common[@]}"
"$PYTHON" slurm_scripts/run_llg_phase_map.py "${common[@]}" --merge
echo "Completed phase map: $OUT"
