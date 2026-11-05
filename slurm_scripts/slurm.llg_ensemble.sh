#!/bin/bash
#SBATCH --job-name=llg-ensemble
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=qhn
#SBATCH --output=llg-ensemble-%j.out
#SBATCH --error=llg-ensemble-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHON=${PYTHON:-python3}
export OUT=${OUT:-mcalc/MnBi2Te4/ensemble_${SLURM_JOB_ID}}
export SAMPLES=${SAMPLES:-300}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$OUT"
srun --kill-on-bad-exit=1 --nodes=8 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=96 \
    --output="$OUT/node-%t-%j.log" --error="$OUT/node-%t-%j.err" \
    bash -c '
        set -eu
        shard="$OUT/entry-$SLURM_PROCID"
        set -- "$PYTHON" -u LLGInput.py --ensemble --workers "$SLURM_CPUS_PER_TASK" \
            --matrix-entry "$SLURM_PROCID" --samples "$SAMPLES" --temperatures 5 7 11.2 \
            --omegas 0.75 2.4 --xy-plus-target 1 --save-trajectories 2 \
            --output "$shard"
        if [[ -f "$shard/config.json" ]]; then set -- "$@" --resume; fi
        exec "$@"
    '

# Each entry owns entire statistical groups, so the tables concatenate directly.
"$PYTHON" - "$OUT" "$SAMPLES" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("summary.csv", "trajectories.csv", "pulse_counts.csv"):
    temp = root / (name + ".tmp")
    with temp.open("w", newline="") as out:
        writer = csv.writer(out)
        for entry in range(8):
            with (root / f"entry-{entry}" / name).open(newline="") as src:
                rows = csv.reader(src)
                header = next(rows)
                if entry == 0:
                    writer.writerow(header)
                data = list(rows)
                if name == "summary.csv":
                    if len(data) != 3 or any(int(r[header.index("samples")]) != int(sys.argv[2]) for r in data):
                        raise ValueError(f"Incomplete summary for entry {entry}")
                writer.writerows(data)
    temp.replace(root / name)
PY
echo "Combined results in $OUT"
