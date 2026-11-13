"""Distribute fluence-duration-energy ensembles across Slurm ranks; export Origin data."""
import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', default='mcalc/MnBi2Te4')
    p.add_argument('--fluences', type=float, nargs='+', default=[.25, .5, .75, 1.])
    p.add_argument('--durations-ps', type=float, nargs='+', default=[3.6, 7.2, 14.4])
    p.add_argument('--omegas', type=float, nargs='+', default=[.75, 2.4])
    p.add_argument('--ref-field', type=float, default=.005, help='Manuscript/LLG amplitude in V/angstrom')
    p.add_argument('--ref-fwhm-ps', type=float, default=7.2)
    p.add_argument('--alpha', type=float, default=.001)
    p.add_argument('--anisotropy-mev', type=float, default=.2)
    p.add_argument('--temperature', type=float, default=11.2)
    p.add_argument('--temperature-seed-index', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--cells', type=int, default=625)
    p.add_argument('--samples', type=int, default=300)
    p.add_argument('--workers', type=int, default=96)
    p.add_argument('--pulses', type=int, default=30)
    p.add_argument('--separation-ps', type=float, default=100.)
    p.add_argument('--relax-ps', type=float, default=2199.9)
    p.add_argument('--fine-step-fs', type=float, default=500.)
    p.add_argument('--coarse-step-fs', type=float, default=1000.)
    p.add_argument('--alignment-threshold', type=float, default=.95)
    p.add_argument('--matrix-entry', type=int, choices=[1, 2], default=1,
                   help='1: initial +z, xy-; 2: initial -z, xy+ (target xy+ = +z)')
    p.add_argument('--merge', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def cases(a):
    for index, (omega, duration, fluence) in enumerate(itertools.product(a.omegas, a.durations_ps, a.fluences)):
        field = a.ref_field * math.sqrt(fluence * a.ref_fwhm_ps / duration)
        # A common start offset keeps every first pulse at least three FWHM from t=0.
        first = 3 * max(a.durations_ps) * 1000
        end = first + (a.pulses - 1) * a.separation_ps * 1000 + a.relax_ps * 1000
        metadata = dict(case=index, omega_eV=omega, wavelength_nm=1239.841984/omega,
                        duration_ps=duration, F_over_Fref=fluence, field_amplitude=field)
        # Vacuum-equivalent fluence for E(t)=2 E_MS Re[epsilon exp(-i wt)], |epsilon|=1.
        metadata['vacuum_equivalent_fluence_mJ_cm2'] = (
            2 * 8.8541878128e-12 * 299792458 * (field*1e10)**2
            * duration*1e-12 * math.sqrt(math.pi/(4*math.log(2))) * .1)
        out = a.output / f'case_{index:03d}'
        options = dict(data=a.data, output=str(out), workers=a.workers, samples=a.samples,
                       temperatures=a.temperature, omegas=omega, cells=a.cells, seed=a.seed,
                       temperature_seed_index=a.temperature_seed_index, matrix_entry=a.matrix_entry,
                       xy_plus_target=1, alpha=a.alpha, anisotropy_mev=a.anisotropy_mev,
                       field_amplitude=field, fwhm_fs=duration*1000, pulses=a.pulses,
                       first_center_fs=first, separation_fs=a.separation_ps*1000,
                       total_time_fs=end, fine_step_fs=a.fine_step_fs,
                       coarse_step_fs=a.coarse_step_fs, alignment_threshold=a.alignment_threshold)
        cmd = [sys.executable, '-u', str(ROOT/'LLGInput.py'), '--ensemble', '--pulse-bins']
        for key, value in options.items():
            cmd += ['--'+key.replace('_', '-'), str(value)]
        yield metadata, out, cmd


def merge(a):
    rows = []
    for metadata, out, _ in cases(a):
        with (out/'summary.csv').open(newline='') as f:
            summaries = list(csv.DictReader(f))
        if len(summaries) != 1 or int(summaries[0]['samples']) != a.samples:
            raise ValueError(f'Incomplete or unexpected ensemble: {out}')
        r = summaries[0]
        n = a.samples
        complete = int(r['aligned_success_count'])
        hemi = int(r['write_success_count'])
        if not 0 <= complete <= hemi <= n:
            raise ValueError(f'Invalid outcome counts: {out}')
        row = dict(metadata, samples=n, target_hemisphere_count=hemi,
                   complete_count=complete, partial_count=hemi-complete, absent_count=n-hemi,
                   target_hemisphere_percent=100*hemi/n,
                   complete_percent=100*complete/n, partial_percent=100*(hemi-complete)/n,
                   absent_percent=100*(n-hemi)/n,
                   mean_target_alignment=float(r['mean_target_alignment']))
        rows.append(row)
    for name, delimiter in [('phase_map.csv', ','), ('phase_map_Origin.txt', '\t')]:
        with (a.output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)
    for omega in a.omegas:
        for metric in ['target_hemisphere_percent', 'complete_percent', 'partial_percent', 'absent_percent', 'mean_target_alignment']:
            with (a.output/f'omega_{omega:g}_{metric}_Origin.txt').open('w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['duration_ps', *[f'F_over_Fref_{v:g}' for v in a.fluences]])
                for duration in a.durations_ps:
                    writer.writerow([duration, *[next(r[metric] for r in rows if
                        r['omega_eV'] == omega and r['duration_ps'] == duration and r['F_over_Fref'] == fluence)
                        for fluence in a.fluences]])
    print(f'Exported {len(rows)} cases to {a.output}', flush=True)


def main():
    p = parser()
    a = p.parse_args()
    for values in [a.fluences, a.durations_ps, a.omegas]:
        if len(set(values)) != len(values) or any(not math.isfinite(v) or v <= 0 for v in values):
            p.error('Scan axes must be positive, finite, and distinct')
    for key in ['ref_field', 'ref_fwhm_ps', 'anisotropy_mev', 'samples', 'workers', 'cells',
                'pulses', 'separation_ps', 'relax_ps', 'fine_step_fs', 'coarse_step_fs']:
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            p.error(f'{key} must be positive and finite')
    if not 0 < a.alignment_threshold <= 1 or not math.isfinite(a.alpha) or a.alpha < 0:
        p.error('Invalid alignment threshold or damping')
    if a.relax_ps <= 3*max(a.durations_ps):
        p.error('Relaxation after the last center must exceed three maximum pulse widths')
    if 6*max(a.durations_ps) >= a.separation_ps:
        p.error('This isolated-pulse fluence scan requires separation > 6 * maximum FWHM')
    a.output = a.output.resolve()
    a.data = str(Path(a.data).resolve())
    spec = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
            if k not in ['merge', 'dry_run', 'workers']}
    manifest = a.output/'scan_config.json'
    if not a.dry_run:
        a.output.mkdir(parents=True, exist_ok=True)
        try:
            with manifest.open('x') as f:
                json.dump(spec, f, indent=2)
        except FileExistsError:
            # All ranks validate the same manifest before launching computations.
            import time
            for attempt in range(50):
                try:
                    existing = json.loads(manifest.read_text())
                    break
                except json.JSONDecodeError:
                    time.sleep(.1)
            else:
                raise RuntimeError('Incomplete scan manifest')
            if existing != spec:
                raise ValueError('Output directory belongs to a different scan; use a new output directory')
    if a.merge:
        if a.dry_run:
            p.error('--merge and --dry-run cannot be combined')
        merge(a)
        return
    rank = int(os.environ.get('SLURM_PROCID', 0))
    ranks = int(os.environ.get('SLURM_NTASKS', 1))
    if not 0 <= rank < ranks:
        raise ValueError('Invalid Slurm rank configuration')
    for metadata, out, cmd in cases(a):
        if metadata['case'] % ranks != rank:
            continue
        if (out/'config.json').exists():
            cmd.append('--resume')
        print(json.dumps(metadata), flush=True)
        if a.dry_run:
            print(' '.join(cmd), flush=True)
        else:
            subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
