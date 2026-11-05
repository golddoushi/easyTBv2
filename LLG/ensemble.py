"""Parallel deterministic LLG from a hemisphere-conditioned thermal ensemble.

This module adds no thermal field during evolution. Times are fs, energies eV.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

KB_EV_K = 8.617333262145e-5
_CACHE = {}
_CONFIG = None


def thermal_direction(rng, temperature, cells, anisotropy, pole):
    """Exact rejection sampling of sin(delta) exp(-a sin(delta)^2).

    z=cos(delta) is on [0,1]. A truncated exponential proposal for y=1-z
    has acceptance exp[-a*y*(1-y)], avoiding overflow at large barriers.
    """
    if temperature <= 0:
        raise ValueError("Thermal seeded ensembles require a positive temperature.")
    a = cells * anisotropy / (KB_EV_K * temperature)
    if not np.isfinite(a) or a > 1e14:
        raise ValueError("Thermal spread is too small to resolve reliably away from the pole.")
    while True:
        u = rng.random()
        y = u if a < 1e-10 else -np.log1p(-u * (-np.expm1(-a))) / a
        # VSH angular conversion uses arccos(z). Reject endpoints that round
        # to an exact pole rather than introducing an arbitrary tilt floor.
        if not 0. < 1.-y < 1.:
            continue
        if rng.random() < np.exp(-a * y * (1. - y)):
            break
    phi = rng.uniform(0., 2. * np.pi)
    transverse = np.sqrt(y * (2. - y))
    return np.array([transverse * np.cos(phi), transverse * np.sin(phi), pole * (1. - y)])

def trajectory_metrics(times, trajectory, initial_pole, target_pole, threshold, centers):
    mz = trajectory[:, 2]
    crossed = np.flatnonzero(initial_pole * mz < 0.)
    crossing_time = None
    if crossed.size:
        i = int(crossed[0])
        crossing_time = float(times[i-1] + (times[i]-times[i-1]) * mz[i-1] / (mz[i-1]-mz[i]))
    aligned = target_pole * mz
    bad = np.flatnonzero(aligned < threshold)
    settled_index = int(bad[-1]+1) if bad.size else 0
    settled_time = float(times[settled_index]) if settled_index < len(times) else None
    return {
        "final_mx": float(trajectory[-1, 0]), "final_my": float(trajectory[-1, 1]),
        "final_mz": float(mz[-1]), "target_alignment": float(aligned[-1]),
        "write_success": bool(aligned[-1] > 0.),
        "aligned_success": bool(aligned[-1] >= threshold),
        "final_reversed": bool(initial_pole * mz[-1] < 0.),
        "first_crossing_fs": crossing_time,
        "crossing_pulse_centers_elapsed": None if crossing_time is None else int(np.searchsorted(centers, crossing_time, side="right")),
        "settled_target_fs": settled_time,
        "settled_pulse_centers_elapsed": None if settled_time is None else int(np.searchsorted(centers, settled_time, side="right")),
        "max_stored_norm_error": float(np.max(np.abs(np.linalg.norm(trajectory, axis=1)-1.))),
    }

def _initialize(config):
    global _CONFIG, _CACHE
    _CONFIG, _CACHE = config, {}

def _run_job(job):
    from . import IFTDevice, PulseTimeStepSchedule
    c = _CONFIG
    started = perf_counter()
    key = (job["omega_eV"], job["polarization"])
    centers = np.array(c["pulse_centers_fs"])
    if key not in _CACHE:
        def pulse(t):
            return c["field_amplitude"] * np.exp(-2*np.log(2)*((t-centers)/c["fwhm_fs"])**2).sum()
        device = IFTDevice(
            data_directory=c["map_path"], omega0=key[0], polarization=key[1],
            light_pulse=pulse, use_DFT_mae=False,
            extra_uniaxial_mae=c["anisotropy_eV"], alpha=c["alpha"],
        )
        schedule = PulseTimeStepSchedule.gaussian(
            center_times=centers, intensity_fwhm=c["fwhm_fs"],
            fine_step=c["fine_step_fs"], coarse_step=c["coarse_step_fs"],
            window_half_width_in_fwhm=3.,
        )
        _CACHE[key] = device, schedule
    device, schedule = _CACHE[key]
    # Paired samples across helicities and frequencies, reproducible across workers.
    rng = np.random.default_rng(np.random.SeedSequence([c["seed"], job["temperature_index"], job["sample"]]))
    initial = thermal_direction(rng, job["temperature_K"], c["cells"], c["anisotropy_eV"], job["initial_pole"])
    device.m, device.time = initial.copy(), 0.
    stopped = []
    def stationary(direction):
        fixed = stationary_pole(device, direction)
        if fixed:
            stopped.append(True)
        return fixed
    times, trajectory = device.run(t_total=c["total_time_fs"], time_step=schedule,
                                   stationary_state=stationary)
    result = dict(job, initial_mx=float(initial[0]), initial_my=float(initial[1]), initial_mz=float(initial[2]),
                  initial_delta_deg=float(np.degrees(np.arccos(abs(initial[2])))),
                  initial_phi_deg=float(np.degrees(np.arctan2(initial[1], initial[0])) % 360.))
    result.update(trajectory_metrics(times, trajectory, job["initial_pole"], job["target_pole"], c["alignment_threshold"], centers))
    result["runtime_seconds"] = perf_counter()-started
    result["stationary_pole_shortcut"] = bool(stopped)
    result["stationary_pole_time_fs"] = float(times[-2]) if stopped else None
    if job["sample"] < c["save_trajectories"]:
        np.savez_compressed(Path(c["output"])/"trajectories"/f'{job["job_id"]:07d}.npz', time_fs=times, magnetization=trajectory, initial=initial)
    return result

def wilson_interval(successes, count):
    z = 1.959963984540054
    p = successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return max(0., center-radius), min(1., center+radius)

def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def summarize(results, output):
    results = sorted(results, key=lambda r: r["job_id"])
    _write_csv(output/"trajectories.csv", results)
    groups = {}
    for r in results:
        key = (r["temperature_K"], r["omega_eV"], r["initial_pole"], r["polarization"], r["target_pole"])
        groups.setdefault(key, []).append(r)
    summaries, histograms = [], []
    for key, rows in groups.items():
        base = dict(zip(["temperature_K", "omega_eV", "initial_pole", "polarization", "target_pole"], key))
        row = dict(base, samples=len(rows))
        for name in ("write_success", "aligned_success", "final_reversed"):
            count = sum(r[name] for r in rows)
            lo, hi = wilson_interval(count, len(rows))
            row.update({name+"_count": count, name+"_fraction": count/len(rows), name+"_wilson95_low": lo, name+"_wilson95_high": hi})
        row["mean_target_alignment"] = float(np.mean([r["target_alignment"] for r in rows]))
        for q in (5, 50, 95):
            row[f"target_alignment_p{q}"] = float(np.percentile([r["target_alignment"] for r in rows], q))
        row["initial_delta_rms_deg"] = float(np.sqrt(np.mean([r["initial_delta_deg"]**2 for r in rows])))
        row["max_stored_norm_error"] = max(r["max_stored_norm_error"] for r in rows)
        summaries.append(row)
        for metric in ("crossing_pulse_centers_elapsed", "settled_pulse_centers_elapsed"):
            counts = {}
            for r in rows:
                value = r[metric]
                label = "not_observed_by_end" if value is None else str(value)
                counts[label] = counts.get(label, 0)+1
            for label, count in counts.items():
                histograms.append(dict(base, event=metric, pulse_centers_elapsed=label, count=count, fraction=count/len(rows)))
    _write_csv(output/"summary.csv", summaries)
    _write_csv(output/"pulse_counts.csv", histograms)

def stationary_pole(device, direction):
    """
    Checking how to interupt the propagation and sign the simulation as completed due to the stationary state
    
    For case of IFT simulation on PMA MnBi2Te4, when the magnetization orientation is close to pole sufficiently,
    both the anisotropy-torque and optical torque will be zero, then the system enter stationary state.

    For other cases, such as the SOTs, this maybe not the case. The current-induced SOTs can drag the magnetization
    out from the pole.

    So, define this function case by case! If not sure, return False always.
    """
    if abs(direction[0]) > 1e-6 or abs(direction[1]) > 1e-6 or 1 - abs(direction[2]) > 1e-6:
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", action="store_true", help="run thermal initial-angle ensembles")
    parser.add_argument("--data", default="mcalc/MnBi2Te4")
    parser.add_argument("--output", default="mcalc/MnBi2Te4/ensemble")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--samples", type=int, default=100, help="samples per temperature/frequency/pole/helicity")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[11.2])
    parser.add_argument("--omegas", type=float, nargs="+", default=[0.75, 2.4])
    parser.add_argument("--cells", type=int, default=625)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--anisotropy-mev", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--field-amplitude", type=float, default=0.06, help="V/angstrom amplitude passed directly to IFTDevice, same convention as run_ift_example; no additional 1/4 applied")
    parser.add_argument("--fwhm-fs", type=float, default=50.)
    parser.add_argument("--pulses", type=int, default=30)
    parser.add_argument("--first-center-fs", type=float, default=100.)
    parser.add_argument("--separation-fs", type=float, default=100000.)
    parser.add_argument("--total-time-fs", type=float, default=5100000.)
    parser.add_argument("--fine-step-fs", type=float, default=10.)
    parser.add_argument("--coarse-step-fs", type=float, default=1000.)
    parser.add_argument("--alignment-threshold", type=float, default=0.95)
    parser.add_argument("--xy-plus-target", type=int, choices=[-1, 1], default=-1, help="target pole for code alias xy+; xy- uses opposite target (verify manuscript helicity labels)")
    parser.add_argument("--save-trajectories", type=int, default=0, help="save first N trajectories per matrix entry as NPZ")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--matrix-entry", type=int, help="run one zero-based frequency/pole/helicity entry at all temperatures")
    args = parser.parse_args()
    if not args.ensemble:
        parser.error("supply --ensemble to run this mode")
    positive = [args.workers, args.samples, args.cells, args.anisotropy_mev, args.fwhm_fs, args.pulses, args.separation_fs, args.total_time_fs, args.fine_step_fs, args.coarse_step_fs]
    numbers = positive + args.temperatures + args.omegas + [args.alpha, args.field_amplitude, args.first_center_fs, args.alignment_threshold]
    if not all(np.isfinite(numbers)) or min(positive) <= 0 or min(args.temperatures) <= 0 or args.alpha < 0 or args.seed < 0 or args.save_trajectories < 0 or args.first_center_fs < 0 or not 0 < args.alignment_threshold < 1:
        parser.error("invalid nonfinite, negative, or out-of-range parameter")
    if len(set(args.temperatures)) != len(args.temperatures) or len(set(args.omegas)) != len(args.omegas):
        parser.error("temperatures and frequencies must not contain duplicates")
    centers = [args.first_center_fs+i*args.separation_fs for i in range(args.pulses)]
    if args.total_time_fs <= centers[-1]+3*args.fwhm_fs:
        parser.error("total time must extend past the final pulse window")
    path = Path(args.data).resolve()
    if path.is_dir():
        path /= "AngleDependentDynamicsMap.npz"
    if not path.is_file():
        parser.error(f"map not found: {path}")
    with np.load(path) as data:
        omega_axis = data["omega_axis_eV"]
        if min(args.omegas) < omega_axis.min() or max(args.omegas) > omega_axis.max():
            parser.error("requested frequency lies outside the stored map")
    output = Path(args.output).resolve()
    if args.matrix_entry is not None and not 0 <= args.matrix_entry < 4*len(args.omegas):
        parser.error("matrix entry outside requested frequencies")
    config = dict(map_path=str(path), output=str(output), samples=args.samples, temperatures=args.temperatures,
                  omegas=args.omegas, cells=args.cells, seed=args.seed, anisotropy_eV=args.anisotropy_mev*1e-3,
                  alpha=args.alpha, field_amplitude=args.field_amplitude, fwhm_fs=args.fwhm_fs,
                  pulse_centers_fs=centers, total_time_fs=args.total_time_fs, fine_step_fs=args.fine_step_fs,
                  coarse_step_fs=args.coarse_step_fs, alignment_threshold=args.alignment_threshold,
                  xy_plus_target=args.xy_plus_target, save_trajectories=args.save_trajectories,
                  map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                  model="deterministic LLG; thermal initial angles only; uniaxial hemisphere-conditioned Boltzmann distribution")
    manifest = output/"config.json"
    if args.matrix_entry is not None:
        config["matrix_entry"] = args.matrix_entry
    if args.resume:
        if not manifest.exists() or json.loads(manifest.read_text()) != config:
            parser.error("resume requires identical configuration, map and LLG source files")
    else:
        if output.exists() and any(output.iterdir()):
            parser.error("output is not empty; use a new directory or --resume")
        output.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output/"results").mkdir(exist_ok=True)
    (output/"trajectories").mkdir(exist_ok=True)
    jobs, results = [], []
    index = 0
    for ti, temperature in enumerate(args.temperatures):
        for omega in args.omegas:
            for pole in (1, -1):
                for pol, target in (("xy+", args.xy_plus_target), ("xy-", -args.xy_plus_target)):
                    for sample in range(args.samples):
                        entry = (index // args.samples) % (4*len(args.omegas))
                        if args.matrix_entry is not None and args.matrix_entry != entry:
                            index += 1
                            continue
                        job = dict(job_id=index, temperature_index=ti, temperature_K=temperature, omega_eV=omega,
                                   initial_pole=pole, polarization=pol, target_pole=target, sample=sample)
                        saved = output/"results"/f"{index:07d}.json"
                        if saved.exists():
                            results.append(json.loads(saved.read_text()))
                        else:
                            jobs.append(job)
                        index += 1
    # Spawned processes import NumPy only after inheriting these limits.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    index = len(results) + len(jobs)
    print(f"{index} trajectories; {len(results)} completed; {len(jobs)} pending; {args.workers} workers", flush=True)
    print("Code aliases xy+/xy- are recorded literally; no manuscript helicity relabeling is applied.", flush=True)
    start = perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=_initialize, initargs=(config,)) as executor:
            futures = {executor.submit(_run_job, job): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                dest = output/"results"/f'{result["job_id"]:07d}.json'
                temporary = dest.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
                temporary.replace(dest)
                results.append(result)
                if len(results) % max(1, min(50, len(jobs)//10)) == 0 or len(results) == index:
                    print(f"Completed {len(results)}/{index}; elapsed {perf_counter()-start:.1f} s", flush=True)
    finally:
        if results:
            summarize(results, output)
    print(f"Results: {output}/summary.csv", flush=True)


if __name__ == "__main__":
    main()
