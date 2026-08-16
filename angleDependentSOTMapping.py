import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout
import argparse
import hashlib
import inspect
import json
import warnings
from pathlib import Path
import numpy as np
import paramReader
paramReader.readGlobalConst()

from SphericalHarmonics import SphericalHarmonicExpansion, VectorSphericalHarmonicExpansion
from SphericalHarmonics import (
    axial_rotation_matrices,
    cartesian_to_spherical,
    fibonacci_sphere,
    project_tangential,
    transform_ift_tensor,
)
from auxiliary import fermiDiracOccupations
from auxiliary import getWholeSymmetryOperator, SYMPREC
from system import System_w90
from task import BZGridTask, solveEnergy, solveIFTorque_LG_DM
from auxiliary import BOHR_MAGNETON_EV_PER_TESLA

_orientation_worker_model = None
_warned_legacy_ift_checkpoint = False
IFT_CHECKPOINT_FORMAT_VERSION = 3

def orientation_indices_for_shard(number_of_orientations, shard_rank, shard_count):
    """Return the deterministic modulo shard owned by one Slurm task."""
    if shard_count < 1: raise ValueError("shard_count must be positive.")
    if not 0 <= shard_rank < shard_count: raise ValueError("shard_rank must satisfy 0 <= rank < shard_count.")
    return list(range(shard_rank, number_of_orientations, shard_count))

def command_line_arguments():
    parser = argparse.ArgumentParser(description="Angle-dependent electronic and IFT mapping.")
    parser.add_argument("--stage", choices=("all", "prepare", "compute", "aggregate"), default="all",
        help="Run locally end-to-end, prepare shared filling, compute one Slurm shard, or aggregate checkpoints.")
    parser.add_argument("--workers-per-node", type=int, default=None,
        help="Local orientation-process count; defaults to SLURM_CPUS_PER_TASK or the detected CPU count.")
    parser.add_argument("--bz-workers-per-orientation", type=int, default=1,
        help="Inner BZ workers for each orientation process. Keep this at 1 for orientation-level parallelism.")
    parser.add_argument("--shard-rank", type=int, default=None,
        help="Override SLURM_PROCID, mainly for testing or manual sharding.")
    parser.add_argument("--shard-count", type=int, default=None,
        help="Override SLURM_NTASKS. Pass the compute-node count to the aggregate stage.")
    parser.add_argument("--harmonic-degree", type=int, default=None,
        help="Override the VSH fitting degree during aggregation without changing electronic checkpoints.")
    return parser.parse_args()

def distributed_run_settings(stage, workers_per_node=None, shard_rank=None, shard_count=None):
    default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    workers_per_node = default_workers if workers_per_node is None else int(workers_per_node)
    if workers_per_node < 1: raise ValueError("workers_per_node must be positive.")
    if stage == "compute":
        shard_rank = int(os.environ.get("SLURM_PROCID", 0)) if shard_rank is None else int(shard_rank)
        shard_count = int(os.environ.get("SLURM_NTASKS", 1)) if shard_count is None else int(shard_count)
    else:
        shard_rank = 0 if shard_rank is None else int(shard_rank)
        shard_count = 1 if shard_count is None else int(shard_count)
    orientation_indices_for_shard(1, shard_rank, shard_count)
    return workers_per_node, shard_rank, shard_count

class StaleIFTCheckpointError(RuntimeError):
    """Raised when a saved IFT sample was produced by incompatible settings or code."""

def build_ift_checkpoint_signature(main_dir,solver_name,occupation_mode,kBT,eta,degen,mag_type,
                                   torque_projection_specs,exchange_provenance):
    """Fingerprint electronic settings and code without coupling checkpoints to VSH postprocessing."""
    source_dir = Path(__file__).resolve().parent
    source_files = [
        source_dir / "calculator" / "__init__.py", source_dir / "task.py",
        source_dir / "auxiliary.py", source_dir / "paramReader.py",
        source_dir / "system" / "__init__.py", source_dir / "system" / "system_w90.py",
    ]
    source_hash = hashlib.sha256()
    for path in source_files:
        source_hash.update(path.relative_to(source_dir).as_posix().encode("utf-8"))
        normalized_source = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        source_hash.update(normalized_source.encode("utf-8"))
    for function in (
            solve_energy_orientation, solve_ift_orientation, collect_bz_energies,
            collect_bz_response, solve_chemical_potential, fixed_number_electronic_state):
        source_hash.update(function.__name__.encode("utf-8"))
        normalized_source = inspect.getsource(function).replace("\r\n", "\n").replace("\r", "\n")
        source_hash.update(normalized_source.encode("utf-8"))

    model_inputs = []
    for path in sorted(Path(main_dir).iterdir(), key=lambda item: item.name):
        is_input = path.name == "POSCAR" or path.name.endswith("_hr.dat") or path.suffix.lower() in {
            ".chk", ".eig", ".mmn", ".soi", ".win",
        }
        if path.is_file() and is_input:
            model_inputs.append((path.name, path.stat().st_size))

    payload = {
        "format_version": IFT_CHECKPOINT_FORMAT_VERSION,
        "solver_name": solver_name, "occupation_mode": occupation_mode,
        "kBT_eV": float(kBT), "eta_eV": float(eta), "degen_eV": float(degen), "mag_type": mag_type,
        "torque_projection_specs": torque_projection_specs,
        "exchange_provenance": exchange_provenance,
        "electronic_source_sha256": source_hash.hexdigest(), "model_inputs": model_inputs,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

def checkpoint_signatures_compatible(saved_signature, expected_signature, saved_format_version):
    """Compare signatures, allowing semantic reuse of version-2 checkpoints."""
    if saved_signature == expected_signature and saved_format_version == IFT_CHECKPOINT_FORMAT_VERSION:
        return True
    try:
        saved_payload = json.loads(saved_signature)
        expected_payload = json.loads(expected_signature)
    except (TypeError, json.JSONDecodeError):
        return False
    if saved_format_version != 2: return False
    saved_payload.pop("format_version", None)
    expected_payload.pop("format_version", None)
    saved_payload.pop("source_sha256", None)
    expected_payload.pop("electronic_source_sha256", None)
    return saved_payload == expected_payload

def initialize_orientation_worker(model=None):
    """Give one process exclusive ownership of a reusable electronic model."""
    global _orientation_worker_model
    if model is not None:
        _orientation_worker_model = model
    if _orientation_worker_model is None:
        raise RuntimeError("The orientation worker did not inherit or receive an electronic model.")

@contextmanager
def orientation_worker_output(verbose):
    """Keep output from many simultaneous calculators from interleaving."""
    if verbose:
        yield
    else:
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
            yield

def solve_energy_orientation(job):
    """Process-pool entry point for one fixed-number energy orientation."""
    model = _orientation_worker_model
    if model is None:
        raise RuntimeError("Orientation worker model is not initialized.")
    with orientation_worker_output(job["worker_verbose"]):
        model.includeSOCFromSOI(theta=job["theta"], phi=job["phi"], strength=1.0, keepRegularSpinAxis=True)
        calc = BZGridTask(model, solveEnergy, args={}, BZGrid=job["bz_grid"], BZSymm=False,
            mode="run", ncores=job["bz_workers"])
        energies = collect_bz_energies(calc, job["bz_grid"])
        sample = fixed_number_electronic_state(
            energies, job["target_electrons_per_cell"], job["number_kpoints"],
            job["occupation_mode"], job["kBT"])
    return job["index"], sample

def solve_ift_orientation(job):
    """Process-pool entry point for one orientation-resolved IFT calculation."""
    model = _orientation_worker_model
    if model is None:
        raise RuntimeError("Orientation worker model is not initialized.")
    with orientation_worker_output(job["worker_verbose"]):
        model.includeSOCFromSOI(theta=job["theta"], phi=job["phi"], strength=1.0, keepRegularSpinAxis=True)
        calc = BZGridTask(
            model, job["ift_solver"],
            args={
                "fermi": np.asarray([job["chemical_potential"]]),
                "omega": job["omega_axis"],
                "mag type": job["mag_type"],
                "projected shells for detecting torque": job["torque_projection_specs"],
                "finite temperature occupations": job["occupation_mode"] == "fermi-dirac",
                "kBT": job["kBT"], "eta": job["eta"], "degen": job["degen"],
            },
            BZGrid=job["bz_grid"], BZSymm=False, mode="run", ncores=job["bz_workers"])
        projected_chi_sample = collect_bz_response(calc)
    return job["index"], projected_chi_sample

def create_orientation_executor(model, number_of_workers):
    """Create a model-isolated process pool, preferring copy-on-write on POSIX."""
    global _orientation_worker_model
    _orientation_worker_model = model
    number_of_workers = int(number_of_workers)
    if number_of_workers <= 1:
        initialize_orientation_worker()
        return None, "serial"

    available_methods = mp.get_all_start_methods()
    if "fork" in available_methods:
        start_method = "fork"
        context = mp.get_context(start_method)
        executor = ProcessPoolExecutor(
            max_workers=number_of_workers, mp_context=context,
            initializer=initialize_orientation_worker)
    else:
        start_method = "spawn"
        context = mp.get_context(start_method)
        executor = ProcessPoolExecutor(
            max_workers=number_of_workers, mp_context=context,
            initializer=initialize_orientation_worker, initargs=(model,))
    return executor, start_method

def completed_orientation_jobs(executor, worker, jobs):
    """Yield completed orientation jobs, propagating worker failures promptly."""
    if executor is None:
        for job in jobs:
            yield worker(job)
        return
    futures = [executor.submit(worker, job) for job in jobs]
    try:
        for future in as_completed(futures):
            yield future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        raise

def collect_bz_energies(calc, bz_grid):
    """Assemble full-grid energies returned by a hybrid FFT calculation."""
    _, _, _, number_of_bands = calc.results[0]["energies"].shape
    energies = np.zeros(tuple(bz_grid) + (number_of_bands,), dtype=float)

    for result in calc.results:
        iq = result["iq"]
        energies[iq[0] :: calc.QGrid[0], iq[1] :: calc.QGrid[1], iq[2] :: calc.QGrid[2]] = result["energies"]
    return energies

def collect_bz_response(calc):
    """Average an integrated response over the phase-shifted FFT grids."""
    projected_response = np.zeros_like(calc.results[0]["projected sigma"], dtype=complex)
    for result in calc.results:
        projected_response += result["projected sigma"]
    projected_response /= len(calc.results)
    return projected_response

def fixed_electron_band_energy(energies, occupied_state_count, number_kpoints):
    """
    Return the zero-temperature force-theorem band energy at fixed filling.

    ``occupied_state_count`` counts states over the complete finite BZ grid.
    The result is energy per primitive cell.
    """
    flat_energies = np.asarray(energies).reshape(-1)
    occupied_state_count = int(occupied_state_count)
    if occupied_state_count < 1 or occupied_state_count >= len(flat_energies):
        raise ValueError("occupied_state_count must lie between zero and the total number of sampled states.")

    partitioned = np.partition(flat_energies, occupied_state_count)
    occupied = partitioned[:occupied_state_count]
    highest_occupied = np.max(occupied)
    lowest_unoccupied = partitioned[occupied_state_count]
    discrete_fermi = 0.5 * (highest_occupied + lowest_unoccupied)
    return np.sum(occupied) / number_kpoints, discrete_fermi, highest_occupied, lowest_unoccupied

def solve_chemical_potential(energies, target_electrons_per_cell, number_kpoints, kBT, tol=1.0e-12):
    """Find the chemical potential giving a fixed Fermi-Dirac electron count."""
    energies = np.asarray(energies, dtype=float)
    number_bands = energies.size / number_kpoints
    if not 0.0 < target_electrons_per_cell < number_bands:
        raise ValueError("target_electrons_per_cell must lie strictly between zero and the number of bands.")
    if kBT <= 0.0:
        raise ValueError("A positive kBT is required for Fermi-Dirac filling.")

    margin = max(40.0 * kBT, 1.0)
    lower = float(np.min(energies) - margin)
    upper = float(np.max(energies) + margin)
    for _ in range(200): # Bisector searching for the chemical potential that gives correct filling
        chemical_potential = 0.5 * (lower + upper)
        occupations = fermiDiracOccupations(energies, [chemical_potential], kBT)[0]
        electron_count = np.sum(occupations) / number_kpoints
        if abs(electron_count - target_electrons_per_cell) < tol:
            return chemical_potential
        if electron_count < target_electrons_per_cell:
            lower = chemical_potential
        else:
            upper = chemical_potential
    raise RuntimeError("The fixed-number chemical-potential search did not converge.")

def fixed_number_electronic_state(energies, target_electrons_per_cell, number_kpoints, occupation_mode, kBT):
    """
    Evaluate the electronic energy at fixed particle number.

    At finite temperature the energy landscape is the Helmholtz free energy,
    including the independent-electron entropy term.  The internal band
    energy is returned separately for diagnostics.
    """
    state_count = target_electrons_per_cell * number_kpoints
    nominal_occupied_count = int(round(state_count))
    flat_energies = np.asarray(energies, dtype=float).reshape(-1)
    if nominal_occupied_count < 1 or nominal_occupied_count >= len(flat_energies):
        raise ValueError("The requested filling is outside the sampled bands.")
    partitioned = np.partition(flat_energies, nominal_occupied_count)
    highest_occupied = np.max(partitioned[:nominal_occupied_count])
    lowest_unoccupied = partitioned[nominal_occupied_count]

    if occupation_mode == "zero-temperature":
        if not np.isclose(state_count, nominal_occupied_count, atol=1.0e-8):
            raise ValueError("A zero-temperature finite-grid filling must contain an integer number of sampled states.")
        internal_energy, chemical_potential, highest_occupied, lowest_unoccupied, = fixed_electron_band_energy(
            energies, nominal_occupied_count, number_kpoints)
        free_energy = internal_energy
    elif occupation_mode == "fermi-dirac":
        chemical_potential = solve_chemical_potential(energies, target_electrons_per_cell, number_kpoints, kBT)
        occupations = fermiDiracOccupations(energies, [chemical_potential], kBT)[0]
        internal_energy = np.sum(occupations * energies) / number_kpoints
        entropy_kernel = np.zeros_like(occupations)
        fractional = (occupations > 0.0) & (occupations < 1.0)
        f = occupations[fractional]
        entropy_kernel[fractional] = f * np.log(f) + (1.0 - f) * np.log1p(-f)
        free_energy =  internal_energy + kBT * np.sum(entropy_kernel) / number_kpoints
    else:
        raise ValueError("occupation_mode must be 'zero-temperature' or 'fermi-dirac'.")

    return {
        "chemical_potential": chemical_potential,
        "internal_energy": internal_energy,
        "free_energy": free_energy,
        "highest_occupied": highest_occupied,
        "lowest_unoccupied": lowest_unoccupied,
        "nominal_occupied_state_count": nominal_occupied_count,
    }

#+--------------------------------------+
#| File I/O for electronic computations |
#+--------------------------------------+
def save_energy_sample(path, direction, theta, phi, bz_grid, target_electrons_per_cell, occupation_mode, kBT, state):
    """Atomically save one completed fixed-number energy calculation."""
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        direction=direction,
        theta=theta,
        phi=phi,
        bz_grid=bz_grid,
        target_electrons_per_cell=target_electrons_per_cell,
        occupation_mode=occupation_mode,
        kBT_eV=kBT,
        chemical_potential_eV=state["chemical_potential"],
        internal_energy_eV_per_cell=state["internal_energy"],
        free_energy_eV_per_cell=state["free_energy"],
        highest_occupied_eV=state["highest_occupied"],
        lowest_unoccupied_eV=state["lowest_unoccupied"],
        nominal_occupied_state_count=state["nominal_occupied_state_count"],
    )
    temporary.replace(path)

def load_energy_sample(path,expected_direction,expected_bz_grid,expected_target_electrons_per_cell,expected_occupation_mode,expected_kBT):
    """Load and validate one fixed-number energy checkpoint."""
    with np.load(path) as data:
        required = {
            "direction",
            "bz_grid",
            "target_electrons_per_cell",
            "occupation_mode",
            "kBT_eV",
            "chemical_potential_eV",
            "internal_energy_eV_per_cell",
            "free_energy_eV_per_cell",
        }
        missing = required.difference(data.files)
        if missing: raise RuntimeError(f"{path} is an obsolete checkpoint; missing {sorted(missing)}.")
        direction = data["direction"]
        assert np.allclose(direction, expected_direction, atol=1.0e-12), (
            f"Sampling direction in {path} does not match this run.")
        signatures_match = (
            np.array_equal(data["bz_grid"], expected_bz_grid)
            and np.isclose(float(data["target_electrons_per_cell"]), expected_target_electrons_per_cell)
            and str(data["occupation_mode"]) == expected_occupation_mode
            and np.isclose(float(data["kBT_eV"]), expected_kBT)
        )
        assert signatures_match,(
                f"Electronic calculation settings in {path} do not match "
                "this run. Remove the sample directory or restore its settings.")
        return {
            "chemical_potential": float(data["chemical_potential_eV"]),
            "internal_energy": float(data["internal_energy_eV_per_cell"]),
            "free_energy": float(data["free_energy_eV_per_cell"]),
            "highest_occupied": float(data["highest_occupied_eV"]),
            "lowest_unoccupied": float(data["lowest_unoccupied_eV"]),
            "nominal_occupied_state_count": int(data["nominal_occupied_state_count"]),
        }

def save_ift_sample(path,direction,chemical_potential,omega_axis,bz_grid,solver_name,occupation_mode,kBT,
                    calculation_signature,projected_chi_ift):
    """Atomically save one completed fixed-number IFT calculation."""
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        direction=direction,
        chemical_potential_eV=chemical_potential,
        omega_axis_eV=omega_axis,
        bz_grid=bz_grid,
        solver_name=solver_name,
        occupation_mode=occupation_mode,
        kBT_eV=kBT,
        checkpoint_format_version=IFT_CHECKPOINT_FORMAT_VERSION,
        calculation_signature=calculation_signature,
        projected_chi_IFT=projected_chi_ift,
    )
    temporary.replace(path)

def load_ift_sample(path,expected_direction,expected_chemical_potential,expected_omega_axis,expected_bz_grid,
                    expected_solver_name,expected_occupation_mode,expected_kBT,expected_calculation_signature):
    """Load and validate one fixed-number IFT checkpoint."""
    global _warned_legacy_ift_checkpoint
    with np.load(path) as data:
        required = {"checkpoint_format_version", "calculation_signature"}
        if not required.issubset(data.files):
            raise StaleIFTCheckpointError(f"{path.name} predates complete IFT checkpoint validation")
        saved_format_version = int(data["checkpoint_format_version"])
        signature_matches = checkpoint_signatures_compatible(
            str(data["calculation_signature"]), expected_calculation_signature, saved_format_version)
        signatures_match = (
            np.allclose(data["direction"], expected_direction, atol=1.0e-12)
            and np.isclose(float(data["chemical_potential_eV"]), expected_chemical_potential)
            and np.array_equal(data["omega_axis_eV"], expected_omega_axis)
            and np.array_equal(data["bz_grid"], expected_bz_grid)
            and str(data["solver_name"]) == expected_solver_name
            and str(data["occupation_mode"]) == expected_occupation_mode
            and np.isclose(float(data["kBT_eV"]), expected_kBT)
            and signature_matches
        )
        if not signatures_match:
            raise StaleIFTCheckpointError(f"IFT settings or implementation changed since {path.name} was written")
        if saved_format_version == 2 and not _warned_legacy_ift_checkpoint:
            warnings.warn(
                "Loading version-2 IFT checkpoints by semantic electronic settings. Their legacy source hash "
                "also covered VSH postprocessing, so postprocessing-only source edits are intentionally ignored.",
                RuntimeWarning)
            _warned_legacy_ift_checkpoint = True
        return data["projected_chi_IFT"].copy()

#+----------------------------------------------------------+
#| Optical contraction to give response at polarized states +
#+----------------------------------------------------------+
def optical_polarization_states():
    """
    Return linear and circular unit Jones vectors in Cartesian coordinates.

    Circular ``ab+`` means ``(e_a + i e_b)/sqrt(2)``.  Responses are
    contracted as ``sum_ab e_a e_b^* chi_ab``.
    """
    cartesian = np.eye(3, dtype=complex)
    labels = ["linear_x", "linear_y", "linear_z"]
    vectors = [cartesian[0], cartesian[1], cartesian[2]]
    for label, ia, ib in (("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)):
        labels.extend((f"circular_{label}+", f"circular_{label}-"))
        vectors.extend((
                (cartesian[ia] + 1j * cartesian[ib]) / np.sqrt(2.0),
                (cartesian[ia] - 1j * cartesian[ib]) / np.sqrt(2.0),
            ))
    return np.asarray(labels), np.asarray(vectors)

def polarization_contracted_diagnostics(reference, reconstructed, polarization_vectors):
    """
    Measure vector-field errors after contraction with optical polarization.

    Input tensor axes must end in ``(drive_1, drive_2, torque_cartesian)``.
    Per-frequency metrics average over orientations and torque components.
    Global metrics additionally combine all frequencies.
    """
    reference = np.asarray(reference)
    reconstructed = np.asarray(reconstructed)
    assert reference.shape == reconstructed.shape, "Reference and reconstructed responses must match."
    assert reference.shape[-3:] == (3, 3, 3), "Response axes must end in (3, 3, 3)."

    polarization_vectors = np.asarray(polarization_vectors, dtype=complex)
    contracted_reference = np.einsum(
        "pa,pb,n...abc->np...c",
        polarization_vectors, polarization_vectors.conj(), reference)
    contracted_reconstructed = np.einsum(
        "pa,pb,n...abc->np...c",
        polarization_vectors, polarization_vectors.conj(), reconstructed)
    error = contracted_reconstructed - contracted_reference

    reference_rms = np.sqrt(np.mean(np.abs(contracted_reference) ** 2, axis=(0, -1)))
    error_rms = np.sqrt(np.mean(np.abs(error) ** 2, axis=(0, -1)))
    relative_rms = np.divide(
        error_rms, reference_rms, out=np.zeros_like(error_rms, dtype=float), where=reference_rms > 0.0
    )
    reference_peak = np.max(np.abs(contracted_reference), axis=(0, -1))
    error_peak = np.max(np.abs(error), axis=(0, -1))
    peak_normalized_error = np.divide(
        error_peak, reference_peak, out=np.zeros_like(error_peak, dtype=float), where=reference_peak > 0.0
    )

    reference_power = np.sum(
        np.abs(contracted_reference) ** 2,
        axis=(0, -2, -1),
    )
    error_power = np.sum(np.abs(error) ** 2, axis=(0, -2, -1))
    global_relative_rms = np.sqrt(
        np.divide(error_power, reference_power, out=np.zeros_like(error_power, dtype=float), where=reference_power > 0.0)
    )
    return {
        "reference_rms": reference_rms,
        "error_rms": error_rms,
        "relative_rms": relative_rms,
        "reference_peak": reference_peak,
        "error_peak": error_peak,
        "peak_normalized_error": peak_normalized_error,
        "global_relative_rms": global_relative_rms,
    }

def crystal_symmetry_operations(model, tolerance=SYMPREC):
    """Return Cartesian point operations and their primitive-cell atom maps."""
    symmetry = getWholeSymmetryOperator(model.posinfo, symprec=tolerance)
    if hasattr(symmetry, "rotations"):
        fractional_rotations = np.asarray(symmetry.rotations, dtype=int)
        translations = np.asarray(symmetry.translations, dtype=float)
    else:
        fractional_rotations = np.asarray(
            symmetry["rotations"],
            dtype=int,
        )
        translations = np.asarray(
            symmetry["translations"],
            dtype=float,
        )
    positions = np.asarray(model.posinfo["pos ions (frac.)"], dtype=float)
    lattice = np.asarray(model.lattice, dtype=float)
    labels = [atom["label"] for atom in model.atomicOrbInfoList]
    cartesian_basis = lattice.T
    cartesian_basis_inverse = np.linalg.inv(cartesian_basis)

    operations = []
    for fractional_rotation, translation in zip(
        fractional_rotations,
        translations,
    ):
        polar_rotation = (
            cartesian_basis
            @ fractional_rotation
            @ cartesian_basis_inverse
        )
        metric_error = np.max(
            abs(polar_rotation @ polar_rotation.T - np.eye(3))
        )
        if metric_error > max(100.0 * tolerance, 1.0e-6):
            raise RuntimeError(
                "Crystal operation is not orthogonal in Cartesian "
                f"coordinates; metric residual={metric_error:.3e}."
            )
        left, _, right = np.linalg.svd(polar_rotation)
        polar_rotation = left @ right
        atom_map = []
        for iatom, position in enumerate(positions):
            transformed = fractional_rotation @ position + translation
            difference = positions - transformed[None, :]
            difference -= np.rint(difference)
            distance = np.linalg.norm(difference @ lattice, axis=1)
            compatible = np.asarray(
                [label == labels[iatom] for label in labels]
            )
            distance[~compatible] = np.inf
            mapped_atom = int(np.argmin(distance))
            if distance[mapped_atom] > tolerance:
                raise RuntimeError(
                    "Could not determine the atom map for a crystal "
                    f"symmetry operation; residual={distance[mapped_atom]:.3e} "
                    f"angstrom for atom {iatom}."
                )
            atom_map.append(mapped_atom)
        operations.append(
            {
                "polar_rotation": polar_rotation,
                "fractional_rotation": fractional_rotation,
                "translation": translation,
                "atom_map": np.asarray(atom_map, dtype=int),
            }
        )
    return operations

def _projection_signature(selection, atomic_orbital_info):
    signature = {}
    for atom_id, shell_ids in selection:
        atom_id = int(atom_id)
        if not 0 <= atom_id < len(atomic_orbital_info):
            raise IndexError(f"Projected atom {atom_id} is out of range.")
        selected = signature.setdefault(atom_id, set())
        for shell_id in shell_ids:
            shell_id = int(shell_id)
            if not 0 <= shell_id < len(
                atomic_orbital_info[atom_id]["orbital index"]
            ):
                raise IndexError(
                    f"Projected shell {shell_id} is out of range for atom "
                    f"{atom_id}."
                )
            selected.add(shell_id)
    return {
        atom_id: tuple(sorted(shell_ids))
        for atom_id, shell_ids in signature.items()
    }

def projection_preserving_operations(
    operations,
    projection,
    atomic_orbital_info,
):
    """
    Select operations preserving source and target atom-shell sets separately.
    """
    source = _projection_signature(
        projection.get("source", []),
        atomic_orbital_info,
    )
    target = _projection_signature(
        projection.get("target", []),
        atomic_orbital_info,
    )

    preserved = []
    for operation in operations:
        atom_map = operation["atom_map"]
        transformed_source = {
            int(atom_map[atom_id]): shell_ids
            for atom_id, shell_ids in source.items()
        }
        transformed_target = {
            int(atom_map[atom_id]): shell_ids
            for atom_id, shell_ids in target.items()
        }
        if transformed_source == source and transformed_target == target:
            preserved.append(operation)
    if not preserved:
        raise RuntimeError(
            "No crystal symmetry operation preserves the requested "
            "projection. The identity should always survive; check the "
            "projection atom and shell indexes."
        )
    return preserved

def fit_projection_resolved_ift_symmetry(
    directions,
    values,
    harmonic_degree,
    rotation_groups,
):
    """Fit every torque-projection slot with its preserving subgroup."""
    values = np.asarray(values)
    if values.ndim < 5 or values.shape[-3:] != (3, 3, 3):
        raise ValueError("IFT values must end in three Cartesian axes.")
    number_projections = values.shape[3]
    if len(rotation_groups) != number_projections:
        raise ValueError(
            "One symmetry rotation group is required per torque projection."
        )

    expansions = []
    diagnostics_by_projection = []
    for iprojection, rotations in enumerate(rotation_groups):
        selected = values[:, :, :, iprojection : iprojection + 1, ...]
        expansion, diagnostics = (
            VectorSphericalHarmonicExpansion.fit_ift_symmetry(
                directions,
                selected,
                harmonic_degree,
                rotations,
                project=False,
            )
        )
        expansions.append(expansion)
        diagnostics_by_projection.append(diagnostics)

    coefficients = np.concatenate(
        [expansion.coefficients for expansion in expansions],
        axis=4,
    )
    expansion = VectorSphericalHarmonicExpansion(
        harmonic_degree,
        expansions[0].modes,
        coefficients,
        values.shape[1:-1],
    )
    diagnostics = {
        "rank": np.asarray(
            [item["rank"] for item in diagnostics_by_projection],
            dtype=int,
        ),
        "condition": np.asarray(
            [item["condition"] for item in diagnostics_by_projection],
            dtype=float,
        ),
        "rms": np.concatenate(
            [item["rms"] for item in diagnostics_by_projection],
            axis=2,
        ),
        "relative_rms": np.concatenate(
            [item["relative_rms"] for item in diagnostics_by_projection],
            axis=2,
        ),
    }
    return expansion, diagnostics

def paired_antipodal_vsh_cross_validation(
    directions,
    values,
    harmonic_degree,
    number_folds=5,
    polar_rotations=None,
):
    """Predict every orientation from fits that exclude its antipodal pair."""
    directions = np.asarray(directions)
    values = np.asarray(values)
    number_orientations = len(directions)
    if number_orientations % 2:
        raise ValueError("Paired-antipodal validation needs an even sample count.")
    number_pairs = number_orientations // 2
    if not np.allclose(
        directions[:number_pairs],
        -directions[number_pairs:],
        atol=1.0e-12,
    ):
        raise ValueError("The second half of directions must be antipodal to the first.")
    number_folds = int(number_folds)
    if not 2 <= number_folds <= number_pairs:
        raise ValueError("number_folds must lie between 2 and the pair count.")

    predictions = np.empty_like(values)
    fold_rank = np.empty(number_folds, dtype=int)
    fold_condition = np.empty(number_folds)
    pair_indices = np.arange(number_pairs)
    for ifold in range(number_folds):
        test_pairs = pair_indices[pair_indices % number_folds == ifold]
        test_indices = np.concatenate((test_pairs, test_pairs + number_pairs))
        training_mask = np.ones(number_orientations, dtype=bool)
        training_mask[test_indices] = False
        if polar_rotations is None:
            expansion, diagnostics = VectorSphericalHarmonicExpansion.fit(
                directions[training_mask],
                values[training_mask],
                harmonic_degree,
                project=False,
            )
        else:
            expansion, diagnostics = (
                VectorSphericalHarmonicExpansion.fit_ift_symmetry(
                    directions[training_mask],
                    values[training_mask],
                    harmonic_degree,
                    polar_rotations,
                    project=False,
                )
            )
        predictions[test_indices] = expansion.evaluate(
            directions[test_indices]
        )
        fold_rank[ifold] = diagnostics["rank"]
        fold_condition[ifold] = diagnostics["condition"]
    return predictions, {
        "rank": fold_rank,
        "condition": fold_condition,
    }

def projected_ift_cross_validation(
    directions,
    values,
    harmonic_degree,
    rotation_groups,
    number_folds=5,
):
    """Cross-validate each projection with the same subgroup used in fitting."""
    predictions = []
    ranks = []
    conditions = []
    for iprojection, rotations in enumerate(rotation_groups):
        selected = values[:, :, :, iprojection : iprojection + 1, ...]
        prediction, diagnostics = paired_antipodal_vsh_cross_validation(
            directions,
            selected,
            harmonic_degree,
            number_folds=number_folds,
            polar_rotations=rotations,
        )
        predictions.append(prediction)
        ranks.append(diagnostics["rank"])
        conditions.append(diagnostics["condition"])
    return np.concatenate(predictions, axis=3), {
        "rank": np.asarray(ranks, dtype=int),
        "condition": np.asarray(conditions, dtype=float),
    }

def ift_symmetry_covariance_diagnostics(
    expansion,
    directions,
    rotation_groups,
):
    """Measure covariance residuals of every fitted projection subgroup."""
    directions = np.asarray(directions, dtype=float)
    operation_errors = []
    for iprojection, rotations in enumerate(rotation_groups):
        selected_coefficients = expansion.coefficients[
            :, :, :, :, iprojection : iprojection + 1, ...
        ]
        selected_shape = list(expansion.value_shape)
        selected_shape[2] = 1
        selected_expansion = VectorSphericalHarmonicExpansion(
            expansion.max_degree,
            expansion.modes,
            selected_coefficients,
            tuple(selected_shape),
        )
        reference = selected_expansion.evaluate(directions)
        errors = []
        for polar_rotation in rotations:
            axial_rotation = axial_rotation_matrices(
                polar_rotation
            )[0]
            transformed_directions = directions @ axial_rotation.T
            left = selected_expansion.evaluate(transformed_directions)
            right = transform_ift_tensor(reference, polar_rotation)
            error_rms = np.sqrt(np.mean(abs(left - right) ** 2))
            reference_rms = np.sqrt(
                0.5 * np.mean(abs(left) ** 2 + abs(right) ** 2)
            )
            errors.append(
                error_rms / reference_rms
                if reference_rms > 0.0
                else 0.0
            )
        operation_errors.append(errors)
    return operation_errors

def global_relative_error_by_projection(reference, reconstructed, projection_axis=3):
    """Return an all-component relative RMS for each projection slot."""
    reference = np.asarray(reference)
    reconstructed = np.asarray(reconstructed)
    assert reference.shape == reconstructed.shape, "Reference and reconstructed responses must match."
    projection_axis = int(projection_axis) % reference.ndim
    reduction_axes = tuple(axis for axis in range(reference.ndim) if axis != projection_axis)
    reference_power = np.sum(np.abs(reference) ** 2, axis=reduction_axes)
    error_power = np.sum(np.abs(reconstructed - reference) ** 2, axis=reduction_axes)
    return np.sqrt(
        np.divide(
            error_power,
            reference_power,
            out=np.zeros_like(reference_power, dtype=float),
            where=reference_power > 0.0,
        )
    )

def output_polarization_validation_table(path, omega_axis, polarization_labels, diagnostics_by_type):
    """Write projection- and frequency-resolved polarization fit errors."""
    with open(path, "w", encoding="utf-8") as output:
        output.write(
            "#validation      polarization     drive_projection_1 drive_projection_2  torque_projection        "
            "fermi_index omega_index           omega_eV      reference_rms          error_rms       "
            "relative_rms     reference_peak         error_peak peak_normalized_error\n"
        )
        leading_shape=diagnostics_by_type["fit"]["relative_rms"].shape[1:-1]
        for iomega, omega in enumerate(omega_axis):
            for leading_index in np.ndindex(leading_shape):
                for ipolarization, label in enumerate(polarization_labels):
                    for validation_name, diagnostics in diagnostics_by_type.items():
                        index = (ipolarization,) + leading_index + (iomega,)
                        output.write(
                            f"{validation_name:16s} {label:16s} "
                            + " ".join(f"{value:18d}" for value in leading_index)
                            + f" {iomega:11d} {omega:18.10e}"
                            f" {diagnostics['reference_rms'][index]:18.10e}"
                            f" {diagnostics['error_rms'][index]:18.10e}"
                            f" {diagnostics['relative_rms'][index]:18.10e}"
                            f" {diagnostics['reference_peak'][index]:18.10e}"
                            f" {diagnostics['error_peak'][index]:18.10e}"
                            f"    {diagnostics['peak_normalized_error'][index]:18.10e}\n"
                        )

def print_polarization_validation_summary(polarization_labels, fit_diagnostics, cross_validation_diagnostics, active_signal_fraction):
    """Print compact full-response diagnostics for each polarization."""
    fit_relative = fit_diagnostics["relative_rms"][:, 0, 0, 0, 0, :]
    cross_relative = cross_validation_diagnostics["relative_rms"][:, 0, 0, 0, 0, :]
    reference_rms = cross_validation_diagnostics["reference_rms"][:, 0, 0, 0, 0, :]
    fit_global = fit_diagnostics["global_relative_rms"][:, 0, 0, 0, 0]
    cross_global = cross_validation_diagnostics["global_relative_rms"][:, 0, 0, 0, 0]

    print(
        "IFT polarization validation for the full response "
        f"(active frequencies >= {active_signal_fraction:.0%} of peak RMS):"
    )
    for ipolarization, label in enumerate(polarization_labels):
        active = reference_rms[ipolarization] >= active_signal_fraction * np.max(reference_rms[ipolarization])
        cross_active = cross_relative[ipolarization, active]
        print(
            f"  {label:16s} fit global={fit_global[ipolarization]:.3%}, "
            f"CV global={cross_global[ipolarization]:.3%}, "
            f"CV active median={np.median(cross_active):.3%}, "
            f"max={np.max(cross_active):.3%}"
        )

def output_energy_table(path, directions, theta, phi, 
                        internal_energy, free_energy, chemical_potential, highest_occupied, lowest_unoccupied):
    relative_energy_meV = (free_energy - np.min(free_energy)) * 1000.0
    with open(path, "w", encoding="utf-8") as output:
        output.write(
            "#index                mx                my                mz "
            "            theta               phi internal_energy_eV_per_cell "
            "free_energy_eV_per_cell relative_free_energy_meV_per_cell "
            "chemical_potential_eV "
            "highest_occupied_eV lowest_unoccupied_eV\n"
        )
        for index, direction in enumerate(directions):
            output.write(
                f"{index:6d}"
                f"{direction[0]:18.10e}{direction[1]:18.10e}"
                f"{direction[2]:18.10e}{theta[index]:18.10e}"
                f"{phi[index]:18.10e}      {internal_energy[index]:22.12e}"
                f"  {free_energy[index]:22.12e}"
                f"            {relative_energy_meV[index]:22.12e}"
                f"    {chemical_potential[index]:18.10e}"
                f"  {highest_occupied[index]:18.10e}"
                f"   {lowest_unoccupied[index]:18.10e}\n"
            )

def run(stage="all", workers_per_node=None, bz_workers_per_orientation=1, shard_rank=None, shard_count=None,
        harmonic_degree_override=None):
    # Electronic calculation.
    main_dir = Path("./mcalc/MnBi2Te4")
    wannierBasis={
        "Fe": (0, 1, 2),
        "Mn": (0, 1, 2),
        "Cl": (0, 1),
        "Bi": (0, 1),
        "Te": (0, 1),
    }
    reference_fermi = 0.95
    reference_direction = np.asarray([0.0, 0.0, 1.0])
    # Set this directly when the electron number is known.  Otherwise it is
    # inferred from reference_fermi at reference_direction.
    fixed_electrons_per_cell = None
    occupation_mode = "zero-temperature" # fermi-dirac | zero-temperature
    electronic_kBT = 0.026
    eta = 0.06582      # eV (6.582 meV), corresponding to hbar/eta about 100 fs
    degen = 1.0e-3     # eV; regularizes r_mn=v_mn/(i DeltaE_mn) inside near-degenerate subspaces

    # Set this only when the fixed moment magnitude is known.  If supplied,
    # the file also contains B_anis=-grad_m(E)/(M mu_B), in tesla.
    moment_magnitude_muB = 5 # mu_B

    omega_axis = np.array([ 0.75, 0.8, 0.85, 2.35, 2.4, 2.45, ])
    bz_grid = (180, 180, 1)
    # Primary parallelism is over magnetization directions.  A hybrid setup
    # is also possible, provided the following worker-count product does not
    # exceed the allocated physical cores.
    number_of_cores, shard_rank, shard_count = distributed_run_settings(
        stage, workers_per_node, shard_rank, shard_count)
    number_of_orientation_workers = number_of_cores
    number_of_bz_workers_per_orientation = int(bz_workers_per_orientation)
    number_of_reference_bz_workers = number_of_cores
    orientation_worker_verbose = False
    torque_projection_list = [(0,(0,1,2,)),]
    torque_projection_specs = [
        #{
        #    "source": torque_projection_list,
        #    "target": torque_projection_list,
        #},
    ]
    ift_solver = solveIFTorque_LG_DM
    mag_type = "spin" # spin | orbital | total
    exchange_provenance = "pauli estimate"

    # Angular representation.  A degree-l tangential VSH expansion has
    # 2*l*(l+2) coefficients for each response-tensor component.
    harmonic_degree = 24
    if harmonic_degree_override is not None:
        harmonic_degree = int(harmonic_degree_override)
        if harmonic_degree < 1: raise ValueError("harmonic_degree must be positive.")
    number_of_orientations = 16*96
    resume_completed_orientations = True
    validation_folds = 5
    validation_active_signal_fraction = 0.05
    enforce_crystal_symmetry = True

    assert occupation_mode in {"zero-temperature", "fermi-dirac"}, "occupation_mode must be 'zero-temperature' or 'fermi-dirac'."
    if occupation_mode == "fermi-dirac" and electronic_kBT <= 0.0: raise ValueError("electronic_kBT must be positive in Fermi-Dirac mode.")
    if number_of_orientation_workers < 1 or number_of_bz_workers_per_orientation < 1:
        raise ValueError("Orientation and inner-BZ worker counts must be positive.")
    if number_of_orientation_workers * number_of_bz_workers_per_orientation > number_of_cores:
        raise ValueError("orientation workers times BZ workers per orientation exceeds number_of_cores.")

    minimum_recommended = 2 * harmonic_degree * (harmonic_degree + 2)
    assert number_of_orientations >= minimum_recommended, f"Use at least {minimum_recommended} orientations for a stable degree-{harmonic_degree} VSH fit."

    model = System_w90(mainDir=str(main_dir) + os.sep, projections=wannierBasis, ispin=2, spinSeparated=True)
    model.symmerizeCollinearAndSOI()
    model.getExchangeR(provenance=exchange_provenance)
    model.setFermi(reference_fermi)
    ift_checkpoint_signature = build_ift_checkpoint_signature(
        main_dir, ift_solver.__name__, occupation_mode, electronic_kBT, eta, degen, mag_type,
        torque_projection_specs, exchange_provenance)
    directions = fibonacci_sphere(number_of_orientations, antipodal=True)
    theta_axis, phi_axis = cartesian_to_spherical(directions)
    selected_indices = (orientation_indices_for_shard(number_of_orientations, shard_rank, shard_count)
                        if stage == "compute" else list(range(number_of_orientations)))
    print(f"Mapping stage={stage}, shard={shard_rank}/{shard_count}, orientations={len(selected_indices)}, "
          f"local workers={number_of_orientation_workers}, BZ workers/orientation={number_of_bz_workers_per_orientation}")
    energy_sample_dir = main_dir / "AngleDependentEnergySamples"
    ift_sample_dir = main_dir / "AngleDependentIFTSamples"
    energy_sample_dir.mkdir(parents=True, exist_ok=True)
    ift_sample_dir.mkdir(parents=True, exist_ok=True)

    internal_energy = np.empty(number_of_orientations)
    free_energy = np.empty(number_of_orientations)
    chemical_potential = np.empty(number_of_orientations)
    highest_occupied = np.empty(number_of_orientations)
    lowest_unoccupied = np.empty(number_of_orientations)
    number_kpoints = int(np.prod(bz_grid))

    # Reference filling.  A metallic target is generally fractional at finite
    # temperature, so store electrons/cell rather than an occupied-band count.
    reference_path = energy_sample_dir / "reference_filling.npz"
    if fixed_electrons_per_cell is not None:
        target_electrons_per_cell = float(fixed_electrons_per_cell)
        filling_source = "explicit electrons/cell"
        print(f"Using explicit filling: {target_electrons_per_cell:.12f} electrons/cell")
    elif (resume_completed_orientations or stage in {"compute", "aggregate"}) and reference_path.exists():
        with np.load(reference_path) as data:
            reference_matches = (
                np.array_equal(data["bz_grid"], np.asarray(bz_grid))
                and np.isclose(
                    float(data["reference_fermi_eV"]),
                    reference_fermi,
                )
                and str(data["occupation_mode"]) == occupation_mode
                and np.isclose(float(data["kBT_eV"]), electronic_kBT)
                and np.allclose(
                    data["reference_direction"],
                    reference_direction,
                    atol=1.0e-12,
                )
            )
            if not reference_matches:
                raise RuntimeError(
                    f"Reference filling settings in {reference_path} do not "
                    "match this run. Remove the energy sample directory or "
                    "restore its settings."
                )
            target_electrons_per_cell = float(
                data["target_electrons_per_cell"]
            )
        print(f"Loaded reference filling: {target_electrons_per_cell:.12f} electrons/cell")
        filling_source = "reference Fermi level"
    elif stage in {"compute", "aggregate"}:
        raise RuntimeError(
            f"Missing {reference_path}. Run '--stage prepare' once before launching compute shards.")
    else:
        reference_theta, reference_phi = cartesian_to_spherical(reference_direction[None, :])
        #exit()
        model.includeSOCFromSOI(
            theta=reference_theta[0],
            phi=reference_phi[0],
            strength=1.0,
            keepRegularSpinAxis=True,
        )
        reference_calc = BZGridTask(model, solveEnergy, args={},
            BZGrid=bz_grid, BZSymm=False,
            mode="run", ncores=number_of_reference_bz_workers)
        reference_energies = collect_bz_energies(reference_calc, bz_grid)
        reference_occupations = fermiDiracOccupations(
            reference_energies, [reference_fermi],
            electronic_kBT if occupation_mode == "fermi-dirac" else 0.0,
        )[0]
        target_electrons_per_cell = (
            np.sum(reference_occupations) / number_kpoints
        )
        temporary = reference_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            reference_direction=reference_direction,
            reference_fermi_eV=reference_fermi,
            bz_grid=np.asarray(bz_grid),
            occupation_mode=occupation_mode,
            kBT_eV=electronic_kBT,
            target_electrons_per_cell=target_electrons_per_cell,
        )
        temporary.replace(reference_path)
        print(f"Determined reference filling: {target_electrons_per_cell:.12f} electrons/cell")
        filling_source = "reference Fermi level"

    if stage == "prepare":
        print(f"Preparation complete: {reference_path}")
        return

    # The model is mutable when its magnetization direction changes.  Give
    # every orientation process a private copy and keep each inner BZ solve
    # serial by default.  On Linux, fork shares the large immutable Wannier
    # input arrays copy-on-write.
    model.removeSOCFromSOI()
    executor = None
    executor_start_method = "not started"
    try:
        # Pass 1: determine mu(m) and the fixed-number energy landscape.
        energy_jobs = []
        loaded_energy_samples = 0
        for index in selected_indices:
            direction, theta, phi = directions[index], theta_axis[index], phi_axis[index]
            sample_path = energy_sample_dir / f"orientation_{index:04d}.npz"
            if (resume_completed_orientations or stage == "aggregate") and sample_path.exists():
                sample = load_energy_sample(
                    sample_path, direction, np.asarray(bz_grid),
                    target_electrons_per_cell, occupation_mode, electronic_kBT)
                internal_energy[index] = sample["internal_energy"]
                free_energy[index] = sample["free_energy"]
                chemical_potential[index] = sample["chemical_potential"]
                highest_occupied[index] = sample["highest_occupied"]
                lowest_unoccupied[index] = sample["lowest_unoccupied"]
                loaded_energy_samples += 1
            elif stage == "aggregate":
                raise RuntimeError(f"Missing energy checkpoint {sample_path}; compute all shards before aggregation.")
            else:
                energy_jobs.append({
                    "index": index, "direction": direction, "theta": theta, "phi": phi,
                    "bz_grid": tuple(bz_grid), "bz_workers": number_of_bz_workers_per_orientation,
                    "number_kpoints": number_kpoints,
                    "target_electrons_per_cell": target_electrons_per_cell,
                    "occupation_mode": occupation_mode, "kBT": electronic_kBT,
                    "worker_verbose": orientation_worker_verbose,
                })
        print(
            f"Energy pass: loaded {loaded_energy_samples}, pending {len(energy_jobs)} "
            f"of {len(selected_indices)} selected orientations."
        )
        if energy_jobs:
            executor, executor_start_method = create_orientation_executor(model, number_of_orientation_workers)
            print(
                f"Orientation backend: {number_of_orientation_workers} process workers "
                f"({executor_start_method}), {number_of_bz_workers_per_orientation} "
                "inner BZ worker(s) per orientation."
            )
            completed = 0
            for index, sample in completed_orientation_jobs(executor, solve_energy_orientation, energy_jobs):
                direction = directions[index]
                save_energy_sample(
                    energy_sample_dir / f"orientation_{index:04d}.npz",
                    direction, theta_axis[index], phi_axis[index], np.asarray(bz_grid),
                    target_electrons_per_cell, occupation_mode, electronic_kBT, sample)
                internal_energy[index] = sample["internal_energy"]
                free_energy[index] = sample["free_energy"]
                chemical_potential[index] = sample["chemical_potential"]
                highest_occupied[index] = sample["highest_occupied"]
                lowest_unoccupied[index] = sample["lowest_unoccupied"]
                completed += 1
                print(
                    f"[energy {completed:4d}/{len(energy_jobs):4d}] orientation "
                    f"{index + 1:4d}, mu={sample['chemical_potential']:.9f} eV"
                )

        # Pass 2: evaluate IFT with the orientation-specific chemical potential
        # and exactly the same occupation convention used in pass 1.
        projected_chi_samples = [None] * number_of_orientations
        ift_jobs = []
        loaded_ift_samples = 0
        stale_ift_samples = []
        for index in selected_indices:
            direction, theta, phi = directions[index], theta_axis[index], phi_axis[index]
            sample_path = ift_sample_dir / f"orientation_{index:04d}.npz"
            if (resume_completed_orientations or stage == "aggregate") and sample_path.exists():
                try:
                    projected_chi_samples[index] = load_ift_sample(
                        sample_path, direction, chemical_potential[index], omega_axis, np.asarray(bz_grid),
                        ift_solver.__name__, occupation_mode, electronic_kBT, ift_checkpoint_signature)
                    loaded_ift_samples += 1
                except StaleIFTCheckpointError as error:
                    stale_ift_samples.append(str(error))
            if projected_chi_samples[index] is None and stage == "aggregate":
                reason = stale_ift_samples[-1] if stale_ift_samples else f"missing checkpoint {sample_path}"
                raise RuntimeError(f"Cannot aggregate orientation {index}: {reason}")
            if projected_chi_samples[index] is None:
                ift_jobs.append({
                    "index": index, "direction": direction, "theta": theta, "phi": phi,
                    "chemical_potential": chemical_potential[index], "omega_axis": omega_axis,
                    "bz_grid": tuple(bz_grid), "bz_workers": number_of_bz_workers_per_orientation,
                    "ift_solver": ift_solver, "mag_type": mag_type,
                    "torque_projection_specs": torque_projection_specs,
                    "occupation_mode": occupation_mode, "kBT": electronic_kBT, "eta": eta, "degen": degen,
                    "worker_verbose": orientation_worker_verbose,
                })
        print(
            f"IFT pass: loaded {loaded_ift_samples}, pending {len(ift_jobs)} "
            f"of {len(selected_indices)} selected orientations."
        )
        if stale_ift_samples:
            print(f"IFT pass: {len(stale_ift_samples)} stale checkpoint(s) will be recomputed. First reason: {stale_ift_samples[0]}")
        if ift_jobs and executor is None:
            executor, executor_start_method = create_orientation_executor(model, number_of_orientation_workers)
            print(
                f"Orientation backend: {number_of_orientation_workers} process workers "
                f"({executor_start_method}), {number_of_bz_workers_per_orientation} "
                "inner BZ worker(s) per orientation."
            )
        if ift_jobs:
            completed = 0
            for index, projected_chi_sample in completed_orientation_jobs(executor, solve_ift_orientation, ift_jobs):
                save_ift_sample(
                    ift_sample_dir / f"orientation_{index:04d}.npz",
                    directions[index], chemical_potential[index], omega_axis, np.asarray(bz_grid),
                    ift_solver.__name__, occupation_mode, electronic_kBT,
                    ift_checkpoint_signature, projected_chi_sample)
                projected_chi_samples[index] = projected_chi_sample
                completed += 1
                print(
                    f"[IFT    {completed:4d}/{len(ift_jobs):4d}] orientation "
                    f"{index + 1:4d}, mu={chemical_potential[index]:.9f} eV"
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if stage == "compute":
        print(f"Compute shard {shard_rank}/{shard_count} complete; checkpoints are ready for aggregation.")
        return

    if stage == "aggregate" and shard_count > 1:
        executor_start_method = "Slurm sharding with node-local process pools"

    projected_chi_ift = np.asarray(projected_chi_samples)
    tangential_chi_ift = project_tangential(projected_chi_ift, directions)

    band_energy = free_energy
    energy_offset = np.mean(band_energy)

    crystal_operations = crystal_symmetry_operations(model)
    if enforce_crystal_symmetry:
        projection_operation_groups = [crystal_operations]
        projection_operation_groups.extend(
            projection_preserving_operations(
                crystal_operations,
                projection,
                model.atomicOrbInfoList,)
            for projection in torque_projection_specs
        )
    else:
        identity_operations = [
            operation
            for operation in crystal_operations
            if np.allclose(
                operation["polar_rotation"],
                np.eye(3),
                atol=1.0e-10,)
        ]
        assert len(identity_operations) == 1, "Expected exactly one identity crystal operation."
        projection_operation_groups = [ identity_operations for _ in range(tangential_chi_ift.shape[3])]

    assert len(projection_operation_groups) == tangential_chi_ift.shape[3], (
            f"The number of torque projection specifications {len(projection_operation_groups)} does not match "
            f"the response projection axis[3]: {tangential_chi_ift.shape=}."
            f"Note: {len(torque_projection_specs)=}"
            )
    rotation_groups = [np.asarray(
            [operation["polar_rotation"] for operation in operations],
            dtype=float,) for operations in projection_operation_groups
    ]
    subgroup_indices = [
        np.asarray([next(index for index, crystal_operation in enumerate(crystal_operations)
                    if crystal_operation is operation) for operation in operations], dtype=int,)
        for operations in projection_operation_groups
    ]

    full_rotations = rotation_groups[0]
    unconstrained_torque_expansion, unconstrained_torque_fit = (
        VectorSphericalHarmonicExpansion.fit(
            directions,
            tangential_chi_ift,
            harmonic_degree,
            project=False,
        )
    )
    if enforce_crystal_symmetry:
        energy_expansion, energy_fit = (
            SphericalHarmonicExpansion.fit_scalar_axial_symmetry(
                directions,
                band_energy - energy_offset,
                harmonic_degree,
                full_rotations,
            )
        )
        torque_expansion, torque_fit = (
            fit_projection_resolved_ift_symmetry(
                directions,
                tangential_chi_ift,
                harmonic_degree,
                rotation_groups,
            )
        )
    else:
        energy_expansion, energy_fit = SphericalHarmonicExpansion.fit(
            directions,
            band_energy - energy_offset,
            harmonic_degree,
        )
        torque_expansion = unconstrained_torque_expansion
        torque_fit = unconstrained_torque_fit
    torque_reconstructed = torque_expansion.evaluate(directions)
    unconstrained_torque_reconstructed = (
        unconstrained_torque_expansion.evaluate(directions)
    )
    torque_fit_global_relative_rms = global_relative_error_by_projection(tangential_chi_ift, torque_reconstructed)
    unconstrained_torque_fit_global_relative_rms = global_relative_error_by_projection(tangential_chi_ift, unconstrained_torque_reconstructed)
    
    torque_cross_validated, torque_cross_validation_fit = (
        projected_ift_cross_validation(
            directions,
            tangential_chi_ift,
            harmonic_degree,
            rotation_groups,
            number_folds=validation_folds,
        )
    )
    covariance_test_directions = fibonacci_sphere(64)
    pre_enforcement_symmetry_covariance_errors = (
        ift_symmetry_covariance_diagnostics(
            unconstrained_torque_expansion,
            covariance_test_directions,
            rotation_groups,
        )
    )
    symmetry_covariance_errors = ift_symmetry_covariance_diagnostics(
        torque_expansion,
        covariance_test_directions,
        rotation_groups,
    )
    pre_enforcement_symmetry_covariance_rms = np.asarray([
            np.sqrt(np.mean(np.square(errors))) for errors in pre_enforcement_symmetry_covariance_errors])
    pre_enforcement_symmetry_covariance_max = np.asarray([
            np.max(errors) for errors in pre_enforcement_symmetry_covariance_errors])
    symmetry_covariance_rms = np.asarray([
            np.sqrt(np.mean(np.square(errors))) for errors in symmetry_covariance_errors])
    symmetry_covariance_max = np.asarray([np.max(errors) for errors in symmetry_covariance_errors])
    polarization_labels, polarization_vectors = optical_polarization_states()
    polarization_fit = polarization_contracted_diagnostics(tangential_chi_ift, torque_reconstructed, polarization_vectors)
    polarization_cross_validation = polarization_contracted_diagnostics(
        tangential_chi_ift, torque_cross_validated, polarization_vectors)

    energy_gradient = np.real_if_close(energy_expansion.surface_gradient(directions))
    half_orientations = number_of_orientations // 2
    energy_reversal_odd_meV = 0.5 * (band_energy[:half_orientations] - band_energy[half_orientations:]) * 1000.0
    
    if moment_magnitude_muB is None:
        anisotropy_field_tesla = np.empty((0, 3))
    else:
        moment_magnitude_muB = float(moment_magnitude_muB)
        if moment_magnitude_muB <= 0.0:
            raise ValueError("moment_magnitude_muB must be positive.")
        anisotropy_field_tesla = -energy_gradient / (moment_magnitude_muB * BOHR_MAGNETON_EV_PER_TESLA)

    output_energy_table(
        main_dir / "AngleDependentEnergy.txt",
        directions, theta_axis, phi_axis,
        internal_energy, free_energy, chemical_potential, highest_occupied, lowest_unoccupied,
    )
    output_polarization_validation_table(
        main_dir / "AngleDependentIFTFitValidation.txt",
        omega_axis,
        polarization_labels,
        {
            "fit": polarization_fit,
            "cross_validation": polarization_cross_validation,
        },
    )

    subgroup_counts = np.asarray([len(rotations) for rotations in rotation_groups], dtype=int)
    maximum_subgroup_size = int(np.max(subgroup_counts))
    subgroup_rotations = np.full((len(rotation_groups), maximum_subgroup_size, 3, 3), np.nan)
    subgroup_operation_indices = np.full((len(rotation_groups), maximum_subgroup_size), -1, dtype=int)
    for iprojection, (rotations, indices) in enumerate(zip(rotation_groups, subgroup_indices)):
        subgroup_rotations[iprojection, : len(rotations)] = rotations
        subgroup_operation_indices[iprojection, : len(indices)] = indices

    output_path = main_dir / "AngleDependentDynamicsMap.npz"
    np.savez_compressed(
        output_path,
        format_version=3,
        directions=directions,
        orientation_weights=np.full(number_of_orientations, 4.*np.pi/number_of_orientations),
        theta=theta_axis, phi=phi_axis, reference_direction=reference_direction,
        reference_fermi_eV=reference_fermi,
        filling_source=filling_source,
        chemical_potential_eV=chemical_potential,
        fermi_axis_eV=chemical_potential[:, None],
        omega_axis_eV=omega_axis,
        bz_grid=np.asarray(bz_grid), occupation_mode=occupation_mode, kBT_eV=electronic_kBT,
        eta_eV=eta, degen_eV=degen,
        number_of_cores=number_of_cores,
        number_of_orientation_workers=number_of_orientation_workers,
        number_of_bz_workers_per_orientation=number_of_bz_workers_per_orientation,
        slurm_shard_count=shard_count,
        orientation_process_start_method=executor_start_method,
        nominal_occupied_state_count=int(round(target_electrons_per_cell * number_kpoints)),
        electrons_per_cell=target_electrons_per_cell,
        internal_energy_eV_per_cell=internal_energy,
        free_energy_eV_per_cell=free_energy,
        band_energy_eV_per_cell=band_energy,
        energy_offset_eV_per_cell=energy_offset,
        energy_surface_gradient_eV_per_cell=energy_gradient,
        energy_reversal_odd_meV_per_cell=energy_reversal_odd_meV,
        moment_magnitude_muB= np.nan if moment_magnitude_muB is None else moment_magnitude_muB,
        anisotropy_field_tesla=anisotropy_field_tesla,
        projected_chi_IFT=projected_chi_ift,
        tangential_projected_chi_IFT=tangential_chi_ift,
        harmonic_degree=harmonic_degree,
        energy_harmonic_modes=energy_expansion.modes,
        energy_harmonic_coefficients=energy_expansion.coefficients,
        energy_fit_rank=energy_fit["rank"],
        energy_fit_condition=energy_fit["condition"],
        energy_fit_rms_eV=energy_fit["rms"],
        energy_fit_relative_rms=energy_fit["relative_rms"],
        torque_harmonic_modes=torque_expansion.modes,
        torque_vsh_coefficients=torque_expansion.coefficients,
        torque_fit_rank=torque_fit["rank"],
        torque_fit_condition=torque_fit["condition"],
        torque_fit_rms=torque_fit["rms"],
        torque_fit_relative_rms=torque_fit["relative_rms"],
        torque_fit_global_relative_rms_by_projection=torque_fit_global_relative_rms,
        torque_unconstrained_fit_rank=unconstrained_torque_fit["rank"],
        torque_unconstrained_fit_condition=unconstrained_torque_fit["condition"],
        torque_unconstrained_fit_rms=unconstrained_torque_fit["rms"],
        torque_unconstrained_fit_relative_rms=unconstrained_torque_fit["relative_rms"],
        torque_unconstrained_fit_global_relative_rms_by_projection=unconstrained_torque_fit_global_relative_rms,
        torque_cross_validation_prediction=torque_cross_validated,
        torque_cross_validation_folds=validation_folds,
        torque_cross_validation_fold_rank=torque_cross_validation_fit["rank"],
        torque_cross_validation_fold_condition=torque_cross_validation_fit["condition"],
        crystal_symmetry_enforced=enforce_crystal_symmetry,
        crystal_symmetry_polar_rotations=np.asarray([operation["polar_rotation"] for operation in crystal_operations]),
        crystal_symmetry_fractional_rotations=np.asarray([operation["fractional_rotation"] for operation in crystal_operations]),
        crystal_symmetry_translations=np.asarray([operation["translation"] for operation in crystal_operations]),
        crystal_symmetry_atom_maps=np.asarray([operation["atom_map"] for operation in crystal_operations]),
        torque_projection_symmetry_operation_counts=subgroup_counts,
        torque_projection_symmetry_operation_indices=subgroup_operation_indices,
        torque_projection_symmetry_polar_rotations=subgroup_rotations,
        torque_projection_specs_json=json.dumps(torque_projection_specs, sort_keys=True),
        torque_pre_enforcement_symmetry_covariance_relative_rms=pre_enforcement_symmetry_covariance_rms,
        torque_pre_enforcement_symmetry_covariance_relative_max=pre_enforcement_symmetry_covariance_max,
        torque_symmetry_covariance_relative_rms=symmetry_covariance_rms,
        torque_symmetry_covariance_relative_max=symmetry_covariance_max,
        polarization_labels=polarization_labels,
        polarization_vectors=polarization_vectors,
        polarization_active_signal_fraction=validation_active_signal_fraction,
        polarization_fit_reference_rms=polarization_fit["reference_rms"],
        polarization_fit_error_rms=polarization_fit["error_rms"],
        polarization_fit_relative_rms=polarization_fit["relative_rms"],
        polarization_fit_reference_peak=polarization_fit["reference_peak"],
        polarization_fit_error_peak=polarization_fit["error_peak"],
        polarization_fit_peak_normalized_error=polarization_fit["peak_normalized_error"],
        polarization_fit_global_relative_rms=polarization_fit["global_relative_rms"],
        polarization_cv_reference_rms=polarization_cross_validation["reference_rms"],
        polarization_cv_error_rms=polarization_cross_validation["error_rms"],
        polarization_cv_relative_rms=polarization_cross_validation["relative_rms"],
        polarization_cv_reference_peak=polarization_cross_validation["reference_peak"],
        polarization_cv_error_peak=polarization_cross_validation["error_peak"],
        polarization_cv_peak_normalized_error=polarization_cross_validation["peak_normalized_error"],
        polarization_cv_global_relative_rms=polarization_cross_validation["global_relative_rms"],
    )

    metadata = {
        "format_version": 3,
        "sampling": "equal-area antipodal Fibonacci sphere",
        "number_of_orientations": number_of_orientations,
        "harmonic_degree": harmonic_degree,
        "energy_model": (
            "fixed-electron-number force-theorem electronic free energy; "
            "for Fermi-Dirac occupations this includes the independent-"
            "electron entropy term and is not a self-consistent DFT total "
            "energy"
        ),
        "two_pass_mapping": (
            "pass 1 solves the orientation-dependent chemical potential and "
            "free energy at fixed electrons/cell; pass 2 evaluates IFT using "
            "that chemical potential and the same occupation convention"
        ),
        "occupation_mode": occupation_mode,
        "kBT_eV": electronic_kBT,
        "eta_eV": eta,
        "degen_eV": degen,
        "reference_fermi_eV": reference_fermi,
        "filling_source": filling_source,
        "target_electrons_per_cell": target_electrons_per_cell,
        "parallelization": {
            "number_of_cores": number_of_cores,
            "orientation_process_workers": number_of_orientation_workers,
            "bz_workers_per_orientation": number_of_bz_workers_per_orientation,
            "reference_bz_workers": number_of_reference_bz_workers,
            "process_start_method": executor_start_method,
            "worker_verbose": orientation_worker_verbose,
            "slurm_shard_count": shard_count,
        },
        "projected_chi_IFT_axes": [
            "orientation",
            "drive_projection_1",
            "drive_projection_2",
            "torque_projection",
            "fermi",
            "omega",
            "drive_cartesian_1",
            "drive_cartesian_2",
            "torque_cartesian",
        ],
        "torque_vsh_coefficient_axes": [
            "family_0_poloidal_1_toroidal",
            "degree_order_mode",
            "drive_projection_1",
            "drive_projection_2",
            "torque_projection",
            "fermi",
            "omega",
            "drive_cartesian_1",
            "drive_cartesian_2",
        ],
        "anisotropy_field": (
            "B_anis=-surface_gradient(E)/(moment_magnitude_muB*mu_B); "
            "the array is empty when moment_magnitude_muB is not supplied"
        ),
        "torque_note": (
            "IFT values remain electronic response coefficients. Contract "
            "with the complex optical field and convert units before LLG."
        ),
        "crystal_symmetry": (
            "Scalar energy and tangent IFT VSH fits are augmented by complete "
            "spatial-symmetry orbits. The total-torque slot uses the full "
            "crystallographic point group. Each projected slot uses the "
            "subgroup that preserves its source and target atom-shell sets "
            "separately. Electric-field axes transform as polar vectors; "
            "moment direction and torque transform as axial vectors."
        ),
        "crystal_symmetry_enforced": enforce_crystal_symmetry,
        "crystal_point_group_order": len(crystal_operations),
        "torque_projection_specs": torque_projection_specs,
        "torque_projection_subgroup_orders": subgroup_counts.tolist(),
        "symmetry_diagnostics": (
            "The pre-enforcement covariance residual is measured from the "
            "ordinary unconstrained VSH fit. The post-enforcement residual "
            "is measured from the saved symmetry-constrained fit. Global "
            "fit errors combine all tensor entries and frequencies within "
            "each torque-projection slot."
        ),
        "polarization_validation": (
            "The fit metrics reconstruct all samples. Cross-validation uses "
            f"{validation_folds} folds and keeps each antipodal pair in the "
            "same held-out fold. Circular ab+ uses "
            "(e_a+i e_b)/sqrt(2), and responses are contracted as "
            "sum_ab e_a e_b^* chi_ab. Console active-frequency summaries "
            f"use a {validation_active_signal_fraction:.1%} signal threshold."
        ),
    }
    with open(main_dir / "AngleDependentDynamicsMap.json", "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)

    relative_energy_meV = (band_energy - np.min(band_energy)) * 1000.0
    torque_relative_rms = np.asarray(torque_fit["relative_rms"])
    torque_fit_condition = np.atleast_1d(torque_fit["condition"])
    print(f"Saved {output_path}")
    print(f"Band-energy anisotropy range: {np.ptp(relative_energy_meV):.9e} meV/cell")
    print(f"Energy reversal-odd residual: RMS={np.sqrt(np.mean(energy_reversal_odd_meV**2)):.6e}, max={np.max(np.abs(energy_reversal_odd_meV)):.6e} meV/cell")
    print(f"Energy fit: condition={energy_fit['condition']:.6e}, relative RMS={float(energy_fit['relative_rms']):.6e}")
    print(
        "IFT VSH fit: condition range="
        f"{np.min(torque_fit_condition):.6e}--"
        f"{np.max(torque_fit_condition):.6e}, "
        f"median relative RMS={np.median(torque_relative_rms):.6e}, "
        f"maximum relative RMS={np.max(torque_relative_rms):.6e}"
    )
    print(
        "IFT symmetry before enforcement: relative RMS="
        f"{pre_enforcement_symmetry_covariance_rms.tolist()}, "
        "relative max="
        f"{pre_enforcement_symmetry_covariance_max.tolist()}"
    )
    print(
        "IFT symmetry after enforcement: subgroup orders="
        f"{subgroup_counts.tolist()}, relative RMS="
        f"{symmetry_covariance_rms.tolist()}, relative max="
        f"{symmetry_covariance_max.tolist()}"
    )
    print(
        "IFT global fit error by projection: unconstrained="
        f"{unconstrained_torque_fit_global_relative_rms.tolist()}, "
        "symmetry-constrained="
        f"{torque_fit_global_relative_rms.tolist()}"
    )
    print(
        "IFT VSH cross-validation: "
        f"fold condition min="
        f"{np.min(torque_cross_validation_fit['condition']):.6e}, "
        f"max={np.max(torque_cross_validation_fit['condition']):.6e}"
    )
    print_polarization_validation_summary(
        polarization_labels,
        polarization_fit,
        polarization_cross_validation,
        validation_active_signal_fraction,
    )
if __name__ == "__main__":
    arguments = command_line_arguments()
    run(stage=arguments.stage, workers_per_node=arguments.workers_per_node,
        bz_workers_per_orientation=arguments.bz_workers_per_orientation,
        shard_rank=arguments.shard_rank, shard_count=arguments.shard_count,
        harmonic_degree_override=arguments.harmonic_degree)
