import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from pathlib import Path

import numpy as np

import paramReader

paramReader.readGlobalConst()

from SphericalHarmonics import (
    SphericalHarmonicExpansion,
    VectorSphericalHarmonicExpansion,
    cartesian_to_spherical,
    fibonacci_sphere,
    project_tangential,
)
from system import System_w90
from task import BZGridTask, solveIFTorque_LG_DM


BOHR_MAGNETON_EV_PER_TESLA = 5.7883818060e-5


def collect_bz_results(calc, bz_grid):
    """Assemble full-grid energies and average the response over phase grids."""
    _, _, _, number_of_bands = calc.results[0]["energies"].shape
    energies = np.zeros(tuple(bz_grid) + (number_of_bands,), dtype=float)
    projected_chi_ift = None

    for result in calc.results:
        iq = result["iq"]
        energies[
            iq[0] :: calc.QGrid[0],
            iq[1] :: calc.QGrid[1],
            iq[2] :: calc.QGrid[2],
        ] = result["energies"]

        if "projected sigma" in result:
            if projected_chi_ift is None:
                projected_chi_ift = np.zeros_like(
                    result["projected sigma"], dtype=complex
                )
            projected_chi_ift += result["projected sigma"]

    if projected_chi_ift is None:
        raise RuntimeError("The IFT task did not return projected sigma.")
    projected_chi_ift /= len(calc.results)
    return energies, projected_chi_ift


def fixed_electron_band_energy(energies, occupied_state_count, number_kpoints):
    """
    Return the zero-temperature force-theorem band energy at fixed filling.

    ``occupied_state_count`` counts states over the complete finite BZ grid.
    The result is energy per primitive cell.
    """
    flat_energies = np.asarray(energies).reshape(-1)
    occupied_state_count = int(occupied_state_count)
    if occupied_state_count < 1 or occupied_state_count >= len(flat_energies):
        raise ValueError(
            "occupied_state_count must lie between zero and the total "
            "number of sampled states."
        )

    partitioned = np.partition(flat_energies, occupied_state_count)
    occupied = partitioned[:occupied_state_count]
    highest_occupied = np.max(occupied)
    lowest_unoccupied = partitioned[occupied_state_count]
    discrete_fermi = 0.5 * (highest_occupied + lowest_unoccupied)
    return (
        np.sum(occupied) / number_kpoints,
        discrete_fermi,
        highest_occupied,
        lowest_unoccupied,
    )


def save_orientation_sample(
    path,
    direction,
    theta,
    phi,
    fermi_axis,
    omega_axis,
    bz_grid,
    solver_name,
    occupied_state_count,
    band_energy,
    discrete_fermi,
    highest_occupied,
    lowest_unoccupied,
    projected_chi_ift,
):
    """Atomically save one expensive electronic-structure orientation."""
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        direction=direction,
        theta=theta,
        phi=phi,
        fermi_axis_eV=fermi_axis,
        omega_axis_eV=omega_axis,
        bz_grid=bz_grid,
        solver_name=solver_name,
        occupied_state_count=occupied_state_count,
        band_energy_eV_per_cell=band_energy,
        discrete_fermi_eV=discrete_fermi,
        highest_occupied_eV=highest_occupied,
        lowest_unoccupied_eV=lowest_unoccupied,
        projected_chi_IFT=projected_chi_ift,
    )
    temporary.replace(path)


def load_orientation_sample(
    path,
    expected_direction,
    expected_fermi_axis,
    expected_omega_axis,
    expected_bz_grid,
    expected_solver_name,
):
    """Load and validate a previously completed orientation."""
    with np.load(path) as data:
        required = {
            "direction",
            "fermi_axis_eV",
            "omega_axis_eV",
            "bz_grid",
            "solver_name",
        }
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(
                f"{path} is an obsolete checkpoint; missing {sorted(missing)}."
            )
        direction = data["direction"]
        if not np.allclose(direction, expected_direction, atol=1.0e-12):
            raise RuntimeError(
                f"Sampling direction in {path} does not match this run."
            )
        signatures_match = (
            np.array_equal(data["fermi_axis_eV"], expected_fermi_axis)
            and np.array_equal(data["omega_axis_eV"], expected_omega_axis)
            and np.array_equal(data["bz_grid"], expected_bz_grid)
            and str(data["solver_name"]) == expected_solver_name
        )
        if not signatures_match:
            raise RuntimeError(
                f"Electronic calculation settings in {path} do not match "
                "this run. Remove the sample directory or restore its settings."
            )
        return {
            "occupied_state_count": int(data["occupied_state_count"]),
            "band_energy": float(data["band_energy_eV_per_cell"]),
            "discrete_fermi": float(data["discrete_fermi_eV"]),
            "highest_occupied": float(data["highest_occupied_eV"]),
            "lowest_unoccupied": float(data["lowest_unoccupied_eV"]),
            "projected_chi_IFT": data["projected_chi_IFT"].copy(),
        }


def output_energy_table(
    path,
    directions,
    theta,
    phi,
    band_energy,
    discrete_fermi,
    highest_occupied,
    lowest_unoccupied,
):
    relative_energy_meV = (band_energy - np.min(band_energy)) * 1000.0
    with open(path, "w", encoding="utf-8") as output:
        output.write(
            "# index mx my mz theta phi band_energy_eV_per_cell "
            "relative_energy_meV_per_cell discrete_fermi_eV "
            "highest_occupied_eV lowest_unoccupied_eV\n"
        )
        for index, direction in enumerate(directions):
            output.write(
                f"{index:6d}"
                f"{direction[0]:18.10e}{direction[1]:18.10e}"
                f"{direction[2]:18.10e}{theta[index]:18.10e}"
                f"{phi[index]:18.10e}{band_energy[index]:22.12e}"
                f"{relative_energy_meV[index]:22.12e}"
                f"{discrete_fermi[index]:18.10e}"
                f"{highest_occupied[index]:18.10e}"
                f"{lowest_unoccupied[index]:18.10e}\n"
            )


def run():
    # Electronic calculation.
    main_dir = Path("./mcalc/FeCl2")
    fermi = -2.5
    fermi_axis = np.asarray([fermi])
    omega_axis = np.linspace(0.5, 2.0, 21)
    bz_grid = (48, 48, 1)
    number_of_cores = 8
    torque_projection_list = []
    ift_solver = solveIFTorque_LG_DM

    # Angular representation.  For l_max=4 the tangential VSH expansion has
    # 48 coefficients.  Fifty orientations provide 100 tangential data
    # values for each response-tensor component.
    harmonic_degree = 4
    number_of_orientations = 50
    resume_completed_orientations = True

    # Set this only when the fixed moment magnitude is known.  If supplied,
    # the file also contains B_anis=-grad_m(E)/(M mu_B), in tesla.
    moment_magnitude_muB = None

    minimum_recommended = 2 * harmonic_degree * (harmonic_degree + 2)
    if number_of_orientations < minimum_recommended:
        raise ValueError(
            f"Use at least {minimum_recommended} orientations for a stable "
            f"degree-{harmonic_degree} VSH fit."
        )

    model = System_w90(
        mainDir=str(main_dir) + os.sep,
        cutoff=-0.002,
        poscar="POSCAR",
        projections={
            "Fe": (0, 1, 2),
            "Cl": (0, 1),
        },
        ispin=2,
        spinSeparated=True,
    )
    model.symmerizeCollinearAndSOI()
    model.setFermi(fermi)

    directions = fibonacci_sphere(number_of_orientations, antipodal=True)
    theta_axis, phi_axis = cartesian_to_spherical(directions)
    sample_dir = main_dir / "AngleDependentDynamicsSamples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    band_energy = np.empty(number_of_orientations)
    discrete_fermi = np.empty(number_of_orientations)
    highest_occupied = np.empty(number_of_orientations)
    lowest_unoccupied = np.empty(number_of_orientations)
    projected_chi_samples = []
    occupied_state_count = None
    number_kpoints = int(np.prod(bz_grid))

    for index, (direction, theta, phi) in enumerate(
        zip(directions, theta_axis, phi_axis)
    ):
        sample_path = sample_dir / f"orientation_{index:04d}.npz"
        if resume_completed_orientations and sample_path.exists():
            sample = load_orientation_sample(
                sample_path,
                direction,
                fermi_axis,
                omega_axis,
                np.asarray(bz_grid),
                ift_solver.__name__,
            )
            if occupied_state_count is None:
                occupied_state_count = sample["occupied_state_count"]
            elif occupied_state_count != sample["occupied_state_count"]:
                raise RuntimeError("Inconsistent filling in saved samples.")
            print(
                f"[{index + 1:4d}/{number_of_orientations:4d}] "
                f"loaded {sample_path.name}"
            )
        else:
            print(
                f"[{index + 1:4d}/{number_of_orientations:4d}] "
                f"m=({direction[0]: .7f}, {direction[1]: .7f}, "
                f"{direction[2]: .7f})"
            )
            model.includeSOCFromSOI(
                theta=theta,
                phi=phi,
                strength=1.0,
                keepRegularSpinAxis=True,
            )
            calc = BZGridTask(
                model,
                ift_solver,
                args={
                    "fermi": fermi_axis,
                    "omega": omega_axis,
                    "mag type": "total",
                    "projected shells for detecting torque":
                        torque_projection_list,
                },
                BZGrid=bz_grid,
                BZSymm=False,
                mode="run",
                ncores=number_of_cores,
            )
            energies, projected_chi_ift = collect_bz_results(calc, bz_grid)

            if occupied_state_count is None: occupied_state_count = int(np.count_nonzero(energies <= fermi))
            energy_value, fermi_value, highest_value, lowest_value = fixed_electron_band_energy(energies, occupied_state_count, number_kpoints)
            sample = {
                "occupied_state_count": occupied_state_count,
                "band_energy": energy_value,
                "discrete_fermi": fermi_value,
                "highest_occupied": highest_value,
                "lowest_unoccupied": lowest_value,
                "projected_chi_IFT": projected_chi_ift,
            }
            save_orientation_sample(
                sample_path,
                direction,
                theta,
                phi,
                fermi_axis,
                omega_axis,
                np.asarray(bz_grid),
                ift_solver.__name__,
                occupied_state_count,
                energy_value,
                fermi_value,
                highest_value,
                lowest_value,
                projected_chi_ift,
            )

        band_energy[index] = sample["band_energy"]
        discrete_fermi[index] = sample["discrete_fermi"]
        highest_occupied[index] = sample["highest_occupied"]
        lowest_unoccupied[index] = sample["lowest_unoccupied"]
        projected_chi_samples.append(sample["projected_chi_IFT"])

    projected_chi_ift = np.asarray(projected_chi_samples)
    tangential_chi_ift = project_tangential(
        projected_chi_ift, directions
    )

    energy_offset = np.mean(band_energy)
    energy_expansion, energy_fit = SphericalHarmonicExpansion.fit(
        directions,
        band_energy - energy_offset,
        harmonic_degree,
    )
    torque_expansion, torque_fit = (
        VectorSphericalHarmonicExpansion.fit(
            directions,
            tangential_chi_ift,
            harmonic_degree,
            project=False,
        )
    )

    energy_gradient = np.real_if_close(
        energy_expansion.surface_gradient(directions)
    )
    half_orientations = number_of_orientations // 2
    energy_reversal_odd_meV = 0.5 * (band_energy[:half_orientations] - band_energy[half_orientations:]) * 1000.0
    
    if moment_magnitude_muB is None:
        anisotropy_field_tesla = np.empty((0, 3))
    else:
        moment_magnitude_muB = float(moment_magnitude_muB)
        if moment_magnitude_muB <= 0.0:
            raise ValueError("moment_magnitude_muB must be positive.")
        anisotropy_field_tesla = (
            -energy_gradient
            / (moment_magnitude_muB * BOHR_MAGNETON_EV_PER_TESLA)
        )

    output_energy_table(
        main_dir / "AngleDependentEnergy.txt",
        directions,
        theta_axis,
        phi_axis,
        band_energy,
        discrete_fermi,
        highest_occupied,
        lowest_unoccupied,
    )

    output_path = main_dir / "AngleDependentDynamicsMap.npz"
    np.savez_compressed(
        output_path,
        format_version=1,
        directions=directions,
        orientation_weights=np.full(
            number_of_orientations,
            4.0 * np.pi / number_of_orientations,
        ),
        theta=theta_axis,
        phi=phi_axis,
        fermi_axis_eV=fermi_axis,
        omega_axis_eV=omega_axis,
        bz_grid=np.asarray(bz_grid),
        occupied_state_count=occupied_state_count,
        electrons_per_cell=occupied_state_count / number_kpoints,
        band_energy_eV_per_cell=band_energy,
        energy_offset_eV_per_cell=energy_offset,
        energy_surface_gradient_eV_per_cell=energy_gradient,
        energy_reversal_odd_meV_per_cell=energy_reversal_odd_meV,
        moment_magnitude_muB=(
            np.nan if moment_magnitude_muB is None else moment_magnitude_muB
        ),
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
    )

    metadata = {
        "format_version": 1,
        "sampling": "equal-area antipodal Fibonacci sphere",
        "number_of_orientations": number_of_orientations,
        "harmonic_degree": harmonic_degree,
        "energy_model": (
            "zero-temperature force-theorem band energy at fixed electron "
            "number; this is not a self-consistent DFT total energy"
        ),
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
    }
    with open(main_dir / "AngleDependentDynamicsMap.json", "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)

    relative_energy_meV = (band_energy - np.min(band_energy)) * 1000.0
    torque_relative_rms = np.asarray(torque_fit["relative_rms"])
    print(f"Saved {output_path}")
    print(f"Band-energy anisotropy range: {np.ptp(relative_energy_meV):.9e} meV/cell")
    print(f"Energy reversal-odd residual: RMS={np.sqrt(np.mean(energy_reversal_odd_meV**2)):.6e}, max={np.max(np.abs(energy_reversal_odd_meV)):.6e} meV/cell")
    print(f"Energy fit: condition={energy_fit['condition']:.6e}, relative RMS={float(energy_fit['relative_rms']):.6e}")
    print(
        f"IFT VSH fit: condition={torque_fit['condition']:.6e}, "
        f"median relative RMS={np.median(torque_relative_rms):.6e}, "
        f"maximum relative RMS={np.max(torque_relative_rms):.6e}"
    )


if __name__ == "__main__":
    run()
