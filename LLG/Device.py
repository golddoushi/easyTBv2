from collections.abc import Callable, Sequence
from pathlib import Path
import numpy as np
from LLG import Field, SingleMomentLLG, Torque
from SphericalHarmonics import SphericalHarmonicExpansion, VectorSphericalHarmonicExpansion

from auxiliary import BOHR_MAGNETON_EV_PER_TESLA, HBAR_EV_FS, ELECTRON_GAMMA_RAD_PER_FS_T

def _initial_direction(theta: float, phi: float) -> np.ndarray:
    return np.array([np.sin(theta) * np.cos(phi),np.sin(theta) * np.sin(phi),np.cos(theta)])

def _as_field(value) -> Field:
    if isinstance(value, Field):
        return value
    if callable(value):
        return Field(value)
    raise TypeError("additional_field must be a Field or callable.")


def _as_torque(value) -> Torque:
    if isinstance(value, Torque):
        return value
    if callable(value):
        return Torque(value)
    raise TypeError("additional_torque must be a Torque or callable.")


def _polarization_vector(polarization) -> np.ndarray:
    aliases = {
        "x": np.array([1.0, 0.0, 0.0], dtype=complex),
        "y": np.array([0.0, 1.0, 0.0], dtype=complex),
        "z": np.array([0.0, 0.0, 1.0], dtype=complex),
    }
    for plane, ia, ib in (("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)):
        plus = np.zeros(3, dtype=complex)
        plus[ia] = 1.0
        plus[ib] = 1j
        aliases[f"{plane}+"] = plus
        aliases[f"circular_{plane}+"] = plus
        aliases[f"{plane}-"] = plus.conj()
        aliases[f"circular_{plane}-"] = plus.conj()
    aliases.update(
        {
            "linear_x": aliases["x"],
            "linear_y": aliases["y"],
            "linear_z": aliases["z"],
        }
    )

    if isinstance(polarization, str):
        try:
            vector = aliases[polarization.lower()].copy()
        except KeyError as error:
            raise ValueError(
                "Unknown polarization. Use x, y, z, xy+/-, xz+/-, yz+/-, "
                "or a complex Cartesian Jones vector."
            ) from error
    else:
        vector = np.asarray(polarization, dtype=complex)
        if vector.shape != (3,):
            raise ValueError("polarization must have shape (3,).")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("polarization must be finite and nonzero.")
    return vector / norm



class SOTDevice:
    def __init__(
        self,
        H_K=np.array([0, 0, 1]),
        H_ext=np.array([-0.3, 0.0, 0.0]),
        H_FL=0.10,
        H_DL=0.30,
        sigma=np.array([0, 1, 0]),
        current_pulse=lambda time: 0,
        gamma=1.0,
        alpha=0.1,
        theta0=0.05,
        phi0=0.0,
    ):
        """Simulate a single-moment current-driven SOT device."""
        H_K = np.asarray(H_K, dtype=float)
        H_ext = np.asarray(H_ext, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        def sot(m, time):
            pulse = current_pulse(time)
            damping_like = -gamma * H_DL * np.cross(
                m, np.cross(m, sigma)
            )
            field_like = -gamma * H_FL * np.cross(m, sigma)
            return pulse * (damping_like + field_like)

        torque = Torque(sot)
        H_K_norm = np.linalg.norm(H_K)
        if H_K_norm == 0.0:
            field = Field.constant(H_ext)
        else:
            field = (
                Field.constant(H_ext)
                + Field.uniaxial_anisotropy(H_K / H_K_norm, H_K_norm)
            )

        self.dynamic = SingleMomentLLG(
            field=field,
            torque=torque,
            gamma=gamma,
            alpha=alpha,
        )
        self.m = _initial_direction(theta0, phi0)
        self.time = 0.0

    def run(self, t_total=1.0, time_step=0.01):
        time_axis, trajectory = self.dynamic.integrate(
            initial_m=self.m,
            time_span=[self.time, self.time + t_total],
            time_step=time_step,
        )
        self.m = trajectory[-1].copy()
        self.time = time_axis[-1]
        return time_axis, trajectory


class IFTDevice:
    """
    Couple an angle-dependent electronic IFT map to a single-moment LLG.

    Time is measured in femtoseconds, fields in tesla, and the light envelope
    returns the electric-field amplitude in V/angstrom.  ``omega0`` is the
    photon energy ``hbar*omega`` in eV, matching the electronic calculation.
    """

    REQUIRED_MAP_KEYS = {
        "energy_harmonic_modes",
        "energy_harmonic_coefficients",
        "torque_harmonic_modes",
        "torque_vsh_coefficients",
        "harmonic_degree",
        "omega_axis_eV",
    }

    def __init__(
        self,
        data_directory,
        omega0: float,
        polarization="xy+",
        light_pulse=lambda time: 0.0,
        moment_magnitude_muB: float | None = None,
        external_field: Sequence[float] = (0.0, 0.0, 0.0),
        additional_field=None,
        additional_torque=None,
        frequency_mode: str = "linear",
        drive_projection_1: int = 0,
        drive_projection_2: int = 0,
        torque_projection: int = 0,
        fermi_index: int = 0,
        gamma: float = ELECTRON_GAMMA_RAD_PER_FS_T,
        alpha: float = 0.1,
        theta0: float = 0.05,
        phi0: float = 0.0,
    ):
        if not callable(light_pulse):
            raise TypeError("light_pulse must be callable.")
        self.light_pulse = light_pulse
        self.polarization = _polarization_vector(polarization)
        self.omega0 = float(omega0)
        self.frequency_mode = str(frequency_mode).lower()

        path = Path(data_directory)
        if path.is_dir():
            path = path / "AngleDependentDynamicsMap.npz"
        if not path.is_file():
            raise FileNotFoundError(f"IFT dynamics map not found: {path}")
        self.map_path = path

        with np.load(path) as data:
            missing = self.REQUIRED_MAP_KEYS.difference(data.files)
            if missing:
                raise RuntimeError(
                    f"{path} lacks required map arrays: {sorted(missing)}"
                )
            harmonic_degree = int(data["harmonic_degree"])
            energy_modes = data["energy_harmonic_modes"].copy()
            energy_coefficients = data[
                "energy_harmonic_coefficients"
            ].copy()
            torque_modes = data["torque_harmonic_modes"].copy()
            torque_coefficients = data["torque_vsh_coefficients"].copy()
            self.omega_axis_eV = data["omega_axis_eV"].copy()
            stored_moment = (
                float(data["moment_magnitude_muB"])
                if "moment_magnitude_muB" in data.files
                else np.nan
            )

        if moment_magnitude_muB is None:
            moment_magnitude_muB = stored_moment
        self.moment_magnitude_muB = float(moment_magnitude_muB)
        if (
            not np.isfinite(self.moment_magnitude_muB)
            or self.moment_magnitude_muB <= 0.0
        ):
            raise ValueError(
                "A positive moment_magnitude_muB is required either as an "
                "IFTDevice argument or in the saved map."
            )

        self.energy_expansion = SphericalHarmonicExpansion(
            harmonic_degree,
            energy_modes,
            energy_coefficients,
            energy_coefficients.shape[1:],
        )

        if torque_coefficients.ndim != 9:
            raise RuntimeError(
                "Expected torque coefficients with axes "
                "(family, mode, drive_proj_1, drive_proj_2, torque_proj, "
                "fermi, omega, drive_1, drive_2)."
            )
        selection = (
            drive_projection_1,
            drive_projection_2,
            torque_projection,
            fermi_index,
        )
        selection_shape = torque_coefficients.shape[2:6]
        for index, size, name in zip(
            selection,
            selection_shape,
            (
                "drive_projection_1",
                "drive_projection_2",
                "torque_projection",
                "fermi_index",
            ),
        ):
            if not 0 <= index < size:
                raise IndexError(f"{name}={index} is outside [0, {size}).")

        selected = torque_coefficients[:,:,drive_projection_1,drive_projection_2,torque_projection,fermi_index,:,:,:]
        selected, self.frequency_bracket = self._select_frequency(selected)
        self.tensor_expansion = VectorSphericalHarmonicExpansion(
            harmonic_degree,
            torque_modes,
            selected,
            (3, 3),
        )
        polarization_weight = self.polarization[:, None] * self.polarization.conj()[None, :]
        contracted_coefficients = np.einsum(
            "ab,fmab->fm",
            polarization_weight,
            selected,
        )
        self.ift_expansion = VectorSphericalHarmonicExpansion(
            harmonic_degree,
            torque_modes,
            contracted_coefficients,
            (),
        )

        anisotropy = Field(self._anisotropy_field)
        field = Field.constant(external_field) + anisotropy
        if additional_field is not None:
            field = field + _as_field(additional_field)

        torque = Torque(self._optical_torque)
        if additional_torque is not None:
            torque = torque + _as_torque(additional_torque)

        self.field = field
        self.torque = torque
        self.dynamic = SingleMomentLLG(
            field=field,
            torque=torque,
            gamma=gamma,
            alpha=alpha,
        )
        self.m = _initial_direction(theta0, phi0)
        self.time = 0.0

    def _select_frequency(self, coefficients):
        omega = self.omega_axis_eV
        if omega.ndim != 1 or len(omega) == 0 or np.any(np.diff(omega) <= 0):
            raise RuntimeError("omega_axis_eV must be nonempty and increasing.")
        if self.frequency_mode == "nearest":
            index = int(np.argmin(np.abs(omega - self.omega0)))
            return coefficients[:, :, index], (index, index, 0.0)
        if self.frequency_mode == "exact":
            matches = np.flatnonzero(
                np.isclose(omega, self.omega0, rtol=1.0e-10, atol=1.0e-12)
            )
            if len(matches) == 0:
                raise ValueError("omega0 is not present on the saved grid.")
            index = int(matches[0])
            return coefficients[:, :, index], (index, index, 0.0)
        if self.frequency_mode != "linear":
            raise ValueError("frequency_mode must be linear, nearest, or exact.")
        if self.omega0 < omega[0] or self.omega0 > omega[-1]:
            raise ValueError("omega0 lies outside the saved frequency range.")

        upper = int(np.searchsorted(omega, self.omega0, side="left"))
        if upper == 0:
            return coefficients[:, :, 0], (0, 0, 0.0)
        if upper == len(omega):
            index = len(omega) - 1
            return coefficients[:, :, index], (index, index, 0.0)
        if np.isclose(omega[upper], self.omega0):
            return coefficients[:, :, upper], (upper, upper, 0.0)
        lower = upper - 1
        weight = (self.omega0 - omega[lower]) / (
            omega[upper] - omega[lower]
        )
        interpolated = (
            (1.0 - weight) * coefficients[:, :, lower]
            + weight * coefficients[:, :, upper]
        )
        return interpolated, (lower, upper, float(weight))

    def anisotropy_field(self, m) -> np.ndarray:
        """Return the fitted anisotropy field in tesla."""
        direction = np.asarray(m, dtype=float)
        gradient = self.energy_expansion.surface_gradient(direction[None, :] )[0]
        return -np.real(gradient) / (self.moment_magnitude_muB * BOHR_MAGNETON_EV_PER_TESLA)

    def _anisotropy_field(self, m, time) -> np.ndarray:
        return self.anisotropy_field(m)

    def ift_tensor(self, m) -> np.ndarray:
        """Return fitted chi_ab,c in mu_B eV/(V/angstrom)^2."""
        direction = np.asarray(m, dtype=float)
        return self.tensor_expansion.evaluate(direction[None, :])[0]

    def ift_response(self, m) -> np.ndarray:
        """Return the polarization-contracted IFT response coefficient."""
        direction = np.asarray(m, dtype=float)
        response = self.ift_expansion.evaluate(direction[None, :])[0]
        return np.real(response)

    def optical_torque(self, m, time) -> np.ndarray:
        """Return the optical source torque on the unit moment in 1/fs."""
        amplitude = complex(self.light_pulse(time))
        return  self.ift_response(m) * abs(amplitude) ** 2 / (HBAR_EV_FS * self.moment_magnitude_muB)

    def _optical_torque(self, m, time) -> np.ndarray:
        return self.optical_torque(m, time)

    def run(self, t_total=1000.0, time_step=0.1):
        """Advance the device from its current state, with time in fs."""
        time_axis, trajectory = self.dynamic.integrate(
            initial_m=self.m,
            time_span=[self.time, self.time + t_total],
            time_step=time_step,
        )
        self.m = trajectory[-1].copy()
        self.time = time_axis[-1]
        return time_axis, trajectory

