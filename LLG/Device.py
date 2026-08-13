from collections.abc import Callable, Sequence
from dataclasses import dataclass
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

@dataclass(frozen=True)
class TangentTorqueMap:
    """Drive-torque vector field sampled on the magnetic orientation sphere."""

    theta: np.ndarray
    phi: np.ndarray
    direction: np.ndarray
    cartesian: np.ndarray
    theta_component: np.ndarray
    phi_component: np.ndarray

    @property
    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.cartesian, axis=-1)

    @property
    def components(self) -> np.ndarray:
        """Return ``(T_theta, T_phi)`` on the last axis."""
        return np.stack((self.theta_component, self.phi_component), axis=-1)

def sample_tangent_torque_map(device, theta_count: int = 31, phi_count: int = 61, drive_amplitude: float = 1.0,) -> TangentTorqueMap:
    """
    Sample the drive-induced tangent torque over magnetic orientations.

    ``drive_amplitude=1`` means a unit current-pulse multiplier for
    :class:`SOTDevice` and a unit electric-field amplitude in V/angstrom for
    :class:`IFTDevice`.  Static fields, anisotropy, and their LLG precession
    are intentionally excluded.

    The theta samples are cell-centered in ``(0, pi)``.  The plot can still
    span ``0 <= theta <= pi``, but avoiding the exact poles removes the
    coordinate singularity of the local spherical basis and of the VSH fit.
    """
    theta_count = int(theta_count)
    phi_count = int(phi_count)
    if theta_count < 2:
        raise ValueError("theta_count must be at least 2.")
    if phi_count < 2:
        raise ValueError("phi_count must be at least 2.")
    if not np.isfinite(drive_amplitude):
        raise ValueError("drive_amplitude must be finite.")
    if not hasattr(device, "drive_torque"):
        raise TypeError("device must provide drive_torque(m, drive_amplitude).")

    theta_axis = (np.arange(theta_count, dtype=float) + 0.5) * np.pi / theta_count
    phi_axis = np.linspace(0.0, 2.0 * np.pi, phi_count, endpoint=False)
    theta, phi = np.meshgrid(theta_axis, phi_axis, indexing="ij")

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    direction = np.stack((
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta,
        ),
        axis=-1,
    )
    theta_basis = np.stack((
            cos_theta * cos_phi,
            cos_theta * sin_phi,
            -sin_theta,
        ),
        axis=-1,
    )
    phi_basis = np.stack((
            -sin_phi,
            cos_phi,
            np.zeros_like(phi),
        ),
        axis=-1,
    )

    flat_direction = direction.reshape(-1, 3)
    flat_torque = np.asarray(
        [
            device.drive_torque(m, drive_amplitude)
            for m in flat_direction
        ],
        dtype=float,
    )
    if flat_torque.shape != flat_direction.shape:
        raise ValueError("device.drive_torque must return a Cartesian vector.")
    cartesian = flat_torque.reshape(direction.shape)
    radial = np.sum(cartesian * direction, axis=-1)
    cartesian = cartesian - radial[..., None] * direction

    return TangentTorqueMap(
        theta=theta,
        phi=phi,
        direction=direction,
        cartesian=cartesian,
        theta_component=np.sum(cartesian * theta_basis, axis=-1),
        phi_component=np.sum(cartesian * phi_basis, axis=-1),
    )

def sample_tangent_torque_at_directions(device, directions, drive_amplitude: float = 1.0) -> TangentTorqueMap:
    """Evaluate the fitted drive torque at specified magnetic directions."""
    if not np.isfinite(drive_amplitude):
        raise ValueError("drive_amplitude must be finite.")
    if not hasattr(device, "drive_torque"):
        raise TypeError("device must provide drive_torque(m, drive_amplitude).")
    direction = np.asarray(directions, dtype=float)
    if direction.ndim < 2 or direction.shape[-1] != 3:
        raise ValueError("directions must have shape (..., 3).")
    direction_norm = np.linalg.norm(direction, axis=-1)
    if np.any(~np.isfinite(direction_norm)) or np.any(direction_norm == 0.0):
        raise ValueError("directions must be finite and nonzero.")
    direction = direction/direction_norm[..., None]

    flat_direction = direction.reshape(-1, 3)
    flat_torque = np.asarray([device.drive_torque(m, drive_amplitude) for m in flat_direction], dtype=float)
    if flat_torque.shape != flat_direction.shape:
        raise ValueError("device.drive_torque must return a Cartesian vector.")
    cartesian = flat_torque.reshape(direction.shape)
    cartesian -= np.sum(cartesian*direction, axis=-1)[..., None]*direction

    theta = np.arccos(np.clip(direction[..., 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(direction[..., 1], direction[..., 0]), 2.0*np.pi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    theta_basis = np.stack((cos_theta*cos_phi, cos_theta*sin_phi, -sin_theta), axis=-1)
    phi_basis = np.stack((-sin_phi, cos_phi, np.zeros_like(phi)), axis=-1)
    return TangentTorqueMap(
        theta=theta,
        phi=phi,
        direction=direction,
        cartesian=cartesian,
        theta_component=np.sum(cartesian*theta_basis, axis=-1),
        phi_component=np.sum(cartesian*phi_basis, axis=-1),
    )

def sample_raw_tangent_torque_map(device, drive_amplitude: float = 1.0) -> TangentTorqueMap:
    """
    Contract and tangent-project the raw IFT samples stored in a dynamics map.

    Unlike :func:`sample_tangent_torque_map`, this function never evaluates
    the VSH expansion. The returned vectors occupy the original electronic
    calculation directions and use the same unit-moment source convention as
    ``IFTDevice.drive_torque``.
    """
    if not np.isfinite(drive_amplitude):
        raise ValueError("drive_amplitude must be finite.")
    required_attributes = (
        "map_path", "polarization", "frequency_bracket", "moment_magnitude_muB",
        "drive_projection_1", "drive_projection_2", "torque_projection", "fermi_index",
    )
    missing_attributes = [name for name in required_attributes if not hasattr(device, name)]
    if missing_attributes:
        raise TypeError(f"device lacks raw-map selection attributes: {missing_attributes}")

    with np.load(device.map_path) as data:
        required_keys = {"directions", "theta", "phi", "projected_chi_IFT"}
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise RuntimeError(f"{device.map_path} lacks raw map arrays: {sorted(missing_keys)}")
        direction = np.asarray(data["directions"], dtype=float)
        theta = np.asarray(data["theta"], dtype=float)
        phi = np.asarray(data["phi"], dtype=float)
        raw_tensor = data["projected_chi_IFT"]

        if raw_tensor.ndim != 9:
            raise RuntimeError(
                "projected_chi_IFT must have axes (orientation, drive_projection_1, "
                "drive_projection_2, torque_projection, fermi, omega, drive_1, drive_2, torque)."
            )
        selection = (
            device.drive_projection_1, device.drive_projection_2,
            device.torque_projection, device.fermi_index,
        )
        for index, size, name in zip(
            selection, raw_tensor.shape[1:5],
            ("drive_projection_1", "drive_projection_2", "torque_projection", "fermi_index"),
        ):
            if not 0 <= index < size:
                raise IndexError(f"{name}={index} is outside [0, {size}).")
        selected = raw_tensor[:,selection[0],selection[1],selection[2],selection[3],:,:,:,:]

    if direction.ndim != 2 or direction.shape[1] != 3 or theta.shape != direction.shape[:1] or phi.shape != theta.shape:
        raise RuntimeError("Raw orientation arrays have inconsistent shapes.")
    lower, upper, weight = device.frequency_bracket
    if not 0 <= lower < selected.shape[1] or not 0 <= upper < selected.shape[1]:
        raise IndexError("The device frequency bracket is outside the raw omega axis.")
    if lower == upper:
        selected = selected[:, lower]
    else:
        selected = (1.0-weight)*selected[:, lower] + weight*selected[:, upper]

    polarization_weight = device.polarization[:, None] * device.polarization.conj()[None, :]
    contracted = np.einsum("ab,nabc->nc", polarization_weight, selected)
    cartesian = -np.real(contracted)*abs(complex(drive_amplitude))**2 / (HBAR_EV_FS*device.moment_magnitude_muB)
    direction_norm = np.linalg.norm(direction, axis=1)
    if np.any(~np.isfinite(direction_norm)) or np.any(direction_norm == 0.0):
        raise RuntimeError("Raw magnetic directions must be finite and nonzero.")
    direction = direction/direction_norm[:, None]
    cartesian = cartesian - np.sum(cartesian*direction, axis=1)[:, None]*direction

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    theta_basis = np.stack((cos_theta*cos_phi, cos_theta*sin_phi, -sin_theta), axis=-1)
    phi_basis = np.stack((-sin_phi, cos_phi, np.zeros_like(phi)), axis=-1)
    return TangentTorqueMap(
        theta=theta,
        phi=phi,
        direction=direction,
        cartesian=cartesian,
        theta_component=np.sum(cartesian*theta_basis, axis=-1),
        phi_component=np.sum(cartesian*phi_basis, axis=-1),
    )

class SOTDevice:
    def __init__(
        self,
        H_K=np.array([0, 0, 1]),
        H_ext=np.array([-0.3, 0.0, 0.0]),
        H_FL=0.10,
        H_DL=0.30,
        sigma=np.array([0, 1, 0]),
        current_pulse=lambda time: 0,
        gamma=ELECTRON_GAMMA_RAD_PER_FS_T,
        alpha=0.1,
        theta0=0.05,
        phi0=0.0,
    ):
        """Simulate a single-moment current-driven SOT device."""
        H_K = np.asarray(H_K, dtype=float)
        H_ext = np.asarray(H_ext, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.gamma = float(gamma)
        self.H_FL = float(H_FL)
        self.H_DL = float(H_DL)
        self.current_pulse = current_pulse

        def sot(m, time):
            return self.drive_torque(m, self.current_pulse(time))

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

    def drive_torque(self, m, drive_amplitude=1.0) -> np.ndarray:
        """Return SOT at a specified dimensionless current-pulse amplitude."""
        m = np.asarray(m, dtype=float)
        damping_like = -self.gamma * self.H_DL * np.cross(
            m,
            np.cross(m, self.sigma),
        )
        field_like = -self.gamma * self.H_FL * np.cross(m, self.sigma)
        return float(drive_amplitude) * (damping_like + field_like)

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

    ``use_DFT_mae`` controls the anisotropy field reconstructed from the
    electronic energy map.  The optional manual contribution uses
    ``E_uni(m) = extra_uniaxial_mae * (1 - (m dot u)^2)``, where ``u`` is
    ``extra_uniaxial_mae_direction``.  Thus a positive MAE makes ``u`` an
    easy axis and equals the hard-axis energy minus the easy-axis energy.
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
        use_DFT_mae: bool = True,
        extra_uniaxial_mae: float = 0.0,
        extra_uniaxial_mae_direction: Sequence[float] = (0.0, 0.0, 1.0),
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
                raise RuntimeError(f"{path} lacks required map arrays: {sorted(missing)}")
            harmonic_degree = int(data["harmonic_degree"])
            energy_modes = data["energy_harmonic_modes"].copy()
            energy_coefficients = data["energy_harmonic_coefficients"].copy()
            torque_modes = data["torque_harmonic_modes"].copy()
            torque_coefficients = data["torque_vsh_coefficients"].copy()
            self.omega_axis_eV = data["omega_axis_eV"].copy()
            stored_moment = float(data["moment_magnitude_muB"]) if "moment_magnitude_muB" in data.files else np.nan

        if moment_magnitude_muB is None:
            moment_magnitude_muB = stored_moment
        self.moment_magnitude_muB = float(moment_magnitude_muB)
        if not np.isfinite(self.moment_magnitude_muB) or self.moment_magnitude_muB <= 0.0:
            raise ValueError(
                "A positive moment_magnitude_muB is required either as an "
                "IFTDevice argument or in the saved map.")

        assert isinstance(use_DFT_mae, (bool, np.bool_)), "use_DFT_mae must be boolean."
        self.use_DFT_mae = bool(use_DFT_mae)
        self.extra_uniaxial_mae = float(extra_uniaxial_mae)
        assert np.isfinite(self.extra_uniaxial_mae), "extra_uniaxial_mae must be finite."
        extra_axis = np.asarray(extra_uniaxial_mae_direction, dtype=float)
        assert extra_axis.shape == (3,) and np.all(np.isfinite(extra_axis)), "extra_uniaxial_mae_direction must contain three finite values."
        extra_axis_norm = np.linalg.norm(extra_axis)
        assert extra_axis_norm != 0.0, "extra_uniaxial_mae_direction must be nonzero."
        self.extra_uniaxial_mae_direction = extra_axis / extra_axis_norm
        self.extra_uniaxial_field_tesla = 2.0 * self.extra_uniaxial_mae / (self.moment_magnitude_muB * BOHR_MAGNETON_EV_PER_TESLA)

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
        selection = (drive_projection_1, drive_projection_2, torque_projection, fermi_index)
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

        self.drive_projection_1 = drive_projection_1
        self.drive_projection_2 = drive_projection_2
        self.torque_projection = torque_projection
        self.fermi_index = fermi_index

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

    def dft_anisotropy_field(self, m) -> np.ndarray:
        """Return the spherical-harmonic DFT anisotropy field in tesla."""
        direction = np.asarray(m, dtype=float)
        gradient = self.energy_expansion.surface_gradient(direction[None, :])[0]
        return -np.real(gradient) / (self.moment_magnitude_muB * BOHR_MAGNETON_EV_PER_TESLA)

    def extra_uniaxial_energy(self, m) -> float:
        """Return the added uniaxial MAE in eV per cell, with zero at the easy axis."""
        direction = np.asarray(m, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("m must contain three finite values and be nonzero.")
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            raise ValueError("m must contain three finite values and be nonzero.")
        direction = direction / norm
        projection = np.dot(direction, self.extra_uniaxial_mae_direction)
        return self.extra_uniaxial_mae * (1.0 - projection**2)

    def extra_uniaxial_anisotropy_field(self, m) -> np.ndarray:
        """Return the field generated by ``extra_uniaxial_mae`` in tesla."""
        direction = np.asarray(m, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("m must contain three finite values and be nonzero.")
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            raise ValueError("m must contain three finite values and be nonzero.")
        direction = direction / norm
        projection = np.dot(direction, self.extra_uniaxial_mae_direction)
        return self.extra_uniaxial_field_tesla * projection * self.extra_uniaxial_mae_direction

    def anisotropy_field(self, m) -> np.ndarray:
        """Return the active DFT plus manually supplied anisotropy field in tesla."""
        field = self.extra_uniaxial_anisotropy_field(m)
        if self.use_DFT_mae:
            field = field + self.dft_anisotropy_field(m)
        return field

    def _anisotropy_field(self, m, time) -> np.ndarray:
        return self.anisotropy_field(m)

    def ift_tensor(self, m) -> np.ndarray:
        """Return fitted chi_ab,c in mu_B eV/(V/angstrom)^2."""
        direction = np.asarray(m, dtype=float)
        return self.tensor_expansion.evaluate(direction[None, :])[0]

    def ift_response(self, m) -> np.ndarray:
        """Return the polarization-contracted magnetic-order torque coefficient."""
        direction = np.asarray(m, dtype=float)
        response = self.ift_expansion.evaluate(direction[None, :])[0]
        return np.real(response)

    def optical_torque(self, m, time) -> np.ndarray:
        """Return the optical source torque on the unit moment in 1/fs."""
        return self.drive_torque(m, self.light_pulse(time))

    def drive_torque(self, m, drive_amplitude=1.0) -> np.ndarray:
        """
        Return the IFT contribution to d(m)/dt at a specified field amplitude.

        The electronic response stores g*mu_B times the angular-momentum
        torque transferred to the magnetic order. The LLG variable points
        along the magnetic moment, opposite to its electronic angular
        momentum, which supplies the reaction minus sign below.
        """
        amplitude = complex(drive_amplitude)
        return -self.ift_response(m) * abs(amplitude) ** 2 / (HBAR_EV_FS * self.moment_magnitude_muB)
    
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

