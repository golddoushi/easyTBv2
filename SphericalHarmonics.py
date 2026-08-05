"""Spherical-harmonic tools for orientation-dependent magnetic data."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
try:
    from scipy.special import sph_harm_y as _sph_harm_y

    def _spherical_harmonic(
        degree: int,
        order: int,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        return _sph_harm_y(degree, order, theta, phi)
except ImportError:
    from scipy.special import sph_harm as _sph_harm

    def _spherical_harmonic(
        degree: int,
        order: int,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        return _sph_harm(order, degree, phi, theta)

def _directions(value) -> np.ndarray:
    directions = np.asarray(value, dtype=float)
    if directions.ndim == 1:
        directions = directions[None, :]
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(
            "directions must have shape (norientation, 3)."
        )
    norms = np.linalg.norm(directions, axis=1)
    if np.any(~np.isfinite(directions)) or np.any(norms == 0.0):
        raise ValueError("directions must be finite and nonzero.")
    return directions / norms[:, None]

def fibonacci_sphere(number: int, antipodal: bool = False) -> np.ndarray:
    """
    Return approximately equal-area directions without duplicated poles.

    The deterministic Fibonacci lattice is intended for least-squares
    spherical-harmonic sampling rather than exact angular quadrature.
    """
    number = int(number)
    if number < 1: raise ValueError("number must be positive.")

    if antipodal:
        if number % 2 != 0:
            raise ValueError("An antipodal Fibonacci sphere needs an even number.")
        half_number = number // 2
        index = np.arange(half_number, dtype=float)
        z = 1.0 - (index + 0.5) / half_number
    else:
        index = np.arange(number, dtype=float)
        z = 1.0 - 2.0 * (index + 0.5) / number
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = np.mod(index * golden_angle, 2.0 * np.pi)
    directions = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    if antipodal: directions = np.concatenate((directions, -directions), axis=0)
    return directions

def cartesian_to_spherical(directions) -> tuple[np.ndarray, np.ndarray]:
    """Return polar ``theta`` and azimuthal ``phi`` for Cartesian directions."""
    unit = _directions(directions)
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * np.pi)
    return theta, phi

def scalar_modes(max_degree: int) -> np.ndarray:
    """Return ``(degree, order)`` rows through ``max_degree``."""
    max_degree = int(max_degree)
    if max_degree < 0:
        raise ValueError("max_degree must be nonnegative.")
    return np.asarray([(degree, order) for degree in range(max_degree + 1) for order in range(-degree, degree + 1)],dtype=int)

def vector_modes(max_degree: int) -> np.ndarray:
    """Return VSH ``(degree, order)`` rows; tangential modes begin at l=1."""
    max_degree = int(max_degree)
    if max_degree < 1:
        raise ValueError("max_degree must be at least one.")
    return np.asarray([(degree, order) for degree in range(1, max_degree + 1) for order in range(-degree, degree + 1)],dtype=int)

def scalar_harmonic_matrix(directions, modes: Sequence[Sequence[int]]) -> np.ndarray:
    """Evaluate complex scalar spherical harmonics at the supplied directions."""
    theta, phi = cartesian_to_spherical(directions)
    mode_array = np.asarray(modes, dtype=int)
    matrix = np.empty((len(theta), len(mode_array)), dtype=complex)
    for imode, (degree, order) in enumerate(mode_array):
        matrix[:, imode] = _spherical_harmonic(int(degree), int(order), theta, phi)
    return matrix

def vector_harmonic_matrices(directions, modes: Sequence[Sequence[int]], zerosAtPoles=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate normalized poloidal and toroidal vector spherical harmonics.

    The returned arrays both have shape ``(norientation, nmode, 3)``.
    The convention is

    ``poloidal = grad_s(Y_lm) / sqrt(l(l+1))`` and
    ``toroidal = r_hat x poloidal``.
    """
    unit = _directions(directions)
    theta, phi = cartesian_to_spherical(unit)
    sin_theta = np.sin(theta)
    if np.any(sin_theta < 1.0e-12):
        if zerosAtPoles:
            print(
                "Warning: Vector harmonics are evaluated away from the coordinate poles."
                f"for {zerosAtPoles=}, fall back to zero outputs! Take your own risk for this feature!"
            )
            return np.zeros((len(theta), len(modes), 3), dtype=complex), np.zeros((len(theta), len(modes), 3), dtype=complex)
        raise ValueError("Vector harmonics are evaluated away from the coordinate poles.")
    e_theta = np.column_stack((np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -sin_theta))
    e_phi = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)))

    mode_array = np.asarray(modes, dtype=int)
    harmonic_cache = {}
    for degree, order in mode_array:
        degree = int(degree)
        for needed_order in (int(order) - 1, int(order), int(order) + 1):
            if abs(needed_order) <= degree:
                key = (degree, needed_order)
                if key not in harmonic_cache:
                    harmonic_cache[key] = _spherical_harmonic(degree, needed_order, theta, phi)

    poloidal = np.empty((len(theta), len(mode_array), 3), dtype=complex)
    for imode, (degree_raw, order_raw) in enumerate(mode_array):
        degree = int(degree_raw)
        order = int(order_raw)
        harmonic = harmonic_cache[(degree, order)]
        derivative_theta = np.zeros(len(theta), dtype=complex)

        if order < degree:
            factor_plus = np.sqrt((degree - order) * (degree + order + 1))
            derivative_theta += 0.5 * factor_plus * np.exp(-1j * phi) * harmonic_cache[(degree, order + 1)]
        if order > -degree:
            factor_minus = np.sqrt((degree + order) * (degree - order + 1))
            derivative_theta -= 0.5 * factor_minus * np.exp(1j * phi) * harmonic_cache[(degree, order - 1)]

        derivative_phi = 1j * order * harmonic
        gradient = derivative_theta[:, None] * e_theta + (derivative_phi / sin_theta)[:, None] * e_phi
        poloidal[:, imode, :] = gradient / np.sqrt(degree * (degree + 1))

    toroidal = np.cross(unit[:, None, :], poloidal, axis=-1)
    return poloidal, toroidal

def project_tangential(values, directions) -> np.ndarray:
    """Remove the radial component from vectors on the orientation sphere."""
    unit = _directions(directions)
    vectors = np.asarray(values)
    if vectors.shape[0] != len(unit) or vectors.shape[-1] != 3:
        raise ValueError("values must have shape (norientation, ..., 3).")
    radial = np.einsum("ni,n...i->n...", unit, vectors)
    return vectors - radial[..., None] * unit[
        (slice(None),) + (None,) * (vectors.ndim - 2) + (slice(None),)
    ]


def _polar_rotation_matrices(rotations) -> np.ndarray:
    matrices = np.asarray(rotations, dtype=float)
    if matrices.ndim == 2:
        matrices = matrices[None, :, :]
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (noperation, 3, 3).")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("rotations must contain only finite values.")
    metric_error = np.max(np.abs(matrices @ matrices.swapaxes(-1, -2) - np.eye(3)[None, :, :]))
    determinant = np.linalg.det(matrices)
    if metric_error > 1.0e-8 or np.max(abs(abs(determinant) - 1.0)) > 1.0e-8:
        raise ValueError("rotations must be orthogonal matrices.")
    return matrices


def axial_rotation_matrices(polar_rotations) -> np.ndarray:
    """Return axial-vector matrices ``det(R) R`` from polar rotations."""
    polar = _polar_rotation_matrices(polar_rotations)
    determinant = np.linalg.det(polar)
    return determinant[:, None, None] * polar


def transform_ift_tensor(values, polar_rotation) -> np.ndarray:
    """
    Transform ``chi_ab,c`` with two polar inputs and one axial output.

    The final three axes of ``values`` must be
    ``(drive_cartesian_1, drive_cartesian_2, torque_cartesian)``.
    """
    tensor = np.asarray(values)
    if tensor.ndim < 3 or tensor.shape[-3:] != (3, 3, 3):
        raise ValueError("IFT values must end in three Cartesian axes.")
    polar = _polar_rotation_matrices(polar_rotation)
    if len(polar) != 1:
        raise ValueError("transform_ift_tensor expects one rotation.")
    polar = polar[0]
    axial = np.linalg.det(polar) * polar
    return np.einsum(
        "ai,bj,ck,...ijk->...abc",
        polar,
        polar,
        axial,
        tensor,
        optimize=True,
    )


def augment_scalar_axial_orbits(directions, values, polar_rotations):
    """Augment scalar orientation data by spatial-symmetry orbits."""
    unit = _directions(directions)
    sample_values = np.asarray(values)
    if sample_values.shape[0] != len(unit):
        raise ValueError("values and directions have different lengths.")
    axial = axial_rotation_matrices(polar_rotations)
    orbit_directions = np.einsum("gij,nj->gni", axial, unit)
    orbit_values = np.broadcast_to(
        sample_values[None, ...],
        (len(axial),) + sample_values.shape,
    )
    return orbit_directions.reshape((-1, 3)), orbit_values.reshape((-1,) + sample_values.shape[1:])


def augment_ift_axial_orbits(directions, values, polar_rotations):
    """
    Augment IFT samples using polar drive and axial moment/torque actions.

    ``values`` must have shape ``(norientation, ..., 3, 3, 3)``.
    """
    unit = _directions(directions)
    sample_values = np.asarray(values)
    assert sample_values.shape[0] == len(unit), "values and directions have different lengths."
    if sample_values.ndim < 4 or sample_values.shape[-3:] != (3, 3, 3):
        raise ValueError("IFT values must end in three Cartesian axes.")

    polar = _polar_rotation_matrices(polar_rotations)
    axial = axial_rotation_matrices(polar)
    orbit_directions = np.einsum("gij,nj->gni", axial, unit)
    orbit_values = np.einsum(
        "gai,gbj,gck,n...ijk->gn...abc",
        polar, polar, axial, sample_values, optimize=True)
    return orbit_directions.reshape((-1, 3)), orbit_values.reshape((-1,) + sample_values.shape[1:])


class SphericalHarmonicExpansion:
    """Least-squares scalar spherical-harmonic expansion."""

    def __init__(self, max_degree: int, modes, coefficients, value_shape):
        self.max_degree = int(max_degree)
        self.modes = np.asarray(modes, dtype=int)
        self.coefficients = np.asarray(coefficients)
        self.value_shape = tuple(value_shape)

    @classmethod
    def fit(cls, directions, values, max_degree: int) -> tuple["SphericalHarmonicExpansion", dict]:
        unit = _directions(directions)
        sample_values = np.asarray(values)
        if sample_values.shape[0] != len(unit): raise ValueError("values and directions have different lengths.")

        modes = scalar_modes(max_degree)
        matrix = scalar_harmonic_matrix(unit, modes)
        rhs = sample_values.reshape(len(unit), -1)
        coefficients, _, rank, singular_values = np.linalg.lstsq(matrix, rhs, rcond=None)
        expansion = cls(max_degree, modes,
            coefficients.reshape((len(modes),) + sample_values.shape[1:]),
            sample_values.shape[1:],
        )
        reconstructed = expansion.evaluate(unit)
        error = reconstructed - sample_values
        rms = np.sqrt(np.mean(np.abs(error) ** 2, axis=0))
        reference_rms = np.sqrt(np.mean(np.abs(sample_values) ** 2, axis=0))
        relative_rms = np.divide(rms, reference_rms, out=np.zeros_like(rms, dtype=float), where=reference_rms > 0.0)
        diagnostics = {
            "rank": rank,
            "condition": (
                singular_values[0] / singular_values[-1]
                if singular_values[-1] > 0.0
                else np.inf
            ),
            "rms": rms,
            "relative_rms": relative_rms,
        }
        return expansion, diagnostics

    @classmethod
    def fit_scalar_axial_symmetry(
        cls,
        directions,
        values,
        max_degree: int,
        polar_rotations,
    ) -> tuple["SphericalHarmonicExpansion", dict]:
        """Fit scalar orientation data after full spatial-orbit augmentation."""
        orbit_directions, orbit_values = augment_scalar_axial_orbits(
            directions, values, polar_rotations)
        return cls.fit(orbit_directions, orbit_values, max_degree)

    def evaluate(self, directions) -> np.ndarray:
        unit = _directions(directions)
        matrix = scalar_harmonic_matrix(unit, self.modes)
        flat_coefficients = self.coefficients.reshape(len(self.modes), -1)
        return (matrix @ flat_coefficients).reshape(
            (len(unit),) + self.value_shape
        )

    def surface_gradient(self, directions) -> np.ndarray:
        """Evaluate the tangential gradient with respect to moment direction."""
        unit = _directions(directions)
        nonconstant = self.modes[:, 0] > 0
        modes = self.modes[nonconstant]
        poloidal, _ = vector_harmonic_matrices(unit, modes)
        normalization = np.sqrt(modes[:, 0] * (modes[:, 0] + 1))
        gradient_basis = poloidal * normalization[None, :, None]
        matrix = gradient_basis.transpose(0, 2, 1).reshape(
            len(unit) * 3, len(modes)
        )
        flat_coefficients = self.coefficients[nonconstant].reshape(
            len(modes), -1
        )
        flat_gradient = matrix @ flat_coefficients
        gradient = flat_gradient.reshape((len(unit), 3) + self.value_shape)
        return np.moveaxis(gradient, 1, -1)

class VectorSphericalHarmonicExpansion:
    """Least-squares expansion of tangential vector fields on the sphere."""

    def __init__(self, max_degree: int, modes, coefficients, value_shape):
        self.max_degree = int(max_degree)
        self.modes = np.asarray(modes, dtype=int)
        self.coefficients = np.asarray(coefficients)
        self.value_shape = tuple(value_shape)

    @staticmethod
    def _design_matrix(directions, modes) -> np.ndarray:
        poloidal, toroidal = vector_harmonic_matrices(directions, modes)
        basis = np.concatenate((poloidal, toroidal), axis=1)
        return basis.transpose(0, 2, 1).reshape(len(basis) * 3, basis.shape[1])

    @classmethod
    def fit(cls, directions, values, max_degree: int, project: bool = True) -> tuple["VectorSphericalHarmonicExpansion", dict]:
        unit = _directions(directions)
        sample_values = np.asarray(values)
        if sample_values.shape[0] != len(unit):
            raise ValueError("values and directions have different lengths.")
        if sample_values.shape[-1] != 3:
            raise ValueError("The last values axis must be Cartesian.")
        if project:
            sample_values = project_tangential(sample_values, unit)

        modes = vector_modes(max_degree)
        matrix = cls._design_matrix(unit, modes)
        value_shape = sample_values.shape[1:-1]
        rhs = np.moveaxis(sample_values, -1, 1).reshape(len(unit) * 3, -1)
        coefficients, _, rank, singular_values = np.linalg.lstsq(matrix, rhs, rcond=None)
        expansion = cls(max_degree, modes, coefficients.reshape((2, len(modes)) + value_shape), value_shape)
        reconstructed = expansion.evaluate(unit)
        error = reconstructed - sample_values
        rms = np.sqrt(np.mean(np.abs(error) ** 2, axis=(0, -1)))
        reference_rms = np.sqrt(np.mean(np.abs(sample_values) ** 2, axis=(0, -1)))
        relative_rms = np.divide(rms, reference_rms, out=np.zeros_like(rms, dtype=float), where=reference_rms > 0.0)
        diagnostics = {
            "rank": rank,
            "condition": (
                singular_values[0] / singular_values[-1]
                if singular_values[-1] > 0.0
                else np.inf
            ),
            "rms": rms,
            "relative_rms": relative_rms,
        }
        return expansion, diagnostics

    @classmethod
    def fit_ift_symmetry(
        cls,
        directions,
        values,
        max_degree: int,
        polar_rotations,
        project: bool = True,
    ) -> tuple["VectorSphericalHarmonicExpansion", dict]:
        """Fit a tangent IFT tensor after full spatial-orbit augmentation."""
        orbit_directions, orbit_values = augment_ift_axial_orbits(
            directions, values, polar_rotations)
        return cls.fit(
            orbit_directions,
            orbit_values,
            max_degree,
            project=project,
        )

    def evaluate(self, directions) -> np.ndarray:
        unit = _directions(directions)
        matrix = self._design_matrix(unit, self.modes)
        flat_coefficients = self.coefficients.reshape(
            2 * len(self.modes), -1
        )
        flat_values = matrix @ flat_coefficients
        values = flat_values.reshape((len(unit), 3) + self.value_shape)
        return np.moveaxis(values, 1, -1)
