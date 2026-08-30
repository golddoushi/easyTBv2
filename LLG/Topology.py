from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from .Device import TangentTorqueMap, sample_tangent_torque_map


class _TorqueFieldDevice:
    """Expose a selected constant-drive device field through the legacy API."""

    def __init__(self, device, torque_mode, time):
        self.device = device
        self.torque_mode = str(torque_mode).lower()
        self.time = float(time)

    def drive_torque(self, direction, drive_amplitude):
        if self.torque_mode == "total":
            return self.device.total_torque(direction, drive_amplitude, self.time)
        if self.torque_mode == "drive":
            return self.device.drive_torque(direction, drive_amplitude)
        raise ValueError("torque_mode must be 'total' or 'drive'.")

    def drive_torque_many(self, directions, drive_amplitude):
        directions = np.asarray(directions, dtype=float)
        if self.torque_mode == "drive" and hasattr(self.device, "drive_torque_many"):
            return self.device.drive_torque_many(directions, drive_amplitude)
        return np.asarray([self.drive_torque(m, drive_amplitude) for m in directions], dtype=float)


@dataclass(frozen=True)
class TangentFixedPoint:
    """An isolated zero and its linearized tangent-plane dynamics."""

    direction: np.ndarray
    theta: float
    phi: float
    residual: float
    jacobian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    classification: str
    poincare_index: int


@dataclass(frozen=True)
class TangentSeparatrix:
    """One traced branch of a stable or unstable saddle manifold."""

    fixed_point_index: int
    manifold: str
    branch: int
    directions: np.ndarray
    endpoint_index: int

    @property
    def theta(self) -> np.ndarray:
        return np.arccos(np.clip(self.directions[:, 2], -1.0, 1.0))

    @property
    def phi(self) -> np.ndarray:
        return np.mod(np.arctan2(self.directions[:, 1], self.directions[:, 0]), 2.0*np.pi)


@dataclass(frozen=True)
class TangentTorqueTopology:
    """Numerical phase portrait of an autonomous tangent field on the sphere."""

    fixed_points: tuple[TangentFixedPoint, ...]
    separatrices: tuple[TangentSeparatrix, ...]
    basin_labels: np.ndarray
    field_scale: float
    poincare_index_sum: int
    basin_converged_fraction: float


def _unit_directions(directions) -> np.ndarray:
    vectors = np.asarray(directions, dtype=float)
    if vectors.ndim < 2 or vectors.shape[-1] != 3:
        raise ValueError("directions must have shape (..., 3).")
    norms = np.linalg.norm(vectors, axis=-1)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValueError("directions must be finite and nonzero.")
    return vectors/norms[..., None]


def _raw_drive_torque_many(device, flat_directions, drive_amplitude) -> np.ndarray:
    flat = np.asarray(flat_directions, dtype=float)
    if hasattr(device, "drive_torque_many"):
        torque = np.asarray(device.drive_torque_many(flat, drive_amplitude), dtype=float)
    else:
        torque = np.asarray([device.drive_torque(m, drive_amplitude) for m in flat], dtype=float)
    if torque.shape != flat.shape:
        raise ValueError("device drive torque must return Cartesian vectors.")
    return torque


def _drive_torque_many(device, directions, drive_amplitude) -> np.ndarray:
    unit = _unit_directions(directions)
    flat = unit.reshape(-1, 3)
    pole_mask = 1.0-np.abs(flat[:, 2]) < 1.0e-13
    torque = np.empty_like(flat)
    if np.any(~pole_mask):
        torque[~pole_mask] = _raw_drive_torque_many(device, flat[~pole_mask], drive_amplitude)
    if np.any(pole_mask):
        # Cartesian VSH fields are smooth at a pole even though their
        # (theta, phi) construction is singular there. Average a tiny ring
        # to evaluate that coordinate-independent limit without forcing zero.
        pole_epsilon = 1.0e-7
        azimuth = np.arange(8, dtype=float)*0.25*np.pi
        for index in np.flatnonzero(pole_mask):
            sign = np.sign(flat[index, 2])
            ring = np.column_stack((
                pole_epsilon*np.cos(azimuth),
                pole_epsilon*np.sin(azimuth),
                np.full(len(azimuth), sign*np.sqrt(1.0-pole_epsilon**2)),
            ))
            torque[index] = np.mean(_raw_drive_torque_many(device, ring, drive_amplitude), axis=0)
    torque -= np.sum(torque*flat, axis=1)[:, None]*flat
    return torque.reshape(unit.shape)


def _tangent_basis(direction) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(direction, dtype=float)
    reference = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.8 else np.array([1.0, 0.0, 0.0])
    first = reference-np.dot(reference, direction)*direction
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return first, second


def _exponential_map(direction, tangent) -> np.ndarray:
    length = np.linalg.norm(tangent)
    if length == 0.0:
        return np.asarray(direction, dtype=float).copy()
    return np.cos(length)*direction+np.sin(length)*tangent/length


def _candidate_directions(torque_map: TangentTorqueMap, maximum_candidates=64) -> np.ndarray:
    magnitude = np.asarray(torque_map.magnitude)
    flat_direction = np.asarray(torque_map.direction).reshape(-1, 3)
    candidates = []
    if magnitude.ndim == 2:
        local_minimum = np.ones_like(magnitude, dtype=bool)
        ntheta = magnitude.shape[0]
        for theta_shift in (-1, 0, 1):
            for phi_shift in (-1, 0, 1):
                if theta_shift == 0 and phi_shift == 0:
                    continue
                neighbor = np.roll(magnitude, phi_shift, axis=1)
                if theta_shift < 0:
                    neighbor = np.vstack((np.full((1, magnitude.shape[1]), np.inf), neighbor[:-1]))
                elif theta_shift > 0:
                    neighbor = np.vstack((neighbor[1:], np.full((1, magnitude.shape[1]), np.inf)))
                local_minimum &= magnitude <= neighbor
        candidates.extend(np.flatnonzero(local_minimum))
    smallest = np.argsort(magnitude.reshape(-1))[:maximum_candidates]
    candidates.extend(smallest.tolist())
    candidates = list(dict.fromkeys(candidates))
    candidates = np.asarray(sorted(candidates, key=lambda index: magnitude.reshape(-1)[index])[:maximum_candidates])
    return flat_direction[candidates]


def _linearize(device, direction, drive_amplitude, step) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second = _tangent_basis(direction)
    basis = np.stack((first, second))
    jacobian = np.empty((2, 2), dtype=float)
    for column, tangent in enumerate(basis):
        plus = _exponential_map(direction, step*tangent)
        minus = _exponential_map(direction, -step*tangent)
        values = _drive_torque_many(device, np.stack((plus, minus)), drive_amplitude)
        jacobian[:, column] = basis@((values[0]-values[1])/(2.0*step))
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    return jacobian, eigenvalues, eigenvectors


def _classify_fixed_point(jacobian, eigenvalues) -> tuple[str, int]:
    determinant = float(np.linalg.det(jacobian))
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0e-30)
    tolerance = 1.0e-6*spectral_scale
    real_parts = np.real(eigenvalues)
    if determinant < -tolerance**2:
        return "saddle", -1
    if np.all(real_parts < -tolerance):
        return ("stable focus" if np.any(np.abs(np.imag(eigenvalues)) > tolerance) else "stable node"), 1
    if np.all(real_parts > tolerance):
        return ("unstable focus" if np.any(np.abs(np.imag(eigenvalues)) > tolerance) else "unstable node"), 1
    return "nonhyperbolic", 0


def _refine_fixed_points(device, torque_map, drive_amplitude, zero_tolerance, merge_angle,
                         jacobian_step, maximum_candidates):
    field_scale = float(np.max(torque_map.magnitude))
    if not np.isfinite(field_scale) or field_scale == 0.0:
        raise RuntimeError("The tangent field vanishes everywhere; isolated fixed-point topology is undefined.")

    directions = []
    residuals = []
    for initial in _candidate_directions(torque_map, maximum_candidates=maximum_candidates):
        def residual(vector):
            norm = np.linalg.norm(vector)
            if norm == 0.0:
                return np.array([0.0, 0.0, 0.0, -1.0])
            direction = vector/norm
            torque = _drive_torque_many(device, direction[None, :], drive_amplitude)[0]
            return np.concatenate((torque/field_scale, [norm-1.0]))

        solution = least_squares(residual, initial, xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12, max_nfev=600)
        direction = solution.x/np.linalg.norm(solution.x)
        torque_residual = float(np.linalg.norm(_drive_torque_many(device, direction[None, :], drive_amplitude)[0]))
        if torque_residual > zero_tolerance*field_scale:
            continue
        if any(np.arccos(np.clip(np.dot(direction, previous), -1.0, 1.0)) < merge_angle for previous in directions):
            continue
        directions.append(direction)
        residuals.append(torque_residual)

    order = sorted(range(len(directions)), key=lambda index: (
        np.arccos(np.clip(directions[index][2], -1.0, 1.0)),
        np.mod(np.arctan2(directions[index][1], directions[index][0]), 2.0*np.pi),
    ))
    fixed_points = []
    for index in order:
        direction = directions[index]
        theta = float(np.arccos(np.clip(direction[2], -1.0, 1.0)))
        phi = 0.0 if abs(np.sin(theta)) < 1.0e-10 else float(
            np.mod(np.arctan2(direction[1], direction[0]), 2.0*np.pi)
        )
        jacobian, eigenvalues, eigenvectors = _linearize(
            device, direction, drive_amplitude, jacobian_step
        )
        classification, poincare_index = _classify_fixed_point(jacobian, eigenvalues)
        fixed_points.append(TangentFixedPoint(
            direction=direction,
            theta=theta,
            phi=phi,
            residual=residuals[index],
            jacobian=jacobian,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            classification=classification,
            poincare_index=poincare_index,
        ))
    return tuple(fixed_points), field_scale


def _advance_normalized_flow(device, directions, drive_amplitude, step, flow_sign=1.0) -> np.ndarray:
    direction = _unit_directions(directions)
    first = flow_sign*_drive_torque_many(device, direction, drive_amplitude)
    first_norm = np.linalg.norm(first, axis=-1)
    first_unit = np.divide(first, first_norm[..., None], out=np.zeros_like(first), where=first_norm[..., None] > 0.0)
    midpoint = _unit_directions(direction+0.5*step*first_unit)
    second = flow_sign*_drive_torque_many(device, midpoint, drive_amplitude)
    second_norm = np.linalg.norm(second, axis=-1)
    second_unit = np.divide(second, second_norm[..., None], out=np.zeros_like(second), where=second_norm[..., None] > 0.0)
    return _unit_directions(direction+step*second_unit)


def _nearest_fixed_point(direction, fixed_points, radius) -> int:
    if not fixed_points:
        return -1
    dots = np.array([np.dot(direction, fixed.direction) for fixed in fixed_points])
    index = int(np.argmax(dots))
    return index if dots[index] >= np.cos(radius) else -1


def _trace_separatrices(device, fixed_points, drive_amplitude, field_scale, zero_tolerance,
                        offset, step, maximum_steps) -> tuple[TangentSeparatrix, ...]:
    paths = []
    for fixed_index, fixed in enumerate(fixed_points):
        if fixed.classification != "saddle":
            continue
        first, second = _tangent_basis(fixed.direction)
        basis = np.stack((first, second))
        for manifold, select, flow_sign in (("stable", np.argmin, -1.0), ("unstable", np.argmax, 1.0)):
            eigen_index = int(select(np.real(fixed.eigenvalues)))
            local_vector = np.real(fixed.eigenvectors[:, eigen_index])
            local_vector /= np.linalg.norm(local_vector)
            cartesian_vector = local_vector[0]*basis[0]+local_vector[1]*basis[1]
            for branch in (-1, 1):
                direction = _exponential_map(fixed.direction, branch*offset*cartesian_vector)
                trace = [fixed.direction.copy(), direction.copy()]
                endpoint = -1
                for iteration in range(maximum_steps):
                    direction = _advance_normalized_flow(
                        device, direction[None, :], drive_amplitude, step, flow_sign=flow_sign
                    )[0]
                    trace.append(direction.copy())
                    if iteration > 5:
                        endpoint = _nearest_fixed_point(direction, fixed_points, 2.0*offset)
                        if endpoint >= 0 and endpoint != fixed_index:
                            break
                    residual = np.linalg.norm(_drive_torque_many(device, direction[None, :], drive_amplitude)[0])
                    if residual <= zero_tolerance*field_scale:
                        endpoint = _nearest_fixed_point(direction, fixed_points, 4.0*offset)
                        break
                paths.append(TangentSeparatrix(
                    fixed_point_index=fixed_index,
                    manifold=manifold,
                    branch=branch,
                    directions=np.asarray(trace),
                    endpoint_index=endpoint,
                ))
    return tuple(paths)


def _estimate_basins(device, directions, fixed_points, drive_amplitude, step, maximum_steps, radius):
    original_shape = np.asarray(directions).shape[:-1]
    points = _unit_directions(directions).reshape(-1, 3)
    labels = np.full(len(points), -1, dtype=int)
    attractors = [index for index, fixed in enumerate(fixed_points) if fixed.classification.startswith("stable")]
    if not attractors:
        return labels.reshape(original_shape), 0.0

    cosine_radius = np.cos(radius)
    for _ in range(maximum_steps):
        active = np.flatnonzero(labels < 0)
        if len(active) == 0:
            break
        dots = points[active]@np.stack([fixed_points[index].direction for index in attractors]).T
        nearest = np.argmax(dots, axis=1)
        converged = dots[np.arange(len(active)), nearest] >= cosine_radius
        labels[active[converged]] = np.asarray(attractors)[nearest[converged]]
        moving = active[~converged]
        if len(moving) == 0:
            break
        points[moving] = _advance_normalized_flow(device, points[moving], drive_amplitude, step)
    return labels.reshape(original_shape), float(np.mean(labels >= 0))


def analyze_tangent_torque_topology(
    device,
    torque_map: TangentTorqueMap | None = None,
    drive_amplitude: float = 1.0,
    compute_basins: bool = True,
    zero_tolerance: float = 1.0e-6,
    merge_angle: float = 2.0e-3,
    jacobian_step: float = 1.0e-5,
    separatrix_offset: float = 1.0e-2,
    flow_step: float = 2.0e-2,
    maximum_flow_steps: int = 800,
    basin_radius: float = 4.0e-2,
    maximum_candidates: int = 256,
    torque_mode: str = "total",
    time: float = 0.0,
) -> TangentTorqueTopology:
    """
    Analyze a constant-drive autonomous phase portrait.

    The default ``torque_mode='total'`` includes the complete explicit LLG
    derivative at the selected time, but freezes the pulse at
    ``drive_amplitude``. ``torque_mode='drive'`` analyzes only the source.
    Basins are numerical estimates; unassigned points can indicate a long
    transient, a non-fixed attractor, or insufficient integration length.
    """
    maximum_candidates = int(maximum_candidates)
    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive.")
    field_device = _TorqueFieldDevice(device, torque_mode, time)
    if torque_map is None:
        torque_map = sample_tangent_torque_map(
            device, drive_amplitude=drive_amplitude, torque_mode=torque_mode, time=time
        )
    fixed_points, field_scale = _refine_fixed_points(
        field_device, torque_map, drive_amplitude, zero_tolerance, merge_angle,
        jacobian_step, maximum_candidates,
    )
    separatrices = _trace_separatrices(
        field_device, fixed_points, drive_amplitude, field_scale, zero_tolerance,
        separatrix_offset, flow_step, maximum_flow_steps,
    )
    if compute_basins:
        basin_labels, converged_fraction = _estimate_basins(
            field_device, torque_map.direction, fixed_points, drive_amplitude,
            2.0*flow_step, maximum_flow_steps, basin_radius,
        )
    else:
        basin_labels = np.full(torque_map.theta.shape, -1, dtype=int)
        converged_fraction = 0.0
    return TangentTorqueTopology(
        fixed_points=fixed_points,
        separatrices=separatrices,
        basin_labels=basin_labels,
        field_scale=field_scale,
        poincare_index_sum=sum(fixed.poincare_index for fixed in fixed_points),
        basin_converged_fraction=converged_fraction,
    )
