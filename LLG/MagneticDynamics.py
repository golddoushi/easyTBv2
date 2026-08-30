"""
Magnetization dynamics for a single, fixed-length magnetic moment.

The module uses the Gilbert form of the Landau-Lifshitz-Gilbert equation

    dm/dt = -gamma m x H_eff + alpha m x dm/dt + tau,

where ``m`` is a unit vector.  Fields and torques use the common callable
interface ``quantity(m, time) -> vector``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


VectorFunction = Callable[[np.ndarray, float], np.ndarray]
TimeStepFunction = Callable[[float, np.ndarray], float]


def _vector3(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _unit_vector(value, name: str) -> np.ndarray:
    vector = _vector3(value, name)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return vector / norm


class PulseTimeStepSchedule:
    """Fine integration steps near pulses and coarse steps between them."""

    def __init__(self, center_times: Sequence[float], active_half_width: float, fine_step: float, coarse_step: float):
        """
        Construct fine-step windows around Gaussian intensity pulses.
                
        Parameters
        ----------
        - center_times : Sequence[float]
            peak positions
        - active_half_width : float
            time radius of the fine sampling around peaks
        - fine_step : float
            time step used to sample the dynamics during the peaks, usually small
        - coarse_step : float
            time step used to sample the dynamics far from the peaks, usually large
        """
        centers = np.asarray(center_times, dtype=float)
        if centers.ndim != 1 or len(centers) == 0:
            raise ValueError(
                "center_times must be a nonempty one-dimensional sequence."
            )
        if not np.all(np.isfinite(centers)):
            raise ValueError("center_times must contain only finite values.")

        self.active_half_width = float(active_half_width)
        self.fine_step = float(fine_step)
        self.coarse_step = float(coarse_step)
        for value, name in (
            (self.active_half_width, "active_half_width"),
            (self.fine_step, "fine_step"),
            (self.coarse_step, "coarse_step"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.fine_step > self.coarse_step:
            raise ValueError("fine_step must not exceed coarse_step.")

        windows = sorted((center - self.active_half_width, center + self.active_half_width,)for center in centers)
        merged = []
        for start, stop in windows:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        self.windows = np.asarray(merged, dtype=float)

    @classmethod
    def gaussian(cls, center_times: Sequence[float], intensity_fwhm: float, 
                 fine_step: float, coarse_step: float, window_half_width_in_fwhm: float = 3.0,
    ) -> "PulseTimeStepSchedule":
        """
        Construct fine-step windows around Gaussian intensity pulses.
        
        Parameters
        ----------
        - center_times : Sequence[float]
            peak positions
        - intensity_fwhm : float
            FWHM of the peaks
        - fine_step : float
            time step used to sample the dynamics during the peaks, usually small, 1/10 of the fwhm
        - coarse_step : float
            time step used to sample the dynamics far from the peaks, usually large
        - window_half_width_in_fwhm : float
            time radius of the fine sampling around peaks, in unit of fwhm. default 3 is good.
        """
        intensity_fwhm = float(intensity_fwhm)
        window_factor = float(window_half_width_in_fwhm)
        if not np.isfinite(intensity_fwhm) or intensity_fwhm <= 0.0:
            raise ValueError("intensity_fwhm must be finite and positive.")
        if not np.isfinite(window_factor) or window_factor <= 0.0:
            raise ValueError("window_half_width_in_fwhm must be finite and positive.")
        return cls(center_times=center_times, active_half_width=window_factor * intensity_fwhm,
            fine_step=fine_step, coarse_step=coarse_step)

    def __call__(self, time: float, m: np.ndarray | None = None) -> float:
        """Return a step that cannot cross a pulse-window boundary."""
        time = float(time)
        if not np.isfinite(time):
            raise ValueError("time must be finite.")

        index = int(np.searchsorted(self.windows[:, 1], time, side="right"))
        if index == len(self.windows):
            return self.coarse_step

        start, stop = self.windows[index]
        if time < start:
            return min(self.coarse_step, start - time)
        return min(self.fine_step, stop - time)


class Field:
    """Composable effective magnetic field ``H_eff(m, time)``."""

    def __init__(self, function: VectorFunction | None = None):
        self._function = function if function is not None else self._zero
        if not callable(self._function):
            raise TypeError("Field function must be callable.")

    @staticmethod
    def _zero(m: np.ndarray, time: float) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def __call__(self, m: np.ndarray, time: float = 0.0) -> np.ndarray:
        return _vector3(self._function(m, time), "effective field")

    def __add__(self, other: "Field") -> "Field":
        if not isinstance(other, Field):
            return NotImplemented
        return Field(lambda m, time: self(m, time) + other(m, time))

    def __radd__(self, other) -> "Field":
        if isinstance(other, (int, float)) and other == 0:
            return self
        return NotImplemented

    @classmethod
    def constant(cls, field: Sequence[float]) -> "Field":
        """Return an orientation- and time-independent external field."""
        field_vector = _vector3(field, "constant field").copy()
        return cls(lambda m, time: field_vector.copy())

    external = constant

    @classmethod
    def uniaxial_anisotropy(
        cls,
        axis: Sequence[float],
        anisotropy_field: float,
    ) -> "Field":
        """
        Return ``H_k (m . u) u`` for a uniaxial anisotropy axis ``u``.

        A positive ``anisotropy_field`` makes ``axis`` an easy axis; a
        negative value makes it a hard axis.
        """
        unit_axis = _unit_vector(axis, "anisotropy axis")
        strength = float(anisotropy_field)
        if not np.isfinite(strength):
            raise ValueError("anisotropy_field must be finite.")

        return cls(
            lambda m, time: strength * np.dot(m, unit_axis) * unit_axis
        )


class Torque:
    """Composable non-field torque ``tau(m, time)``."""

    def __init__(self, function: VectorFunction | None = None):
        self._function = function if function is not None else self._zero
        if not callable(self._function):
            raise TypeError("Torque function must be callable.")

    @staticmethod
    def _zero(m: np.ndarray, time: float) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def __call__(self, m: np.ndarray, time: float = 0.0) -> np.ndarray:
        return _vector3(self._function(m, time), "torque")

    def __add__(self, other: "Torque") -> "Torque":
        if not isinstance(other, Torque):
            return NotImplemented
        return Torque(lambda m, time: self(m, time) + other(m, time))

    def __radd__(self, other) -> "Torque":
        if isinstance(other, (int, float)) and other == 0:
            return self
        return NotImplemented

    @classmethod
    def constant(cls, torque: Sequence[float]) -> "Torque":
        """Return an orientation- and time-independent torque."""
        torque_vector = _vector3(torque, "constant torque").copy()
        return cls(lambda m, time: torque_vector.copy())


class SingleMomentLLG:
    """Integrate the LLG equation for one fixed-length magnetic moment."""

    def __init__(
        self,
        field: Field | VectorFunction,
        torque: Torque | VectorFunction | None,
        gamma: float,
        alpha: float,
    ):
        if not callable(field):
            raise TypeError("field must be a Field or a callable.")
        if torque is not None and not callable(torque):
            raise TypeError("torque must be a Torque, a callable, or None.")

        self.field = field
        self.torque = Torque() if torque is None else torque
        self.gamma = float(gamma)
        self.alpha = float(alpha)

        if not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite.")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and nonnegative.")

        self.time: np.ndarray | None = None
        self.trajectory: np.ndarray | None = None

    def derivative_from_terms(self, m: Sequence[float], effective_field, torque) -> np.ndarray:
        """Evaluate the explicit Gilbert equation from already selected terms.

        This supports constant-drive phase portraits: the field is the one
        used by the trajectory, while a chosen drive amplitude replaces the
        time-dependent pulse envelope.
        """
        direction = _unit_vector(m, "magnetic moment")
        effective_field = _vector3(effective_field, "effective field")
        torque = _vector3(torque, "torque")
        torque -= direction * np.dot(direction, torque)

        force = -self.gamma * np.cross(direction, effective_field) + torque
        return (force + self.alpha * np.cross(direction, force)) / (1.0 + self.alpha**2)

    def derivative_from_terms_many(self, directions, effective_fields, torques) -> np.ndarray:
        """Vectorized form of :meth:`derivative_from_terms` for topology flows."""
        directions = np.asarray(directions, dtype=float)
        effective_fields = np.asarray(effective_fields, dtype=float)
        torques = np.asarray(torques, dtype=float)
        if directions.ndim < 2 or directions.shape[-1] != 3:
            raise ValueError("directions must have shape (..., 3).")
        if effective_fields.shape != directions.shape or torques.shape != directions.shape:
            raise ValueError("effective_fields and torques must match directions.")
        norms = np.linalg.norm(directions, axis=-1)
        if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
            raise ValueError("directions must be finite and nonzero.")
        direction = directions/norms[..., None]
        tangent_torque = torques-np.sum(direction*torques, axis=-1)[..., None]*direction
        force = -self.gamma*np.cross(direction, effective_fields)+tangent_torque
        return (force+self.alpha*np.cross(direction, force))/(1.0+self.alpha**2)

    def derivative(self, time: float, m: Sequence[float]) -> np.ndarray:
        """
        Evaluate the explicit Gilbert equation at ``(time, m)``.

        The supplied torque is projected perpendicular to ``m`` because this
        class describes orientational dynamics at fixed moment magnitude.
        """
        direction = _unit_vector(m, "magnetic moment")
        return self.derivative_from_terms(
            direction, self.field(direction, time), self.torque(direction, time)
        )

    def _rk4_step(self, time: float, m: np.ndarray, step: float) -> np.ndarray:
        k1 = self.derivative(time, m)
        k2 = self.derivative(time + 0.5 * step, m + 0.5 * step * k1)
        k3 = self.derivative(time + 0.5 * step, m + 0.5 * step * k2)
        k4 = self.derivative(time + step, m + step * k3)
        updated = m + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return _unit_vector(updated, "updated magnetic moment")

    def integrate(
        self,
        initial_m: Sequence[float],
        time_span: Sequence[float],
        time_step: float | TimeStepFunction,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate the trajectory with fourth-order Runge-Kutta.

        Parameters
        ----------
        initial_m
            Initial moment direction.  It is normalized before integration.
        time_span
            Two values ``(start, stop)`` in the chosen unit of time.
        time_step
            Positive maximum integration step, or a callable
            ``time_step(time, m)`` returning the next positive maximum step.
            The last step is shortened when necessary to end exactly at
            ``time_span[1]``.

        Returns
        -------
        time, trajectory
            ``time`` has shape ``(nstep + 1,)`` and ``trajectory`` has shape
            ``(nstep + 1, 3)``.  Every trajectory vector has unit norm.

        Notes
        -----
        Units must be mutually consistent.  For example, with time in seconds
        and field in tesla, ``gamma`` is in rad/(s T) and torque in 1/s.
        """
        interval = np.asarray(time_span, dtype=float)
        if interval.shape != (2,) or not np.all(np.isfinite(interval)):
            raise ValueError("time_span must contain two finite values.")
        start, stop = interval
        if stop <= start:
            raise ValueError("time_span stop must be greater than start.")

        if callable(time_step):
            step_function = time_step
        else:
            step = float(time_step)
            if not np.isfinite(step) or step <= 0.0:
                raise ValueError("time_step must be finite and positive.")
            step_function = lambda time, m: step

        time_values = [start]
        trajectory = [_unit_vector(initial_m, "initial_m")]

        while time_values[-1] < stop:
            current_time = time_values[-1]
            requested_step = float(step_function(current_time, trajectory[-1]))
            if not np.isfinite(requested_step) or requested_step <= 0.0:
                raise ValueError(
                    "time_step callable must return a finite positive value."
                )
            current_step = min(requested_step, stop - current_time)
            next_m = self._rk4_step(current_time, trajectory[-1], current_step)
            time_values.append(current_time + current_step)
            trajectory.append(next_m)

        self.time = np.asarray(time_values)
        self.trajectory = np.asarray(trajectory)
        return self.time, self.trajectory
