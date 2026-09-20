"""Values returned by response solvers, independent of Grid entry points."""

from dataclasses import dataclass, field, fields
import numpy as np


@dataclass
class ConductivityResult:
    sigma: np.ndarray
    sigma_bare: np.ndarray | None = None
    sigma_correction: np.ndarray | None = None
    sigma_correction_channels: dict | None = None
    sigma_correction_by_mode: np.ndarray | None = None
    displacement_factors: np.ndarray | None = None
    sum_rule_residual: np.ndarray | None = None
    correction_sum_rule_residual: np.ndarray | None = None
    static_epc_info: dict | None = None
    sigma_samples: np.ndarray | None = None
    sigma_stderr_real: np.ndarray | None = None
    sigma_stderr_imag: np.ndarray | None = None
    phonon_initial_positions: np.ndarray | None = None
    phonon_initial_velocities: np.ndarray | None = None
    phonon_trajectory_cosine: np.ndarray | None = None
    phonon_trajectory_sine: np.ndarray | None = None
    phonon_energies: np.ndarray | None = None
    phonon_frequency: float | None = None
    phonon_harmonics: np.ndarray | None = None
    phonon_dynamics_info: dict | None = None
    conductivity_harmonics: np.ndarray | None = None
    solver_info: list = field(default_factory=list)
    integration_info: list = field(default_factory=list)
    time_traces: dict = field(default_factory=dict)

    @property
    def projected_sigma(self):
        return self.sigma[None, None]


@dataclass
class FirstOrderResponseResult:
    kernel: object
    projected_channels: dict
    response_channels: dict
    fermi_indices: tuple
    omega_indices: tuple = ()
    pole_residual: object = None


def publish_conductivity(calculator, result):
    """Expose a returned result on the convenient Grid entry point."""
    calculator.result = result
    for item in fields(result):
        value = getattr(result, item.name)
        if value is not None:
            setattr(calculator, item.name, value)
    calculator.projected_sigma = result.projected_sigma
