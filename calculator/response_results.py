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


@dataclass
class SecondOrderResponseResult:
    """BZ responses and selected k integrands from a nonlinear contraction."""
    value: np.ndarray | None = None
    channels: dict = field(default_factory=dict)
    projected_value: np.ndarray | None = None
    projected_channels: dict = field(default_factory=dict)
    integrands: np.ndarray | None = None
    integrand_channels: dict = field(default_factory=dict)
    kernel: object = None
    channel_names: tuple = ()
    channel_descriptions: dict = field(default_factory=dict)
    fermi_indices: tuple = ()
    omega_indices: tuple = ()


def publish_second_order(calculator, result, observable):
    """Publish named observables only at the Grid entry-point boundary."""
    calculator.result = result
    calculator.density_matrix_kernel = result.kernel
    calculator.channel_names = result.channel_names
    calculator.channel_descriptions = result.channel_descriptions
    calculator.responseFermiIndexList = result.fermi_indices
    calculator.responseOmegaIndexList = result.omega_indices
    calculator.projected_response = result.integrands
    calculator.projected_response_channels = result.integrand_channels
    calculator.projected_sigma = result.projected_value
    calculator.projected_sigma_channels = result.projected_channels
    setattr(calculator, observable, result.value)
    setattr(calculator, observable + '_channels', result.channels)
    if observable.startswith('sigma'):
        calculator.sigma_channels = result.channels
    if observable == 'chi_IFT':
        calculator.projected_chi_IFT = result.projected_value
        calculator.projected_chi_IFT_channels = result.projected_channels
    for name in result.channel_names:
        setattr(calculator, observable + '_' + name, result.channels[name])
        setattr(calculator, 'projected_response_' + name, result.integrand_channels[name])
        setattr(calculator, 'projected_sigma_' + name, result.projected_channels[name])
        if observable == 'chi_IFT':
            setattr(calculator, 'projected_chi_IFT_' + name, result.projected_channels[name])
