#
# The module is for calculations on energy, berry curv., and several types of conductances
#
from phonon import PhononBath
from system import System
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from auxiliary import showElapsedTime, print_matrix, projectionOperator
from auxiliary import PauliBlock, semiCirle, fermiDiracOccupations
from scipy.special import p_roots
from scipy.constants import elementary_charge, electron_mass, hbar, angstrom
from scipy.constants import k as kB
from .density_matrix import (AC_CHANNEL_DESCRIPTIONS as AC_DM_CHANNEL_DESCRIPTIONS,
                             AC_CHANNEL_NAMES as AC_DM_CHANNEL_NAMES,
                             DC_CHANNEL_DESCRIPTIONS as DC_DM_CHANNEL_DESCRIPTIONS,
                             DC_CHANNEL_NAMES as DC_DM_CHANNEL_NAMES,
                             LengthGaugeDCFirstOrderDMKernel,
                             LengthGaugeFirstOrderDMKernel,
                             VelocityGaugeFirstOrderDMKernel,
                             diagonal_matrix, fermi_divided_difference,
                             LengthGaugeSecondOrderDMKernel,
                             VelocityGaugeSecondOrderDMKernel)

from .collision import (CollisionSpecification, PreparedCollision, resolve_collision, collision_mode,
                        prepare_phonon_background)
from .electronic_stages import (Calculator, Energy, Velocity, Dipole, Spin, Orbital,
                                SpinTorque, SpinExchangeTorque, SpinSOCTorque, SpinFluxTorque, OrbitalSOCTorque, OrbitalFluxTorque, OrbitalTorque, SpinVelocity, OrbitalVelocity,
                                Orbital_BlochSelfRotate, OrbitalVelocity_BlochSelfRotate, OrbitalVelocity_BlochSelfRotateV2)
from .response_inputs import ResponseInputs
from .response_results import publish_conductivity, publish_second_order, FirstOrderResponseResult
from .response_contractions import (
    _normalize_response_index_list, _contract_second_order_observable, _sum_density_matrix_channels, solve_bpve_lg, solve_bpve_vg, solve_ife_lg, solve_ife_vg, solve_torque_lg, solve_torque_vg)
from .response_stages import ElectronicResponse, DensityMatrixResponse, PhononResponse, response_requires_full_basis


def _contract_dc_first_order_density_matrix(inputs, drive_vertices, observable_vertices,
                                            observable_component_shape, responseFermiIndexList,
                                            finiteTemperatureOccupations, fullCovariantDerivative, *, collision):
    """Evaluate Tr(rho O) through exact band-resolved response weights.

    The expensive band-pair algebra is independent of Fermi level. Reducing it
    to one weight per band before the Fermi scan preserves the density-matrix
    result exactly while avoiding repeated ``(..., band, band)`` allocations.
    """
    kernel = LengthGaugeDCFirstOrderDMKernel(inputs, fullCovariantDerivative, finiteTemperatureOccupations,
                                           collision=collision)
    responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(inputs.fermiList))
    fermi_indices = tuple(responseFermiIndexList)
    responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}

    projected_shape = ((len(drive_vertices), len(observable_vertices), len(inputs.fermiList), 3)
                       + tuple(observable_component_shape))
    response_shape = ((len(drive_vertices), len(observable_vertices)) + inputs.kgrid
                      + (len(responseFermiIndexList), 3) + tuple(observable_component_shape))
    projected_channels = {name: np.zeros(projected_shape, dtype=float) for name in kernel.channel_names}
    response_channels = {name: np.zeros(response_shape, dtype=float) for name in kernel.channel_names}

    # A mesh-global EPC collision couples populations at different k points,
    # so the band-resolved Fermi-weight reduction below is no longer valid.
    # Contract each Fermi-level density matrix directly instead.
    if kernel.collision.mesh_coupled:
        k_slices = (slice(None),) * len(inputs.kgrid)
        k_axes = tuple(range(len(inputs.kgrid)))
        for ifermi, fermi in enumerate(inputs.fermiList):
            state = kernel.fermi_state(ifermi, fermi)
            irecord = responseFermiRecord.get(ifermi)
            for idrive, drive_vertex in enumerate(drive_vertices):
                for ialpha in range(3):
                    block = state.block(ialpha, drive_vertex)
                    for iobservable, observable_vertex in enumerate(observable_vertices):
                        for observable_components in np.ndindex(*observable_component_shape):
                            vertex_index = k_slices + observable_components + (slice(None), slice(None))
                            observable_nm = observable_vertex[vertex_index].swapaxes(-1, -2)
                            for name, rho in block.rho1.items():
                                value = np.sum(rho*observable_nm, axis=(-2, -1)).real
                                projected_channels[name][(idrive, iobservable, ifermi, ialpha)
                                                         + observable_components] = np.sum(value, axis=k_axes)
                                if irecord is not None:
                                    response_index = ((idrive, iobservable) + k_slices
                                                      + (irecord, ialpha) + observable_components)
                                    response_channels[name][response_index] = value
        return FirstOrderResponseResult(kernel, projected_channels, response_channels, fermi_indices)

    nbands = inputs.energies.shape[-1]
    weight_shape = ((len(drive_vertices), len(observable_vertices)) + inputs.kgrid
                    + (3,) + tuple(observable_component_shape) + (nbands,))
    weights_e = np.zeros(weight_shape, dtype=float)
    weights_i = np.zeros(weight_shape, dtype=float) if kernel.full_covariant_derivative else None
    k_slices = (slice(None),) * len(inputs.kgrid)

    for idrive, drive_vertex in enumerate(drive_vertices):
        for ialpha in range(3):
            drive_a = drive_vertex[..., ialpha, :, :]
            coherence_prefactor = (
                1j * kernel.deltaE_inverse_pure * kernel.g0_mn
                * drive_a * kernel.off_diag
            )
            drive_diag = np.diagonal(drive_a, axis1=-2, axis2=-1).real
            for iobservable, observable_vertex in enumerate(observable_vertices):
                for observable_components in np.ndindex(*observable_component_shape):
                    vertex_index = k_slices + observable_components + (slice(None), slice(None))
                    observable_nm = observable_vertex[vertex_index].swapaxes(-1, -2)
                    pair_weight = coherence_prefactor * observable_nm
                    band_weight_e = (np.sum(pair_weight, axis=-2) - np.sum(pair_weight, axis=-1)).real
                    weight_index = ((idrive, iobservable) + k_slices + (ialpha,)
                                    + observable_components + (slice(None),))
                    weights_e[weight_index] = band_weight_e
                    if weights_i is not None:
                        observable_diag = np.diagonal(observable_vertex[vertex_index], axis1=-2, axis2=-1)
                        weights_i[weight_index] = (drive_diag * observable_diag * kernel.collision.population_inverse).real

    fermi_shape = ((1, 1) + inputs.kgrid + (1,)*(1 + len(observable_component_shape)) + (nbands,))
    k_axes = tuple(range(2, 2 + len(inputs.kgrid)))
    for ifermi, fermi in enumerate(inputs.fermiList):
        f, fp = kernel.occupations(ifermi, fermi)
        response_e = np.sum(weights_e * f.reshape(fermi_shape), axis=-1)
        projected_channels["E"][:, :, ifermi] = np.sum(response_e, axis=k_axes)
        irecord = responseFermiRecord.get(ifermi)
        if irecord is not None:
            response_index = ((slice(None), slice(None)) + k_slices + (irecord,)
                              + (slice(None),)*(1 + len(observable_component_shape)))
            response_channels["E"][response_index] = response_e

        if weights_i is not None:
            response_i = np.sum(weights_i * fp.reshape(fermi_shape), axis=-1)
            projected_channels["I"][:, :, ifermi] = np.sum(response_i, axis=k_axes)
            if irecord is not None:
                response_channels["I"][response_index] = response_i
    return FirstOrderResponseResult(kernel, projected_channels, response_channels, fermi_indices)

def _first_order_observable_shape(inputs, observable_vertices):
    """Validate first-order vertices and return the observable component axes."""
    if not observable_vertices:
        raise ValueError("At least one observable vertex is required")
    nk_dimensions = len(inputs.kgrid)
    nbands = inputs.energies.shape[-1]
    drive_shape = tuple(inputs.kgrid) + (3, nbands, nbands)
    for vertex in observable_vertices:
        if (tuple(vertex.shape[:nk_dimensions]) != tuple(inputs.kgrid)
                or vertex.shape[-2:] != (nbands, nbands)):
            raise ValueError("Observable vertices must have shape kgrid + components + (nbands, nbands)")
    component_shape = tuple(observable_vertices[0].shape[nk_dimensions:-2])
    if any(tuple(vertex.shape[nk_dimensions:-2]) != component_shape for vertex in observable_vertices):
        raise ValueError("All observable vertices must have the same component shape")
    return drive_shape, component_shape

def _contract_first_order_observable(rho, observable, component_shape):
    """Evaluate Tr(rho O) for an observable with arbitrary component axes."""
    expanded = rho.reshape(rho.shape[:-2] + (1,)*len(component_shape) + rho.shape[-2:])
    return np.sum(expanded*observable.swapaxes(-1, -2), axis=(-2, -1))

def _contract_first_order_density_matrix_response(inputs, kernel_class, drive_vertices,
                                                   observable_vertices, frequencies,
                                                   responseFermiIndexList,
                                                   responseOmegaIndexList,
                                                   finiteTemperatureOccupations,
                                                   fullCovariantDerivative, *, collision):
    """Gauge-independent contraction engine for a first-order DM kernel.

    Returned projected arrays contain raw BZ sums with axes
    ``drive_projection, observable_projection, fermi, frequency,
    drive_direction, observable_components``. Response arrays retain the
    selected k-resolved integrands. Units and observable-specific signs belong
    to the consuming calculator.
    """
    kernel = kernel_class(inputs, fullCovariantDerivative, finiteTemperatureOccupations,
                          collision=collision)
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies must be a finite one-dimensional array")
    if not drive_vertices:
        raise ValueError("At least one drive vertex is required")
    drive_shape, component_shape = _first_order_observable_shape(inputs, observable_vertices)
    for vertex in drive_vertices:
        if vertex.shape != drive_shape:
            raise ValueError(f"Drive vertices must have shape {drive_shape}")

    fermi_indices = tuple(
        _normalize_response_index_list(responseFermiIndexList, len(inputs.fermiList)))
    omega_indices = tuple(
        _normalize_response_index_list(responseOmegaIndexList, len(frequencies)))
    fermi_records = {value:index for index, value in enumerate(fermi_indices)}
    omega_records = {value:index for index, value in enumerate(omega_indices)}
    projected_shape = ((len(drive_vertices), len(observable_vertices), len(inputs.fermiList),
                        len(frequencies), 3) + component_shape)
    response_shape = ((len(drive_vertices), len(observable_vertices)) + tuple(inputs.kgrid)
                      + (len(fermi_records), len(omega_records), 3) + component_shape)
    projected_channels = {
        name:np.zeros(projected_shape, dtype=complex) for name in kernel.channel_names}
    response_channels = {
        name:np.zeros(response_shape, dtype=complex) for name in kernel.channel_names}
    k_slices = (slice(None),)*len(inputs.kgrid)
    k_axes = tuple(range(len(inputs.kgrid)))
    response_tail = (slice(None),)*(1 + len(component_shape))

    if kernel.collision.mesh_coupled:
        # Mesh-global collisions cannot be reduced to independent band-pair
        # weights. Preserve the exact density-matrix solve for this path.
        for ifermi, fermi in enumerate(inputs.fermiList):
            state = kernel.fermi_state(ifermi, fermi)
            irecord_fermi = fermi_records.get(ifermi)
            for iomega, omega in enumerate(frequencies):
                frequency = state.at_frequency(omega)
                irecord_omega = omega_records.get(iomega)
                for idrive, drive_vertex in enumerate(drive_vertices):
                    for direction in range(3):
                        block = frequency.block(direction, drive_vertex)
                        for iobservable, observable_vertex in enumerate(observable_vertices):
                            for name, rho in block.rho1.items():
                                value = _contract_first_order_observable(
                                    rho, observable_vertex, component_shape)
                                projected_channels[name][idrive, iobservable, ifermi,
                                                         iomega, direction] = np.sum(value, axis=k_axes)
                                if irecord_fermi is not None and irecord_omega is not None:
                                    index = ((idrive, iobservable) + k_slices
                                             + (irecord_fermi, irecord_omega, direction)
                                             + (slice(None),)*len(component_shape))
                                    response_channels[name][index] = value
        return FirstOrderResponseResult(kernel, projected_channels, response_channels, fermi_indices, omega_indices)

    # For local CRTA/SERTA collisions, reduce regular interband pairs to one
    # band weight per frequency. Near-degenerate pairs retain the confluent
    # Fermi divided difference and are accumulated separately.
    nbands = inputs.energies.shape[-1]
    off_diagonal = ~kernel.diag
    inverse_gap = np.zeros_like(inputs.deltaE)
    if kernel.full_covariant_derivative:
        near_tolerance = 1e-5*inputs.kBT
        regular = off_diagonal & (np.abs(inputs.deltaE) > near_tolerance)
        np.divide(1., inputs.deltaE, out=inverse_gap, where=regular)
        near = off_diagonal & ~regular
    else:
        inverse_gap[...] = inputs.deltaEInverse_pure
        regular = inverse_gap != 0.
        near = np.zeros_like(regular)

    nk_dimensions = len(inputs.kgrid)
    near_indices = np.nonzero(near)
    if near_indices[0].size:
        near_k_flat = np.ravel_multi_index(near_indices[:nk_dimensions], inputs.kgrid)
        near_left_energy = inputs.energies[
            near_indices[:nk_dimensions] + (near_indices[nk_dimensions],)]
        near_right_energy = inputs.energies[
            near_indices[:nk_dimensions] + (near_indices[nk_dimensions + 1],)]
        near_band_left = near_indices[nk_dimensions]
        near_band_right = near_indices[nk_dimensions + 1]
    else:
        near_k_flat = near_left_energy = near_right_energy = None
        near_band_left = near_band_right = None

    fermi_data = []
    for fermi in inputs.fermiList:
        occupations = (inputs.fdOccupation(fermi) if kernel.full_covariant_derivative
                       or finiteTemperatureOccupations else (inputs.energies < fermi).astype(float))
        derivative = (inputs.fdEnergyDerivatives(occupations)[0]
                      if kernel.full_covariant_derivative else None)
        near_difference = (fermi_divided_difference(
            near_left_energy, near_right_energy, fermi, inputs.kBT)
            if near_k_flat is not None else None)
        fermi_data.append((occupations, derivative, near_difference))

    nk = int(np.prod(inputs.kgrid))
    component_count = int(np.prod(component_shape)) if component_shape else 1
    for iomega, omega in enumerate(frequencies):
        resolvent = kernel.collision.resolvent(inputs.deltaE, omega)
        pair_factor = -1j*resolvent*inverse_gap
        population_factor = np.diagonal(-1j*resolvent, axis1=-2, axis2=-1)
        near_resolvent = -1j*resolvent[near_indices] if near_k_flat is not None else None
        irecord_omega = omega_records.get(iomega)
        for idrive, drive_vertex in enumerate(drive_vertices):
            drive_diagonal = np.diagonal(drive_vertex, axis1=-2, axis2=-1)
            near_drive = None
            if near_k_flat is not None:
                drive_at_k = drive_vertex.reshape((nk, 3, nbands, nbands))[near_k_flat]
                near_drive = drive_at_k[
                    np.arange(len(near_k_flat))[:, None], np.arange(3)[None, :],
                    near_band_left[:, None], near_band_right[:, None]]
            for iobservable, observable_vertex in enumerate(observable_vertices):
                observable_diagonal = np.diagonal(observable_vertex, axis1=-2, axis2=-1)
                drive_diag = drive_diagonal.reshape(tuple(inputs.kgrid) + (3,)
                                                    + (1,)*len(component_shape) + (nbands,))
                observable_diag = observable_diagonal.reshape(
                    tuple(inputs.kgrid) + (1,) + component_shape + (nbands,))
                pop_factor = population_factor.reshape(
                    tuple(inputs.kgrid) + (1,)*(1 + len(component_shape)) + (nbands,))
                weights_i = drive_diag*observable_diag*pop_factor
                weights_e = np.zeros(tuple(inputs.kgrid) + (3,) + component_shape + (nbands,),
                                     dtype=complex)
                for direction in range(3):
                    drive_component = drive_vertex[..., direction, :, :]
                    for components in np.ndindex(component_shape):
                        observable_index = k_slices + components + (slice(None), slice(None))
                        observable_nm = observable_vertex[observable_index].swapaxes(-1, -2)
                        pair_weight = pair_factor*drive_component*observable_nm
                        weight_index = k_slices + (direction,) + components + (slice(None),)
                        weights_e[weight_index] = (np.sum(pair_weight, axis=-1)
                                                   - np.sum(pair_weight, axis=-2))

                near_observable = None
                if near_k_flat is not None:
                    observable_at_k = observable_vertex.reshape(
                        (nk, component_count, nbands, nbands))[near_k_flat]
                    near_observable = observable_at_k[
                        np.arange(len(near_k_flat))[:, None],
                        np.arange(component_count)[None, :], near_band_right[:, None],
                        near_band_left[:, None]].reshape((len(near_k_flat),) + component_shape)

                for ifermi, (occupations, derivative, near_difference) in enumerate(fermi_data):
                    fermi_weight_shape = tuple(inputs.kgrid) + (1,)*(1 + len(component_shape)) + (nbands,)
                    value_e = np.sum(weights_e*occupations.reshape(fermi_weight_shape), axis=-1)
                    if near_k_flat is not None:
                        near_weight = near_resolvent*near_difference
                        near_value = (near_weight.reshape((-1, 1) + (1,)*len(component_shape))
                                      * near_drive.reshape((-1, 3) + (1,)*len(component_shape))
                                      * near_observable.reshape((-1, 1) + component_shape))
                        np.add.at(value_e.reshape((nk, 3) + component_shape),
                                  near_k_flat, near_value)
                    value_i = (np.sum(weights_i*derivative.reshape(fermi_weight_shape), axis=-1)
                               if derivative is not None else np.zeros_like(value_e))
                    projected_channels["I"][idrive, iobservable, ifermi, iomega] = np.sum(
                        value_i, axis=k_axes)
                    projected_channels["E"][idrive, iobservable, ifermi, iomega] = np.sum(
                        value_e, axis=k_axes)
                    irecord_fermi = fermi_records.get(ifermi)
                    if irecord_fermi is not None and irecord_omega is not None:
                        index = ((idrive, iobservable) + k_slices
                                 + (irecord_fermi, irecord_omega) + response_tail)
                        response_channels["I"][index] = value_i
                        response_channels["E"][index] = value_e
    return FirstOrderResponseResult(kernel, projected_channels, response_channels, fermi_indices, omega_indices)

def _contract_first_order_lg_response(inputs, drive_vertices, observable_vertices,
                                      frequencies, responseFermiIndexList=None,
                                      responseOmegaIndexList=None,
                                      finiteTemperatureOccupations=True,
                                      fullCovariantDerivative=True, *, collision):
    """Contract a length-gauge first-order density matrix with observables."""
    return _contract_first_order_density_matrix_response(
        inputs, LengthGaugeFirstOrderDMKernel, drive_vertices, observable_vertices,
        frequencies, responseFermiIndexList, responseOmegaIndexList,
        finiteTemperatureOccupations, fullCovariantDerivative, collision=collision)

def _contract_first_order_vg_response(inputs, drive_vertices, observable_vertices,
                                      contact_vertices, frequencies,
                                      responseFermiIndexList=None,
                                      responseOmegaIndexList=None,
                                      finiteTemperatureOccupations=True,
                                      fullCovariantDerivative=True, *, collision):
    """Contract the pole-subtracted VG response and its observable contact term.

    ``contact_vertices[idrive][iobservable]`` is ``dO/dA`` with shape
    ``kgrid + observable_components + (drive_direction, nbands, nbands)``.
    The returned pole residue has no frequency axis and enters channel ``C``
    as ``i*residue/Omega``. At exactly zero frequency its periodic-BZ limit is
    removed analytically.
    """
    result = _contract_first_order_density_matrix_response(
        inputs, VelocityGaugeFirstOrderDMKernel, drive_vertices, observable_vertices,
        frequencies, responseFermiIndexList, responseOmegaIndexList,
        finiteTemperatureOccupations, fullCovariantDerivative, collision=collision)
    kernel = result.kernel
    projected_channels, response_channels = result.projected_channels, result.response_channels
    fermi_indices, omega_indices = result.fermi_indices, result.omega_indices
    _, component_shape = _first_order_observable_shape(inputs, observable_vertices)
    if len(contact_vertices) != len(drive_vertices) or any(
            len(values) != len(observable_vertices) for values in contact_vertices):
        raise ValueError("contact_vertices must be indexed by drive and observable projection")
    expected_contact_shape = tuple(inputs.kgrid) + component_shape + (
        3, inputs.energies.shape[-1], inputs.energies.shape[-1])
    if any(vertex.shape != expected_contact_shape for values in contact_vertices for vertex in values):
        raise ValueError(f"VG contact vertices must have shape {expected_contact_shape}")

    projected_channels["C"] = np.zeros_like(projected_channels["I"])
    response_channels["C"] = np.zeros_like(response_channels["I"])
    pole_shape = ((len(drive_vertices), len(observable_vertices), len(inputs.fermiList), 3)
                  + component_shape)
    pole_residual = np.zeros(pole_shape, dtype=complex)
    fermi_records = {value:index for index, value in enumerate(fermi_indices)}
    omega_records = {value:index for index, value in enumerate(omega_indices)}
    k_slices = (slice(None),)*len(inputs.kgrid)
    k_axes = tuple(range(len(inputs.kgrid)))
    nbands = inputs.energies.shape[-1]
    for ifermi, fermi in enumerate(inputs.fermiList):
        state = kernel.fermi_state(ifermi, fermi)
        irecord_fermi = fermi_records.get(ifermi)
        occupation_shape = tuple(inputs.kgrid) + (1,)*len(component_shape) + (nbands,)
        for idrive, drive_vertex in enumerate(drive_vertices):
            for iobservable, observable_vertex in enumerate(observable_vertices):
                contact_vertex = contact_vertices[idrive][iobservable]
                for direction in range(3):
                    equilibrium_derivative = _contract_first_order_observable(
                        state.derivative(direction, drive_vertex), observable_vertex,
                        component_shape)
                    contact = contact_vertex[..., direction, :, :]
                    contact_diagonal = np.diagonal(contact, axis1=-2, axis2=-1)
                    equilibrium_derivative += np.sum(
                        state.f.reshape(occupation_shape)*contact_diagonal, axis=-1)
                    residue = -equilibrium_derivative
                    pole_residual[idrive, iobservable, ifermi, direction] = np.sum(
                        residue, axis=k_axes)
                    for iomega, omega in enumerate(np.asarray(frequencies, dtype=float)):
                        if omega == 0:
                            continue
                        value = 1j*residue/omega
                        projected_channels["C"][idrive, iobservable, ifermi,
                                                iomega, direction] = np.sum(value, axis=k_axes)
                        irecord_omega = omega_records.get(iomega)
                        if irecord_fermi is not None and irecord_omega is not None:
                            index = ((idrive, iobservable) + k_slices
                                     + (irecord_fermi, irecord_omega, direction)
                                     + (slice(None),)*len(component_shape))
                            response_channels["C"][index] = value
    result.pole_residual = pole_residual
    return result

def _store_dc_galvanic_density_matrix(owner, kernel, projected_channels,
                                       response_channels, units, pole_residual=None):
    """Store a zero-frequency first-order response in the galvanic API."""
    owner.density_matrix_kernel = kernel
    owner.channel_names = tuple(projected_channels)
    owner.channel_descriptions = dict(kernel.channel_descriptions)
    if "C" in owner.channel_names:
        owner.channel_descriptions["C"] = "equilibrium observable-contact pole residue (zero in the analytic dc limit)"
    projected_channels = {name:value[:, :, :, 0].real
                          for name, value in projected_channels.items()}
    frequency_axis = 2 + len(owner.kgrid) + 1
    response_channels = {name:np.take(value, 0, axis=frequency_axis).real
                         for name, value in response_channels.items()}

    scale = 1.0/np.prod(owner.kgrid)
    owner.galvanic_units = units
    owner.conductivity_units = units
    owner.projected_sigma_channels = {
        name: response*scale for name, response in projected_channels.items()
    }
    if pole_residual is not None:
        owner.projected_sum_rule_residual = pole_residual*scale
        owner.sum_rule_residual = owner.projected_sum_rule_residual[0, 0]
        owner.sum_rule_residual_units = units+"*eV"
    owner.projected_response_channels = response_channels
    owner.projected_sigma = _sum_density_matrix_channels(owner.projected_sigma_channels)
    owner.projected_response = _sum_density_matrix_channels(owner.projected_response_channels)
    owner.sigma_channels = {
        name: response[0, 0] for name, response in owner.projected_sigma_channels.items()
    }
    owner.sigma = _sum_density_matrix_channels(owner.sigma_channels)
    for name in owner.channel_names:
        setattr(owner, f"projected_sigma_{name}", owner.projected_sigma_channels[name])
        setattr(owner, f"projected_response_{name}", owner.projected_response_channels[name])
        setattr(owner, f"sigma_{name}", owner.sigma_channels[name])

def _solve_dc_galvanic_lg_density_matrix(inputs, observable_vertices,
                                          responseFermiIndexList,
                                          finiteTemperatureOccupations,
                                          fullCovariantDerivative, units, *, collision):
    """Contract the first-order LG kernel with a vector observable."""
    full_covariant_derivative = bool(fullCovariantDerivative)
    drive_vertices = [inputs.velocity_K] + inputs.projected_velocity_K_list
    result = _contract_first_order_lg_response(
        inputs, drive_vertices, observable_vertices, np.array([0.]),
        responseFermiIndexList=responseFermiIndexList, responseOmegaIndexList=[0],
        finiteTemperatureOccupations=finiteTemperatureOccupations,
        fullCovariantDerivative=full_covariant_derivative, collision=collision)
    return result

def _solve_dc_galvanic_vg_density_matrix(inputs, observable_vertices,
                                          observable_contact_vertices,
                                          responseFermiIndexList,
                                          finiteTemperatureOccupations,
                                          fullCovariantDerivative, units, *, collision):
    """Contract the pole-subtracted first-order VG kernel at zero frequency."""
    full_covariant_derivative = bool(fullCovariantDerivative)
    if inputs.projected_velocity_K_list:
        raise NotImplementedError("VG galvanic response does not support projected drive currents")
    result = _contract_first_order_vg_response(
        inputs, [inputs.velocity_K], observable_vertices, [observable_contact_vertices],
        np.array([0.]), responseFermiIndexList=responseFermiIndexList,
        responseOmegaIndexList=[0], finiteTemperatureOccupations=finiteTemperatureOccupations,
        fullCovariantDerivative=full_covariant_derivative, collision=collision)
    return result

def _zero_vg_observable_contacts(observable_vertices):
    """Return dO/dA=0 vertices for field-independent local observables."""
    return [np.zeros(vertex.shape[:-2]+(3,)+vertex.shape[-2:], dtype=complex)
            for vertex in observable_vertices]

#
# Implemented calculaors
# Energy, velocity (ao and eigen gauges), spin velocity (ao and eigen gauges), 
# Berry curvature, spin Berry curvature
#
    

class FermiScan(Energy):
    def __init__(self, **kwarg):
        '''
        Compute the fermi function and search the band in range, based on the band energies, fermi energie(s), and beta.
        |E_{km} - E_f| < 20 kBT are thought to be involved
        self.ibmin, self.ibmax are integers, giving the involved min and max index of band (on whole BZ)

        Indexing
        --------
        self.fermiWeight[ifermi,k1,k2,k3,iband] gives the value of fermi weight the \partial_E f(E_{k,n})
        NOTE: iband only indexes the band in window, that is, iband=0 corresponding to the band self.ibmin.
        '''
        super().__init__(**kwarg)
        Emax, Emin=np.max(self.fermiList)+20*self.kBT, np.min(self.fermiList)-20*self.kBT
        condition_energyInRange=np.sum((self.energies>Emin) * (self.energies<Emax),axis=(0,1,2,))
        self.ibmin, self.ibmax= 0, self.energies.shape[-1]-1
        while(condition_energyInRange[self.ibmin]<0.5):
            self.ibmin+=1
        while(condition_energyInRange[self.ibmax]<0.5):
            self.ibmax-=1
        # Fermi weight for each band (in energy window)
        self.fermiWeight = self.fd_energy_derivative[..., self.ibmin:self.ibmax+1]
        if self.verbose==2: showElapsedTime(extraInfo=f"Band index window: {self.ibmin=} {self.ibmax=}")

class PDOS(Energy):
    def __init__(self, energyAxis:np.ndarray, eta=0.05, **kwarg) -> None:
        '''
        Compute the PDOS

        Parameters:
        -----------
        - energy Axis: energy axis for pdos intepolation, in absolute frame, NOT relative to Fermi
        - sigma: gaussian width
        '''
        super().__init__(**kwarg)

        NEDOS,=energyAxis.shape
        self.pdos=np.zeros((NEDOS,self.model.norbs),dtype=float)
        for ik1 in range(self.kgrid[0]):
            for ik2 in range(self.kgrid[1]):
                for ik3 in range(self.kgrid[2]):
                    for iband in range(self.model.norbs-self.ibmin):
                        E_band = self.energies[ik1,ik2,ik3,iband]
                        vecs   = self.vectors[ik1,ik2,ik3,:,iband]
                        rho    = abs(vecs)**2
                        if E_band < energyAxis[0] - 10*eta or E_band > energyAxis[-1] + 10*eta: continue
                        tdos=np.exp(-((E_band-energyAxis)/eta)**2/2)
                        self.pdos += tdos[:,None] * rho
        self.pdos /= np.sqrt(2*np.pi)*eta*np.prod(self.kgrid)


class Spin_diag(FermiScan):
    def __init__(self, spinProjectionList=[], gS:float=2.0, **kwarg):
        '''
        Compute spin magnetic-moment expectations in eigen-basis, in mu_B.

        Indexing
        --------
        self.spin_diag_K[ik1, ik2, ik3, idirection, ieig] gives the spin
        magnetic-moment expectation of an eigenstate.
        idirection can be 0 | 1 | 2, corresponding to x | y | z.
        '''
        super().__init__(**kwarg)

        vectors=self.vectors[:,:,:,None,:,self.ibmin:self.ibmax+1]
        spin_operator=-gS*self.model.spin/2
        if self.model.overlap:
            _sv=spin_operator[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
            self.spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
        else:
            _sv=spin_operator[None,None,None,:,:,:] @ vectors
            self.spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
        if self.verbose==2: showElapsedTime(extraInfo=f"spin magnetic moment of eig.-basis calculated")

        self.projected_spin_diag_K_list=[]
        for spinProjection in spinProjectionList:
            proj1=projectionOperator(spinProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            spin_proj=(proj1@spin_operator@proj2 + proj2@spin_operator@proj1)/2
            if self.model.overlap:
                _sv=spin_proj[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
                projected_spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
            else:
                _sv=spin_proj[None,None,None,:,:,:] @ vectors
                projected_spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
            self.projected_spin_diag_K_list.append(projected_spin_diag_K)
            #proj1_K=proj1[None,None,None,:,:]@self.vectors[...,self.ibmin:self.ibmax+1]
            #proj2_K=proj2[None,None,None,:,:]@self.vectors[...,self.ibmin:self.ibmax+1]

class Orbital_diag(FermiScan):
    def __init__(self, orbitalProjectionList=[], gL:float=1.0, **kwarg):
        '''
        Compute local orbital magnetic-moment expectations in eigen-basis,
        in mu_B, using the individual-atom approximation.

        Indexing
        --------
        self.orbital_diag_K[ik1, ik2, ik3, idirection, ieig] gives the
        orbital magnetic-moment expectation of an eigenstate.
        idirection can be 0 | 1 | 2, corresponding to x | y | z.
        '''
        super().__init__(**kwarg)

        vectors=self.vectors[:,:,:,None,:,self.ibmin:self.ibmax+1]
        orbital_operator=-gL*self.model.orbital
        if self.model.overlap:
            _ov=orbital_operator[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
            self.orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
        else:
            _ov=orbital_operator[None,None,None,:,:,:] @ vectors
            self.orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
        if self.verbose==2: showElapsedTime(extraInfo=f"local orbital magnetic moment of eig.-basis calculated")

        self.projected_orbital_diag_K_list=[]
        for orbitalProjection in orbitalProjectionList:
            proj1=projectionOperator(orbitalProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(orbitalProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            orbital_proj=(proj1@orbital_operator@proj2 + proj2@orbital_operator@proj1)/2
            if self.model.overlap:
                _ov=orbital_proj[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
                projected_orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
            else:
                _ov=orbital_proj[None,None,None,:,:,:] @ vectors
                projected_orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
            self.projected_orbital_diag_K_list.append(projected_orbital_diag_K)

class Velocity_diag(FermiScan):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Compute group velocity in original basis and eigen-basis

        Indexing
        --------
        self.RHam_K[ik1, ik2, ik3, idirection, iorb, jorb] gives velocity matrix in original basis,
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.

        self.velocity_diag_K[ik1, ik2, ik3, idirection, ieig] gives group velocity of each eigen-basis.
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.
        '''
        super().__init__(**kwarg)
        
        RH_R_W=self.model.RHam_R*self.phaseFactorR[:,None,None,None]
        self.RHam_R_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
        for iR, Rvec in enumerate(self.model.iRvec):
            self.RHam_R_intepolated[tuple(Rvec)]=RH_R_W[iR]
        self.RHam_K= 1j*np.fft.ifftn(self.RHam_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)

        if self.verbose==2: showElapsedTime(extraInfo=f"FFT RHam_R to RHam_K to calc. velocity in ao-basis finished")
        vectors=self.vectors[:,:,:,None,:,self.ibmin:self.ibmax+1]
        _RHam_K_vector=self.RHam_K @ vectors
        self.velocity_diag_K=np.sum(vectors.conj() * _RHam_K_vector, axis=(4,))
        if self.verbose==2: showElapsedTime(extraInfo=f"group velocity of eig.-basis finished")

        # extra modifications to velocity due to orbital overlaps
        if self.model.overlap:
            RO_R_W=self.model.RO_R
            self.RO_R_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
            RO_R_W=RO_R_W*self.phaseFactorR[:,None,None,None]
            for iR, Rvec in enumerate(self.model.iRvec):
                self.RO_R_intepolated[tuple(Rvec)]=RO_R_W[iR]
            self.RO_K= 1j*np.fft.ifftn(self.RO_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
            _RO_K_vector=self.RO_K @ (vectors *self.energies[...,None,None,self.ibmin:self.ibmax+1])
            self.velocity_diag_K -= np.sum(vectors.conj() *_RO_K_vector, axis=(4,))

        self.projected_velocity_diag_K_list=[]
        for velocityProjection in velocityProjectionList:
            proj1=projectionOperator(velocityProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(velocityProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            proj1=proj1[None,None,None,None,:,:]
            proj2=proj2[None,None,None,None,:,:]
            RHam_K_proj=(proj1@self.RHam_K@proj2 + proj2@self.RHam_K@proj1)/2
            _RHam_K_proj_vector=RHam_K_proj @ vectors
            projected_velocity_diag_K=np.sum(vectors.conj() * _RHam_K_proj_vector, axis=(4,))

            if self.model.overlap:
                RO_K_proj=(proj1@self.RO_K@proj2 + proj2@self.RO_K@proj1)/2
                _RO_K_proj_vector=RO_K_proj @ (vectors *self.energies[...,None,None,:])
                projected_velocity_diag_K -= np.sum(vectors.conj() *_RO_K_proj_vector, axis=(4,))
            self.projected_velocity_diag_K_list.append(projected_velocity_diag_K)

class SpinVelocity_diag(Velocity, Spin):
    def __init__(self, velocityProjectionList=[], spinvelocityProjectionList=[], **kwarg) -> None:
        '''
        Compute the diagonal elements of spin velocity matrix in original basis and eigen-basis
        and projected spin velocity matrix.
        Note, the projection is on the velocity component, corresponding to the spin-currents running in specific basis.

        Indexing:
        self.SRHam_K[ik1, ik2, ik3, idirection, igamma, iorb, jorb] gives spin velocity matrix in original basis,
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z spin magnetic moment.

        self.spinVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives spin velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z spin magnetic moment.

        if spinvelocityProjctionList is not empty
        self.spinVelocity_K_proj[iproj] gives each projected spin velocity matrix in eigen-basis.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList+spinvelocityProjectionList, **kwarg)
        
        self.spinVelocity_diag_K = np.sum(self.spin_K[...,None,:,:,:] * self.velocity_K[...,None,:,:].swapaxes(-2,-1),axis=-1).real

        self.projected_spinVelocity_diag_K_list=[]
        for iproj, proj in enumerate(spinvelocityProjectionList):
            projected_velocity_K=self.projected_velocity_K_list.pop(iproj+len(velocityProjectionList))
            self.projected_spinVelocity_diag_K_list.append(np.sum(self.spin_K[...,None,:,:,:] * projected_velocity_K[...,None,:,:].swapaxes(-2,-1),axis=-1).real)
        
        if self.verbose==2: showElapsedTime(extraInfo=f"diagonal elements of spin velocity in eig.-basis finished")

class OrbitalVelocity_diag(Velocity, Orbital):
    def __init__(self, velocityProjectionList=[], orbitalvelocityProjectionList=[], **kwarg) -> None:
        '''
        Compute the diagonal elements of orbital velocity matrix in original basis and eigen-basis
        and projected orbital velocity matrix.
        Note, the projection is on the velocity component, corresponding to the orbital-currents running in specific basis.

        Indexing:
        self.ORHam_K[ik1, ik2, ik3, idirection, igamma, iorb, jorb] gives orbital velocity matrix in original basis,
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z orbital magnetic moment.

        self.orbitalVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives orbital velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z orbital magnetic moment.

        if orbitalvelocityProjctionList is not empty
        self.orbitalVelocity_K_proj[iproj] gives each projected orbital velocity matrix in eigen-basis.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList+orbitalvelocityProjectionList, **kwarg)
        
        self.orbitalVelocity_K = np.sum(self.orbital_K[...,None,:,:,:] * self.velocity_K[...,None,:,:].swapaxes(-2,-1),axis=-1).real

        self.projected_orbitalVelocity_K_list=[]
        for iproj, proj in enumerate(orbitalvelocityProjectionList):
            projected_velocity_K=self.projected_velocity_K_list.pop(iproj+len(velocityProjectionList))
            self.projected_orbitalVelocity_K_list.append(np.sum(self.orbital_K[...,None,:,:,:] * projected_velocity_K[...,None,:,:].swapaxes(-2,-1),axis=-1).real)
        
        if self.verbose==2: showElapsedTime(extraInfo=f"diagonal elements of orbital velocity in eig.-basis finished")
        
class BerryCurv(Velocity):
    
    def __init__(self, velocityProjectionList=[],
                 _simultaneousVelocityProjection=False, **kwarg) -> None:
        '''
        Calc. Berry curvature \Omega_{alpha,beta}_{k,mn}.

        alpha and beta can be x, y, z corresponds the (cart.) space direction, k denote the crystal moment, m, n index the bands.

        Indexing: BerryCurv[ik1, ik2, ik3, ialpha, ibeta, iband, jband]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse2 # The real response kernel, \frac{1}{(Em - En)(Em - En - j \eta)}

        velocity_K_list = [self.velocity_K]+self.projected_velocity_K_list
        berry_shape = self.kgrid+(3,3,self.energies.shape[-1])
        if _simultaneousVelocityProjection:
            self.projected_BerryCurv=np.zeros((len(velocity_K_list),)+berry_shape)
            velocity_pairs = (
                (iprojection, iprojection, velocity_K, velocity_K)
                for iprojection, velocity_K in enumerate(velocity_K_list)
            )
        else:
            self.projected_BerryCurv=np.zeros(
                (len(velocity_K_list),len(velocity_K_list))+berry_shape
            )
            velocity_pairs = (
                (ivelocity_alpha, ivelocity_beta,
                 proj_velocity_alpha_K, proj_velocity_beta_K)
                for ivelocity_alpha, proj_velocity_alpha_K in enumerate(velocity_K_list)
                for ivelocity_beta, proj_velocity_beta_K in enumerate(velocity_K_list)
            )

        for ivelocity_alpha, ivelocity_beta, proj_velocity_alpha_K, proj_velocity_beta_K in velocity_pairs:
            if _simultaneousVelocityProjection:
                berry_curv = self.projected_BerryCurv[ivelocity_alpha]
            else:
                berry_curv = self.projected_BerryCurv[ivelocity_alpha,ivelocity_beta]
            for ialpha in range(3):
                for ibeta in range(3):
                    berry_curv[...,ialpha,ibeta,:]=-2*np.diagonal(
                        proj_velocity_beta_K[...,ibeta,:,:] @
                        (dEInverse2*proj_velocity_alpha_K[...,ialpha,:,:]),
                        0,-2,-1
                    ).imag

        if _simultaneousVelocityProjection:
            self.BerryCurv=self.projected_BerryCurv[0]
        else:
            self.BerryCurv=self.projected_BerryCurv[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"Berry curvature finished")

class OrbitalModern_IC(BerryCurv):
    def __init__(self, orbitalProjectionList=[], **kwarg) -> None:
        '''
        Compute the itinerant-circulation (IC) contribution to the
        equilibrium orbital magnetization, in mu_B.

        Unlike the LC magnetic-moment operator, this is a band-diagonal
        ground-state contribution. A projected IC term applies the same
        projection to both velocity vertices of the Berry curvature.

        Indexing:
        orbital_IC_K[ik1, ik2, ik3, idirection, iband].
        '''
        super().__init__(
            velocityProjectionList=orbitalProjectionList,
            _simultaneousVelocityProjection=True,
            orbitalProjectionList=orbitalProjectionList,
            **kwarg,
        )

        hbar2_over_2me = hbar**2 / (
            2 * electron_mass * elementary_charge * angstrom**2
        )

        def berryCurvatureToItinerantMoment(berryCurvature):
            orbital_tensor = (
                (self.energies[...,None,None,:] - self.model.fermi)
                * berryCurvature / hbar2_over_2me
            )
            return np.stack((
                orbital_tensor[...,1,2,:],
                orbital_tensor[...,2,0,:],
                orbital_tensor[...,0,1,:],
            ),axis=-2)

        self.orbital_IC_K = berryCurvatureToItinerantMoment(self.BerryCurv)
        self.projected_orbital_IC_K_list = [
            berryCurvatureToItinerantMoment(projected_BerryCurv)
            for projected_BerryCurv in self.projected_BerryCurv[1:]
        ]
        if self.verbose==2:
            showElapsedTime(extraInfo="modern IC orbital magnetization finished")

class OrbitalModern_Full(OrbitalModern_IC, Orbital_BlochSelfRotate):
    def __init__(self, orbitalProjectionList=[], **kwarg) -> None:
        '''
        Compute Bloch self-rotation and the equilibrium Berry correction
        using one eigensystem and one velocity calculation.

        ``orbital_K`` is the self-rotation moment matrix. The band-diagonal
        equilibrium integrand including IC is ``orbital_full_diag_K``.
        '''
        super().__init__(orbitalProjectionList=orbitalProjectionList, **kwarg)
        orbital_LC_diag_K = np.diagonal(
            self.orbital_K,axis1=-2,axis2=-1
        )
        self.orbital_full_diag_K = orbital_LC_diag_K+self.orbital_IC_K
        self.projected_orbital_full_diag_K_list = [
            np.diagonal(projected_orbital_K,axis1=-2,axis2=-1)
            + projected_orbital_IC_K
            for projected_orbital_K, projected_orbital_IC_K in zip(
                self.projected_orbital_K_list,
                self.projected_orbital_IC_K_list,
            )
        ]

class SpinOrbital(Spin, Orbital):
    '''
    Compute spin, LC orbital, and IC orbital magnetism while sharing the
    eigensystem and velocity calculation.
    '''
    def __init__(self, magneticProjectionList=[], gS = 2, **kwarg):
        super().__init__(spinProjectionList=magneticProjectionList, orbitalProjections=magneticProjectionList, gS=gS, **kwarg)

class AnormalousHallConductance(BerryCurv):

    def __init__(self, currentProjectionList=[], finiteTemperatureOccupations=False, **kwarg) -> None:
        super().__init__(velocityProjectionList=currentProjectionList, **kwarg)
        self.Iodd = False  # inversion gives minus
        self.Todd = True   # timereversal gives minus

        nProjections=self.projected_BerryCurv.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3))

        occupations = self.fd_occupations if finiteTemperatureOccupations else None
        for ifermi, fermi in enumerate(self.fermiList):
            if occupations is None:
                f = np.int8(self.energies < fermi)
            else:
                f = occupations[ifermi]
            for ivelocity_alpha in range(nProjections[0]):
                for ivelocity_beta in range(nProjections[1]):
                    for ialpha in range(3):
                        for ibeta in range(3):
                            projected_sigma[ivelocity_alpha,ivelocity_beta,ifermi,ialpha,ibeta]=np.sum(self.projected_BerryCurv[ivelocity_alpha,ivelocity_beta,...,ialpha,ibeta,:]*f)
        
        self.projected_sigma=projected_sigma*elementary_charge**2 / (hbar * angstrom * abs(self.model.Vcell) * np.prod(self.kgrid))
        #self.projected_sigma=projected_sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Anormalous Hall conductance finished")

class VelocityVelocityCorrelation_diag(Velocity_diag):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. intra-band contributions on the correlation func of velocity velocity \Omega^{alpha,beta}_{k,m}.

        alpha and beta can be x, y, z, denoting the (cart.) space direction of applied currents and the propagation of Hall currents
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        BerryCurv_diag[ik1, ik2, ik3, ialpha, ibeta, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,**kwarg)

        self.projected_BerryCurv_diag=np.zeros((len(velocityProjectionList)+1,len(velocityProjectionList)+1)+self.kgrid+(3,3,self.ibmax+1-self.ibmin))
        for ivelocity_alpha, proj_velocity_diag_K_alpha in enumerate([self.velocity_diag_K]+self.projected_velocity_diag_K_list):
            for ivelocity_beta, proj_velocity_diag_K_beta in enumerate([self.velocity_diag_K]+self.projected_velocity_diag_K_list):
                for ialpha in range(3):
                    for ibeta in range(3):
                        self.projected_BerryCurv_diag[ivelocity_alpha,ivelocity_beta,...,ialpha,ibeta,:]=-(proj_velocity_diag_K_beta[...,ibeta,:]*proj_velocity_diag_K_alpha[...,ialpha,:]).real/self.eta
        
        self.BerryCurv_diag=self.projected_BerryCurv_diag[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on intra band contributions on correlation functions <velocity><velocity> finished")

class AnormalousHallConductance_diag(VelocityVelocityCorrelation_diag):
    def __init__(self, currentProjectionList=[], **kwarg) -> None:
        '''
        Intra band contributions on the spin Hall conductance in (S/m)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, **kwarg)

        nProjections=self.projected_BerryCurv_diag.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3))
        
        for ifermi, weight in enumerate(self.fermiWeight):
            for ivelocity_alpha in range(nProjections[0]):
                for ivelocity_beta in range(nProjections[1]):
                    for ialpha in range(3):
                        for ibeta in range(3):
                            projected_sigma[ivelocity_alpha,ivelocity_beta,ifermi,ialpha,ibeta]=np.sum(self.projected_BerryCurv_diag[ivelocity_alpha,ivelocity_beta,...,ialpha,ibeta,:]*weight)
        
        self.projected_sigma=projected_sigma*elementary_charge**2 / (hbar * angstrom * abs(self.model.Vcell) * np.prod(self.kgrid))
        #self.projected_sigma=projected_sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Intra-band spin Galvanic conductance finished")

class AnormalousHallConductance_LG_DM(DensityMatrixResponse):

    def __init__(self, currentProjectionList=[], responseFermiIndexList=None,
                 finiteTemperatureOccupations=False, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Calculate the dc charge-current response from the common first-order
        length-gauge density-matrix kernel. ``fullCovariantDerivative=True``
        includes both intraband (I) and interband (E) channels. Note the CRTA
        Boltzmann conductivity is the I channel only. 
        The full ``sigma`` additionally contains the broadened E channel.
        '''
        super().__init__(electronic_stages=(), velocityProjectionList=currentProjectionList, **kwarg)
        self.Iodd = False
        self.Todd = True
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        self.channel_names = DC_DM_CHANNEL_NAMES
        self.channel_descriptions = dict(DC_DM_CHANNEL_DESCRIPTIONS)

        velocity_vertices = [self.velocity_K] + self.projected_velocity_K_list
        current_vertices = [-vertex for vertex in velocity_vertices]  # j=-v in electron units
        result = _contract_dc_first_order_density_matrix(
            self.inputs, velocity_vertices, current_vertices, (3,), responseFermiIndexList,
            finiteTemperatureOccupations, self.fullCovariantDerivative, collision=self.collision
        )
        (kernel, projected_channels, response_channels) = result.kernel, result.projected_channels, result.response_channels
        self.accept_first_order_result(result)
        self.density_matrix_kernel = kernel

        # Unit conversion
        scale=elementary_charge**2 / (hbar * angstrom * abs(self.model.Vcell) * np.prod(self.kgrid))
        #scale = 2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))

        self.projected_sigma_channels = {
            name: response * scale for name, response in projected_channels.items()
        }
        self.projected_response_channels = response_channels
        self.projected_sigma = _sum_density_matrix_channels(self.projected_sigma_channels)
        self.projected_response = _sum_density_matrix_channels(self.projected_response_channels)
        self.sigma_channels = {
            name: response[0, 0] for name, response in self.projected_sigma_channels.items()
        }
        self.sigma = _sum_density_matrix_channels(self.sigma_channels)
        for name in self.channel_names:
            setattr(self, f"projected_sigma_{name}", self.projected_sigma_channels[name])
            setattr(self, f"projected_response_{name}", self.projected_response_channels[name])
            setattr(self, f"sigma_{name}", self.sigma_channels[name])
        if self.verbose==2: showElapsedTime(extraInfo=f"Length-gauge density-matrix AHE finished")

class _LinearConductivityDM(DensityMatrixResponse):
    """Common storage/contraction for linear optical and dc conductivity."""

    gauge = "LG"
    dc = False

    def __init__(self, currentProjectionList=None, responseFermiIndexList=None,
                 responseOmegaIndexList=None, finiteTemperatureOccupations=True,
                 fullCovariantDerivative=True, **kwarg):
        projections = [] if currentProjectionList is None else currentProjectionList
        if self.gauge == "VG" and projections:
            raise NotImplementedError("VG currently supports total current; projected contact vertices are not implemented")
        if self.gauge == "VG" and (response_requires_full_basis(kwarg)):
            raise NotImplementedError("VG conductivity requires the complete orthogonal model basis")
        kwarg.pop('highOrderVelocity', None)
        if self.dc:
            kwarg['omegaList'] = np.array([0.0])
        super().__init__(electronic_stages=(), velocityProjectionList=projections, highOrderVelocity=self.gauge == "VG", **kwarg)
        self.Iodd, self.Todd = False, True
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        vertices = [self.velocity_K] + self.projected_velocity_K_list
        current_vertices = [-vertex for vertex in vertices]
        if self.gauge == "VG":
            result = _contract_first_order_vg_response(
                self.inputs, vertices, current_vertices, [[-self.w_K]], self.omegaList,
                responseFermiIndexList=responseFermiIndexList,
                responseOmegaIndexList=responseOmegaIndexList,
                finiteTemperatureOccupations=finiteTemperatureOccupations,
                fullCovariantDerivative=self.fullCovariantDerivative, collision=self.collision)
            (kernel, sigma, response, pole_residual) = result.kernel, result.projected_channels, result.response_channels, result.pole_residual
            self.accept_first_order_result(result)
            self.channel_names = ("I", "E", "C")
        else:
            result = _contract_first_order_lg_response(
                self.inputs, vertices, current_vertices, self.omegaList,
                responseFermiIndexList=responseFermiIndexList,
                responseOmegaIndexList=responseOmegaIndexList,
                finiteTemperatureOccupations=finiteTemperatureOccupations,
                fullCovariantDerivative=self.fullCovariantDerivative, collision=self.collision)
            (kernel, sigma, response) = result.kernel, result.projected_channels, result.response_channels
            self.accept_first_order_result(result)
            pole_residual = None
            self.channel_names = kernel.channel_names
        self.density_matrix_kernel = kernel
        self.channel_descriptions = dict(I="first-order diagonal population",
                                         E="first-order interband coherence")
        if self.gauge == "VG":
            self.channel_descriptions['C'] = "equilibrium contact/pole residue (vanishes on a converged periodic BZ)"
        scale = elementary_charge**2 / (
            hbar*angstrom*abs(self.model.Vcell)*np.prod(self.kgrid))
        self.conductivity_units = "S/m"
        self.fourier_convention = "exp(-i*Omega*t), Omega in eV, t in hbar/eV"
        sigma = {name:value*scale for name, value in sigma.items()}
        self.sum_rule_residual = (pole_residual[0, 0]*scale if pole_residual is not None
                                  else np.zeros((len(self.fermiList), 3, 3), complex))
        if self.dc:
            sigma = {name: value[:,:,:,0].real for name,value in sigma.items()}
            # The dc record has no frequency dimension, like the LG dc class.
            response = {name: np.take(value, 0, axis=6).real for name,value in response.items()}
        self.projected_sigma_channels = sigma
        self.projected_response_channels = response
        self.projected_sigma = _sum_density_matrix_channels(sigma)
        self.projected_response = _sum_density_matrix_channels(response)
        self.sigma_channels = {name: value[0,0] for name,value in sigma.items()}
        self.sigma = self.projected_sigma[0,0]
        setattr(self, 'sigma_'+self.gauge+'_DM', self.sigma)
        for name in self.channel_names:
            setattr(self, 'sigma_'+name, self.sigma_channels[name])
            setattr(self, 'projected_sigma_'+name, sigma[name])
            setattr(self, 'projected_response_'+name, response[name])

class AnormalousHallConductance_VG_DM(_LinearConductivityDM):
    """Full dc conductivity in S/m; sigma[fermi, drive, current].

    Uses the analytic zero-frequency limit of the first-order VG kernel,
    removing the equilibrium pure-gauge pole by the periodic BZ sum rule.
    sum_rule_residual diagnoses that sum rule before subtraction. Requires
    finite kBT and population relaxation; supports CRTA and frozen SERTA.
    """
    gauge = "VG"
    dc = True

    def __init__(self, **kwarg):
        kwarg['responseOmegaIndexList'] = [0]
        super().__init__(**kwarg)

class Opticalconductivity_LG_DM(_LinearConductivityDM):
    """Linear optical conductivity.

    sigma[fermi, omega, drive, current] is complex, in S/m, for exp(-i*Omega*t).
    omegaList contains photon energies in eV, including zero. I is Drude and
    E interband; fullCovariantDerivative=True includes both (default).
    projected_sigma has two leading current-projection axes. Optional
    projected_response records unscaled k-resolved integrands.
    """

class Opticalconductivity_VG_DM(_LinearConductivityDM):
    """Linear optical conductivity with covariant VG relaxation and contact current.

    Same output axes/units as Opticalconductivity_LG_DM. C is the unsuppressed
    i/Omega equilibrium residue on a finite k mesh. At exactly zero frequency
    the analytic periodic-BZ limit is used. Only total current is supported.
    """
    gauge = "VG"

class Opticalconductivity_VG_DM_StaticEPC(PhononResponse):
    """Complete static Gaussian EPC correction, second order in displacement.

    Linear electric response including Fan-type repeated linear vertices,
    quadratic/Debye-Waller vertices, occupations and optical current changes.
    Use a commensurate supercell carrying the desired phonon covariance and a
    CRTA background. No numerical EPC delta, displaced eigensolve, displacement
    finite difference or extra SERTA linewidth enters this calculation.

    sigma = sigma_bare + sigma_correction, all shaped (fermi,omega,drive,current)
    in S/m. movingCenters=True matches static makeDisplacedModel snapshots;
    False matches hopping-only snapshots and the phonon TDSE basis convention.
    At omega=0 the same analytic VG pole removal as Opticalconductivity_VG_DM
    is used. At nonzero omega the contact residue is retained and reported.
    Only first electric-field order is implemented; this is not a BPVE kernel.
    """
    def __init__(self, phononBath=PhononBath(),
                 movingCenters=True, displacementFactors=None, kBatchSize=4,
                 currentProjectionList=None, fullCovariantDerivative=True,
                 finiteTemperatureOccupations=True, **kwarg):
        if currentProjectionList or response_requires_full_basis(kwarg):
            raise NotImplementedError('Static EPC VG DM requires total current and a complete orthogonal basis')
        if not fullCovariantDerivative or not finiteTemperatureOccupations:
            raise ValueError('Static EPC VG DM requires full covariant finite-temperature occupations')
        kwarg.pop('highOrderVelocity', None)
        super().__init__(electronic_stages=(), highOrderVelocity=False, **kwarg)
        if np.any(~np.isfinite(self.omegaList)):
            raise ValueError('Optical frequencies must be finite')
        self.Iodd, self.Todd = False, True
        self.conductivity_units = 'S/m'
        self.fourier_convention = 'exp(-i*Omega*t), Omega in eV, t in hbar/eV'
        from .phonon_static import solve_static_epc_conductivity
        result = solve_static_epc_conductivity(self.inputs, phononBath,
            movingCenters, displacementFactors, kBatchSize, epc_model=self.epcModel, collision=self.collision)
        publish_conductivity(self, result)

class Opticalconductivity_VG_DM_Phonon(PhononResponse):
    """Frequency-domain VG DM for a prescribed periodic SSH phonon bath.

    Full conditional density matrices and phonon harmonics replace the
    linewidth-only EPC approximation. This is the stationary counterpart of
    Opticalconductivity_TDSE(phononDynamics=True), including its positive CRTA
    background and instantaneous Fermi relaxation target. It is neither
    quantum SERTA nor ECEID. ``phononDamping > 0`` samples a stationary periodic
    approximation to a damped-Langevin spectrum; zero retains exact harmonic
    motion and requires equal Einstein energies. Use the same bath parameters,
    seed, and mesh as TDSE. Converge phononPhasePoints (odd), phononBathCycles,
    the trajectory ensemble, and solverTolerance.

    sigma is the central complex-drive response; conductivity_harmonics also
    stores sidebands. lockinConductivity(times, sample, ifermi, iomega) applies
    TDSE's real-drive measurement window to the stationary spectral solution,
    separating finite-window effects from timestep/solver discrepancies.
    """
    def __init__(self, phononBath=PhononBath(statistics='classical'),
                 nPhononTrajectories=4, phononSeed=2026, phononPhasePoints=33,
                 phononDamping=0., phononBathCycles=8.,
                 solverTolerance=1e-9, solverMaxiter=100, solverRestart=30,
                 currentProjectionList=None, fullCovariantDerivative=True,
                 finiteTemperatureOccupations=True, **kwarg):
        if currentProjectionList or response_requires_full_basis(kwarg):
            raise NotImplementedError("Phonon VG DM requires total current and a complete orthogonal basis")
        if not fullCovariantDerivative or not finiteTemperatureOccupations:
            raise ValueError("Phonon VG DM requires full covariant finite-temperature equilibrium")
        kwarg.pop('highOrderVelocity', None)
        super().__init__(electronic_stages=(), highOrderVelocity=False, **kwarg)
        if self.kBT <= 0 or np.any(~np.isfinite(self.omegaList)) or np.any(np.asarray(self.omegaList) <= 0):
            raise ValueError("Phonon VG DM requires kBT > 0 and positive finite optical frequencies")
        self.Iodd, self.Todd = False, True
        self.sigma = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3), complex)
        self.projected_sigma = self.sigma[None, None]
        self.conductivity_units = 'S/m'
        self.fourier_convention = 'exp(-i*Omega*t), Omega in eV, t in hbar/eV'
        from .phonon_frequency import solve_phonon_conductivity
        result = solve_phonon_conductivity(self.inputs, phononBath,
            nPhononTrajectories, phononSeed, phononPhasePoints,
            solverTolerance, solverMaxiter, solverRestart,
            damping=phononDamping, bath_cycles=phononBathCycles, epc_model=self.epcModel, collision=self.collision)
        publish_conductivity(self, result)

    def lockinConductivity(self, times, sample=0, ifermi=0, iomega=0):
        from .phonon_frequency import lockin_conductivity
        return lockin_conductivity(self.result, times, self.omegaList[iomega], sample, ifermi, iomega)

class Opticalconductivity_TDSE(ElectronicResponse):
    """Independent real-time VG benchmark with collision relaxation.

    Propagate the occupied-state ensemble using unitary TDSE steps for
    H(A)=H0+A*v+A**2*w/2 and relaxation toward f(H(A)). CRTA uses the exact
    frozen-H step; SERTA uses symmetric collision/unitary splitting with
    equilibrium-band linewidths. With relaxation this is ensemble density
    propagation, not a single isolated pure-state TDSE.

    E(t)=E0*cos(Omega*t), A(t)=-E0*sin(Omega*t)/Omega. E0 is PEAK field in
    V/Angstrom, Omega is eV, time hbar/eV. Opposite drives remove even powers;
    2*mean(j_odd*exp(i*Omega*t))/E0 gives sigma[fermi,omega,drive,current]
    in S/m. Converge E0, n_steps_per_cycle, and transient_lifetimes. Requires
    strictly positive frequencies and linewidths for a periodic steady state.

    phononDynamics=True instead propagates electrons in prescribed SSH
    trajectories on the supplied (usually supercell) model. Positions
    and velocities are sampled from phononBath; its statistics is
    'classical' or 'wigner'. The latter gives symmetrized quantum noise, not
    quantum detailed balance. ``phononDamping > 0`` uses a stationary periodic
    damped-Langevin spectrum, including the noise required to retain the chosen
    temperature; zero gives undamped harmonic motion. No electronic force acts
    on the phonon bath.
    This path uses the exact field derivative of each TDSE step (E0 sets
    only recorded field units), sharing one eigensolve across all drives.
    It requires a positive CRTA background and forbids adding SERTA on top
    of explicit EPC. Hopping/current/contact vertices evolve in a fixed
    equilibrium point-orbital basis. sigma_samples and separate real/imaginary
    standard errors report the nPhononTrajectories ensemble; time_traces keys
    are (trajectory, fermi, omega, drive). phononCycles sets a minimum number
    of slowest-phonon periods in the measurement window, rounded up to whole
    optical periods. Converge duration, timestep, ensemble and supercell size.
    """

    def __init__(self, E0=1e-4, n_steps_per_cycle=128, n_cycles_measure=2,
                 transient_lifetimes=12., max_phase_step=0.3, recordCurrent=False,
                 currentProjectionList=None, fullCovariantDerivative=True,
                 finiteTemperatureOccupations=True, phononDynamics=False,
                 phononBath=PhononBath(statistics='classical'),
                 nPhononTrajectories=4, phononSeed=2026, phononCycles=1.,
                 phononDamping=0., phononBathCycles=8., phononPhasePoints=33,
                 collisionSpecification=None, preparedCollision=None, epcModel=None, **kwarg):
        if phononDynamics and preparedCollision is not None:
            raise ValueError("Explicit phonon dynamics does not accept preparedCollision")
        if currentProjectionList:
            raise NotImplementedError("TDSE currently supports total current only")
        if not fullCovariantDerivative or not finiteTemperatureOccupations:
            raise ValueError("TDSE benchmark requires full covariant finite-temperature equilibrium")
        if response_requires_full_basis(dict(kwarg, preparedCollision=preparedCollision)):
            raise NotImplementedError("TDSE requires the complete orthogonal model basis")
        kwarg.pop('highOrderVelocity', None)
        super().__init__(highOrderVelocity=not phononDynamics, preparedCollision=preparedCollision, **kwarg)
        if np.any(~np.isfinite(self.omegaList)) or np.any(np.asarray(self.omegaList) <= 0):
            raise ValueError("TDSE requires positive photon energies; use a DM calculator for dc")
        self.Iodd, self.Todd = False, True
        mode = collision_mode(collisionSpecification, preparedCollision)
        if not phononDynamics and mode == "epc":
            raise NotImplementedError("Opticalconductivity_TDSE does not propagate the mesh-global EPC collision operator")
        if phononDynamics:
            self.collision = prepare_phonon_background(self.electronic, collisionSpecification, kBT=self.kBT, epc_model=epcModel)
        else:
            self.preparedCollision = resolve_collision(
                self.electronic, collisionSpecification, kBT=self.kBT, prepared=preparedCollision, epc_model=epcModel)
            self.collision = self.preparedCollision.operator
        gamma = self.collision.coherence_linewidth
        if np.min(gamma) <= self.collision.specification.linewidthFloor:
            raise ValueError("Steady-state TDSE requires strictly positive relaxation linewidths")
        self.sigma = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3), complex)
        self.sigma_TDSE = self.sigma
        self.projected_sigma = self.sigma[None,None]
        self.conductivity_units = 'S/m'
        self.fourier_convention = 'exp(-i*Omega*t), Omega in eV, t in hbar/eV'
        self.E0 = E0
        self.time_traces = {}
        self.integration_info = []
        if phononDynamics:
            from .phonon_tdse import propagate_phonon_conductivity
            result = propagate_phonon_conductivity(
                self.inputs, n_steps_per_cycle=n_steps_per_cycle, n_cycles_measure=n_cycles_measure,
                transient_lifetimes=transient_lifetimes, max_phase_step=max_phase_step,
                recordCurrent=recordCurrent, phononBath=phononBath, trajectories=nPhononTrajectories,
                seed=phononSeed, phonon_cycles=phononCycles,
                phonon_damping=phononDamping, bath_cycles=phononBathCycles,
                phase_points=phononPhasePoints, E0=E0, epc_model=epcModel, collision=self.collision)
            publish_conductivity(self, result)
            self.sigma_TDSE = result.sigma
            return
        from .tdse_response import solve_electronic_conductivity
        result = solve_electronic_conductivity(self.inputs, collision=self.collision, E0=E0,
            n_steps_per_cycle=n_steps_per_cycle, n_cycles_measure=n_cycles_measure,
            transient_lifetimes=transient_lifetimes, max_phase_step=max_phase_step,
            recordCurrent=recordCurrent)
        publish_conductivity(self, result)
        self.sigma_TDSE = result.sigma

class BerryConnectionPolarizability(Velocity):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Compute the Berry-connection polarizability (BCP) tensor
        c.f. eq. S2, PHYSICAL REVIEW LETTERS 130, 126303 (2023)

                            v^a_nm v^b_mn
        G^n_ab = 2 Re sum ------------------- .
                            (E_n-E_m)^3
                      m != n

        ``velocity_K`` is dH/dk in eV*angstrom, so ``G_K`` is in
        angstrom^2/eV. Degenerate terms selected by ``degen`` are omitted.

        Indexing:
        G_K[ik1, ik2, ik3, ia, ib, n]

        projected_G_K[iproj_a, iproj_b, ik1, ik2, ik3, ia, ib, n]
        applies the two projections independently to the two velocity
        vertices.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg,)

        self.calcDeltaEInverse()
        deltaEInverse3 = self.deltaEInverse_pure**3

        velocity_K_list = [self.velocity_K]+self.projected_velocity_K_list
        bcp_shape = self.kgrid+(3,3,self.energies.shape[-1])
        self.projected_G_K = np.zeros((len(velocity_K_list),len(velocity_K_list))+bcp_shape,dtype=float,)

        for iproj_a, velocity_a_K in enumerate(velocity_K_list):
            for iproj_b, velocity_b_K in enumerate(velocity_K_list):
                self.projected_G_K[iproj_a, iproj_b] = 2*np.einsum(
                    "...anm,...bmn,...nm->...abn",
                    velocity_a_K,
                    velocity_b_K,
                    deltaEInverse3,
                    optimize="greedy",
                ).real

        self.G_K = self.projected_G_K[0,0]
        if self.verbose==2: showElapsedTime(extraInfo="Berry-connection polarizability finished")

class BerryConnectionPolarizabilityDipole(BerryConnectionPolarizability):
    Iodd = True
    Todd = False

    def __init__(self, **kwarg) -> None:
        '''
        Compute the analytical k-space dipole of the BCP,

        G_Dipole_K[..., a, b, c, n] = partial_{k_a} G^n_{bc}.

        The derivative is evaluated locally at each k point using velocity
        matrices, their gauge-covariant derivatives, and band-energy
        derivatives. Its unit is angstrom^3/eV. The isolated-band expression
        omits intermediate states separated by less than ``degen``.
        '''
        super().__init__(highOrderVelocity=True, **kwarg)
        nbands = self.energies.shape[-1]

        eigenstateDeltaEInverse   = self.deltaEInverse_pure
        eigenstateDeltaEInverse_T = self.deltaEInverse_pure.swapaxes(-1, -2)
        deltaEInverse3 = self.deltaEInverse_pure**3
        deltaEInverse4 = self.deltaEInverse_pure**4
        bandVelocity = np.diagonal(self.velocity_K, axis1=-2, axis2=-1)

        self.G_Dipole_K = np.zeros(
            self.kgrid + (3, 3, 3, nbands),
            dtype=float,
        )
        for a in range(3):
            velocity_a = self.velocity_K[..., a, :, :]
            braDerivative = velocity_a * eigenstateDeltaEInverse
            ketDerivative = velocity_a * eigenstateDeltaEInverse_T
            # D_a V_b includes changes of both eigenstates. Keeping only one
            # derivative direction in memory matters for large Wannier models.
            velocityDerivative_K = np.zeros(self.kgrid + (3, nbands, nbands), dtype=complex)
            for b in range(3):
                velocity_b = self.velocity_K[..., b, :, :]
                velocityDerivative_K[..., b, :, :] = (
                    self.w_K[..., a, b, :, :]
                    + braDerivative @ velocity_b
                    + velocity_b @ ketDerivative
                )

            numeratorDerivative = np.einsum(
                "...bnm,...cmn,...nm->...bcn",
                velocityDerivative_K,
                self.velocity_K,
                deltaEInverse3,
                optimize="greedy",
            ) + np.einsum(
                "...bnm,...cmn,...nm->...bcn",
                self.velocity_K,
                velocityDerivative_K,
                deltaEInverse3,
                optimize="greedy",
            )
            
            deltaVelocity =  bandVelocity[..., a, :, None] - bandVelocity[..., a, None, :]
            
            denominatorDerivative = -3*np.einsum(
                "...bnm,...cmn,...nm->...bcn",
                self.velocity_K,
                self.velocity_K,
                deltaVelocity*deltaEInverse4,
                optimize="greedy",
            )
            self.G_Dipole_K[..., a, :, :, :] =  2*(numeratorDerivative + denominatorDerivative).real

        if self.verbose==2:
            showElapsedTime(extraInfo="Berry-connection polarizability dipole finished")

class SecondOrderAHC_BCP_FermiSea(BerryConnectionPolarizabilityDipole):
    Iodd = True
    Todd = True

    def __init__(self, **kwarg) -> None:
        '''
        Second-order nonlinear DC conductivity in the Fermi-sea form,
        c.f. eq. S6, PHYSICAL REVIEW LETTERS 130, 126303 (2023)

        chi_abc = sum_n integral[dk] f_n
                  (partial_b G^n_ac - partial_a G^n_bc).

        Indexing
        --------
        - sigma[ifermi, a, b, c]
            3rd-ranked conductivity. a: orientation of current; b, c: orientations of applied efield
        - response[ik1, ik2, ik3, ifermi, a, b, c]
            response function on BZ
        '''
        super().__init__(**kwarg)

        self.response=np.zeros(self.kgrid+(len(self.fermiList),3,3,3),dtype=float)
        for ifermi, occupation in enumerate(self.fd_occupations):
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        self.response[...,ifermi,a,b,c]=np.sum(
                            (self.G_Dipole_K[...,b,a,c,:]-self.G_Dipole_K[...,a,b,c,:])*occupation,axis=-1)
        
        self.sigma=np.sum(self.response,axis=(0,1,2,))/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo="2nd-order anomalous Hall conductance (Fermi sea) finished")

class SecondOrderAHC_BCP_FermiSurf(BerryConnectionPolarizability, FermiScan):
    Iodd = True
    Todd = True

    def __init__(self, **kwarg) -> None:
        '''
        Second-order nonlinear DC conductivity in the Fermi-surface form,
        obtained from Eq. S6 of PHYSICAL REVIEW LETTERS 130, 126303 (2023)
        by integration by parts:

        chi_abc = sum_n integral[dk] f'_n
                  (v^n_a G^n_bc - v^n_b G^n_ac).

        ``fermiWeight`` is f'_n and is evaluated only for the band window
        selected by ``FermiScan``.

        Indexing
        --------
        - sigma[ifermi, a, b, c]
            3rd-rank conductivity. a: current direction; b, c: electric-field
            directions.
        - response[ik1, ik2, ik3, ifermi, a, b, c]
            Fermi-surface response on the BZ after summing over bands.
        '''
        super().__init__(**kwarg)

        bandWindow = slice(self.ibmin, self.ibmax+1)
        bandVelocity = np.diagonal(self.velocity_K, axis1=-2, axis2=-1).real[..., bandWindow]

        responseFermiFirst = np.einsum(
            "f...n,...an,...bcn->f...abc",
            self.fermiWeight,
            bandVelocity,
            self.G_K[..., bandWindow],
            optimize="greedy",
        )
        responseFermiFirst = responseFermiFirst - responseFermiFirst.swapaxes(-3, -2)
        self.response = np.moveaxis(responseFermiFirst, 0, 3)
        self.sigma = np.sum(self.response, axis=(0, 1, 2)) / np.prod(self.kgrid)

        if self.verbose==2: showElapsedTime(extraInfo="2nd-order anomalous Hall conductance (Fermi surface) finished")

class BerryConnectionPolarizabilityMagSusceptibility(BerryConnectionPolarizability,Spin,Orbital_BlochSelfRotate,):
    Iodd = False
    Todd = True

    def __init__(self, mag_type="total", gS=2.0, **kwarg) -> None:
        '''
        Compute the magnetic susceptibility of the BCP from Eq. S7,

        Lambda_K[..., a, b, d, n] = partial_{B_d} G^n_{ab}.

        ``mag_type`` selects ``"spin"``, modern local-circulation
        ``"orbital"``, or their ``"total"``. Magnetic moments are represented
        in mu_B, so Lambda_K is in mu_B*angstrom^2/eV^2. Degenerate
        intermediate states selected by ``degen`` are omitted.
        '''
        validMagTypes = ("total", "spin", "orbital")
        if mag_type not in validMagTypes:
            raise ValueError(
                f"mag_type must be one of {validMagTypes}, got {mag_type!r}"
            )

        super().__init__(gS=gS,_computeSpin=mag_type in ("total", "spin"),_computeOrbitalBlochSelfRotate=mag_type in ("total", "orbital"),**kwarg,)
        nbands = self.energies.shape[-1]

        self.magneticMoment_K = np.zeros(self.kgrid + (3, nbands, nbands), dtype=complex)
        if mag_type in ("total", "spin"):
            self.magneticMoment_K += self.spin_K
        if mag_type in ("total", "orbital"):
            self.magneticMoment_K += self.orbital_K
        self.mag_type = mag_type

        nondegenerate = np.abs(self.deltaE) > self.degen
        eigenstateDeltaEInverse   = self.deltaEInverse_pure
        eigenstateDeltaEInverse_T = self.deltaEInverse_pure.swapaxes(-1, -2)
        deltaEInverse3 = self.deltaEInverse**3
        deltaEInverse4 = self.deltaEInverse**4
        momentDiagonal = np.diagonal(self.magneticMoment_K, axis1=-2, axis2=-1)

        self.Lambda_K = np.zeros(self.kgrid + (3, 3, 3, nbands), dtype=float)
        for d in range(3):
            # H_B=-M_d B_d. These two terms are the changes of the bra and
            # ket eigenstates in each velocity matrix element.
            magneticPerturbation = -self.magneticMoment_K[..., d, :, :]
            braDerivative = magneticPerturbation * eigenstateDeltaEInverse
            ketDerivative = magneticPerturbation * eigenstateDeltaEInverse_T
            velocityMagDerivative_K = np.zeros(self.kgrid + (3, nbands, nbands),dtype=complex,)
            for a in range(3):
                velocity_a = self.velocity_K[..., a, :, :]
                velocityMagDerivative_K[..., a, :, :] =  braDerivative @ velocity_a + velocity_a @ ketDerivative

            numeratorDerivative = np.einsum(
                "...anm,...bmn,...nm->...abn",
                velocityMagDerivative_K, self.velocity_K, deltaEInverse3,
                optimize="greedy",
            ) + np.einsum(
                "...anm,...bmn,...nm->...abn",
                self.velocity_K, velocityMagDerivative_K, deltaEInverse3,
                optimize="greedy",
            )
            deltaMoment =  momentDiagonal[..., d, :, None] - momentDiagonal[..., d, None, :]
            denominatorDerivative = 3*np.einsum(
                "...anm,...bmn,...nm->...abn",
                self.velocity_K, self.velocity_K, deltaMoment*deltaEInverse4,
                optimize="greedy",
            )
            self.Lambda_K[..., :, :, d, :] = 2*(numeratorDerivative + denominatorDerivative).real

        if self.verbose==2: showElapsedTime(extraInfo="Berry-connection polarizability magnetic susceptibility finished")

class IntrinsicNPHE_BCP(BerryConnectionPolarizabilityDipole,BerryConnectionPolarizabilityMagSusceptibility,FermiScan):
    Iodd = True
    Todd = False

    def __init__(self, mag_type="total", gS=2.0, **kwarg) -> None:
        '''
        Compute the intrinsic nonlinear planar Hall response from Eq. S6.

        Indexing:
        ---------
        - BCP_susc_K[..., a, b, c, d, n]
            Band-resolved geometric bracket in Eq. S6, before f'_0.
        - BCP_susc_integrand_K[ifermi, ..., a, b, c, d]
            Full k-resolved integrand after the Fermi-surface band sum.
        - Upsilon[ifermi, a, b, c, d]
            Uniform-k average of the integrand. a: current direction, b and c: electric field direction, d: mag. field direction

        Magnetic moments are in mu_B, so Upsilon is in
        mu_B*angstrom^3/eV^2 before the later SI conversion.
        '''
        super().__init__(mag_type=mag_type, gS=gS, **kwarg)

        bandVelocity = np.diagonal(self.velocity_K,axis1=-2,axis2=-1).real
        momentDiagonal = np.diagonal(self.magneticMoment_K,axis1=-2,axis2=-1,).real

        self.BCP_susc_K = np.einsum(
            "...an,...bcdn->...abcdn",
            bandVelocity,self.Lambda_K,
            optimize="greedy",
        )
        self.BCP_susc_K -= np.einsum(
            "...bn,...acdn->...abcdn",
            bandVelocity,self.Lambda_K,
            optimize="greedy",
        )
        self.BCP_susc_K += np.einsum(
            "...abcn,...dn->...abcdn",
            self.G_Dipole_K,momentDiagonal,
            optimize="greedy",
        )
        self.BCP_susc_K -= np.einsum(
            "...abcn,...dn->...abcdn",
            self.G_Dipole_K.swapaxes(-4, -3),momentDiagonal,
            optimize="greedy",
        )

        bandWindow = slice(self.ibmin, self.ibmax+1)
        BCP_susc_integrand_K = np.einsum(
            "f...n,...abcdn->f...abcd",
            self.fermiWeight,self.BCP_susc_K[..., bandWindow],
            optimize="greedy",
        )
        if self.model.ispin == 1:
            BCP_susc_integrand_K *= 2

        self.response = np.moveaxis(BCP_susc_integrand_K, 0, 3)
        self.Upsilon = np.sum(BCP_susc_integrand_K,axis=(1, 2, 3)) / np.prod(self.kgrid)
        self.projected_Upsilon=self.Upsilon[None,None,...]

        if self.verbose==2: showElapsedTime(extraInfo="intrinsic NPHE from BCP finished")

class BulkPhotovoltaicEffect(Velocity):
    def __init__(self, velocityProjectionList=[], responseOmegaIndexList=[], **kwarg) -> None:
        '''
        Only the I3 contribution, the Tr {va vb vc} term
        The projections are implemented.

        Parameters
        ----------
        - responseOmegaIndexList : list
            the BZ distribution of response is only recorded for the frequencies pointed here to save memory
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg)
        self.Iodd = True  # inversion gives minus
        self.Todd = True  # timereversal gives minus

        # energy difference, Ekm - Ekn, index: [ik1, ik2, ik3, ibm, ibn]
        dEmn = self.energies[...,:,None] - self.energies[...,None,:]
        # energy difference with three band axes, and fermi energy axis, 3-spatial axes 
        # index: [ik1, ik2, ik3, 1(ifermi), 1(ialpha1), 1(ialpha2), 1(ibeta), 1(ibl), ibm, ibn]
        dEnm    = -dEmn[..., None, None,None,None, None,:,:] - 1j* self.eta # (En-Em-i\eta)
        dEnl    = -dEmn[..., None, None,None,None, :,None,:] - 1j* self.eta # (En-El-i\eta)

        # occupation number [ik1, ik2, ik3, ifermi, ib]
        f = np.int8(self.energies[..., None, :] < self.fermiList[None, None, None, :, None])
        # occupation number difference fl - fn, index: [ik1, ik2, ik3, ifermi, 1(ialpha1), 1(ialpha2), 1(ibeta), ibl, 1(ibm), ibn]
        fnl = f[..., None,None,None, None,None,:] - f[..., None,None,None, :,None,None]
        
        self.projected_sigma = np.zeros((len(velocityProjectionList)+1, len(velocityProjectionList)+1, len(velocityProjectionList)+1, # iproj1, iproj2, iproj3
                                         len(self.fermiList), len(self.omegaList),   # ifermi, iomega 
                                         3, 3, 3), dtype=float)                      # ialpha1, ialpha2, ibeta
        self.projected_response = np.zeros((len(velocityProjectionList)+1, len(velocityProjectionList)+1, len(velocityProjectionList)+1, # iproj1, iproj2, iproj3
                                         self.kgrid[0], self.kgrid[1], self.kgrid[2],
                                         len(self.fermiList), len(responseOmegaIndexList),   # ifermi, iomega 
                                         3, 3, 3), dtype=float)                      # ialpha1, ialpha2, ibeta
        irecord=0
        for iomega, Omega in enumerate(self.omegaList):
            # compute the energy terms
            den    = 1/dEnm * (1/(dEnl - Omega) + 1/(dEnl + Omega))
            # compute the velocity products
            for ivelocity_alpha1, proj_velocity_alpha1_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                # index: [ik1, ik2, ik3, 1(ifermi), ialpha1, 1(ialpha2), 1(ialpha3), 1(iorbA), ibn, 1(ibm), ibl]
                v_alpha1 = proj_velocity_alpha1_K.swapaxes(-1, -2)[..., None, :, None, None, :, None, :]
                for ivelocity_alpha2, proj_velocity_alpha2_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                    # index: [ik1, ik2, ik3, 1(ifermi), 1(ialpha1), ialpha2, 1(ialpha3), ibl, ibm, 1(ibn)]
                    v_alpha2 = proj_velocity_alpha2_K[..., None, None, :, None, :, :, None]
                    for ivelocity_beta, proj_velocity_beta_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                        # index: [ik1, ik2, ik3, 1(ifermi), 1(ialpha1), 1(ialpha2), ialpha3, 1(ibl), ibm, ibn]
                        v_beta   = proj_velocity_beta_K[..., None, None, None, :, None, :, :]
                        # Sum over the 3 band indices to compute trace
                        bubble_sum = np.sum(
                            v_alpha1 * v_alpha2 * v_beta * fnl * den, 
                            axis=(0, 1, 2, -3, -2, -1)
                        ).real
                        
                        # E(t) = -dA(t)/dt = -A0 (i Omega) exp(i Omega t). So, E(Omega) = -i Omega A(Omega), E(Omega)E(-Omega) = Omega^2 A(Omega) A(-Omega)
                        self.projected_sigma[ivelocity_alpha1, ivelocity_alpha2, ivelocity_beta, :, iomega, :, :, :] = -2 * bubble_sum / (-Omega**2)

                        if iomega not in responseOmegaIndexList: continue
                        bubble_sum_k = np.sum(
                            v_alpha1 * v_alpha2 * v_beta * fnl * den, 
                            axis=(-3, -2, -1)
                        ).real                        
                        self.projected_response[ivelocity_alpha1, ivelocity_alpha2, ivelocity_beta, :, :, :, :, irecord, :, :, :] = -2 * bubble_sum_k / (-Omega**2)
                        irecord+=1
            
        self.projected_sigma /= np.prod(self.kgrid)
        # unit conversion
        factor_SI = 4*elementary_charge**2/hbar/self.model.Vcell 
        if self.model.ispin==1: factor_SI*=2
        self.projected_sigma *= factor_SI # in unit A / V^2

        # to responsibity (uA/W)
        #factor_responsivity = (753.46 / self.model.refractive_index) * 1e6 # 1e6 for micro-Amps
        #self.projected_responsibility = self.projected_sigma * factor_responsivity
        if self.verbose == 2: showElapsedTime(extraInfo=f"Bulk photovoltaic conductance finished")

class BulkPhotovoltaicEffect_VG(Velocity):
    def __init__(self, velocityProjectionList=[], responseOmegaIndexList=[], **kwarg) -> None:
        '''
        Calculates the COMPLETE Velocity Gauge BPVE by summing all four Feynman diagrams, using resonator denominator approach
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True, **kwarg)
        
        self.projected_response = np.zeros((len(velocityProjectionList)+1, len(velocityProjectionList)+1, len(velocityProjectionList)+1, # iproj1, iproj2, iproj3
                                         self.kgrid[0], self.kgrid[1], self.kgrid[2],
                                         len(self.fermiList), len(responseOmegaIndexList),   # ifermi, iomega 
                                         3, 3, 3), dtype=complex)                      # ialpha1, ialpha2, ibeta
        self.sigma_VG_Full = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)
        
        # 3D energy array, index (ik1, ik2, ik3, m, n, l)
        E_m = self.energies[..., :, None, None]
        E_n = self.energies[..., None, :, None]
        E_l = self.energies[..., None, None, :]
        D_mn = 1.0 / (E_m - E_n - 1j * self.eta)

        # 2D energy arrays, index [ik1, ik2, ik3, m, n]
        E_m_2 = self.energies[..., :, None]
        E_n_2 = self.energies[..., None, :]
        D_mn_2 = 1.0 / (E_m_2 - E_n_2 - 1j * self.eta)

        for ifermi, fermi in enumerate(self.fermiList):
            # occupation number: index [ik1, ik2, ik3, ib]
            f = np.int8(self.energies < fermi)
            # 3D occupation arrays
            f_m =             f[..., :, None, None]
            f_n =             f[..., None, :, None]
            f_l =             f[..., None, None, :]
            f_nl= f_n - f_l
            f_lm= f_l - f_m

            # 2D occupation arrays
            f_m_2 =             f[..., :, None]
            f_n_2 =             f[..., None, :]
            f_mn_2= f_m_2 - f_n_2    

            for iomega, Omega in enumerate(self.omegaList):
                # Resonant Denominators
                S_ln_pos = 1.0 / (E_l - E_n + Omega - 1j * self.eta)
                S_ln_neg = 1.0 / (E_l - E_n - Omega - 1j * self.eta)
                S_ml_pos = 1.0 / (E_m - E_l + Omega - 1j * self.eta)
                S_ml_neg = 1.0 / (E_m - E_l - Omega - 1j * self.eta)

                term_ba = D_mn * (f_nl * S_ln_pos - f_lm * S_ml_neg)
                term_ab = D_mn * (f_nl * S_ln_neg - f_lm * S_ml_pos)
            
                S_mn_pos = 1.0 / (E_m_2 - E_n_2 - Omega - 1j * self.eta)
                S_mn_neg = 1.0 / (E_m_2 - E_n_2 + Omega - 1j * self.eta)
                G_mn_2= f_mn_2 * D_mn_2
            
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                        
                            # (ik1, ik2, ik3, norbs, norbs)
                            v_a = self.velocity_K[..., a, :, :]
                            v_b = self.velocity_K[..., b, :, :]
                            v_c = self.velocity_K[..., c, :, :]
                            v_c_nm = v_c.swapaxes(-1, -2)
                        
                            w_ab = 0.5 * (self.w_K[..., a, b, :, :] + self.w_K[..., b, a, :, :])
                            w_ca = self.w_K[..., c, a, :, :]
                            w_cb = self.w_K[..., c, b, :, :]
                            w_ca_nm = w_ca.swapaxes(-1, -2)
                            w_cb_nm = w_cb.swapaxes(-1, -2)
                        
                            u_cab = 0.5 * (self.u_K[..., c, a, b, :, :] + self.u_K[..., c, b, a, :, :])
                            
                            # =======================================================
                            # 1. Three-Point Bubble: Tr[ v^c v^a v^b ]
                            # =======================================================
                            vvv_ba = np.einsum('abcmn, abcml, abcln -> abcmnl', v_c_nm, v_b, v_a)
                            vvv_ab = np.einsum('abcmn, abcml, abcln -> abcmnl', v_c_nm, v_a, v_b)
                            I3 = np.sum(vvv_ba * term_ba + vvv_ab * term_ab, axis=(-3, -2, -1))
                            
                            # =======================================================
                            # 2. Two-Point Bubble A (Diamagnetic Perturbation): Tr[ v^c w^{ab} ]
                            # =======================================================
                            rho_w = w_ab * G_mn_2
                            I2A = np.einsum('abcmn, abcmn -> abc', v_c_nm, rho_w)
                            
                            # =======================================================
                            # 3. Two-Point Bubble B (Diamagnetic Current): Tr[ w^{ca} v^b ]
                            # =======================================================
                            rho_v_a = v_a * f_mn_2 * S_mn_neg
                            rho_v_b = v_b * f_mn_2 * S_mn_pos
                            I2B = np.einsum('abcmn, abcmn -> abc', w_cb_nm, rho_v_a) + \
                                  np.einsum('abcmn, abcmn -> abc', w_ca_nm, rho_v_b)
                            
                            # =======================================================
                            # 4. One-Point Bubble (Ground State): Tr[ u^{cab} ]
                            # =======================================================
                            u_nn = np.einsum('abcnn -> abcn', u_cab)
                            I1 = np.einsum('abcn, abcn -> abc', u_nn, f)
                            
                            # =======================================================
                            # Total Summation
                            # =======================================================
                            sigma_total_k = (I3 + I2A + I2B + I1) / Omega**2
                            # only the paramagentic contribution
                            #sigma_total_k = I1 / Omega**2

                            # BZ integration
                            self.sigma_VG_Full[ifermi, iomega, a, b, c] = np.sum(sigma_total_k) / np.prod(self.kgrid)
                            
                            # saving BZ response 
                            for irecord, iomega_record in enumerate(responseOmegaIndexList):
                                if iomega_record == iomega: break
                            else:
                                continue
                            self.projected_response[0, 0, 0, :, :, :, ifermi, irecord, a, b, c] = sigma_total_k
                            irecord+=1

        self.projected_sigma = self.sigma_VG_Full[None,None,None,...]
        # unit conversion
        #factor_SI = 4*elementary_charge**2/hbar/self.model.Vcell 
        #if self.model.ispin==1: factor_SI*=2
        #self.projected_sigma *= factor_SI # in unit A / V^2

        # to responsibity (uA/W)
        #factor_responsivity = (753.46 / self.model.refractive_index) * 1e6 # 1e6 for micro-Amps
        #self.projected_responsibility = self.projected_sigma * factor_responsivity
        if self.verbose == 2: showElapsedTime(extraInfo=f"Full Velocity Gauge BPVE finished")

class BulkPhotovoltaicEffect_VG_DM(DensityMatrixResponse):
    def __init__(self, velocityProjectionList=[], responseOmegaIndexList=[], responseFermiIndexList=None,
                 finiteTemperatureOccupations=False, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Calculates the Velocity Gauge BPVE using the frequency-domain
        Liouville density-matrix approach. ``fullCovariantDerivative=True``
        adds the gauge-covariant relaxation target and retains II, EI, IE and
        EE channels. The default reproduces the historical interband result.
        '''
        currentProjectionList = kwarg.pop("currentProjectionList", None)
        if currentProjectionList is not None and len(velocityProjectionList) == 0:
            velocityProjectionList = currentProjectionList
        super().__init__(electronic_stages=(Dipole,), velocityProjectionList=velocityProjectionList, highOrderVelocity=True,
                         highOrderDipole=fullCovariantDerivative, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        result = solve_bpve_vg(self.inputs, collision=self.collision,
            responseOmegaIndexList=responseOmegaIndexList, responseFermiIndexList=responseFermiIndexList, fullCovariantDerivative=fullCovariantDerivative, finiteTemperatureOccupations=finiteTemperatureOccupations, velocityProjectionList=velocityProjectionList)
        publish_second_order(self, result, "sigma_VG_DM")

class BulkPhotovoltaicEffect_LG_Shift(Dipole):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Calculates the Universal Shift Current (BPVE) using the Sipe-Shkrebtii Length Gauge.
        Natively evaluates BOTH the Linear Shift Current and Circular Shift Current.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderDipole=True, **kwarg)
        omega_mn = self.deltaE
        
        # Tensor Shape: [ifermi, iomega, ialpha(drive), ibeta(drive), igamma(current)]
        self.sigma_shift = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)

        for iomega, Omega in enumerate(self.omegaList):
            L_mn_pos = (1/np.pi) * self.eta / ((omega_mn - Omega)**2 + self.eta**2)
            L_mn_neg = (1/np.pi) * self.eta / ((omega_mn + Omega)**2 + self.eta**2)
            L_mn = L_mn_pos + L_mn_neg

            for ialpha in range(3): # Drive E-field a
                r_a = self.r_K[..., ialpha, :, :]
                for ibeta in range(3): # Drive E-field b
                    r_b = self.r_K[..., ibeta, :, :]
                    for igamma in range(3): # Measured Current c
                        r_a_deriv_c = self.r_deriv_K[..., igamma, ialpha, :, :]
                        r_b_deriv_c = self.r_deriv_K[..., igamma, ibeta, :, :]

                        # Shift current is symmetric in the two optical fields.
                        I_mn = -0.5j * (r_b.conj() * r_a_deriv_c + r_a.conj() * r_b_deriv_c)
                        
                        for ifermi, fermi in enumerate(self.fermiList):
                            f = np.int8(self.energies < fermi)
                            f_nm = f[..., :, None] - f[..., None, :] 
                            
                            shift_integrand = f_nm * I_mn * L_mn
                            shift_sum = np.sum(shift_integrand) * np.pi
                            
                            self.sigma_shift[ifermi, iomega, ialpha, ibeta, igamma] = shift_sum / np.prod(self.kgrid)

        self.projected_sigma = self.sigma_shift[None,None,None,...]
        
        # Unit conversion (Ensure it supports complex broadcasting)
        #factor_SI = 4 * elementary_charge**2 / hbar / self.model.Vcell 
        #if self.model.ispin == 1: factor_SI *= 2
        #self.projected_sigma *= factor_SI # in unit A / V^2

        if self.verbose == 2: showElapsedTime(extraInfo=f"Length Gauge Shift Current finished")

class BulkPhotovoltaicEffect_LG_Injection(Dipole):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Calculates the Circular Injection Current (Circular BPVE) using the 
        Sipe-Shkrebtii Length Gauge formalism.
        '''
        # No high order velocity needed! This makes LG much faster than VG.
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=False, **kwarg)
        
        omega_mn = self.deltaE
        
        # Tensor Shape: [ifermi, iomega, ialpha(drive), ibeta(drive), igamma(current)]
        # Must be COMPLEX because the Circular BPVE susceptibility is purely imaginary!
        self.sigma_inj = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)

        # 1. Compute Group Velocity Differences (Delta^c_nm)
        v_diag = np.diagonal(self.velocity_K, axis1=-2, axis2=-1) 
        Delta_v = v_diag[..., :, :, None] - v_diag[..., :, None, :] 

        # 3. Relaxation Time (tau = 1 / 2\eta in natural units)
        # Injection current scales linearly with the scattering time!
        tau = 1.0 / (2.0 * self.eta)

        for iomega, Omega in enumerate(self.omegaList):
            # Energy conservation delta functions (Lorentzians)
            L_mn_pos = (1/np.pi) * self.eta / ((omega_mn - Omega)**2 + self.eta**2)
            # Use the positive-frequency transition once, with the explicit
            # factor two from the two optical time orderings. Adding the
            # reverse transition here would cancel the ordered circular phase.
            L_mn = 2.0 * L_mn_pos

            for ialpha in range(3): # Drive E-field a
                r_a = self.r_K[..., ialpha, :, :]
                
                for ibeta in range(3): # Drive E-field b
                    r_b = self.r_K[..., ibeta, :, :]
                    
                    # Full Complex Optical Transition Tensor
                    # Real part drives Linear Injection (if magnetic)
                    # Imaginary part drives Circular Injection
                    rho_ab = r_a.conj() * r_b
                    
                    for igamma in range(3): # Measured Current c
                        Delta_c = Delta_v[..., igamma, :, :]
                        
                        # Universal Sipe Injection Integrand
                        I_mn = tau * Delta_c * rho_ab
                        
                        for ifermi, fermi in enumerate(self.fermiList):
                            f = np.int8(self.energies < fermi)
                            f_mn = f[..., :, None] - f[..., None, :] 
                            
                            inj_integrand = f_mn * I_mn * L_mn
                            inj_sum = np.sum(inj_integrand) * (-2 * np.pi)
                            
                            self.sigma_inj[ifermi, iomega, ialpha, ibeta, igamma] = inj_sum / np.prod(self.kgrid)

        self.projected_sigma = self.sigma_inj[None,None,None,...]
        
        # Unit conversion
        #factor_SI = 4 * elementary_charge**2 / hbar / self.model.Vcell 
        #if self.model.ispin == 1: factor_SI *= 2
        #self.projected_sigma *= factor_SI # in unit A / V^2

        if self.verbose == 2: showElapsedTime(extraInfo=f"Length Gauge Injection Current finished")

class BulkPhotovoltaicEffect_LG_DM(DensityMatrixResponse):
    def __init__(self, velocityProjectionList=[], responseOmegaIndexList=[], responseFermiIndexList=None,
                 finiteTemperatureOccupations=False, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Calculates the length-gauge BPVE using the frequency-domain Liouville
        density-matrix equation with an analytic covariant derivative.

        This is the finite-eta density-matrix counterpart of the resonant
        shift/injection formulas. It keeps the full complex denominators, so it
        is not expected to be identical to the Lorentzian-only LG formulas away
        from the clean resonant limit. ``fullCovariantDerivative=True`` uses
        the complete common LG kernel; the default preserves the historical
        analytic interband derivative for backward compatibility.
        '''
        currentProjectionList = kwarg.pop("currentProjectionList", None)
        if currentProjectionList is not None and len(velocityProjectionList) == 0:
            velocityProjectionList = currentProjectionList
        kwarg.pop("highOrderVelocity", None)
        super().__init__(electronic_stages=(Dipole,), velocityProjectionList=velocityProjectionList, highOrderVelocity=True, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        result = solve_bpve_lg(self.inputs, collision=self.collision,
            responseOmegaIndexList=responseOmegaIndexList, responseFermiIndexList=responseFermiIndexList, fullCovariantDerivative=fullCovariantDerivative, finiteTemperatureOccupations=finiteTemperatureOccupations, velocityProjectionList=velocityProjectionList)
        publish_second_order(self, result, "sigma_LG_DM")

class BulkPhotovoltaicEffect_TDSE(Velocity):
    def __init__(self, velocityProjectionList=[], E0=0.001, n_steps_per_cycle=100,
                 highOrderVelocity=True, include_offdiag=True, **kwarg) -> None:
        '''
        Calculates the 2nd order DC photovoltaic conductance using Real-Time TDSE.
        Evolves the density matrix using a 4th-Order Runge-Kutta (RK4) integrator.
        
        Outputs
        -------
        self.sigma_TDSE[ifermi, iomega, ialpha_drive_1, ialpha_drive_2, ibeta_measure]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=highOrderVelocity, **kwarg)
        
        self.sigma_TDSE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=float)
        self.sigma_TDSE_symmetric = np.zeros_like(self.sigma_TDSE, dtype=float)
        self.sigma_TDSE_quadrature = np.zeros_like(self.sigma_TDSE, dtype=float)
        
        # 1. Create the unperturbed Hamiltonian diagonal matrix H0
        H_diag = np.zeros(self.energies.shape + (self.model.norbs,), dtype=complex)
        for i in range(self.model.norbs):
            H_diag[..., i, i] = self.energies[..., i]

        for ifermi, fermi in enumerate(self.fermiList):
            # 2. Initial equilibrium density matrix (Fermi-Dirac)
            f = np.int8(self.energies < fermi)
            rho0 = np.zeros_like(H_diag)
            for i in range(self.model.norbs):
                rho0[..., i, i] = f[..., i]

            J0 = np.zeros(3)
            for ibeta in range(3):
                j0_val = -np.sum(rho0 * self.velocity_K[..., ibeta, :, :].swapaxes(-1, -2), axis=(-1, -2)).real
                J0[ibeta] = np.sum(j0_val)

            for iomega, Omega in enumerate(self.omegaList):
                T = 2 * np.pi / Omega
                dt = T / n_steps_per_cycle
                
                # Compute cycles needed for the transient to decay (e.g., 10 lifetimes)
                t_transient = 10.0 / self.eta
                n_cycles_transient = int(np.ceil(t_transient / T))
                n_cycles_measure = 1
                
                if self.verbose:
                    print(f"TDSE [Fermi={fermi:.2f}, Omega={Omega:.2f}]: Simulating {n_cycles_transient} transient cycles + {n_cycles_measure} measurement cycle.")

                def run_drive(directions, phases):
                    rho = rho0.copy()
                    t = 0.0
                    
                    # Phase 1: Evolve through transients to reach Non-Equilibrium Steady State
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        rho = self._rk4_step(rho, t, dt, H_diag, directions, phases, Omega, rho0, E0)
                        t += dt
                        
                    # Phase 2: Time-average the current over the final optical cycle
                    J_dc = np.zeros(3)
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        rho = self._rk4_step(rho, t, dt, H_diag, directions, phases, Omega, rho0, E0)
                        t += dt

                        A_t = [-2 * E0 / Omega * np.sin(Omega * t + phase) for phase in phases]
                        
                        # Measure current in all 3 directions
                        for ibeta in range(3):
                            J_op = -self.velocity_K[..., ibeta, :, :]
                            for i, A_i in enumerate(A_t):
                                J_op = J_op - A_i * self.w_K[..., ibeta, directions[i], :, :]
                            for i, A_i in enumerate(A_t):
                                for j, A_j in enumerate(A_t):
                                    J_op = J_op - 0.5 * A_i * A_j * self.u_K[..., ibeta, directions[i], directions[j], :, :]

                            # Physical charge current Tr(rho * (-dH/dA)).
                            j_val = np.sum(rho * J_op.swapaxes(-1, -2), axis=(-1,-2)).real
                            J_dc[ibeta] += np.sum(j_val) / (n_cycles_measure * n_steps_per_cycle)

                    return (J_dc - J0) / E0**2 / np.prod(self.kgrid)

                diagonal_response = np.zeros((3, 3))
                for ialpha in range(3): # Direction of the driving field
                    response = run_drive((ialpha,), (0.0,))
                    # Conductance = J / E^2
                    # Divide by volume/N_k
                    self.sigma_TDSE[ifermi, iomega, ialpha, ialpha, :] = response
                    self.sigma_TDSE_symmetric[ifermi, iomega, ialpha, ialpha, :] = response
                    diagonal_response[ialpha] = response

                if include_offdiag:
                    for ialpha in range(3):
                        for ibeta in range(ialpha + 1, 3):
                            same_phase = run_drive((ialpha, ibeta), (0.0, 0.0))
                            cross_symmetric = same_phase - diagonal_response[ialpha] - diagonal_response[ibeta]

                            quadrature_phase = run_drive((ialpha, ibeta), (0.0, -np.pi / 2))
                            cross_quadrature = quadrature_phase - diagonal_response[ialpha] - diagonal_response[ibeta]

                            self.sigma_TDSE_symmetric[ifermi, iomega, ialpha, ibeta, :] = cross_symmetric
                            self.sigma_TDSE_symmetric[ifermi, iomega, ibeta, ialpha, :] = cross_symmetric
                            self.sigma_TDSE_quadrature[ifermi, iomega, ialpha, ibeta, :] = cross_quadrature
                            self.sigma_TDSE_quadrature[ifermi, iomega, ibeta, ialpha, :] = -cross_quadrature

                            # Same-phase driving gives Re[chi_ab + chi_ba]. The
                            # quadrature drive gives the phase-sensitive cross term.
                            self.sigma_TDSE[ifermi, iomega, ialpha, ibeta, :] = 0.5 * (cross_symmetric - cross_quadrature)
                            self.sigma_TDSE[ifermi, iomega, ibeta, ialpha, :] = 0.5 * (cross_symmetric + cross_quadrature)
        self.projected_sigma=self.sigma_TDSE[None,None,None,...]
        # unit conversion
        #factor_SI = 4*elementary_charge**2/hbar/self.model.Vcell 
        #if self.model.ispin==1: factor_SI*=2
        #self.projected_sigma *= factor_SI # in unit A / V^2

        # to responsibity (uA/W)
        #factor_responsivity = (753.46 / self.model.refractive_index) * 1e6 # 1e6 for micro-Amps
        #self.projected_responsibility = self.projected_sigma * factor_responsivity
        if self.verbose == 2: showElapsedTime(extraInfo=f"TDSE BPVE finished")

    def _rk4_step(self, rho, t, dt, H_diag, directions, phases, Omega, rho0, E0):
        # A(t) = A0 exp(-iwt)
        # E(t) = -dA/dt = (iw) A0 exp(-iw)
        # E(w) = iw A(w), A(w) = (-i/w) E(w)
        # Assuming real electric field: E(W) = E(-W) = E0, E(t) = E(W) exp(-iWt) + E(-W) exp(iWt) = 2 E0 cos(Wt)
        # A(t) = A(W) exp(-iWt) + A(-W) exp(iWt) = (-i/W) E(W) exp(-iWt) + (i/W) E(-W) exp(iWt) = -2 E0/W sin(Wt)
        def dRho_dt(rho_state, time_val):
            A_t = [-2 * E0 / Omega * np.sin(Omega * time_val + phase) for phase in phases]
            # H(k+A) through the second order needed by this response.
            H_t = H_diag
            for i, A_i in enumerate(A_t):
                H_t = H_t + A_i * self.velocity_K[..., directions[i], :, :]
            for i, A_i in enumerate(A_t):
                for j, A_j in enumerate(A_t):
                    H_t = H_t + 0.5 * A_i * A_j * self.w_K[..., directions[i], directions[j], :, :]
            comm = H_t @ rho_state - rho_state @ H_t
            return -1j * comm - self.eta * (rho_state - rho0)

        k1 = dRho_dt(rho, t)
        k2 = dRho_dt(rho + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = dRho_dt(rho + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = dRho_dt(rho + dt * k3, t + dt)
        return rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
class BulkPhotovoltaicEffect_TDSE_LG_Naive(Dipole):
    def __init__(self, velocityProjectionList=[], E0=0.001, n_steps_per_cycle=100, **kwarg) -> None:
        '''
        Calculates the LG DC response using isolated k-point TDSE.
        WARNING: Because it lacks the k-derivative (Nabla_k rho), this will ONLY capture 
        the Injection-like optical transitions. It will FAIL to reproduce the Shift current.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=False, **kwarg)
        
        self.sigma_TDSE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=float)
        
        # 1. Diagonal Hamiltonian
        H_diag = np.zeros(self.energies.shape + (self.model.norbs,), dtype=complex)
        for i in range(self.model.norbs):
            H_diag[..., i, i] = self.energies[..., i]

        for ifermi, fermi in enumerate(self.fermiList):
            f = np.int8(self.energies < fermi)
            rho0 = np.zeros_like(H_diag)
            for i in range(self.model.norbs):
                rho0[..., i, i] = f[..., i]

            for iomega, Omega in enumerate(self.omegaList):
                T = 2 * np.pi / Omega
                dt = T / n_steps_per_cycle
                
                n_cycles_transient = int(np.ceil((10.0 / self.eta) / T))
                n_cycles_measure = 1

                for ialpha in range(3): # Direction of the driving E-field
                    r_alpha = self.r_K[..., ialpha, :, :]
                    rho = rho0.copy()
                    t = 0.0
                    
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        rho = self._rk4_step_LG(rho, t, dt, H_diag, r_alpha, Omega, rho0, E0)
                        t += dt
                        
                    J_dc = np.zeros(3)
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        rho = self._rk4_step_LG(rho, t, dt, H_diag, r_alpha, Omega, rho0, E0)
                        t += dt

                        for ibeta in range(3):
                            v_beta = self.velocity_K[..., ibeta, :, :]
                            # Charge current in LG is -v.
                            j_val = -np.sum(rho * v_beta.swapaxes(-1, -2), axis=(-1,-2)).real
                            J_dc[ibeta] += np.sum(j_val) / (n_cycles_measure * n_steps_per_cycle)
                            
                    self.sigma_TDSE[ifermi, iomega, ialpha, ialpha, :] = J_dc / E0**2 / np.prod(self.kgrid)
        
        self.projected_sigma = self.sigma_TDSE[None,None,None,...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"TDSE Naive LG finished")

    def _rk4_step_LG(self, rho, t, dt, H_diag, r_alpha, Omega, rho0, E0):
        def dRho_dt(rho_state, time_val):
            # A(t) = -2E0/Omega * sin(Omega t) -> E(t) = -dA/dt = 2*E0*cos(Omega t)
            E_t = 2.0 * E0 * np.cos(Omega * time_val)
            H_t = H_diag + E_t * r_alpha
            
            comm = H_t @ rho_state - rho_state @ H_t
            return -1j * comm - self.eta * (rho_state - rho0)

        k1 = dRho_dt(rho, t)
        k2 = dRho_dt(rho + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = dRho_dt(rho + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = dRho_dt(rho + dt * k3, t + dt)
        return rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class InverseFaradayEffect_LG(Dipole, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], responseOmegaIndexList=[],
                 responseFermiIndexList=None, **kwarg) -> None:
        '''
        Computing IFE with length gauge, using resonator denominator approach.
        unit: \muB / (V/\angst.)^2

        Indexing
        --------
        - chi_IFE[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure] : np.ndarray(dtype=complex)
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=False, **kwarg)
        
        nbasis=self.energies.shape[-1]
        M_K = np.zeros(self.kgrid + (3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        self.chi_IFE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)

        responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(self.omegaList))
        if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
            responseFermiIndexList = range(len(self.fermiList))
        responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(self.fermiList))
        responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
        responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
        self.projected_response = np.zeros((1, 1, 1) + self.kgrid +
                                           (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3),
                                           dtype=complex)

        # Band-axis convention below is (m, n, l), matching
        # Tr[M rho^(2)] = sum_mn M_mn rho^(2)_nm.
        E_m_3D = self.energies[..., :, None, None]
        E_n_3D = self.energies[..., None, :, None]
        E_l_3D = self.energies[..., None, None, :]
        D_nm_3D = 1.0 / (E_n_3D - E_m_3D - 1j * self.eta)

        for ifermi, fermi in enumerate(self.fermiList):
            f = np.int8(self.energies < fermi)
            f_m_3D = f[..., :, None, None]
            f_n_3D = f[..., None, :, None]
            f_l_3D = f[..., None, None, :]
            
            f_lm = f_l_3D - f_m_3D
            f_nl = f_n_3D - f_l_3D

            for iomega, Omega in enumerate(self.omegaList):
                S_lm_pos = 1.0 / (E_l_3D - E_m_3D + Omega - 1j * self.eta)
                S_nl_pos = 1.0 / (E_n_3D - E_l_3D + Omega - 1j * self.eta)
                S_lm_neg = 1.0 / (E_l_3D - E_m_3D - Omega - 1j * self.eta)
                S_nl_neg = 1.0 / (E_n_3D - E_l_3D - Omega - 1j * self.eta)
                
                term_ba = D_nm_3D * (-f_lm * S_lm_pos + f_nl * S_nl_neg)
                term_ab = D_nm_3D * ( f_nl * S_nl_pos - f_lm * S_lm_neg)
                irecord_omega = responseOmegaRecord.get(iomega)

                for c in range(3):
                    M_c = M_K[..., c, :, :]
                    for a in range(3):
                        r_a = self.r_K[..., a, :, :]
                        for b in range(3):
                            r_b = self.r_K[..., b, :, :]
                            
                            Mrr_ba = np.einsum('...mn, ...nl, ...lm -> ...mnl', M_c, r_b, r_a)
                            Mrr_ab = np.einsum('...mn, ...nl, ...lm -> ...mnl', M_c, r_a, r_b)
                            
                            integrand = Mrr_ba * term_ba + Mrr_ab * term_ab
                            chi_k = np.sum(integrand, axis=(-3, -2, -1))
                            self.chi_IFE[ifermi, iomega, a, b, c] = np.sum(chi_k) / np.prod(self.kgrid)
                            irecord_fermi = responseFermiRecord.get(ifermi)
                            if irecord_fermi is not None and irecord_omega is not None:
                                self.projected_response[0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, c] = chi_k

        self.projected_sigma = self.chi_IFE[None,None,None,...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"Length Gauge IFE finished")

class InverseFaradayEffect_LG_DM(DensityMatrixResponse):
    def __init__(self, ife_type="total", velocityProjectionList=[], responseOmegaIndexList=[],
                 responseFermiIndexList=None, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute IFE in the length gauge with a frequency-domain density matrix.
        unit: \muB / (V/\angst.)^2

        ``fullCovariantDerivative=True`` evaluates ``i D_k rho`` through stable
        divided differences of f(H), preserving II, EI, IE and EE separately.
        The first letter labels the second-order density-matrix sector and the
        second letter labels the first-order sector. In full mode, degen groups
        reporting blocks (I: within-block, E: between-block), not the total.

        Indexing
        --------
        - chi_IFE[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure]
        '''
        highOrderVelocity = bool(kwarg.pop("highOrderVelocity", False) or fullCovariantDerivative)
        super().__init__(electronic_stages=(Dipole, Spin, Orbital,), velocityProjectionList=velocityProjectionList, highOrderVelocity=highOrderVelocity, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        result = solve_ife_lg(self.inputs, collision=self.collision,
            responseOmegaIndexList=responseOmegaIndexList, responseFermiIndexList=responseFermiIndexList, fullCovariantDerivative=fullCovariantDerivative, ife_type=ife_type)
        publish_second_order(self, result, "chi_IFE")

class InverseFaradayEffect_VG_DM(DensityMatrixResponse):
    def __init__(self, ife_type="total", velocityProjectionList=[], responseOmegaIndexList=[],
                 responseFermiIndexList=None, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute IFE in the velocity gauge with a frequency-domain density matrix.
        unit: \muB / (V/\angst.)^2

        ``fullCovariantDerivative=True`` adds the collision-covariant
        equilibrium-restoring vertices built from ``C_a=iD_a``. They reduce to
        ``eta C_a f`` and ``i eta {C_a,C_b}f/2`` in CRTA and use the frozen
        secular collision superoperator in SERTA. These vertices supply the
        intraband population response missing from the clean commutator
        ``[v_a,f]``. II, EI, IE and EE density-matrix channels are retained
        separately. The magnetic-moment operator is assumed to have no
        explicit vector-potential dependence.

        Indexing
        --------
        - chi_IFE[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure] : np.ndarray(dtype=complex)
        '''
        super().__init__(electronic_stages=(Dipole, Spin, Orbital,), velocityProjectionList=velocityProjectionList, highOrderVelocity=True,
                         highOrderDipole=fullCovariantDerivative, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        result = solve_ife_vg(self.inputs, collision=self.collision,
            responseOmegaIndexList=responseOmegaIndexList, responseFermiIndexList=responseFermiIndexList, fullCovariantDerivative=fullCovariantDerivative, ife_type=ife_type)
        publish_second_order(self, result, "chi_IFE")

class InverseFaradayEffect_VG(Velocity, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], responseOmegaIndexList=[],
                 responseFermiIndexList=None, **kwarg) -> None:
        '''
        Computing IFE with velocity-gauge (vector potential/current coupling), using resonator denominator approach
        unit: \muB / (V/\angst.)^2

        Indexing
        --------
        - chi_IFE[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure] : np.ndarray(dtype=complex)
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True, **kwarg)
        
        nbasis=self.energies.shape[-1]
        M_K = np.zeros(self.kgrid + (3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        self.chi_IFE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)

        responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(self.omegaList))
        if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
            responseFermiIndexList = range(len(self.fermiList))
        responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(self.fermiList))
        responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
        responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
        self.projected_response = np.zeros((1, 1, 1) + self.kgrid +
                                           (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3),
                                           dtype=complex)
        # Matrix axes are ordered as (m, n), same convention as omega_mn in LG.
        E_m = self.energies[..., :, None, None]
        E_n = self.energies[..., None, :, None]
        E_l = self.energies[..., None, None, :]
        D_mn = 1.0 / (E_m - E_n - 1j * self.eta)

        E_m_2D = self.energies[..., :, None]
        E_n_2D = self.energies[..., None, :]
        D_mn_2D = 1.0 / (E_m_2D - E_n_2D - 1j * self.eta)

        for ifermi, fermi in enumerate(self.fermiList):
            f = np.int8(self.energies < fermi)
            f_m = f[..., :, None, None]
            f_n = f[..., None, :, None]
            f_l = f[..., None, None, :]
            f_nl = f_n - f_l
            f_lm = f_l - f_m
            f_mn_2D = f[..., :, None] - f[..., None, :]

            for iomega, Omega in enumerate(self.omegaList):
                irecord_omega = responseOmegaRecord.get(iomega)
                S_ln_pos = 1.0 / (E_l - E_n + Omega - 1j * self.eta)
                S_ln_neg = 1.0 / (E_l - E_n - Omega - 1j * self.eta)
                S_ml_pos = 1.0 / (E_m - E_l + Omega - 1j * self.eta)
                S_ml_neg = 1.0 / (E_m - E_l - Omega - 1j * self.eta)

                term_ba = D_mn * (f_nl * S_ln_pos - f_lm * S_ml_neg)
                term_ab = D_mn * (f_nl * S_ln_neg - f_lm * S_ml_pos)

                
                for a in range(3):
                    v_a = self.velocity_K[..., a, :, :]
                    for b in range(3):
                        v_b = self.velocity_K[..., b, :, :]
                        w_ab = 0.5 * (self.w_K[..., a, b, :, :] + self.w_K[..., b, a, :, :])
                        for c in range(3):
                            M_c_mn = M_K[..., c, :, :]
                            M_c_nm = M_c_mn.swapaxes(-1, -2)

                            Mvv_ba = np.einsum('...mn, ...ml, ...ln -> ...mnl', M_c_nm, v_b, v_a)
                            Mvv_ab = np.einsum('...mn, ...ml, ...ln -> ...mnl', M_c_nm, v_a, v_b)
                            para_k = np.sum(Mvv_ba * term_ba + Mvv_ab * term_ab, axis=(-3, -2, -1))

                            dia_k = np.sum(M_c_nm * w_ab * f_mn_2D * D_mn_2D, axis=(-2, -1))
                            sigma_k = (para_k + dia_k) / Omega**2
                            self.chi_IFE[ifermi, iomega, a, b, c] = np.sum(sigma_k) / np.prod(self.kgrid)
                            irecord_fermi = responseFermiRecord.get(ifermi)
                            if irecord_fermi is not None and irecord_omega is not None:
                                self.projected_response[0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, c] = sigma_k

        self.projected_sigma = self.chi_IFE[None,None,None,...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"Velocity Gauge IFE finished")

class InverseFaradayEffect_TDSE(Dipole, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], E0=0.001, n_steps_per_cycle=100,
                 include_offdiag=True, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute the velocity-gauge IFE by time-domain density-matrix propagation.
        unit: \muB / (V/\angst.)^2

        The propagation is resolved to first and second order in the vector
        potential. ``fullCovariantDerivative=True`` relaxes toward the
        field-shifted equilibrium density matrix through second order,

            rho_eq(A) = f + i A_a C_a f - A_a A_b C_a C_b f / 2,

        where C_a=iD_a. This is the time-domain counterpart of the covariant
        relaxation vertices in ``InverseFaradayEffect_VG_DM``. The II, EI, IE
        and EE channels are retained separately. No k-resolved response is
        stored by this validation calculator.

        Indexing
        --------
        - chi_TDSE[ifermi, iomega, drive1, drive2, measure]
        - chi_IFE_channels[name][ifermi, iomega, drive1, drive2, measure]

        For off-diagonal driving, ``chi_TDSE_symmetric`` and
        ``chi_TDSE_quadrature`` contain the directly measured same-phase and
        quadrature cross responses, respectively.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True,
                         highOrderDipole=fullCovariantDerivative, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        if self.fullCovariantDerivative and self.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.fullCovariantDerivative and abs(self.eta) < 1e-15:
            raise ValueError("fullCovariantDerivative requires finite eta for gauge-covariant relaxation")
        if np.any(np.abs(self.omegaList) < 1e-15):
            raise ValueError("velocity-gauge TDSE IFE is singular at exactly Omega=0")

        nbasis = self.energies.shape[-1]
        M_K = np.zeros(self.kgrid + (3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        response_shape = (len(self.fermiList), len(self.omegaList), 3, 3, 3)
        self.channel_names = AC_DM_CHANNEL_NAMES
        self.channel_descriptions = {
            "II": "second-order intraband from first-order intraband",
            "EI": "second-order interband from first-order intraband",
            "IE": "second-order intraband from first-order interband",
            "EE": "second-order interband from first-order interband and the contact vertex",
        }
        self.chi_IFE_channels = {name: np.zeros(response_shape, dtype=complex) for name in self.channel_names}
        self.chi_TDSE_channels = self.chi_IFE_channels
        self.chi_TDSE_symmetric_channels = {
            name: np.zeros(response_shape, dtype=float) for name in self.channel_names
        }
        self.chi_TDSE_quadrature_channels = {
            name: np.zeros(response_shape, dtype=float) for name in self.channel_names
        }
        diag = np.eye(nbasis)
        off_diag = 1.0 - diag
        indices = np.arange(nbasis)

        for ifermi, fermi in enumerate(self.fermiList):
            if self.fullCovariantDerivative:
                f, dfdk, d2fdk2 = self.fermiKDerivatives(ifermi)
            else:
                f = np.int8(self.energies < fermi)
                dfdk = d2fdk2 = None
            rho0 = np.zeros(self.energies.shape + (nbasis,), dtype=complex)
            rho0[..., indices, indices] = f

            for iomega, Omega in enumerate(self.omegaList):
                period = 2 * np.pi / Omega
                dt = period / n_steps_per_cycle
                n_cycles_transient = int(np.ceil((10.0 / self.eta) / period))
                n_cycles_measure = 1

                def run_drive(directions, phases):
                    drive_data = self._prepare_covariant_tdse_drive(
                        directions, rho0, f, dfdk, d2fdk2, diag, off_diag
                    )
                    zero = np.zeros_like(rho0)
                    state = tuple(zero.copy() for _ in range(6))
                    t = 0.0
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        state = self._rk4_step_channels(state, t, dt, phases, Omega, E0,
                                                        drive_data, diag, off_diag)
                        t += dt

                    M_dc = {name: np.zeros(3) for name in self.channel_names}
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        state = self._rk4_step_channels(state, t, dt, phases, Omega, E0,
                                                        drive_data, diag, off_diag)
                        t += dt
                        for ichannel, name in enumerate(self.channel_names, start=2):
                            rho2 = state[ichannel]
                            for c in range(3):
                                value_k = np.sum(rho2 * M_K[..., c, :, :].swapaxes(-1, -2), axis=(-2, -1)).real
                                M_dc[name][c] += np.sum(value_k) / (n_cycles_measure * n_steps_per_cycle)
                    normalization = E0**2 * np.prod(self.kgrid)
                    return {name: value / normalization for name, value in M_dc.items()}

                diagonal_response = {name: np.zeros((3, 3)) for name in self.channel_names}
                for a in range(3):
                    response = run_drive((a,), (0.0,))
                    for name in self.channel_names:
                        diagonal_response[name][a] = response[name]
                        self.chi_IFE_channels[name][ifermi, iomega, a, a, :] = response[name]
                        self.chi_TDSE_symmetric_channels[name][ifermi, iomega, a, a, :] = response[name]

                if include_offdiag:
                    for a in range(3):
                        for b in range(a + 1, 3):
                            same_phase = run_drive((a, b), (0.0, 0.0))
                            quadrature_phase = run_drive((a, b), (0.0, -np.pi / 2))
                            for name in self.channel_names:
                                cross_symmetric = same_phase[name] - diagonal_response[name][a] - diagonal_response[name][b]
                                cross_quadrature = quadrature_phase[name] - diagonal_response[name][a] - diagonal_response[name][b]
                                self.chi_TDSE_symmetric_channels[name][ifermi, iomega, a, b, :] = cross_symmetric
                                self.chi_TDSE_symmetric_channels[name][ifermi, iomega, b, a, :] = cross_symmetric
                                self.chi_TDSE_quadrature_channels[name][ifermi, iomega, a, b, :] = cross_quadrature
                                self.chi_TDSE_quadrature_channels[name][ifermi, iomega, b, a, :] = -cross_quadrature
                                self.chi_IFE_channels[name][ifermi, iomega, a, b, :] = 0.5 * (
                                    cross_symmetric - 1j * cross_quadrature
                                )
                                self.chi_IFE_channels[name][ifermi, iomega, b, a, :] = 0.5 * (
                                    cross_symmetric + 1j * cross_quadrature
                                )

        self.chi_IFE = sum(self.chi_IFE_channels.values())
        self.chi_TDSE = self.chi_IFE
        self.chi_TDSE_symmetric = sum(self.chi_TDSE_symmetric_channels.values())
        self.chi_TDSE_quadrature = sum(self.chi_TDSE_quadrature_channels.values())
        self.projected_sigma = self.chi_IFE[None, None, None, ...]
        self.projected_sigma_channels = {
            name: response[None, None, None, ...] for name, response in self.chi_IFE_channels.items()
        }
        self.projected_sigma_complex = self.projected_sigma
        self.projected_sigma_complex_channels = self.projected_sigma_channels
        for name in self.channel_names:
            setattr(self, f"chi_IFE_{name}", self.chi_IFE_channels[name])
            setattr(self, f"chi_TDSE_{name}", self.chi_TDSE_channels[name])
            setattr(self, f"projected_sigma_{name}", self.projected_sigma_channels[name])
        if self.verbose == 2: showElapsedTime(extraInfo="TDSE IFE finished")

    def _prepare_covariant_tdse_drive(self, directions, rho0, f, dfdk, d2fdk2, diag, off_diag):
        def commutator(left, right):
            return left @ right - right @ left

        data = {
            "velocity": [self.velocity_K[..., direction, :, :] for direction in directions],
            "comm_v_rho0": [], "comm_w_rho0": [], "q1_I": [], "q1_E": [], "c2": [],
        }
        data["comm_v_rho0"] = [commutator(v, rho0) for v in data["velocity"]]
        for direction_i in directions:
            row = []
            for direction_j in directions:
                w_ij = self.w_K[..., direction_i, direction_j, :, :]
                row.append(commutator(w_ij, rho0))
            data["comm_w_rho0"].append(row)

        if self.fullCovariantDerivative:
            for direction in directions:
                c1_i, c1_e = self.fermiCovariantFirstDerivativeChannels(f, dfdk, direction)
                data["q1_I"].append(-1j * c1_i * diag)
                data["q1_E"].append(-1j * c1_e * off_diag)
            for direction_i in directions:
                row = []
                for direction_j in directions:
                    row.append(self.fermiCovariantSecondDerivativeChannels(
                        f, dfdk, d2fdk2, direction_i, direction_j
                    ))
                data["c2"].append(row)
        return data

    def _rk4_step_channels(self, state, t, dt, phases, Omega, E0, drive_data, diag, off_diag):
        def commutator(left, right):
            return left @ right - right @ left

        def free_evolution(rho):
            return -1j * self.deltaE * rho - self.eta * rho

        def derivative(current_state, time):
            rho1_i, rho1_e, rho2_ii, rho2_ei, rho2_ie, rho2_ee = current_state
            A_t = [-2 * E0 / Omega * np.sin(Omega * time + phase) for phase in phases]
            result = [free_evolution(rho) for rho in current_state]

            for i, A_i in enumerate(A_t):
                result[1] -= 1j * A_i * drive_data["comm_v_rho0"][i]
                if self.fullCovariantDerivative:
                    result[0] += self.eta * A_i * drive_data["q1_I"][i]
                    result[1] += self.eta * A_i * drive_data["q1_E"][i]

                source_i = -1j * A_i * commutator(drive_data["velocity"][i], rho1_i)
                source_e = -1j * A_i * commutator(drive_data["velocity"][i], rho1_e)
                result[3] += source_i * off_diag
                result[4] += source_e * diag
                result[5] += source_e * off_diag

                for j, A_j in enumerate(A_t):
                    result[5] -= 0.5j * A_i * A_j * drive_data["comm_w_rho0"][i][j] * off_diag
                    if self.fullCovariantDerivative:
                        coefficient = -0.5 * self.eta * A_i * A_j
                        for ichannel, name in enumerate(self.channel_names, start=2):
                            result[ichannel] += coefficient * drive_data["c2"][i][j][name]
            return tuple(result)

        k1 = derivative(state, t)
        k2 = derivative(tuple(rho + 0.5 * dt * k for rho, k in zip(state, k1)), t + 0.5 * dt)
        k3 = derivative(tuple(rho + 0.5 * dt * k for rho, k in zip(state, k2)), t + 0.5 * dt)
        k4 = derivative(tuple(rho + dt * k for rho, k in zip(state, k3)), t + dt)
        return tuple(rho + (dt / 6.0) * (a + 2*b + 2*c + d)
                     for rho, a, b, c, d in zip(state, k1, k2, k3, k4))

class InverseFaradayEffect_BlochAcceleration_TDSE(Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], E0=0.001,
                 n_steps_per_cycle=100, include_offdiag=True, **kwarg) -> None:
        '''
        Computing IFE by time propagation with the finite Bloch-acceleration
        Hamiltonian H(k-A(t)), evaluated directly from real-space hoppings.

        This is the nonperturbative counterpart of InverseFaradayEffect_TDSE.
        It currently supports orthogonal local bases only.

        Indexing
        --------
        - chi_IFE[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure] : np.ndarray(dtype=float)
        '''
        assert len(velocityProjectionList) == 0, "Velocity projections are not used in Bloch-acceleration IFE"
        super().__init__(**kwarg)
        assert not self.model.overlap, "Bloch-acceleration TDSE currently supports only orthogonal bases"

        nbasis = self.energies.shape[-1]
        M_K = np.zeros(self.kgrid + (3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        self._R_cart = self.model.iRvec @ self.model.lattice
        self._r_cart = self.model.rList
        self._ham_R_phase = self.model.ham_R * self.phaseFactorR[:, None, None]

        self.chi_TDSE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=float)
        self.chi_TDSE_symmetric = np.zeros_like(self.chi_TDSE, dtype=float)
        self.chi_TDSE_quadrature = np.zeros_like(self.chi_TDSE, dtype=float)

        H_diag = np.zeros(self.energies.shape + (nbasis,), dtype=complex)
        for iband in range(nbasis):
            H_diag[..., iband, iband] = self.energies[..., iband]

        for ifermi, fermi in enumerate(self.fermiList):
            f = np.int8(self.energies < fermi)
            rho0 = np.zeros_like(H_diag)
            for iband in range(nbasis):
                rho0[..., iband, iband] = f[..., iband]

            M0 = np.zeros(3)
            for ic in range(3):
                M_val0 = np.sum(rho0 * M_K[..., ic, :, :].swapaxes(-1, -2), axis=(-1, -2)).real
                M0[ic] = np.sum(M_val0)

            for iomega, Omega in enumerate(self.omegaList):
                period = 2 * np.pi / Omega
                dt = period / n_steps_per_cycle
                n_cycles_transient = int(np.ceil((10.0 / self.eta) / period))
                n_cycles_measure = 1

                def run_drive(directions, phases):
                    rho = rho0.copy()
                    t = 0.0
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        rho = self._rk4_step_bloch_acceleration(rho, t, dt, directions, phases, Omega, rho0, E0)
                        t += dt

                    M_dc = np.zeros(3)
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        rho = self._rk4_step_bloch_acceleration(rho, t, dt, directions, phases, Omega, rho0, E0)
                        t += dt

                        for ic in range(3):
                            M_val = np.sum(rho * M_K[..., ic, :, :].swapaxes(-1, -2), axis=(-1, -2)).real
                            M_dc[ic] += np.sum(M_val) / (n_cycles_measure * n_steps_per_cycle)

                    return (M_dc - M0) / E0**2 / np.prod(self.kgrid)

                diagonal_response = np.zeros((3, 3))
                for ialpha in range(3):
                    response = run_drive((ialpha,), (0.0,))
                    diagonal_response[ialpha] = response
                    self.chi_TDSE[ifermi, iomega, ialpha, ialpha, :] = response
                    self.chi_TDSE_symmetric[ifermi, iomega, ialpha, ialpha, :] = response

                if include_offdiag:
                    for ialpha in range(3):
                        for ibeta in range(ialpha + 1, 3):
                            same_phase = run_drive((ialpha, ibeta), (0.0, 0.0))
                            cross_symmetric = same_phase - diagonal_response[ialpha] - diagonal_response[ibeta]

                            quadrature_phase = run_drive((ialpha, ibeta), (0.0, -np.pi / 2))
                            cross_quadrature = quadrature_phase - diagonal_response[ialpha] - diagonal_response[ibeta]

                            self.chi_TDSE_symmetric[ifermi, iomega, ialpha, ibeta, :] = cross_symmetric
                            self.chi_TDSE_symmetric[ifermi, iomega, ibeta, ialpha, :] = cross_symmetric
                            self.chi_TDSE_quadrature[ifermi, iomega, ialpha, ibeta, :] = cross_quadrature
                            self.chi_TDSE_quadrature[ifermi, iomega, ibeta, ialpha, :] = -cross_quadrature

                            self.chi_TDSE[ifermi, iomega, ialpha, ibeta, :] = 0.5 * (cross_symmetric - cross_quadrature)
                            self.chi_TDSE[ifermi, iomega, ibeta, ialpha, :] = 0.5 * (cross_symmetric + cross_quadrature)

        self.chi_IFE = self.chi_TDSE
        self.projected_sigma = self.chi_TDSE[None, None, None, ...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"Bloch-acceleration TDSE IFE finished")

    def _hamiltonian_eigenbasis_at_vector_potential(self, A_cart):
        phase_A = np.exp(1j * (self._R_cart @ A_cart))
        orbital_phase = np.exp(-1j * (A_cart @ self._r_cart))
        ham_R_A = (self._ham_R_phase * phase_A[:, None, None]
                   * orbital_phase[None, :, None] * orbital_phase.conj()[None, None, :])
        ham_R_grid = np.zeros(self.kgrid + (self.model.norbs, self.model.norbs), dtype=complex)
        for iR, Rvec in enumerate(self.model.iRvec):
            ham_R_grid[tuple(Rvec)] = ham_R_A[iR]
        ham_K_AO = np.fft.ifftn(ham_R_grid, axes=(0, 1, 2)) * np.prod(self.kgrid)
        ham_K_AO = 0.5 * (ham_K_AO + ham_K_AO.conj().swapaxes(-1, -2))
        return self.vectors.conj().swapaxes(-2, -1) @ ham_K_AO @ self.vectors

    def _rk4_step_bloch_acceleration(self, rho, t, dt, directions, phases, Omega, rho0, E0):
        def dRho_dt(rho_state, time_val):
            A_t = np.zeros(3)
            for direction, phase in zip(directions, phases):
                A_t[direction] += -2 * E0 / Omega * np.sin(Omega * time_val + phase)
            H_t = self._hamiltonian_eigenbasis_at_vector_potential(A_t)
            comm = H_t @ rho_state - rho_state @ H_t
            return -1j * comm - self.eta * (rho_state - rho0)

        k1 = dRho_dt(rho, t)
        k2 = dRho_dt(rho + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = dRho_dt(rho + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = dRho_dt(rho + dt * k3, t + dt)
        return rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class InverseFaradayTorque_TDSE(Dipole, SpinExchangeTorque, OrbitalTorque):
    def __init__(self, ife_type="total", velocityProjectionList=[], torqueProjectionList=[], E0=0.001,
                 n_steps_per_cycle=100, include_offdiag=True, exchangeVelocity=True,
                 fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute light-induced d.c. torque by velocity-gauge time-domain density-matrix propagation.
        unit: \muB eV / (V/\angst.)^2; divide by \hbar in eV fs for \muB/fs / (V/\angst.)^2.

        ``fullCovariantDerivative=True`` uses the field-shifted relaxation
        target through second order and preserves II, EI, IE and EE. The
        field-dependent torque vertices are included consistently: T1*rho1_I
        and T2*rho0 are assigned to II, while T1*rho1_E is assigned to EE.
        ``chi_TDSE`` is the directly comparable complex response tensor.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True,
                         highOrderDipole=fullCovariantDerivative, exchangeVelocity=exchangeVelocity,
                         spinTorqueProjectionList=torqueProjectionList,
                         orbitalTorqueProjectionList=torqueProjectionList, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        if self.fullCovariantDerivative and self.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.fullCovariantDerivative and abs(self.eta) < 1e-15:
            raise ValueError("fullCovariantDerivative requires finite eta for gauge-covariant relaxation")
        if np.any(np.abs(self.omegaList) < 1e-15):
            raise ValueError("velocity-gauge TDSE IFT is singular at exactly Omega=0")
        if ife_type in ["total", "spin"]: self.buildSpinTorqueVelocityCommutator()
        if ife_type in ["total", "orbital"]: self.buildOrbitalTorqueVelocityCommutator()

        nbasis = self.energies.shape[-1]
        nproj = len(torqueProjectionList) + 1
        proj_T_K = np.zeros((nproj,) + self.kgrid + (3, nbasis, nbasis), dtype=complex)
        proj_TA_K = np.zeros((nproj,) + self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        proj_TAA_K = np.zeros((nproj,) + self.kgrid + (3, 3, 3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]:
            proj_T_K[0] += self.spinTorque_K
            proj_TA_K[0] += self.spinTorqueA_K
            proj_TAA_K[0] += self.spinTorqueAA_K
        if ife_type in ["total", "orbital"]:
            proj_T_K[0] += self.orbitalTorque_K
            proj_TA_K[0] += self.orbitalTorqueA_K
            proj_TAA_K[0] += self.orbitalTorqueAA_K
        for iproj in range(len(torqueProjectionList)):
            if ife_type in ["total", "spin"]:
                proj_T_K[iproj+1] += self.projected_spinTorque_K_list[iproj]
                proj_TA_K[iproj+1] += self.projected_spinTorqueA_K_list[iproj]
                proj_TAA_K[iproj+1] += self.projected_spinTorqueAA_K_list[iproj]
            if ife_type in ["total", "orbital"]:
                proj_T_K[iproj+1] += self.projected_orbitalTorque_K_list[iproj]
                proj_TA_K[iproj+1] += self.projected_orbitalTorqueA_K_list[iproj]
                proj_TAA_K[iproj+1] += self.projected_orbitalTorqueAA_K_list[iproj]

        response_shape = (1, 1, nproj, len(self.fermiList), len(self.omegaList), 3, 3, 3)
        self.channel_names = ("II", "EI", "IE", "EE")
        self.channel_descriptions = {
            "II": "intraband channel, including T1*rho1_I and T2*rho0",
            "EI": "second-order interband from first-order intraband",
            "IE": "second-order intraband from first-order interband",
            "EE": "interband channel, including T1*rho1_E and the contact Hamiltonian vertex",
        }
        self.projected_chi_IFT_channels = {
            name: np.zeros(response_shape, dtype=complex) for name in self.channel_names
        }
        self.projected_chi_TDSE_symmetric_channels = {
            name: np.zeros(response_shape, dtype=float) for name in self.channel_names
        }
        self.projected_chi_TDSE_quadrature_channels = {
            name: np.zeros(response_shape, dtype=float) for name in self.channel_names
        }
        diag = np.eye(nbasis)
        off_diag = 1.0 - diag
        indices = np.arange(nbasis)

        for ifermi, fermi in enumerate(self.fermiList):
            if self.fullCovariantDerivative:
                f, dfdk, d2fdk2 = self.fermiKDerivatives(ifermi)
            else:
                f = np.int8(self.energies < fermi)
                dfdk = d2fdk2 = None
            rho0 = np.zeros(self.energies.shape + (nbasis,), dtype=complex)
            rho0[..., indices, indices] = f

            for iomega, Omega in enumerate(self.omegaList):
                period = 2 * np.pi / Omega
                dt = period / n_steps_per_cycle
                n_cycles_transient = int(np.ceil((10.0 / self.eta) / period))
                n_cycles_measure = 1

                def run_drive(directions, phases):
                    drive_data = self._prepare_covariant_tdse_drive(
                        directions, rho0, f, dfdk, d2fdk2, diag, off_diag
                    )
                    zero = np.zeros_like(rho0)
                    state = tuple(zero.copy() for _ in range(6))
                    t = 0.0
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        state = self._rk4_step_channels(state, t, dt, phases, Omega, E0,
                                                        drive_data, diag, off_diag)
                        t += dt

                    T_dc = {name: np.zeros((nproj, 3)) for name in self.channel_names}
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        state = self._rk4_step_channels(state, t, dt, phases, Omega, E0,
                                                        drive_data, diag, off_diag)
                        t += dt
                        A_t = [-2 * E0 / Omega * np.sin(Omega * t + phase) for phase in phases]
                        rho1_i, rho1_e = state[:2]
                        rho2 = dict(zip(self.channel_names, state[2:]))
                        for iproj_T in range(nproj):
                            for c in range(3):
                                T0_nm = proj_T_K[iproj_T, ..., c, :, :].swapaxes(-1, -2)
                                value_k = {
                                    name: np.sum(rho2[name] * T0_nm, axis=(-2, -1))
                                    for name in self.channel_names
                                }
                                T1 = np.zeros_like(proj_T_K[iproj_T, ..., c, :, :])
                                T2 = np.zeros_like(T1)
                                for i, A_i in enumerate(A_t):
                                    T1 += A_i * proj_TA_K[iproj_T, ..., c, directions[i], :, :]
                                    for j, A_j in enumerate(A_t):
                                        T2 += 0.5 * A_i * A_j * proj_TAA_K[
                                            iproj_T, ..., c, directions[i], directions[j], :, :
                                        ]
                                T1_nm = T1.swapaxes(-1, -2)
                                value_k["II"] += np.sum(rho1_i * T1_nm + rho0 * T2.swapaxes(-1, -2),
                                                        axis=(-2, -1))
                                value_k["EE"] += np.sum(rho1_e * T1_nm, axis=(-2, -1))
                                for name in self.channel_names:
                                    T_dc[name][iproj_T, c] += np.sum(value_k[name].real) / (
                                        n_cycles_measure * n_steps_per_cycle
                                    )
                    normalization = E0**2 * np.prod(self.kgrid)
                    return {name: value / normalization for name, value in T_dc.items()}

                diagonal_response = {name: np.zeros((3, nproj, 3)) for name in self.channel_names}
                for a in range(3):
                    response = run_drive((a,), (0.0,))
                    for name in self.channel_names:
                        diagonal_response[name][a] = response[name]
                        self.projected_chi_IFT_channels[name][0, 0, :, ifermi, iomega, a, a, :] = response[name]
                        self.projected_chi_TDSE_symmetric_channels[name][
                            0, 0, :, ifermi, iomega, a, a, :
                        ] = response[name]

                if include_offdiag:
                    for a in range(3):
                        for b in range(a + 1, 3):
                            same_phase = run_drive((a, b), (0.0, 0.0))
                            quadrature_phase = run_drive((a, b), (0.0, -np.pi / 2))
                            for name in self.channel_names:
                                cross_symmetric = same_phase[name] - diagonal_response[name][a] - diagonal_response[name][b]
                                cross_quadrature = quadrature_phase[name] - diagonal_response[name][a] - diagonal_response[name][b]
                                self.projected_chi_TDSE_symmetric_channels[name][
                                    0, 0, :, ifermi, iomega, a, b, :
                                ] = cross_symmetric
                                self.projected_chi_TDSE_symmetric_channels[name][
                                    0, 0, :, ifermi, iomega, b, a, :
                                ] = cross_symmetric
                                self.projected_chi_TDSE_quadrature_channels[name][
                                    0, 0, :, ifermi, iomega, a, b, :
                                ] = cross_quadrature
                                self.projected_chi_TDSE_quadrature_channels[name][
                                    0, 0, :, ifermi, iomega, b, a, :
                                ] = -cross_quadrature
                                self.projected_chi_IFT_channels[name][
                                    0, 0, :, ifermi, iomega, a, b, :
                                ] = 0.5 * (cross_symmetric - 1j * cross_quadrature)
                                self.projected_chi_IFT_channels[name][
                                    0, 0, :, ifermi, iomega, b, a, :
                                ] = 0.5 * (cross_symmetric + 1j * cross_quadrature)

        self.projected_chi_IFT = sum(self.projected_chi_IFT_channels.values())
        self.chi_IFT_channels = {
            name: response[0, 0, 0] for name, response in self.projected_chi_IFT_channels.items()
        }
        self.chi_IFT = sum(self.chi_IFT_channels.values())
        self.chi_TDSE = self.chi_IFT
        self.chi_TDSE_channels = self.chi_IFT_channels
        self.projected_sigma = self.projected_chi_IFT
        self.projected_sigma_channels = self.projected_chi_IFT_channels
        self.projected_chi_TDSE_symmetric = sum(self.projected_chi_TDSE_symmetric_channels.values())
        self.projected_chi_TDSE_quadrature = sum(self.projected_chi_TDSE_quadrature_channels.values())
        self.chi_TDSE_symmetric_channels = {
            name: response[0, 0, 0] for name, response in self.projected_chi_TDSE_symmetric_channels.items()
        }
        self.chi_TDSE_quadrature_channels = {
            name: response[0, 0, 0] for name, response in self.projected_chi_TDSE_quadrature_channels.items()
        }
        self.chi_TDSE_symmetric = self.projected_chi_TDSE_symmetric[0, 0, 0]
        self.chi_TDSE_quadrature = self.projected_chi_TDSE_quadrature[0, 0, 0]
        for name in self.channel_names:
            setattr(self, f"chi_IFT_{name}", self.chi_IFT_channels[name])
            setattr(self, f"chi_TDSE_{name}", self.chi_TDSE_channels[name])
            setattr(self, f"projected_chi_IFT_{name}", self.projected_chi_IFT_channels[name])
        if self.verbose == 2: showElapsedTime(extraInfo="TDSE IFT finished")

    _prepare_covariant_tdse_drive = InverseFaradayEffect_TDSE._prepare_covariant_tdse_drive
    _rk4_step_channels = InverseFaradayEffect_TDSE._rk4_step_channels

class InverseFaradayTorque_LG_DM(DensityMatrixResponse):
    def __init__(self, ife_type="total", velocityProjectionList=[], torqueProjectionList=[],
                 responseOmegaIndexList=[], responseFermiIndexList=None,
                 finiteTemperatureOccupations=False, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute light-induced d.c. torque in the length gauge with a frequency-domain density matrix.
        unit: \muB eV / (V/\angst.)^2; divide by \hbar in eV fs for \muB/fs / (V/\angst.)^2.
        The spin part is the exchange electronic torque -i[mu_S,H_XC(t)].

        ``fullCovariantDerivative=True`` replaces the interband-only dipole
        commutators by divided-difference iD_k and retains II, EI, IE and EE
        separately. degen groups channel-reporting blocks, not the total DM.
        The orbital electric-field vertex still uses the masked dipole. Explicit
        first-order torque vertices contracted with rho1_I and rho1_E are
        assigned to II and EE, respectively.

        Indexing
        --------
        - chi_IFT[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure] : np.ndarray(dtype=complex)
        '''
        highOrderVelocity = bool(kwarg.pop("highOrderVelocity", False) or fullCovariantDerivative)
        super().__init__(electronic_stages=(Dipole, SpinExchangeTorque, OrbitalTorque,), velocityProjectionList=velocityProjectionList,
                         highOrderVelocity=highOrderVelocity,
                         spinTorqueProjectionList=torqueProjectionList,
                         orbitalTorqueProjectionList=torqueProjectionList, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        if ife_type in ("total", "spin"):
            self.prepare_torque_vertices("spin", "dipole")
        if ife_type in ("total", "orbital"):
            self.prepare_torque_vertices("orbital", "dipole")
        result = solve_torque_lg(self.inputs, collision=self.collision,
            responseOmegaIndexList=responseOmegaIndexList, responseFermiIndexList=responseFermiIndexList, fullCovariantDerivative=fullCovariantDerivative, finiteTemperatureOccupations=finiteTemperatureOccupations, ife_type=ife_type, torqueProjectionList=torqueProjectionList)
        publish_second_order(self, result, "chi_IFT")

class InverseFaradayTorque_VG_DM(DensityMatrixResponse):
    def __init__(self, ife_type="total", velocityProjectionList=[], torqueProjectionList=[],
                 responseOmegaIndexList=[], responseFermiIndexList=None,
                 finiteTemperatureOccupations=False, exchangeVelocity=True,
                 fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute light-induced d.c. torque in the velocity gauge with a frequency-domain density matrix.
        unit: \muB eV / (V/\angst.)^2; divide by \hbar in eV fs for \muB/fs / (V/\angst.)^2.

        ``fullCovariantDerivative=True`` adds the collision-covariant
        equilibrium-restoring vertices and preserves II, EI, IE and EE. CRTA
        and frozen secular SERTA collision operators are supported. T1*rho1_I
        and T2*rho0 are assigned to II; T1*rho1_E is assigned to EE.
        '''
        super().__init__(electronic_stages=(Dipole, SpinExchangeTorque, OrbitalTorque,), velocityProjectionList=velocityProjectionList, highOrderVelocity=True,
                         highOrderDipole=fullCovariantDerivative, exchangeVelocity=exchangeVelocity,
                         spinTorqueProjectionList=torqueProjectionList,
                         orbitalTorqueProjectionList=torqueProjectionList, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        if ife_type in ("total", "spin"):
            self.prepare_torque_vertices("spin", "velocity")
        if ife_type in ("total", "orbital"):
            self.prepare_torque_vertices("orbital", "velocity")
        result = solve_torque_vg(self.inputs, collision=self.collision,
            responseOmegaIndexList=responseOmegaIndexList, responseFermiIndexList=responseFermiIndexList, fullCovariantDerivative=fullCovariantDerivative, finiteTemperatureOccupations=finiteTemperatureOccupations, ife_type=ife_type, torqueProjectionList=torqueProjectionList)
        publish_second_order(self, result, "chi_IFT")

class SpinBerryCurv(SpinVelocity):
    def __init__(self, velocityProjectionList=[], spinvelocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. spin Berry curvature \Omega^{alpha,gamma}_{k,mn}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gamma can be x, y, or z, corresponding to the spin magnetic-moment orientation.
        k denotes the crystal moment, m, n index the bands.

        Indexing: spinBerryCurv[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, spinvelocityProjectionList=spinvelocityProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2

        self.projected_spinBerryCurv=np.zeros((len(velocityProjectionList)+1,len(spinvelocityProjectionList)+1)+self.kgrid+(3,3,3,self.model.norbs-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
            for ispinvelocty, proj_spinVelocity_K in enumerate([self.spinVelocity_K]+self.projected_spinVelocity_K_list):
                for ialpha in range(3):
                    for ibeta in range(3):
                        for igamma in range(3):
                            self.projected_spinBerryCurv[ivelocity,ispinvelocty,...,ialpha,ibeta,igamma,:]=-2*np.diagonal(proj_spinVelocity_K[...,ibeta,igamma,:,:] @ (dEInverse2*proj_velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        
        self.spinBerryCurv=self.projected_spinBerryCurv[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"Spin Berry curvature finished")

class SpinHallConductance(SpinBerryCurv):
    def __init__(self, currentProjectionList=[], spincurrentProjectionList=[], **kwarg) -> None:
        '''
        Spin hall conductance in (S/m hbar/2e)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinvelocityProjectionList=spincurrentProjectionList, **kwarg)

        nProjections=self.projected_spinBerryCurv.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3,3))
        
        for ifermi, fermi in enumerate(self.fermiList):
            occupy=self.energies<fermi
            for ivelocity in range(nProjections[0]):
                for ispinvelocity in range(nProjections[1]):
                    for ialpha in range(3):
                        for ibeta in range(3):
                            for igamma in range(3):
                                projected_sigma[ivelocity,ispinvelocity,ifermi,ialpha,ibeta,igamma]=np.sum(self.projected_spinBerryCurv[ivelocity,ispinvelocity,...,ialpha,ibeta,igamma,:][occupy])
            #exit()
        
        #self.projected_sigma=projected_sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        self.projected_sigma=projected_sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Spin Hall conductance finished")

class SpinHallConductance_LG_DM(DensityMatrixResponse):
    def __init__(self, currentProjectionList=[], spincurrentProjectionList=[], responseFermiIndexList=None,
                 finiteTemperatureOccupations=False, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Calculate the dc spin magnetic-moment current response from the common first-order
        length-gauge density-matrix kernel. Tensor axes are electric-field
        direction, spin-current flow direction and spin polarization. The
        observable is {mu_S,v}/2, with mu_S/mu_B=-gS*S/hbar; there is no
        charge-current minus. The historical 2D normalization below is retained.
        '''
        super().__init__(electronic_stages=(SpinVelocity,), velocityProjectionList=currentProjectionList,
                         spinvelocityProjectionList=spincurrentProjectionList, **kwarg)
        self.Iodd = False
        self.Todd = False
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        self.channel_names = DC_DM_CHANNEL_NAMES
        self.channel_descriptions = dict(DC_DM_CHANNEL_DESCRIPTIONS)

        drive_vertices = [self.velocity_K] + self.projected_velocity_K_list
        observable_vertices = [self.spinVelocity_K] + self.projected_spinVelocity_K_list
        result = _contract_dc_first_order_density_matrix(
            self.inputs, drive_vertices, observable_vertices, (3, 3), responseFermiIndexList,
            finiteTemperatureOccupations, self.fullCovariantDerivative, collision=self.collision
        )
        (kernel, projected_channels, response_channels) = result.kernel, result.projected_channels, result.response_channels
        self.accept_first_order_result(result)
        self.density_matrix_kernel = kernel

        # unit conversion
        #scale=1/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        scale = 2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))

        self.projected_sigma_channels = {
            name: response * scale for name, response in projected_channels.items()
        }
        self.projected_response_channels = response_channels
        self.projected_sigma = _sum_density_matrix_channels(self.projected_sigma_channels)
        self.projected_response = _sum_density_matrix_channels(self.projected_response_channels)
        self.sigma_channels = {
            name: response[0, 0] for name, response in self.projected_sigma_channels.items()
        }
        self.sigma = _sum_density_matrix_channels(self.sigma_channels)
        for name in self.channel_names:
            setattr(self, f"projected_sigma_{name}", self.projected_sigma_channels[name])
            setattr(self, f"projected_response_{name}", self.projected_response_channels[name])
            setattr(self, f"sigma_{name}", self.sigma_channels[name])
        if self.verbose==2: showElapsedTime(extraInfo=f"Length-gauge density-matrix SHC finished")

class OrbitalBerryCurv(OrbitalVelocity):
    def __init__(self, velocityProjectionList=[], orbitalvelocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. orbital Berry curvature \Omega^{alpha,gamma}_{k,mn}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be Lx, Ly, Lz, corresponding to the orbital-polarization. 
        k denotes the crystal moment, m, n index the bands.

        Indexing: orbitalBerryCurv[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, orbitalvelocityProjectionList=orbitalvelocityProjectionList, **kwarg)
        self.calcDeltaEInverse()

        self.projected_orbitalBerryCurv=np.zeros((len(velocityProjectionList)+1,len(orbitalvelocityProjectionList)+1)+self.kgrid+(3,3,3,self.model.norbs-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
            for iorbitalvelocty, proj_orbitalVelocity_K in enumerate([self.orbitalVelocity_K]+self.projected_orbitalVelocity_K_list):
                for ialpha in range(3):
                    for ibeta in range(3):
                        for igamma in range(3):
                            self.projected_orbitalBerryCurv[ivelocity,iorbitalvelocty,...,ialpha,ibeta,igamma,:]=-2*np.diagonal(proj_orbitalVelocity_K[...,ibeta,igamma,:,:] @ (self.deltaEInverse2*proj_velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        
        self.orbitalBerryCurv=self.projected_orbitalBerryCurv[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"Orbital Berry curvature finished")

class OrbitalHallConductance(OrbitalBerryCurv):
    def __init__(self, fermiList=[], currentProjectionList=[], orbitalcurrentProjectionList=[], **kwarg) -> None:
        '''
        Orbital hall conductance in (S/m hbar/e)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, orbitalvelocityProjectionList=orbitalcurrentProjectionList, **kwarg)

        nProjections=self.projected_orbitalBerryCurv.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(fermiList),3,3,3))
        for ifermi, fermi in enumerate(fermiList):
            occupy=self.energies<fermi
            for ivelocity in range(nProjections[0]):
                for iorbitalvelocity in range(nProjections[1]):
                    for ialpha in range(3):
                        for ibeta in range(3):
                            for igamma in range(3):
                                #print(f"{ifermi=}{ivelocity=}{iorbitalvelocity=}{ialpha=}{ibeta=}{igamma=},{projected_sigma.shape=}{self.projected_orbitalBerryCurv.shape=}")
                                projected_sigma[ivelocity,iorbitalvelocity,ifermi,ialpha,ibeta,igamma]=np.sum(self.projected_orbitalBerryCurv[ivelocity,iorbitalvelocity,...,ialpha,ibeta,igamma,:][occupy])
            #exit()
        
        #self.projected_sigma=projected_sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        self.projected_sigma=projected_sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Orbital Hall conductance finished")

class _OrbitalHallConductance_LG_DM(DensityMatrixResponse):
    def __init__(self, currentProjectionList=[], orbitalcurrentProjectionList=[], responseFermiIndexList=None,
                 finiteTemperatureOccupations=False, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Calculate the dc orbital magnetic-moment current response from the common first-order
        length-gauge density-matrix kernel. Tensor axes are electric-field
        direction, orbital-current flow direction and orbital polarization. The
        observable is {mu_L,v}/2 for the selected moment stage; there is no
        charge-current minus. The historical 2D normalization below is retained.
        '''
        super().__init__(electronic_stages=(self.orbital_velocity_stage,), velocityProjectionList=currentProjectionList,
                         orbitalvelocityProjectionList=orbitalcurrentProjectionList, **kwarg)
        self.Iodd = False
        self.Todd = False
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        self.channel_names = DC_DM_CHANNEL_NAMES
        self.channel_descriptions = dict(DC_DM_CHANNEL_DESCRIPTIONS)

        drive_vertices = [self.velocity_K] + self.projected_velocity_K_list
        observable_vertices = [self.orbitalVelocity_K] + self.projected_orbitalVelocity_K_list
        result = _contract_dc_first_order_density_matrix(
            self.inputs, drive_vertices, observable_vertices, (3, 3), responseFermiIndexList,
            finiteTemperatureOccupations, self.fullCovariantDerivative, collision=self.collision
        )
        (kernel, projected_channels, response_channels) = result.kernel, result.projected_channels, result.response_channels
        self.accept_first_order_result(result)
        self.density_matrix_kernel = kernel

        # unit conversion
        #scale=1/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        scale = 2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))

        self.projected_sigma_channels = {
            name: response * scale for name, response in projected_channels.items()
        }
        self.projected_response_channels = response_channels
        self.projected_sigma = _sum_density_matrix_channels(self.projected_sigma_channels)
        self.projected_response = _sum_density_matrix_channels(self.projected_response_channels)
        self.sigma_channels = {
            name: response[0, 0] for name, response in self.projected_sigma_channels.items()
        }
        self.sigma = _sum_density_matrix_channels(self.sigma_channels)
        for name in self.channel_names:
            setattr(self, f"projected_sigma_{name}", self.projected_sigma_channels[name])
            setattr(self, f"projected_response_{name}", self.projected_response_channels[name])
            setattr(self, f"sigma_{name}", self.sigma_channels[name])
        if self.verbose==2: showElapsedTime(extraInfo=f"Length-gauge density-matrix OHC finished")

class OrbitalHallConductance_LG_DM(_OrbitalHallConductance_LG_DM):
    """Length-gauge DC response of the atom-centered orbital moment current."""
    orbital_velocity_stage = OrbitalVelocity

class OrbitalHallConductance_BlochSelfRotate_LG_DM(_OrbitalHallConductance_LG_DM):
    """DC response of {m_BlochSelfRotate, v}/2 to the electric field.

    Uses the Bloch self-rotation approximation, with the same tensor axes,
    channels and 2D normalization as OrbitalHallConductance_LG_DM. This is
    not the full Wannier LC/IC orbital-current response.
    """
    orbital_velocity_stage = OrbitalVelocity_BlochSelfRotate

class OrbitalHallConductance_BlochSelfRotateV2_LG_DM(_OrbitalHallConductance_LG_DM):
    """Length-gauge DC OHC using the PRB 114, 214403 Bloch moment."""
    orbital_velocity_stage = OrbitalVelocity_BlochSelfRotateV2


class SpinvelocityVelocityCorrelation_diag(SpinVelocity_diag, Velocity_diag):
    def __init__(self, velocityProjectionList=[], spinvelocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. intra-band contributions on the correlation func of spin-velocity and velocity \Omega^{alpha,beta,gamma}_{k,m}.

        alpha and beta can be x, y, z, denoting the (cart.) space direction of applied currents and the propagation of spin currents
        gammma can be Sx, Sy, Sz, corresponding to the orientation of spin-polarization. 
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        spinVelocityCorrelation_diag[ik1, ik2, ik3, ialpha, ibeta, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,spinvelocityProjectionList=spinvelocityProjectionList, **kwarg)

        self.projected_spinvelocityVelocityCorrelation_diag=np.zeros((len(velocityProjectionList)+1,len(spinvelocityProjectionList)+1)+self.kgrid+(3,3,3,self.ibmax+1-self.ibmin))
        for ivelocity, proj_velocity_diag_K in enumerate([self.velocity_diag_K]+self.projected_velocity_diag_K_list):
            for ispinvelocity, proj_spinvelocity_diag_K in enumerate([self.spinVelocity_diag_K]+self.projected_spinVelocity_diag_K_list):
                for ialpha in range(3):
                    for ibeta in range(3):
                        for igamma in range(3):
                            self.projected_spinvelocityVelocityCorrelation_diag[ivelocity,ispinvelocity,...,ialpha,ibeta,igamma,:]=-(proj_spinvelocity_diag_K[...,ibeta,igamma,:]*proj_velocity_diag_K[...,ialpha,:]).real/self.eta
        
        self.spinvelocityVelocityCorrelation_diag=self.projected_spinvelocityVelocityCorrelation_diag[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on intra band contributions on correlation functions <Spinvelocity><velocity> finished")

class SpinHallConductance_diag(SpinvelocityVelocityCorrelation_diag):
    def __init__(self, currentProjectionList=[], spincurrentProjectionList=[], **kwarg) -> None:
        '''
        Intra band contributions on the spin Hall conductance in (S/m hbar/2e)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinvelocityProjectionList=spincurrentProjectionList, **kwarg)

        nProjections=self.projected_spinvelocityVelocityCorrelation_diag.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3,3))
        
        for ifermi, weight in enumerate(self.fermiWeight):
            for ivelocity in range(nProjections[0]):
                for ispinvelocity in range(nProjections[1]):
                    for ialpha in range(3):
                        for ibeta in range(3):
                            for igamma in range(3):
                                projected_sigma[ivelocity,ispinvelocity,ifermi,ialpha,ibeta,igamma]=np.sum(self.projected_spinvelocityVelocityCorrelation_diag[ivelocity,ispinvelocity,...,ialpha,ibeta,igamma,:]*weight)
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        #self.projected_sigma=projected_sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Intra-band spin Galvanic conductance finished")

class SpinVelocityCorrelation(Spin,Velocity):
    def __init__(self, velocityProjectionList=[], spinProjectionList=[], **kwarg) -> None:
        '''
        Calc. inter-band contributions on the correlation func of spin and velocity \Omega^{alpha,gamma}_{k,m}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be Sx, Sy, Sz, corresponding to the orientation of spin-polarization. 
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        spinVelocityCorrelation[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,spinProjectionList=spinProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse2

        self.projected_spinVelocityCorrelation=np.zeros((len(velocityProjectionList)+1,len(spinProjectionList)+1)+self.kgrid+(3,3,self.model.norbs-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
            for ispin, proj_spin_K in enumerate([self.spin_K]+self.projected_spin_K_list):
                for ialpha in range(3):
                    for igamma in range(3):
                        self.projected_spinVelocityCorrelation[ivelocity,ispin,...,ialpha,igamma,:]=-2*np.diagonal(proj_spin_K[...,igamma,:,:] @ (dEInverse2*proj_velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        
        self.spinVelocityCorrelation=self.projected_spinVelocityCorrelation[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on inter band contributions on correlation functions <Spin \dot velocity> finished")

class SpinGalvanicConductivity_LG_DM(DensityMatrixResponse):
    """Dc electric-field-induced spin magnetic moment from the LG DM kernel.

    ``sigma[fermi, field_direction, spin_direction]`` is measured in
    ``mu_B/(V/angstrom)/cell``. The ``I`` and ``E`` channels respectively
    contain the intraband displaced population and interband coherence.
    ``fullCovariantDerivative=True`` enables both channels and supports the
    CRTA, frozen-SERTA, and mesh-coupled EPC collision specifications.
    """
    Iodd = True
    Todd = True

    def __init__(self, currentProjectionList=None, spinProjectionList=None,
                 responseFermiIndexList=None, finiteTemperatureOccupations=True,
                 fullCovariantDerivative=True, **kwarg):
        currentProjectionList = [] if currentProjectionList is None else currentProjectionList
        spinProjectionList = [] if spinProjectionList is None else spinProjectionList
        super().__init__(electronic_stages=(Spin,), velocityProjectionList=currentProjectionList,
                         spinProjectionList=spinProjectionList, **kwarg)
        observable_vertices = [self.spin_K] + self.projected_spin_K_list
        result = _solve_dc_galvanic_lg_density_matrix(
            self.inputs, observable_vertices, responseFermiIndexList,
            finiteTemperatureOccupations, fullCovariantDerivative,
            "mu_B/(V/angstrom)/cell", collision=self.collision
        )
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        self.accept_first_order_result(result)
        _store_dc_galvanic_density_matrix(self, result.kernel, result.projected_channels, result.response_channels, 'mu_B/(V/angstrom)/cell', result.pole_residual)
        if self.verbose==2: showElapsedTime(extraInfo="Length-gauge density-matrix spin galvanic conductivity finished")

class SpinGalvanicConductivity_VG_DM(DensityMatrixResponse):
    """Dc electric-field-induced spin moment from the pole-subtracted VG DM kernel.

    The spin operator is field independent, so its explicit ``dO/dA`` contact
    vertex vanishes. At zero frequency the analytic periodic-BZ limit removes
    the pure-gauge pole; ``sum_rule_residual`` records its finite-mesh residue.
    """
    Iodd = True
    Todd = True

    def __init__(self, currentProjectionList=None, spinProjectionList=None,
                 responseFermiIndexList=None, finiteTemperatureOccupations=True,
                 fullCovariantDerivative=True, **kwarg):
        currentProjectionList = [] if currentProjectionList is None else currentProjectionList
        spinProjectionList = [] if spinProjectionList is None else spinProjectionList
        if currentProjectionList:
            raise NotImplementedError("VG spin galvanic conductivity requires the total drive current")
        if response_requires_full_basis(kwarg):
            raise NotImplementedError("VG spin galvanic conductivity requires the complete orthogonal model basis")
        kwarg.pop('highOrderVelocity', None)
        super().__init__(electronic_stages=(Spin,), velocityProjectionList=[], spinProjectionList=spinProjectionList,
                         highOrderVelocity=False, **kwarg)
        observable_vertices = [self.spin_K] + self.projected_spin_K_list
        result = _solve_dc_galvanic_vg_density_matrix(
            self.inputs, observable_vertices, _zero_vg_observable_contacts(observable_vertices),
            responseFermiIndexList, finiteTemperatureOccupations,
            fullCovariantDerivative, "mu_B/(V/angstrom)/cell", collision=self.collision)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        self.accept_first_order_result(result)
        _store_dc_galvanic_density_matrix(self, result.kernel, result.projected_channels, result.response_channels, 'mu_B/(V/angstrom)/cell', result.pole_residual)
        if self.verbose==2: showElapsedTime(extraInfo="Velocity-gauge density-matrix spin galvanic conductivity finished")

class SpinGalvanicConductance(SpinVelocityCorrelation):
    def __init__(self, fermiList=[], currentProjectionList=[], spinProjectionList=[], **kwarg) -> None:
        '''
        Inter band contributions on the spin Galvanic conductance in (1/cell 1/(V/angst.) hbar/2)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinProjectionList=spinProjectionList, **kwarg)

        nProjections=self.projected_spinVelocityCorrelation.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(fermiList),3,3))

        for ifermi, fermi in enumerate(fermiList):
            occupy=self.energies<fermi
            for ivelocity in range(nProjections[0]):
                for ispin in range(nProjections[1]):
                    for ialpha in range(3):
                        for igamma in range(3):
                            projected_sigma[ivelocity,ispin,ifermi,ialpha,igamma]=np.sum(self.projected_spinVelocityCorrelation[ivelocity,ispin,...,ialpha,igamma,:][occupy])
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"Inter-band spin Galvanic conductance finished")

class OrbitalVelocityCorrelation(Orbital,Velocity):
    def __init__(self, velocityProjectionList=[], orbitalProjectionList=[], **kwarg) -> None:
        '''
        Calc. inter-band contributions to the correlation function of
        orbital magnetic moment and velocity.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be Lx, Ly, Lz, corresponding to the orientation of orbital-polarization. 
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        orbitalVelocityCorrelation[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,orbitalProjectionList=orbitalProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse2

        self.projected_orbitalVelocityCorrelation=np.zeros((len(velocityProjectionList)+1,len(orbitalProjectionList)+1)+self.kgrid+(3,3,self.model.norbs-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
            for iorbital, proj_orbital_K in enumerate([self.orbital_K]+self.projected_orbital_K_list):
                for ialpha in range(3):
                    for igamma in range(3):
                        self.projected_orbitalVelocityCorrelation[ivelocity,iorbital,...,ialpha,igamma,:]=-2*np.diagonal(proj_orbital_K[...,igamma,:,:] @ (dEInverse2*proj_velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        
        self.orbitalVelocityCorrelation=self.projected_orbitalVelocityCorrelation[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on inter band contributions on correlation functions <L \dot velocity> finished")

class _OrbitalGalvanicConductivity_LG_DM(DensityMatrixResponse):
    """DC LG response of the selected orbital magnetic-moment matrix."""
    Iodd = True
    Todd = True

    def __init__(self, currentProjectionList=None, orbitalProjectionList=None,
                 responseFermiIndexList=None, finiteTemperatureOccupations=True,
                 fullCovariantDerivative=True, **kwarg):
        currentProjectionList = [] if currentProjectionList is None else currentProjectionList
        orbitalProjectionList = [] if orbitalProjectionList is None else orbitalProjectionList
        super().__init__(electronic_stages=(self.orbital_stage,), velocityProjectionList=currentProjectionList,
                         orbitalProjectionList=orbitalProjectionList, **kwarg)
        observable_vertices = [self.orbital_K] + self.projected_orbital_K_list
        result = _solve_dc_galvanic_lg_density_matrix(
            self.inputs, observable_vertices, responseFermiIndexList,
            finiteTemperatureOccupations, fullCovariantDerivative,
            "mu_B/(V/angstrom)/cell", collision=self.collision
        )
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        self.accept_first_order_result(result)
        _store_dc_galvanic_density_matrix(self, result.kernel, result.projected_channels, result.response_channels, 'mu_B/(V/angstrom)/cell', result.pole_residual)
        if self.verbose==2: showElapsedTime(extraInfo="Length-gauge density-matrix orbital galvanic conductivity finished")

class _OrbitalGalvanicConductivity_VG_DM(DensityMatrixResponse):
    """DC VG response with the contact appropriate to the selected moment."""
    orbital_contact = False
    Iodd = True
    Todd = True

    def __init__(self, currentProjectionList=None, orbitalProjectionList=None,
                 responseFermiIndexList=None, finiteTemperatureOccupations=True,
                 fullCovariantDerivative=True, **kwarg):
        currentProjectionList = [] if currentProjectionList is None else currentProjectionList
        orbitalProjectionList = [] if orbitalProjectionList is None else orbitalProjectionList
        if currentProjectionList:
            raise NotImplementedError("VG orbital galvanic conductivity requires the total drive current")
        if response_requires_full_basis(kwarg):
            raise NotImplementedError("VG orbital galvanic conductivity requires the complete orthogonal model basis")
        kwarg.pop('highOrderVelocity', None)
        super().__init__(electronic_stages=(self.orbital_stage,), velocityProjectionList=[], orbitalProjectionList=orbitalProjectionList,
                         highOrderVelocity=False, orbitalContact=self.orbital_contact, **kwarg)
        observable_vertices = [self.orbital_K] + self.projected_orbital_K_list
        contacts = ([self.electronic.orbitalA_K] + self.electronic.projected_orbitalA_K_list
                    if self.orbital_contact else _zero_vg_observable_contacts(observable_vertices))
        result = _solve_dc_galvanic_vg_density_matrix(
            self.inputs, observable_vertices, contacts,
            responseFermiIndexList, finiteTemperatureOccupations,
            fullCovariantDerivative, "mu_B/(V/angstrom)/cell", collision=self.collision)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        self.accept_first_order_result(result)
        _store_dc_galvanic_density_matrix(self, result.kernel, result.projected_channels, result.response_channels, 'mu_B/(V/angstrom)/cell', result.pole_residual)
        if self.verbose==2: showElapsedTime(extraInfo="Velocity-gauge density-matrix orbital galvanic conductivity finished")

class OrbitalGalvanicConductivity_LG_DM(_OrbitalGalvanicConductivity_LG_DM):
    """DC atom-centered orbital moment response, in mu_B/(V/angstrom)/cell.

    sigma axes are (fermi, field_direction, orbital_direction).
    """
    orbital_stage = Orbital

class OrbitalGalvanicConductivity_VG_DM(_OrbitalGalvanicConductivity_VG_DM):
    """DC VG response of the field-independent atom-centered moment matrix."""
    orbital_stage = Orbital

class OrbitalGalvanicConductivity_BlochSelfRotate_LG_DM(_OrbitalGalvanicConductivity_LG_DM):
    """Electric-field-induced Bloch self-rotation moment from Tr(rho1 m_self).

    sigma axes are (fermi, field_direction, orbital_direction), with units
    mu_B/(V/angstrom)/cell. The unperturbed self-rotation matrix is the
    observable; field-induced corrections to that matrix and Wannier IC
    contributions are outside this approximation.
    """
    orbital_stage = Orbital_BlochSelfRotate

class OrbitalGalvanicConductivity_BlochSelfRotate_VG_DM(_OrbitalGalvanicConductivity_VG_DM):
    """DC VG BSR moment response including its vector-potential contact.

    Uses D_a m_self, including derivatives of observable projections, for
    the normalized vector potential a=e*A/hbar. The analytic derivative uses
    energies, velocities and ordered velocity derivatives at the same k point;
    no displaced meshes or finite-difference step are needed.
    The DC finite part is pole-subtracted; inspect sum_rule_residual under
    mesh refinement to verify the contact cancellation. A complete
    orthogonal basis, total drive, and fullCovariantDerivative=True are required.
    This implements the gauge transform of the BSR observable, not additional
    physical field-induced changes of its definition or Wannier IC terms.
    """
    orbital_stage = Orbital_BlochSelfRotate
    orbital_contact = True

class OrbitalGalvanicConductance(OrbitalVelocityCorrelation):
    def __init__(self, fermiList=[], currentProjectionList=[], orbitalProjectionList=[], **kwarg) -> None:
        '''
        Inter band contributions on the orbital Galvanic conductance in (1/cell 1/(V/angst.) hbar)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, orbitalProjectionList=orbitalProjectionList, **kwarg)

        nProjections=self.projected_orbitalVelocityCorrelation.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(fermiList),3,3))

        for ifermi, fermi in enumerate(fermiList):
            occupy=self.energies<fermi
            for ivelocity in range(nProjections[0]):
                for iorbital in range(nProjections[1]):
                    for ialpha in range(3):
                        for igamma in range(3):
                            projected_sigma[ivelocity,iorbital,ifermi,ialpha,igamma]=np.sum(self.projected_orbitalVelocityCorrelation[ivelocity,iorbital,...,ialpha,igamma,:][occupy])
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"Inter-band orbital Galvanic conductance finished")

class SpinVelocityCorrelation_diag(Spin_diag, Velocity_diag):
    def __init__(self, velocityProjectionList=[], spinProjectionList=[], **kwarg) -> None:
        '''
        Calc. intra-band contributions on the correlation func of spin and velocity \Omega^{alpha,gamma}_{k,m}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be Sx, Sy, Sz, corresponding to the orientation of spin-polarization. 
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        spinVelocityCorrelation_diag[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,spinProjectionList=spinProjectionList, **kwarg)

        self.projected_spinVelocityCorrelation_diag=np.zeros((len(velocityProjectionList)+1,len(spinProjectionList)+1)+self.kgrid+(3,3,self.ibmax+1-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_diag_K]+self.projected_velocity_diag_K_list):
            for ispin, proj_spin_K in enumerate([self.spin_diag_K]+self.projected_spin_diag_K_list):
                for ialpha in range(3):
                    for igamma in range(3):
                        self.projected_spinVelocityCorrelation_diag[ivelocity,ispin,...,ialpha,igamma,:]=(proj_spin_K[...,igamma,:]*proj_velocity_K[...,ialpha,:]).real/self.eta
        
        self.spinVelocityCorrelation_diag=self.projected_spinVelocityCorrelation_diag[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on intra band contributions on correlation functions <Spin><velocity> finished")

class SpinGalvanicConductance_diag(SpinVelocityCorrelation_diag):
    def __init__(self, currentProjectionList=[], spinProjectionList=[], **kwarg) -> None:
        '''
        Intra band contributions on the spin Galvanic conductance in (1/cell 1/(V/angst.) hbar/2)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinProjectionList=spinProjectionList, **kwarg)

        nProjections=self.projected_spinVelocityCorrelation_diag.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3))
        
        for ifermi, weight in enumerate(self.fermiWeight):
            for ivelocity in range(nProjections[0]):
                for ispin in range(nProjections[1]):
                    for ialpha in range(3):
                        for igamma in range(3):
                            projected_sigma[ivelocity,ispin,ifermi,ialpha,igamma]=np.sum(self.projected_spinVelocityCorrelation_diag[ivelocity,ispin,...,ialpha,igamma,:]*weight)
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"Intra-band spin Galvanic conductance finished")

class OrbitalVelocityCorrelation_diag(Orbital_diag, Velocity_diag):
    def __init__(self, velocityProjectionList=[], orbitalProjectionList=[], **kwarg) -> None:
        '''
        Calc. intra-band contributions to the correlation function of
        orbital magnetic moment and velocity.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be Lx, Ly, Lz, corresponding to the orientation of orbital-polarization. 
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        orbitalVelocityCorrelation_diag[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,orbitalProjectionList=orbitalProjectionList, **kwarg)

        self.projected_orbitalVelocityCorrelation_diag=np.zeros((len(velocityProjectionList)+1,len(orbitalProjectionList)+1)+self.kgrid+(3,3,self.ibmax+1-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_diag_K]+self.projected_velocity_diag_K_list):
            for iorbital, proj_orbital_K in enumerate([self.orbital_diag_K]+self.projected_orbital_diag_K_list):
                for ialpha in range(3):
                    for igamma in range(3):
                        self.projected_orbitalVelocityCorrelation_diag[ivelocity,iorbital,...,ialpha,igamma,:]=(proj_orbital_K[...,igamma,:]*proj_velocity_K[...,ialpha,:]).real/self.eta
        
        self.orbitalVelocityCorrelation_diag=self.projected_orbitalVelocityCorrelation_diag[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on intra band contributions on correlation functions <L><velocity> finished")

class OrbitalGalvanicConductance_diag(OrbitalVelocityCorrelation_diag):
    def __init__(self, currentProjectionList=[], orbitalProjectionList=[], **kwarg) -> None:
        '''
        Intra band contributions on the orbital Galvanic conductance in (1/cell 1/(V/angst.) hbar)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, orbitalProjectionList=orbitalProjectionList, **kwarg)

        nProjections=self.projected_orbitalVelocityCorrelation_diag.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3))
        
        for ifermi, weight in enumerate(self.fermiWeight):
            for ivelocity in range(nProjections[0]):
                for iorbital in range(nProjections[1]):
                    for ialpha in range(3):
                        for igamma in range(3):
                            projected_sigma[ivelocity,iorbital,ifermi,ialpha,igamma]=np.sum(self.projected_orbitalVelocityCorrelation_diag[ivelocity,iorbital,...,ialpha,igamma,:]*weight)
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"Intra-band orbital Galvanic conductance finished")

class SpintorqueVelocityCorrelation(SpinExchangeTorque,Velocity):
    def __init__(self, velocityProjectionList=[], spinTorqueProjectionList=[], **kwarg) -> None:
        '''
        Calc. correlation func of spin torque and velocity \Omega^{alpha,gamma}_{k,mn}.
        Only the exchange torque, not the full torque, suitable to fed to LLG dynamics 

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gamma can be x, y, or z, corresponding to the spin magnetic-moment torque orientation.
        k denotes the crystal moment, m, n index the bands.

        Indexing
        --------
        spinBerryCurv[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,
                         spinTorqueProjectionList=spinTorqueProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2

        self.projected_spintorqueVelocityCorrelation=np.zeros((len(velocityProjectionList)+1,len(spinTorqueProjectionList)+1)+self.kgrid+(3,3,self.model.norbs-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
            for ispinTorque, proj_spinTorque_K in enumerate([self.spinTorque_K]+self.projected_spinTorque_K_list):
                for ialpha in range(3):
                    for igamma in range(3):
                        self.projected_spintorqueVelocityCorrelation[ivelocity,ispinTorque,...,ialpha,igamma,:]=-2*np.diagonal(proj_spinTorque_K[...,igamma,:,:] @ (dEInverse2*proj_velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on inter band contributions on correlation function <Torque_Spin velocity> finished")

def _angular_torque_vertices(owner, contact=False):
    """Read physical stage tensors for the selected moment, with both projection orders."""
    kind = owner.torque_stage.angular_kind
    suffix = 'A_K' if contact else '_K'
    return (getattr(owner.electronic, kind+'Torque'+suffix),
            getattr(owner.electronic, 'projected_'+kind+'Torque'+suffix+'_list'),
            getattr(owner.electronic, 'projected'+kind.capitalize()+'_torque'+suffix+'_list'))

def _publish_angular_torque_projection_results(owner):
    """Expose both observable groups without repeating the density-matrix solve.

    The combined observable axis is total, projected torque, then torque of
    projected moment. Named group arrays exclude the total, which remains sigma.
    """
    _, projected_torque, projected_moment_torque = _angular_torque_vertices(owner)
    nt, ns = len(projected_torque), len(projected_moment_torque)
    owner.torque_observable_labels = (('total', 0),)+tuple(
        ('projected_angularTorque', i) for i in range(nt))+tuple(
        ('projectedAngular_torque', i) for i in range(ns))
    for prefix, selection in (('projected_angularTorque', slice(1, 1+nt)),
                              ('projectedAngular_torque', slice(1+nt, 1+nt+ns))):
        for name in ('sigma', 'response'):
            setattr(owner, prefix+'_'+name, getattr(owner, 'projected_'+name)[:, selection])
            setattr(owner, prefix+'_'+name+'_channels', {key: value[:, selection]
                for key, value in getattr(owner, 'projected_'+name+'_channels').items()})
        if hasattr(owner, 'projected_sum_rule_residual'):
            setattr(owner, prefix+'_sum_rule_residual', owner.projected_sum_rule_residual[:, selection])

class _AngularTorqueGalvanicConductivity_LG_DM(DensityMatrixResponse):
    """LG response with two independent, simultaneously available projections.

    angularTorqueProjectionList projects -i[mu,H]; angularProjectionList selects
    mu_P first and measures -i[mu_P,H]. Combined observable axes are ordered
    total, projected torque, torque of projected moment. The separate
    projected_angularTorque_sigma and projectedAngular_torque_sigma arrays exclude
    the total; corresponding response, channel and residual arrays use the
    same grouping. Empty input lists produce empty groups, never a fallback
    to the other projection definition.
    """
    Iodd = True
    Todd = False

    def __init__(self, currentProjectionList=None, angularProjectionList=None, angularTorqueProjectionList=None,
                 responseFermiIndexList=None, finiteTemperatureOccupations=True,
                 fullCovariantDerivative=True, **kwarg):
        currentProjectionList = [] if currentProjectionList is None else currentProjectionList
        angularProjectionList = [] if angularProjectionList is None else angularProjectionList
        angularTorqueProjectionList = [] if angularTorqueProjectionList is None else angularTorqueProjectionList
        kind = self.torque_stage.angular_kind
        super().__init__(electronic_stages=(self.torque_stage,), velocityProjectionList=currentProjectionList,
            **{kind+'ProjectionList': angularProjectionList, kind+'TorqueProjectionList': angularTorqueProjectionList}, **kwarg)
        total, projected_torque, projected_moment_torque = _angular_torque_vertices(self)
        observable_vertices = [total]+projected_torque+projected_moment_torque
        result = _solve_dc_galvanic_lg_density_matrix(
            self.inputs, observable_vertices, responseFermiIndexList,
            finiteTemperatureOccupations, fullCovariantDerivative,
            "mu_B*eV/(V/angstrom)/cell", collision=self.collision
        )
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        self.accept_first_order_result(result)
        _store_dc_galvanic_density_matrix(self, result.kernel, result.projected_channels, result.response_channels, 'mu_B*eV/(V/angstrom)/cell', result.pole_residual)
        _publish_angular_torque_projection_results(self)
        if self.verbose==2: showElapsedTime(extraInfo="Length-gauge density-matrix angular-torque galvanic conductivity finished")

class _AngularTorqueGalvanicConductivity_VG_DM(DensityMatrixResponse):
    """VG response with both projection orders and complete first-order contacts."""
    Iodd = True
    Todd = False

    def __init__(self, currentProjectionList=None, angularProjectionList=None, angularTorqueProjectionList=None,
                 responseFermiIndexList=None, finiteTemperatureOccupations=True,
                 fullCovariantDerivative=True, **kwarg):
        if currentProjectionList:
            raise NotImplementedError('VG angular torque requires the total drive current')
        if response_requires_full_basis(kwarg):
            raise NotImplementedError('VG angular torque requires the complete orthogonal model basis')
        angularProjectionList = [] if angularProjectionList is None else angularProjectionList
        angularTorqueProjectionList = [] if angularTorqueProjectionList is None else angularTorqueProjectionList
        kind = self.torque_stage.angular_kind
        super().__init__(electronic_stages=(self.torque_stage,), velocityProjectionList=[],
            **{kind+'ProjectionList': angularProjectionList, kind+'TorqueProjectionList': angularTorqueProjectionList},
            **self.contact_options, **kwarg)
        total, projected_torque, projected_moment_torque = _angular_torque_vertices(self)
        observables = [total]+projected_torque+projected_moment_torque
        total, projected_torque, projected_moment_torque = _angular_torque_vertices(self, contact=True)
        contacts = [total]+projected_torque+projected_moment_torque
        units = 'mu_B*eV/(V/angstrom)/cell'
        result = _solve_dc_galvanic_vg_density_matrix(self.inputs, observables, contacts,
            responseFermiIndexList, finiteTemperatureOccupations, fullCovariantDerivative,
            units, collision=self.collision)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        self.accept_first_order_result(result)
        _store_dc_galvanic_density_matrix(self, result.kernel, result.projected_channels,
            result.response_channels, units, result.pole_residual)
        _publish_angular_torque_projection_results(self)

class SpintorqueGalvanicConductivity_LG_DM(_AngularTorqueGalvanicConductivity_LG_DM):
    """LG electronic exchange torque in mu_B*eV/(V/angstrom)/cell.

    sigma[fermi, field_direction, torque_direction] responds to the electric
    field. Divide by hbar in eV*fs and apply the electron-to-order reaction
    sign for the magnetic-moment rate used by an LLG equation.
    """
    torque_stage = SpinExchangeTorque

class SpintorqueGalvanicConductivity_VG_DM(_AngularTorqueGalvanicConductivity_VG_DM):
    """VG exchange torque with analytic contacts for both projection orders.

    Includes spin and projector derivatives as well as D_a H_XC.
    """
    torque_stage = SpinExchangeTorque
    contact_options = {'spinExchangeContact': True}

    def __init__(self, *, exchangeVelocity=True, **kwarg):
        if not exchangeVelocity:
            raise ValueError('VG exchange torque requires the complete exchange contact')
        super().__init__(**kwarg)

class SpinSOCTorqueGalvanicConductivity_LG_DM(_AngularTorqueGalvanicConductivity_LG_DM):
    """DC LG response of -i[mu_S,H_SOC] to the electric field.

    sigma[fermi, field_direction, torque_direction] has units
    mu_B*eV/(V/angstrom)/cell. This is the electronic SOC spin-moment torque;
    no electron-to-order reaction sign is applied. The SOC Hamiltonian is
    supplied explicitly by model.soc_R.
    """
    torque_stage = SpinSOCTorque

class SpinSOCTorqueGalvanicConductivity_VG_DM(_AngularTorqueGalvanicConductivity_VG_DM):
    """VG SOC torque with both projection orders and analytic contacts."""
    torque_stage = SpinSOCTorque
    contact_options = {'spinSOCContact': True}

class SpinFluxTorqueGalvanicConductivity_LG_DM(_AngularTorqueGalvanicConductivity_LG_DM):
    """DC LG response of -i[mu_S,H0] and -i[projected_spin,H0].

    angularProjectionList selects spin moments before the commutator. Equal
    source/target projections describe regional spin inflow from H0.
    projected_sigma[drive, observable, fermi, field, spin] includes the
    unprojected observable at index zero; sigma is that total response.
    Units are mu_B*eV/(V/angstrom)/cell, with no LLG reaction sign.
    """
    torque_stage = SpinFluxTorque

class SpinFluxTorqueGalvanicConductivity_VG_DM(_AngularTorqueGalvanicConductivity_VG_DM):
    """VG H0-driven spin torque with both projection orders and analytic contacts."""
    torque_stage = SpinFluxTorque
    contact_options = {'spinFluxContact': True}

class OrbitalSOCTorqueGalvanicConductivity_LG_DM(_AngularTorqueGalvanicConductivity_LG_DM):
    """LG ACA orbital-moment response of -i[mu_L,H_SOC], in mu_B*eV/(V/angstrom)/cell."""
    torque_stage = OrbitalSOCTorque

class OrbitalSOCTorqueGalvanicConductivity_VG_DM(_AngularTorqueGalvanicConductivity_VG_DM):
    """VG ACA SOC orbital torque, with both projection orders and analytic contacts."""
    torque_stage = OrbitalSOCTorque
    contact_options = {'orbitalTorqueContact': True}

class OrbitalFluxTorqueGalvanicConductivity_LG_DM(_AngularTorqueGalvanicConductivity_LG_DM):
    """LG ACA orbital-moment response of -i[mu_L,H0], in mu_B*eV/(V/angstrom)/cell.

    H0 can exert crystal-field torque on orbital moment; this contribution
    includes that torque as well as regional transport, and need not vanish.
    """
    torque_stage = OrbitalFluxTorque

class OrbitalFluxTorqueGalvanicConductivity_VG_DM(_AngularTorqueGalvanicConductivity_VG_DM):
    """VG H0-driven ACA orbital torque, with both projection orders and analytic contacts."""
    torque_stage = OrbitalFluxTorque
    contact_options = {'orbitalTorqueContact': True}

class SpintorqueGalvanicConductance(SpintorqueVelocityCorrelation):
    def __init__(self, currentProjectionList=[], spinTorqueProjectionList=[], **kwarg) -> None:
        '''
        Inter band contributions on the spin torque Galvanic conductance in (1/cell eV /(V/angst.) hbar/2)
        Only the exchange torque, not the full torque, suitable to fed to LLG dynamics 
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinTorqueProjectionList=spinTorqueProjectionList, **kwarg)

        nProjections=self.projected_spintorqueVelocityCorrelation.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3))

        for ifermi, fermi in enumerate(self.fermiList):
            occupy=self.energies<fermi
            for ivelocity in range(nProjections[0]):
                for ispin in range(nProjections[1]):
                    for ialpha in range(3):
                        for igamma in range(3):
                            projected_sigma[ivelocity,ispin,ifermi,ialpha,igamma]=np.sum(self.projected_spintorqueVelocityCorrelation[ivelocity,ispin,...,ialpha,igamma,:][occupy])
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"Inter-band spin Galvanic conductance finished")

class Chern_FHS(Energy):
    def __init__(self, **kwarg):
        '''
        Computes the exact Chern number OF 2D MATERIAL (z-surf) using the Fukui-Hatsugai-Suzuki (FHS) 
        U(N) link variable method.
        
        This algorithm guarantees an exact integer output and converges 
        orders of magnitude faster than the Kubo velocity integration.

        Parameters
        ----------
        - model : System
            the model for computation
        - kgrid : tuple(3)
            kgrid for sampling the BZ
        '''
        # Hijack the Energy class to build the Hamiltonian and diagonalize it over the grid
        super().__init__(**kwarg)
        
        # 1. Identify the occupied manifold
        # Assuming a global gap, we find the number of occupied bands at the Gamma point
        n_occ_Gamma = np.sum(self.energies[0, 0, 0] <= self.model.fermi)
        n_occ = np.sum(self.energies <= self.model.fermi)//np.prod(self.kgrid)
        assert n_occ==n_occ_Gamma, f"The occ. at Gamma {n_occ_Gamma} is inconsistent to occ. on BZ {n_occ}, recheck {self.model.fermi=}"
        self.n_occ=n_occ
        if self.verbose: 
            print(f"\n--- FHS Chern Number Calculation ---")
            print(f"Occupied bands identified: {self.n_occ} out of {self.model.norbs}")
            
        # Extract the occupied eigenvectors over the 2D grid
        # Shape: (Nx, Ny, norbs, n_occ)
        V = self.vectors[:, :, 0, :, :self.n_occ]
        
        # Determine the conjugate transpose, factoring in overlap if non-orthogonal
        if self.model.overlap:
            O_K = self.O_K_AO[:, :, 0, :, :]
            V_dag = np.matmul(V.conj().transpose(0, 1, 3, 2), O_K)
        else:
            V_dag = V.conj().transpose(0, 1, 3, 2)
            
        # 2. Compute U(N) Link Variables
        # Shift vectors to get adjacent k-points. np.roll natively handles the periodic boundary!
        V_shift_x = np.roll(V, shift=-1, axis=0)
        V_shift_y = np.roll(V, shift=-1, axis=1)
        
        # Overlap matrices: M_x = V^\dagger(k) V(k+dx)
        M_x = np.matmul(V_dag, V_shift_x)
        M_y = np.matmul(V_dag, V_shift_y)
        
        # Determinants of the overlap matrices
        det_x = np.linalg.det(M_x)
        det_y = np.linalg.det(M_y)
        
        # Link variables (normalized U(1) phases of the U(N) determinant)
        U_x = det_x / np.abs(det_x)
        U_y = det_y / np.abs(det_y)
        
        # 3. Compute Plaquette Flux (Lattice Berry Curvature)
        U_x_shift_y = np.roll(U_x, shift=-1, axis=1)
        U_y_shift_x = np.roll(U_y, shift=-1, axis=0)
        
        # F_12 = ln( U_1(k) * U_2(k+1) * U_1(k+2)^-1 * U_2(k)^-1 )
        # np.log automatically takes the principal branch, wrapping the phase to (-pi, pi]
        F_12 = np.log(U_x * U_y_shift_x * np.conj(U_x_shift_y) * np.conj(U_y))
        
        # The Chern number is the sum of the fluxes divided by 2*pi*i
        self.chern_number = np.sum(F_12.imag) / (2 * np.pi)
        
        if self.verbose:
            print(f"Computed Chern Number: {self.chern_number:.6f}")
            print(f"Nearest Integer: {int(np.round(self.chern_number))}")
            showElapsedTime(extraInfo="FHS Chern calculation finished")

class Z2_FHS(Energy):
    def __init__(self, **kwarg):
        '''
        Computes the Z2 topological invariant OF 2D MATERIAL (z-surf) using the Spin-Chern / FHS method.
        
        Projects the S_z operator onto the occupied bands, dynamically splits them 
        into '+' and '-' spin sectors, and applies the FHS algorithm to the '+' sector.
        Z2 = C_+ mod 2.

        Parameters
        ----------
        - model : System
            the model for computation
        - kgrid : tuple(3)
            kgrid for sampling the BZ
        '''
        super().__init__(**kwarg)
        
        n_occ_Gamma = np.sum(self.energies[0, 0, 0] <= self.model.fermi)
        n_occ = np.sum(self.energies <= self.model.fermi)//np.prod(self.kgrid)
        assert n_occ==n_occ_Gamma, f"The occ. at Gamma {n_occ_Gamma} is inconsistent to occ. on BZ {n_occ}, recheck {self.model.fermi=}"
        self.n_occ=n_occ
        if self.verbose: 
            print(f"\n--- FHS Z2 (Spin-Chern) Calculation ---")
            print(f"Occupied bands identified: {self.n_occ} out of {self.model.norbs}")
            
        V = self.vectors[:, :, 0, :, :self.n_occ]
        
        if self.model.overlap:
            O_K = self.O_K_AO[:, :, 0, :, :]
            V_dag = np.matmul(V.conj().transpose(0, 1, 3, 2), O_K)
        else:
            V_dag = V.conj().transpose(0, 1, 3, 2)
            
        # 1. Project S_z onto the occupied subspace
        Sz_op = self.model.spin[2] # sigma_z from your Spin class matrices
        
        # P S_z P = V^\dagger S_z V
        Sz_matrix = np.matmul(Sz_op, V)
        if self.model.overlap:
            Sz_matrix = np.matmul(O_K, Sz_matrix)
            
        P_Sz_P = np.matmul(V_dag, Sz_matrix)
        
        # 2. Diagonalize the projected S_z
        # eigh automatically sorts eigenvalues from lowest to highest
        spin_evals, spin_evecs = np.linalg.eigh(P_Sz_P)
        
        # The upper half of the sorted eigenvectors belongs to the "+" spin sector
        half_occ = self.n_occ // 2
        V_plus_coeff = spin_evecs[:, :, :, half_occ:]
        
        # Rotate the occupied tight-binding vectors into the purified "+" spin sector
        V_plus = np.matmul(V, V_plus_coeff)
        
        # 3. Apply FHS logic strictly to the "+" sector
        if self.model.overlap:
            V_plus_dag = np.matmul(V_plus.conj().transpose(0, 1, 3, 2), O_K)
        else:
            V_plus_dag = V_plus.conj().transpose(0, 1, 3, 2)
            
        V_plus_shift_x = np.roll(V_plus, shift=-1, axis=0)
        V_plus_shift_y = np.roll(V_plus, shift=-1, axis=1)
        
        M_x = np.matmul(V_plus_dag, V_plus_shift_x)
        M_y = np.matmul(V_plus_dag, V_plus_shift_y)
        
        det_x = np.linalg.det(M_x)
        det_y = np.linalg.det(M_y)
        
        U_x = det_x / np.abs(det_x)
        U_y = det_y / np.abs(det_y)
        
        U_x_shift_y = np.roll(U_x, shift=-1, axis=1)
        U_y_shift_x = np.roll(U_y, shift=-1, axis=0)
        
        F_12 = np.log(U_x * U_y_shift_x * np.conj(U_x_shift_y) * np.conj(U_y))
        
        self.chern_plus = np.sum(F_12.imag) / (2 * np.pi)
        self.z2_invariant = int(np.round(np.abs(self.chern_plus))) % 2
        
        if self.verbose:
            print(f"Spin-Chern C_+ : {self.chern_plus:.6f}")
            print(f"Z2 Invariant   : {self.z2_invariant}")
            showElapsedTime(extraInfo="FHS Z2 calculation finished")

class RetardedLatticeGreenFunc(Energy):
    def __init__(self, latVecList=None, relevantOrbitalIndexList=None, calcGreen=False, **kwarg) -> None:
        '''
        Compute retarded lattice Green's functions on selected real-space
        lattice vectors and selected local orbitals.

        Indexing:
        self.rG_w_lat[iomega, ilat, iorb, jorb]

        The orbital indexes of the last two axes follow
        self.relevantOrbitalIndexList.
        '''
        super().__init__(**kwarg)
        self.rG_w_lat = None
        self.latVecList = None
        self.relevantOrbitalIndexList = None
        self.orbitalIndexMap = None
        if calcGreen:
            assert latVecList is not None, "latVecList is required when calcGreen=True"
            assert relevantOrbitalIndexList is not None, "relevantOrbitalIndexList is required when calcGreen=True"
            self.calcRetardedLatticeGreenFunc(
                omegaList=self.omegaList,
                latVecList=latVecList,
                relevantOrbitalIndexList=relevantOrbitalIndexList,
            )

    def calcRetardedLatticeGreenFunc(self, omegaList, latVecList, relevantOrbitalIndexList):
        '''
        Calculate G_R(z) = 1/Nk sum_k exp(-i k.R) G_k(z).

        Parameters
        ----------
        omegaList : array-like(complex)
            Complex energy points.
        latVecList : array-like(int), shape (nLat, 3)
            Bravais lattice vectors where the Green's function is needed.
        relevantOrbitalIndexList : array-like(int)
            Local-basis orbital indexes retained in the Green's function.
        '''
        if self.verbose: showElapsedTime("start computations on retarded lattice Green's function")
        omegaList = np.atleast_1d(np.asarray(omegaList, dtype=complex))
        latVecList = np.asarray(latVecList, dtype=int).reshape((-1,3))
        relevantOrbitalIndexList = np.asarray(relevantOrbitalIndexList, dtype=int)

        self.omegaList = omegaList
        self.latVecList = latVecList
        self.relevantOrbitalIndexList = relevantOrbitalIndexList
        self.orbitalIndexMap = np.zeros(self.model.norbs, dtype=int) - 1
        for iorb_reduced, iorb in enumerate(relevantOrbitalIndexList):
            self.orbitalIndexMap[iorb] = iorb_reduced

        nkpt = int(np.prod(self.kgrid))
        nlat = len(latVecList)
        nbands = self.energies.shape[-1]
        norb_reduced = len(relevantOrbitalIndexList)

        energies = self.energies.reshape((nkpt, nbands))
        vectors = self.vectors.reshape((nkpt, self.model.norbs, nbands))
        selected_vectors = vectors[:, relevantOrbitalIndexList, :]

        k_index = np.indices(self.kgrid).reshape((3, nkpt)).T
        k_frac = k_index / np.array(self.kgrid)[None,:]
        lattice_factors = np.exp(-2j*np.pi*(k_frac @ latVecList.T)).T

        selected_vectors_conj = selected_vectors.conj()

        def calc_omega_batch(iomega_start, omega_batch):
            denominator = 1.0/(omega_batch[:, None, None] - energies[None, :, :] + 1j*self.eta)
            G_k = np.einsum(
                'kai,wki,kbi->wkab',
                selected_vectors, denominator, selected_vectors_conj,
                optimize=True,
            )
            rG_batch = np.einsum('lk,wkab->wlab', lattice_factors, G_k, optimize=True) / nkpt
            return iomega_start, rG_batch

        self.rG_w_lat = np.zeros((len(omegaList), nlat, norb_reduced, norb_reduced), dtype=complex)
        omega_batches = [
            (iomega_start, omegaList[iomega_start:iomega_start+self.greenFuncOmegaBatch])
            for iomega_start in range(0, len(omegaList), self.greenFuncOmegaBatch)
        ]
        if self.greenFuncNWorkers == 1 or len(omega_batches) == 1:
            for iomega_start, omega_batch in omega_batches:
                _, rG_batch = calc_omega_batch(iomega_start, omega_batch)
                self.rG_w_lat[iomega_start:iomega_start+len(omega_batch)] = rG_batch
        else:
            with ThreadPoolExecutor(max_workers=self.greenFuncNWorkers) as executor:
                futures = [
                    executor.submit(calc_omega_batch, iomega_start, omega_batch)
                    for iomega_start, omega_batch in omega_batches
                ]
                for future in futures:
                    iomega_start, rG_batch = future.result()
                    self.rG_w_lat[iomega_start:iomega_start+rG_batch.shape[0]] = rG_batch
        
        if self.verbose:
            showElapsedTime(extraInfo=f"retarded lattice Green function on {len(omegaList)} energies, {nlat} lattice vectors finished")
        return self.rG_w_lat

class ExchangeCoupling(RetardedLatticeGreenFunc):
    def __init__(self, exchangeRoutes=None, erange=(-500,500), extraEnergyEmpty=5.0,
                 lowOrder=5, highOrder=20, Jcvg=1e-3,
                 symmetricExchangeSign=-1.0, dmiSign=1.0, **kwarg) -> None:
        '''
        Compute exchange couplings from retarded lattice Green's functions.

        ``exchangeEnergy``, ``Jani``, ``DMI``, and the values copied into the
        route dictionaries are fully normalized magnetic-force energies in eV.
        Output code must not apply an additional factor of one half.

        Parameters
        ----------
        exchangeRoutes : list[dict] | list[list[dict]]
            Routes returned by ``System.classifyExchangeRoutes``. A flat route
            list is also accepted.
        erange : tuple(float, float)
            Hard energy integration window. The upper bound is also clipped by
            each Fermi level.
        extraEnergyEmpty : float
            Extra energy margin below the band bottom.
        lowOrder, highOrder : int
            Gaussian-quadrature orders tested sequentially. The interval is
            ``range(lowOrder, highOrder)``.
        Jcvg : float
            Convergence threshold for all computed couplings. Negative values
            disable convergence checking and use the first quadrature order.
        symmetricExchangeSign, dmiSign : float
            Sign-convention factors. The default symmetric-exchange sign matches
            the historical ``couplingCalc.py`` output for the current route and
            Green-function convention.
        '''
        self.exchangeRoutes = self._flattenExchangeRoutes(exchangeRoutes)
        self.erange = erange
        self.extraEnergyEmpty = extraEnergyEmpty
        self.lowOrder = lowOrder
        self.highOrder = highOrder
        self.Jcvg = Jcvg
        self.symmetricExchangeSign = symmetricExchangeSign
        self.dmiSign = dmiSign
        super().__init__(**kwarg)
        assert self.model.ispin == 2, "ExchangeCoupling requires a spinful model with ispin=2"
        assert len(self.exchangeRoutes) > 0, "no exchange route is provided"
        self.calcExchangeCoupling()

    @staticmethod
    def _flattenExchangeRoutes(exchangeRoutes):
        if exchangeRoutes is None:
            return []
        if len(exchangeRoutes) == 0:
            return []
        if isinstance(exchangeRoutes[0], dict):
            return list(exchangeRoutes)
        routes = []
        for group in exchangeRoutes:
            routes.extend(group)
        return routes

    def _prepareRouteMaps(self):
        overlat_set = {(0,0,0)}
        relevant_orbitals = set()
        half_norbs = self.model.norbs//2
        for route in self.exchangeRoutes:
            overlat = tuple(int(x) for x in route["overlat"])
            overlat_set.add(overlat)
            overlat_set.add(tuple(-np.array(overlat, dtype=int)))
            for iorb in route["source_orbitals"]:
                relevant_orbitals.add(int(iorb))
                relevant_orbitals.add(int(iorb) + half_norbs)
            for iorb in route["target_orbitals"]:
                relevant_orbitals.add(int(iorb))
                relevant_orbitals.add(int(iorb) + half_norbs)

        self.overlatList = np.array(sorted(overlat_set), dtype=int)
        self.overlatIndex = {tuple(overlat): ilat for ilat, overlat in enumerate(self.overlatList)}
        self.relevantOrbitalIndexList = np.array(sorted(relevant_orbitals), dtype=int)

    def _routeOrbitalIndexes(self, route):
        half_norbs = self.model.norbs//2
        source_up = [int(iorb) for iorb in route["source_orbitals"]]
        target_up = [int(iorb) for iorb in route["target_orbitals"]]
        source_full = source_up + [iorb + half_norbs for iorb in source_up]
        target_full = target_up + [iorb + half_norbs for iorb in target_up]
        source_reduced = list(self.orbitalIndexMap[source_full])
        target_reduced = list(self.orbitalIndexMap[target_full])
        return source_reduced, target_reduced

    @staticmethod
    def _contractExchange(p_source, p_target, G_forward, G_backward):
        G_forward_blocks = PauliBlock(G_forward)
        G_backward_blocks = PauliBlock(G_backward)
        AMat = np.zeros((4,4), dtype=complex)
        for i in range(4):
            route1 = p_source @ G_forward_blocks[i]
            for j in range(4):
                route2 = p_target @ G_backward_blocks[j]
                AMat[i,j] = np.sum(route1 * route2.T)
        return AMat

    @staticmethod
    def _spinVertex(p_mat, iaxis):
        norb = p_mat.shape[0]
        vertex = np.zeros((2*norb, 2*norb), dtype=complex)
        if iaxis == 0:
            vertex[:norb, norb:] = p_mat
            vertex[norb:, :norb] = p_mat
        elif iaxis == 1:
            vertex[:norb, norb:] = -1j * p_mat
            vertex[norb:, :norb] = 1j * p_mat
        elif iaxis == 2:
            vertex[:norb, :norb] = p_mat
            vertex[norb:, norb:] = -p_mat
        else:
            raise ValueError(f"spin vertex axis should be 0, 1, or 2, got {iaxis}")
        return vertex

    @classmethod
    def _contractSpinRotation(cls, p_source, iaxis_source, p_target, iaxis_target,
                              G_forward, G_backward):
        vertex_source = cls._spinVertex(p_source, iaxis_source)
        vertex_target = cls._spinVertex(p_target, iaxis_target)
        return np.trace(vertex_source @ G_forward @ vertex_target @ G_backward)

    def calcExchangeCoupling(self):
        self._prepareRouteMaps()
        nfermi = len(self.fermiList)
        nroutes = len(self.exchangeRoutes)
        self.exchangeEnergy = np.zeros((nfermi, nroutes))
        self.Jani = np.zeros((nfermi, nroutes, 2, 2))
        self.DMI = np.zeros((nfermi, nroutes, 3))
        self.rotationCurvature = np.zeros((nfermi, nroutes, 3, 3))
        self.exchangeInfo = []

        bandEnergyMin = float(self.energies.min())
        emin = max(self.erange[0], bandEnergyMin) - self.extraEnergyEmpty

        for ifermi, fermi in enumerate(self.fermiList):
            emax = min(self.erange[1], float(fermi))
            J_old = np.zeros((nroutes, 3, 3))
            for iteration_num, gaussianOrder in enumerate(range(self.lowOrder, self.highOrder)):
                points, weights = p_roots(gaussianOrder)
                omega_path, domega_path = semiCirle(emin, emax, points)
                self.calcRetardedLatticeGreenFunc(
                    omegaList=omega_path,
                    latVecList=self.overlatList,
                    relevantOrbitalIndexList=self.relevantOrbitalIndexList,
                )

                J_new = np.zeros((nroutes, 3, 3))
                for iroute, route in enumerate(self.exchangeRoutes):
                    overlat = np.array(route["overlat"], dtype=int)
                    i_overlat_forward = self.overlatIndex[tuple(overlat)]
                    i_overlat_backward = self.overlatIndex[tuple(-overlat)]
                    source_reduced, target_reduced = self._routeOrbitalIndexes(route)
                    p_source = self.model.getAtomicSelfEnergy(route["source_orbitals"])
                    p_target = self.model.getAtomicSelfEnergy(route["target_orbitals"])

                    J = 0j
                    J_a_xx = 0j
                    J_a_yy = 0j
                    J_a_xy = 0j
                    C_xy = 0j
                    C_yx = 0j
                    DMI = np.zeros(3, dtype=complex)
                    for iomega in range(len(omega_path)):
                        G_forward = self.rG_w_lat[iomega, i_overlat_forward][source_reduced,:][:,target_reduced]
                        G_backward = self.rG_w_lat[iomega, i_overlat_backward][target_reduced,:][:,source_reduced]
                        AMat = self._contractExchange(p_source, p_target, G_forward, G_backward)

                        weight = domega_path[iomega] * weights[iomega]
                        J += 2*(AMat[0,0]-AMat[1,1]-AMat[2,2]-AMat[3,3]) * weight
                        J_a_xx += 2*(AMat[1,1]+AMat[1,1]) * weight
                        J_a_yy += 2*(AMat[2,2]+AMat[2,2]) * weight
                        J_a_xy += 2*(AMat[1,2]+AMat[2,1]) * weight
                        DMI[0] += 2j*(AMat[0,1]-AMat[1,0]) * weight
                        DMI[1] += 2j*(AMat[0,2]-AMat[2,0]) * weight
                        C_xy += self._contractSpinRotation(p_source, 0, p_target, 1, G_forward, G_backward) * weight
                        C_yx += self._contractSpinRotation(p_source, 1, p_target, 0, G_forward, G_backward) * weight

                    # The accumulated contractions carry a historical factor
                    # of two.  The magnetic-force theorem contributes 1/pi.
                    exchange_prefactor = 1.0/(2*np.pi)
                    self.exchangeEnergy[ifermi, iroute] = self.symmetricExchangeSign * J.imag * exchange_prefactor
                    self.Jani[ifermi, iroute] = self.symmetricExchangeSign * np.array([[J_a_xx, J_a_xy], [J_a_xy, J_a_yy]]).imag * exchange_prefactor
                    self.rotationCurvature[ifermi, iroute, 0, 1] = self.symmetricExchangeSign * C_xy.imag * exchange_prefactor
                    self.rotationCurvature[ifermi, iroute, 1, 0] = self.symmetricExchangeSign * C_yx.imag * exchange_prefactor
                    DMI[2] = 0.5 * (C_yx - C_xy)
                    self.DMI[ifermi, iroute] = self.dmiSign * DMI.imag * exchange_prefactor
                    J_new[iroute] = np.diag([self.exchangeEnergy[ifermi, iroute]]*3)

                dJ_max = float(np.max(np.abs(J_new - J_old)))
                if self.verbose:
                    print(f"ExchangeCoupling fermi {fermi:.6f}, Gaussian order {gaussianOrder}, max dJ {dJ_max:.6e}")
                if self.Jcvg < 0:
                    break
                if iteration_num > 0 and dJ_max < self.Jcvg:
                    break
                J_old = J_new.copy()

            self.exchangeInfo.append({
                "fermi": float(fermi),
                "emin": float(emin),
                "emax": float(emax),
                "gaussianOrder": int(gaussianOrder),
                "max_dJ": dJ_max,
            })

        for iroute, route in enumerate(self.exchangeRoutes):
            route["exchangeEnergy"] = self.exchangeEnergy[:, iroute]
            route["Jani"] = self.Jani[:, iroute]
            route["DMI"] = self.DMI[:, iroute]
        return self.exchangeEnergy
