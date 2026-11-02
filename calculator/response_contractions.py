"""Nonlinear density-matrix contractions with explicit inputs and returned results.

No electronic preparation or mutation of Grid calculators occurs here.
"""
import numpy as np
from .response_inputs import ResponseInputs
from .response_results import SecondOrderResponseResult
from .density_matrix import (AC_CHANNEL_NAMES as AC_DM_CHANNEL_NAMES,
    AC_CHANNEL_DESCRIPTIONS as AC_DM_CHANNEL_DESCRIPTIONS,
    LengthGaugeSecondOrderDMKernel, VelocityGaugeSecondOrderDMKernel)

def _normalize_response_index_list(index_list, axis_length):
    if index_list is None:
        return []
    return [int(index) for index in index_list if 0 <= int(index) < axis_length]


def _contract_second_order_observable(rho, observable):
    """Contract one density matrix with every component of an observable."""
    return np.einsum("...mn,...cmn->...c", rho, observable.swapaxes(-1, -2), optimize=True)


def _sum_density_matrix_channels(channels):
    """Sum channel arrays in place to avoid large intermediate responses."""
    values = iter(channels.values())
    total = np.array(next(values), copy=True)
    for value in values:
        np.add(total, value, out=total)
    return total


def solve_bpve_lg(inputs: ResponseInputs, *, collision,
    responseOmegaIndexList=(), responseFermiIndexList=None, fullCovariantDerivative=False, finiteTemperatureOccupations=False, velocityProjectionList=()):
    result = SecondOrderResponseResult()
    inputs.electronic.require('velocity_K', 'r_K')

    result.channel_names = AC_DM_CHANNEL_NAMES
    result.channel_descriptions = dict(AC_DM_CHANNEL_DESCRIPTIONS)
    response_shape = (len(inputs.fermiList), len(inputs.omegaList), 3, 3, 3)
    result.channels = {
        name: np.zeros(response_shape, dtype=complex) for name in result.channel_names
    }

    responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(inputs.omegaList))
    if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
        responseFermiIndexList = range(len(inputs.fermiList))
    responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(inputs.fermiList))
    responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
    responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
    nproj = len(velocityProjectionList) + 1
    projected_shape = ((nproj, nproj, nproj) + inputs.kgrid
                       + (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
    result.integrand_channels = {
        name: np.zeros(projected_shape, dtype=complex) for name in result.channel_names
    }
    result.value = np.zeros(response_shape, dtype=complex)
    result.integrands = np.zeros(projected_shape, dtype=complex)

    kernel = LengthGaugeSecondOrderDMKernel(
        inputs, fullCovariantDerivative, finiteTemperatureOccupations,
        collision=collision
    )
    result.kernel = kernel
    result.channel_descriptions = dict(kernel.channel_descriptions)
    velocity_observable = -inputs.velocity_K  # physical electron current
    for ifermi, fermi in enumerate(inputs.fermiList):
        fermi_state = kernel.fermi_state(ifermi, fermi)
        for iomega, Omega in enumerate(inputs.omegaList):
            frequency_state = fermi_state.at_frequency(Omega)
            irecord_fermi = responseFermiRecord.get(ifermi)
            irecord_omega = responseOmegaRecord.get(iomega)
            for a in range(3):
                for b in range(3):
                    block = frequency_state.block(a, b)
                    sigma_total_k = _contract_second_order_observable(block.total_second_order, velocity_observable)
                    result.value[ifermi, iomega, a, b, :] = sigma_total_k.mean(axis=(0, 1, 2))
                    if irecord_fermi is not None and irecord_omega is not None:
                        result.integrands[0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, :] = sigma_total_k
                    for name in kernel.active_rho2_channels:
                        sigma_k = _contract_second_order_observable(block.rho2[name], velocity_observable)
                        result.channels[name][ifermi, iomega, a, b, :] = (
                            np.sum(sigma_k, axis=tuple(range(len(inputs.kgrid)))) / np.prod(inputs.kgrid)
                        )
                        if irecord_fermi is not None and irecord_omega is not None:
                            result.integrand_channels[name][
                                0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, :
                            ] = sigma_k

    result.projected_value = result.value[None, None, None, ...]
    result.projected_channels = {
        name: sigma[None, None, None, ...] for name, sigma in result.channels.items()
    }
    result.fermi_indices = tuple(responseFermiIndexList)
    result.omega_indices = tuple(responseOmegaIndexList)
    return result

def solve_bpve_vg(inputs: ResponseInputs, *, collision,
    responseOmegaIndexList=(), responseFermiIndexList=None, fullCovariantDerivative=False, finiteTemperatureOccupations=False, velocityProjectionList=()):
    result = SecondOrderResponseResult()
    inputs.electronic.require('velocity_K', 'r_K', 'w_K', 'u_K')

    result.channel_names = AC_DM_CHANNEL_NAMES
    result.channel_descriptions = dict(AC_DM_CHANNEL_DESCRIPTIONS)
    result.channel_descriptions["II"] += ", including current contact vertices"
    response_shape = (len(inputs.fermiList), len(inputs.omegaList), 3, 3, 3)
    result.channels = {
        name: np.zeros(response_shape, dtype=complex) for name in result.channel_names
    }

    responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(inputs.omegaList))
    if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
        responseFermiIndexList = range(len(inputs.fermiList))
    responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(inputs.fermiList))
    responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
    responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
    nproj = len(velocityProjectionList) + 1
    projected_shape = ((nproj, nproj, nproj) + inputs.kgrid
                       + (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
    result.integrand_channels = {
        name: np.zeros(projected_shape, dtype=complex) for name in result.channel_names
    }

    kernel = VelocityGaugeSecondOrderDMKernel(
        inputs, fullCovariantDerivative, finiteTemperatureOccupations,
        collision=collision
    )
    result.kernel = kernel
    velocity_observable = -inputs.velocity_K  # j(A)=-v-A*w-A*A*u/2
    for ifermi, fermi in enumerate(inputs.fermiList):
        fermi_state = kernel.fermi_state(ifermi, fermi)
        relaxation_response_k = {
            name: np.zeros(inputs.kgrid + (3, 3, 3), dtype=complex) for name in result.channel_names
        }
        if fullCovariantDerivative:
            for a in range(3):
                for b in range(3):
                    rho2_relaxation = fermi_state.relaxation(a, b)
                    for name in result.channel_names:
                        relaxation_response_k[name][..., a, b, :] = _contract_second_order_observable(
                            rho2_relaxation[name], velocity_observable
                        )
                    del rho2_relaxation

        for iomega, Omega in enumerate(inputs.omegaList):
            frequency_state = fermi_state.at_frequency(Omega)
            irecord_fermi = responseFermiRecord.get(ifermi)
            irecord_omega = responseOmegaRecord.get(iomega)
            for a in range(3):
                for b in range(3):
                    block = frequency_state.block(a, b)
                    response_k = {
                        name: relaxation_response_k[name][..., a, b, :].copy()
                        for name in result.channel_names
                    }
                    for name in kernel.active_rho2_channels:
                        response_k[name] += _contract_second_order_observable(
                            block.rho2[name], velocity_observable
                        )
                    response_k["EE"] -= (
                        _contract_second_order_observable(block.rho1_minus["E"], inputs.w_K[..., :, b, :, :])
                        + _contract_second_order_observable(block.rho1_plus["E"], inputs.w_K[..., :, a, :, :])
                    )
                    u_cab = 0.5 * (inputs.u_K[..., :, a, b, :, :] + inputs.u_K[..., :, b, a, :, :])
                    response_k["II"] -= np.einsum(
                        "...n,...cn->...c", fermi_state.f,
                        np.diagonal(u_cab, axis1=-2, axis2=-1), optimize=True
                    )
                    if fullCovariantDerivative:
                        response_k["II"] -= (
                            _contract_second_order_observable(block.rho1_minus["I"], inputs.w_K[..., :, b, :, :])
                            + _contract_second_order_observable(block.rho1_plus["I"], inputs.w_K[..., :, a, :, :])
                        )
                    for name in result.channel_names:
                        sigma_k = response_k[name] / Omega**2
                        result.channels[name][ifermi, iomega, a, b, :] = (
                            np.sum(sigma_k, axis=tuple(range(len(inputs.kgrid)))) / np.prod(inputs.kgrid)
                        )
                        if irecord_fermi is not None and irecord_omega is not None:
                            result.integrand_channels[name][
                                0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, :
                            ] = sigma_k

    result.value = _sum_density_matrix_channels(result.channels)
    result.integrands = _sum_density_matrix_channels(result.integrand_channels)
    result.projected_value = result.value[None, None, None, ...]
    result.projected_channels = {
        name: sigma[None, None, None, ...] for name, sigma in result.channels.items()
    }
    result.fermi_indices = tuple(responseFermiIndexList)
    result.omega_indices = tuple(responseOmegaIndexList)
    return result

def solve_ife_lg(inputs: ResponseInputs, *, collision,
    responseOmegaIndexList=(), responseFermiIndexList=None, fullCovariantDerivative=False, ife_type="total"):
    result = SecondOrderResponseResult()
    inputs.electronic.require('velocity_K', 'r_K')
    if ife_type in ("total", "spin"):
        inputs.electronic.require("spin_K")
    if ife_type in ("total", "orbital"):
        inputs.electronic.require("orbital_K")
    if fullCovariantDerivative and inputs.kBT <= 0:
        raise ValueError("fullCovariantDerivative requires kBT > 0")

    nbasis = inputs.energies.shape[-1]
    M_K = np.zeros(inputs.kgrid + (3, nbasis, nbasis), dtype=complex)
    if ife_type in ["total", "spin"]: M_K += inputs.spin_K
    if ife_type in ["total", "orbital"]: M_K += inputs.orbital_K

    response_shape = (len(inputs.fermiList), len(inputs.omegaList), 3, 3, 3)
    result.channel_names = AC_DM_CHANNEL_NAMES
    result.channel_descriptions = dict(AC_DM_CHANNEL_DESCRIPTIONS)
    result.channels = {name: np.zeros(response_shape, dtype=complex) for name in result.channel_names}

    responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(inputs.omegaList))
    if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
        responseFermiIndexList = range(len(inputs.fermiList))
    responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(inputs.fermiList))
    responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
    responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
    projected_shape = ((1, 1, 1) + inputs.kgrid +
                       (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
    result.integrand_channels = {
        name: np.zeros(projected_shape, dtype=complex) for name in result.channel_names
    }
    result.value = np.zeros(response_shape, dtype=complex)
    result.integrands = np.zeros(projected_shape, dtype=complex)

    kernel = LengthGaugeSecondOrderDMKernel(inputs, fullCovariantDerivative, collision=collision)
    result.kernel = kernel
    result.channel_descriptions = dict(kernel.channel_descriptions)
    for ifermi, fermi in enumerate(inputs.fermiList):
        fermi_state = kernel.fermi_state(ifermi, fermi)
        for iomega, Omega in enumerate(inputs.omegaList):
            irecord_fermi = responseFermiRecord.get(ifermi)
            irecord_omega = responseOmegaRecord.get(iomega)
            frequency_state = fermi_state.at_frequency(Omega)
            for a in range(3):
                for b in range(3):
                    block = frequency_state.block(a, b)
                    chi_total_k = _contract_second_order_observable(block.total_second_order, M_K)
                    result.value[ifermi, iomega, a, b, :] = chi_total_k.mean(axis=(0, 1, 2))
                    if irecord_fermi is not None and irecord_omega is not None:
                        result.integrands[0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, :] = chi_total_k
                    for name in kernel.active_rho2_channels:
                        chi_k = _contract_second_order_observable(block.rho2[name], M_K)
                        result.channels[name][ifermi, iomega, a, b, :] = (
                            np.sum(chi_k, axis=tuple(range(len(inputs.kgrid)))) / np.prod(inputs.kgrid)
                        )
                        if irecord_fermi is not None and irecord_omega is not None:
                            result.integrand_channels[name][
                                0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, :
                            ] = chi_k

    result.projected_value = result.value[None, None, None, ...]
    result.projected_channels = {
        name: response[None, None, None, ...] for name, response in result.channels.items()
    }
    result.fermi_indices = tuple(responseFermiIndexList)
    result.omega_indices = tuple(responseOmegaIndexList)
    return result

def solve_ife_vg(inputs: ResponseInputs, *, collision,
    responseOmegaIndexList=(), responseFermiIndexList=None, fullCovariantDerivative=False, ife_type="total"):
    result = SecondOrderResponseResult()
    inputs.electronic.require('velocity_K', 'r_K')
    if ife_type in ("total", "spin"):
        inputs.electronic.require("spin_K")
    if ife_type in ("total", "orbital"):
        inputs.electronic.require("orbital_K")
    if fullCovariantDerivative and inputs.kBT <= 0:
        raise ValueError("fullCovariantDerivative requires kBT > 0")

    nbasis = inputs.energies.shape[-1]
    M_K = np.zeros(inputs.kgrid + (3, nbasis, nbasis), dtype=complex)
    if ife_type in ["total", "spin"]: M_K += inputs.spin_K
    if ife_type in ["total", "orbital"]: M_K += inputs.orbital_K

    response_shape = (len(inputs.fermiList), len(inputs.omegaList), 3, 3, 3)
    result.channel_names = AC_DM_CHANNEL_NAMES
    result.channel_descriptions = dict(AC_DM_CHANNEL_DESCRIPTIONS)
    result.channel_descriptions["EE"] += " and the contact vertex"
    result.channels = {name: np.zeros(response_shape, dtype=complex) for name in result.channel_names}

    responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(inputs.omegaList))
    if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
        responseFermiIndexList = range(len(inputs.fermiList))
    responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(inputs.fermiList))
    responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
    responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
    projected_shape = ((1, 1, 1) + inputs.kgrid +
                       (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
    result.integrand_channels = {
        name: np.zeros(projected_shape, dtype=complex) for name in result.channel_names
    }

    kernel = VelocityGaugeSecondOrderDMKernel(inputs, fullCovariantDerivative, collision=collision)
    result.kernel = kernel
    for ifermi, fermi in enumerate(inputs.fermiList):
        fermi_state = kernel.fermi_state(ifermi, fermi)

        relaxation_response_k = {
            name: np.zeros(inputs.kgrid + (3, 3, 3), dtype=complex) for name in result.channel_names
        }
        if fullCovariantDerivative:
            for a in range(3):
                for b in range(3):
                    rho2_relaxation = fermi_state.relaxation(a, b)
                    for name in result.channel_names:
                        relaxation_response_k[name][..., a, b, :] = _contract_second_order_observable(
                            rho2_relaxation[name], M_K
                        )
                    del rho2_relaxation

        for iomega, Omega in enumerate(inputs.omegaList):
            irecord_fermi = responseFermiRecord.get(ifermi)
            irecord_omega = responseOmegaRecord.get(iomega)
            frequency_state = fermi_state.at_frequency(Omega)
            for a in range(3):
                for b in range(3):
                    block = frequency_state.block(a, b)
                    for name in kernel.active_rho2_channels:
                        sigma_k = (
                            _contract_second_order_observable(block.rho2[name], M_K)
                            + relaxation_response_k[name][..., a, b, :]
                        ) / Omega**2
                        result.channels[name][ifermi, iomega, a, b, :] = (
                            np.sum(sigma_k, axis=tuple(range(len(inputs.kgrid)))) / np.prod(inputs.kgrid)
                        )
                        if irecord_fermi is not None and irecord_omega is not None:
                            result.integrand_channels[name][
                                0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, :
                            ] = sigma_k

    result.value = _sum_density_matrix_channels(result.channels)
    result.integrands = _sum_density_matrix_channels(result.integrand_channels)
    result.projected_value = result.value[None, None, None, ...]
    result.projected_channels = {
        name: response[None, None, None, ...] for name, response in result.channels.items()
    }
    result.fermi_indices = tuple(responseFermiIndexList)
    result.omega_indices = tuple(responseOmegaIndexList)
    return result

def solve_torque_lg(inputs: ResponseInputs, *, collision,
    responseOmegaIndexList=(), responseFermiIndexList=None, fullCovariantDerivative=False, finiteTemperatureOccupations=False, ife_type="total", torqueProjectionList=()):
    result = SecondOrderResponseResult()
    inputs.electronic.require('velocity_K', 'r_K')
    if ife_type in ("total", "spin"):
        inputs.electronic.require('spinTorque_K', 'spinTorqueE_K')
    if ife_type in ("total", "orbital"):
        inputs.electronic.require('orbitalTorque_K', 'orbitalTorqueE_K')
    if fullCovariantDerivative and inputs.kBT <= 0:
        raise ValueError("fullCovariantDerivative requires kBT > 0")

    nbasis = inputs.energies.shape[-1]
    # Slot 0 is the full torque; following slots are projected torque operators.
    proj_T_K = np.zeros((len(torqueProjectionList)+1,) + inputs.kgrid + (3, nbasis, nbasis), dtype=complex)
    if ife_type in ["total", "spin"]: proj_T_K[0] += inputs.spinTorque_K
    if ife_type in ["total", "orbital"]: proj_T_K[0] += inputs.orbitalTorque_K
    for iproj in range(len(torqueProjectionList)):
        if ife_type in ["total", "spin"]: proj_T_K[iproj+1] += inputs.projected_spinTorque_K_list[iproj]
        if ife_type in ["total", "orbital"]: proj_T_K[iproj+1] += inputs.projected_orbitalTorque_K_list[iproj]

    # H_LG(t) = H0 + E_a(t) r_a, hence the full torque contains
    # T_c(E) = -i[mu_c,H_LG] = T_c(0) - i E_a [mu_c,r_a].
    proj_torqueE_K = np.zeros((len(torqueProjectionList)+1,) + inputs.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
    if ife_type in ["total", "spin"]:
        proj_torqueE_K[0] += inputs.spinTorqueE_K
    if ife_type in ["total", "orbital"]:
        proj_torqueE_K[0] += inputs.orbitalTorqueE_K
    for iproj in range(len(torqueProjectionList)):
        if ife_type in ["total", "spin"]: proj_torqueE_K[iproj+1] += inputs.projected_spinTorqueE_K_list[iproj]
        if ife_type in ["total", "orbital"]: proj_torqueE_K[iproj+1] += inputs.projected_orbitalTorqueE_K_list[iproj]

    response_shape = (1, 1, len(torqueProjectionList)+1,
                      len(inputs.fermiList), len(inputs.omegaList), 3, 3, 3)
    result.channel_names = AC_DM_CHANNEL_NAMES
    result.channel_descriptions = {
        "II": "intraband channel, including torqueE*rho1_I",
        "EI": "second-order interband from first-order intraband",
        "IE": "second-order intraband from first-order interband",
        "EE": "interband channel, including torqueE*rho1_E",
    }
    result.projected_channels = {
        name: np.zeros(response_shape, dtype=complex) for name in result.channel_names
    }
    responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(inputs.omegaList))
    if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
        responseFermiIndexList = range(len(inputs.fermiList))
    responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(inputs.fermiList))
    responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
    responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
    projected_response_shape = ((1, 1, len(torqueProjectionList)+1) + inputs.kgrid +
                                (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
    result.integrand_channels = {
        name: np.zeros(projected_response_shape, dtype=complex) for name in result.channel_names
    }
    result.projected_value = np.zeros(response_shape, dtype=complex)
    result.integrands = np.zeros(projected_response_shape, dtype=complex)

    kernel = LengthGaugeSecondOrderDMKernel(
        inputs, fullCovariantDerivative, finiteTemperatureOccupations,
        collision=collision
    )
    result.kernel = kernel
    result.channel_descriptions = dict(kernel.channel_descriptions)
    result.channel_descriptions["II"] += ", including torqueE*rho1_I"
    result.channel_descriptions["EE"] += ", including torqueE*rho1_E"
    for ifermi, fermi in enumerate(inputs.fermiList):
        fermi_state = kernel.fermi_state(ifermi, fermi)
        for iomega, Omega in enumerate(inputs.omegaList):
            irecord_omega = responseOmegaRecord.get(iomega)
            frequency_state = fermi_state.at_frequency(Omega)
            for a in range(3):
                for b in range(3):
                    block = frequency_state.block(a, b)
                    for iproj_T, T_K in enumerate(proj_T_K):
                        total_k = _contract_second_order_observable(block.total_second_order, T_K)
                        response_k = {
                            name: np.zeros(inputs.kgrid + (3,), dtype=complex)
                            for name in result.channel_names
                        }
                        for name in kernel.active_rho2_channels:
                            response_k[name] = _contract_second_order_observable(block.rho2[name], T_K)
                        contact_e = (
                            _contract_second_order_observable(
                                block.rho1_minus["E"], proj_torqueE_K[iproj_T, ..., :, b, :, :]
                            )
                            + _contract_second_order_observable(
                                block.rho1_plus["E"], proj_torqueE_K[iproj_T, ..., :, a, :, :]
                            )
                        )
                        response_k["EE"] += contact_e
                        total_k += contact_e
                        if fullCovariantDerivative:
                            contact_i = (
                                _contract_second_order_observable(
                                    block.rho1_minus["I"], proj_torqueE_K[iproj_T, ..., :, b, :, :]
                                )
                                + _contract_second_order_observable(
                                    block.rho1_plus["I"], proj_torqueE_K[iproj_T, ..., :, a, :, :]
                                )
                            )
                            response_k["II"] += contact_i
                            total_k += contact_i
                        irecord_fermi = responseFermiRecord.get(ifermi)
                        result.projected_value[0, 0, iproj_T, ifermi, iomega, a, b, :] = total_k.mean(axis=(0, 1, 2))
                        if irecord_fermi is not None and irecord_omega is not None:
                            result.integrands[0, 0, iproj_T, :, :, :, irecord_fermi, irecord_omega, a, b, :] = total_k
                        for name in result.channel_names:
                            result.projected_channels[name][
                                0, 0, iproj_T, ifermi, iomega, a, b, :
                            ] = np.sum(
                                response_k[name], axis=tuple(range(len(inputs.kgrid)))
                            ) / np.prod(inputs.kgrid)
                            if irecord_fermi is not None and irecord_omega is not None:
                                result.integrand_channels[name][
                                    0, 0, iproj_T, :, :, :, irecord_fermi, irecord_omega, a, b, :
                                ] = response_k[name]

    result.channels = {
        name: response[0, 0, 0] for name, response in result.projected_channels.items()
    }
    result.value = result.projected_value[0, 0, 0]
    result.fermi_indices = tuple(responseFermiIndexList)
    result.omega_indices = tuple(responseOmegaIndexList)
    return result

def solve_torque_vg(inputs: ResponseInputs, *, collision,
    responseOmegaIndexList=(), responseFermiIndexList=None, fullCovariantDerivative=False, finiteTemperatureOccupations=False, ife_type="total", torqueProjectionList=()):
    result = SecondOrderResponseResult()
    inputs.electronic.require('velocity_K', 'r_K')
    if ife_type in ("total", "spin"):
        inputs.electronic.require('spinTorque_K', 'spinTorqueA_K', 'spinTorqueAA_K')
    if ife_type in ("total", "orbital"):
        inputs.electronic.require('orbitalTorque_K', 'orbitalTorqueA_K', 'orbitalTorqueAA_K')
    if fullCovariantDerivative and inputs.kBT <= 0:
        raise ValueError("fullCovariantDerivative requires kBT > 0")

    nbasis = inputs.energies.shape[-1]
    nproj = len(torqueProjectionList) + 1
    proj_T_K = np.zeros((nproj,) + inputs.kgrid + (3, nbasis, nbasis), dtype=complex)
    proj_TA_K = np.zeros((nproj,) + inputs.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
    proj_TAA_K = np.zeros((nproj,) + inputs.kgrid + (3, 3, 3, nbasis, nbasis), dtype=complex)
    if ife_type in ["total", "spin"]:
        proj_T_K[0] += inputs.spinTorque_K
        proj_TA_K[0] += inputs.spinTorqueA_K
        proj_TAA_K[0] += inputs.spinTorqueAA_K
    if ife_type in ["total", "orbital"]:
        proj_T_K[0] += inputs.orbitalTorque_K
        proj_TA_K[0] += inputs.orbitalTorqueA_K
        proj_TAA_K[0] += inputs.orbitalTorqueAA_K
    for iproj in range(len(torqueProjectionList)):
        if ife_type in ["total", "spin"]:
            proj_T_K[iproj+1] += inputs.projected_spinTorque_K_list[iproj]
            proj_TA_K[iproj+1] += inputs.projected_spinTorqueA_K_list[iproj]
            proj_TAA_K[iproj+1] += inputs.projected_spinTorqueAA_K_list[iproj]
        if ife_type in ["total", "orbital"]:
            proj_T_K[iproj+1] += inputs.projected_orbitalTorque_K_list[iproj]
            proj_TA_K[iproj+1] += inputs.projected_orbitalTorqueA_K_list[iproj]
            proj_TAA_K[iproj+1] += inputs.projected_orbitalTorqueAA_K_list[iproj]

    response_shape = (1, 1, nproj, len(inputs.fermiList), len(inputs.omegaList), 3, 3, 3)
    result.channel_names = AC_DM_CHANNEL_NAMES
    result.channel_descriptions = {
        "II": "intraband channel, including T1*rho1_I and T2*rho0",
        "EI": "second-order interband from first-order intraband",
        "IE": "second-order intraband from first-order interband",
        "EE": "interband channel, including T1*rho1_E and the contact Hamiltonian vertex",
    }
    result.projected_channels = {
        name: np.zeros(response_shape, dtype=complex) for name in result.channel_names
    }
    responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(inputs.omegaList))
    if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
        responseFermiIndexList = range(len(inputs.fermiList))
    responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(inputs.fermiList))
    responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
    responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
    projected_response_shape = ((1, 1, nproj) + inputs.kgrid +
                                (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
    result.integrand_channels = {
        name: np.zeros(projected_response_shape, dtype=complex) for name in result.channel_names
    }

    kernel = VelocityGaugeSecondOrderDMKernel(
        inputs, fullCovariantDerivative, finiteTemperatureOccupations,
        collision=collision
    )
    result.kernel = kernel
    for ifermi, fermi in enumerate(inputs.fermiList):
        fermi_state = kernel.fermi_state(ifermi, fermi)

        relaxation_response_k = {
            name: np.zeros((nproj,) + inputs.kgrid + (3, 3, 3), dtype=complex)
            for name in result.channel_names
        }
        if fullCovariantDerivative:
            for a in range(3):
                for b in range(3):
                    rho2_relaxation = fermi_state.relaxation(a, b)
                    for name in result.channel_names:
                        for iproj_T in range(nproj):
                            relaxation_response_k[name][iproj_T, ..., a, b, :] = (
                                _contract_second_order_observable(
                                    rho2_relaxation[name], proj_T_K[iproj_T]
                                )
                            )
                    del rho2_relaxation

        for iomega, Omega in enumerate(inputs.omegaList):
            irecord_fermi = responseFermiRecord.get(ifermi)
            irecord_omega = responseOmegaRecord.get(iomega)
            frequency_state = fermi_state.at_frequency(Omega)
            for a in range(3):
                for b in range(3):
                    block = frequency_state.block(a, b)
                    for iproj_T in range(nproj):
                        response_k = {
                            name: relaxation_response_k[name][iproj_T, ..., a, b, :].copy()
                            for name in result.channel_names
                        }
                        for name in kernel.active_rho2_channels:
                            response_k[name] += _contract_second_order_observable(
                                block.rho2[name], proj_T_K[iproj_T]
                            )
                        response_k["EE"] += (
                            _contract_second_order_observable(
                                block.rho1_minus["E"], proj_TA_K[iproj_T, ..., :, b, :, :]
                            )
                            + _contract_second_order_observable(
                                block.rho1_plus["E"], proj_TA_K[iproj_T, ..., :, a, :, :]
                            )
                        )
                        T2_ab = 0.5 * (
                            proj_TAA_K[iproj_T, ..., :, a, b, :, :]
                            + proj_TAA_K[iproj_T, ..., :, b, a, :, :]
                        )
                        response_k["II"] += np.einsum(
                            "...n,...cn->...c", fermi_state.f,
                            np.diagonal(T2_ab, axis1=-2, axis2=-1), optimize=True
                        )
                        if fullCovariantDerivative:
                            response_k["II"] += (
                                _contract_second_order_observable(
                                    block.rho1_minus["I"], proj_TA_K[iproj_T, ..., :, b, :, :]
                                )
                                + _contract_second_order_observable(
                                    block.rho1_plus["I"], proj_TA_K[iproj_T, ..., :, a, :, :]
                                )
                            )
                        for name in result.channel_names:
                            sigma_k = response_k[name] / Omega**2
                            result.projected_channels[name][
                                0, 0, iproj_T, ifermi, iomega, a, b, :
                            ] = np.sum(
                                sigma_k, axis=tuple(range(len(inputs.kgrid)))
                            ) / np.prod(inputs.kgrid)
                            if irecord_fermi is not None and irecord_omega is not None:
                                result.integrand_channels[name][
                                    0, 0, iproj_T, :, :, :, irecord_fermi, irecord_omega, a, b, :
                                ] = sigma_k

    result.projected_value = _sum_density_matrix_channels(result.projected_channels)
    result.integrands = _sum_density_matrix_channels(result.integrand_channels)
    result.channels = {
        name: response[0, 0, 0] for name, response in result.projected_channels.items()
    }
    result.value = _sum_density_matrix_channels(result.channels)
    result.fermi_indices = tuple(responseFermiIndexList)
    result.omega_indices = tuple(responseOmegaIndexList)
    return result


def _torque_fd_response(sample, *, fermiList, omegaList, kBT, eta, step=1e-3, record=None):
    """Independent central differences in a common Wannier basis (Cartesian k)."""
    from scipy.special import expit
    if not all(np.isfinite(x) and x > 0 for x in (kBT, eta, step)):
        raise ValueError('kBT, eta and step must be finite and positive')
    step = float(step)
    cache = {}

    def point(q):
        key = tuple(q)
        if key not in cache:
            h, a, t = map(np.asarray, sample(np.asarray(q)*step))
            if h.shape[-1] != h.shape[-2] or a.shape != h.shape[:-2]+(3,)+h.shape[-2:] or t.shape != a.shape:
                raise ValueError('Expected H (...,n,n) and A,T (...,3,n,n)')
            for matrix in (h, a, t):
                if not np.all(np.isfinite(matrix)) or np.max(np.abs(matrix-matrix.conj().swapaxes(-1,-2))) > 1e-8*max(1.,np.max(np.abs(matrix))):
                    raise ValueError(f'H, A and T must be finite Hermitian matrices; residual={np.max(np.abs(matrix-matrix.conj().swapaxes(-1,-2)))} scale={np.max(np.abs(matrix))}')
            e, u = np.linalg.eigh(h)
            cache[key] = e, u, a, t
        return cache[key]

    def solve(x, q, energy):
        e, u, _, _ = point(q)
        ud = u.conj().swapaxes(-1, -2)
        return u @ ((ud @ x @ u)/(e[..., :, None]-e[..., None, :]-energy-1j*eta)) @ ud

    def derivative(fun, q, axis):
        shift = np.eye(3, dtype=int)[axis]
        x = fun(q)
        a = point(q)[2][..., axis, :, :]
        return (fun(q+shift)-fun(q-shift))/(2*step)-1j*(a@x-x@a)

    origin = np.zeros(3, dtype=int)
    torque = point(origin)[3]
    value = np.empty((len(fermiList),len(omegaList),3,3,3), complex)
    for im, mu in enumerate(fermiList):
        def equilibrium(q):
            e, u, _, _ = point(q)
            return (u*expit((mu-e)/kBT)[...,None,:]) @ u.conj().swapaxes(-1,-2)
        for iw, omega in enumerate(omegaList):
            first = {}
            def rho1(q, axis, sign):
                key = (tuple(q), axis, sign)
                if key not in first:
                    first[key] = solve(-1j*derivative(equilibrium,q,axis),q,sign*omega)
                return first[key]
            for a in range(3):
                for b in range(3):
                    source = -1j*(derivative(lambda q: rho1(q,a,-1),origin,b)
                                  + derivative(lambda q: rho1(q,b,1),origin,a))
                    rho2 = solve(source,origin,0.)
                    integrand = np.einsum('...mn,...cnm->...c',rho2,torque)
                    value[im,iw,a,b] = integrand.reshape(-1,3).mean(axis=0)
                    if record is not None:
                        record(im, iw, a, b, integrand)
    return value

def _fd_wannier_sampler(prepare_inputs):
    """Adapt prepare_inputs(cartesian_offset) -> complete ResponseInputs.

    The factory must prepare spin exchange torque at each displaced mesh and
    convert Cartesian offsets to fractional kOffset using lattice/(2*pi).
    Uses only energies, eigenvectors, AA_R_W and the exchange torque vertex.
    """
    def sample(offset):
        inputs = prepare_inputs(offset)
        s = inputs.states
        if s.model.overlap or s.skip_low or s.ibmin:
            raise NotImplementedError('Requires a complete orthogonal Wannier basis')
        u = s.vectors
        ud = u.conj().swapaxes(-1,-2)
        h = (u*s.energies[...,None,:]) @ ud
        t = u[...,None,:,:] @ inputs.spinTorque_K @ ud[...,None,:,:]
        ar = s.model.AA_R_W*s.phaseFactorR[:,None,None,None]
        mesh = np.zeros(tuple(s.kgrid)+ar.shape[1:],complex)
        for r, value in zip(s.model.iRvec, ar):
            mesh[tuple(np.asarray(r)%s.kgrid)] += value
        a = np.fft.ifftn(mesh,axes=(0,1,2))*np.prod(s.kgrid)
        return h, (a+a.conj().swapaxes(-1,-2))/2, t
    return sample

def solve_torque_fd(inputs: ResponseInputs, *, collision, displaced_inputs,
                    finiteDifferenceStep=1e-3, responseOmegaIndexList=(),
                    responseFermiIndexList=None):
    """CRTA spin exchange IFT, returned through the standard result contract.

    displaced_inputs(offset) prepares inputs at Cartesian momentum offsets.
    Electronic preparation is owned by the Grid boundary. No analytic
    derivatives enter the FD recursion. I/E channel separation is not provided.
    """
    eta = collision.requireConstant('IFT finite differences')
    oi = _normalize_response_index_list(responseOmegaIndexList, len(inputs.omegaList))
    fi = _normalize_response_index_list(
        range(len(inputs.fermiList)) if responseFermiIndexList is None and oi else responseFermiIndexList,
        len(inputs.fermiList))
    result = SecondOrderResponseResult()
    result.omega_indices, result.fermi_indices = tuple(oi), tuple(fi)
    result.integrands = np.zeros((1,1,1)+tuple(inputs.kgrid)+(len(fi),len(oi),3,3,3),complex)
    def record(im, iw, a, b, integrand):
        if im in fi and iw in oi:
            result.integrands[0,0,0,...,fi.index(im),oi.index(iw),a,b,:] = integrand
    result.value = _torque_fd_response(_fd_wannier_sampler(displaced_inputs),
        fermiList=inputs.fermiList, omegaList=inputs.omegaList, kBT=inputs.kBT,
        eta=eta, step=finiteDifferenceStep, record=record)
    result.projected_value = result.value[None,None,None,...]
    return result
