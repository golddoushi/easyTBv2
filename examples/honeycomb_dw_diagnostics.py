"""Diagnose DW cancellation and higher displacement orders in the gap benchmark.

python examples/honeycomb_dw_diagnostics.py --qgrid 6 6 1 --beta 100
python examples/honeycomb_dw_diagnostics.py --beta-bond-factors 1 0 0

Uses common Gaussian samples at several coupling scales. Subtracting each
sample's second-order gap expansion isolates the higher-order remainder with
much less noise than subtracting two independently averaged results. This
diagnostic requires nondegenerate, isolated finite-supercell band edges.
"""
from phonon import PhononBath

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigvalsh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calculator import Energy, ModelElectronSelfEnergy
from calculator.model_self_energy import uniform_qmesh
from examples.honeycomb_four_band import build_model, write_hr
from elph import SSHModelEPC


def gaussian_mean_quadratic_hamiltonian(epc, temperature):
    """Independent Cartesian-covariance contraction for the Einstein bath.

    This deliberately uses orbital positions and ionic variances directly,
    without calling the second-displacement vertex used by the DW calculator.
    """
    model = epc.electronic_model
    variance = epc.phonon_model.meanSquareDisplacements(temperature)
    atoms = epc.orbital_atoms
    bonds = ((model.iRvec @ model.lattice)[:, :, None, None]
             + model.rList[None, :, None, :] - model.rList[None, :, :, None])
    lengths = np.linalg.norm(bonds, axis=1)
    units = np.divide(bonds, lengths[:, None], out=np.zeros_like(bonds),
                      where=lengths[:, None] > 1e-12)
    orbital_variance = variance[atoms].T
    stretch_variance = np.sum(units**2 * (orbital_variance[None, :, :, None]
                                         + orbital_variance[None, :, None, :]), axis=1)
    same_ion = (model.iRvec == 0).all(axis=1)[:, None, None] & (atoms[:, None] == atoms[None, :])[None]
    stretch_variance[same_ion] = 0
    return 0.5*epc.beta_R*stretch_variance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='tmp/honeycomb_dw_diagnostics')
    parser.add_argument('--qgrid', nargs=3, type=int, default=[6, 6, 1])
    parser.add_argument('--scales', nargs='+', type=float, default=[0.1, 0.05, 0.025])
    parser.add_argument('--beta', type=float, default=100)
    parser.add_argument('--beta-bond-factors', nargs=3, type=float, default=[1, 1, 1],
                        help='relative beta on home, (-1,0,0), and (0,-1,0) A-to-B bonds')
    parser.add_argument('--temperature', type=float, default=300)
    parser.add_argument('--pairs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=5740)
    args = parser.parse_args()
    if args.pairs < 2 or not np.all(np.isfinite(args.scales)) or min(args.scales) <= 0:
        parser.error('pairs must be at least two, and scales must be positive and finite')
    if not np.isfinite(args.beta) or not np.all(np.isfinite(args.beta_bond_factors)):
        parser.error('beta and bond factors must be finite')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    grid, scale0 = tuple(args.qgrid), max(args.scales)
    with contextlib.redirect_stdout(io.StringIO()):
        model, epc = build_model(out/'model', alpha=0.5*scale0)
        beta = np.zeros_like(epc.alpha_R[:, :2, :2])
        for R, factor in zip(((0, 0, 0), (-1, 0, 0), (0, -1, 0)), args.beta_bond_factors):
            value = args.beta*scale0**2*factor
            beta[model.iRvecIndex[R], 0, 1] = value
            beta[model.iRvecIndex[tuple(-np.array(R))], 1, 0] = value
        write_hr(out/'model'/'quadratic_hr.dat', model.iRvec, beta, 'quadratic diagnostic coefficients')
        epc = SSHModelEPC.from_files(model, epc.phonon_model, betaR='quadratic')
        sc_epc = epc.supercell(np.diag(grid))
        sc = sc_epc.electronic_model
    ks = uniform_qmesh(grid)
    se = ModelElectronSelfEnergy(
        epc,
        ks,
        adiabatic=True,
        delta=1e-07,
        phononBath=PhononBath(qgrid=grid, temperature=args.temperature),
    )
    vb = np.unravel_index(np.argmax(se.energies[:, :2]), (len(ks), 2))
    cb0 = np.unravel_index(np.argmin(se.energies[:, 2:]), (len(ks), 2))
    cb = cb0[0], cb0[1]+2
    fan0 = float((se.fan_migdal[cb]-se.fan_migdal[vb]).real)
    dw0 = float(se.debye_waller[cb]-se.debye_waller[vb])
    mean_R = gaussian_mean_quadratic_hamiltonian(epc, args.temperature)
    mean_K = np.einsum('kr,rij->kij', np.exp(2j*np.pi*(ks @ model.iRvec.T)), mean_R)
    covariance_dw = se.vectors.conj().swapaxes(-1, -2) @ mean_K @ se.vectors

    solver = Energy(model=sc, epcModel=sc_epc, kgrid=(1, 1, 1))
    energies = solver.energies.reshape(-1)
    vectors = solver.vectors.reshape(sc.norbs, sc.norbs)
    occupied = sc.norbs//2
    edges = np.array([occupied-1, occupied])
    edge_vectors = vectors[:, edges]
    gap0 = float(energies[occupied]-energies[occupied-1])
    denominators = energies[edges][None, :]-energies[:, None]
    denominators[edges, np.arange(2)] = np.inf
    separation = float(np.min(np.abs(denominators)))
    if gap0 <= 0 or separation < 1e-8:
        raise ValueError('This diagnostic needs an insulating gap and isolated, nondegenerate edges')

    def diagonal_gap(matrix):
        shifts = np.einsum('ic,ic->c', edge_vectors.conj(), matrix @ edge_vectors).real
        return shifts[1]-shifts[0]

    def mixing_gap(matrix):
        coupling = vectors.conj().T @ (matrix @ edge_vectors)
        shifts = np.sum(np.abs(coupling)**2/denominators, axis=0).real
        return shifts[1]-shifts[0]

    h0 = solver.ham_K.reshape(sc.norbs, sc.norbs)
    displacements = sc_epc.phonon_model.sampleDisplacements(args.temperature, nSamples=args.pairs, seed=args.seed)
    residual = np.zeros((len(args.scales), args.pairs))
    beta_residual = np.zeros_like(residual)
    beta_fourth = np.zeros(args.pairs)
    exact_shifts = np.zeros_like(residual)
    for sample, u in enumerate(displacements):
        plus = sc_epc.deltaHamiltonian(u).sum(axis=0)
        minus = sc_epc.deltaHamiltonian(-u).sum(axis=0)
        h1, h2 = (plus-minus)/2, (plus+minus)/2
        dw_sample = diagonal_gap(h2)
        second_order_sample = dw_sample+mixing_gap(h1)
        beta_fourth[sample] = mixing_gap(h2)
        for i, scale in enumerate(args.scales):
            f = scale/scale0
            pair_shift = 0.0
            for sign in (-1, 1):
                e = eigvalsh(h0+sign*f*h1+f*f*h2, subset_by_index=tuple(edges))
                pair_shift += 0.5*(e[1]-e[0]-gap0)
            exact_shifts[i, sample] = pair_shift
            residual[i, sample] = pair_shift-f*f*second_order_sample
            e_beta = eigvalsh(h0+f*f*h2, subset_by_index=tuple(edges))
            beta_residual[i, sample] = e_beta[1]-e_beta[0]-gap0-f*f*dw_sample
        if (sample+1) % 100 == 0:
            print(f'Processed {sample+1}/{args.pairs} common Gaussian pairs', flush=True)

    def stderr(values):
        return float(values.std(ddof=1)/np.sqrt(len(values)))

    rows = []
    for i, scale in enumerate(args.scales):
        f = scale/scale0
        leading = f*f*(fan0+dw0)
        rows.append(dict(scale=scale, fan_gap_eV=f*f*fan0, dw_gap_eV=f*f*dw0,
                         leading_gap_shift_eV=leading,
                         raw_sampled_gap_shift_eV=float(exact_shifts[i].mean()),
                         raw_standard_error_eV=stderr(exact_shifts[i]),
                         beyond_second_order_eV=float(residual[i].mean()),
                         remainder_standard_error_eV=stderr(residual[i]),
                         controlled_sampled_gap_shift_eV=leading+float(residual[i].mean()),
                         remainder_divided_by_scale4=float(residual[i].mean()/scale**4),
                         beta_only_remainder_eV=float(beta_residual[i].mean()),
                         beta_only_remainder_divided_by_scale4=float(beta_residual[i].mean()/scale**4)))
    result = dict(qgrid=grid, temperature_K=args.temperature, beta=args.beta,
                  beta_bond_factors=args.beta_bond_factors, pairs=args.pairs, seed=args.seed,
                  bare_gap_eV=gap0, nearest_edge_level_separation_eV=separation,
                  vbm_k=ks[vb[0]].tolist(), cbm_k=ks[cb[0]].tolist(),
                  max_dw_matrix_error_eV=float(np.max(np.abs(covariance_dw-se.debye_waller_matrix))),
                  max_mean_quadratic_hopping_eV=float(np.max(np.abs(mean_R))),
                  beta_only_fourth_order_coefficient=float(beta_fourth.mean()/scale0**4),
                  beta_only_fourth_order_coefficient_stderr=stderr(beta_fourth)/scale0**4,
                  results=rows)
    (out/'diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez(out/'samples.npz', scales=args.scales, exact_gap_shifts=exact_shifts,
             beyond_second_order=residual, beta_only_remainder=beta_residual, beta_fourth_order=beta_fourth)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
