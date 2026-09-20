"""Generate and exercise a four-band honeycomb SSH/Rashba/exchange model.

From the repository root:
    python examples/honeycomb_four_band.py --responses
Outputs go to tmp/honeycomb_four_band by default. No external material data
are required. The response calculation uses CRTA; model EPC is provided for
subsequent collision-operator work, not silently converted into linewidths.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from system import System_tb
from elph import SSHModelEPC
from phonon import EinsteinPhonon

def write_hr(path, iRvec, values, title):
    """Write dense hr-format blocks, with unit degeneracies and 1-based indices."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = values.shape[-1]
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"{title}\n{n}\n{len(iRvec)}\n")
        for start in range(0, len(iRvec), 15):
            stream.write(" ".join(["1"] * min(15, len(iRvec)-start)) + "\n")
        for R, block in zip(iRvec, values):
            for j in range(n):
                for i in range(n):
                    value = complex(block[i, j])
                    stream.write(f"{R[0]:3d} {R[1]:3d} {R[2]:3d} {i+1:3d} {j+1:3d} "
                                 f"{value.real:.16e} {value.imag:.16e}\n")

def build_model(directory, hopping=(1.0, 0.9, 1.1), delta=0.45, rashba=0.2,
                exchange=0.35, direction=(0.3, 0.4, np.sqrt(0.75)), alpha=0.5,
                bondLength=1.0, mass=12.0, phononEnergy=0.02):
    """Write inputs and return (System_tb, SSHModelEPC). Energy/length units: eV/Angstrom.

    H_hop=-t, H_staggered=(Delta,-Delta), H_XC=-J*n.sigma. Only scalar
    nearest-neighbor hopping is displaced; Rashba and exchange stay fixed.
    Slight bond anisotropy and tilted n reduce accidental response zeros.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lattice = np.array([[np.sqrt(3)/2, 1.5, 0], [-np.sqrt(3)/2, 1.5, 0], [0, 0, 10.0]])
    lattice *= bondLength
    positions = np.array([[0, 0, 0], [0, bondLength, 0]])
    with (directory / "POSCAR").open("w", encoding="utf-8") as stream:
        stream.write("Four-band honeycomb (one s orbital per atom)\n1.0\n")
        for row in lattice:
            stream.write(" ".join(f"{x:.16e}" for x in row) + "\n")
        stream.write("A B\n1 1\nCartesian\n")
        for row in positions:
            stream.write(" ".join(f"{x:.16e}" for x in row) + "\n")

    iRvec = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]])
    lookup = {tuple(R): ir for ir, R in enumerate(iRvec)}
    scalar = np.zeros((len(iRvec), 2, 2))
    lambdas, alphas = np.zeros_like(scalar), np.zeros_like(scalar)
    scalar[0] = np.diag([delta, -delta])
    for R, t in zip(((0, 0, 0), (-1, 0, 0), (0, -1, 0)), hopping):
        ir, reverse = lookup[R], lookup[tuple(-np.array(R))]
        scalar[ir, 0, 1] = scalar[reverse, 1, 0] = -t
        lambdas[ir, 0, 1] = lambdas[reverse, 1, 0] = rashba
        alphas[ir, 0, 1] = alphas[reverse, 1, 0] = alpha
    ham = np.zeros((len(iRvec), 4, 4))
    ham[:, :2, :2] = ham[:, 2:, 2:] = scalar
    write_hr(directory / "honeycomb_hr.dat", iRvec, ham, "Spin-independent H=-t plus staggered Delta; eV")
    write_hr(directory / "rashba_hr.dat", iRvec, lambdas, "Scalar Rashba lambda; spatial indices; eV")
    write_hr(directory / "epc_hr.dat", iRvec, alphas, "SSH alpha=dH/dr for H=-t; spatial indices; eV/Angstrom")

    model = System_tb(str(directory), hamR="honeycomb", projections={"A": [0], "B": [0]}, ispin=2)
    model.verbose = False
    n = np.asarray(direction, dtype=float)
    if n.shape != (3,) or not np.all(np.isfinite(n)) or np.linalg.norm(n) == 0:
        raise ValueError("Exchange direction must be a finite nonzero three-vector")
    model.addExchangeSplitting([(0, (0,)), (1, (0,))], -exchange*n/np.linalg.norm(n))
    model.addRashbaSOC()
    phonons = EinsteinPhonon(np.full(2, mass), phononEnergy, directions=(0, 1))
    epc_model = SSHModelEPC.from_files(model, phonons)
    return model, epc_model


def response_arrays(model, grid=18):
    """Nonzero clean-model responses and LG/VG comparisons at finite CRTA eta."""
    from calculator import (AnormalousHallConductance_LG_DM, CollisionSpecification,
                            BulkPhotovoltaicEffect_LG_DM, BulkPhotovoltaicEffect_VG_DM,
                            InverseFaradayEffect_LG_DM, InverseFaradayEffect_VG_DM,
                            InverseFaradayTorque_LG_DM, InverseFaradayTorque_VG_DM)
    args = dict(model=model, epcModel=epc, kgrid=(grid, grid, 1), eta=0.05, kBT=0.025,
                collisionSpecification=CollisionSpecification("crta", constantLinewidth=.05),
                fermiList=np.array([-0.8, 0.0]), omegaList=np.array([0.5, 1.0, 1.5, 2.0]),
                fullCovariantDerivative=True, verbose=0)
    arrays = {"fermi_eV": args["fermiList"], "photon_energy_eV": args["omegaList"]}
    arrays["sigma_dc"] = AnormalousHallConductance_LG_DM(**args).sigma
    for gauge, bpve, ife, ift in (
            ("LG", BulkPhotovoltaicEffect_LG_DM, InverseFaradayEffect_LG_DM, InverseFaradayTorque_LG_DM),
            ("VG", BulkPhotovoltaicEffect_VG_DM, InverseFaradayEffect_VG_DM, InverseFaradayTorque_VG_DM)):
        arrays[f"BPVE_{gauge}"] = bpve(**args).projected_sigma[0, 0, 0]
        arrays[f"IFE_{gauge}"] = ife(ife_type="spin", **args).chi_IFE
        arrays[f"IFT_{gauge}"] = ift(ife_type="spin", **args).chi_IFT
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tmp/honeycomb_four_band")
    parser.add_argument("--responses", action="store_true")
    parser.add_argument("--grid", type=int, default=18)
    args = parser.parse_args()
    output = Path(args.output)
    model, epc = build_model(output)
    k, q = np.array([0.17, 0.23, 0]), np.array([1/3, 0, 0])
    g = epc.matrixElements(k, q)
    supercell_epc = epc.supercell(np.diag([3, 3, 1]))
    supercell = supercell_epc.electronic_model
    displacements = supercell_epc.phonon_model.sampleDisplacements(300, seed=2026)
    snapshot = supercell_epc.makeDisplacedModel(displacements)
    arrays = dict(k=k, q=q, g_eV=g, phonon_energy_eV=epc.phonon_model.modes(q)[0],
                  displacements_A=displacements, snapshot_ham_R=snapshot.ham_R,
                  snapshot_iRvec=snapshot.iRvec, snapshot_lattice=snapshot.lattice,
                  snapshot_positions_A=snapshot.posinfo['pos ions (cart.)'])
    summary = dict(nbands=model.norbs, phonon_modes=g.shape[0],
                   max_abs_g_eV=float(np.max(np.abs(g))), supercell_bands=snapshot.norbs,
                   rms_displacement_A=float(np.sqrt(np.mean(displacements**2))))
    if args.responses:
        arrays.update(response_arrays(model, args.grid))
        # Hall means the antisymmetric part, not the symmetric anisotropic conductivity.
        dc = arrays["sigma_dc"]
        summary["max_abs_AHC_S_per_m"] = float(np.max(np.abs((dc[..., 0, 1]-dc[..., 1, 0])/2)))
        for name in ("BPVE", "IFE", "IFT"):
            lg, vg = arrays[f"{name}_LG"], arrays[f"{name}_VG"]
            summary[f"max_abs_{name}"] = float(np.max(np.abs(lg)))
            summary[f"relative_LG_VG_error_{name}"] = float(np.max(np.abs(lg-vg)) / max(np.max(np.abs(lg)), 1e-30))
            if name in ("IFE", "IFT"):
                summary[f"max_abs_helicity_odd_{name}"] = float(np.max(np.abs((lg[..., 0, 1, :]-lg[..., 1, 0, :])/(2j))))
    np.savez(output / "model_results.npz", **arrays)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Inputs and outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
