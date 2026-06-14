import copy
import numpy as np

from system import System_tb
from calculator import Energy, ExchangeCoupling


mainDir = "./mcalc/1D-chain-3orb/"
fermi = 0.0
theta_list = (0.02, 0.01, 0.005)
primitive_kgrid = (28, 1, 1)
supercell_kgrid = (28, 1, 1)
supercell_matrix = np.diag([10, 1, 1])
projections = {
    "B": (0,),
    "N": (0,),
    "H": (0,),
}


def build_primitive_model():
    model = System_tb(
        mainDir=mainDir,
        cutoff=-0.002,
        poscar="POSCAR",
        projections=projections,
        ispin=2,
    )
    model.verbose = False
    return model


def occupied_energy(model):
    calc = Energy(model=model, kgrid=supercell_kgrid, verbose=0)
    return float(np.sum(calc.energies[calc.energies < fermi]) / np.prod(supercell_kgrid))


def occupied_energy_with_kgrid(model, kgrid):
    calc = Energy(model=model, kgrid=kgrid, verbose=0)
    return float(np.sum(calc.energies[calc.energies < fermi]) / np.prod(kgrid))


def rotate_supercell_model(reference_model, rotations):
    model = copy.deepcopy(reference_model)
    for atom_id, theta, phi in rotations:
        model.rotateExchangeField([(atom_id, (0,))], theta, phi)
    return model


def direction_to_angles(direction):
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
    phi = np.arctan2(direction[1], direction[0])
    return theta, phi


def rotate_atom_exchange_from_z(model, atom_id, direction):
    theta, phi = direction_to_angles(direction)
    model.rotateExchangeField([(atom_id, (0,))], theta, phi)


def magnetic_atom_ids(model):
    return [
        atom_id for atom_id, atom_info in enumerate(model.atomicOrbInfoList)
        if atom_info["label"] == "H" and atom_id % 3 == 2
    ]


def directed_model_from_z(reference_model, reference_direction, updates):
    model = copy.deepcopy(reference_model)
    update_atom_ids = {atom_id for atom_id, _ in updates}
    for atom_id in magnetic_atom_ids(model):
        if atom_id not in update_atom_ids:
            rotate_atom_exchange_from_z(model, atom_id, reference_direction)
    for atom_id, direction in updates:
        rotate_atom_exchange_from_z(model, atom_id, direction)
    return model


def mixed_second_derivative(reference_model, atom_i, atom_j, theta, phi_i, phi_j, e00):
    ei = occupied_energy(rotate_supercell_model(reference_model, [(atom_i, theta, phi_i)]))
    ej = occupied_energy(rotate_supercell_model(reference_model, [(atom_j, theta, phi_j)]))
    eij = occupied_energy(rotate_supercell_model(reference_model, [(atom_i, theta, phi_i), (atom_j, theta, phi_j)]))
    return (eij - ei - ej + e00) / theta**2


def central_mixed_second_derivative_from_directions(reference_model, atom_i, atom_j, ref_dir, dir_i, dir_j, theta):
    ref_dir = np.asarray(ref_dir, dtype=float)
    dir_i = np.asarray(dir_i, dtype=float)
    dir_j = np.asarray(dir_j, dtype=float)

    def tilted(sign_i, sign_j):
        ni = np.cos(theta) * ref_dir + sign_i * np.sin(theta) * dir_i
        nj = np.cos(theta) * ref_dir + sign_j * np.sin(theta) * dir_j
        return occupied_energy(directed_model_from_z(reference_model, ref_dir, [(atom_i, ni), (atom_j, nj)]))

    e_pp = tilted( 1,  1)
    e_pm = tilted( 1, -1)
    e_mp = tilted(-1,  1)
    e_mm = tilted(-1, -1)
    return (e_pp - e_pm - e_mp + e_mm) / (4 * theta**2)


def central_mixed_second_derivative(reference_model, atom_i, atom_j, theta, phi_i, phi_j):
    dir_i = np.array([np.cos(phi_i), np.sin(phi_i), 0.0])
    dir_j = np.array([np.cos(phi_j), np.sin(phi_j), 0.0])
    return central_mixed_second_derivative_from_directions(
        reference_model, atom_i, atom_j, np.array([0.0, 0.0, 1.0]), dir_i, dir_j, theta
    )


def exchange_for_reference(direction):
    model = build_primitive_model()
    rotate_atom_exchange_from_z(model, 2, direction)
    routes_ref = model.classifyExchangeRoutes(
        sourceAtoms=[2],
        pbc=(0,),
        maxRange=5,
        calRange=5,
        verbose=False,
    )
    return ExchangeCoupling(
        model=model,
        kgrid=primitive_kgrid,
        fermiList=np.array([fermi]),
        eta=0.0001,
        exchangeRoutes=routes_ref,
        symmetricExchangeSign=1,
        lowOrder=60,
        highOrder=80,
        Jcvg=-1e-4,
        verbose=0,
    )


primitive_model = build_primitive_model()
routes = primitive_model.classifyExchangeRoutes(
    sourceAtoms=[2],
    pbc=(0,),
    maxRange=5,
    calRange=5,
    verbose=False,
)
exchange = ExchangeCoupling(
    model=primitive_model,
    kgrid=primitive_kgrid,
    fermiList=np.array([fermi]),
    eta=0.0001,
    exchangeRoutes=routes,
    symmetricExchangeSign=1,
    lowOrder=60,
    highOrder=80,
    Jcvg=-1e-4,
    verbose=0,
)

j_iso = exchange.exchangeEnergy[0, 0]
j_xx = exchange.exchangeEnergy[0, 0] + exchange.Jani[0, 0, 0, 0]
j_yy = exchange.exchangeEnergy[0, 0] + exchange.Jani[0, 0, 1, 1]
j_xy = exchange.Jani[0, 0, 0, 1]
d_x, d_y, d_z = exchange.DMI[0, 0]
gf_cxy = exchange.rotationCurvature[0, 0, 0, 1]
gf_cyx = exchange.rotationCurvature[0, 0, 1, 0]

supercell_model = build_primitive_model()
supercell_model.make_supercell(supercell_matrix)
supercell_model.verbose = False

mag_atom_i = 5 * 3 + 2
mag_atom_j = 6 * 3 + 2
e00 = occupied_energy(supercell_model)
primitive_energy = occupied_energy_with_kgrid(primitive_model, (supercell_matrix[0, 0] * supercell_kgrid[0], 1, 1))
supercell_energy_per_pc = e00 / supercell_matrix[0, 0]

print("Supercell folding check")
print(f"primitive energy/cell = {primitive_energy: .12f} eV")
print(f"supercell energy/cell = {supercell_energy_per_pc: .12f} eV")
print(f"difference             = {supercell_energy_per_pc - primitive_energy: .3e} eV")
print()
print("Green-function reference for route 0")
print(f"J_iso/2      = {j_iso / 2 * 1000: .6f} meV")
print(f"(J+Jani_xx)/2= {j_xx / 2 * 1000: .6f} meV")
print(f"(J+Jani_yy)/2= {j_yy / 2 * 1000: .6f} meV")
print(f"Jani_xy/2    = {j_xy / 2 * 1000: .6f} meV")
print(f"Dx/2, Dy/2, Dz/2 = {d_x / 2 * 1000: .6f}, {d_y / 2 * 1000: .6f}, {d_z / 2 * 1000: .6f} meV")
print(f"Cxy/2, Cyx/2 = {gf_cxy / 2 * 1000: .6f}, {gf_cyx / 2 * 1000: .6f} meV")
print()
print("Finite-rotation mixed second derivative in 10x supercell")
print("theta       Jxx/2 (meV)     err_xx (meV)     Jyy/2 (meV)     err_yy (meV)")
for theta in theta_list:
    coeff_xx = mixed_second_derivative(supercell_model, mag_atom_i, mag_atom_j, theta, 0.0, 0.0, e00)
    coeff_yy = mixed_second_derivative(supercell_model, mag_atom_i, mag_atom_j, theta, np.pi/2, np.pi/2, e00)
    coeff_xx_mev = coeff_xx / 2 * 1000
    coeff_yy_mev = coeff_yy / 2 * 1000
    ref_xx_mev = j_xx / 2 * 1000
    ref_yy_mev = j_yy / 2 * 1000
    print(f"{theta:7.4f} {coeff_xx_mev:16.6f} {coeff_xx_mev-ref_xx_mev:16.6f} {coeff_yy_mev:16.6f} {coeff_yy_mev-ref_yy_mev:16.6f}")

print()
print("Off-diagonal transverse finite-rotation check")
print("Jxy is extracted from 0.5*(Cxy + Cyx); Dz is extracted from 0.5*(Cxy - Cyx).")
print("The primitive route direction reverses the individual GF Cxy/Cyx labels, while the reported Dz follows the ordered supercell pair.")
print("theta       Cxy/2 (meV)     Cyx/2 (meV)     Jxy/2 (meV)     err_xy (meV)      Dz/2 (meV)      err_Dz (meV)")
for theta in theta_list:
    coeff_xy = central_mixed_second_derivative(supercell_model, mag_atom_i, mag_atom_j, theta, 0.0, np.pi/2)
    coeff_yx = central_mixed_second_derivative(supercell_model, mag_atom_i, mag_atom_j, theta, np.pi/2, 0.0)
    jxy_energy = 0.5 * (coeff_xy + coeff_yx)
    dz_energy = 0.5 * (coeff_xy - coeff_yx)
    jxy_energy_mev = jxy_energy / 2 * 1000
    dz_energy_mev = dz_energy / 2 * 1000
    ref_jxy_mev = j_xy / 2 * 1000
    ref_dz_mev = d_z / 2 * 1000
    print(f"{theta:7.4f} {coeff_xy / 2 * 1000:16.6f} {coeff_yx / 2 * 1000:16.6f} {jxy_energy_mev:16.6f} {jxy_energy_mev-ref_jxy_mev:16.6f} {dz_energy_mev:16.6f} {dz_energy_mev-ref_dz_mev:16.6f}")

print()
print("DMI finite-rotation check with rotated reference exchange fields")
print("Dx is extracted around an x reference from 0.5*(Cyz - Czy).")
print("Dy is extracted around a y reference from 0.5*(Czx - Cxz).")
print("The finite-difference sign follows the ordered supercell pair; the GF values below are sign-adjusted to that convention.")
print("theta        Dx_GF/2        Dx_FD/2       err_Dx        Dy_GF/2        Dy_FD/2       err_Dy")
exchange_x_ref = exchange_for_reference(np.array([1.0, 0.0, 0.0]))
exchange_y_ref = exchange_for_reference(np.array([0.0, 1.0, 0.0]))
d_x_ref = exchange_x_ref.DMI[0, 0, 0]
d_y_ref = exchange_y_ref.DMI[0, 0, 1]

for theta in theta_list:
    coeff_yz = central_mixed_second_derivative_from_directions(
        supercell_model, mag_atom_i, mag_atom_j,
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), theta
    )
    coeff_zy = central_mixed_second_derivative_from_directions(
        supercell_model, mag_atom_i, mag_atom_j,
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), theta
    )
    d_x_fd = 0.5 * (coeff_yz - coeff_zy)

    coeff_zx = central_mixed_second_derivative_from_directions(
        supercell_model, mag_atom_i, mag_atom_j,
        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), theta
    )
    coeff_xz = central_mixed_second_derivative_from_directions(
        supercell_model, mag_atom_i, mag_atom_j,
        np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), theta
    )
    d_y_fd = 0.5 * (coeff_zx - coeff_xz)

    dx_gf_mev = -d_x_ref / 2 * 1000
    dy_gf_mev = -d_y_ref / 2 * 1000
    dx_fd_mev = d_x_fd / 2 * 1000
    dy_fd_mev = d_y_fd / 2 * 1000
    print(f"{theta:7.4f} {dx_gf_mev:14.6f} {dx_fd_mev:14.6f} {dx_fd_mev-dx_gf_mev:12.6f} {dy_gf_mev:14.6f} {dy_fd_mev:14.6f} {dy_fd_mev-dy_gf_mev:12.6f}")
