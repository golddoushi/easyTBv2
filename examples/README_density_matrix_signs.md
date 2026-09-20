# Physical density-matrix convention (2026-09-12)

The reusable kernels use `hbar=e=1`, `exp(-i omega t)`,

    H_V = H(k+A),   E = -dA/dt,   H_L = H0+E_a r_a.

All response orders are physical density matrices. An observable is evaluated
as `Tr(rho O)`. Only charge-current calculators insert `j=-dH/dA`; positive
SI conversion factors then restore the magnitudes of the electron charges.
Magnetic moments already have their physical signs in their own operators.
The magnetic-order reaction torque in the LLG layer is separate and unchanged.

## First order

Writing `Df=D_a f(H0)` and `Delta_mn=E_m-E_n`, the physical electric-field
response satisfies

    (Gamma-i omega+i ad_H0) rho1 = Df
    rho1 = -i Df/(Delta-omega-i Gamma).

For the dc interband-only inverse-gap construction this becomes

    rho_E,mn = +i (f_n-f_m) v_a,mn/[Delta_mn (Delta_mn-i eta)]
    rho_I,nn = f'_n v_a,nn/eta.

The same convention applies to CRTA, SERTA, and the mesh-coupled full collision.
The latter retains its fixed-particle-number source projection. Neither the
generic contraction nor a collision solve inserts an observable charge.

The linear VG `block()` returns the stable, pole-subtracted electric-field
density. `vector_potential_density()` returns the full physical `delta-rho/A`:

    R(omega) = Df+i omega rho1.

Consequently its conductivity is

    -Tr(rho1 v) + i/omega Tr(Df v+f w).

The contact residue already combines `j=-v` with `E=i omega A`, so it must
not be flipped along with the density contraction. This also applies to the
static EPC expansion: all of its density coefficients have the physical
sign, its measured current has a minus, and its VG residue keeps a plus.

## Second order and torque vertices

LG first-order blocks multiply electric fields. VG second-order-kernel
first-order blocks multiply vector potentials. `rho2` is the mixed coefficient
for the two ordered input fields at `(-omega,+omega)`, including both drive
orderings; it is not divided by two. For `E(t)=Eamp cos(omega t)`, its dc density
contribution is `rho2*Eamp^2/4`. Convert the VG pair using
`A(-omega)A(+omega)=E(-omega)E(+omega)/omega^2`.

The migration reverses exposed first-order blocks but preserves every
second-order density, including relaxation and nonlinear Pauli terms. The
driving recurrence, density derivatives, and LG-to-VG transformations change
together. BPVE measures physical charge current:

    LG: -Tr(rho2 v)
    VG: -Tr(rho2 v + rho1_minus w_b + rho1_plus w_a + rho0 u_ab)/omega^2.

IFE and IFT are direct observables and have no charge-current minus. For
`T=-i[mu,H]`, the torque vertices are

    T_A = -i[mu,v],   T_AA = -i[mu,w],   T_E = -i[mu,r].

Both first-order torque vertices and first-order densities reverse relative
to the previous implementation, preserving their IFT contact products.
The IFE/IFT channel TDSE helper, BPVE TDSE drives/current, and Bloch-acceleration
Hamiltonian were migrated to the same `H(k+A)` convention. The linear optical
and moving-phonon TDSE implementations already had the physical current sign.

## Effect on existing results

| Quantity | Relative to the immediately preceding implementation |
|---|---|
| First-order density blocks | Reverse sign |
| Second-order density blocks | Unchanged |
| Linear AHC/optical charge conductivity, including saved k records | Unchanged |
| Static EPC optical conductivity and VG contact residue | Unchanged |
| Direct spin/orbital/torque galvanic response | Unchanged |
| IFE/IFT DM response | Unchanged |
| BPVE LG/VG DM charge response | Reverse sign |
| SpinHallConductance_LG_DM | Reverse sign |

`spinVelocity_K={mu_S,v}/2` is a **spin magnetic-moment current**, with
`mu_S/mu_B=-gS*S/hbar`. It is not the spin-angular-momentum current `{S,v}/2`.
Its historical 2D output normalization is retained. Legacy closed-form
Berry/response calculators retain their own conventions; their agreement
with a DM result is not used to override the physical density/current sign.

Do not compare old and new BPVE or DM spin-Hall output files without accounting
for this sign change. User code directly consuming first-order blocks must
remove compensating density minuses and supply physical observable vertices.

## Verification

`tests/test_physical_dm_signs.py` checks density equations before any observable
contraction, both gauges' first-order blocks inside second-order kernels,
physical torque derivatives, static EPC density recurrences, TDSE Hamiltonian
derivatives, and IFT time/frequency agreement. The nonlinear EPC propagation
test also measures `j=-v` from independently time-integrated densities.

The before/after audit in `tmp/dm_sign_migration/` covers 91 arrays across
interband, full-derivative, covariant-SERTA, and full-EPC branches. Expected
unchanged arrays and sign reversals agree exactly in that audit.
