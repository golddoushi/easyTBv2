# Direct-position OBC orbital Hall comparison

```text
python examples/orbital_torque_hex/orbital_hall_obc.py --sizes 4 6 8 12 --linewidths 0.02 0.05
python examples/orbital_torque_hex/orbital_hall_obc.py --delta 0.4 --t2 0.15 --phase 0 --output tmp/orbital_torque_hex/obc_trivial
python examples/orbital_torque_hex/orbital_hall_obc.py --delta 0.4 --t2 0.15 --phase 1.5707963267948966 --output tmp/orbital_torque_hex/obc_chern
python tests/test_orbital_hall_obc.py
```

`--sizes` specifies the side lengths of open N by N parallelograms of unit
cells. No hopping wraps across a boundary. `--phase` is in radians.
The code diagonalizes the finite Hamiltonian, rather than inserting bulk
Bloch eigenstates into an open sample. The two spin sectors are identical;
one 2N^2-dimensional block is diagonalized and responses include a factor 2.
Dense diagonalization scales as N^6 and matrix memory as N^4. Increase sizes
gradually. `--bulk-grid` selects the independent PBC reference mesh.

In the orthogonal site basis, position is diagonal, measured from the
sample's geometric center. The definitions are

```
V = hbar*v = -i [r,H]
m_z/mu_B = -(x V_y - y V_x - V_x y + V_y x)/(4 C)
C = hbar^2/(2 m_e), in eV Angstrom^2
J_a = {m_z, V_a}/2
```

This is the full sample-centered finite orbital moment. There is no
band-diagonal position removal and no reconstruction with energy
denominators. It is not identical by definition to a bulk self-rotation
moment. Angular momentum depends on its reference origin; we use the same
geometric centering rule for every size.

The finite density-matrix solver uses the scalar-potential perturbation
`H'=E*x` and the same constant-relaxation convention as the bulk LG example:

```
(epsilon_m - epsilon_n - i eta) rho1_mn = (f_m - f_n) x_mn
```

The source is evaluated directly from the position matrix and occupations.
No inverse-gap cutoff or periodic intraband population term is added.
Diagonal source elements vanish; exactly degenerate pairs have zero source
at equal equilibrium occupation. The stationary equation
`-i[H,rho1] - i[x,f(H)] - eta*rho1 = 0` is checked independently, including
a site-basis check in the tests. Magnetic moments are in mu_B, and Hall
responses use the bulk example's native `2*pi/area` normalization, with area
N^2 times the primitive cell area. These are finite-flake current responses,
not lead-measured conductances.

Outputs include:

- `obc_comparison.png`: diagonal finite-state orbital moments, whole-sample
  Ex-to-Jy response, its antisymmetric xy/yx part, and an interior partition.
- `responses.npz`: full in-plane response tensors, induced total orbital
  moment, PBC V1/V2 references, finite spectra and diagonal orbital moments.
- `flake_n*.npz`: Hamiltonian, coordinates, origin, eigensystem and full
  orbital-moment matrix for each sample.
- `hall_n*_eta*.csv` and `summary.json`: curves, parameters, commutator and
  stationary-equation residuals.

`--edge-layers` excludes that many outer cell layers from the interior
partition. We use `{P_interior,J}/2`, normalized by the retained cell area.
This partitions the same sample-centered current; it is not a local bulk
orbital-magnetization marker. If no interior cells remain the curve is NaN.
`--plot-only` regenerates the figure from saved arrays.

## What a size comparison establishes

Zero bulk Chern number removes the requirement for topological chiral edge
states, but does not guarantee negligible boundary contributions to orbital
magnetism or to this current operator. A moment weights motion by distance
from the origin, so the surface-to-volume counting argument alone is
insufficient. Ordinary boundary states and finite-size level spacing also
matter. The order of the sample-size and relaxation/DC limits matters for
an isolated finite system. Compare several sizes at fixed eta, then vary
eta, and converge the PBC mesh separately. Do not require equality between
the full finite-sample observable and the bulk BSR vertex as a pass/fail
test. The default staggered honeycomb and the phase=0 Haldane case provide
topologically trivial comparisons; phase=pi/2 at M=.4,t2=.15 adds the Chern
case. All these spin-degenerate models still have no SOC.

The former direct-coordinate/PBC-velocity diagnostic and its tests have
been removed. Existing archived output files are left untouched.
