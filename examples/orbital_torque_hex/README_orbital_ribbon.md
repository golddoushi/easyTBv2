# Stage 2: ribbon BSR moment response

```text
python examples/orbital_torque_hex/orbital_ribbon.py
python examples/orbital_torque_hex/orbital_ribbon.py --plot-only
python examples/orbital_torque_hex/orbital_ribbon.py --mesh-shift 0.5 --output tmp/orbital_torque_hex/ribbon_shifted
python tests/test_hex_orbital_ribbon.py
```

The stage-1 spinful s-only honeycomb is repeated along a2, then cut along
that direction. It remains periodic along a1 (Cartesian x), with finite y.
Defaults: 8 primitive transverse cells, 16 atoms at different y coordinates,
32 bands; EF=0 (gap) and 0.8 eV (metallic); kBT=0.01 eV and linewidth=0.02 eV.
The original staggered onsite energies and hopping are retained. The two
terminations are inequivalent because of the sublattice potential; an odd
profile or zero total moment is not imposed.

## Calculator audit and scope

The public LG calculator contracts the unperturbed ribbon BSR moment with
the established first-order density matrix, `Tr(rho1 m_BSR)`. Tests verify
the direct trace, Hermiticity/formula of the BSR operator, units, channels,
projector normalization and calculator/task entry points. The new ribbon
test additionally evaluates projected traces in the AO basis and compares
different projector batch sizes. The tests retain gap and metallic fillings.

The upgraded VG calculator includes the covariant observable contact `D_x m`,
including derivatives of the row projectors. The static gauge sum rule is now
the Brillouin-zone average of `Tr[(D_x f)m + f(D_x m)]`, rather than the first
term alone. It is the integral of a periodic derivative and should vanish
with converged integration. Equality of the finite LG/VG responses alone is
not this test: the analytic pole subtraction makes their dc densities equal.

The contact passes independent finite-difference, band-phase covariance,
nonlocal-position/projector and finite-frequency tests in
`tests/test_observable_contacts.py`. The ribbon regression also tests the
unsubtracted static sum rule for all row moments. For width 3, half-step
meshes of 128/256/512 points give maximum metallic Ex residuals
2.80e-3 / 2.76e-6 / 4.02e-13; the midgap residual is below 6e-14.
Thus the previous missing-contact obstruction is resolved in these tests.

The default width-8 ribbon gives metallic residuals 1.71e-3 / 1.65e-6 /
4.01e-13 on the same half-step meshes, taking the maximum over all 16 row
observables. Its midgap residual stays below 4e-13. The unshifted 256/512
results are 0.01352 / 0.006759. `--mesh-shift 0.5` exposes this integration
check without changing the Hamiltonian or contact implementation; the default
remains zero to preserve the original profile-mesh comparison.

Unshifted meshes sample exact band degeneracies and converge more slowly:
the width-3 residual is 0.00617 / 0.00308 / 0.00154 on the same meshes.
The BSR contact holds the excluded-gap cutoff fixed; it is not a general
degenerate-subspace formulation. Mesh convergence and sensitivity to that
cutoff remain necessary, especially for wider ribbons. This audit concerns
the gauge transform of the chosen BSR observable, not a proof that it equals
the full orbital magnetization including all Wannier contributions.

## Spatial assignment

BSR is reconstructed from the **ribbon** eigenstates and velocity matrices,
including the transverse velocity of intra-ribbon bonds. It is not the bulk
BSR moment pasted onto the edges and it is not an onsite atomic angular
momentum. The moment assigned to atom i is

`m_i = (P_i m_BSR + m_BSR P_i)/2`.

This is the existing orbital-observable projection convention, distinct from
the source/target projected driving-velocity convention `P_i v P_i`.
Because the BSR matrix is nonlocal, replacing m_i with `P_i m_BSR P_i`
would lose cross-atom terms. With the symmetrized assignment the sum of local
moments equals the total. This spatial assignment is a defined partition of
a nonlocal observable, not a unique microscopic on-site magnetization.
The applied Ex is uniform; there are no row-projected drives in this stage.

## Outputs and interpretation

`tmp/orbital_torque_hex/ribbon/` contains:

- `profile_f0.png` and `profile_f1.png`: BSR z/Ex profiles at each filling,
  LG mesh comparison, VG finite part, and a run with
  both temperature and linewidth halved. The next panel separates I/E;
  the final panel shows ACA and spin zeros without rounding the data.
- `responses.npz`: full field/moment tensors and channels for every mesh,
  VG pole residuals including the contact and reduced-temperature/linewidth runs.
  Profile axes: observed row, Fermi, field, moment. Total tensors omit row.
- `profiles.csv`: finest-mesh z/Ex data with positions and control values.
- `summary.json`: settings, band edges, sum/gauge residuals, zeros,
  mesh differences and peak response for each filling.
- `model/ribbon_model.npz`: actual cut Hamiltonian, geometry and onsite
  moment matrices, plus primitive inputs.

Units are mu_B/(V/Angstrom) per longitudinal repeat per assigned row. These
are accumulation susceptibilities, not the stage-1 Hall-current units.
The hopping-off control is omitted because isolated atoms are not a transport system.
ACA is exactly zero in every model because all basis orbitals are s-type.

An insulating bulk BSR Hall plateau does not force a finite, relaxation-
independent edge accumulation. In the initial runs the midgap response is
interband and approximately halves when linewidth and temperature are halved;
the metallic response is much larger and dominated by the population channel.
This is why both fillings and relaxation settings are retained. A single
finite-linewidth profile does not establish a clean-limit insulating edge
current or torque source. It also does not establish edge localization or a
decay length without width convergence. The 64/128 meshes were inadequate
for the metallic case; defaults use 256/512 and report remaining differences.

At Nk=512, the maximum z/Ex response is 0.00786228 at EF=0 and 11.33479065
at EF=0.8 eV. Halving both temperature and linewidth gives 0.00393163 and
22.82692282 respectively. The 256-to-512 maximum profile differences are
3.74e-5 and 2.64e-4; these are finite-mesh comparisons, not a claim of exact
convergence. The local-sum error is below 6e-15, ACA is exactly zero, and
the maximum spin response is below 4e-20. These response values use the
unshifted meshes; the contact changes the static gauge residual, not the
pole-subtracted dc finite part. Residual units are mu_B eV/(V/Angstrom).

Options: `--width`, `--grids`, `--fermi`, `--kbt`, `--linewidth`,
`--batch-size`, `--mesh-shift`, `--output`, `--plot-only`. Projector batches limit memory
and reuse the band states. Equal interleaved k submeshes cap each calculator
at 64 points; their union is exactly the requested grid. Response tensors
are averaged, while band edges use the appropriate maximum/minimum. This
is valid for the k-local BSR/scalar-CRTA kernels and is checked against an
unsplit mesh. BLAS is restricted to one thread for these small
matrix tests. No SOC, magnetic exchange or spin cancellation is imposed by
post-processing.
