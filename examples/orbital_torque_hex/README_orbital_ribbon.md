# Stage 2: ribbon BSR moment response

```text
python examples/orbital_torque_hex/orbital_ribbon.py
python examples/orbital_torque_hex/orbital_ribbon.py --plot-only
python examples/orbital_torque_hex/orbital_ribbon.py --mesh-shift 0.5 --output tmp/orbital_torque_hex/ribbon_shifted
python examples/orbital_torque_hex/orbital_ribbon.py --edge armchair --width 4 --mesh-shift 0.5 --grids 128 256 --output tmp/orbital_torque_hex/ribbon_armchair
python tests/test_hex_orbital_ribbon.py
```

The stage-1 spinful s-only honeycomb is repeated along a2, then cut along
that direction. It remains periodic along a1 (Cartesian x), with finite y.
Defaults: 8 primitive transverse cells, 16 atoms at different y coordinates,
32 bands; EF=0 (gap) and 0.8 eV (metallic); kBT=0.01 eV and linewidth=0.02 eV.
The original staggered onsite energies and hopping are retained. The two
terminations are inequivalent because of the sublattice potential; an odd
profile or zero total moment is not imposed.

`--edge armchair` instead uses the rectangular supercell with longitudinal
vector a1+2*a2 and transverse vector -width*a1, rotated so the longitudinal
direction is again Cartesian x. The basis is wrapped before cutting, and
the diagonal TB position vertex is rebuilt at the new centers. Tests check
nearest-neighbor distances, four two-neighbor edge atoms, and three-neighbor
interior atoms. `geometry.png` shows the actual bonds and both sublattices.
Armchair width W contains 4W atoms (2W dimer lines), versus 2W zigzag atoms.
Width 4 armchair and width 8 zigzag both have 32 bands but different physical
widths and repeat lengths; their raw per-cell susceptibilities are not a
matched-width comparison. Saved arrays and CSV retain each atom separately.
Armchair profile figures sum the two atoms at equal y and plot against the
physical dimer-line coordinate. These are sums, not averages.

The glide `(x,y) -> (x+Lx/2, ymin+ymax-y)` preserves each sublattice and
the Hamiltonian. Ex is even and the axial moment mz is odd, requiring
opposite responses at glide-related sites, for each I/E channel separately.
The earlier atom-index plots could appear asymmetric because floating-point
y sorting ordered A/B differently within equal-y dimers. Reversing that
index sequence is not the glide permutation. At EF=0.8, the lower-edge A/B
responses are -3.34168330/+10.02672103; their upper-edge partners have the
opposite signs. Dimer-line edge sums are +6.68503772/-6.68503772. No symmetry
averaging is imposed; the corrected plot simply groups equal-y sites.

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
  LG mesh comparison, VG finite part, half linewidth at fixed temperature,
  and half temperature at fixed linewidth. The next panel separates I/E;
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

The earlier simultaneous change of temperature and linewidth could not
identify which parameter caused the difference. The current example varies
them independently: `narrower_*` means half linewidth only, while `colder_*`
means half kBT only. At EF=0 the zigzag spectrum has edges +/-0.5 eV, so the
occupied-to-empty transition gap is at least 1 eV, much larger than linewidth
0.02 eV. Fermi occupation corrections at kBT=0.01 eV are exponentially small
(exp(-50)); small splittings within occupied or empty manifolds do not supply
a sizable occupation difference. Interband therefore does not imply a
small-gap resonance.

The CRTA coherence kernel is proportional to
`i*(f_n-f_m)*v_nm/[Delta_nm*(Delta_nm-i*eta)]` in the code's convention.
For this time-reversal-symmetric model's time-reversal-odd moment response,
the surviving dissipative contribution contains a factor proportional to
`eta/(Delta_nm^2+eta^2)` (with the other matrix elements and gap factors).
When eta is far below the occupied-to-empty gap it scales approximately
linearly with eta. This explains why a finite interband accumulation can
halve with linewidth, despite saturated occupations. It does not demonstrate
a relaxation-independent insulating edge accumulation. The BSR operator's
own excluded-gap prescription is a separate issue from these response
denominators.

Independent controls at EF=0 (maximum absolute atom-projected z response):

| Ribbon / shifted mesh | kBT=.01, eta=.02 | kBT=.005, eta=.02 | kBT=.01, eta=.01 |
|---|---:|---:|---:|
| Zigzag W=8 / 512 | 0.0078996691 | 0.0078996691 | 0.0039503235 |
| Armchair W=4 / 256 | 0.0064458894 | 0.0064458894 | 0.0032235118 |

Units are mu_B/(V/Angstrom) per longitudinal repeat. Temperature halving
changes neither midgap profile at numerical precision; linewidth halving
approximately halves both. The armchair 128-to-256 midgap profile change is
below 9e-13, whereas the metallic change is 0.00458 and needs further mesh
convergence for higher precision. These are fixed-width examples, not width
convergence or evidence that one termination is more efficient.

For reference, in the original unshifted run with jointly varied parameters,
at Nk=512 the maximum z/Ex response was 0.00786228 at EF=0 and 11.33479065
at EF=0.8 eV. Halving both temperature and linewidth gives 0.00393163 and
22.82692282 respectively. The 256-to-512 maximum profile differences are
3.74e-5 and 2.64e-4; these are finite-mesh comparisons, not a claim of exact
convergence. The local-sum error is below 6e-15, ACA is exactly zero, and
the maximum spin response is below 4e-20. These response values use the
unshifted meshes; the contact changes the static gauge residual, not the
pole-subtracted dc finite part. Residual units are mu_B eV/(V/Angstrom).

Options: `--edge`, `--width`, `--grids`, `--fermi`, `--kbt`, `--linewidth`,
`--batch-size`, `--mesh-shift`, `--output`, `--plot-only`. Projector batches limit memory
and reuse the band states. Equal interleaved k submeshes cap each calculator
at 64 points; their union is exactly the requested grid. Response tensors
are averaged, while band edges use the appropriate maximum/minimum. This
is valid for the k-local BSR/scalar-CRTA kernels and is checked against an
unsplit mesh. BLAS is restricted to one thread for these small
matrix tests. No SOC, magnetic exchange or spin cancellation is imposed by
post-processing.
