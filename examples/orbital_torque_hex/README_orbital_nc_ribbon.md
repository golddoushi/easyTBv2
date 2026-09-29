# Stage 3: hexagonal BSR source / p-shell converter

Run from the repository root:

```text
python examples/orbital_torque_hex/orbital_nc_ribbon.py
python examples/orbital_torque_hex/orbital_nc_ribbon.py --edge armchair
python examples/orbital_torque_hex/orbital_nc_ribbon.py --edge armchair --workers 8
python tests/test_hex_nc_ribbon.py
```

Defaults are 4 zigzag or 2 armchair source transverse cells, followed by
3 converter rows. Both choices have 8 s-shell source atoms and 3 p-shell
converter atoms (34 spinful bands). They have different physical widths and
longitudinal repeat lengths, so their absolute per-cell responses should not
be interpreted as a matched-width efficiency comparison. The model retains
the source hoppings and staggered +/-0.5 eV onsite energies exactly.

## Converter and interface

C is a rectangular lattice with one full p shell per site, in the repository
basis (pz,px,py). Its onsite term is shift*I + lambda*L.sigma/2, and every
nearest-neighbor C hopping is -t_C*I. Defaults: lambda=0.4, t_C=0.5, shift=0 eV.
Thus isolated bulk C has H(k)=epsilon(k)*I+H_SOC with k-independent internal
eigenvectors and velocity proportional to identity. It has no interband
velocity or BSR moment, and its transverse ACA OHC and SHC vanish. The script
also checks all components of those three bulk Hall tensors numerically at
the requested fillings using the public DM Hall calculators on a 32x32 mesh.
This is a dispersive transport system, not an isolated-atom control.

The uppermost N atomic line connects to C's first row through oblique bonds.
The spin-independent s-to-p matrix is t_I*(0,dx,dy)/|d|, with Hermitian reverse
hoppings and default t_I=0.4 eV. C is displaced longitudinally by half a
repeat relative to one top N atom, ensuring both px and py interface mixing.
Longitudinal home-cell conventions are retained when assigning these bonds.
`geometry.png` shows the actual connectivity. Other interface registries
would define physically different toy junctions, and can change conversion.

All atoms lie in one xy plane. The s,px,py orbitals are mirror-even and pz
is mirror-odd; atomic L.S preserves the spinful mirror. Ex is even, whereas
magnetic moments mx,my are odd. Tests verify the Hamiltonian mirror and
time-reversal symmetries, and computed in-plane moments are checked for
numerical zero. Figures show only z, but saved arrays retain xyz.

## Responses and projection convention

The uniform-drive profiles sum moments over each physical transverse row;
an armchair dimer line contains both atoms. BSR is calculated from the full
junction eigenstates, rather than copied from the isolated source. Its row
observable is {P_row,m_BSR}/2, and the C-total observable is {P_C,m_BSR}/2.
The spin observable is the magnetic moment -sigma in mu_B. The BSR and spin
profiles use the established length-gauge first-order density-matrix kernel,
including population I and coherence E channels, with scalar CRTA.

The row-drive figure reports the moment summed over C while applying one
vertex v_x^(r)={P_r,v_x}/2 at a time. Hexagonal and interface bonds have
inter-row v_x elements. This symmetric assignment allocates half of each
such bond to each endpoint row; sum_r v_x^(r)=v_x. It is NOT the strictly
internal projection P_r*v_x*P_r, whose sum omits cross-row bonds. The public
source/target projection realizes the chosen vertex with source=row,
target=all atoms. Direct public-calculator tests verify every row drive
against the optimized dual contraction. Summing the row-drive responses
reconstructs the uniform-drive C-total response. Summing N rows corresponds
to {P_N,v_x}/2, including half the interface vertex; it is not an electric
field strictly confined to N or a self-consistent electrostatic solution.

Each calculation includes full, SOC-off, and interface-disconnected cases.
The last case retains hopping and SOC within C. Disconnected C has zero
BSR/spin accumulation, including under the row drives; turning SOC off
eliminates spin response but need not eliminate junction BSR response.

## Output and validation

Outputs live in `tmp/orbital_torque_hex/nc_zigzag` or `nc_armchair`:

- `profile_f0.png`, `profile_f1.png`: uniform Ex, row BSR/spin z moments.
- `drive_rows_f0.png`, `drive_rows_f1.png`: C-total BSR/spin z versus driven row.
- `geometry.png`: N/C sites and interface bonds.
- `responses.npz`: full xyz arrays, profile I/E channels, controls, and each mesh.
- `profiles.csv`, `drive_rows.csv`: z-component tables for all cases and fillings.
- `summary.json`: parameters, mesh differences, symmetry/control tests,
  converter Hall zeros and row-drive reconstruction errors.
- Case subdirectories: reproducible primitive source and junction Hamiltonians.
  `model.npz` includes the final SOC Hamiltonian; the hr input precedes SOC.

Response units are mu_B/(V/Angstrom) per longitudinal repeat. Defaults use
EF=0 and 0.8 eV, kBT=0.01 eV, linewidth=0.02 eV, and 512/1024 shifted k meshes.
Chunks of at most 32 points bound memory and exactly partition the mesh.
`--workers N` runs independent chunks concurrently in a thread pool, with
one BLAS thread per worker to avoid nested thread oversubscription. The
default is 1 (serial). Workers are capped at the available chunk count
(16 chunks for Nk=512, 32 for Nk=1024); they need not divide that count.
Mesh points, chunk weights, and reduction order are identical to serial.
Both uniform profiles and row-drive contractions are parallelized. The small
isolated-converter bulk check and plotting remain serial. No dense array of
all drive/observable row pairs is built. Memory grows with simultaneous
workers; choose the worker count by timing representative runs on your machine.
Actual worker counts and per-case/per-mesh timings are printed and saved in
`summary.json`. Tests compare serial and parallel outputs for both edges.

For the validation runs at 1024 points, the full-model maximum in-plane
response is below 2e-13 and row-drive reconstruction errors below 5e-14.
The isolated converter Hall tensors are below 2e-12. The 512-to-1024 maximum
profile changes are about 2.67e-4 (BSR), 1.08e-4 (spin) for zigzag, and
2.32e-4 (BSR), 1.87e-4 (spin) for armchair. These are mesh comparisons, not
width or linewidth convergence. SOC-off spin and disconnected C responses
are numerical zero. Tests independently compare profiles and row drives
against public calculators, including the spin-moment sign.

Finite C spin response establishes a junction response requiring SOC and
coupling to N. It does not by itself isolate a conserved BSR current flowing
from N and converted into spin: the junction changes the eigenstates, C
driving can contribute, and BSR is a nonlocal observable with an explicitly
chosen spatial partition. EF=0 denotes the isolated N midgap; C is metallic
for these parameters, so the entire junction must not be called insulating.
ACA moments in C are distinct from the BSR observable requested here.

Options include `--edge`, `--n-width`, `--c-width`, `--soc`,
`--interface-hopping`, `--converter-hopping`, `--converter-shift`, `--grids`,
`--fermi`, `--kbt`, `--linewidth`, `--workers`, `--output`, and `--plot-only`.

## Profiles for individual drive rows

The C-total curve alone cannot tell where the induced moment resides. Compute
the two-row susceptibility chi[drive_row, observed_row, fermi] with:

```text
python examples/orbital_torque_hex/orbital_nc_ribbon.py --row-profiles --workers 8
python examples/orbital_torque_hex/orbital_nc_ribbon.py --row-profiles-only --output tmp/orbital_torque_hex/nc_zigzag --workers 2
python examples/orbital_torque_hex/orbital_nc_ribbon.py --row-profiles-only --output tmp/orbital_torque_hex/nc_zigzag --profile-drive-rows 15 16 32 63 --workers 2
```

The first command adds profiles to a new calculation. The other commands use
the saved model parameters and finest mesh, independently of ordinary CLI
parameter defaults, and leave the original outputs untouched. Omit row indices
to calculate all rows. Indices are zero-based physical transverse rows; both
atoms of an armchair dimer line belong to one row. Saved Hamiltonians are
compared with rebuilt models before calculation. Uniform and C-total marginal
arrays alone cannot reconstruct this matrix, so a new response calculation
is necessary.

The `row_profiles/` subdirectory contains:

- `profile_drive_rowNNN_fM.png`: BSR/spin z versus observed y for a specified
  drive, with full/SOC-off/disconnected models. A vertical marker identifies
  the driven row. These are raw responses, with no spatial symmetrization.
- `response_matrix_fM.png`: full-model heatmaps; horizontal index is observed
  row and vertical index enumerates selected drives in `drive_rows`.
- `responses.npz`: all cases, z-only total/I/E matrices with axes selected
  drive row, observed row, Fermi; plus positions, regions and drive indices.
- `profiles.csv`: total/I/E values with drive and observed row indices,
  coordinates, region labels and Fermi energies.
- `summary.json`: source parameters, mesh, timing, and marginal residuals.
- `checkpoint.npz`: intermediate completed cases; model inputs in `models/`.

Replot stored matrices without recalculation using `--plot-only --row-profiles`
with the same `--output` directory.
Use `--plot-only --row-profiles-only` to regenerate just that subdirectory,
leaving the original uniform and C-total plots untouched.

For each drive, summing observed C rows must reproduce the old C-total curve.
When all drives are requested, summing drive rows must reproduce the uniform
profile. Both identities are checked, and independent tests compare every
matrix element and I/E channel with public projected DM calculators. The
implementation contracts low-rank row projectors one k point and eight
observed rows at a time. It reuses the eigensystem and does not allocate a
k*drive*observed*band*band tensor. This still costs more than computing only
the C total, especially for hundreds of bands; selected drives reduce cost.

## Interpreting a wide converter

The 2026-09-29 saved zigzag run with N width 8, C width 48, lambda=0.1,
eta=0.02 and kBT=0.01 eV uses only 48/64 k points. Its 48-to-64 maximum
differences are 12.63 (BSR profile), 6.40 (spin profile), 0.569 (C-total BSR
versus drive) and 3.92 (C-total spin versus drive). The finest-mesh maximum
spin profile is only 2.14 and maximum C-total spin versus drive is 4.25.
These discrepancies prevent treating oscillation amplitudes or decay lengths
as converged predictions. Tiny marginal residuals verify algebra, not k-mesh
convergence. Newly generated row profiles at that same mesh are diagnostic,
not an additional convergence check.

In that saved 64-point calculation, driving the last C row (63) produces its
largest BSR moment at the first C row (16), adjacent to N: approximately
-2.082 mu_B/(V/Angstrom). This demonstrates why a C-integrated response cannot
locate the accumulation. The spin profile from the same drive has its largest
magnitude near row 45, approximately -0.0658, and its population contribution
is substantial. These are diagnostic observations at the unconverged mesh,
not established transport lengths or oscillation periods.

Nonlocal response is possible here: C is metallic at EF=0, eigenstates extend
across the finite ribbon, and the interface changes those states even though
isolated C has no Hall response. Standing-wave interference and SOC-related
phase differences can produce sign changes in a local spin susceptibility.
This is not an exchange-driven precession model (there is no ferromagnet).
The scalar CRTA relaxes eigenstate populations/coherences; it does not solve
a spatial diffusion equation with local reservoirs. In particular, the I
channel can redistribute occupation of states extended across the whole C.

There is an additional issue for BSR: {P_row,m_BSR}/2 assigns part of a
nonlocal operator to the row. A large C-total BSR response is not by itself
evidence that a locally conserved orbital current propagated that distance.
The spin operator is local in the atomic basis, so a nonlocal spin response
has no such observable-partition explanation, although the state/kernel
and finite-mesh issues remain. Deep inside C the symmetrized row drive equals
the internal row vertex because vertical C-C hopping has zero vx; half-bond
assignment matters at the oblique N/interface bonds.

Before extracting transport lengths, converge k sampling at fixed geometry,
then compare converter width, linewidth and interface coupling. Inspect the
two-row matrix and its I/E components to distinguish a moment near the drive
from a response near the interface. The projected-drive protocol is not a
self-consistent electrostatic bias or proof of diffusion.
