# Weak-EPC chain conductivity comparison

`--static-epc-dm` now adds the complete static Gaussian EPC response through
second displacement order, with analytic density-matrix coefficients. See
[the static DM scheme](README_chain_static_epc_dm.md) for equations and validation.

Use `--phonon-q`, `--phonon-branches` and `--phonon-weights` to restrict every
approach to the same mode sector. See [selected phonon modes](README_chain_selected_phonons.md)
for normalization, commensurability and examples.

The example now also accepts `--tdse` for prescribed moving-phonon dynamics
and `--tdse-frequency-check` for an independent same-trajectory frequency
solver. See [the TDSE audit and usage guide](README_chain_TDSE_alignment.md)
for controls, numerical validation and the distinction from quantum EPC.

Numerical investigation performed on 2026-09-08 using
`examples/chain_transport_alignment.py`. Data and individual run metadata are in
`tmp/chain_alignment/`; `comparison.csv` collects the complex conductivities.
The full-operator results use this repository's `CollisionOperator_full`, not an
independent EPW calculation. Only its population sector has the aiBTE structure;
interband coherences still use secular damping.

## Controlled model and normalization

`--phonon-statistics quantum|wigner|classical` controls all EPC methods,
including snapshots, snapshot curvature, complete static DM, SERTA, full
collision/balanced RTA, TDSE and its frequency-domain check. The default
`quantum` preserves the previous default calculations: quantum absorption and
emission rates, quantum static variance, and Wigner trajectories as the TDSE
approximation. Explicit `wigner` uses that same variance but symmetric
positive/negative-frequency noise rates. `classical` uses equipartition
variance and classical noise rates throughout. The old `--tdse-statistics`
spelling remains an alias, but now affects every method even without `--tdse`.

For classical/Wigner noise, `full` and `balanced_rta` retain their thermal
pair-balancing approximation; they are not exact trajectory collision operators.
Historical array names such as `dynamic_serta` are retained; the chosen bath
and its mapping to each approach are saved in `metadata.json`.

Masses are independent of this statistics switch. At 300 K and 40 meV,
classical displacement variance is 0.838946 times the quantum/Wigner variance
(amplitudes are 0.915940 times as large). Complete static DM and snapshot
curvature corrections scale by the variance ratio for this equal-frequency
model. Full finite-amplitude snapshot conductivities need not scale exactly.
The frequency-dependent mass adjustment below preserves **quantum** variance;
it does not preserve classical variance when phonon energy is varied.

Both classical and Wigner TDSE propagate moving phonons; this option changes
statistics, not whether dynamics are present. Use `--fixed-centers` and compare
`static_epc_dm_fixed`/`fixed_center_snapshot` with TDSE to match optical vertices.

The electronic model is the unchanged four-band spinful Rice-Mele chain, with
nearest-neighbor hoppings 1 and 0.72 eV and SSH derivatives 0.45 and 0.35 eV/A.
There is no quadratic SSH beta term. The chemical potential is -0.7 eV, inside
the valence manifold, so the model is metallic. Electronic and phonon
temperatures are both 300 K (kBT = 25.852 meV).

The default phonon energy is 40 meV. `--scale s` changes masses from (12,14) amu
to (12,14)/s^2, making both displacements and EPC matrix elements proportional
to s. It also scales the moving-center optical vertices consistently. At
s=0.25 the RMS hopping fluctuations are 12.56 and 9.77 meV (1.26% and 1.36% of
the hoppings); at s=0.125 these are halved. The respective atomic RMS
displacements at s=0.25 are 0.02048 and 0.01896 A.

Every method includes the same 20 meV background linewidth. SERTA receives
Gamma_EPC + Gamma_background explicitly. Full collision uses its
number-conserving background. Snapshot calculations use CRTA. The comparisons
are at fixed chemical potential, not fixed carrier density across snapshots.
The chain's arbitrary transverse area is 100 A^2; relative changes are more
meaningful than the absolute S/m normalization.

Define Delta sigma = sigma - sigma0, where sigma0 is the undisplaced CRTA
conductivity. Distinguish disagreement relative to sigma from disagreement
relative to Delta sigma: making the coupling weak trivially improves the first
without necessarily improving the second.

## Matched finite-grid ensemble

Runs `main` and `half`: 512 electronic k points, 32 phonon q points, 32-cell
supercells with 16 supercell k points, 20 meV EPC energy-conservation smearing,
and 200 independent antithetic pairs (400 displaced calculations). Pair means,
not the individual correlated +/- configurations, determine standard errors.
Identical random seeds are used for both coupling strengths.

| Method | DC Delta sigma / sigma0, s=0.25 | DC Delta sigma / sigma0, s=0.125 |
|---|---:|---:|
| Snapshots | -4.06385% +/- 0.08606% | -1.06540% +/- 0.02355% |
| Static SERTA + background | -3.34800% | -0.86089% |
| Dynamic quantum SERTA + background | -2.04497% | -0.51962% |
| Full quantum collision + background | -2.55701% | -0.65335% |

Errors are one standard error in percentage points. Halving s divides the
snapshot DC correction by 3.81; the deviation from four is resolved with the
paired samples and is consistent with higher-order corrections. The
second-order curvature calculation below separates these from sampling noise.

At s=0.125, static SERTA, dynamic SERTA, and full collision differ from snapshot
DC by only 0.207%, 0.552%, and 0.416% of the total conductivity. However, the
differences are 19.2%, 51.2%, and 38.7% of the EPC-induced correction.

The following compares complex AC corrections at s=0.125:

| Photon energy (eV) | Static SERTA error / abs(Delta sigma_snapshot) | Dynamic SERTA | Full collision |
|---|---:|---:|---:|
| 0.03 | 19.8% | 51.6% | 39.1% |
| 0.10 | 17.2% | 48.6% | 35.8% |
| 0.60 | 149.5% | 110.0% | 103.2% |
| 1.50 | 54.6% | 69.1% | 70.7% |

At 0.6 eV the total-response errors are just 0.0305%, 0.0224%, and 0.0211%.
Thus total-response agreement is not a sufficient optical EPC validation.
These are matched finite-supercell results, not fully converged bulk optical
corrections; the high-frequency correction retains appreciable supercell-size
dependence.

## Numerical controls and second-order snapshot reference

For independent Einstein displacements, the leading Gaussian average is

    <sigma> = sigma0 + (1/2) sum_i <u_i^2> d_i^2 sigma + O(u^4).

Primitive translation symmetry leaves only two inequivalent atomic
derivatives. `--curvature` evaluates these with centered finite differences
and multiplies by the number of cells. It removes Monte Carlo noise and
isolates the leading thermal correction of the complete frozen response.
It is not a Debye-Waller-only calculation; even a linear Hamiltonian produces
a nonzero conductivity curvature through state mixing and current vertices.

At s=0.25, 32 cells, 1024 electronic k points, the DC second-order snapshot
correction is -4.52840% (equivalently -1.13210% at s=0.125).
Halving the finite-difference displacement from 0.005 to 0.0025 A changes this
by 0.00017 percentage points. At 512 k points it is -4.52873%; increasing the
supercell from 32 to 64 at this grid changes it to -4.50395%. The leading DC
correction is consequently stable at roughly the percent level in these
checks. The Monte Carlo ensemble has sampling uncertainty and higher orders;
it is not substituted for the curvature reference.

Undisplaced supercell and primitive conductivities agree to about 6e-15
relative error. Primitive LG/VG agreement at 512 k points is about 1e-15.
The 1.5 eV baseline changes by 0.0021% between 512 and 1024 k points; low-energy
baselines are more tightly converged. Sum-rule agreement alone does not test
the convergence of the small EPC correction.

The rates have a separate numerical limit. At s=0.25, 40 meV phonons and
20 meV smearing, increasing q from 32 to 512 changes the static SERTA DC
correction from -3.3480% to -2.8704%, whereas full collision changes from
-2.5570% to -2.5457%. At q=512, reducing smearing from 20 to 5 meV changes full
collision to -2.9442%. A small q-grid dependence therefore does not establish
smearing convergence. Finite-smearing detailed-balance balancing also changes
the rate matrix; its norm change is saved in each metadata file. A broadening
cannot be decreased safely while keeping a coarse q grid fixed.

The same-balanced-rate diagonal RTA gives -1.4785% in `main`, versus -2.5570%
for full collision. This isolates the significant scattering-in contribution
from the separate finite-smearing rate-balancing change relative to raw SERTA.

## Isolating phonon dynamics

With quantum/Wigner statistics, changing phonon energy automatically adjusts
the masses to preserve <u_atom^2> at the selected temperature and scale. The snapshot distribution,
including its hopping and center fluctuations, is then exactly unchanged.
Only the phonon dynamics and quantum absorption/emission weights change.
For classical statistics the same mass convention is retained, so changing
phonon energy also changes the classical displacement variance.

The following refined DC calculations use s=0.25, 1024 k and q points, and
2.5 meV energy-conservation smearing:

| Phonon energy | Static SERTA | Dynamic SERTA | Full collision |
|---|---:|---:|---:|
| 40 meV | -3.03624% | -2.11474% | -2.98444% |
| 4 meV | -3.03624% | -3.01925% | -4.22055% |

Entries are DC changes relative to the bare background response. The frozen
second-order result is about -4.50% at either phonon energy. Slow phonons
therefore make dynamic and static SERTA nearly equal, while scattering-in
substantially improves the full result relative to the frozen reference.
This is a controlled trend, not a claim of exact equality or an exhausted
delta->0 / q->infinity limit. At 4 meV and q=k=512, reducing the smearing from
20 to 5 meV moves the full correction from -3.2549% to -4.0972%.

## Physical interpretation

* **Energy exchange and Pauli blocking.** The default 40 meV phonons are not
  small compared with kBT=25.85 meV or the 20 meV background linewidth.
  Frozen disorder is elastic, while quantum rates resolve absorption/emission.
  A Born-Oppenheimer separation by itself does not make long-time current
  correlations insensitive to this distinction. Zero-point frozen disorder
  likewise differs from quantum scattering at a low-temperature Fermi surface.
* **Momentum redistribution.** SERTA omits scattering-in. In this chain it
  contributes substantially to current relaxation, as shown by the comparison
  with diagonal RTA using exactly the same balanced rates. Full collision need
  not be closest to a static reference when it describes a different bath.
* **Optical vertices and spectral changes.** Snapshots change states, energies,
  hoppings, and orbital centers. Neither linewidth-only SERTA nor the present
  secular coherence extension of full collision contains all corresponding
  optical corrections. Holding snapshot centers fixed changes about 27% of
  the 0.6 eV EPC correction at s=0.125, but has negligible influence on DC.
  Real Fan shifts remain possible even with beta=0 and DW=0.
* **Background relaxation and probe frequency.** A larger background masks
  relative differences and cuts off long-time disorder dynamics. At s=0.25,
  the second-order 32-cell DC snapshot correction changes from -9.1645% to
  -4.5287% to -1.3145% as the background increases from 10 to 20 to 65.8 meV.
  High-frequency total-response agreement can conceal incorrect EPC changes.
* **Coupling, density and dimensionality.** Stronger disorder increases higher
  orders; carrier filling determines allowed processes and Pauli blocking.
  Static one-dimensional disorder can also produce coherent localization
  beyond Boltzmann transport at long times. The present finite-background
  study does not establish a localization regime or the zero-background DC
  limit. Fixed density would additionally require adjusting chemical potentials.

## Reproduction

    python examples/chain_transport_alignment.py --output tmp/chain_alignment/main --grid 512 --qgrid 32 --supercell 32 --pairs 200 --scale .25 --fixed-centers
    python examples/chain_transport_alignment.py --output tmp/chain_alignment/half --grid 512 --qgrid 32 --supercell 32 --pairs 200 --scale .125 --fixed-centers
    python examples/chain_transport_alignment.py --output tmp/chain_alignment/curvature_k1024_h0.0025 --grid 1024 --qgrid 32 --supercell 32 --pairs 0 --scale .25 --curvature --curvature-step .0025
    python examples/chain_transport_alignment.py --output tmp/chain_alignment/soft_fine_dc --grid 1024 --qgrid 1024 --pairs 0 --scale .25 --phonon-energy .004 --delta .0025 --omega 0

Use separate processes for large q-grid scans: dense-in-practice rate matrices
and cached population factorizations can be memory-intensive. One exploratory
multi-case large-grid process exhausted memory; the final refined DC runs
above completed independently. A 128-cell curvature scan was stopped after the
64-cell result to limit cost; no result from that unfinished run is used.

The implementation reuses the established conductivity calculators. No
collision or conductivity library algorithms were changed for this study.
