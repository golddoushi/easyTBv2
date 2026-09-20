# Moving-phonon optical conductivity: audit and chain example

Audit date: 2026-09-09. The existing entry point is
`examples/chain_transport_alignment.py` (singular `alignment`, in `examples`).

## Running the comparison

The new `--tdse` switch adds the moving-phonon `Opticalconductivity_TDSE`
calculator. For a small numerical benchmark:

```powershell
python examples/chain_transport_alignment.py --grid 16 --qgrid 4 --supercell 4 --pairs 0 --scale 0.125 --phonon-energy 0.02 --temperature 300 --background 0.026 --delta 0.05 --fermi -0.7 --omega 0.6 --tdse --tdse-trajectories 2 --tdse-steps 256 --tdse-max-phase-step 100 --tdse-transient-lifetimes 16 --tdse-phonon-cycles 1 --tdse-frequency-check --tdse-phase-points 17 --output tmp/chain_tdse_demo
```

This reproduces the small-model **time-integrator validation**, not a
bulk-converged conductivity. The large max-phase-step in this command lets
`--tdse-steps` control timestep refinement directly; the default 0.3 provides
an additional electronic/phonon frequency-resolution requirement.

TDSE uses the same supercell and folded electronic grid as the snapshots:
`Nk_supercell = grid/supercell`, and its phonon q grid is `supercell`.
Set `qgrid=supercell` when comparing with the other EPC schemes on the same
finite phonon mesh. The script reports intentional mismatches. TDSE cost grows
rapidly with supercell size; a 128-cell setting used for a rates-only run is
not an inexpensive TDSE setting.

Options:

- `--tdse-trajectories`: independent trajectories, default 4; the seed is
  the existing `--seed`. These are independent of `--pairs`, which controls
  static antithetic snapshots. Use `--pairs 0` to skip snapshots.
- `--phonon-statistics quantum|wigner|classical`: shared by every EPC method.
  Default `quantum` uses quantum collision rates and static variance, with
  Wigner trajectories as the TDSE approximation. Explicit `wigner` uses
  symmetric Wigner noise rates as well. `classical` reduces the variance in
  snapshots/static DM and trajectories together and uses classical noise rates.
  `--tdse-statistics` is a compatibility alias with the same shared scope.
  Neither trajectory option is a quantum detailed-balance bath. See
  [shared statistics](README_chain_alignment.md#controlled-model-and-normalization).
- `--tdse-steps`, `--tdse-max-phase-step`: timestep controls. The actual step
  count is the larger requirement and is saved in metadata.
- `--tdse-transient-lifetimes`: initial transient discarded, default 12
  background-relaxation times, rounded to complete optical cycles.
- `--tdse-measure-cycles`, `--tdse-phonon-cycles`: minimum measurement durations,
  default 2 optical cycles and 2 slowest-phonon periods. The longer requirement
  is used, rounded to complete optical cycles.
- `--tdse-phonon-damping`: default zero. Positive damping selects the existing
  periodically represented stationary Langevin spectrum. `--tdse-bath-cycles`
  and `--tdse-phase-points` then determine the sampled bath spectrum as well.
- `--tdse-frequency-check`: independently solve the same sampled trajectories
  with `Opticalconductivity_VG_DM_Phonon`, including the same electronic
  reservoir. Default off because this adds work. For the undamped chain's
  equal Einstein energies, `--tdse-phase-points` controls Fourier convergence.
- `--tdse-record-current`: save x-drive current traces; conductivity and
  per-trajectory results are always saved.

TDSE only runs at positive frequencies. A mixed `--omega 0 0.6` retains the DC
frequency-domain calculation and stores NaN in the TDSE DC slot. `--tdse` with
only zero/nonpositive frequencies fails before doing calculations.

## Outputs and comparison references

The existing `results.npz` and `metadata.json` now also contain:

- `tdse`, `tdse_samples`, `tdse_stderr_real`, `tdse_stderr_imag`: xxx optical
  conductivity in S/m, with independent-trajectory standard errors. One
  trajectory has undefined (NaN) error bars.
- `tdse_clean`: a zero-EPC moving-path TDSE calculation with the same time
  controls. All paths give the same clean result, so only one is propagated.
- `tdse_clean_dm` and `tdse_clean_minus_dm`: the clean supercell VG result
  and its time-discretization/transient discrepancy.
- `tdse_change = tdse - tdse_clean`: the EPC correction. Subtracting the
  independently propagated clean response avoids contaminating this small
  quantity with most of the pristine time-step error. The direct difference
  from clean DM is also retained as `tdse_minus_clean_dm`.
- `static_serta_vg`, `dynamic_serta_vg`, `balanced_rta_vg`: VG counterparts
  of the existing LG calculations, enabled with TDSE. Its comparison table
  uses VG throughout, including the pristine reference.
- `tdse_bath_static_rta`, `tdse_bath_dynamic_rta`: additional Born linewidth
  references using the trajectory's classical/Wigner statistics and q grid.
  They retain the background once and do not add a collision to TDSE itself.
  The dynamic noise-RTA source uses undamped phonon poles, even when damping
  is requested for TDSE. This limitation is reported explicitly.
- `tdse_frequency_carrier`, `tdse_frequency_lockin`, and per-trajectory
  lock-in values when the frequency check is enabled. The lock-in applies
  exactly the TDSE measurement times to the stationary spectral solution,
  including sideband leakage and the conjugate real-drive contribution.
- `tdse_change_minus_frequency_change`: compare EPC corrections, after each
  method's own clean reference is removed.

The clean LG/VG mismatch is saved as `baseline_vg_minus_lg`. EPC changes use
the corresponding gauge's baseline. This also corrects the old summary's
subtraction of an LG baseline from `full`, which is evaluated in VG. Snapshot
changes use the undisplaced supercell. `change_references` records the mapping.
Raw historical result arrays remain available.

## What the existing TDSE actually solves

In time units hbar/eV, the implemented equation is

\[
\dot\rho=-i[H(u(t),A(t)),\rho]
 -\Gamma_{\rm bg}\{\rho-f_{\mu,T}[H(u(t),A(t))]\}.
\]

It is a one-particle ensemble density matrix with a prescribed classical or
Wigner-sampled harmonic trajectory. Ionic positions and momenta are sampled;
the ensuing harmonic motion is evaluated analytically. There is **no
electronic force on the ions**. Thus it is neither self-consistent Ehrenfest
dynamics nor a propagation of the quantum electron-plus-phonon state.

The propagator evolves the zero-field density and its exact derivative with
respect to electric-field amplitude. It does not solve a frequency-domain
response equation. E0 only scales recorded field units on this moving-phonon
path: this is an infinitesimal-field linear response, not a finite-amplitude
nonlinear propagation.

At each midpoint it diagonalizes the actual instantaneous displaced
Hamiltonian. That eigensolve is needed for the time-dependent propagator and
Fermi target; the stationary eigensystem alone cannot replace it. The step
combines exact frozen-H unitary evolution and scalar relaxation. Linearizing
the exponential and Fermi matrix includes both the field-induced unitary
response and the derivative of the reservoir target. A sinc expression makes
the exponential derivative finite at degeneracies. The midpoint treatment of
time-dependent coefficients is second-order accurate in dt.

The current includes both the density response and the Peierls contact
vertex. Fourier extraction uses the peak-field convention
`2 mean(j_linear exp(i omega t))`. Currents and hopping vertices evolve in a
**fixed equilibrium point-orbital basis**. This differs from the default
snapshots, which move orbital centers. `--fixed-centers` exposes the
hopping-only snapshot comparison.

The reservoir term is a physical approximation, not merely a plotting
broadening. Its instantaneous fixed-mu target generally exchanges particles
and energy with the electrons as the lattice moves. It establishes a steady
driven state but does not implement a microscopic quantum EPC thermalization
mechanism. Explicit EPC trajectories must not have SERTA added on top; the
calculator already rejects this double counting.

## Numerical audit

No error was found in the tested midpoint propagation, field linearization,
current/contact vertices or normalization. The core propagation algorithm was
left unchanged. The example-reference issue described above was corrected.

The tests `test_phonon_tdse.py`, `test_phonon_frequency.py`, and
`test_chain_transport_tdse.py` passed all **16 checks**. These cover finite-field
step differentiation, a newly added independent continuous-time finite-field
ODE reference with a moving Hamiltonian, second-order time convergence,
Hermiticity and zero-field occupation bounds, harmonic bath statistics,
Peierls derivatives, zero-EPC recovery, supercell folding, same-trajectory
frequency/time agreement for undamped and damped baths, and the new CLI.

The independent ODE reference integrates finite positive/negative fields with
adaptive DOP853, then differences the resulting density matrices. It uses
neither the TDSE Frechet derivative nor a DM response kernel.

For the 4-cell, 16-primitive-k-point command above, the stationary spectral EPC
correction is approximately **34.09733 - 6.14867 i S/m**. Comparing this with
the TDSE correction gives:

| Steps per optical period | Absolute error in complex EPC correction (S/m) | Relative error |
|---|---:|---:|
| 128 | 0.00340594 | 0.0098303% |
| 256 | 0.00085304 | 0.0024621% |

Halving dt reduces error by a factor of about four. Increasing the bath
Fourier phase points from 17 to 33 changes the matched per-trajectory lock-in
conductivity by less than 1.8e-11 S/m. The maximum phase-33 linear-solve
relative residual is 8.5e-10. Data are in `tmp/chain_tdse_audit_steps128/`,
`tmp/chain_tdse_audit_steps256/`, and `tmp/chain_tdse_audit_summary.json`.

This validates two solvers of the **same specified stochastic equation**.
It does not validate that equation against quantum EPC. Two trajectories are
also insufficient for a precise thermal ensemble average. The deliberately
small electronic grid has a large pristine LG/VG mismatch (imaginary part
about 1.48e4 S/m); at grid=256 that difference falls below 1e-6 S/m for these
parameters. No bulk-converged SERTA-versus-TDSE alignment is inferred from
this integrator test.

## Suitability for crystalline EPC

The approach can model nonadiabatic electronic response to classical lattice
fluctuations, including finite-q mixing in a supercell, time-dependent optical
vertices and repeated coherent scattering. It is useful as a controlled
benchmark for a prescribed stochastic lattice and, with suitable convergence,
for a high-temperature semiclassical approximation to a solid.

However, a real classical harmonic path has a symmetric noise spectrum.
Wigner sampling reproduces the symmetrized quantum displacement correlation,
not the ordered quantum correlation:

\[
\langle\hat u(t)\hat u(0)\rangle
=u_{\rm zp}^2[(N+1)e^{-i\Omega t}+Ne^{i\Omega t}],
\qquad
\langle u(t)u(0)\rangle_{\rm Wigner}
=u_{\rm zp}^2(2N+1)\cos\Omega t.
\]

The missing imaginary/antisymmetric part supplies the unequal quantum
emission and absorption weights. Zero-point Wigner motion therefore does not
restore quantum detailed balance. At 300 K and 20 meV, their quantum weight
ratio is about 2.17, rather than one; this is not a parametrically classical
limit. The asymmetry becomes relatively less important when the relevant
phonon energies are much smaller than kBT, although other approximations and
transport convergence still need testing.

The current bath has Einstein oscillators, no acoustic dispersion or sum
rule, no electronic backreaction/phonon drag/hot-phonon evolution, and no
electron-phonon entanglement. The fixed centers and background reservoir are
additional assumptions. All q points having the same frequency does not
eliminate q-dependent scattering matrix elements.

Consequently, this TDSE is a sound numerical reference for the specified
semiclassical bath; it should not be labeled a general quantum-SERTA or
phonon-limited-solid benchmark. The paper previously discussed develops
ECEID specifically to retain electron-ion correlations and recover
thermalization beyond Ehrenfest; that method is not implemented by these
trajectories. See [Rizzi et al., PRB 93, 024306](https://arxiv.org/abs/1511.03875)
and the quantum Fan/BTE formulas in [EPW theory](https://docs.epw-code.org/Theory.html).

For a controlled quantum benchmark, a future small electron-plus-quantized-
phonon calculation with a converged phonon occupation cutoff, or an explicitly
quantum correlation-based dynamical method, would address a different and
necessary validation level. Adding electronic mean forces alone would not
supply the missing quantum correlations.

## Selected-mode extension

The chain benchmark now supports shared wavevector/branch/weight selection in
TDSE, its frequency-domain check and the other approaches. See
[selected phonon modes](README_chain_selected_phonons.md). The audit above
describes the original full-bath runs; their physical limitations also apply to
selected-mode trajectories.
