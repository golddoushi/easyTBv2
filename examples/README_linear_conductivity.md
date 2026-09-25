# Linear conductivity: LG, VG, and real time

See [collision preparation and reuse](README_collision_preparation.md) for the
explicit collision stage, shared LG/VG rates, and coupled-mesh task scheduling.

The new calculators exported by `calculator` are:

| Calculator | Output `sigma` axes | Method |
|---|---|---|
| `AnormalousHallConductance_VG_DM` | Fermi, drive, current | Analytic dc limit of first-order VG |
| `Opticalconductivity_LG_DM` | Fermi, photon energy, drive, current | First-order LG |
| `Opticalconductivity_VG_DM` | Fermi, photon energy, drive, current | First-order VG with contact current |
| `Opticalconductivity_TDSE` | Fermi, photon energy, drive, current | Real-time VG ensemble propagation |

Here “optical conductivity” means the finite-frequency linear conductivity,
not the second-order dc photocurrent calculated by the BPVE classes. `sigma` is in
S/m; the optical result is complex. `projected_sigma` has two leading
projection axes, with `[0,0]` the total response. The last two axes follow
the existing dc API: electric-field direction first, current direction second.
Use `(sigma[...,0,1]-sigma[...,1,0])/2` for the antisymmetric xy component.
The 2D model uses its three-dimensional cell volume including vacuum; multiply
S/m by the cell height in metres for sheet conductance.

## Example

```python
import numpy as np
from calculator import (Opticalconductivity_LG_DM, Opticalconductivity_VG_DM,
                        Opticalconductivity_TDSE, CollisionSpecification)

options = dict(model=model, kgrid=(24,24,1), kBT=0.15,
               collisionSpecification=CollisionSpecification('crta', constantLinewidth=.2),
               fermiList=np.array([-0.7]), omegaList=np.array([0.6,1.3]),
               fullCovariantDerivative=True)
lg = Opticalconductivity_LG_DM(**options)
vg = Opticalconductivity_VG_DM(**options)
# Use a small grid initially; compare TDSE with VG on that same grid.
small = dict(options, kgrid=(4,4,1))
reference = Opticalconductivity_VG_DM(**small)
td = Opticalconductivity_TDSE(E0=1e-4, n_steps_per_cycle=256, **small)
```

Run the standalone benchmark:

```text
python examples/honeycomb_conductivity.py --electronic-k-grid 32 --clamped-tdse-k-grid 4
python examples/honeycomb_conductivity.py --electronic-k-grid 32 --skip-tdse --snapshots 20 --phonon-supercell 2
python examples/honeycomb_conductivity.py --electronic-k-grid 24 --serta both --skip-tdse
python examples/honeycomb_conductivity.py --electronic-k-grid 12 --serta static --snapshots 200 --phonon-supercell 2 --skip-tdse
```

It writes `tmp/honeycomb_conductivity/summary.json` and `conductivity.npz`.
An example run with the defaults above gave relative maximum errors of
`7.2e-16` for dc LG/VG, `7.8e-7` for optical LG/VG, and `9.7e-5` for
TDSE/VG on the common 4x4 mesh. These are comparisons at the chosen meshes,
temperature, broadening, and time step, not universal accuracy guarantees.

`task.py` also exports `solveAHC_VG_DM`, `solveOpticalconductivity_LG_DM`,
`solveOpticalconductivity_VG_DM`, and `solveOpticalconductivity_TDSE` for
`BZGridTask`. Use the existing `fermi`, `omega`, `collisionSpecification`,
`record fermi ids`, and `record frequency ids` parameter keys. TDSE controls
use their Python argument names. Use `BZSymm=False` and average the phase-grid
results, preserving their complex values.

## Conventions and the dc sign correction

Photon energies `Omega` and linewidths are in eV. The Fourier convention is
`exp(-i*Omega*t)` with time in hbar/eV. The physical electron Hamiltonian is
`H(k+A)`, electric field is `E=-dA/dt`, and charge current is proportional to
`-Tr[rho*v(A)]`. Every kernel returns the **physical density matrix**.
Charge-current calculators supply the minus sign explicitly; generic
observable contractions always evaluate `Tr[rho*O]`.

For `Delta_mn=E_m-E_n` and `Df=f[E_m,E_n]*v`, the first-order electric-field
density is `-i*Df/(Delta-Omega-i*Gamma_mn)`. Contracting with `-v` gives the Drude
conductivity `(-f')*v_a*v_b/(Gamma_n-i*Omega)`. The full finite-temperature
kernel uses the stable divided difference, including the `f'` limit at
degeneracy, rather than dropping small energy gaps. I/E denote diagonal and
off-diagonal contributions; only the total is invariant under arbitrary
rotations of exactly degenerate bands.

The September 12 sign migration removes the former negative-density/positive-
current compensation. Both dc density channels and the optimized generic
contraction now have the physical sign. Linear charge conductivities, including
their records, remain unchanged because the charge-current minus moved to the
calculator. Direct spin/orbital/torque galvanic responses also remain unchanged.
DM spin Hall results reverse sign: `spinVelocity_K` is the magnetic-moment
current `{mu_S,v}/2`, not a charge current or an angular-momentum current.
The separate historical Berry-curvature calculators retain their existing
conventions. When comparing Hall signs, also account for drive/current axis
ordering. The dc LG kernel retains its small-gap cutoff; the full optical
kernels retain the finite degenerate limit.

The VG contact `+i*residue/Omega` retains its sign: it already combines `j=-v`
with `E=i*Omega*A`. See [the sign migration](README_density_matrix_signs.md)
for second-order densities, BPVE and torque vertices.

## Why the VG contact and relaxation terms matter

The VG equilibrium must transform with the field, and the observable includes
the diamagnetic contact term. This requirement, including its finite-band
formulation, is discussed by [Passos et al., Sections III–V and Appendix A](https://arxiv.org/html/1712.04924).
Within a consistently coupled finite tight-binding model, a finite number of
bands by itself does not force the gauges to disagree.

In this implementation the physical density per vector potential is

```text
R(Omega) = -([v,f] + i*Collision(Df)) / (Delta-Omega-i*Gamma)
sigma    = (i/Omega) * BZ_average(Tr[R*v_current] + Tr[f*w_current,drive])
```

Here `w[b,a]=D_a v_b` is the existing covariant velocity derivative. The
equilibrium residue `Tr[Df*v+f*w]` integrates to zero over a periodic BZ.
For numerical stability the kernel analytically evaluates
`(R(Omega)-R(0))/Omega`, and the calculator restores the residue separately.
The `C` channel is this `i*residue/Omega` contribution. It is retained at
nonzero frequency, so mesh errors are observable. `sum_rule_residual` stores
the integrated residue in (S/m)*eV. At exactly zero frequency the equilibrium
pole is subtracted analytically, giving the periodic-BZ dc limit; this avoids
an arbitrary tiny probe frequency. Consequently dc LG/VG agreement alone is
not an independent sum-rule convergence test. Check finite-frequency results
and the residual as well, especially near zero frequency.

The DM variants use `CollisionSpecification` for CRTA or frozen secular SERTA.
VG requires the full covariant response, positive electronic temperature,
an orthogonal model, and the complete model basis (`skip_low=False`). Its
contact vertices currently support total current only; LG also supports the
existing current projections. Optional DM `projected_response` records remain
unscaled k-resolved integrands, matching the dc convention.

## Time-domain benchmark and EPC scope

The TDSE variant does not call a response kernel. Each time step propagates
the occupied ensemble with a unitary exponential of
`H0+A*v+A^2*w/2`, together with relaxation toward the independently evaluated
spectral function `f(H(A))`. This is a density ensemble of TDSE trajectories
with relaxation, not isolated pure-state evolution. CRTA gives an exact
frozen-H step; SERTA uses symmetric splitting with frozen equilibrium-band
linewidths. These approximate a midpoint time step with second-order time
accuracy. `E0` is the **peak** field in V/Angstrom; it is not the positive-
frequency amplitude used in the older BPVE TDSE class.

After `transient_lifetimes` relaxation times, the code measures an integer
number of cycles. Opposite fields remove even-order current terms, and
`2*mean(j_odd*exp(i*Omega*t))/E0` gives the complex conductivity. The current
vertex is `v+A*w`; the finite-field Hamiltonian/current Taylor expansion is
intended for convergence to **first order**, not strong-field dynamics.
Refine `n_steps_per_cycle`, reduce `E0`, and increase the discarded transient
to test convergence. `max_phase_step` can increase the requested step count;
inspect `integration_info`. Steady-state extraction requires positive photon
energies and strictly positive relaxation linewidths. It deliberately rejects
dc and undamped cases. `recordCurrent=True` stores time, field, and current
divided by the peak field in S/m for each `(fermi,omega,drive)`.

The optional snapshot example averages displaced SSH supercells at fixed CRTA
eta and reports the equilibrium supercell result plus separate real/imaginary
sampling errors. It is a **static disorder benchmark**, not an implementation
of inelastic EPC transport. Two snapshots are only a smoke test. Converge the
ensemble, supercell, k mesh, and residual broadening before interpreting EPC
changes.

## Toy-EPC SERTA linewidths

`ModelEPCLinewidth` turns the diagonal on-shell imaginary Fan self-energy into
a sampled linewidth source:

```python
from phonon import PhononBath
from calculator import CollisionSpecification, ModelEPCLinewidth

source = ModelEPCLinewidth(
    epc,
    fermi=-0.7,
    delta=0.02,
    adiabatic=False,
    phononBath=PhononBath(qgrid=(12, 12, 1), temperature=300),
)
collision = CollisionSpecification("serta", linewidthSource=source)
sigma = Opticalconductivity_LG_DM(**dict(options, collisionSpecification=collision)).sigma
```

It evaluates the calculator's k mesh and caches the result for LG, VG, and TDSE
instances using the same unchanged electronic/EPC model pair and mesh. A shifted `BZGridTask` supplies its
`kOffset`. `adiabatic=False` is standard dynamical SERTA: the linewidth is
`-2 Im Sigma_Fan(epsilon_nk)` with phonon emission/absorption energies and Pauli
factors, as in the [VASP SERTA definition](https://vasp.at/wiki/ELPH_SCATTERING_APPROX).
`adiabatic=True` gives an elastic Born rate for static Gaussian displacements.
We call this **static/adiabatic RTA**, since “SERTA” normally refers to the
dynamical on-shell Fan rate. Its Lorentzian `eta` approximates the elastic
delta function and must be converged together with the q mesh.

The source supplies only Gamma. The transport Hamiltonian, energies,
occupations, and current vertices remain bare. It does not add Re(Fan)+DW or
optical vertex/scattering-in corrections.

## Frozen supercells versus the two rates

Agreement of static Fan+DW **band-energy shifts** with stochastic snapshots
does not imply equal conductivity. That comparison concerns a one-particle
pole shift through second order. Conductivity is also sensitive to eigenstates,
current vertices, and how scattering redistributes momentum.

A frozen calculation evaluates the full independent-electron conductivity of
each `H[u]` and then averages. It contains real spectral changes, modified
eigenvectors/current vertices, elastic state mixing allowed by the supercell,
and all displacement orders in `H[u]`. Its CRTA `eta` is an additional residual
broadening. Static/adiabatic RTA keeps only the elastic Born **out-scattering
lifetime** in a diagonal secular collision operator. It omits scattering-in
and vertex corrections. They therefore are not formally equal.

They may approach one another for weak EPC, large supercells, and slow classical
phonons after matching displacement covariance, k/q resolution, chemical
potential, and broadening, if vertex corrections are negligible. Dynamical
SERTA resolves phonon energy exchange and Pauli blocking. A static snapshot
cannot reproduce these, except approximately when `kBT >> hbar*Omega` and the
electronic probe cannot resolve the phonon dynamics.

The example now makes meshes commensurate: an `S x S` supercell uses
`(grid/S) x (grid/S)`, sampling the same primitive momenta. The undisplaced 2x2
supercell then agrees with the primitive 12x12 tensor to about `5.9e-15`
relative error. At 300 K, 200 frozen 2x2 snapshots differed from pure static
RTA by about 38% under the report normalization. That raw comparison also
replaces the snapshots' 0.2 eV CRTA background by the EPC rate. Adding the same
0.2 eV background to static RTA reduces the discrepancy to about 10.8%. The
largest componentwise sampling standard error was about 1.6% of the response,
and the raw discrepancy reached roughly 51 standard errors. The example stores
both comparisons. This finite benchmark rejects automatic equality while also
showing that unmatched background broadening can dominate the apparent error.

`--tdse-collision static|dynamic` makes the time-domain comparison use model
EPC rates. Small linewidths require very long transients, so this optional path
applies `--tdse-min-linewidth 0.02` eV by default and reports the actual minimum.
Set it to zero for unmodified SERTA and converge the transient. DM rates are
never floored.

A diagram-by-diagram snapshot comparison requires a static-disorder collision
kernel with scattering-in/vertex terms, or an explicit second-order expansion
of ensemble conductivity. The present comparison diagnoses the diagonal
lifetime approximation; it does not establish it as the exact frozen-phonon
limit.

## Moving harmonic phonons

`Opticalconductivity_TDSE(phononDynamics=True, ...)` propagates electrons in
a prescribed harmonic SSH bath. Enable the supercell example with:

```text
python examples/honeycomb_conductivity.py --phonon-dynamics --serta none --omega 1.3 --phonon-trajectories 4
```

The default `--phonon-supercell 4` has 64 electronic states and 64 in-plane phonon
degrees of freedom. `--supercell-k-grid 1` samples the supercell at Gamma. Its
frequency-domain comparisons use primitive electronic and phonon meshes
`(4,4,1)`. More generally, supercell S and reduced electronic grid K correspond
to a primitive electronic grid `(S*K,S*K,1)` and phonon grid `(S,S,1)`.
`--electronic-k-grid`, `--clamped-tdse-k-grid`, and `--serta-q-grid` control separate
calculations; they do not override these matched moving-phonon grids.
`--skip-tdse` and `--phonon-dynamics` are mutually exclusive.

With the default `--phonon-damping 0`, each ion follows
`u(t)=x*cos(Eph*t)+y*sin(Eph*t)`, with independent Gaussian quadratures x and
y, so both its initial position and velocity are thermal. Time is in hbar/eV;
velocity is in Angstrom/(hbar/eV). This integrates the Einstein oscillator
exactly. There is no electronic backreaction or quantized joint
electron-phonon wavefunction. The orbital
basis and centers stay at their equilibrium reference: SSH changes hopping
amplitudes, including optional quadratic beta, while Rashba and exchange
remain fixed. Velocity and contact matrices are differentiated from the
instantaneous hopping Hamiltonian. Moving-basis/convective electronic
current and ionic current are outside this hopping-only model.

The electronic algorithm linearizes the exact midpoint TDSE step. It
propagates the zero-field density and three field derivatives using one
eigensolve per timestep; it uses no optical DM frequency kernel or fitted EPC
linewidth. `--optical-field` only rescales the saved drive trace in this linear path.
The scalar `--background-linewidth` is a background relaxation toward
`f(H[u(t),A(t)])`; it must be positive. The corresponding DM comparisons add
the same background to their EPC linewidths. Adding `--tdse-collision dynamic`
or `static` to moving-phonon propagation is rejected to avoid double counting.
The background also controls electronic heating and the stationary ensemble;
reduce it only while increasing transient time and checking the result.

`--phonon-statistics classical` is the default: the mode noise has equal
positive/negative-frequency weights `C=kBT/Eph`, without zero-point motion.
`--phonon-statistics wigner` uses `C=N_Bose+1/2`, reproducing the harmonic
symmetrized correlation but not quantum detailed balance. The frequency-domain
reference with matching statistics uses

```text
Sigma_nn(E) = sum_q,nu,m w_q |g_mn|^2 C_qnu *
  [1/(E-e_m-Eph+i*delta) + 1/(E-e_m+Eph+i*delta)].
```

Both have DW factor `2*C`. Standard quantum Fan/SERTA instead has pole
weights `N+1-f_m` and `N+f_m`; it is saved separately as a diagnostic.
Classical/Wigner sources use the existing diagonal RTA collision machinery,
but their bath statistics differ from quantum SERTA. The default quantum
self-energy API remains unchanged. `--epc-delta` regulates the Born sum and
has no literal counterpart in the finite trajectory; converge it jointly
with mesh resolution rather than interpreting it as phonon damping.

Matching k/q meshes removes one source of inconsistency, not every difference:
trajectory conductivity retains changing spectra/vertices and repeated
scattering, while a linewidth-only RTA omits these and scattering-in terms.
The bath-driven steady electronic state also need not equal Fermi equilibrium.
Equality is therefore a hypothesis to test under weak-coupling/classical and
convergence limits, not an imposed pass criterion. A fully quantum benchmark
would require a quantum bath method. See
[Rizzi et al., electron-phonon thermalization](https://arxiv.org/abs/1511.03875)
for the detailed-balance limitations of classical electron-phonon dynamics.

`--phonon-cycles 1` enforces a measurement window of at least one slowest
phonon period, rounded up to whole optical periods. `--measure-cycles 2` is
an additional optical-period minimum. The transient uses
`--transient-lifetimes 12`. These are starting settings, not convergence
claims: increase duration, trajectories, supercell, and steps separately.
At 0.02 eV a phonon period is about 207 fs, so following moving phonons takes
many more steps than a short optical-only trace. The same sampled phonon
histories are used for all field directions and frequencies.

`conductivity.npz` stores the trajectory tensors, complex ensemble mean,
separate real/imaginary standard errors, electronic current traces, initial
ionic positions/velocities, and oscillator energies (sufficient to reconstruct
the entire ionic motion). A single trajectory has undefined standard error
and stores NaN. `summary.json` records both meshes, bath assumptions, actual
integration lengths, matched-bath RTA discrepancy, and quantum-SERTA
discrepancy. The seed is controlled by `--phonon-seed`.

## Frequency-domain counterpart of the moving-ion TDSE

For the stationary Pauli collision approximation beyond SERTA, including
`--epc-markov`, see [the Markov EPC documentation](README_epc_markov.md).

`Opticalconductivity_VG_DM_Phonon` solves the same prescribed harmonic-bath
model as `Opticalconductivity_TDSE(phononDynamics=True)`. It is now included
automatically in the `--phonon-dynamics` example. For instance:

```text
python examples/honeycomb_conductivity.py --phonon-dynamics --serta none --omega 1.3 --phonon-trajectories 2 --tdse-steps-per-cycle 128 --phonon-phase-points 33
```

The moving-ion implementation is a classical-path approximation
with a phenomenological electronic reservoir. It is not self-consistent
Ehrenfest dynamics, since electronic forces never change the ionic paths,
and it is not the ECEID algorithm of Rizzi et al. Adding a scalar damping
rate does not restore the quantum electron-phonon correlations in ECEID.
The code's continuum electronic equation, in hbar=1 units, is

```text
d rho/dt = -i[H(u(t),A(t)),rho] - gamma*(rho-f(H(u(t),A(t)))).
```

Thus the target of background relaxation itself changes with phonons and
the optical field. Replacing the bath by a linewidth on the bare bands
misses this effect as well as correlations between density, hopping, and
current vertices. A 50% discrepancy cannot be assigned solely to omitted
scattering-in without checking these other differences.

Finite phonon lifetime is enabled with, for example:

```text
--phonon-damping 0.005 --phonon-bath-cycles 8 --phonon-phase-points 33
```

Here `phonon-damping` is the amplitude-correlation decay rate kappa in eV for
`u''+2*kappa*u'+omega0^2*u=noise`. A bare factor `exp(-kappa*t)` is not used,
because it would cool the ions to zero. Instead, the code samples a stationary
Gaussian Fourier series from

```text
S_u(nu) proportional to 1/[(omega0^2-nu^2)^2+(2*kappa*nu)^2]
```

and normalizes it to the requested classical or Wigner mean-square
displacement. This is the fluctuation-dissipation-consistent classical shape;
the finite Fourier series is periodic, with a period of
`phonon-bath-cycles` slowest-mode periods. Increase both that period and the
odd `phonon-phase-points` until the correlation function and conductivity are
stable. The saved cosine and sine coefficients reconstruct the complete path.

For equal Einstein energies and zero damping, each sampled harmonic trajectory is periodic
in theta=nu*t, even though different ions have independent phases. Define
H(theta), v_a(theta)=dH/dA_a, and w_ba(theta)=d2H/dA_b dA_a. The new solver
first obtains the periodic zero-field stationary density, then the response
R_a(theta;omega) to a complex vector potential exp(-i*omega*t):

```text
L(omega) = nu*d/dtheta + gamma - i*omega + i[H(theta), .]
L(0) rho0(theta) = gamma*f(H(theta))
L(omega) R_a(theta) = -i[v_a(theta),rho0(theta)]
                    + gamma*D f(H(theta))[v_a(theta)]
sigma_ab(omega) = i/omega * average_theta,k,samples
                 Tr(R_a(theta)*v_b(theta) + rho0(theta)*w_ba(theta))
                 * [usual conductivity unit/volume factor].
```

`D f(H)[v]` is the Frechet derivative with its finite degenerate-band limit.
The current includes the instantaneous contact vertex. This is a true
frequency-domain density-matrix solve: Fourier differentiation in phonon
phase and matrix-free GMRES solve the coupled harmonic linear systems; there
is no electronic time propagation, optical response kernel reuse, or fitted
EPC rate. All matrix elements and harmonics within the chosen phase grid are
retained, including repeated scattering, quadratic SSH modulation, and the
periodic background density (which generally is not instantaneous f(H)).

This can also be viewed as a memory collision operator. Split harmonic space
into the central sector 0 and sidebands Q, then eliminate Q:

```text
L_eff(omega) = L_00(omega) - L_0Q(omega) L_QQ(omega)^(-1) L_Q0(omega).
```

The eliminated sector also modifies the drive and current vertices. The
result is frequency dependent and nonsecular, unlike the current array of
population/coherence linewidths. The new `HarmonicPhononLiouvillian` operates
in the enlarged conditional-DM space; it is intentionally not presented as
another scalar mode of `CollisionSpecification`. The only phenomenological
collision retained is the same positive CRTA background used in TDSE.

`sigma` and `sigma_samples` are central complex-drive coefficients.
`conductivity_harmonics` retains the phonon sidebands. A finite-duration
real-drive TDSE extraction also contains sideband leakage and, at coincident
frequencies, conjugate-drive contributions. `lockinConductivity(times, ...)`
reconstructs only the scalar spectral current and applies the exact same
measurement times as TDSE. It performs no dynamical evolution. The example
stores both quantities:

- `sigma_ac_phonon_VG_DM`: central ensemble conductivity.
- `sigma_ac_phonon_VG_DM_lockin`: spectral solution under the TDSE window.
- `frequency_TDSE_relative_difference`: comparison using that same window.
- `frequency_TDSE_carrier_relative_difference`: central coefficient versus
  the finite TDSE extraction, which also tests measurement convergence.
- `frequency_solver_max_relative_residual`: numerical linear-system check.

The original RTA and quantum-SERTA discrepancy fields remain, now as tests
of those approximations against an independently checked model response.
Using identical samples in TDSE and frequency calculations cancels sampling
noise in their implementation comparison; estimating the physical ensemble
conductivity still requires trajectory and supercell convergence.

`--phonon-phase-points` must be odd (default 33); increase it, e.g. to 49 or
65, to resolve sharper phase dependence at lower temperature, stronger EPC,
or a longer damped-bath period. `--phonon-solver-tolerance` controls the
linear-system residual, not harmonic truncation. Failed solves raise an error.
The undamped one-phase implementation requires equal active Einstein energies.
The damped periodic representation puts distinct mode spectra on one common
Fourier grid and therefore accepts unequal energies, subject to its frequency
resolution check. A positive electronic background is required for a unique
stationary solution. Quantum detailed balance and electronic force feedback
remain outside this benchmark in both domains.

On the saved 4x4x1, two-trajectory, 300 K, omega=1.3 eV, gamma=0.2 eV run,
the 33-point frequency solve took about 7 s. Its maximum in-plane discrepancy
from the original 128-step TDSE, under the identical measurement window, was
0.0251%. Increasing to 49 phase points changed the central tensor by
1.8e-11 relative. Small-system timestep tests show the remaining time-domain
error falls quadratically with timestep. These numbers validate agreement
between the two representations of this model; they do not validate its
classical bath as a quantum EPC approximation.

The TDSE cycle rounding now preserves exactly commensurate periods: at
omega=1.3 eV and nu=0.02 eV, one phonon period is 65 optical cycles, not 66
from floating-point ceil noise. With that correction the central coefficient
and matched-window comparison both differ from TDSE by at most 0.0251% in
the in-plane components. The full rerun took 6.8 s for the frequency solve
and 66.5 s for TDSE; matched classical RTA and quantum SERTA still differ by
up to 56.1% and 55.4%, respectively. The same trajectories were used in all
time/frequency implementation checks.
