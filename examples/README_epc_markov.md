# Fixed-number Markov EPC collision

`CollisionSpecification('epc', ...)` binds `CollisionOperator_full` to the
calculator's existing Energy eigensystem. It is a fixed-number secular Pauli
approximation with population scattering-in, not a full non-Markov optical
collision kernel. The name `full` remains an alias for compatibility.

The `epc` object explicitly pairs the electronic and phonon models; see
[model construction](README_epc_models.md).

```python
collision = CollisionSpecification(
    'epc', qgrid=(4, 4, 1), bathStatistics='quantum',
    phononTemperature=300., delta=.03, constantLinewidth=.2,
    populationSolver='auto', solverTolerance=1e-10,
    stationaryTolerance=1e-12, enforceDetailedBalance=True,
)
calc = Opticalconductivity_VG_DM(
    model=model, epcModel=epc, kgrid=(16, 16, 1), kBT=.025852,
    fermiList=np.array([-.7]), omegaList=np.array([1.3]),
    fullCovariantDerivative=True, collisionSpecification=collision,
)
state = calc.density_matrix_kernel.collision.forFermi(-.7)
print(state.stationary_info)
```

The electronic `kBT` must match `phononTemperature` when detailed balance is
enabled. The complete orthogonal model basis is required. Each q-grid dimension must
divide the corresponding electronic k-grid dimension. Use the complete
coupled k mesh in one calculator; independent shifted-grid task partitions do
not reproduce a collision connecting states across those partitions. `BZGridTask`
now schedules EPC as one complete `BZGrid` task, even when more workers are
requested. See [collision preparation and reuse](README_collision_preparation.md)
for sharing rates across response calculators. Projected
observables are supported where the existing calculator supports them, but
the electric drive must use the total velocity vertex.

The honeycomb example exposes the calculation alongside CRTA/SERTA:

```text
python examples/honeycomb_conductivity.py --electronic-k-grid 16 --skip-tdse --serta none --epc-markov quantum --markov-q-grid 4 --omega 1.3 --background-linewidth .2
```

For comparison with the prescribed moving-phonon benchmark:

```text
python examples/honeycomb_conductivity.py --electronic-k-grid 4 --phonon-dynamics --serta none --epc-markov classical --markov-q-grid 4 --phonon-supercell 4 --omega 1.3 --background-linewidth .2 --phonon-trajectories 2 --tdse-steps-per-cycle 128
```

The latter stores `sigma_ac_VG_phonon_Markov` on the same folded electronic
and phonon meshes as TDSE. `markov_relative_difference` uses the TDSE component
as denominator. Add `--epc-raw-rates` to inspect the unbalanced Lorentzian
rates. Equal meshes do not make the two physical approximations equal.

## Stationary state and nonlinear equation

Write R[i,j] for a transition from state j to state i, with rates in eV and
time in hbar/eV. Finite Lorentzian tails do not obey thermal detailed balance
exactly. By default each unordered pair retains its raw total activity
`S_ij=Rraw[i,j]+Rraw[j,i]`, while its two directions are replaced by

```text
R[i,j] = S_ij * expit(-(energy[i]-energy[j])/kBT)
R[j,i] = S_ij - R[i,j]
```

Consequently `R[i,j]/R[j,i]=exp(-(energy[i]-energy[j])/kBT)` and every Fermi
distribution at that temperature is an exact stationary state. The nonlinear
fixed-number equation is

```text
C_ep[p] = (1-p)*(R p) - p*(R.T (1-p))
dp/dt = C_ep[p] - gamma*(p-f_mu[p])
```

Here `f_mu[p]` is the Fermi distribution whose total occupation equals
`sum(p)`. Both the EPC and BGK terms therefore conserve particle number. The
response solver projects sources onto the trace-free tangent space and uses a
bordered sparse system to impose `sum(delta_p)=0`. With balanced rates,
`forFermi(mu)` uses the requested Fermi state directly. With
`enforceDetailedBalance=False`, a bounded, fixed-number Newton solve finds the
stationary state of the raw rates.

For quantum phonons, pair balancing repairs the thermodynamics lost through
finite energy broadening. Applying it to classical or Wigner rate activities
produces a thermally balanced surrogate. The spectrum of an undamped real
classical trajectory is symmetric and does not itself satisfy finite-temperature
quantum detailed balance; use raw rates when that distinction is under study.

For a general density matrix, the diagonal follows the Pauli equation while
the off-diagonal collision is

```text
C[rho]_ij = -(ell_i(p)+ell_j(p))*rho_ij/2,  i != j
ell(p) = gamma + R p + R.T (1-p)
```

`collision_integral(rho)` evaluates this closed fixed-number model. `apply(x)`
returns -D C[rho0] x. `second_variation(x,y)` returns the symmetric bilinear
D^2 C[rho0][x,y], with no conjugation and no factor 1/2. It includes population
Pauli terms and population/coherence cross terms. The ordered (-omega,+omega)
second-order response includes this source; multiplying a real cosine field
of peak amplitude E yields a dc density E^2*rho2/4.

All EPC response clients require `fullCovariantDerivative=True`. First-order
AHC/SHC/optical kernels use p0 and its covariant derivative. For second-order
BPVE/IFE/IFT, the solved first-order density is differentiated on the k mesh,
so derivatives of k-dependent collision coefficients are included. The
derivative is taken in the complete orbital basis with its position connection,
avoiding derivatives of arbitrary eigenvector phases. An analytic CRTA
reference is subtracted before mesh differentiation, retaining the established
zero-EPC limit on coarse meshes. Even-grid Nyquist derivatives are zero.
The electronic mesh must resolve the collision-induced correction; these
derivatives do not eliminate k-grid convergence requirements.

VG responses are gauge transforms of the same LG kinetic equation, including
the transformation of the collision and stationary reference. The existing
Grid observable/contact contractions are retained. Nonlinear terms involving
population perturbations are assigned to intraband-origin reporting channels;
the total response is the physically meaningful quantity.

## Cost and phonon dispersion

Rates are summed over modes before storage and written directly into CSC.
This removes mode-resolved COO triplets and their concatenation copies.
No numerical rate cutoff is introduced. Storage still scales as
O(Nk Nq Nb^2); a Lorentzian can produce an essentially dense transition matrix
when q spans the entire electronic mesh. This remains a real limitation.

Automatic population solves use cached sparse LU up to 512 states and
diagonally preconditioned GMRES above that size, avoiding large LU fill-in.
At most two frequency factorizations are retained per stationary state, and
zero sources require no solve. Iterative failures and excessive stationary or
linear residuals raise errors instead of returning unchecked conductivities.

The rate builder calls `modes(q)` and `modeDisplacements(q)` at each q. It
supports different positive mode energies and q-dependent polarizations supplied
by that phonon interface; it does not require equal Einstein frequencies.
There is a regression fixture with dispersive frequencies. This change does
not add a force-constant phonon model. Exactly zero acoustic modes require an
explicit limiting prescription and currently raise an error.

## Scope of a common collision interface

The reusable density-matrix architecture is appropriate, but a general reduced
electronic equation has memory and possibly initial-correlation terms. In the
frequency domain its linear response involves a superoperator K(omega), not
necessarily constant population/coherence rates. At second optical order, the
source also includes nonlinear collision derivatives and field-induced changes
of the interaction, reference state, and observable vertices.

The current Markov class omits coherence-transfer vertices, pure dephasing from
diagonal noise, real Fan/Debye-Waller energy shifts, and phonon-assisted optical
current vertices. Its semiclassical rate weights do not recover all correlations
of the moving-phonon TDSE. No claim of TDSE agreement follows from fixing the
stationary state and nonlinear response.

The harmonic benchmark retains these trajectory correlations within its
prescribed bath. Generalizing its one-phase grid to d independent frequencies
with P points per phase costs P^d in a direct product representation. Commensurate
frequencies can use one common period, but that period and required resolution
may be large. A dispersive weak-coupling frequency-domain memory kernel summed
over (q,nu), or auxiliary phonon-assisted density matrices, is a more scalable
next approximation. It still requires its own truncation and convergence tests.

Useful background: [many-electron Pauli master equations](https://arxiv.org/abs/1411.5324)
and [generalized master equations with memory](https://arxiv.org/abs/2208.14273).

## Validation

`python -m unittest tests.test_epc_collision` covers the stationary equation,
number conservation, finite-difference Jacobian and Hessian, independent
classical steady states, direct/iterative solve agreement, factor reuse,
mode-aggregated rates versus scalar loops with dispersive phonons, and weak-EPC
recovery of both gauges. Independent nonlinear population propagation checks
the rectified second-order response. An isolated intracell dimer additionally
checks first- and second-order optical density matrices against direct nonlinear
DM propagation and checks LG/VG agreement without k-interpolation error.

For the 4x4x1, two-trajectory command at 300 K, photon energy 0.3 eV,
background 0.0685 eV and delta 0.03 eV, harmonic frequency DM and TDSE agree
within 3.1e-5 relative componentwise. The pair-balanced classical Markov result
differs by 61-228% componentwise and 84% in the global Frobenius norm. Raw
fixed-number rates differ by 65-230% and 90% globally. Pair balancing changes
the raw transition matrix by 40% in this coarse example. It fixes equilibrium
thermodynamics but cannot restore the coherent sidebands and memory discarded
by the Markov reduction. These are comparisons to a two-trajectory finite
sample, not ensemble-convergence claims.

Repeating the same benchmark with a stationary damped trajectory at
`phononDamping=0.005 eV` and an eight-period correlation window changed the
global Markov-versus-TDSE error only from 84.33% to 83.08%. The matched-window
Fourier-versus-TDSE error was 1.63% globally (selected components 0.26-4.98%)
at 33 phase points. Thus damping removes the infinite-correlation idealization
and gives a controlled route toward a Markov limit, but it does not by itself
repair the present weak-coupling collision approximation. The phonon damping,
finite Fourier window, phase resolution, supercell, and trajectory ensemble
must all be converged; `epc_delta` remains a separate broadening unless a
specific microscopic identification is justified.

With k=64x64x1, q=4x4x1, background .026 eV, delta .01 eV and photon energy
1.5 eV, the example's four quantum Markov DC/AC calculations together took
about 7.3 s locally. The steady-state residual was 7.6e-13 eV. This timing
includes four independent rate builds and is not a comparison to calculations
using a different phonon grid.
