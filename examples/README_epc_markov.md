# Stationary Markov EPC collision

`CollisionSpecification('epc', ...)` binds `CollisionOperator_full` to the
calculator's existing Energy eigensystem. It is a secular Pauli approximation
with population scattering-in, not a full non-Markov optical collision kernel.
The name `full` remains an alias for compatibility.

```python
collision = CollisionSpecification(
    'epc', qgrid=(4, 4, 1), bathStatistics='quantum',
    phononTemperature=300., delta=.03, constantLinewidth=.2,
    populationSolver='auto', solverTolerance=1e-10,
    stationaryTolerance=1e-12,
)
calc = Opticalconductivity_VG_DM(
    model=model, kgrid=(16, 16, 1), kBT=.025852,
    fermiList=np.array([-.7]), omegaList=np.array([1.3]),
    fullCovariantDerivative=True, collisionSpecification=collision,
)
state = calc.density_matrix_kernel.collision.forFermi(-.7)
print(state.stationary_info)
```

The complete orthogonal model basis is required. Each q-grid dimension must
divide the corresponding electronic k-grid dimension. Use the complete
coupled k mesh in one calculator; independent shifted-grid task partitions do
not reproduce a collision connecting states across those partitions. Projected
observables are supported where the existing calculator supports them, but
the electric drive must use the total velocity vertex.

The honeycomb example exposes the calculation alongside CRTA/SERTA:

```text
python examples/honeycomb_conductivity.py --grid 16 --skip-tdse --serta none --epc-markov quantum --epc-qgrid 4 --omega 1.3 --constantLinewidth .2
```

For comparison with the prescribed moving-phonon benchmark:

```text
python examples/honeycomb_conductivity.py --grid 4 --phonon-dynamics --serta none --epc-markov classical --epc-qgrid 4 --supercell 4 --omega 1.3 --constantLinewidth .2 --phonon-trajectories 2 --steps 128
```

The latter stores `sigma_ac_VG_phonon_Markov` on the same folded electronic
and phonon meshes as TDSE. `markov_relative_difference` uses the TDSE component
as denominator. Equal meshes do not make the two physical approximations equal.

## Stationary state and nonlinear equation

Write R[i,j] for a transition from state j to state i, with rates in eV and
time in hbar/eV. The implemented equation is

```text
C_ep[p] = (1-p)*(R p) - p*(R.T (1-p))
dp/dt = C_ep[p] - gamma*(p-f_reservoir)
```

`forFermi(mu)` specifies the reservoir chemical potential. A bounded Newton
solve finds p0 in [0,1] satisfying this equation; no constant collision residual
is discarded and no transition rates are adjusted. The damping used in response
is G = gamma I - D C_ep[p0]. The EPC part conserves total particle number; the
background reservoir fixes its stationary value to sum(f_reservoir).

The quantum broadened rates need not obey exact detailed balance at finite
`delta`, even at equal electronic/phonon temperatures. A nonzero
`reservoir_residual_eV` or `max_population_shift` exposes this limitation.
Solving for p0 makes the response consistent with the specified rates; it
does not turn Lorentzian-broadened rates into exact equilibrium quantum kinetics.
Classical/Wigner symmetric rates instead tend to equalize connected occupations
without the reservoir. Their nonthermal stationary state is expected.

For a general density matrix, the diagonal follows the Pauli equation while
the off-diagonal collision is

```text
C[rho]_ij = -(ell_i(p)+ell_j(p))*rho_ij/2,  i != j
ell(p) = gamma + R p + R.T (1-p).
```

`collision_integral(rho)` evaluates this closed nonlinear model. `apply(x)`
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

On the saved 4x4x1, two-trajectory moving-phonon benchmark (300 K, photon
energy 1.3 eV, background .2 eV, delta .03 eV), the corrected classical Markov
calculation differs from TDSE by at most 36.60% across the four in-plane
components, versus 44.34% before the fix. Quantum rates give 36.71% and
represent a different bath. The harmonic frequency solver remains at 0.0251%.
These are comparisons to that finite sample, not ensemble convergence claims.

With k=64x64x1, q=4x4x1, background .026 eV, delta .01 eV and photon energy
1.5 eV, the example's four quantum Markov DC/AC calculations together took
about 7.3 s locally. The steady-state residual was 7.6e-13 eV. This timing
includes four independent rate builds and is not a comparison to calculations
using a different phonon grid.
