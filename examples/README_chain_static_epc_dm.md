# Complete static EPC correction in the density-matrix formalism

`Opticalconductivity_VG_DM_StaticEPC` implements the complete Gaussian static
displacement expansion of the linear conductivity through second order in
displacement. It includes both DC and finite photon frequencies. It uses the
equilibrium electronic eigensystem from the existing `Energy`/`Velocity`
calculator and constructs density-matrix coefficients analytically. It does not
diagonalize displaced Hamiltonians, finite-difference displacements, introduce
an EPC delta regulator, or add a SERTA lifetime to the result.

This is the static response completion discussed in the selected-mode benchmark.
It is not a dynamical quantum EPC solver or a second-order electric-field/BPVE
calculator. For static Gaussian displacements it should agree with the full
snapshot average through second displacement order; higher displacement orders
remain in the snapshot average.

## Chain example

```powershell
python examples/chain_transport_alignment.py --grid 2048 --qgrid 16 --supercell 4 --phonon-q 0.5 --phonon-branches 0 --phonon-weights 0.0625 --pairs 0 --fixed-centers --scale 0.125 --background 0.026 --delta 0.1 --omega 0 0.8 --static-epc-dm --curvature --curvature-step 0.00125 --output tmp/chain_static_epc_dm_validation
```

* `--static-epc-dm` adds the complete DM correction on the same supercell and
  folded electronic grid as the snapshots. No snapshot samples are required.
* The default output `static_epc_dm` uses displaced orbital centers, matching
  `makeDisplacedModel`. Adding `--fixed-centers` also computes
  `static_epc_dm_fixed`, matching hopping-only snapshots and the TDSE basis.
* `--static-epc-kbatch` controls the maximum electronic batch size (default 4).
  Large band counts automatically reduce it. This is a memory/performance
  control, not a physical or convergence parameter.
* `--delta` continues to affect the old SERTA/full comparison entries only.
  Neither the new solver nor snapshot curvature consumes it.
* `--curvature` remains an independent finite-displacement verification path.
  It uses the LG calculator, while the new solver uses VG; at finite frequency
  their contact-residue difference must be converged with the electronic grid.
  Its `--curvature-step` has no effect on the new DM solver.

The existing `--phonon-q`, `--phonon-branches` and `--phonon-weights` selection
is shared automatically. Without these flags the full supercell displacement
covariance is used. For that case match qgrid and supercell when comparing the
new solver to the primitive rate-based approaches.

## Equations and completeness

Factor the real-space covariance as

    C_ij = sum_s B_si B_sj,   u_i = sum_s B_si x_s,   <x_s x_t> = delta_st.

The existing phonon model supplies B in Angstrom, including absolute q weights,
branch selection, mass, phonon energy and thermal/zero-point variance. Each
independent normal coordinate can be expanded separately and its second-order
contribution summed. Arbitrary real covariance factors may also be supplied.

For one coordinate, use Taylor coefficients (not raw second derivatives):

    H(x) = H0 + x H1 + x^2 H2,
    v_a(x) = v_a0 + x v_a1 + x^2 v_a2,
    w_ba(x) = w_ba0 + x w_ba1 + x^2 w_ba2 + ... .

Here H2 includes beta/2. Moving-center vertices additionally differentiate the
bond vector in v=i*d*H and w=-d_b*d_a*H. Consequently v2 and w2 may be nonzero
even with a linear SSH Hamiltonian. The reference has diagonal point-orbital
positions and an orthogonal complete electronic basis.

Let f(H) be the Fermi matrix at the specified chemical potential and electronic
temperature. Its first three Frechet derivatives determine

    f0 = f(H0)
    f1 = Df[H1]
    f2 = Df[H2] + D^2f[H1,H1]/2

and the coefficients of the gauge-covariant electric source Df(H(x))[v_a(x)]:

    d_a0 = Df[v_a0]
    d_a1 = Df[v_a1] + D^2f[H1,v_a0]
    d_a2 = Df[v_a2] + D^2f[H1,v_a1] + D^2f[H2,v_a0]
            + D^3f[H1,H1,v_a0]/2.

In the equilibrium eigenbasis these are ordered matrix products weighted by
Fermi divided differences. Confluent limits are treated analytically with a
Taylor series, so degeneracies require no inverse-gap cutoff. No neighboring
electronic k points are needed for these derivatives.

With L_omega = gamma_bg - i*omega + i[H0,.], the smooth electric-response
coefficients obey

    L_omega r_a0 = d_a0
    L_omega r_a1 = d_a1 - i[H1,r_a0]
    L_omega r_a2 = d_a2 - i[H1,r_a1] - i[H2,r_a0].

These are physical density coefficients without an observable charge sign.
L_omega is inverted elementwise in the equilibrium
basis. These equations follow from expanding the same instantaneous-Fermi,
gauge-covariant CRTA background used by the existing VG response.

The conductivity correction contains

    -Tr(r_a2 v_b0 + r_a1 v_b1 + r_a0 v_b2)

plus i/omega times the second-order coefficient of the VG equilibrium residue

    Tr(Df(H)[v_a] v_b + f(H) w_ba).

The residue is retained at finite frequency. At exactly zero frequency it is
removed by the analytic periodic-BZ limit, exactly as in Opticalconductivity_VG_DM.
Its value before removal is saved to diagnose electronic-grid convergence.

Together these terms retain repeated linear-Hamiltonian insertions (Fan-type
effects), explicit quadratic/Debye-Waller-type vertices, equilibrium occupation
changes, propagation corrections, drive corrections, measured-current changes
and contact-current terms. Only diagonal energy shifts or a modified linewidth
would omit part of this expansion. None of the channel groups below is a
separately gauge-invariant observable or a uniquely isolated Fan self-energy.

## API and outputs

```python
from phonon import PhononBath
import numpy as np
from calculator import Opticalconductivity_VG_DM_StaticEPC, CollisionSpecification

# model is already the commensurate supercell, carrying its phonon covariance.
response = Opticalconductivity_VG_DM_StaticEPC(
    model=model,
    kgrid=(512, 1, 1),
    kBT=0.025851999786435,
    fermiList=np.array([-0.7]),
    omegaList=np.array([0.0, 0.8]),
    collisionSpecification=CollisionSpecification('crta', constantLinewidth=0.026),
    movingCenters=True,
    phononBath=PhononBath(temperature=300.0, statistics='quantum'),
)
```

`sigma_bare`, `sigma_correction` and `sigma` have axes
`(fermi, omega, drive, current)` and units S/m, with
`sigma = sigma_bare + sigma_correction`.

`sigma_correction_channels` contains `density` (r2*v0), `mixed_current`
(r1*v1), `quadratic_current` (r0*v2) and `contact` (the VG equilibrium-pole
residue). `sigma_correction_by_mode` resolves covariance factors.
`sum_rule_residual` and `correction_sum_rule_residual` report the bare and
second-order pole residues. `static_epc_info` records the approximation,
statistics, center convention and covariance source.

The chain example saves their xx components under `static_epc_dm_*` and, when
requested, `static_epc_dm_fixed_*`. Corrections are stored directly from the
expansion, avoiding subtraction of two large conductivities.

In `chain_transport_alignment.py`, `--phonon-statistics` now selects the
statistics for this solver, snapshots/curvature, collision rates and TDSE
together. Use `--phonon-statistics classical` for an equipartition comparison;
`--tdse-statistics` is retained as a shared-scope compatibility alias.

The default statistics are the static quantum Gaussian marginal; `wigner`
has the same covariance, and `classical` uses equipartition. This does not
introduce quantum phonon dynamics. Custom `displacementFactors` must be a real
array `(nfactor,natom,3)` whose outer products give the desired covariance.

## Validation

At the parameters in the example, the moving-center DC correction is
**-5.253329256 S/m**. The independent finite-displacement curvature at a
0.00125 Angstrom step is -5.253269150 S/m; its step-extrapolated value is
-5.253328889 S/m. The full nine-node Gaussian snapshot average is
-5.252595214 S/m, differing from the DM correction by **0.013975%**. The
Gaussian average includes fourth and higher displacement orders.

At photon energy 0.8 eV, the new DM correction is
0.119841688 - 0.095204870i S/m. Independent displacement curvature gives
0.119842128 - 0.095204934i S/m. Fixed-center DC gives -5.252145338 S/m;
center conventions must therefore be kept consistent in a comparison.

Data are in `tmp/chain_static_epc_dm_validation/`. Unit tests independently
verify matrix-function derivatives at degeneracies, both DC and AC displaced
model responses, linear/mixed/pure-quadratic EPC, covariance-factor rotations,
weight scaling, zero EPC, streaming/batching and delta independence. The
targeted regression run passed 69 tests.

## Scope and cost

This implementation supports static Gaussian covariance, SSH hopping modulation,
point-orbital positions, and a positive scalar CRTA background at finite
electronic temperature. Adding SERTA or full collision is rejected because
the new correction already contains the static EPC response at this order.
It is not a dynamical phonon collision operator, and should not be compared to
quantum dynamical SERTA as if both represented the same bath.

Automatic covariance construction uses the existing Einstein phonon model.
Supplied real covariance factors can also encode the equal-time displacement
distribution of dispersive phonons; their dispersion and covariance must be
constructed externally. Static response depends on that covariance, not on
explicit phonon motion.

The kernel is reusable below the calculator layer, but only linear electric
conductivity is connected in this change. BPVE/IFE/IFT still require their
own consistent response expansions. Accuracy requires weak enough displacement
amplitudes and converged electronic grids; the method does not resum strong
disorder or capture higher Gaussian orders.

Dense third divided differences involve fourth powers of the supercell band
count. Small tensors are cached, while larger tensors stream their last band
index; k batching limits memory. Small commensurate selected-mode supercells
are the intended first benchmark. Large supercells can be expensive despite
having no displaced eigensolves. Each electronic k block is independent, so
future k-point parallelization needs only a reduction of response tensors.
