# Stage 3: N/C ribbon spin transport and SOC conversion

The source is an s-only honeycomb ribbon (zigzag or armchair); C is a p-shell
ribbon with onsite lambda L.sigma/2 and scalar spin-independent hopping.
The interface has directional s-p hopping. Isolated C has no bulk Hall
response; that property is covered by tests, not recomputed on every run.

## Run

```text
python examples/orbital_torque_hex/orbital_nc_ribbon.py --workers 8 --row-profiles
python examples/orbital_torque_hex/orbital_nc_ribbon.py --edge armchair --workers 8
python examples/orbital_torque_hex/orbital_nc_ribbon.py --n-width 6 --c-width 32 --soc .01 --grids 256 512 --fermi 0 --workers 16 --row-profiles --output tmp/orbital_torque_hex/nc_zigzag_flux
```

Defaults: N width 4 zigzag / 2 armchair cells, C width 3 rows, SOC .4 eV,
interface hopping .4 eV, converter hopping .5 eV, EF=0 and .8 eV,
kBT=.01 eV, linewidth=.02 eV, shifted meshes 512/1024. Converter onsite
shift defaults to zero. Width counts do not imply equal physical widths
between edge geometries. Armchair rows collect both atoms of a dimer line.

## Observables and decomposition

Production retains four spin observables and adds five ACA orbital-rate diagnostics. Spin NPZ names:

- `spin`: local spin magnetic moment M_r={P_r,mu_S}/2, mu_S=-sigma mu_B.
- `soc_torque`: {P_r,i[H_SOC,mu_S]}/2. Because SOC is onsite here, this
  equals i[H_SOC,M_r].
- `flux_torque`: i[H0,M_r], positive for spin-moment influx into row r.
- `total_torque`: i[H,M_r], evaluated independently from the full Hamiltonian.

H=H0+H_SOC and H_XC=0 are explicitly verified during model construction.
Consequently total_torque=flux_torque+soc_torque. The code checks this
identity in both I/E channels and the C-total row-drive responses; tests
also verify the full drive/observed-row matrices. The sum of flux over all
rows vanishes. The total Hamiltonian rate need not vanish in a driven
steady state: driving and collision terms must also enter its balance.
It is not a reaction torque on a magnetic order parameter.

Moments have units mu_B/(V/Angstrom) per longitudinal repeat. All
rates have units mu_B*eV/(V/Angstrom), i.e. hbar times moment rate. Divide
by hbar=.6582119569 eV fs for rates in mu_B/fs. No area normalization is
applied. Only z is plotted; xyz is retained for symmetry checks.

BSR and ACA accumulation plots are omitted to reduce cost, not because their
irrelevance to spin generation has been established. The Hamiltonian and
its orbital degrees of freedom are unchanged.

## Projection and general nonlocal interactions

A drive on row d is {P_d,v_x}/2. Inter-row bonds are half assigned to each
endpoint; summing all drives recovers uniform driving. This is not P_d v_x P_d
and not a self-consistent spatial electric-field solution.

For any H_a, i[H_a,M_r] equals the projected spin torque
{P_r,i[H_a,mu_S]}/2 plus {i[H_a,P_r],mu_S}/2. The SOC/exchange calculators
provide both orders simultaneously: angularTorqueProjectionList selects the
former, angularProjectionList selects the complete local rate. Equal source
and target specify a Hermitian regional moment. For nonlocal SOC/exchange,
their complete local rates contain additional transport. SpinFluxTorque
includes H0 transport only. For a general model the three complete local
rates sum to i[H,M_r] only if H0+H_SOC+H_XC=H is actually satisfied.

The optimized example contracts the established LG density matrix, using
cyclic traces for local rates and batched row vertices for two-row matrices.
It is tested against the public flux/SOC LG calculators. VG is validation
only, with contacts for both orders; raw gauge residuals must be examined
separately from pole-subtracted finite-response agreement.

## Outputs and controls

`profile_fN.png` and `drive_rows_fN.png` have nine panels for uniform-drive
row profiles and C-total response versus driven row, respectively. Each
includes full, SOC-off, and interface-disconnected controls. `responses.npz`
retains total/I/E uniform profiles, row drives, and mesh comparisons;
`profiles.csv` and `drive_rows.csv` contain z components. `summary.json`
records parameters, timing, decomposition, mirror and control checks.
`geometry.png` shows actual bonds; model files are in case subdirectories.

With `--row-profiles`, the finest-mesh calculation also produces
`row_profiles/profile_drive_rowNNN_fM.png` for every drive, nine full-model
heatmaps in `response_matrix_fM.png`, all-control total/I/E arrays in
`responses.npz`, CSV tables and reconstruction residuals. Use
`--profile-drive-rows 0 12 43` to select drives. Summing observed C rows
reconstructs each C total; summing all drives reconstructs uniform profiles.
Small residuals verify algebra, not k-mesh convergence.

`--row-profiles-only --output PATH` rebuilds saved model parameters and
checks existing shared observables before adding the new row matrices.
`--plot-only --row-profiles --output PATH` replots current-format files.
Old BSR/ACA files need recalculation for the new nine-panel output.

Parallelism uses fixed chunks of at most 32 k points and one BLAS thread
per worker. `--workers N` changes scheduling, not the mesh or reduction
order. Workers are capped by chunk count. Row matrices share the finest
mesh preparation with uniform calculations; no second production run is
needed. Memory grows with concurrent workers.

## Audit

The audit covers independent Wannier-basis commutators, both projection
orders, displaced-mesh contact derivatives including nonlocal position
connections, LG/VG agreement, raw finite-frequency VG contraction, gauge
sum rules, and mesh partitioning. Additional tests reassign nonlocal SOC
between SOC/exchange and verify the complete local-rate decomposition.
Tests exercise no-SOC and disconnected controls, both ribbon geometries,
serial/parallel equality, and public-calculator agreement for all profiles.
The 2026-09-30 audit passed all 27 tests in the four modules below; no
calculator-core correction was required within the complete orthogonal-basis
scope tested. The example uses LG for production and VG only in validation.

The separate validation output `tmp/orbital_torque_hex/nc_zigzag_flux` uses
N width 6, C width 32, SOC .01 eV, EF=0, 256/512 points and 16 workers.
All controls passed. Maximum uniform/row-drive decomposition residual is
9.78e-15; the full two-row matrix decomposition residual is below 6.70e-16.
Global flux residual is below 5.60e-16, and marginal reconstruction errors
are below 1.57e-14. All 44 drive profiles are included. These algebraic
checks do not establish mesh convergence; the saved mesh differences should
still be inspected before interpreting amplitudes.

```text
python -m unittest tests.test_spin_flux_torque tests.test_spin_soc_torque tests.test_spin_torque_projection_orders tests.test_hex_nc_ribbon
```


## Orbital transfer diagnostics

The five added observables use mu_L=-L mu_B and M_L,r={P_r,mu_L}/2:

- `orbital_soc`: projected i[H_SOC,mu_L], equal to i[H_SOC,M_L,r] for onsite SOC.
- `orbital_h0`: i[H0,M_L,r], the complete H0-induced local rate.
- `orbital_crystal`: {P_r,i[H0,mu_L]}/2, the projected crystal/interface torque.
- `orbital_transfer`: {i[H0,P_r],mu_L}/2, evaluated as orbital_h0 minus orbital_crystal.
- `orbital_total`: i[H,M_L,r], evaluated independently from the full band energies.

OrbitalFluxTorqueGalvanicConductivity measures orbital_h0 when supplied
angularProjectionList; with angularTorqueProjectionList it measures
orbital_crystal. H0 does not conserve ACA orbital moment at the s-p interface,
so orbital_h0 alone is NOT an orbital influx. All five quantities use the
same energy-weighted rate units as spin torque.

The checks verify orbital_total=orbital_h0+orbital_soc, zero global summed
orbital_transfer, and spin SOC torque + 2*orbital SOC torque = 0. The factor
2 comes from magnetic g factors, not a transfer efficiency. Both projection
orders and all I/E channels are compared with the public LG calculators;
VG residuals and independent nonlocal-SOC contact derivatives are tested.
Obsolete spin/orbital projection keyword names are now rejected explicitly
at the response-calculator layer; use the generic angular names above.

`orbital_cut_current_fN.png` and `orbital_cut_currents.npz/.csv` show
J_L(r+1/2)=-sum_{q<=r} orbital_transfer_q, positive toward increasing y.
The row_profiles folder additionally contains `orbital_cut_current_fN.png`
for all three controls on a shared color scale, and `orbital_cut_currents.npz`
with axes [driving row, cut after observed row, Fermi]. The cut-current unit
is mu_B*eV/(V/Angstrom), or hbar times moment flow per electric field.
Uniform cut arrays retain xyz; row-drive cut arrays retain z.

Cuts strictly inside C connect identical p shells with scalar H0 hopping,
so these are clean ACA orbital-current diagnostics. At N/C the same
symmetrized partition is a convention: s-only N has no ACA moment, and
orbital_crystal and orbital_transfer can cancel there although the actual
local ACA rate is zero. Nonzero interface ACA transfer alone does not prove
a conserved BSR current arriving from N. No BSR operator is computed here.

Validation: all 32 orbital/spin/ribbon tests pass. The C16 weak-SOC benchmark
is saved in `tmp/orbital_torque_hex/nc_zigzag_n6_c16_soc0.01_orbital_transfer`
(N width 6, C width 16, SOC .01 eV, EF=0, 512/1024 points, eight workers).
All controls pass; uniform and two-row orbital identities have maximum
residual 8.17e-15, and marginal reconstruction errors are below 1.17e-14.
Spin matches the prior C16 run exactly. Maximum internal-C orbital cut
current magnitude is .10869 with SOC and .10528 without SOC; the disconnected
control is below 1.18e-14, in the rate units specified above. These values
show ACA orbital transport in the coupled ribbon, not proof of a conserved
BSR current originating in N. `orbital_audit.json` records identity residuals.
