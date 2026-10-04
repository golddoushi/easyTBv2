# Step 3: s-p orbital source N and onsite-SOC converter C

```text
python examples/orbital_torque/orbital_nc_ribbon.py --widths 8 6 --widths 12 6 --widths 8 12 --workers 8 --row-profiles
python examples/orbital_torque/orbital_nc_ribbon.py --plot-only
python tests/test_nc_ribbon.py
```

The original model is retained: every row carries s and p shells. N has the
source's directional s-p hybridization (`--tsp`, default 0.7 eV); C has
onsite energies (-2,0,0,0)+shift, orbital-independent nearest-neighbor
hopping (`--converter-hopping`, default 0.5 eV), and onsite isotropic SOC
(`--soc`, default 0.4 eV). A spin-independent -t_interface I4 link joins
N/C (`--interface-hopping`, default 0.3 eV). No exchange field is present.
The Hamiltonian explicitly registers H0, SOC and zero exchange components.

Controls now vary widths: repeat `--widths N C`, in physical rows. Defaults
are (8,6), (12,6), (8,12). Old switch-off/disconnected/N-only-drive comparison
sets are no longer run. All other model parameters are common to the cases.
The earlier single-width CLI and old result layout have been replaced;
old results must be recomputed for the torque panels.

The response uses the same length-gauge density-matrix evaluator as the
hexagonal N/C investigation, including k-shard parallelism (`--workers`).
Defaults: EF=0.8 eV, kBT=linewidth=0.05 eV, meshes 256/512/1024. These can
be changed with `--fermi`, `--kbt`, `--linewidth`, and `--grids`.

## Observables and conservation

The twelve z-component panels are spin accumulation, spin SOC torque,
H0 spin influx, full spin rate, orbital SOC torque, H0 orbital local rate,
crystal orbital torque, orbital transfer, full orbital rate, direct spin
source, direct orbital source and ACA orbital accumulation. BSR is not
computed. All xyz components and uniform I/E channels remain in NPZ data.
Spin/orbital relaxation and independently evaluated continuity residuals
are preserved in data without redundant plot panels.

The orbital split is:

- Local H0 rate: i[H0,{P_r,mu_L}/2].
- Crystal torque: {P_r,i[H0,mu_L]}/2.
- Orbital transfer: local H0 rate minus crystal torque.

Onsite SOC has no inter-row transfer term. Spin and orbital continuities,
full-rate decompositions, the magnetic-moment SOC relation T_S+2*T_L=0,
and row-drive sums are checked independently. Rate vertices use energy
units: mu_B eV/(V/Angstrom); divide by hbar for rates per unit time.
Accumulations use mu_B/(V/Angstrom), per longitudinal repeat.

Driving uses {P_row,v_x}/2, with a distinct density-matrix source term kept
in the continuity balance. A driving-row contribution is a numerical
partition, not a self-consistent spatial electric-field calculation.

## Alignment and outputs

Last N row = -1; first C row = 0; interface = -0.5. Native row spacing is
preserved. N extends left and C right, with no interpolation or rescaling.
Missing outer rows appear blank/NaN. Display coordinates count rows, not
Angstrom. C totals sum actual physical rows before display alignment.

Under `--output` (default `tmp/orbital_torque/nc_ribbon`):

- `profile_f*.png`: all width cases under uniform Ex.
- `drive_rows_f*.png`: C-total responses versus driving row.
- `orbital_cut_current_f*.png`: cumulative orbital transfer across native cuts.
- With `--row-profiles`, `row_profiles/` contains aligned response matrices
  per case with common color scales, and comparison profiles for each
  driving row. Cases lacking that driving row are omitted.
- Each `nN_cC/` contains the actual `model.npz`, `responses_n*.npz`, finest
  `responses.npz`, and `profiles.csv`/`drive_rows.csv`. Full row matrices
  (total/I/E, z only) are computed only on the finest mesh when requested.
- `summary.json` stores parameters, timing, successive-mesh differences,
  regional totals, and conservation/projection checks per case.
- `aligned_display.npz` records selection matrices and missing-row masks;
  it is a plotting cache. Physical positions and arrays remain per case.

The existing NC builder is also used by the N/C/F example; its return
signature and Hamiltonian construction are retained.
