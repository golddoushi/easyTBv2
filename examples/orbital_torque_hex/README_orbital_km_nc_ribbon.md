# Honeycomb N / intrinsic Kane-Mele C ribbon

`orbital_km_nc_ribbon.py` is a separate s-only N/C example. Both regions
share one honeycomb lattice, periodic along x and open along y. Zigzag and
armchair edges are available. N has nearest-neighbor hopping and a staggered
potential (default +/-0.5 eV), while C has the same nearest-neighbor hopping,
a selectable staggered potential (default zero), and intrinsic SOC:

`H_SOC[i,j] = i lambda nu_ij sigma_z`.

`nu_ij` is the sign of the z component of the turn along the two nearest-
neighbor bonds from i through their common neighbor to j, for the matrix
convention used by the builder. Reversing the link reverses its sign. NNN
bonds are generated in the infinite primitive lattice before cutting the
ribbon, so edge bonds do not depend on whether an intermediate atom survives
the cut. SOC is retained only if both endpoints are in C. Nearest-neighbor
N/C links use `--interface-hopping`; there is no Rashba SOC or exchange field.
The s labels describe this effective one-orbital model, not a microscopic
atomic derivation of intrinsic SOC from an isolated s shell.

## Run

```text
python examples/orbital_torque_hex/orbital_km_nc_ribbon.py --workers 8 --row-profiles
python examples/orbital_torque_hex/orbital_km_nc_ribbon.py --widths 6 8 --widths 12 8 --widths 6 16 --workers 8 --row-profiles --output tmp/orbital_torque_hex/km_widths
python examples/orbital_torque_hex/orbital_km_nc_ribbon.py --edge armchair --n-width 6 --c-width 8 --soc .1 --grids 512 1024 --workers 8 --row-profiles
python examples/orbital_torque_hex/orbital_km_nc_ribbon.py --plot-only
python tests/test_hex_km_nc_ribbon.py
```

Widths count transverse cells, not individual atoms. The source uses the
same rectangular-strip convention as the earlier hexagonal examples.
Defaults are N=6, C=8, lambda=0.1 eV, nearest-neighbor hopping magnitude
1 eV, kBT=0.01 eV, linewidth=0.02 eV and EF=0. The ribbon meshes are 256/512;
`--bulk-grid` (default 48) sets the separate two-dimensional bulk reference.
`--fermi` accepts several values. Output defaults to
`tmp/orbital_torque_hex/km_nc_zigzag` or `km_nc_armchair`.
BSR is included by default in this example and can require substantially
finer meshes than the spin response. Mesh differences are recorded, not
silently treated as proof of convergence. `--workers` uses the existing
threaded k-shard evaluator with BLAS restricted to one thread per worker.

## Comparing N/C widths

Repeat `--widths N C` to run width controls with all other parameters shared.
These pairs override the single-case `--n-width` and `--c-width` settings.
Each case has its own `nN_cC/` directory, including physical coordinates,
responses, convergence checks, bands and native plots. The parent directory
contains comparison figures and a case index in `summary.json`.

For comparisons, the last N row is at -1, the first C row is at 0 and the
interface is at -0.5. All physical rows retain unit index spacing; wider N
regions extend left and wider C regions extend right. This coordinate counts
rows, not Angstrom. No interpolation or averaging is applied. Missing outer
rows are NaN in `aligned_display.npz` and appear blank in the plots.

Parent `profile_f*.png` and `drive_rows_f*.png` overlay all cases. C totals
use each model's actual C rows. With `--row-profiles`, the parent
`row_profiles/` contains aligned matrices with common color scales and
overlaid profiles for each driving row; cases without that row are omitted.
`--plot-only --output <comparison directory>` regenerates these comparisons
without recomputing responses. Single-case operation remains available when
`--widths` is omitted.

## What is, and is not, conversion here

ACA moment is identically zero throughout the model. Intrinsic Kane-Mele
SOC also commutes with total Sz. Consequently, the z component of the
projected spin conversion torque `{P_r, i[H_SOC,mu_S]}/2` vanishes.
The local rate `i[H_SOC,{P_r,mu_S}/2]` need not vanish: it is spin transport
through SOC hopping. This rate is plotted as **SOC hopping influx**.

The independently evaluated continuity check is

`H0 influx + SOC hopping influx + direct spin source - eta*spin = 0`.

It is also checked against `full Hamiltonian rate + source - eta*spin`.
The zero conversion torque is retained as a diagnostic panel. This distinction
is essential for nonlocal SOC; using only the projected global torque in
this balance would omit SOC-mediated transport.

A finite spin response with zero ACA moment does **not** prove BSR-to-spin
conversion. Spin-dependent hopping is already present explicitly and can
redistribute conserved Sz. BSR is a separate observable evaluated from the
full junction; it is partitioned as `{P_r,mu_BSR}/2`. No separately driven
BSR injection or conservation equation for BSR is assumed.

Unlike the previous p-shell C model, this converter has its own bulk spin
Hall response. Its infinite bulk is in the intrinsic quantum spin Hall
regime when `abs(delta_C) < 3*sqrt(3)*abs(lambda)` (nonzero lambda). Finite
ribbons can have edge-state hybridization gaps. The band plot resolves spin
sectors and colors states by mean y to inspect boundary localization; it
is not an independently computed Z2 invariant or a lead-based conductance.
The implementation is checked against the analytic bulk Dirac masses.

## Results

- `profile_f*.png`: uniform Ex, seven z-component panels: BSR, spin, SOC
  hopping influx, SOC conversion torque, H0 influx, total rate, direct
  source. Relaxation is retained in data.
- `drive_rows_f*.png`: corresponding C-integrated responses versus driving
  row. The drive is the existing `{P_row,v_x}/2` numerical decomposition,
  not a self-consistent spatial electric-field profile.
- With `--row-profiles`, `row_profiles/response_matrix_f*.png` and individual
  drive-row profiles contain the same seven observables.
- `responses_n*.npz`, `responses.npz`: native arrays, all moment components,
  I/E channels and continuity residuals; full row matrices are z only.
  ACA arrays are retained and zero. `profiles.csv` and `drive_rows.csv`
  contain the plotted z components.
- `bands.png`/`bands.npz`: spin-resolved ribbon spectrum (figure displays
  +/-1.5 eV; NPZ keeps all bands).
- `bulk_hall.png`, `bulk_N.npz`, `bulk_C.npz`: isolated bulk BSR/ACA OHC and
  SHC at requested Fermi energies. Bulk Hall normalization differs from
  the local moment response normalization; do not compare their magnitudes
  directly.
- `summary.json`: parameters, analytic bulk phase criterion, checks and
  successive-mesh differences. `model.npz` is the final junction Hamiltonian
  with the explicit H0/SOC split. Files under `model/` are the primitive
  construction inputs, not the final finite junction Hamiltonian.

No disconnected or SOC-off comparison sweep is run automatically. The CLI
allows parameter variation in separate output directories.

References: [Kane and Mele, Quantum Spin Hall Effect in Graphene](https://arxiv.org/abs/cond-mat/0411737)
and [Z2 Topological Order and the Quantum Spin Hall Effect](https://arxiv.org/abs/cond-mat/0506581).

