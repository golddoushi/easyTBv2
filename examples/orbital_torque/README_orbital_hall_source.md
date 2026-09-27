# Spin-independent orbital Hall source

Run from the repository root:

```text
python examples/orbital_torque/orbital_hall_source.py
```

For a quick calculation at one metallic filling:

```text
python examples/orbital_torque/orbital_hall_source.py --grids 24 48 --fermi 0.8 --output tmp/orbital_torque/source_quick
```

The example uses `OrbitalHallConductance_LG_DM` and
`SpinHallConductance_LG_DM` directly. No response formula is reimplemented,
and no small spin responses are rounded or replaced by zeros. NumPy, SciPy,
Matplotlib, threadpoolctl, and the normal repository runtime are required.

## Hamiltonian

One atom per square cell, lattice constant 3 Angstrom, with the full `s,p`
shells. The local basis is `(s,pz,px,py)` in each spin block, as required by
`System_tb`. The eight-band Hamiltonian is `diag(h,h)`. There is no SOC,
exchange, or magnetic field. All orbital centers coincide with the atom.
The out-of-plane cell height is 10 Angstrom, with no z hopping.

With `qx=kx*a`, `qy=ky*a`, and `H(k)=sum_R H_R exp(+i k.R)`, the spatial
Hamiltonian has diagonal entries (eV):

```text
h_ss   = -2.0 - 1.0*(cos(qx)+cos(qy))
h_pzpz =      - 0.2*(cos(qx)+cos(qy))
h_pxpx =        cos(qx) - 0.2*cos(qy)
h_pypy = -0.2*cos(qx) +       cos(qy)
```

The only off-diagonal entries are
`h_s,px=2i*tsp*sin(qx)`, `h_s,py=2i*tsp*sin(qy)`, and their Hermitian
conjugates, with `tsp=0.7 eV`. The pz orbital completes the angular-momentum
shell for future atomic SOC, but does not hybridize here. The source follows
the orbital-texture construction discussed by
[Go et al., PRL 121, 086602 (2018)](https://arxiv.org/abs/1804.02118);
these numerical parameters are an illustrative choice, not a fit to a material
or a reproduction of that paper's parameter set.

Spin independence implies `delta_rho=diag(delta_rho_orb,delta_rho_orb)`
under the charge drive and scalar CRTA collision operator. Every spin-current
trace therefore vanishes: the z contributions cancel between identical spin
blocks, and x/y spin-current vertices connect the two blocks. Orbital-current
contributions instead add between spins. This is an exact symmetry statement;
the numerical zero check uses an absolute tolerance of `1e-10`.

## Response and conventions

Defaults: `kBT=0.05 eV`, CRTA linewidth `0.05 eV`, Gamma-centered
`48x48x1`, `96x96x1`, and `192x192x1` meshes, and 56 chemical potentials from
`-3.5` to `2.0 eV`. Both finite-temperature occupations and the full dc
covariant derivative are enabled. The intraband `I` and interband `E`
channels are retained and saved. These are responses to electric field,
not conductivities divided by the longitudinal charge conductivity.

Tensor axes are `[fermi, drive, flow, polarization]`.
`sigma[:,0,1,2]` measures y-directed, z-polarized current induced by Ex.
The script checks its opposite transverse component for antisymmetry, and
checks the entire SHC tensor, not just this Hall component.

**Values are reported in the existing calculators' native normalization.**
Their current vertices are magnetic-moment currents `{mu_L,v}/2` and
`{mu_S,v}/2`, with `mu_L/mu_B=-L/hbar` and `mu_S/mu_B=-2*S/hbar`.
The stored response is the contraction multiplied by
`2*pi/(cell_area*Nkx*Nky)`, where the area is in Angstrom squared and
the velocity vertex uses the repository energy-length convention.
It is not an SI S/m conductivity. In the same numerical normalization,
replacing the moment vertices by dimensionless angular-momentum vertices
requires `OHC_L/hbar=-OHC_native` and `SHC_S/hbar=-SHC_native/2`.
The sign in the table below is the native magnetic-moment-current sign.

The example concerns conventional local atomic orbital current. It does not
establish orbital transmission through an interface or subsequent conversion
into spin. Those require the later junction calculations.

## Validation and outputs

The script checks identical spin blocks after reading the model, zero SHC
on every mesh, finite OHC somewhere in the scan, transverse OHC antisymmetry,
and zero OHC/SHC in an otherwise identical `tsp=0` control. The latter removes
all orbital hybridization in this model. A failed check raises an error after
saving the diagnostic results. Custom Fermi lists outside the active bands
can legitimately fail the finite-OHC check.

The reference run with `--grids 48 96 192` gives, at Fermi energy 0.8 eV:

| Mesh | OHC, Ex to Jy with z polarization (native) | Maximum absolute SHC, all fillings/components |
| --- | ---: | ---: |
| 48 x 48 | -2.99245577196 | 0.0 |
| 96 x 96 | -2.99241556417 | 0.0 |
| 192 x 192 | -2.99241556197 | 0.0 |

The last two OHC values agree within approximately `7.3e-10` relative at this
filling. This is mesh convergence at the stated nonzero temperature and
linewidth, not a claim about their zero limits. Exact floating-point zero
SHC occurred in this run; other BLAS/eigensolver implementations may leave
roundoff residuals. The analytic symmetry establishes the strict zero.

Outputs in `tmp/orbital_torque/source/` (the earlier run remains in
`tmp/orbital_hall_source/`):

- `model/` and `no_hybridization/`: reproducible POSCAR and hr model inputs.
- `responses.npz`: full tensors and I/E channels for every mesh and control.
- `hall_scan.csv`: finest-mesh OHC, selected SHC, full-tensor SHC maximum,
  and control OHC as functions of Fermi energy.
- `summary.json`: parameters, mesh convergence, tolerance, and pass/fail checks.
- `hall_scan.png`: OHC mesh comparison, control, and numerical SHC residuals.

All files in the selected output directory are regenerated on rerun.
