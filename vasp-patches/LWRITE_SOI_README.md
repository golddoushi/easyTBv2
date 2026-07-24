# VASP 6.1.2 `LWRITE_SOI` patch

This patch writes the Cartesian Bloch-space matrix elements

\[
A_{\mathbf{k},nm}^{\sigma\sigma',j}
=
\left\langle\bar\psi_{\mathbf{k}n\sigma}\middle|
\sum_\alpha \xi_\alpha(\mathbf r-\mathbf r_\alpha)L_j^\alpha
\middle|\bar\psi_{\mathbf{k}m\sigma'}\right\rangle ,
\qquad j=x,y,z ,
\]

from a scalar-relativistic, collinear, no-SOC VASP calculation. The PAW
operator is evaluated using the same radial SOC strength and real-spherical-
harmonic angular-momentum matrices that VASP uses internally for `LSORBIT`.
It is stored separately and is never added to `CDIJ`.

The output is intended for subsequent disentanglement and Wannierization
with standalone Wannier90. It does not modify Wannier90.

## Files changed

- `src/relativistic.F`
  - Adds `LWRITE_SOI` and storage for the Cartesian PAW projector matrices.
  - Extends `SPINORB_STRENGTH` with an optional orbital-only output.
  - Obtains the Cartesian matrices by decomposing the same four `LS` spin
    blocks used by VASP's native `LSORBIT` implementation. This avoids an
    independent assumption about the internal `DUMMY` tensor.
- `src/paw.F`
  - Evaluates the orbital-only SOC operator from the spin-averaged collinear
    potential without adding SOC to the Hamiltonian.
- `src/mlwf.F`
  - Reads `LWRITE_SOI`.
  - Collects band-distributed projector columns through `COMM_INTER`.
  - Contracts the PAW matrices with the projectors local to each
    `COMM_INB` rank.
  - Transfers each completed k-point matrix from its owning `COMM_KINTER`
    group before writing.
  - Writes `wannier90.soi` in the same band and k-point order as the
    corresponding Wannier90 interface files.

## Applying the patch on Linux

Place `vasp-6.1.2-lwrite-soi.patch` outside or inside the clean VASP 6.1.2
source directory, then run from the VASP source root:

```bash
git apply --check /path/to/vasp-6.1.2-lwrite-soi.patch
git apply /path/to/vasp-6.1.2-lwrite-soi.patch
```

For a source directory that is not a Git worktree:

```bash
patch --dry-run -p1 < /path/to/vasp-6.1.2-lwrite-soi.patch
patch -p1 < /path/to/vasp-6.1.2-lwrite-soi.patch
```

Compile the ordinary complex collinear executable (`vasp_std`) with the
same Wannier90 interface options you normally use. VASP 6.1.x commonly uses
`-DVASP2WANNIER90v2` when linked to Wannier90 2.x.

## Required calculation mode

The writer checks and requires:

```text
ISPIN          = 2
LSORBIT        = .FALSE.
LNONCOLLINEAR  = .FALSE.
ISYM           = -1
LWANNIER90     = .TRUE.
LWRITE_SOI     = .TRUE.
```

Run `vasp_std`, not `vasp_ncl`.

`NPAR`/`NCORE` and `KPAR` may be chosen normally. For each owned k point, the
writer broadcasts the distributed band columns of `W%CPROJ` within
`COMM_INTER`, sums the ion/projector contributions within `COMM_INB`, and
then reduces the completed matrix across `COMM_KINTER`. The coordinates in
each output block are metadata and are not used as a substitute for these
MPI ownership operations.

`ISYM=-1` remains required so every point of the Wannier mesh is explicitly
present and uses the same order as the Wannier90 interface files. The
band-column gathering temporarily allocates
`NPROD * NB_TOT * 2` projector elements per MPI rank for one k point. This is
usually much smaller than the plane-wave storage, but it should still be
included in memory estimates for calculations with many projectors/bands.

A practical workflow is:

1. Converge the collinear and spinful ground state without SOC.
2. Perform a fixed-charge run on the uniform Wannier mesh, for example with
   `ICHARG=11`, the converged `CHGCAR`, the desired `NBANDS`, and the tags
   above.
3. Supply the normal `wannier90.win` containing `num_wann`, projections,
   excluded bands, and disentanglement windows.
4. Run patched `vasp_std`; it writes the normal spin-resolved Wannier90
   interface files plus `wannier90.soi`.
5. Run standalone Wannier90 separately for the up and down seeds.

The excluded-band map used for `wannier90.soi` is the same map produced by
`wannier_setup`, so its compact band indices match `wannier90.up.eig`,
`wannier90.dn.eig`, and the corresponding `.amn` and `.mmn` files.

## `wannier90.soi` format

Line 1 is a comment describing conventions.

Line 2 contains:

```text
num_bands  num_kpoints  3  2
```

For every k point, Cartesian component, bra spin, and ket spin, one block is
written. Its header is:

```text
k_index  component  spin_bra  spin_ket  kx  ky  kz
```

Conventions:

- `component=1,2,3` means `x,y,z`.
- `spin=1,2` means `up,down`.
- k coordinates are fractional reciprocal coordinates.
- Values are in eV.

Each block contains `num_bands**2` entries:

```text
band_bra  band_ket  real_part  imaginary_part
```

Rows (`band_bra`) run fastest inside each matrix column (`band_ket`), matching
Fortran column-major ordering. The expected total line count is

```text
2 + num_kpoints * 3 * 2 * 2 * (1 + num_bands**2)
```

The three full spin-block matrices at each k point are

\[
\mathcal A^j_{\mathbf k} =
\begin{pmatrix}
A_{\mathbf k}^{\uparrow\uparrow,j} &
A_{\mathbf k}^{\uparrow\downarrow,j}\\
A_{\mathbf k}^{\downarrow\uparrow,j} &
A_{\mathbf k}^{\downarrow\downarrow,j}
\end{pmatrix}.
\]

Form the ladder components in postprocessing:

\[
A^+_{\mathbf k}=A^x_{\mathbf k}+iA^y_{\mathbf k},\qquad
A^-_{\mathbf k}=A^x_{\mathbf k}-iA^y_{\mathbf k}.
\]

## Later Wannier transformation

For each spin pair,

\[
A_W^{\sigma\sigma',j}(\mathbf k)
=
\left[U_\sigma^{\mathrm{eff}}(\mathbf k)\right]^\dagger
A_B^{\sigma\sigma',j}(\mathbf k)
U_{\sigma'}^{\mathrm{eff}}(\mathbf k),
\]

where

\[
U_\sigma^{\mathrm{eff}}(\mathbf k)
=U_\sigma^{\mathrm{dis}}(\mathbf k)U_\sigma(\mathbf k).
\]

If no disentanglement is performed, `U_eff=U`. The Fourier transform is

\[
A_{\mathbf R}^{\sigma\sigma',j}
=\frac{1}{N_k}\sum_{\mathbf k}
e^{-2\pi i\mathbf k_{\rm frac}\cdot\mathbf R}
A_W^{\sigma\sigma',j}(\mathbf k).
\]

For reliable interpolation away from the source mesh, inspect the decay of
`A_R`. A large norm on the Wigner-Seitz boundary means that the independently
Wannierized up/down gauges are not sufficiently compatible or that the mesh is
too coarse. Values reconstructed on the original mesh remain exact, but a
denser mesh or better aligned spin-channel gauges are then required for a
smooth band path.

## Validation

The writer checks Cartesian Hermiticity:

\[
A_j^{\sigma\sigma'}(n,m)
=
\left[A_j^{\sigma'\sigma}(m,n)\right]^* .
\]

The maximum error is printed in `OUTCAR`:

```text
LWRITE_SOI: maximum Cartesian Hermiticity error = ...
```

Checks recommended on the Linux server:

1. Compare a no-SOC calculation with `LWRITE_SOI=.FALSE.` and `.TRUE.`.
   Eigenvalues and total energy should be unchanged to numerical precision.
2. Confirm that the reported Hermiticity error is small. An error much larger
   than normal floating-point/SCF noise indicates a projector-index or MPI
   layout problem.
3. Initially test with one MPI rank. Then repeat the identical fixed-charge
   calculation with at least:
   - multiple `COMM_INB` ranks and `NPAR=1`, `KPAR=1`;
   - multiple band groups (`NPAR>1` or the corresponding `NCORE`);
   - multiple k-point groups (`KPAR>1`);
   - both band and k-point parallelism enabled.
   The resulting `.soi` files should agree to floating-point reduction
   tolerance.
4. Reconstruct the SOC Hamiltonian for the z spin axis and compare its bands
   against a fixed-charge conventional `LSORBIT=.TRUE.` calculation. This
   catches normalization, overall-sign, and `L_+`/`L_-` convention mistakes.
   The physical comparison must use `strength=+1`.
5. Repeat the comparison for at least one nontrivial magnetization direction
   after implementing the angular formulas from the paper.