# Linux Server Handoff

This note records the verified workflow for the remote Linux server used for
`easyTBv2` calculations. It is intended for a later Codex task working from
this Windows checkout.

## Connection

From PowerShell in this repository, connect with:

```powershell
ssh -i "C:\Users\86307\.ssh\codex_server" codex@58.194.174.88
```

The remote project is:

```text
/home/codex/easyTBv2
```

Use `~/easyTBv2` after logging in. The SSH key name is exactly
`codex_server` (no intermediate `codex` directory and no `.pem` extension).

The remote host previously reported the hostname `ls-MZ72-HB2-00` and has 192
logical CPUs. The `codex` account had CPU affinity to `0-191`; there was no
observed cgroup CPU restriction.

## Python Environment

For non-interactive SSH commands, do **not** invoke plain `python3`: it may
resolve to `/usr/bin/python3`, whose non-interactive environment lacks NumPy.
Use the verified interpreter explicitly:

```bash
/home/codex/soft/python312/bin/python3
```

Example:

```powershell
ssh -i "C:\Users\86307\.ssh\codex_server" codex@58.194.174.88 "cd ~/easyTBv2 && /home/codex/soft/python312/bin/python3 spinHallCondInput.py"
```

An interactive login can set a different PATH through shell startup files, so
the explicit interpreter remains the most reproducible choice.

## Run And Inspect A Calculation

The standard sample command is:

```bash
cd ~/easyTBv2
/home/codex/soft/python312/bin/python3 spinHallCondInput.py
```

Before an expensive run, inspect the active numerical settings rather than
assuming they match the local checkout:

```bash
grep -nE 'fermiAxis|omegaAxis|BZGrid=' spinHallCondInput.py | head -n 24
```

For example, the reduced FeCl2 sample used:

```python
fermiAxis = np.array([fermi,])
omegaAxis = np.arange(0.1, 3.05, 0.5)
BZGrid = (99, 99, 1)
```

Check whether a run is still active:

```bash
ps -eo pid,ppid,nlwp,pcpu,etime,cmd | grep '[s]pinHallCondInput.py'
```

Inspect fresh FeCl2 outputs after completion:

```bash
cd ~/easyTBv2/mcalc/FeCl2
find . -maxdepth 1 -type f -mmin -10 -printf '%TY-%Tm-%Td %TH:%TM %10s %f\n' | sort
```

Typical final files include:

```text
IFTtensor.proj0-0-0.real.txt
IFTtensor.proj0-0-0.imag.txt
IFTcoeffs.proj0-0-0.txt
IFTcoeffs.proj0-0-0.{x,y,xy+,xy-}.png
IFTtensor.proj0-0-0.{real,imag}.{xx,xy,yx,yy}.png
```

The reduced run was confirmed to finish and write these artifacts.

## Process Backend And Measurements (2026-08-27)

`task.py` now defaults to `parallelBackend="process"` for `BZGridTask` when
more than one phase task/worker is available. The model is installed once per
worker, shared copy-on-write with Linux `fork`, and is not serialized for each
phase task. `workerThreads=1` controls BLAS/OpenMP threads per worker. The
small `threadpoolctl` dependency was installed in both the remote Python312
and local Python310 environments. Results, channel tensors and `iq` placement
retain their existing API; completed results are now kept in phase order.

The previous thread-only bottleneck is confirmed by the following measured
FeCl2 IFT timings. The sample had one Fermi point, six photon energies, a
99x99x1 BZ grid, an 11x11x1 FFT grid, and 81 phase tasks. Times cover BZ task
construction and integration, excluding model loading and plotting.

| Backend | Actual workers | Wall time (s) | Average busy CPU cores |
| --- | ---: | ---: | ---: |
| Old threads (96 requested) | 81 | 36.72 | 7.68 |
| Processes | 8 | 8.63 | 7.45 |
| Processes | 16 | 5.69 | 14.50 |
| Processes | 24 | 6.27 | 22.30 |
| Processes | 48 | 7.68 | 40.04 |
| Processes (96 requested, fresh interpreter) | 81 | 10.22 | 67.98 |

CPU averages are total parent+child CPU seconds divided by wall time, not a
single instantaneous `top` reading. All benchmark tensors agreed exactly.
Windows and Linux regression checks also compared all eight LG/VG DM solver
wrappers, both full-covariant modes, channel tensors, and recorded BZ data.
Process error propagation, cleanup, thread limits, and worker caps were tested.

Use `ncores=16` as a starting point for this small sample; it was about 6.5x
faster than the old 96-thread run. Larger grids and other materials need their
own measurements. Do not optimize CPU percentage alone. Independent worker
processes remove shared-Python contention but still compete for cache/memory.
The GIL is not established as the sole cause by this timing comparison.

Example input change:

```python
calc = BZGridTask(model, solveIFTorque_LG_DM, args=settings, BZGrid=BZGrid,
                  ncores=16, parallelBackend="process", workerThreads=1)
```

`ncores=96` remains valid and uses 81 workers for the current phase grid.
Select `parallelBackend="thread"` for a thread comparison. Set
`processStartMethod="spawn"` for a parent with background/native threads or
fork-unsafe libraries. Spawn costs more memory/startup; Linux scripts normally
use fork. Keep the model read-only during a BZ run. Avoid nesting a large BZ
pool inside the angle-mapping process pool.

With the new backend, the parent Python PID mostly coordinates work. In `top`,
inspect its child processes too (often around 100% CPU each), or use:

```bash
ps -u codex -o pid,ppid,nlwp,pcpu,rss,etime,args
```

Repeat a small benchmark without rewriting the input script's response outputs:

```bash
cd ~/easyTBv2
/home/codex/soft/python312/bin/python3 -u benchmark_bz_parallel.py --case thread:96 --case process:16 --output tmp/bz_check
```

The harness intercepts `BZGridTask` in `spinHallCondInput.run()`, keeps the
actual model and settings, checks every returned field against the first case,
then stops before final tensor/figure output. Its `--output` writes separate
JSON timings and NPZ tensors. Keep the Fermi/frequency axes small. For reliable
process startup timings, also run a process-only case in a fresh interpreter:
a preceding large thread run can retain substantial allocator memory. In this
test, 81 processes after the old thread run took 24.45 s versus 10.22 s fresh.

Remote benchmark records and the original task.py backup are in:

```text
~/easyTBv2/tmp/bz_parallel_20260827/
    task_original.py
    benchmark.log, fecl2.json, fecl2.*.npz
    process_scaling.log, process_scaling.json, process_scaling.*.npz
    fresh96.log, fresh96.json, fresh96.*.npz
    tests.log
```

A local copy is in `tmp/bz_parallel_20260827/`. The remote input settings were
not replaced with the Windows input (their magnetization angles differ).
The final unchanged `spinHallCondInput.py` run with 96 requested workers exited
with status 0: BZ integration took 10.42 s, and the complete run including model
loading and figures took 25.57 s. Its logs are `final_sample.log` and
`final_sample.time` in the same benchmark directory. Tensor/coefficient files
in `mcalc/FeCl2` were refreshed at server time 2026-08-27 15:51.

## Remote Safety And Practical Notes

- `rg` was not available on the server; use `grep`/`find` as needed.
- Do not use plain non-interactive `python3`; see the interpreter section.
- Treat active remote calculations as user work. Check the exact PID before
  stopping anything, and only terminate a run when the user explicitly asks.
- The project files on the server may differ from the Windows checkout; inspect
  the remote configuration before modifying or benchmarking it.
- The empty marker file `~/easyTBv2/mcalc/FeCl2/SUCCESS` was created during an
  earlier connection test. It is harmless and confirms prior remote access.
