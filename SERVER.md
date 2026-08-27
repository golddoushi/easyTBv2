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

## Parallelism Diagnosis

The Linux machine itself is not the immediate cause of low utilization. The
FeCl2 IFT calculation used an outer BZ phase grid with 81 tasks
(`Phase grid: [9 9 1]`) and requested `ncores=96`, yet `top` showed roughly
500-1800% CPU and `NLWP=82`: many threads existed but only about 5-18 cores
were productive.

The likely cause is Python-thread execution in the LG density-matrix/IFT path:
substantial Python-level work remains under the GIL, so thread-level BZ task
parallelism cannot scale like the legacy numerical path. The legacy `solveSHC`
path did reach about 9533% CPU on the same host, so scheduler access and CPU
allocation are healthy.

For future optimization, prioritize process-level BZ parallelism, with each
worker initializing or receiving the read-only model/calculator state safely,
then reduce partial tensors in the parent. Avoid assuming that increasing
`ncores` in the current threaded `BZGridTask` will improve LG/VG density-matrix
calculators.

## Remote Safety And Practical Notes

- `rg` was not available on the server; use `grep`/`find` as needed.
- Do not use plain non-interactive `python3`; see the interpreter section.
- Treat active remote calculations as user work. Check the exact PID before
  stopping anything, and only terminate a run when the user explicitly asks.
- The project files on the server may differ from the Windows checkout; inspect
  the remote configuration before modifying or benchmarking it.
- The empty marker file `~/easyTBv2/mcalc/FeCl2/SUCCESS` was created during an
  earlier connection test. It is harmless and confirms prior remote access.
