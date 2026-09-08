# SHTns 3.7 mathematical parity

SHTnsKit tracks the mathematical surface of SHTns 3.7 (`SHTNS_INTERFACE`
`0x307A0`) across CPU, CUDA, AMDGPU, MPI/CPU, MPI/CUDA, and MPI/AMDGPU.
The fixture oracles are generated independently with the public SHTns 3.7 C
API and checked by the backend-specific runners shown below.

The matrix uses only three status values:

- **verified** means an immutable CI artifact and its digest are recorded for
  the named parity runner. A local passing command is not certification.
- **required** is an executable compatibility obligation that has not yet been
  certified. Checked-in local readiness summaries may demonstrate that the
  runner is ready without changing this status.
- **unverified hardware** means the implementation and runner exist, but no
  physical hardware run is claimed by the checked-in contract.

The table and CI inventory are rendered directly from
`shtns37_capabilities()` and the contract's CI inventory. They are not copied
into this page by hand.

```@eval
using Markdown
using SHTnsKit
Markdown.parse(SHTnsKit._shtns37_parity_markdown())
```

## Manual CUDA testing on one server

`test/support/manual_cuda.py` runs the CUDA suite on your Linux x86-64 NVIDIA
server and can report the outcome to GitHub from your laptop. No Actions runner
registration is needed. The server needs Python 3.9+, Git, Julia compatible with
`test/gpu/cuda/Project.toml`, a working NVIDIA driver, and network access for Julia
dependencies. Only the reporting machine needs the GitHub CLI and authentication.
Run commits you have reviewed and trust: manual testing still executes their code
on the server.

From a checkout containing the helper on the server, select the commit to test:

```sh
git fetch origin pull/50/head
python3 test/support/manual_cuda.py run --ref FETCH_HEAD \
  --devices 0 --output "$HOME/cuda-results/pr50-run1"
```

This tests a separate clean snapshot of the selected commit, leaving the checkout
intact. `--ref` defaults to `HEAD`; uncommitted changes are excluded. The command
installs the CUDA test environment, requires functional CUDA hardware, then runs
`test/gpu/cuda/runtests.jl`. A missing GPU fails the run instead of reporting a
successful hardware skip. Progress goes to the terminal and full output goes to
`cuda-test.log` in the output directory (`tail -f` can follow it). The first run
may take a while to install and precompile dependencies.

Configuration options:

| Option | Purpose |
| --- | --- |
| `--repo /path/to/SHTnsKit.jl` | Source checkout; defaults to the helper's repository. |
| `--ref COMMIT_OR_REF` | Commit to test; resolved to a full SHA before execution. |
| `--julia /path/to/julia` | Julia executable; defaults to `julia` on `PATH`. |
| `--devices 0` | Set `CUDA_VISIBLE_DEVICES`; otherwise inherit the environment. |
| `--output /path/to/new-directory` | Required result destination; use a fresh directory for each run. |
| `--mpi --devices 0,1` | Also run MPI/CUDA parity with two ranks and two visible GPUs on the same box. |

The default suite needs one GPU. The optional MPI suite includes the single-GPU
suite and `test/gpu/cuda/mpi_runtests.jl`; it does not require a second server.
Ordinary dependency or test failures still produce a reportable `result.json`
and log, and return a nonzero exit code. An interrupted/incomplete record cannot
be published as a success.

Copy the entire result directory to your laptop, replacing `user@gpu-server` with
your SSH destination. The laptop also needs a checkout containing the helper:

```sh
scp -r user@gpu-server:cuda-results/pr50-run1 ./pr50-run1
python3 test/support/manual_cuda.py report ./pr50-run1/result.json
```

The `report` command first checks the record and log digest, then prints a preview
without contacting GitHub. To publish, authenticate the GitHub CLI on the laptop
and explicitly add `--publish`:

```sh
gh auth login
python3 test/support/manual_cuda.py report ./pr50-run1/result.json \
  --repository subhk/SHTnsKit.jl --publish
```

The status appears on PRs containing the recorded commit as **CUDA / remote
manual**, or **CUDA + MPI / remote manual** when `--mpi` was selected. New commits
need a new run. Setup failures publish `error`, test failures publish `failure`,
and only a complete passing suite publishes `success`. Publishing needs repository
access that allows commit-status writes; a fine-grained token needs **Commit
statuses: write**. See the [GitHub commit statuses API](https://docs.github.com/en/rest/commits/statuses).

Logs are kept locally, not uploaded automatically. If you upload the log yourself,
pass `--target-url https://...` to link it from the status. These are manually
reported results, not signed attestations or Actions artifacts, and they do not
automatically change the capability matrix's `unverified hardware` entries.

## Dispatch and backend selection

Strict typed dispatch makes execution intent explicit:

```julia
cpu_coefficients = analysis(CPU(), cfg, host_field)
gpu_coefficients = analysis(GPU(), cfg, device_field)
distributed_coefficients = analysis(CPU(), cfg, spatial_pencil)
```

Ordinary calls infer CUDA or AMDGPU from the vendor array type, and infer the
distributed implementation from a `PencilArray`:

```julia
gpu_coefficients = analysis(cfg, device_field)
distributed_coefficients = analysis(cfg, spatial_pencil)
```

For a `PencilArray` whose parent is a CUDA or AMDGPU array, only the
`DistTransposePlan` scalar, spheroidal/toroidal, and QST bang transforms are
currently device-native. Their Legendre work stays on the device and only the
MPI transpose/all-reduce boundary may use bounded pinned staging when the MPI
library is not GPU-aware. Other ordinary MPI+GPU mathematical APIs throw
`BackendUnavailableError` before copying, mutating, or incrementing staging
counters. They remain `unverified hardware`; the package does not claim parity
for a whole-call CPU-staged implementation.

Strict `GPU()` calls report an unavailable backend rather than silently moving
data to the CPU. The legacy `gpu_analysis_safe` and `gpu_synthesis_safe`
wrappers are the explicit compatibility path when automatic host fallback is
desired; callers should account for the transfer and allocation cost.

## Covered mathematical conventions

The parity fixtures and generated sweeps cover:

- Gauss--Legendre, on-the-fly Gauss, regular Fejer, and regular grids with
  poles, including either latitude ordering.
- Orthonormal, four-pi, and Schmidt normalization; Condon--Shortley phase on
  and off; `real_norm` on and off; and Robert-form vector conventions.
- Dense `(l, m)`, SHTns-compatible packed, fixed-`l`, fixed-`m`, batched, and
  distributed `PencilArray` layouts, including representative `mres > 1`.
- `Float32`/`ComplexF32` and `Float64`/`ComplexF64` execution where the backend
  supports those types.

The SHTns grid selectors `sht_reg_fast` and `sht_reg_dct` are two planning
choices for the same Fejer mathematical grid capability. They therefore share
one parity cell rather than appearing as two mathematical features.

## Scope exclusions

This is a mathematical compatibility matrix, not an ABI emulation claim.
SHTns C configuration allocation/destruction, build and compiler options,
thread or FFT planner tuning, timing/profiling counters, and other lifecycle
or performance-control APIs are outside its scope.
