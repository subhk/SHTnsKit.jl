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
