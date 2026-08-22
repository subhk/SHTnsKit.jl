# SHTnsKit.jl

SHTnsKit.jl is a pure-Julia spherical harmonic transform library for scalar,
vector, and QST fields. The same mathematical conventions are available across
serial CPU, CUDA, AMDGPU, and MPI/PencilArrays backends.

[![Build Status](https://github.com/subhk/SHTnsKit.jl/workflows/CI/badge.svg)](https://github.com/subhk/SHTnsKit.jl/actions)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/SHTnsKit.jl/stable)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Highlights

- Scalar, complex, spheroidal/toroidal, and QST transforms.
- Gauss–Legendre, regular Fejér, pole-inclusive regular, and
  Driscoll–Healy grids.
- Orthonormal, four-pi, and Schmidt normalization; configurable
  Condon–Shortley phase, real-field normalization, and Robert form.
- Reusable serial and distributed plans, packed layouts, batch transforms,
  rotations, differential operators, and diagnostics.
- Vendor-native CUDA and AMDGPU array dispatch, plus MPI-distributed CPU/GPU
  paths with explicitly tracked parity status.
- ForwardDiff, Zygote, ChainRulesCore, and LoopVectorization extensions.

See the [SHTns 3.7 parity matrix](shtns37-parity.md) for the executable
capability inventory and backend certification status.

## Install

```julia
using Pkg
Pkg.add("SHTnsKit")
```

Optional functionality is activated by loading its dependencies:

```julia
Pkg.add(["CUDA", "GPUArrays", "KernelAbstractions"])       # NVIDIA
Pkg.add(["AMDGPU", "GPUArrays", "KernelAbstractions"])     # AMD
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])             # distributed
```

## First transform

```@example home-roundtrip
using SHTnsKit

cfg = create_gauss_config(16, 18)

# Start with band-limited coefficients so analysis/synthesis is an exact
# roundtrip at the configured resolution.
coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
coefficients[3, 1] = 1.0
field = synthesis(cfg, coefficients)
recovered = analysis(cfg, field)

@assert maximum(abs, recovered - coefficients) < 1e-12
nothing
```

For repeated transforms, preallocate outputs and reuse an [`SHTPlan`](@ref):

```@example home-plan
using SHTnsKit

cfg = create_gauss_config(16, 18)
plan = SHTPlan(cfg; use_rfft=true)
field = rand(cfg.nlat, cfg.nlon)
coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
reconstructed = zeros(cfg.nlat, cfg.nlon)

analysis!(plan, coefficients, field)
synthesis!(plan, reconstructed, coefficients)
size(reconstructed)
```

!!! note "GPU dispatch"
    `analysis(cfg, device_array)` and `synthesis(cfg, device_coefficients)` keep
    data on the detected CUDA or AMDGPU device. Strict `GPU()` dispatch never
    silently falls back to the CPU. See [GPU Acceleration](gpu.md) for device
    transfer and fallback details.

## Where to go next

- [Quick Start](quickstart.md) — scalar, vector, planned, and batch workflows.
- [Grid Types](grids.md) — sampling constraints and constructor choices.
- [Normalization and Phase](norms.md) — coefficient conventions.
- [Migrating to v2.0](migration.md) — breaking API and numerical changes.
- [GPU Acceleration](gpu.md) — CUDA and AMDGPU execution.
- [Distributed Computing](distributed.md) — MPI/PencilArrays transforms.
- [API Reference](api/index.md) — generated public API documentation.

```@contents
Pages = [
    "quickstart.md",
    "grids.md",
    "norms.md",
    "migration.md",
    "gpu.md",
    "distributed.md",
    "api/index.md",
]
Depth = 1
```
