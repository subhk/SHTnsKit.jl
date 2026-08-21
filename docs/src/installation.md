# Installation

SHTnsKit 2.0 supports Julia 1.10, 1.11, and 1.12 on Linux, macOS, and
Windows. The serial package is pure Julia; FFTW.jl is installed automatically.

## Core package

```julia
using Pkg
Pkg.add("SHTnsKit")
```

Verify the installation with a band-limited roundtrip:

```@example installation-core
using SHTnsKit

cfg = create_gauss_config(8, 10)
coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
coefficients[3, 2] = 0.5 - 0.25im
field = synthesis(cfg, coefficients)
recovered = analysis(cfg, field)

@assert maximum(abs, recovered - coefficients) < 1e-12
(lmax=cfg.lmax, nlat=cfg.nlat, nlon=cfg.nlon)
```

`SHTConfig` values are managed by Julia's garbage collector.
[`destroy_config`](@ref) remains as a no-op compatibility function; explicit
teardown is not required.

## Optional extensions

SHTnsKit uses Julia package extensions. Install and load the packages in the
row you need; no build step is required for SHTnsKit itself.

| Capability | Packages to add and load |
|---|---|
| NVIDIA GPU | `CUDA`, `GPUArrays`, `GPUArraysCore`, `KernelAbstractions` |
| AMD GPU | `AMDGPU`, `GPUArrays`, `GPUArraysCore`, `KernelAbstractions` |
| MPI distributed | `MPI`, `PencilArrays`, `PencilFFTs` |
| MPI + NVIDIA | MPI row plus CUDA row |
| MPI + AMD | MPI row plus AMDGPU row |
| SIMD helpers | `LoopVectorization` |
| Forward-mode AD | `ForwardDiff` |
| Reverse-mode AD | `Zygote`; add `ChainRulesCore` for advanced rules |

For example:

```julia
using Pkg

Pkg.add(["CUDA", "GPUArrays", "GPUArraysCore", "KernelAbstractions"])
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

## GPU setup

For NVIDIA:

```julia
using SHTnsKit, CUDA, GPUArrays, GPUArraysCore, KernelAbstractions
CUDA.functional() || error("CUDA is not functional")

cfg = create_gauss_config(32, 34)
field_device = CUDA.rand(Float64, cfg.nlat, cfg.nlon)
coefficients_device = analysis(cfg, field_device)
recovered_device = synthesis(cfg, coefficients_device)
```

For AMD:

```julia
using SHTnsKit, AMDGPU, GPUArrays, GPUArraysCore, KernelAbstractions
AMDGPU.functional() || error("AMDGPU is not functional")

cfg = create_gauss_config(32, 34)
field_device = AMDGPU.ROCArray(rand(Float64, cfg.nlat, cfg.nlon))
coefficients_device = analysis(cfg, field_device)
recovered_device = synthesis(cfg, coefficients_device)
```

These generic calls preserve device storage. See [GPU Acceleration](gpu.md)
for strict `GPU()` dispatch, transfer helpers, plans, and the CUDA compatibility
wrappers.

## MPI setup

```julia
using Pkg
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

MPI.jl can use its bundled MPI or a system MPI. Configure that choice through
MPI.jl, then run Julia under the matching launcher. A minimal package check is:

```bash
mpiexec -n 2 julia --project -e 'using MPI; MPI.Init(); println(MPI.Comm_rank(MPI.COMM_WORLD)); MPI.Finalize()'
```

Continue with [Distributed Computing](distributed.md) for array construction
and transform plans. All ranks must construct identical `SHTConfig` values.

## Development checkout

```julia
using Pkg
Pkg.develop(url="https://github.com/subhk/SHTnsKit.jl.git")
Pkg.test("SHTnsKit")
```

To build the web documentation from a checkout:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Common errors

### Invalid grid size

Gauss grids require `nlat >= lmax + 1` and `nlon >= 2*mmax + 1`.
Regular grids have additional constraints described in [Grid Types](grids.md).
Constructors now reject invalid combinations immediately.

### GPU backend unavailable

Strict `GPU()` dispatch raises [`BackendUnavailableError`](@ref) when no loaded
adapter is functional. Confirm that the vendor package and the three shared GPU
dependencies are loaded. If both CUDA and AMDGPU are functional, pass an
existing device array as the input or as the `to_device` prototype so the
vendor is unambiguous.

### Distributed extension not loaded

Load all three packages before calling distributed APIs:

```julia
using MPI, PencilArrays, PencilFFTs, SHTnsKit
```

### Migrating older code

If an older application uses C-style flags, symbol devices, removed getters,
or legacy plan keywords, see [Migrating to SHTnsKit v2.0](migration.md).
