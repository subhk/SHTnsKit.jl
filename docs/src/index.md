# SHTnsKit.jl

SHTnsKit.jl transforms scalar and vector fields between values on a spherical
grid and spherical-harmonic coefficients. It is written in Julia and supports
the same transform conventions on CPUs, NVIDIA and AMD GPUs, and MPI-distributed
arrays.

## What it provides

- Scalar, tangential-vector, and three-component QST transforms.
- Gauss–Legendre and equiangular grids, with explicit quadrature conventions.
- Reusable plans, batch transforms, rotations, operators, and energy spectra.
- Optional CUDA, AMDGPU, MPI, automatic-differentiation, and SIMD extensions.

## Install

```julia
using Pkg
Pkg.add("SHTnsKit")
```

See [Installation](installation.md) only if you need a GPU, MPI, or help with
setup.

## First transform

Start with known band-limited coefficients so the roundtrip has an exact answer:

```@example home-roundtrip
using SHTnsKit

cfg = create_gauss_config(16, 18)
coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
coefficients[3, 1] = 1.0             # degree l=2, order m=0
coefficients[5, 3] = 0.25 - 0.1im   # degree l=4, order m=2

field = synthesis(cfg, coefficients)
recovered = analysis(cfg, field)

@assert maximum(abs, recovered - coefficients) < 1e-12
size(field)
```

Spatial fields use `(latitude, longitude)` order. Dense coefficients use
`(l + 1, m + 1)` indexing because Julia arrays start at one.

## Choose your path

| I want to… | Read… |
|---|---|
| understand the basic arrays and transform direction | [Quick Start](quickstart.md) |
| choose the right spherical sampling | [Grid Types](grids.md) |
| adapt a working scientific recipe | [Examples Gallery](examples/index.md) |
| keep transforms on an NVIDIA or AMD GPU | [GPU Acceleration](gpu.md) |
| distribute fields across MPI ranks | [Distributed Computing](distributed.md) |
| make repeated transforms faster | [Performance Guide](performance.md) |
| use packed storage, operators, rotations, or AD | [Advanced Usage](advanced.md) |
| exchange coefficients with another library | [Normalization and Phase](norms.md) |

The [API Reference](api/index.md) lists the complete public surface. Most users
can begin with the default Gauss–Legendre grid and orthonormal convention.
