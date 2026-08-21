# Quick Start

This guide covers the stable v2 workflow: construct a configuration, transform
between spatial and dense spectral arrays, and opt into plans or batches when
the same grid is reused.

## Data layout

For an [`SHTConfig`](@ref) named `cfg`:

| Representation | Shape | Indexing |
|---|---:|---|
| Spatial field | `(cfg.nlat, cfg.nlon)` | latitude × longitude |
| Dense coefficients | `(cfg.lmax + 1, cfg.mmax + 1)` | `alm[l + 1, m + 1]` |
| Batch of fields | `(cfg.nlat, cfg.nlon, nfields)` | field index last |
| Batch of coefficients | `(cfg.lmax + 1, cfg.mmax + 1, nfields)` | field index last |

Dense real-field storage keeps non-negative `m`; entries with `l < m` are
unused. Packed APIs use the SHTns-compatible `cfg.nlm` layout instead.

## Scalar roundtrip

```@example quickstart-scalar
using SHTnsKit

lmax = 32
cfg = create_gauss_config(lmax, lmax + 2)

alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
alm[3, 1] = 1.0
alm[5, 3] = 0.25 - 0.1im

field = synthesis(cfg, alm)
alm_recovered = analysis(cfg, field)
@assert maximum(abs, alm_recovered - alm) < 1e-12
nothing
```

Use `synthesis(...; real_output=false)` or [`synthesis_cplx`](@ref) when the
desired spatial output is genuinely complex.

## In-place and planned transforms

Output arguments come before input arguments:

```@example quickstart-inplace
using SHTnsKit

cfg = create_gauss_config(32, 34)
field = rand(cfg.nlat, cfg.nlon)
alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
field_out = zeros(cfg.nlat, cfg.nlon)

fft_scratch = scratch_fft(cfg)
analysis!(cfg, alm, field; fft_scratch)
synthesis!(cfg, field_out, alm; fft_scratch)

plan = SHTPlan(cfg; use_rfft=true)
analysis!(plan, alm, field)
synthesis!(plan, field_out, alm)
size(field_out)
```

A plan owns mutable scratch and is not safe for simultaneous use by multiple
threads. Create one plan per worker or task when calls may overlap.

## Vector fields

Tangential fields use spheroidal (`S`) and toroidal (`T`) coefficients:

```@example quickstart-vector
using SHTnsKit

cfg = create_gauss_config(24, 26)
S = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
T = zeros(ComplexF64, size(S))
S[4, 2] = 0.5 + 0.2im
T[5, 3] = -0.3im

vtheta, vphi = synthesis_sphtor(cfg, S, T)
S_recovered, T_recovered = analysis_sphtor(cfg, vtheta, vphi)
@assert maximum(abs, S_recovered - S) < 1e-11
@assert maximum(abs, T_recovered - T) < 1e-11
nothing
```

For three-component vector fields use [`analysis_qst`](@ref) and
[`synthesis_qst`](@ref).

## Batch transforms

```@example quickstart-batch
using SHTnsKit

cfg = create_gauss_config(16, 18)
coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1, 3)
coefficients[2, 1, 1] = 1
coefficients[3, 2, 2] = 0.5im
coefficients[4, 1, 3] = -0.25

fields = synthesis_batch(cfg, coefficients)
recovered = analysis_batch(cfg, fields)
size(recovered)
```

## Conventions and grids

All transform families honor convention keywords stored in the configuration:

```@example quickstart-conventions
using SHTnsKit

cfg = create_gauss_config(
    32, 34;
    norm=:schmidt,
    cs_phase=false,
    real_norm=true,
)
(cfg.norm, cfg.cs_phase, cfg.real_norm)
```

Use [`create_regular_config`](@ref) for equiangular sampling. See
[Grid Types](grids.md) for latitude constraints and [Normalization and
Phase](norms.md) before exchanging coefficients with another library.

## Device selection

CPU execution can be explicit or inferred:

```@example quickstart-device
using SHTnsKit

cfg = create_gauss_config(8, 10)
field = zeros(cfg.nlat, cfg.nlon)
@assert analysis(cfg, field) == analysis(CPU(), cfg, field)
on_device(field)
```

For CUDA or AMDGPU, move the input to the device and call the same `analysis`
and `synthesis` functions. See [GPU Acceleration](gpu.md).

## Next steps

- [Performance Guide](performance.md) for plans, scratch, tables, and batches.
- [Advanced Usage](advanced.md) for packed layouts, operators, and diagnostics.
- [Distributed Computing](distributed.md) for MPI execution.
- [API Reference](api/index.md) for method signatures.
