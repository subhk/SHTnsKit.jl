# Advanced Usage

This page collects current v2 building blocks for applications that go beyond a
single allocating scalar transform. All examples use the public dense or packed
layouts documented by the package.

## Reusable plans

[`SHTPlan`](@ref) owns FFT plans and scratch for repeated scalar and vector
transforms:

```@example advanced-plan
using SHTnsKit

cfg = create_gauss_config(48, 50)
plan = SHTPlan(cfg; use_rfft=true)

field = rand(cfg.nlat, cfg.nlon)
alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
field_out = zeros(cfg.nlat, cfg.nlon)

analysis!(plan, alm, field)
synthesis!(plan, field_out, alm)
(size(alm), size(field_out))
```

For complex scalar output, allocate a complex destination and use a plan with
`use_rfft=false`. The real-FFT plan requires real input/output semantics.

!!! warning "Plan concurrency"
    A plan contains mutable workspace. Do not call the same plan concurrently;
    create one plan per thread or task.

## Packed coefficient storage

Dense coefficients are convenient for Julia indexing. Packed transforms use a
length-`cfg.nlm` vector compatible with SHTns real-field storage:

```@example advanced-packed
using SHTnsKit

cfg = create_gauss_config(24, 26)
dense = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
dense[4, 2] = 0.75 - 0.2im
field = synthesis(cfg, dense)

packed = analysis_packed(cfg, vec(field))
field_vector = synthesis_packed(cfg, packed)

@assert length(packed) == cfg.nlm
@assert length(field_vector) == cfg.nlat * cfg.nlon
@assert maximum(abs, reshape(field_vector, cfg.nlat, cfg.nlon) - field) < 1e-12
nothing
```

Use `LM_index`, `LiM_index`, `cfg.li`, and `cfg.mi` to map packed modes. The
`analysis_packed_cplx` family stores both signs of `m` for complex fields.

## Fixed-degree and fixed-order transforms

The `_l` and `_ml` families evaluate truncated or single-order transforms
without requiring an application to materialize an unrelated full result:

- `analysis_packed_l` / `synthesis_packed_l`
- `analysis_packed_ml` / `synthesis_packed_ml`
- `analysis_sphtor_l` / `synthesis_sphtor_l`
- `analysis_qst_l` / `synthesis_qst_l`
- `synthesis_point`, `synthesis_point_cplx`, `SH_to_lat`, and `SHqst_to_point`

Their coefficient convention is still the one stored in `cfg`.

## Batch transforms

Batch APIs put the field index in the third dimension:

```@example advanced-batch
using SHTnsKit

cfg = create_gauss_config(20, 22)
alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1, 4)
for k in axes(alm, 3)
    alm[k + 1, 1, k] = 1 / k
end

fields = synthesis_batch(cfg, alm; real_output=true)
recovered = analysis_batch(cfg, fields)
@assert size(fields) == (cfg.nlat, cfg.nlon, 4)
@assert maximum(abs, recovered - alm) < 1e-12
nothing
```

Vector and QST equivalents are `analysis_sphtor_batch`,
`synthesis_sphtor_batch`, `analysis_qst_batch`, and `synthesis_qst_batch`.
For non-FFTW element types, the public batch methods select the compatible
plan-free path automatically.

## Spectral vector operators

Spheroidal coefficients encode divergence and toroidal coefficients encode
vorticity. The inverse maps are also available:

```@example advanced-vector-operators
using SHTnsKit

cfg = create_gauss_config(24, 26)
S = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
T = zeros(ComplexF64, size(S))
S[5, 3] = 0.4im
T[6, 2] = -0.3

divergence = divergence_from_spheroidal(cfg, S)
vorticity = vorticity_from_toroidal(cfg, T)
S_recovered = spheroidal_from_divergence(cfg, divergence)
T_recovered = toroidal_from_vorticity(cfg, vorticity)

@assert maximum(abs, S_recovered - S) < 1e-12
@assert maximum(abs, T_recovered - T) < 1e-12
nothing
```

For general nearest-neighbor-in-degree operators, construct coefficients with
`mul_ct_matrix` or [`st_dt_matrix`](@ref), then apply them to packed storage
with [`SH_mul_mx`](@ref).

## Diagnostics

Diagnostics consume the configured coefficient convention directly:

```@example advanced-diagnostics
using SHTnsKit

cfg = create_gauss_config(24, 26; norm=:fourpi, cs_phase=false)
alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
alm[4, 2] = 0.5 + 0.25im
field = synthesis(cfg, alm)

spectral_energy = energy_scalar(cfg, alm)
grid_energy = grid_energy_scalar(cfg, field)
degree_spectrum = energy_scalar_l_spectrum(cfg, alm)

@assert spectral_energy >= 0
@assert sum(degree_spectrum) ≈ spectral_energy
(spectral_energy, grid_energy)
```

Vector energy, enstrophy, per-degree/per-order spectra, and gradient helpers
are listed in the [API Reference](api/index.md).

## Rotations

Packed rotations follow the SHTns-compatible layout:

```@example advanced-rotation
using SHTnsKit

cfg = create_gauss_config(12, 14)
input = zeros(ComplexF64, cfg.nlm)
input[LM_index(cfg.lmax, cfg.mres, 3, 2) + 1] = 1
rotated = similar(input)
restored = similar(input)

SH_Zrotate(cfg, input, 0.3, rotated)
SH_Zrotate(cfg, rotated, -0.3, restored)
@assert maximum(abs, restored - input) < 1e-12
nothing
```

Y rotations require `mres == 1` because they mix azimuthal orders. Use
[`SHTRotation`](@ref) and the `shtns_rotation_*` functions when a rotation
object will be reused.

## Axisymmetric transforms

`analysis_axisym` and `synthesis_axisym` operate on a latitude vector for
`m = 0`. In v2, analysis includes the full longitude integral; values are
`2π` times those from releases that omitted that factor. Do not apply a second
manual `2π` correction.

## Automatic differentiation

Load the matching optional dependency before using its wrappers:

```julia
using ForwardDiff, SHTnsKit

cfg = create_gauss_config(8, 10)
field = zeros(cfg.nlat, cfg.nlon)
gradient = fdgrad_scalar_energy(cfg, field)
```

Zygote wrappers include `zgrad_scalar_energy`, `zgrad_vector_energy`,
`zgrad_enstrophy_Tlm`, and rotation-angle gradients. ChainRulesCore activates
the broader rule extension. See the API docstrings for supported input storage.

## Backend-aware advanced code

Prefer ordinary dispatch on the storage type:

```julia
coefficients = analysis(cfg, field)  # CPU, CUDA, AMDGPU, or PencilArray
```

Use `analysis(CPU(), ...)` or `analysis(GPU(), ...)` when strict execution
intent matters. See [GPU Acceleration](gpu.md) and [Distributed
Computing](distributed.md) for backend constraints.
