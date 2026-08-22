# Examples Gallery

These six copy-ready examples cover the workflows most users need. Every Julia
block is executed by the package test suite so it cannot silently drift from
the public API.

| Task | Example |
|---|---|
| transform a scalar field | [Scalar roundtrip](#Scalar-roundtrip) |
| inspect energy by degree | [Power spectrum](#Power-spectrum) |
| decompose a tangential vector field | [Vector decomposition](#Vector-decomposition) |
| recover a stream function from vorticity | [Stream function](#Stream-function) |
| measure local performance | [Benchmark repeated transforms](#Benchmark-repeated-transforms) |
| change coordinate orientation | [Rotate a field](#Rotate-a-field) |

## Scalar roundtrip

This latitude-only temperature pattern is band-limited, so the configured
transform can reconstruct it to floating-point accuracy:

```julia
using SHTnsKit

cfg = create_gauss_config(16, 18; nlon=33)
temperature = [
    273.15 + 30 * (1 - cfg.x[i]^2)
    for i in 1:cfg.nlat, _ in 1:cfg.nlon
]

coefficients = analysis(cfg, temperature)
reconstructed = synthesis(cfg, coefficients)
error = maximum(abs, reconstructed - temperature)

@assert error < 1e-10
println("roundtrip error: $error")
```

Use the same `(latitude, longitude)` layout for measured data. Its roundtrip is
the band-limited projection represented by `lmax` and `mmax`.

## Power spectrum

[`energy_scalar_l_spectrum`](@ref) groups spectral energy by spherical-harmonic
degree `l`:

```julia
using SHTnsKit

cfg = create_gauss_config(32, 34; nlon=65)
coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
coefficients[3, 1] = 2.0          # l=2, m=0: dominant large scale
coefficients[7, 4] = 0.5 - 0.1im # l=6, m=3: weaker small scale

field = synthesis(cfg, coefficients)
power = energy_scalar_l_spectrum(cfg, analysis(cfg, field))
peak_degree = argmax(power) - 1

@assert peak_degree == 2
println("peak energy occurs at degree l=$peak_degree")
```

## Vector decomposition

Tangential components are ordered `(Vθ, Vφ)`: increasing colatitude first,
increasing longitude second. Analysis separates divergent spheroidal modes
from rotational toroidal modes:

```julia
using SHTnsKit

cfg = create_gauss_config(64, 66; nlon=129)
Vθ = zeros(cfg.nlat, cfg.nlon)
Vφ = zeros(cfg.nlat, cfg.nlon)

for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ, φ = cfg.θ[i], cfg.φ[j]
    Vθ[i, j] = 5 * cos(θ) * cos(φ)
    Vφ[i, j] = -5 * sin(φ) + 20 * sin(θ)
end

S, T = analysis_sphtor(cfg, Vθ, Vφ)
Vθ_reconstructed, Vφ_reconstructed = synthesis_sphtor(cfg, S, T)
velocity_error = max(
    maximum(abs, Vθ_reconstructed - Vθ),
    maximum(abs, Vφ_reconstructed - Vφ),
)

@assert velocity_error < 1e-10
println("vector roundtrip error: $velocity_error")
```

## Stream function

For a vorticity mode `ζ[l,m]`, solve `-l(l+1)ψ[l,m] = ζ[l,m]` in spectral
space, leaving the undetermined constant mode at zero:

```julia
using SHTnsKit

cfg = create_gauss_config(24, 26; nlon=49)
vorticity_coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
vorticity_coefficients[5, 3] = 1.0 - 0.25im # l=4, m=2

stream_coefficients = similar(vorticity_coefficients)
fill!(stream_coefficients, 0)
for l in 1:cfg.lmax, m in 0:min(l, cfg.mmax)
    stream_coefficients[l + 1, m + 1] =
        -vorticity_coefficients[l + 1, m + 1] / (l * (l + 1))
end

stream_function = synthesis(cfg, stream_coefficients)
zero_spheroidal = zero(stream_coefficients)
Vθ, Vφ = synthesis_sphtor(cfg, zero_spheroidal, stream_coefficients)

@assert all(isfinite, stream_function)
println("stream-function range: ", extrema(stream_function))
println("maximum flow speed: ", maximum(hypot.(Vθ, Vφ)))
```

## Benchmark repeated transforms

BenchmarkTools handles warmup and repeated samples. Keep setup values outside
the timed expression and always pair a timing with an accuracy check:

```julia
using SHTnsKit, BenchmarkTools

cfg = create_gauss_config(64, 66; nlon=129)
spatial = [
    sin(2cfg.θ[i]) * cos(cfg.φ[j]) +
    0.5 * sin(cfg.θ[i])^3 * cos(3cfg.φ[j])
    for i in 1:cfg.nlat, j in 1:cfg.nlon
]

analysis_time = @belapsed analysis($cfg, $spatial) seconds=0.2
coefficients = analysis(cfg, spatial)
synthesis_time = @belapsed synthesis($cfg, $coefficients) seconds=0.2

recovered = synthesis(cfg, coefficients)
max_error = maximum(abs, recovered - spatial)
@assert max_error < 1e-10

println("analysis: $(round(analysis_time * 1e3, digits=2)) ms")
println("synthesis: $(round(synthesis_time * 1e3, digits=2)) ms")
```

For a long-running application, move next to reusable plans and batch calls in
the [Performance Guide](../performance.md).

## Rotate a field

Reusable Euler rotations operate on packed, SHTns-compatible real-field
coefficients and preserve scalar energy:

```julia
using SHTnsKit

cfg = create_gauss_config(32, 34; nlon=65)
input = zeros(ComplexF64, cfg.nlm)
input[LM_index(cfg.lmax, cfg.mres, 3, 2) + 1] = 1.0

rotation = SHTRotation(cfg.lmax, cfg.mmax)
shtns_rotation_set_angles_ZYZ(rotation, π / 4, π / 6, π / 8)
rotated = similar(input)
shtns_rotation_apply_real(rotation, input, rotated)

rotated_field = reshape(synthesis_packed(cfg, rotated), cfg.nlat, cfg.nlon)
orig_power = energy_scalar_packed(cfg, input)
rot_power = energy_scalar_packed(cfg, rotated)

@assert isapprox(orig_power, rot_power; rtol=1e-10)
println("rotated field range: ", extrema(rotated_field))
```

## Scale out with MPI

The repository keeps two longer multi-process programs as executable scripts:

```bash
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl
mpiexec -n 2 julia --project=. examples/parallel_fft_roundtrip.jl
```

See [Distributed Computing](../distributed.md) for package setup, data layout,
and the choice between replicated and distributed spectral coefficients.
