# API Reference

```@raw html
<div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
    <h2 style="margin: 0 0 0.5rem 0; color: white; border: none;">API Reference</h2>
    <p style="margin: 0; opacity: 0.9;">Complete function and type documentation</p>
</div>
```

Complete reference for all SHTnsKit.jl functions and types.

## Configuration Management

### Configuration Creation

```@docs
create_config
create_gauss_config
create_regular_config
destroy_config
```

### Index Utilities

```@docs
nlm_calc
LM_index
```

## Scalar Field Transforms

### Forward Transform (Synthesis)

```@docs
synthesis
synthesis!
```

### Backward Transform (Analysis)

```@docs
analysis
analysis!
```

## Complex Field Transforms

```@docs
synthesis_packed_cplx
analysis_packed_cplx
```

## Vector Field Transforms

Vector fields on the sphere are decomposed into **spheroidal** and **toroidal** components:
- **Spheroidal**: Poloidal component (has radial component)
- **Toroidal**: Azimuthal component (purely horizontal)

### Vector Synthesis

```@docs
synthesis_sphtor
```

### Vector Analysis

```@docs
analysis_sphtor
```

## QST Transforms (3D Vector Fields)

```@docs
synthesis_qst
analysis_qst
```

## Rotations

```@docs
SH_Zrotate
SH_Yrotate
SH_Yrotate90
SH_Xrotate90
```

## Energy/Power Spectrum Analysis

```@docs
energy_scalar_l_spectrum
energy_vector_l_spectrum
energy_scalar
energy_vector
enstrophy
```

## Threading Control

```@docs
shtns_use_threads
```

## Buffer Helpers

```@docs
scratch_spatial
scratch_fft
```

## Distributed Transforms (MPI)

When using MPI with PencilArrays, the following functions are available via the parallel extension:

- `dist_analysis(cfg, fθφ)` - Distributed spatial to spectral transform
- `dist_synthesis(cfg, Alm; prototype_θφ, real_output)` - Distributed spectral to spatial transform
- `dist_analysis_sphtor(cfg, Vθ, Vφ)` - Distributed vector analysis
- `dist_synthesis_sphtor(cfg, Slm, Tlm; prototype_θφ)` - Distributed vector synthesis

See the [Distributed Guide](../distributed.md) for detailed usage.

## Gradient and Differential Operators

```@docs
synthesis_grad
divergence_from_spheroidal
vorticity_from_toroidal
```

## Usage Examples

### Basic Transform

```julia
using SHTnsKit

lmax = 16
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create test pattern
spatial = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat
    x = cfg.x[i]
    spatial[i, :] .= (3*x^2 - 1)/2
end

# Transform roundtrip
Alm = analysis(cfg, spatial)
recovered = synthesis(cfg, Alm)

destroy_config(cfg)
```

### Vector Field Transform

```julia
using SHTnsKit

lmax = 32
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create velocity field
Vθ = zeros(cfg.nlat, cfg.nlon)
Vφ = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ = cfg.θ[i]
    φ = cfg.φ[j]
    Vθ[i,j] = cos(θ) * sin(φ)
    Vφ[i,j] = cos(φ)
end

# Decompose into spheroidal/toroidal
Slm, Tlm = analysis_sphtor(cfg, Vθ, Vφ)

# Reconstruct
Vθ_rec, Vφ_rec = synthesis_sphtor(cfg, Slm, Tlm)

destroy_config(cfg)
```

## Error Handling

### Common Errors

- **`BoundsError`**: Invalid lmax/mmax values
- **`AssertionError` / `DimensionMismatch`**: Array size mismatches

### Best Practices

```julia
# Always check array sizes
@assert size(Alm) == (cfg.lmax+1, cfg.mmax+1) "Wrong spectral array size"
@assert size(spatial) == (cfg.nlat, cfg.nlon) "Wrong spatial array size"

# Always destroy configurations
cfg = create_gauss_config(32, 34; nlon=65)
# ... work with cfg ...
destroy_config(cfg)
```
