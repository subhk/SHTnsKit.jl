# API Reference

Complete reference for all SHTnsKit.jl functions and types.

## Configuration Management

### Configuration Creation

```@docs
create_config
create_gauss_config
create_regular_config
```

### Configuration Queries

```@docs
get_lmax
get_mmax
get_nlat
get_nphi
get_nlm
lmidx
```

### Grid Information

```@docs
get_theta
get_phi
get_gauss_weights
```

### Configuration Cleanup

```@docs
destroy_config
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
SH_to_spat_cplx
spat_cplx_to_SH
```

## Vector Field Transforms

Vector fields on the sphere are decomposed into **spheroidal** and **toroidal** components:
- **Spheroidal**: Poloidal component (has radial component)
- **Toroidal**: Azimuthal component (purely horizontal)

### Vector Synthesis

```@docs
SHsphtor_to_spat
```

### Vector Analysis

```@docs
spat_to_SHsphtor
```

## QST Transforms

```@docs
SHqst_to_spat
spat_to_SHqst
```

## Field Rotations

```@docs
rotate_real!
rotate_complex!
```

## Energy/Power Spectrum Analysis

```@docs
energy_scalar_l_spectrum
energy_vector_l_spectrum
```

## Threading Control

```@docs
set_threading!
get_threading
set_fft_threads
get_fft_threads
set_optimal_threads!
```

## Buffer Helpers

```@docs
scratch_spatial
scratch_fft
```

## Distributed Transforms (MPI)

When using MPI with PencilArrays:

```@docs
SHTnsKit.dist_analysis
SHTnsKit.dist_synthesis
SHTnsKit.dist_spat_to_SHsphtor
SHTnsKit.dist_SHsphtor_to_spat
```

## Helper Functions

```@docs
lm_from_index
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
