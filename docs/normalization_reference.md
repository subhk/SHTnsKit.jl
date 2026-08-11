# SHTnsKit Normalization Reference

This document explains the normalization and phase conventions implemented by
SHTnsKit.

## Normalization Conventions

### 1. `:orthonormal` (default)
**Description**: Orthonormal spherical harmonics
**Formula**: Y_l^m has unit norm over the unit sphere
**Integration Factor**: 4π (for analysis from spatial data)
**Real Coefficients**: Factor of 2 for m > 0 (real/imaginary separation)

### 2. `:fourpi`
**Description**: 4π normalization convention
**Formula**: Basis functions are scaled by `sqrt(4π)` relative to orthonormal
**Integration Factor**: 4π
**Synthesis**: Includes factorial corrections for proper reconstruction

### 3. `:schmidt`
**Description**: Schmidt semi-normalized harmonics
**Formula**: Removes sqrt(2) factor for m > 0 terms
**Integration Factor**: 4π with m-dependent corrections
**Usage**: Common in geophysics applications

The independent `real_norm=true` option controls the real-field coefficient
convention. The `cs_phase` option controls whether the Condon–Shortley
`(-1)^m` phase is included.

## Implementation Details

SHTnsKit computes with an internal orthonormal, Condon–Shortley basis and
converts coefficients at the transform boundary according to `cfg.norm` and
`cfg.cs_phase`. Scalar, vector, QST, packed, and batch transforms share those
configuration values.

## Mathematical Background

The spherical harmonic functions Y_l^m(θ,φ) satisfy:

∫∫ Y_l^m(θ,φ) Y_{l'}^{m'}*(θ,φ) sin(θ) dθ dφ = δ_{ll'} δ_{mm'} N_{lm}

Where N_{lm} is the normalization constant depending on the convention:

- **Orthonormal**: N_{lm} = 1
- **4π**: N_{lm} = 4π
- **Schmidt, m=0**: N_{lm} = 4π/(2l+1)
- **Schmidt, m>0**: N_{lm} = 8π/(2l+1)

## Usage Examples

```julia
# Orthonormal harmonics with the Condon–Shortley phase
cfg = create_gauss_config(10, 12; norm=:orthonormal, cs_phase=true)

# Geodesy-style 4π normalization without the Condon–Shortley phase
cfg_fourpi = create_gauss_config(10, 12; norm=:fourpi, cs_phase=false)

field = rand(cfg.nlat, cfg.nlon)
alm = analysis(cfg, field)
recovered = synthesis(cfg, alm)
```

## Notes

1. **Consistency**: Analysis and synthesis use the same configuration convention.
2. **Performance**: Conversion scale matrices are cached on the configuration.
3. **Accuracy**: Degree/order factors and phase changes are applied coefficient-wise.
