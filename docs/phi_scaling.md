# φ-Scaling in SHTnsKit

## Overview

The `phi_scale` field in `SHTConfig` controls how the longitude (φ) dimension is scaled during FFT operations in spherical harmonic transforms. This is critical for ensuring correct round-trip accuracy.

## Scaling Modes

### `:dft` - DFT Scaling
- **Value**: `inv_scale = nlon`
- **Use cases**:
  - Gauss-Legendre grids (default)
  - Driscoll-Healy grids
- **Rationale**: These grids use exact quadrature in latitude with FFT-based longitude integration

### `:quad` - Quadrature Scaling
- **Value**: `inv_scale = nlon / (2π)`
- **Use cases**:
  - Regular/equiangular grids without special quadrature
  - Regular grids with poles (simple trapezoidal rule)
- **Rationale**: Adjusts for the φ integration measure `dφ` where ∫₀²ᵖ f dφ ≈ (2π/nlon) Σ f_j

### `:auto` - Automatic Selection
- **Behavior**: Chooses based on `grid_type`
  - `:gauss` → `:dft`
  - `:driscoll_healy` → `:dft`
  - `:regular`, `:regular_poles` → `:quad`

## Configuration

### In Code
```julia
# Explicit control
cfg = create_gauss_config(lmax, nlat; phi_scale=:dft)
cfg = create_regular_config(lmax, nlat; phi_scale=:quad)

# Automatic (recommended)
cfg = create_gauss_config(lmax, nlat)  # Uses :dft
cfg = create_regular_config(lmax, nlat)  # Uses :quad
```

### Via Environment Variable
```bash
# Override for all grids
export SHTNSKIT_PHI_SCALE=dft
export SHTNSKIT_PHI_SCALE=quad
```

## Implementation Details

The scaling is applied in `phi_inv_scale(cfg)` which is called during synthesis:

```julia
function phi_inv_scale(cfg::SHTConfig)
    # 1. Check environment variable override
    mode = get(ENV, "SHTNSKIT_PHI_SCALE", "")
    if mode == "quad"
        return cfg.nlon / (2π)
    elseif mode == "dft"
        return cfg.nlon
    end

    # 2. Use config-specified mode
    if cfg.phi_scale === :quad
        return cfg.nlon / (2π)
    elseif cfg.phi_scale === :dft
        return cfg.nlon
    end

    # 3. Fall back to grid-type heuristic
    return cfg.grid_type == :gauss ? cfg.nlon : cfg.nlon / (2π)
end
```

## Why This Matters

Incorrect φ-scaling leads to round-trip errors:
- Analysis transforms spatial grid → spectral coefficients
- Synthesis transforms spectral coefficients → spatial grid
- Round-trip: `alm_rt = analysis(synthesis(alm))` should satisfy `alm_rt ≈ alm`

The φ-scaling factor must match the quadrature weight convention to ensure:
```
∫∫ f(θ,φ) Y_lm(θ,φ) sin(θ) dθ dφ ≈ Σᵢⱼ f[i,j] Y_lm[i,j] w[i] * φ_scale
```

## History

- Original: All grids used DFT scaling (`nlon`)
- Commit fc1d114: Regular grids changed to quadrature scaling (`nlon/(2π)`)
- Commit 2441db0: Formalized with auto-detection
- Current: Explicit `phi_scale` field for clarity and control
