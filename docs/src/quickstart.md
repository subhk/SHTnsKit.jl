# Quick Start Guide

```@raw html
<div style="background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
    <h2 style="margin: 0 0 0.5rem 0; color: white; border: none;">Get Started in 5 Minutes</h2>
    <p style="margin: 0; opacity: 0.9;">From installation to your first spherical harmonic transform</p>
</div>
```

Get up and running with SHTnsKit.jl in minutes. This guide covers the essential concepts and common workflows.

---

## Installation

```julia
using Pkg
Pkg.add("SHTnsKit")
```

For GPU support, also add:
```julia
Pkg.add(["CUDA", "KernelAbstractions"])
```

---

## Core Concepts

### Two Representations of Data

Spherical harmonics provide two ways to represent functions on a sphere:

| Domain | Description | Array Shape | Best For |
|--------|-------------|-------------|----------|
| **Spatial** | Values at grid points | `(nlat, nlon)` | Visualization, physical intuition |
| **Spectral** | Coefficient amplitudes | `(lmax+1, mmax+1)` | Analysis, filtering, derivatives |

**Think of it like audio:**
- **Spatial** = the sound wave (amplitude over time)
- **Spectral** = frequency components (which notes are playing)

### Key Parameters

| Parameter | Meaning | Typical Values |
|-----------|---------|----------------|
| `lmax` | Maximum spherical harmonic degree | 32, 64, 128, 256 |
| `mmax` | Maximum azimuthal order (usually = lmax) | Same as lmax |
| `nlat` | Number of latitude points | lmax + 2 or more |
| `nlon` | Number of longitude points | 2*lmax + 1 or more |

---

## Your First Transform

```julia
using SHTnsKit

# 1. Create configuration
lmax = 32
cfg = create_gauss_config(lmax, lmax + 2)

# 2. Create a simple pattern (P_2 Legendre polynomial)
spatial = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat
    x = cfg.x[i]  # cos(θ) at this latitude
    spatial[i, :] .= (3*x^2 - 1) / 2
end

# 3. Analysis: spatial → spectral
Alm = analysis(cfg, spatial)

# 4. Synthesis: spectral → spatial
recovered = synthesis(cfg, Alm)

# 5. Check accuracy
error = maximum(abs.(spatial - recovered))
println("Roundtrip error: $error")  # Should be ~1e-14
```

**Output:**
```
Roundtrip error: 8.881784197001252e-15
```

---

## Common Workflows

### Creating Test Fields

```julia
cfg = create_gauss_config(64, 66)

# Method 1: From spherical harmonic coefficients
Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Alm[1, 1] = 1.0    # l=0, m=0: constant (mean value)
Alm[3, 1] = 0.5    # l=2, m=0: latitude variation
Alm[3, 3] = 0.3im  # l=2, m=2: longitude variation
spatial = synthesis(cfg, Alm)

# Method 2: From analytical function
spatial2 = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ = acos(cfg.x[i])           # Colatitude
    φ = 2π * (j-1) / cfg.nlon    # Longitude
    spatial2[i, j] = cos(θ) * sin(2φ)  # Some pattern
end
```

### Spectral Filtering

```julia
cfg = create_gauss_config(64, 66)
spatial = rand(cfg.nlat, cfg.nlon)

# Transform to spectral space
Alm = analysis(cfg, spatial)

# Low-pass filter: keep only l ≤ 10
l_cutoff = 10
for l in (l_cutoff+1):cfg.lmax
    for m in 0:min(l, cfg.mmax)
        Alm[l+1, m+1] = 0
    end
end

# Transform back
smoothed = synthesis(cfg, Alm)
```

### Computing Derivatives

```julia
cfg = create_gauss_config(32, 34)

# Create test function
Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Alm[5, 3] = 1.0  # Y_4^2

# Apply Laplacian: Δ Y_l^m = -l(l+1) Y_l^m
Alm_laplacian = copy(Alm)
for l in 0:cfg.lmax
    for m in 0:min(l, cfg.mmax)
        Alm_laplacian[l+1, m+1] *= -l * (l + 1)
    end
end

laplacian_field = synthesis(cfg, Alm_laplacian)
```

---

## GPU Acceleration

For large problems, use GPU acceleration:

```julia
using SHTnsKit, CUDA

# Check GPU availability
println("GPU available: ", CUDA.functional())

cfg = create_gauss_config(128, 130)
spatial = rand(cfg.nlat, cfg.nlon)

# GPU transforms
Alm = gpu_analysis(cfg, spatial)
recovered = gpu_synthesis(cfg, Alm)

# Safe version (auto-fallback to CPU if GPU fails)
Alm_safe = gpu_analysis_safe(cfg, spatial)
```

### When to Use GPU

GPU acceleration is most beneficial for larger problems. As a general guideline:
- **lmax < 32**: CPU is typically faster due to data transfer overhead
- **lmax 32-128**: GPU becomes beneficial
- **lmax > 128**: GPU strongly recommended

---

## Vector Fields

Decompose vector fields into spheroidal (divergent) and toroidal (rotational) components:

```julia
cfg = create_gauss_config(32, 34)

# Create vector field (θ and φ components)
vθ = rand(cfg.nlat, cfg.nlon)
vφ = rand(cfg.nlat, cfg.nlon)

# Decompose: spatial → spectral
Slm, Tlm = analysis_sphtor(cfg, vθ, vφ)

# Reconstruct: spectral → spatial
vθ_out, vφ_out = synthesis_sphtor(cfg, Slm, Tlm)

# Check accuracy
println("θ error: ", maximum(abs.(vθ - vθ_out)))
println("φ error: ", maximum(abs.(vφ - vφ_out)))
```

### Physical Meaning

| Component | Physical Meaning | Examples |
|-----------|------------------|----------|
| **Spheroidal (S)** | Divergent/compressible flow | Pressure gradients, density waves |
| **Toroidal (T)** | Rotational/incompressible flow | Vortices, circulation patterns |

---

## Grid Types

### Gauss-Legendre Grid (Recommended)

```julia
cfg = create_gauss_config(lmax, nlat)
```

- **Points**: Non-uniform spacing (denser near poles)
- **Accuracy**: Optimal for spectral transforms
- **Use for**: Most scientific applications

### Regular (Equiangular) Grid

```julia
cfg = create_regular_config(lmax, nlat)
```

- **Points**: Uniform spacing in θ and φ
- **Accuracy**: Slightly lower than Gauss
- **Use for**: Visualization, interfacing with GIS data

---

## Performance Tips

### 1. Preallocate Arrays

```julia
# Allocate once, reuse
spatial_buffer = zeros(cfg.nlat, cfg.nlon)
Alm_buffer = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)

# Multiple transforms without allocation
for i in 1:100
    fill!(spatial_buffer, 0)
    # ... fill with data ...
    Alm_buffer .= analysis(cfg, spatial_buffer)
end
```

### 2. Use In-Place Operations

```julia
# Out-of-place (allocates new array)
Alm = analysis(cfg, spatial)

# In-place (writes to existing array)
analysis!(cfg, spatial, Alm)
```

### 3. Choose Appropriate Resolution

Rule of thumb for accuracy:
- `nlat ≥ lmax + 2` (minimum)
- `nlat ≈ 3/2 * lmax` (comfortable margin)
- `nlon ≥ 2*lmax + 1` (Nyquist for longitude)

---

## Unified Loop Abstraction

For custom operations that need to work on both CPU and GPU:

```julia
using SHTnsKit

A = rand(100, 100)
B = similar(A)

# Works on CPU arrays (uses SIMD)
@sht_loop B[I] = sin(A[I]) over I ∈ CartesianIndices(A)

# Same code works on GPU arrays (uses CUDA kernels)
using CUDA
A_gpu = CuArray(A)
B_gpu = similar(A_gpu)
@sht_loop B_gpu[I] = sin(A_gpu[I]) over I ∈ CartesianIndices(A_gpu)
```

### Helper Functions

```julia
# Iterate over spectral coefficients
for idx in spectral_range(lmax, mmax)
    l, m = idx[1] - 1, idx[2] - 1
    # ... work with Alm[l+1, m+1] ...
end

# Iterate over spatial grid
for idx in spatial_range(nlat, nlon)
    i_lat, i_lon = idx[1], idx[2]
    # ... work with field[i_lat, i_lon] ...
end
```

---

## Quick Reference

### Configuration

```julia
# Gauss-Legendre grid (recommended)
cfg = create_gauss_config(lmax, nlat; nlon=nlon, mmax=mmax)

# Regular grid
cfg = create_regular_config(lmax, nlat; nlon=nlon)
```

### Scalar Transforms

```julia
# Forward (spectral → spatial)
spatial = synthesis(cfg, Alm)
spatial = synthesis(cfg, Alm; real_output=false)  # Complex output

# Backward (spatial → spectral)
Alm = analysis(cfg, spatial)
```

### Vector Transforms

```julia
# Spheroidal-Toroidal decomposition
Slm, Tlm = analysis_sphtor(cfg, vθ, vφ)
vθ, vφ = synthesis_sphtor(cfg, Slm, Tlm)

# Gradient
dθ, dφ = synthesis_grad(cfg, Alm)
```

### GPU Transforms

```julia
Alm = gpu_analysis(cfg, spatial)
spatial = gpu_synthesis(cfg, Alm)
Alm = gpu_analysis_safe(cfg, spatial)  # With CPU fallback
```

### Operators

```julia
# Laplacian in spectral space: Δf_lm = -l(l+1) f_lm
for l in 0:cfg.lmax, m in 0:min(l, cfg.mmax)
    Alm[l+1, m+1] *= -l * (l + 1)
end
```

---

## Next Steps

- **[GPU Guide](gpu.md)**: Detailed GPU acceleration documentation
- **[Distributed Guide](distributed.md)**: MPI parallelization
- **[API Reference](api/index.md)**: Complete function documentation
- **[Examples](examples/index.md)**: Real-world applications
- **[Performance Guide](performance.md)**: Optimization strategies
