# Advanced Automatic Differentiation Implementation

This document summarizes AD capabilities for SHTnsKit.jl, enabling applications in scientific machine learning, PINNs, and inverse problems.

## AD Capabilities

### Supported Functions
- **Basic transforms**: `synthesis`, `analysis`, `synthesis_packed_cplx`, `analysis_packed_cplx`
- **Vector transforms**: `synthesis_sphtor`, `analysis_sphtor`
- **Diagnostics**: `energy_scalar_l_spectrum`, `energy_vector_l_spectrum`

### ForwardDiff Support
- **FFT-based transforms**: Works with ForwardDiff.Dual numbers
- **Legendre polynomials**: Recurrence relations compatible with Dual numbers

### Zygote Integration
- **Transform adjoints**: Efficient pullbacks for synthesis/analysis
- **Memory efficiency**: Optimized gradient computation

## Usage Example

### Basic AD with Transforms

```julia
using SHTnsKit, Zygote

# Setup
lmax = 32
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create test coefficients (2D matrix format)
Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Alm[1, 1] = 1.0
Alm[3, 1] = 0.5 + 0.2im

# Define a loss function using transforms
function loss_fn(Alm)
    spatial = synthesis(cfg, Alm)
    return sum(abs2, spatial)
end

# Compute gradient with Zygote
grad = Zygote.gradient(loss_fn, Alm)[1]
println("Gradient computed successfully")

destroy_config(cfg)
```

### Roundtrip Gradient Verification

```julia
using SHTnsKit, Zygote, LinearAlgebra

lmax = 16
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create test data
spatial = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ = cfg.θ[i]
    φ = cfg.φ[j]
    spatial[i,j] = sin(2θ) * cos(φ)
end

# Loss through roundtrip
function roundtrip_loss(spatial_in)
    Alm = analysis(cfg, spatial_in)
    reconstructed = synthesis(cfg, Alm)
    return sum(abs2, reconstructed - spatial_in)
end

# Gradient computation
grad = Zygote.gradient(roundtrip_loss, spatial)[1]

# Verify gradient is small (roundtrip should be exact)
println("Gradient norm: ", norm(grad))

destroy_config(cfg)
```

## Applications

### Optimization on the Sphere

```julia
using SHTnsKit, Zygote

lmax = 24
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Target field
target = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat
    x = cfg.x[i]
    target[i, :] .= (3*x^2 - 1)/2  # Y_2^0
end

# Optimize coefficients to match target
function fitting_loss(Alm)
    reconstructed = synthesis(cfg, Alm)
    return sum(abs2, reconstructed - target)
end

# Initialize coefficients
Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)

# Gradient descent
learning_rate = 0.01
for iter in 1:100
    grad = Zygote.gradient(fitting_loss, Alm)[1]
    Alm .-= learning_rate .* grad

    if iter % 20 == 0
        println("Iter $iter: loss = $(fitting_loss(Alm))")
    end
end

destroy_config(cfg)
```

## Notes

- AD works with the 2D coefficient matrix format `(lmax+1, mmax+1)`
- ForwardDiff and Zygote are both supported
- Distributed transforms (`dist_analysis`, `dist_synthesis`) are experimental for AD
