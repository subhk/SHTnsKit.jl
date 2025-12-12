# Quick Start Guide

This tutorial will get you up and running with SHTnsKit.jl for spherical harmonic transforms in just a few minutes.

**What you'll learn:**
- Basic concepts: spectral vs spatial domains
- How to perform your first transform
- Working with real geophysical data
- Vector field analysis
- Performance optimization

**Prerequisites:** Basic Julia knowledge and familiarity with arrays and functions.

## Your First Transform

Let's start with a simple example to understand the basic workflow:

```julia
using SHTnsKit

# Step 1: Create a spherical harmonic configuration
lmax = 32        # Maximum degree (controls resolution)
mmax = 32        # Maximum order (typically same as lmax)
cfg = create_gauss_config(lmax, mmax)

# Step 2: Create simple test coefficients
# These represent the "recipe" for building a function on the sphere
sh_coeffs = zeros(ComplexF64, cfg.nlm)
sh_coeffs[1] = 1.0  # Y_0^0 constant term
if cfg.nlm > 3
    sh_coeffs[3] = 0.5  # Y_2^0 term if available
end
println("Number of coefficients: ", length(sh_coeffs))

# Step 3: Transform from spectral to spatial domain (synthesis)
# This builds the actual function values on a grid
spatial_field = synthesis(cfg, sh_coeffs)
println("Spatial field size: ", size(spatial_field))

# Step 4: Transform back to spectral domain (analysis)
# This recovers the coefficients from the spatial data
recovered_coeffs = analysis(cfg, spatial_field)

# Step 5: Check accuracy (should be very small)
error = norm(sh_coeffs - recovered_coeffs)
println("Round-trip error: $error")

# Step 6: Always clean up
destroy_config(cfg)
```

**What just happened?**
1. **Configuration**: We set up the transform parameters (resolution and grid type)
2. **Coefficients**: Created random spherical harmonic coefficients 
3. **Synthesis**: Converted coefficients → spatial values (spectral to physical)
4. **Analysis**: Converted spatial values → coefficients (physical to spectral)
5. **Verification**: The tiny error confirms the transforms are working correctly

## Understanding the Basics

### Spectral vs Spatial Domains

Understanding the two ways to represent data is key to using spherical harmonics effectively:

- **Spatial Domain**: Values at specific points on the sphere
  - Like having temperature measurements at weather stations
  - 2D array: `field[latitude, longitude]`
  - Easy to visualize and interpret physically

- **Spectral Domain**: Coefficients of mathematical basis functions (spherical harmonics)
  - Like having the "recipe" ingredients for recreating the field
  - 1D array: `coeffs[mode_index]`
  - Compact representation, efficient for analysis

**Analogy**: Think of a recipe vs a finished dish
- **Spatial** = the finished dish (what you see/taste)
- **Spectral** = the recipe (ingredients that make the dish)

```julia
cfg = create_gauss_config(16, 16)

# Spectral domain: 1D array of coefficients
nlm = cfg.nlm        # Number of (l,m) coefficients  
sh = zeros(ComplexF64, nlm)           # Initialize spectral coefficients
sh[1] = 1.0               # Set Y_0^0 = constant field (global average)
println("Spectral domain: ", length(sh), " coefficients")

# Spatial domain: 2D array of values on sphere
nlat, nphi = cfg.nlat, cfg.nlon
println("Spatial domain: $nlat × $nphi = $(nlat*nphi) grid points")

# Transform: spectral → spatial (synthesis)
spatial = synthesis(cfg, reshape(sh, cfg.lmax+1, cfg.mmax+1))
println("Result: all values should be the same (constant field)")
println("Min/max values: ", extrema(spatial))

destroy_config(cfg)
```

**Key insight**: Setting only the first coefficient (`sh[1]`) creates a perfectly constant field over the entire sphere, demonstrating how spherical harmonics work as building blocks.

### Grid Types

SHTnsKit supports different ways to arrange points on the sphere. Think of it like choosing between different types of graph paper:

```julia
# Gauss-Legendre grid (optimal for spectral accuracy)
cfg_gauss = create_gauss_config(32, 32)
println("Gauss grid: $(get_nlat(cfg_gauss)) × $(get_nphi(cfg_gauss))")

# Regular equiangular grid  
cfg_regular = create_regular_config(32, 32)
println("Regular grid: $(get_nlat(cfg_regular)) × $(get_nphi(cfg_regular))")

destroy_config(cfg_gauss)
destroy_config(cfg_regular)
```

**Which grid should you use?**

- **Gauss-Legendre grid** (`create_gauss_config`):
  - **Best for**: Most scientific applications
  - **Pros**: Optimal mathematical properties, highest accuracy
  - **Cons**: Uneven spacing (denser near poles)
  - **Use when**: You want the best accuracy and don't need uniform spacing

- **Regular grid** (`create_regular_config`):
  - **Best for**: Visualization, interfacing with other software  
  - **Pros**: Uniform spacing, easier to understand
  - **Cons**: Slightly less accurate
  - **Use when**: You need uniform spacing or are working with external data

## Working with Real Data

Now let's move beyond random numbers and work with realistic geophysical data patterns.

### Creating Realistic Test Fields

```julia
cfg = create_gauss_config(24, 24)

# Get grid coordinate matrices
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)
println("Grid coordinates:")
println("  θ (colatitude): 0 to π (north pole to south pole)")
println("  φ (longitude): 0 to 2π (around the equator)")

# Create a realistic temperature pattern
# Cold at poles, warm at equator, with some longitude variation
base_temp = 273.15  # 0°C in Kelvin
equatorial_warming = 30  # 30K warmer at equator
longitude_variation = 5   # 5K variation with longitude

temperature = @. base_temp + equatorial_warming * sin(θ)^2 + 
                 longitude_variation * cos(3*φ) * sin(θ)

println("Temperature field stats:")
println("  Min: $(minimum(temperature)) K ($(minimum(temperature)-273.15)°C)")
println("  Max: $(maximum(temperature)) K ($(maximum(temperature)-273.15)°C)")

# Analyze to get spectral coefficients
temp_coeffs = analysis(cfg, temperature)

# Find the most important modes
coeffs_magnitude = abs.(temp_coeffs)
sorted_indices = sortperm(coeffs_magnitude, rev=true)

println("\nTop 5 most important modes:")
for i in 1:5
    idx = sorted_indices[i]
    l, m = SHTnsKit.lm_from_index(cfg, idx)
    println("  Mode $i: l=$l, m=$m, magnitude=$(coeffs_magnitude[idx])")
end

destroy_config(cfg)
```

**What this shows:**
- How to create realistic geophysical patterns using trigonometric functions
- The relationship between spatial patterns and spherical harmonic modes
- How to identify which modes are most important in your data

### Physical Fields

```julia
cfg = create_gauss_config(32, 32)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Temperature-like field with equatorial maximum
temperature = 300 .+ 50 * cos.(2 * θ) .* cos.(φ)

# Transform to spectral domain
temp_sh = analysis(cfg, temperature)

# Reconstruct and compare
temp_reconstructed = synthesis(cfg, temp_sh)
reconstruction_error = norm(temperature - temp_reconstructed)
println("Temperature reconstruction error: $reconstruction_error")

destroy_config(cfg)
```

## Vector Fields

Vector fields on the sphere are decomposed into spheroidal and toroidal components:

```julia
cfg = create_gauss_config(20, 20)

# Create simple spheroidal and toroidal coefficients
S_lm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
T_lm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
S_lm[1, 1] = 1.0  # Simple spheroidal mode
T_lm[2, 1] = 0.5  # Simple toroidal mode

# Synthesize vector field components
V_theta, V_phi = SHsphtor_to_spat(cfg, S_lm, T_lm)

println("Vector field size: ", size(V_theta), " and ", size(V_phi))

# Analyze back to get coefficients
S_recovered, T_recovered = spat_to_SHsphtor(cfg, V_theta, V_phi)

# Check accuracy
S_error = norm(S_lm - S_recovered)
T_error = norm(T_lm - T_recovered)
println("Spheroidal error: $S_error, Toroidal error: $T_error")

destroy_config(cfg)
```

### Gradient and Curl

```julia
cfg = create_gauss_config(20, 20)

# Example: compute spatial derivatives via FFT in φ
# Create simple test function
θ, φ = cfg.θ, cfg.φ
spatial = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    spatial[i,j] = sin(θ[i]) * cos(φ[j])
end
dφ = SHTnsKit.spatial_derivative_phi(cfg, spatial)

println("Spatial derivative field size: ", size(dφ))

destroy_config(cfg)
```

## Complex Fields

For complex-valued fields (e.g., wave functions):

```julia
cfg = create_gauss_config(16, 16)

# Create simple complex spectral coefficients
sh_complex = zeros(ComplexF64, cfg.nlm)
sh_complex[1] = 1.0 + 0.5im  # Complex Y_0^0 coefficient
if cfg.nlm > 2
    sh_complex[2] = 0.3 - 0.2im  # Complex Y_1^0 coefficient
end

# Complex field synthesis
spatial_complex = synthesis(cfg, reshape(sh_complex, cfg.lmax+1, cfg.mmax+1); real_output=false)

# Complex field analysis
recovered_complex = vec(analysis(cfg, spatial_complex))

# Check accuracy
complex_error = norm(sh_complex - recovered_complex)
println("Complex field error: $complex_error")

destroy_config(cfg)
```

## Performance and Threading

### Threading and FFTW threads

```julia
# Enable parallel loops and set FFTW threads sensibly
summary = set_optimal_threads!()
println(summary)  # (threads=..., fft_threads=...)

# Fine-tune
set_threading!(true)           # enable/disable parallel loops
set_fft_threads(4); get_fft_threads()
```

### Benchmarking

```julia
cfg = create_gauss_config(64, 64)
# Create bandlimited test coefficients (avoids high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end

# Time forward transform
@time spatial = synthesis(cfg, sh)

# Time backward transform  
@time recovered = analysis(cfg, spatial)

# Multiple runs for better statistics
println("Forward transform timing:")
@time for i in 1:10
    synthesis(cfg, sh)
end

destroy_config(cfg)
```

## GPU Acceleration

This package is CPU‑focused and does not include GPU support.

## Common Patterns

### In-Place Operations

For memory efficiency:

```julia
cfg = create_gauss_config(24, 24)

# Pre-allocate arrays
sh = allocate_spectral(cfg)
spatial = allocate_spatial(cfg)

# In-place operations (no additional allocation)
rand!(sh)
synthesize!(cfg, sh, spatial)  # spatial = synthesis(cfg, sh)
analyze!(cfg, spatial, sh)     # sh = analysis(cfg, spatial)

destroy_config(cfg)
```

### Batch Processing

```julia
cfg = create_gauss_config(20, 20)

# Process multiple fields
n_fields = 100
results = []

for i in 1:n_fields
    # Generate bandlimited test field (avoids roundtrip errors)
    sh = zeros(cfg.nlm)
    sh[1] = 1.0 + 0.1 * sin(i)  # Smooth variation
    
    # Process
    spatial = synthesis(cfg, sh)
    
    # Store result (example: compute mean)
    push!(results, mean(spatial))
    
    # Progress indicator
    i % 20 == 0 && println("Processed $i/$n_fields fields")
end

println("Mean of field means: ", mean(results))
destroy_config(cfg)
```

## Error Handling

```julia
cfg = create_gauss_config(16, 16)

try
    # Wrong array size
    wrong_sh = zeros(10)  # Should be cfg.nlm
    spatial = synthesis(cfg, wrong_sh)
catch e
    println("Caught expected error: ", e)
end

# Proper size check
# Create bandlimited test data (avoids high-frequency roundtrip errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0  # Simple bandlimited test
@assert length(sh) == cfg.nlm "Wrong spectral array size"

spatial = synthesis(cfg, sh)
println("Successful transform with proper size")

destroy_config(cfg)
```

## Next Steps

Now that you've mastered the basics:

1. **Read the [API Reference](api/index.md)** for complete function documentation
2. **Explore [Examples](examples/index.md)** for real-world applications  
3. **Check [Performance Guide](performance.md)** for optimization tips
4. **See [Advanced Usage](advanced.md)** for complex workflows

## Quick Reference

```julia
# Configuration
cfg = create_gauss_config(lmax, nlat; nlon=nlon)
cfg = create_regular_config(lmax, nlat; nlon=nlon)

# Basic transforms
spatial = synthesis(cfg, spectral)
spectral = analysis(cfg, spatial)

# Vector transforms (spheroidal/toroidal decomposition)
Vθ, Vφ = SHsphtor_to_spat(cfg, S_lm, T_lm)
S_lm, T_lm = spat_to_SHsphtor(cfg, Vθ, Vφ)

# Complex fields
spatial_c = SH_to_spat_cplx(cfg, spectral_c)
spectral_c = spat_cplx_to_SH(cfg, spatial_c)

# Threading
set_threading!(true)
set_optimal_threads!()

# Cleanup
destroy_config(cfg)
```
