# Examples Gallery

```@raw html
<div style="background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
    <h2 style="margin: 0 0 0.5rem 0; color: white; border: none;">Examples Gallery</h2>
    <p style="margin: 0; opacity: 0.9;">Real-world examples and tutorials from beginner to advanced</p>
</div>
```

Real-world examples and tutorials demonstrating SHTnsKit.jl capabilities, organized by difficulty level.

```@raw html
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">

<div style="background: #eff6ff; border-radius: 8px; padding: 1rem; border-left: 4px solid #2563eb;">
    <strong style="color: #1e40af;">Beginner</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
        Start here if you're new to spherical harmonics
    </p>
</div>

<div style="background: #fef3c7; border-radius: 8px; padding: 1rem; border-left: 4px solid #f59e0b;">
    <strong style="color: #92400e;">Intermediate</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
        For users comfortable with basic transforms
    </p>
</div>

<div style="background: #f3e8ff; border-radius: 8px; padding: 1rem; border-left: 4px solid #7c3aed;">
    <strong style="color: #5b21b6;">Advanced</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #475569;">
        Complex workflows and specialized applications
    </p>
</div>

</div>
```

!!! tip "Learning Path"
    Work through the examples in order for the best learning experience.

## Beginner Examples

Start here if you're new to spherical harmonics. These examples teach fundamental concepts with simple, well-explained code.

### Example 1: Your First Transform

**Goal:** Learn the basic workflow of spherical harmonic transforms

```julia
using SHTnsKit

# Step 1: Create a configuration (like setting up your workspace)
lmax = 16
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)
println("Created configuration for degree up to $lmax")

# Step 2: Create a simple temperature pattern
# Simple pattern: warm equator, cold poles (Y_2^0 harmonic)
temperature = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat
    x = cfg.x[i]  # cos(θ) at this latitude
    temperature[i, :] .= 273.15 + 30 * (1 - x^2)  # Warmer at equator
end

println("Temperature range: $(extrema(temperature)) K")

# Step 3: Transform to spherical harmonic coefficients (analysis)
Alm = analysis(cfg, temperature)
println("Coefficient matrix size: $(size(Alm))")

# Step 4: Find the most important coefficient (skip l=0 global mean)
max_val, max_idx = findmax(abs.(Alm[2:end, :]))
println("Largest non-constant mode magnitude: $max_val")

# Step 5: Reconstruct the original field (synthesis)
T_reconstructed = synthesis(cfg, Alm)
error = maximum(abs.(temperature - T_reconstructed))
println("Reconstruction error: $error (should be tiny!)")

destroy_config(cfg)
```

**Key concepts learned:**
- Configuration setup (`create_gauss_config`)
- Creating realistic data patterns
- Analysis: spatial → spectral (`analysis`)
- Synthesis: spectral → spatial (`synthesis`)
- Understanding (l,m) mode indices

### Example 2: Pure Spherical Harmonic Patterns

**Goal:** Understand how individual spherical harmonic modes look

```julia
using SHTnsKit

lmax = 32
cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)

# Create pure Y_2^0 spherical harmonic (zonal mode)
# Coefficients are stored as (lmax+1) × (mmax+1) matrix
Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Alm[3, 1] = 1.0  # l=2, m=0 (index is l+1 for row, m+1 for column)
println("Creating Y₂⁰ pattern (zonal, m=0)")

# Synthesize to spatial domain
Y20_pattern = synthesis(cfg, Alm)

# This creates a pattern that varies only with latitude
println("Pattern statistics:")
println("  Min value: $(minimum(Y20_pattern))")
println("  Max value: $(maximum(Y20_pattern))")
println("  At north pole: $(Y20_pattern[1,1])")
println("  At equator: $(Y20_pattern[div(cfg.nlat,2),1])")

# The Y_2^0 pattern = (3cos²θ - 1)/2
# Positive at poles, negative at equator

destroy_config(cfg)
```

**Key concepts learned:**
- How to create pure spherical harmonic patterns
- Understanding zonal (m=0) vs sectoral (m≠0) modes
- The relationship between (l,m) indices and spatial patterns
- Coefficient indexing: `Alm[l+1, m+1]`

**Try this:** Change to `Alm[3, 3] = 1.0` (l=2, m=2) to see a sectoral pattern!

### Example 3: Understanding Power Spectra

**Goal:** Learn how energy is distributed across different spatial scales

```julia
using SHTnsKit

lmax = 32
cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)

# Create a field with multiple scales (like weather patterns)
field = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ = cfg.θ[i]
    φ = cfg.φ[j]
    field[i,j] = 2*sin(2*θ)*cos(φ) +      # Large scale
                 0.5*sin(6*θ)*cos(3*φ) +   # Medium scale
                 0.1*sin(12*θ)*cos(6*φ)    # Small scale
end

println("Created multi-scale field with 3 different spatial scales")

# Transform to spectral domain
Alm = analysis(cfg, field)

# Compute power spectrum using energy_scalar_l_spectrum
power = energy_scalar_l_spectrum(cfg, Alm)

# Find which scales dominate
max_power_degree = argmax(power[2:end])  # Skip l=0 (global mean)
println("Peak energy at degree l = $max_power_degree")
println("This corresponds to ~$(360/max_power_degree)° wavelength")

# Print first few power values
println("Power spectrum (first 10 degrees):")
for l in 0:min(9, length(power)-1)
    println("  l=$l: $(power[l+1])")
end

destroy_config(cfg)
```

**Key concepts learned:**
- How to create multi-scale patterns
- Power spectrum analysis shows energy distribution
- Relationship between degree l and spatial wavelength
- Use `energy_scalar_l_spectrum` for power spectrum analysis

**Physical meaning:** In meteorology, this tells you whether your weather system is dominated by large-scale patterns (like jet streams) or small-scale features (like thunderstorms).

## Intermediate Examples

Ready to tackle more complex problems? These examples introduce vector fields, real-world data patterns, and scientific applications.

### Vector Field Decomposition

### Vorticity-Divergence Decomposition

```julia
using SHTnsKit
using LinearAlgebra

lmax = 64
cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)

# Create a realistic atmospheric flow pattern
u = zeros(cfg.nlat, cfg.nlon)  # Zonal wind (east-west)
v = zeros(cfg.nlat, cfg.nlon)  # Meridional wind (north-south)

for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ = cfg.θ[i]
    φ = cfg.φ[j]
    u[i,j] = 20 * sin(2θ) * (1 + 0.4 * cos(4φ))  # Jet stream
    v[i,j] = 5 * cos(3θ) * sin(2φ)                # Meridional flow
end

# Decompose into spheroidal (divergent) and toroidal (rotational)
Slm, Tlm = spat_to_SHsphtor(cfg, u, v)

# Analyze energy distribution
spheroidal_energy = sum(abs2, Slm)
toroidal_energy = sum(abs2, Tlm)
println("Spheroidal (divergent) energy: $spheroidal_energy")
println("Toroidal (rotational) energy: $toroidal_energy")

# Reconstruct original velocity
u_recon, v_recon = SHsphtor_to_spat(cfg, Slm, Tlm)
velocity_error = norm(u - u_recon) + norm(v - v_recon)
println("Velocity reconstruction error: $velocity_error")

destroy_config(cfg)
```

### Stream Function from Vorticity

```julia
using SHTnsKit
using LinearAlgebra

lmax = 48
cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)

# Create vorticity field (e.g., from observations)
vorticity = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ = cfg.θ[i]
    φ = cfg.φ[j]
    vorticity[i,j] = exp(-((θ - π/2)^2 + (φ - π)^2) / 0.5^2) * sin(4φ)
end

# Transform vorticity to spectral domain
ζ_lm = analysis(cfg, vorticity)

# Solve ∇²ψ = ζ for stream function ψ
# In spectral domain: -l(l+1) ψ_lm = ζ_lm
ψ_lm = similar(ζ_lm)
for l in 0:cfg.lmax
    for m in 0:min(l, cfg.mmax)
        if l > 0
            ψ_lm[l+1, m+1] = -ζ_lm[l+1, m+1] / (l * (l + 1))
        else
            ψ_lm[l+1, m+1] = 0.0  # l=0 mode: constant not uniquely determined
        end
    end
end

# Get velocity from stream function (toroidal component only)
Slm_zero = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
u_stream, v_stream = SHsphtor_to_spat(cfg, Slm_zero, ψ_lm)

# Convert stream function to spatial domain
stream_function = synthesis(cfg, ψ_lm)

println("Stream function range: ", extrema(stream_function))
println("Max velocity from stream: ", maximum(sqrt.(u_stream.^2 .+ v_stream.^2)))

destroy_config(cfg)
```

## Parallel Computing Examples

### MPI Distributed Computing

**Goal:** Learn how to use MPI for large-scale parallel spherical harmonic computations

```julia
# Save as parallel_example.jl and run with: mpiexec -n 4 julia parallel_example.jl
using MPI
MPI.Init()

using SHTnsKit, PencilArrays, PencilFFTs

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if rank == 0
    println("Running SHTnsKit parallel example with $nprocs processes")
end

# Create configuration (same on all processes)
lmax = 64
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

if rank == 0
    println("Problem size: $(cfg.nlm) spectral coefficients")
    println("Grid: $(cfg.nlat) × $(cfg.nlon) spatial points")
end

# Create distributed array using PencilArrays
pen = Pencil((nlat, nlon), comm)
fθφ = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))

# Fill with test data (Y_2^0 pattern)
ranges = PencilArrays.range_local(pen)
for (i_local, i_global) in enumerate(ranges[1])
    x = cfg.x[i_global]
    for j in 1:length(ranges[2])
        fθφ[i_local, j] = (3*x^2 - 1)/2
    end
end

# Benchmark parallel transforms
MPI.Barrier(comm)
start_time = MPI.Wtime()

n_iter = 50
for i in 1:n_iter
    Alm = SHTnsKit.dist_analysis(cfg, fθφ)
    fθφ_out = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)
end

MPI.Barrier(comm)
end_time = MPI.Wtime()

if rank == 0
    avg_time = (end_time - start_time) / n_iter
    println("Parallel roundtrip: $(avg_time*1000) ms per iteration")
end

# Verify accuracy
Alm = SHTnsKit.dist_analysis(cfg, fθφ)
fθφ_recovered = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)

max_err = maximum(abs.(parent(fθφ_recovered) .- parent(fθφ)))
global_max_err = MPI.Allreduce(max_err, MPI.MAX, comm)

if rank == 0
    println("Roundtrip error: $global_max_err")
    println(global_max_err < 1e-10 ? "SUCCESS!" : "FAILED")
end

destroy_config(cfg)
MPI.Finalize()
```

**Key concepts:**
- MPI initialization and communicator setup
- PencilArrays for domain decomposition
- Distributed transforms with `dist_analysis` and `dist_synthesis`
- Error verification across MPI ranks

### Run Example Scripts

```bash
# Per-rank SHT scalar roundtrip (safe PencilArrays allocation)
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl

# Distributed FFT roundtrip along φ using PencilFFTs
mpiexec -n 2 julia --project=. examples/parallel_fft_roundtrip.jl
```

### Single-Node Performance Example

**Goal:** Benchmark and optimize single-node performance

```julia
using SHTnsKit, BenchmarkTools

lmax = 64
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

println("Single-Node Performance Benchmark")
println("="^40)
println("Grid size: $(cfg.nlat) × $(cfg.nlon)")
println("Spectral coefficients: $(cfg.nlm)")

# Create test data
spatial = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ = cfg.θ[i]
    φ = cfg.φ[j]
    spatial[i,j] = sin(2θ) * cos(φ) + 0.5 * sin(4θ) * cos(3φ)
end

# Benchmark analysis (spatial → spectral)
analysis_time = @belapsed analysis($cfg, $spatial)
println("Analysis time: $(analysis_time*1000) ms")

Alm = analysis(cfg, spatial)

# Benchmark synthesis (spectral → spatial)
synthesis_time = @belapsed synthesis($cfg, $Alm)
println("Synthesis time: $(synthesis_time*1000) ms")

# Benchmark roundtrip
roundtrip_time = @belapsed begin
    alm = analysis($cfg, $spatial)
    synthesis($cfg, alm)
end
println("Roundtrip time: $(roundtrip_time*1000) ms")

# Verify accuracy
recovered = synthesis(cfg, Alm)
max_error = maximum(abs.(spatial - recovered))
println("Roundtrip error: $max_error")

# Threading info
println("\nThreading configuration:")
println("  Julia threads: $(Threads.nthreads())")

destroy_config(cfg)
```

**Key concepts:**
- Performance benchmarking with BenchmarkTools
- Forward and inverse transform timing
- Accuracy verification

### Parallel Vector Transform Example

**Goal:** Perform distributed vector field transforms

```julia
# Save as parallel_vector.jl, run with: mpiexec -n 4 julia parallel_vector.jl
using MPI
MPI.Init()

using SHTnsKit, PencilArrays, PencilFFTs

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# Create configuration
lmax = 64
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

if rank == 0
    println("Parallel Vector Transform Example")
    println("Problem: $(cfg.nlm) coefficients, $(cfg.nlat)×$(cfg.nlon) grid")
    println("MPI processes: $nprocs")
end

# Create distributed arrays for velocity field
pen = Pencil((nlat, nlon), comm)
Vθ = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))
Vφ = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))

# Fill with test vector field (solid body rotation)
ranges = PencilArrays.range_local(pen)
for (i_local, i_global) in enumerate(ranges[1])
    θ = cfg.θ[i_global]
    for (j_local, j_global) in enumerate(ranges[2])
        φ = cfg.φ[j_global]
        Vθ[i_local, j_local] = cos(θ) * sin(φ)
        Vφ[i_local, j_local] = cos(φ)
    end
end

# Benchmark distributed vector transforms
MPI.Barrier(comm)
start_time = MPI.Wtime()

n_iter = 20
for i in 1:n_iter
    Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vθ, Vφ)
    Vθ_out, Vφ_out = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ=Vθ)
end

MPI.Barrier(comm)
end_time = MPI.Wtime()

if rank == 0
    avg_time = (end_time - start_time) / n_iter
    println("Vector roundtrip: $(avg_time*1000) ms per iteration")
end

# Verify accuracy
Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vθ, Vφ)
Vθ_rec, Vφ_rec = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ=Vθ)

θ_err = maximum(abs.(parent(Vθ_rec) .- parent(Vθ)))
φ_err = maximum(abs.(parent(Vφ_rec) .- parent(Vφ)))
global_θ_err = MPI.Allreduce(θ_err, MPI.MAX, comm)
global_φ_err = MPI.Allreduce(φ_err, MPI.MAX, comm)

if rank == 0
    println("Vθ roundtrip error: $global_θ_err")
    println("Vφ roundtrip error: $global_φ_err")
    println((global_θ_err < 1e-10 && global_φ_err < 1e-10) ? "SUCCESS!" : "FAILED")
end

destroy_config(cfg)
MPI.Finalize()
```

### Scaling Test Example

**Goal:** Test parallel scaling with different process counts

```julia
# Save as scaling_test.jl, run with: mpiexec -n 4 julia scaling_test.jl
using MPI
MPI.Init()

using SHTnsKit, PencilArrays, PencilFFTs

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if rank == 0
    println("Parallel Scaling Test")
    println("MPI processes: $nprocs")
end

# Test different problem sizes
for lmax in [32, 64, 128]
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)

    # Create distributed array
    pen = Pencil((nlat, nlon), comm)
    fθφ = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))

    # Fill with test data
    ranges = PencilArrays.range_local(pen)
    for (i_local, i_global) in enumerate(ranges[1])
        x = cfg.x[i_global]
        for j in 1:length(ranges[2])
            fθφ[i_local, j] = (3*x^2 - 1)/2
        end
    end

    # Warmup
    for _ in 1:5
        Alm = SHTnsKit.dist_analysis(cfg, fθφ)
        SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)
    end

    # Benchmark
    MPI.Barrier(comm)
    start_time = MPI.Wtime()

    n_iter = 50
    for _ in 1:n_iter
        Alm = SHTnsKit.dist_analysis(cfg, fθφ)
        SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)
    end

    MPI.Barrier(comm)
    end_time = MPI.Wtime()

    if rank == 0
        avg_time = (end_time - start_time) / n_iter * 1000
        println("lmax=$lmax: $(round(avg_time, digits=2)) ms/roundtrip")
    end

    destroy_config(cfg)
end

MPI.Finalize()
```

**Key concepts:**
- Testing performance across problem sizes
- Proper warmup before timing
- Parallel synchronization for accurate timing

## Advanced Applications

### Multiscale Analysis

```julia
using SHTnsKit

# Create different resolution configurations
lmax_values = [16, 32, 64, 128]
cfgs = []
for lmax in lmax_values
    nlat = lmax + 2
    nlon = 2*lmax + 1
    push!(cfgs, create_gauss_config(lmax, nlat; nlon=nlon))
end

# Use highest resolution for reference field
cfg_hi = cfgs[end]

# Create test field with multiple scales at highest resolution
field = zeros(cfg_hi.nlat, cfg_hi.nlon)
for i in 1:cfg_hi.nlat, j in 1:cfg_hi.nlon
    θ = cfg_hi.θ[i]
    φ = cfg_hi.φ[j]
    field[i,j] = sin(2θ) * cos(φ) +           # Large scale
                 0.3 * sin(8θ) * cos(4φ) +     # Medium scale
                 0.1 * sin(16θ) * cos(8φ)      # Small scale
end

# Analyze at different resolutions
powers = []
for cfg in cfgs
    # Create field at this resolution
    field_i = zeros(cfg.nlat, cfg.nlon)
    for i in 1:cfg.nlat, j in 1:cfg.nlon
        θ = cfg.θ[i]
        φ = cfg.φ[j]
        field_i[i,j] = sin(2θ) * cos(φ) +
                       0.3 * sin(8θ) * cos(4φ) +
                       0.1 * sin(16θ) * cos(8φ)
    end

    # Analyze and compute power spectrum
    f_lm = analysis(cfg, field_i)
    power_i = energy_scalar_l_spectrum(cfg, f_lm)
    push!(powers, power_i)

    println("Resolution lmax=$(cfg.lmax): $(length(power_i)) modes")
end

# Compare power spectra at common degrees
println("\nPower at l=2 (large scale):")
for (i, cfg) in enumerate(cfgs)
    println("  lmax=$(cfg.lmax): $(powers[i][3])")
end

# Cleanup
for cfg in cfgs
    destroy_config(cfg)
end
```

### Field Rotation and Coordinate Transformations

```julia
using SHTnsKit

lmax = 32
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create field in one coordinate system
original_field = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ = cfg.θ[i]
    φ = cfg.φ[j]
    original_field[i,j] = sin(3θ) * cos(2φ)
end

# Transform to spectral domain
f_lm = analysis(cfg, original_field)

# Rotate using Euler angles (ZYZ convention)
α, β, γ = π/4, π/6, π/8
f_rot = copy(f_lm)
rotate_real!(cfg, f_rot; alpha=α, beta=β, gamma=γ)

# Transform back to spatial domain
rotated_field = synthesis(cfg, f_rot)

println("Original field range: ", extrema(original_field))
println("Rotated field range: ", extrema(rotated_field))

# Verify rotation preserves power
orig_power = sum(abs2, f_lm)
rot_power = sum(abs2, f_rot)
println("Power preserved: ", isapprox(orig_power, rot_power, rtol=1e-10))

destroy_config(cfg)
```

## High-Performance Examples

### Multi-threaded Batch Processing

```julia
using SHTnsKit
using Base.Threads
using Statistics

lmax = 64
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Large batch of fields to process
n_batch = 100
# Create bandlimited test fields (smooth functions prevent errors)
input_fields = []
for i in 1:n_batch
    field = zeros(cfg.nlat, cfg.nlon)
    for j in 1:cfg.nlat, k in 1:cfg.nlon
        θ = cfg.θ[j]
        φ = cfg.φ[k]
        field[j,k] = 1.0 + 0.3 * sin(2θ) * cos(φ) * (1 + 0.1*sin(i))
    end
    push!(input_fields, field)
end

# Process with threading
println("Processing $n_batch fields with $(nthreads()) Julia threads...")
results = Vector{Float64}(undef, n_batch)

@time @threads for i in 1:n_batch
    # Each thread gets its own work
    field = input_fields[i]

    # Transform and compute power spectrum
    sh = analysis(cfg, field)
    power = energy_scalar_l_spectrum(cfg, sh)

    # Store result (total energy)
    results[i] = sum(power)
end

println("Mean energy per field: ", mean(results))
println("Energy std dev: ", std(results))

destroy_config(cfg)
```

## Validation and Testing Examples

### Analytical Test Cases

```julia
using SHTnsKit

lmax = 24
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Test pure spherical harmonics Y_l^m
# For simplicity, test real zonal harmonics (m=0)
test_cases = [
    (l=0, m=0, desc="Y₀⁰ (constant)"),
    (l=1, m=0, desc="Y₁⁰ (dipole)"),
    (l=2, m=0, desc="Y₂⁰ (quadrupole)"),
]

println("Analytical validation tests:")
for case in test_cases
    # Create spectral coefficients with single mode
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Alm[case.l+1, case.m+1] = 1.0

    # Synthesize to spatial domain
    spatial = synthesis(cfg, Alm)

    # Analyze back to spectral
    recovered = analysis(cfg, spatial)

    # Check coefficient at expected location
    coeff_val = recovered[case.l+1, case.m+1]

    # Find largest coefficient magnitude
    max_val, max_idx = findmax(abs.(recovered))

    println("Test: $(case.desc)")
    println("  Expected: l=$(case.l), m=$(case.m)")
    println("  Recovered coefficient: $coeff_val")
    println("  Max magnitude at: $(Tuple(max_idx)) = $max_val")

    # Check if roundtrip preserves the mode
    is_correct = max_idx == CartesianIndex(case.l+1, case.m+1)
    println("  Result: ", is_correct ? "PASS" : "FAIL")
    println()
end

destroy_config(cfg)
```

### Numerical Accuracy Tests

```julia
using SHTnsKit
using LinearAlgebra

# Test different resolutions
resolutions = [16, 32, 64]

println("Accuracy vs Resolution Test:")
println("=" ^ 40)

for lmax in resolutions
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)

    # Create bandlimited test coefficients (prevents roundtrip errors)
    Alm_original = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Alm_original[1, 1] = 1.0  # l=0, m=0
    Alm_original[3, 1] = 0.5  # l=2, m=0
    if cfg.lmax >= 4
        Alm_original[5, 1] = 0.2  # l=4, m=0
    end

    # Round-trip transform
    spatial = synthesis(cfg, Alm_original)
    Alm_recovered = analysis(cfg, spatial)

    # Measure error
    error = norm(Alm_original - Alm_recovered) / norm(Alm_original)

    println("lmax=$lmax: relative error = $(round(error, sigdigits=3))")

    destroy_config(cfg)
end

# Test with complex pattern (non-zonal modes)
println("\nNon-zonal mode test:")
lmax = 32
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

Alm_original = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Alm_original[3, 3] = 1.0 + 0.5im  # l=2, m=2

spatial = synthesis(cfg, Alm_original)
Alm_recovered = analysis(cfg, spatial)

error = norm(Alm_original - Alm_recovered) / norm(Alm_original)
println("l=2, m=2 roundtrip error: $(round(error, sigdigits=3))")

destroy_config(cfg)
```

These examples demonstrate the full range of SHTnsKit.jl capabilities from basic transforms to advanced scientific applications. Each example can serve as a starting point for your specific research needs.
