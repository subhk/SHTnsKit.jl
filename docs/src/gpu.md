# GPU Acceleration

```@raw html
<div style="background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
    <h2 style="margin: 0 0 0.5rem 0; color: white; border: none;">CUDA-Accelerated Transforms</h2>
    <p style="margin: 0; opacity: 0.9;">Significant speedup on NVIDIA GPUs with automatic CPU fallback</p>
</div>
```

SHTnsKit.jl provides GPU-accelerated spherical harmonic transforms using CUDA and KernelAbstractions.jl for significant performance improvements on NVIDIA GPUs.

!!! tip "When to Use GPU"
    GPU acceleration is most beneficial for **lmax ≥ 64**. For smaller problems, CPU is often faster due to data transfer overhead.

---

## Quick Start

```julia
using SHTnsKit, CUDA

# Check GPU availability
println("CUDA available: ", CUDA.functional())
println("GPU device: ", CUDA.name(CUDA.device()))

# Create configuration
lmax = 128
cfg = create_gauss_config(lmax, lmax + 2)

# Create test data
spatial = rand(cfg.nlat, cfg.nlon)

# GPU-accelerated analysis (spatial → spectral)
Alm = gpu_analysis(cfg, spatial)

# GPU-accelerated synthesis (spectral → spatial)
recovered = gpu_synthesis(cfg, Alm)

# Verify accuracy
println("Max error: ", maximum(abs.(spatial - recovered)))
```

---

## Installation

### Requirements

- Julia 1.10+
- NVIDIA GPU with CUDA support
- CUDA toolkit (automatically installed via CUDA.jl)

### Setup

```julia
using Pkg
Pkg.add(["SHTnsKit", "CUDA", "GPUArrays", "KernelAbstractions"])

# Verify installation
using CUDA
println("CUDA version: ", CUDA.version())
println("GPU: ", CUDA.name(CUDA.device()))
println("Memory: ", CUDA.available_memory() / 1e9, " GB available")
```

---

## API Reference

### Core Transforms

#### `gpu_analysis`

```julia
gpu_analysis(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true)
```

GPU-accelerated spherical harmonic analysis transform.

**Algorithm:**
1. Transfer data to GPU
2. FFT along longitude (φ) using cuFFT
3. Legendre integration along latitude (θ) with parallel kernels
4. Transfer coefficients back to CPU

**Example:**
```julia
cfg = create_gauss_config(64, 66)
spatial = rand(cfg.nlat, cfg.nlon)

# Basic usage
Alm = gpu_analysis(cfg, spatial)

# Complex output
Alm_complex = gpu_analysis(cfg, spatial; real_output=false)

# Force CPU fallback
Alm_cpu = gpu_analysis(cfg, spatial; device=CPU_DEVICE)
```

#### `gpu_synthesis`

```julia
gpu_synthesis(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)
```

GPU-accelerated spherical harmonic synthesis transform.

**Algorithm:**
1. Transfer coefficients to GPU
2. Legendre summation with parallel kernels
3. Inverse FFT along longitude using cuFFT
4. Transfer spatial field back to CPU

**Example:**
```julia
cfg = create_gauss_config(64, 66)
Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Alm[3, 1] = 1.0  # Y_2^0 mode

# Synthesis
spatial = gpu_synthesis(cfg, Alm)

# Complex output (no Hermitian symmetry enforcement)
spatial_complex = gpu_synthesis(cfg, Alm; real_output=false)
```

### Memory-Safe Variants

#### `gpu_analysis_safe` / `gpu_synthesis_safe`

```julia
gpu_analysis_safe(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true)
gpu_synthesis_safe(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)
```

Memory-safe versions that automatically fall back to CPU if:
- Insufficient GPU memory
- GPU encounters an error
- CUDA is not available

**Example:**
```julia
# Safe for large problems - automatically falls back to CPU if needed
Alm = gpu_analysis_safe(cfg, large_spatial_data)
```

### Vector Field Transforms

#### `gpu_analysis_sphtor`

```julia
gpu_analysis_sphtor(cfg::SHTConfig, vθ, vφ; device=get_device())
```

GPU-accelerated spheroidal-toroidal decomposition.

**Example:**
```julia
cfg = create_gauss_config(32, 34)

# Create vector field components
vθ = rand(cfg.nlat, cfg.nlon)
vφ = rand(cfg.nlat, cfg.nlon)

# Decompose into spheroidal and toroidal
Slm, Tlm = gpu_analysis_sphtor(cfg, vθ, vφ)
```

#### `gpu_synthesis_sphtor`

```julia
gpu_synthesis_sphtor(cfg::SHTConfig, Slm, Tlm; device=get_device(), real_output=true)
```

GPU-accelerated vector field synthesis.

**Example:**
```julia
# Reconstruct vector field from coefficients
vθ_out, vφ_out = gpu_synthesis_sphtor(cfg, Slm, Tlm)
```

### Spectral Operators

#### `gpu_apply_laplacian!`

```julia
gpu_apply_laplacian!(cfg::SHTConfig, coeffs; device=get_device())
```

Apply the spherical Laplacian operator in spectral space: `Δf_lm = -l(l+1) f_lm`

**Example:**
```julia
Alm = rand(ComplexF64, cfg.lmax+1, cfg.mmax+1)
gpu_apply_laplacian!(cfg, Alm)  # Modifies Alm in-place
```

---

## Device Management

### Checking Device Status

```julia
using SHTnsKit, CUDA

# Get current device
device = get_device()
println("Current device: ", device)  # CUDA_DEVICE or CPU_DEVICE

# Available GPUs
gpus = get_available_gpus()
for gpu in gpus
    println("GPU $(gpu.id): $(gpu.name)")
end
```

### Selecting GPU

```julia
# Set active GPU by ID
set_gpu_device(0)  # First GPU
set_gpu_device(1)  # Second GPU (if available)

# Force CPU execution
Alm = gpu_analysis(cfg, spatial; device=CPU_DEVICE)
```

### Memory Management

```julia
# Check GPU memory
info = gpu_memory_info()
println("Free memory: ", info.free / 1e9, " GB")
println("Total memory: ", info.total / 1e9, " GB")

# Estimate memory for operation
bytes_needed = estimate_memory_usage(cfg, :analysis)
println("Memory needed: ", bytes_needed / 1e6, " MB")

# Check if operation will fit
if check_gpu_memory(bytes_needed)
    Alm = gpu_analysis(cfg, spatial)
else
    println("Using CPU fallback")
    Alm = analysis(cfg, spatial)
end

# Clear GPU cache
gpu_clear_cache!()
```

---

## Multi-GPU Support

For large problems, distribute work across multiple GPUs.

### Setup

```julia
using SHTnsKit, CUDA

# Create multi-GPU configuration
mgpu = create_multi_gpu_config(128, 130;
    strategy=:latitude,      # Split by latitude bands
    gpu_ids=[0, 1]           # Use GPUs 0 and 1
)

println("Using $(length(mgpu.gpu_devices)) GPUs")
```

### Transforms

```julia
# Multi-GPU analysis
spatial = rand(130, 257)
Alm = multi_gpu_analysis(mgpu, spatial)

# Multi-GPU synthesis
recovered = multi_gpu_synthesis(mgpu, Alm)
```

### Memory Streaming

For problems larger than GPU memory:

```julia
# Automatic chunking based on available memory
Alm = multi_gpu_analysis_streaming(mgpu, huge_spatial_data;
    max_memory_per_gpu = 4 * 1024^3  # 4 GB per GPU
)

# Estimate chunks needed
n_chunks = estimate_streaming_chunks(mgpu, size(huge_spatial_data))
println("Will use $n_chunks chunks per GPU")
```

---

## Unified Loop Abstraction

The `@sht_loop` macro provides unified CPU/GPU execution:

```julia
using SHTnsKit

# Works on both CPU and GPU arrays
A = rand(100, 100)        # CPU array
B = similar(A)

@sht_loop B[I] = A[I] * 2.0 over I ∈ CartesianIndices(A)

# Same code works on GPU
using CUDA
A_gpu = CuArray(A)
B_gpu = similar(A_gpu)

@sht_loop B_gpu[I] = A_gpu[I] * 2.0 over I ∈ CartesianIndices(A_gpu)
```

### Backend Control

```julia
# Check current backend mode
println(loop_backend())  # "auto" (default)

# Force CPU SIMD (useful for debugging)
set_loop_backend("SIMD")

# Restore auto-detection
set_loop_backend("auto")
```

---

## Performance Tips

### 1. Batch Operations

Minimize data transfers by batching operations:

```julia
# Inefficient: many small transfers
for field in fields
    Alm = gpu_analysis(cfg, field)
    # ... process ...
end

# Efficient: keep data on GPU
gpu_data = CuArray(stack(fields))
# Process all at once on GPU
```

### 2. Use Appropriate Resolution

GPU overhead is significant for small problems. As a general guideline:
- **lmax < 32**: CPU is typically faster due to data transfer overhead
- **lmax 32-128**: GPU becomes beneficial
- **lmax > 128**: GPU strongly recommended

### 3. Preallocate GPU Buffers

For repeated transforms:

```julia
# Create cuFFT plans once
plan = create_cufft_plan(cfg.nlat, cfg.nlon)

# Reuse buffer
buffer = CUDA.zeros(ComplexF64, cfg.nlat, cfg.nlon)
copyto!(buffer, data)

# Use preplanned FFT
gpu_fft!(plan, buffer)
```

### 4. Monitor Memory

```julia
# Profile memory usage
CUDA.@time begin
    Alm = gpu_analysis(cfg, spatial)
end

# Check for memory leaks
for i in 1:100
    gpu_analysis(cfg, spatial)
    if i % 10 == 0
        info = gpu_memory_info()
        println("Iteration $i: $(info.free / 1e9) GB free")
    end
end
```

---

## Troubleshooting

### Common Issues

#### "CUDA not available"

```julia
# Check CUDA installation
using CUDA
println(CUDA.functional())  # Should be true
println(CUDA.version())     # Should show version

# If false, reinstall CUDA.jl
using Pkg
Pkg.rm("CUDA")
Pkg.add("CUDA")
Pkg.build("CUDA")
```

#### "Out of GPU memory"

```julia
# Use safe variants
Alm = gpu_analysis_safe(cfg, spatial)

# Or reduce problem size
cfg_small = create_gauss_config(64, 66)  # Instead of 256

# Or use streaming
Alm = multi_gpu_analysis_streaming(mgpu, spatial; max_memory_per_gpu=2*1024^3)
```

#### "Numerical differences from CPU"

GPU floating-point operations may have small differences due to:
- Different reduction order (non-associativity)
- FMA (fused multiply-add) usage

Typical differences are ~1e-14 for Float64, which is acceptable for most applications.

---

## Example: Complete Workflow

```julia
using SHTnsKit, CUDA

# Setup
println("=== GPU Spherical Harmonic Transform ===")
println("GPU: ", CUDA.name(CUDA.device()))
println("Memory: ", CUDA.available_memory() / 1e9, " GB")

# Configuration
lmax = 128
cfg = create_gauss_config(lmax, lmax + 2)
println("Grid: $(cfg.nlat) × $(cfg.nlon)")

# Create test field: Y_4^2 + Y_6^0
Alm_true = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Alm_true[5, 3] = 1.0 + 0.5im  # Y_4^2
Alm_true[7, 1] = 0.8          # Y_6^0

# Synthesis on GPU
spatial = gpu_synthesis(cfg, Alm_true)
println("Spatial field: min=$(minimum(real(spatial))), max=$(maximum(real(spatial)))")

# Analysis on GPU
Alm_recovered = gpu_analysis(cfg, real.(spatial))

# Verify
error = maximum(abs.(Alm_true - Alm_recovered))
println("Roundtrip error: $error")

# Benchmark
using BenchmarkTools
println("\nBenchmarks:")
@btime gpu_analysis($cfg, $spatial)
@btime gpu_synthesis($cfg, $Alm_true)

println("Done!")
```

---

## See Also

- [Performance Guide](performance.md) - General optimization tips
- [Distributed Computing](distributed.md) - MPI parallelization
- [API Reference](api/index.md) - Complete function documentation
