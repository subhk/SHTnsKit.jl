# SHTnsKit.jl

[![Build Status](https://github.com/subhk/SHTnsKit.jl/workflows/CI/badge.svg)](https://github.com/subhk/SHTnsKit.jl/actions)

<!-- Badges -->
 <p align="left">
    <a href="https://subhk.github.io/SHTnsKit.jl/stable">
        <img alt="stable docs" src="https://img.shields.io/badge/documentation-stable%20-blue">
    </a>
      <a href="https://subhk.github.io/SHTnsKit.jl/dev">
        <img alt="latest docs" src="https://img.shields.io/badge/documentation-dev%20-orange">
    </a>
</p>

<a href="https://github.com/subhk/SHTnsKit.jl/actions/workflows/mpi-examples.yml">
  <img alt="MPI Examples" src="https://github.com/subhk/SHTnsKit.jl/actions/workflows/mpi-examples.yml/badge.svg">
</a>

**High-Performance Spherical Harmonic Transforms in Julia**

SHTnsKit.jl provides a comprehensive, pure-Julia implementation of spherical harmonic transforms with **parallel computing support** for scalable scientific computing. 
## Key Features

### **High-Performance Computing**
- **Pure Julia**: No C dependencies, seamless Julia ecosystem integration
- **Multi-threading**: Optimized with Julia threads and FFTW parallelization
- **MPI Parallel**: Distributed computing with MPI + PencilArrays + PencilFFTs
- **SIMD Optimized**: Vectorization with LoopVectorization.jl support
- **Extensible**: Modular architecture for CPU/GPU/distributed computing

### **Complete Scientific Functionality**  
- **Transform Types**: Scalar, vector, and complex field transforms
- **Grid Support**: Gauss-Legendre and regular (equiangular) grids
- **Vector Analysis**: Spheroidal-toroidal decomposition for flow fields
- **Differential Operators**: Laplacian, gradient, divergence, vorticity
- **Spectral Analysis**: Power spectra, correlation functions, filtering

### **Advanced Capabilities**
- **Automatic Differentiation**: Native ForwardDiff.jl and Zygote.jl support  
- **Field Rotations**: Wigner D-matrix rotations and coordinate transforms
- **Matrix Operators**: Efficient spectral differential operators
- **Performance Tuning**: Comprehensive benchmarking and optimization tools


## Installation

### Basic Installation (Serial Computing)

```julia
using Pkg
Pkg.add("SHTnsKit")
```

### Full Installation (Parallel Computing)

For high-performance parallel computing on clusters:

```julia
using Pkg
Pkg.add(["SHTnsKit", "MPI", "PencilArrays", "PencilFFTs", "LoopVectorization"])
```

### System Requirements

**MPI Setup** (for parallel computing):
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev

# macOS  
brew install open-mpi

# Configure Julia MPI
julia -e 'using Pkg; Pkg.build("MPI")'
```

## Quick Start

### Hello World (Serial)

```julia
using SHTnsKit

# Step 1: Create configuration
lmax = 16                      # Maximum spherical harmonic degree
nlat = lmax + 2                # Number of latitude points
nlon = 2*lmax + 1              # Number of longitude points
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Step 2: Create a test pattern (Y_2^0 harmonic)
spatial = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat
    x = cfg.x[i]  # cos(θ) at this latitude
    spatial[i, :] .= (3*x^2 - 1)/2
end

# Step 3: Transform to spectral coefficients
Alm = analysis(cfg, spatial)

# Step 4: Transform back to spatial domain
recovered = synthesis(cfg, Alm)

# Step 5: Verify accuracy
max_error = maximum(abs.(spatial - recovered))
println("Roundtrip error: $max_error")  # Should be ~1e-14

destroy_config(cfg)
```

### Parallel Computing (MPI)

```julia
using SHTnsKit, MPI, PencilArrays, PencilFFTs

MPI.Init()

# Create configuration
lmax = 32
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create distributed array using PencilArrays
pen = Pencil((nlat, nlon), MPI.COMM_WORLD)
fθφ = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))

# Fill local data (Y_2^0 pattern)
ranges = PencilArrays.range_local(pen)
for (i_local, i_global) in enumerate(ranges[1])
    x = cfg.x[i_global]  # cos(θ)
    for j in 1:length(ranges[2])
        fθφ[i_local, j] = (3*x^2 - 1)/2
    end
end

# Distributed transforms
Alm = SHTnsKit.dist_analysis(cfg, fθφ)
fθφ_recovered = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)

# Vector transforms (spheroidal/toroidal)
Vt, Vp = copy(fθφ), copy(fθφ)
Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vt, Vp)
Vt2, Vp2 = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ=Vt, real_output=true)

destroy_config(cfg)
MPI.Finalize()
```

### High-Performance SIMD

```julia
using SHTnsKit
using LoopVectorization
using Printf

# Use larger problem size where SIMD benefits outweigh overhead
cfg = create_gauss_config(64, 66; nlon=129)

# Test turbo-optimized Laplacian operation
sh_coeffs = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
for l in 0:min(cfg.lmax, 20), m in 0:min(l, cfg.mmax)
    sh_coeffs[l+1, m+1] = (0.1 + 0.05im) * exp(-0.05*l)
end

println("Testing SIMD-optimized operations...")
sh_copy = copy(sh_coeffs)
turbo_apply_laplacian!(cfg, sh_copy)
println(" Turbo Laplacian completed")

# Benchmark analysis/synthesis transforms
results = benchmark_turbo_vs_simd(cfg)
@printf("Analysis speedup: %.2fx\n", results.analysis_speedup)
@printf("Synthesis speedup: %.2fx\n", results.synthesis_speedup)

destroy_config(cfg)

```

## Performance Optimization

### Threading Configuration

```julia
using SHTnsKit, FFTW

# Manual FFTW thread control
FFTW.set_num_threads(8)    # Set FFTW thread count

# SHTnsKit threading (for internal calculations)
shtns_use_threads(8)       # Set number of threads for SHTnsKit operations

# Check current settings
println("Julia threads: $(Threads.nthreads())")
println("FFTW threads: $(FFTW.get_num_threads())")
```

### Environment Variables

Control behavior at startup:

```bash
# Julia threading
export JULIA_NUM_THREADS=8

# SHTnsKit configuration
export SHTNSKIT_FORCE_FFTW=1           # Force FFTW usage over DFT fallback
export SHTNSKIT_PHI_SCALE=quad         # Use φ quadrature scaling (default: dft)
export SHTNSKIT_CACHE_SIZE=64          # L1 cache size in KB for optimization
export SHTNSKIT_CACHE_PENCILFFTS=1     # Cache PencilFFTs plans for MPI

julia --project=.
```

### Benchmarking Tools

```julia
using SHTnsKit, LoopVectorization, BenchmarkTools

# SIMD performance comparison (requires LoopVectorization.jl)
cfg = create_gauss_config(64, 66; nlon=129)
results = benchmark_turbo_vs_simd(cfg; trials=3)
println("Analysis speedup: $(results.analysis_speedup)x")
println("Synthesis speedup: $(results.synthesis_speedup)x")

# Manual performance timing
lmax = 32
cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)
f = randn(cfg.nlat, cfg.nlon)

# Time basic operations
@btime analysis($cfg, $f)
@btime synthesis($cfg, analysis($cfg, $f))

# Memory allocation benchmarking
allocs = @allocated analysis(cfg, f)
println("Analysis allocations: $allocs bytes")
```

### Performance Tips

- use_rfft (distributed plans): Enable real-to-complex transforms in `DistAnalysisPlan` and `DistSphtorPlan` to cut (θ,k) memory and speed real-output paths. Falls back to complex FFTs if not available.
- with_spatial_scratch (distributed vector/QST): Set to `true` to keep a single complex (θ,φ) buffer inside the plan and avoid per-call allocations for iFFT when outputs are real.
- Plan reuse: Build plans once per problem size and reuse across calls to avoid planner churn and allocations.
- Tables vs on-the-fly Plm: Precompute with `enable_plm_tables!(cfg)` to reduce CPU if your grid is fixed; results are identical to on-the-fly recurrence.

## Parallel Computing Guide

### Running Examples

```bash
# Parallel scalar roundtrip (2 processes)
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl

# Include vector field roundtrip
# (Use in-place plans; add spatial scratch to avoid allocs on real outputs)
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl --vector

# Include 3D (Q,S,T) roundtrip
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl --qst

# Ensure required optional packages are available (first time)
julia --project=. -e 'using Pkg; Pkg.add(["MPI","PencilArrays","PencilFFTs"])'
 
# Distributed FFT roundtrip (2 processes)
mpiexec -n 2 julia --project=. examples/parallel_fft_roundtrip.jl
```

Spectral operator demo (cosθ application in spectral space):

```bash
mpiexec -n 2 julia --project=. examples/operator_parallel.jl           # dense
mpiexec -n 2 julia --project=. examples/operator_parallel.jl --halo    # per-m Allgatherv halo

Y-rotation demo (per-l Allgatherv over m):

```bash
mpiexec -n 2 julia --project=. examples/rotate_y_parallel.jl
```

```

Enable rfft in distributed plans (when supported):

```julia
using SHTnsKit, MPI, PencilArrays, PencilFFTs
MPI.Init()
cfg = create_gauss_config(16, 18; nlon=33)

# Let SHTnsKit suggest a balanced θ/φ pencil layout
pθ, pφ = suggest_pencil_grid(MPI.COMM_WORLD, cfg.nlat, cfg.nlon; allow_one_dim=false)
Pθφ = PencilArrays.Pencil((:θ,:φ), (cfg.nlat, cfg.nlon); comm=MPI.COMM_WORLD)
# (Pass `procgrid=(pθ,pφ)` or similar keyword if your PencilArrays version supports it.)

# Scalar analysis with rfft
aplan = DistAnalysisPlan(cfg, PencilArrays.zeros(Pθφ; eltype=Float64); use_rfft=true)

# Vector transforms with rfft + optional spatial scratch to avoid iFFT allocs for real outputs
vplan = DistSphtorPlan(cfg, PencilArrays.zeros(Pθφ; eltype=Float64); use_rfft=true, with_spatial_scratch=true)

# Cache PencilFFT plans across calls once warm-up completes
enable_fft_plan_cache!()
MPI.Finalize()
```

### Automatic Differentiation

Full support for gradient-based optimization:

```julia
using SHTnsKit, ForwardDiff, Zygote

# Setup
cfg = create_gauss_config(12, 14; nlon=25)
sh_coeffs = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
sh_coeffs[1,1] = 1.0

# Forward-mode differentiation
objective(sh) = sum(abs2, synthesis(cfg, sh))
gradient = ForwardDiff.gradient(objective, sh_coeffs)

# Reverse-mode differentiation (better for many parameters)
loss_val, grad = Zygote.withgradient(objective, sh_coeffs)

destroy_config(cfg)
```

Supported functions include all core transforms, vector operations, spectral analysis, and differential operators.

### Allocation Benchmarks

```bash
# Serial and (if available) MPI allocation benchmarks
julia --project=. examples/alloc_benchmark.jl 16
mpiexec -n 2 julia --project=. examples/alloc_benchmark.jl 16

Tip: To avoid allocations for real-output distributed synthesis, construct plans with `with_spatial_scratch=true`, which keeps a single complex (θ,φ) scratch buffer inside the plan. This modest, fixed footprint removes per-call allocations for iFFT writes when outputs are real.
```

##  Contributing

Contributions are welcome! Areas of particular interest:

- **GPU Computing**: CUDA/ROCm support for massive parallelism
- **Performance Optimization**: Architecture-specific tuning

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Citation

If you use SHTnsKit.jl in your research, please cite:
```bibtex
@article{schaeffer2013efficient,
  title={Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations},
  author={Schaeffer, Nathana{\"e}l},
  journal={Geochemistry, Geophysics, Geosystems},
  volume={14},
  number={3},
  pages={751--758},
  year={2013},
  publisher={Wiley Online Library}
}
```

##  License

SHTnsKit.jl is released under the GNU General Public License v3.0 (GPL-3.0), ensuring compatibility with the underlying SHTns library.

## References

- **[SHTns Documentation](https://nschaeff.bitbucket.io/shtns/)**: Original C library
- **[Spherical Harmonics Theory](https://en.wikipedia.org/wiki/Spherical_harmonics)**: Mathematical background  
