# Installation Guide

This guide provides detailed instructions for installing SHTnsKit.jl and its dependencies.

## Quick Installation

```julia
using Pkg
Pkg.add("SHTnsKit")
```

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL
- **Julia**: Version 1.9 or later (1.11+ recommended)
- **Memory**: At least 4GB RAM (16GB+ for large parallel problems)
- **Storage**: 2GB free space for dependencies (including MPI)
- **MPI Library**: OpenMPI or MPICH for parallel functionality

### Required Dependencies

SHTnsKit.jl is pure Julia and does not require an external C library. Core functionality uses Julia's standard libraries and FFTW.jl (installed automatically). Parallel features require additional packages.

## Installing SHTnsKit.jl

### Basic Installation (Serial Only)

```julia
using Pkg
Pkg.add("SHTnsKit")
```

### Full Installation (Parallel + SIMD)

```julia
using Pkg
Pkg.add(["SHTnsKit", "MPI", "PencilArrays", "PencilFFTs", "LoopVectorization"])
```

### Development Installation

For the latest features or contributing:

```julia
using Pkg
Pkg.add(url="https://github.com/username/SHTnsKit.jl.git")
```

### Local Development Setup

```julia
using Pkg
Pkg.develop(path="/path/to/SHTnsKit.jl")
```

## Parallel Computing Setup

### MPI Installation

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install libopenmpi-dev openmpi-bin
```

**macOS:**
```bash
brew install open-mpi
```

**Configure Julia MPI:**
```julia
using Pkg
Pkg.add("MPI")
Pkg.build("MPI")
```

### Verify MPI Installation

```julia
using MPI
MPI.Init()
rank = Comm_rank(COMM_WORLD)
size = Comm_size(COMM_WORLD)
println("Process $rank of $size")
MPI.Finalize()
```

### Optional Performance Packages

```julia
using Pkg
Pkg.add(["LoopVectorization", "BenchmarkTools"])
```

## Verification

### Basic Functionality Test

```julia
using SHTnsKit

# Create simple configuration
cfg = create_gauss_config(8, 8)
println("lmax: ", get_lmax(cfg))
println("nlat: ", cfg.nlat)  
println("nphi: ", cfg.nlon)

# Test basic transform
# Create bandlimited test coefficients (avoids high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
spat = synthesis(cfg, sh)
println("Transform successful: ", size(spat))

destroy_config(cfg)
println("SHTnsKit.jl installation verified!")
```

### Parallel Functionality Test

Save this as `test_mpi.jl` and run with `mpiexec -n 2 julia --project test_mpi.jl`:

```julia
using MPI
MPI.Init()

using SHTnsKit, PencilArrays, PencilFFTs

# Create configuration
lmax = 16
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create distributed array
pen = Pencil((nlat, nlon), MPI.COMM_WORLD)
fθφ = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))

# Fill with test data (Y_2^0 pattern)
ranges = PencilArrays.range_local(pen)
for (i_local, i_global) in enumerate(ranges[1])
    x = cfg.x[i_global]
    for j in 1:length(ranges[2])
        fθφ[i_local, j] = (3*x^2 - 1)/2
    end
end

# Distributed roundtrip test
Alm = SHTnsKit.dist_analysis(cfg, fθφ)
fθφ_recovered = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)

# Verify accuracy
max_err = maximum(abs.(parent(fθφ_recovered) .- parent(fθφ)))
global_max_err = MPI.Allreduce(max_err, MPI.MAX, MPI.COMM_WORLD)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("Parallel roundtrip error: $global_max_err")
    println(global_max_err < 1e-10 ? "SUCCESS!" : "FAILED")
end

destroy_config(cfg)
MPI.Finalize()
```

### Extended Verification

```julia
using SHTnsKit, Test, LinearAlgebra

@testset "Installation Verification" begin
    # Basic functionality
    lmax = 16
    cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)

    # Create bandlimited test pattern
    spatial = zeros(cfg.nlat, cfg.nlon)
    for i in 1:cfg.nlat
        x = cfg.x[i]
        spatial[i, :] .= (3*x^2 - 1)/2  # Y_2^0
    end

    # Roundtrip test
    Alm = analysis(cfg, spatial)
    recovered = synthesis(cfg, Alm)
    @test norm(spatial - recovered) < 1e-12

    # Memory management
    destroy_config(cfg)
    @test true  # No crash
end
```

## Troubleshooting

### Common Issues

**1. Array size mismatch:**
```
ERROR: DimensionMismatch: spatial_data size (X, Y) must be (nlat, nlon)
```

**Fix:** Ensure `size(Alm) == (cfg.lmax+1, cfg.mmax+1)` and `size(spatial) == (cfg.nlat, cfg.nlon)`.

**2. Memory issues:**
```
ERROR: Out of memory
```

**Solutions:**
- Reduce problem size (lmax)
- Increase system swap space
- Reuse allocations with in‑place APIs (`synthesis!`, `analysis!`)

### Advanced Debugging

**Julia environment check:**
```julia
using Libdl
println(Libdl.dllist())  # List all loaded libraries
```

## Performance Optimization

### System-Level Optimizations

Threading and memory tips:
```julia
# Enable SHTnsKit internal threading and FFTW threads
set_optimal_threads!()
println((threads=get_threading(), fft_threads=get_fft_threads()))

# Prevent oversubscription with BLAS/FFTW (optional)
ENV["OPENBLAS_NUM_THREADS"] = "1"
```

### Julia-Specific

**Precompilation:**
```julia
using PackageCompiler
create_sysimage([:SHTnsKit]; sysimage_path="shtns_sysimage.so")
```

**Memory:**
```bash
julia --heap-size-hint=8G script.jl
```

## Docker Installation

For containerized environments:

```dockerfile
FROM julia:1.11

# Install Julia packages
RUN julia -e 'using Pkg; Pkg.add(["SHTnsKit"])'

# Verify installation
RUN julia -e 'using SHTnsKit; cfg = create_gauss_config(8,8); destroy_config(cfg)'
```

## Getting Help

- **Documentation**: [SHTnsKit.jl Docs](https://subhk.github.io/SHTnsKit.jl/)
- **Issues**: [GitHub Issues](https://github.com/subhk/SHTnsKit.jl/issues)
- **Julia Discourse**: [Julia Community](https://discourse.julialang.org/)
