# Advanced Optimizations Implementation

This document summarizes performance optimization capabilities available in SHTnsKit.jl.

## Performance Features

### Threading Support
- Julia multi-threading for internal operations
- FFTW threading for FFT computations

### MPI Parallelism
- Distributed transforms with PencilArrays
- Domain decomposition across MPI ranks

### SIMD Vectorization
- Automatic vectorization of inner loops
- Complex arithmetic optimization

## Usage Examples

### 1. Basic Serial Usage

```julia
using SHTnsKit

lmax = 64
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create coefficients (2D matrix format)
Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Alm[1, 1] = 1.0
Alm[3, 1] = 0.5

# Perform transform
spatial_data = synthesis(cfg, Alm)

destroy_config(cfg)
```

### 2. Distributed MPI Usage

```julia
using MPI
MPI.Init()

using SHTnsKit, PencilArrays, PencilFFTs

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# Create configuration
lmax = 128
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create distributed array using PencilArrays
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

# Distributed transforms
Alm = SHTnsKit.dist_analysis(cfg, fθφ)
fθφ_out = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)

destroy_config(cfg)
MPI.Finalize()
```

### 3. Vector Field Transforms

```julia
using SHTnsKit

lmax = 64
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create velocity field
Vθ = zeros(cfg.nlat, cfg.nlon)
Vφ = zeros(cfg.nlat, cfg.nlon)

for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ = cfg.θ[i]
    φ = cfg.φ[j]
    Vθ[i,j] = cos(θ) * sin(φ)
    Vφ[i,j] = cos(φ)
end

# Decompose into spheroidal/toroidal components
Slm, Tlm = spat_to_SHsphtor(cfg, Vθ, Vφ)

# Reconstruct velocity field
Vθ_rec, Vφ_rec = SHsphtor_to_spat(cfg, Slm, Tlm)

destroy_config(cfg)
```

## Performance Guidelines

### Threading
- Set `JULIA_NUM_THREADS` for Julia threading
- FFTW automatically uses available threads

### Memory
- Use in-place operations for large problems (`synthesis!`, `analysis!`)
- Reuse coefficient matrices when possible

### MPI
- Use PencilArrays for domain decomposition
- Ensure proper precompilation before MPI runs

## Hardware Requirements

- **Minimum**: Any system supported by Julia
- **Recommended**: Multi-core CPU
- **Optimal for MPI**: Cluster with high-speed interconnect
- **Dependencies**: MPI.jl, PencilArrays.jl, PencilFFTs.jl for parallel features