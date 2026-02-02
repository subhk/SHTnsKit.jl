# Distributed Computing

```@raw html
<div style="background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
    <h2 style="margin: 0 0 0.5rem 0; color: white; border: none;">MPI Parallelization</h2>
    <p style="margin: 0; opacity: 0.9;">Scale your transforms across thousands of cores with PencilArrays</p>
</div>
```

SHTnsKit.jl integrates seamlessly with MPI through PencilArrays.jl for distributed memory parallelization. This enables scaling spherical harmonic transforms to large HPC clusters.

!!! tip "When to Use Distributed Computing"
    MPI parallelization is most beneficial for **large problems (lmax > 128)** or when processing many fields simultaneously. For single-field transforms on smaller problems, consider GPU acceleration instead.

---

## Quick Start

```julia
using MPI
MPI.Init()

using SHTnsKit, PencilArrays, PencilFFTs

# Configuration
lmax = 64
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create distributed array
comm = MPI.COMM_WORLD
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

# Distributed transforms
Alm = SHTnsKit.dist_analysis(cfg, fθφ)
fθφ_recovered = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)

# Verify
max_err = maximum(abs.(parent(fθφ_recovered) .- parent(fθφ)))
global_max_err = MPI.Allreduce(max_err, MPI.MAX, comm)

if MPI.Comm_rank(comm) == 0
    println("Roundtrip error: $global_max_err")
end

destroy_config(cfg)
MPI.Finalize()
```

---

## Installation

### Requirements

- Julia 1.10+
- MPI library (OpenMPI or MPICH)
- PencilArrays.jl and PencilFFTs.jl

### Setup

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```

**macOS:**
```bash
brew install open-mpi
```

**Julia packages:**
```julia
using Pkg
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

---

## API Reference

### Distributed Transforms

| Function | Description |
|----------|-------------|
| `dist_analysis(cfg, fθφ)` | Distributed spatial → spectral transform |
| `dist_synthesis(cfg, Alm; prototype_θφ)` | Distributed spectral → spatial transform |
| `dist_analysis_sphtor(cfg, Vθ, Vφ)` | Distributed vector field analysis |
| `dist_synthesis_sphtor(cfg, Slm, Tlm; prototype_θφ)` | Distributed vector field synthesis |

### Spectral Operators

| Function | Description |
|----------|-------------|
| `dist_scalar_laplacian(cfg, fθφ)` | Distributed Laplacian computation |
| `dist_spatial_divergence(cfg, Vθ, Vφ)` | Distributed divergence |
| `dist_spatial_vorticity(cfg, Vθ, Vφ)` | Distributed vorticity |

---

## Running MPI Programs

Save your script as `transform_mpi.jl` and run with:

```bash
mpiexec -n 4 julia --project transform_mpi.jl
```

---

## Working with PencilArrays

### Understanding Domain Decomposition

PencilArrays distributes data across MPI ranks by splitting along one or more dimensions:

```julia
using MPI, PencilArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Create pencil decomposition
pen = Pencil((nlat, nlon), comm)

# Each rank owns a portion of the data
local_size = PencilArrays.size_local(pen)
global_range = PencilArrays.range_local(pen)

println("Rank $rank: local size = $local_size, global range = $global_range")
```

### Accessing Local Data

```julia
# Create distributed array
fθφ = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))

# Access local data directly
local_data = parent(fθφ)

# Use local_range() helper from SHTnsKit
for idx in local_range(fθφ)
    local_data[idx] = some_value
end
```

---

## Vector Field Transforms

```julia
using MPI
MPI.Init()

using SHTnsKit, PencilArrays, PencilFFTs

comm = MPI.COMM_WORLD
lmax = 64
nlat, nlon = lmax + 2, 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create distributed vector field components
pen = Pencil((nlat, nlon), comm)
Vθ = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))
Vφ = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))

# Fill with test data
ranges = PencilArrays.range_local(pen)
for (i_local, i_global) in enumerate(ranges[1])
    θ = cfg.θ[i_global]
    for (j_local, j_global) in enumerate(ranges[2])
        φ = cfg.φ[j_global]
        Vθ[i_local, j_local] = cos(θ) * sin(φ)
        Vφ[i_local, j_local] = cos(φ)
    end
end

# Distributed vector transform
Slm, Tlm = SHTnsKit.dist_analysis_sphtor(cfg, Vθ, Vφ)
Vθ_out, Vφ_out = SHTnsKit.dist_synthesis_sphtor(cfg, Slm, Tlm; prototype_θφ=Vθ)

destroy_config(cfg)
MPI.Finalize()
```

---

## Best Practices

!!! note "Performance Tips"
    1. **Problem size**: MPI overhead is significant for small problems. Use lmax > 64 for benefit.
    2. **Process count**: Scale processes with problem size. Too many processes hurt efficiency.
    3. **Memory**: Each rank needs memory for local data plus communication buffers.
    4. **I/O**: Use parallel I/O (HDF5, NetCDF) for large datasets.

---

## See Also

- [GPU Acceleration](gpu.md) - Single-node GPU speedup
- [Performance Guide](performance.md) - Optimization strategies
- [API Reference](api/index.md) - Complete function documentation

