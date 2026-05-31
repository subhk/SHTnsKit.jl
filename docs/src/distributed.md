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

# Create distributed array — decompose LATITUDE (θ), the scalable axis.
comm = MPI.COMM_WORLD
pen = SHTnsKit.create_spatial_pencil(cfg; comm)   # = Pencil((nlat, nlon), (1,), comm)
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

!!! warning "Decompose latitude (θ), not longitude (φ)"
    PencilArrays splits the **last** dimension by default, so the bare
    `Pencil((nlat, nlon), comm)` decomposes **longitude (φ)** — the
    non-scaling axis. The φ-distributed analysis path `Allgatherv!`s the full
    longitude onto every rank and then replicates the Legendre transform, so it
    does **not** strong-scale (it usually gets *slower* as you add ranks).

    Decompose **latitude (θ)** instead. Use `SHTnsKit.create_spatial_pencil(cfg; comm)`
    (or `create_spatial_array(cfg; comm)`), which builds `Pencil((nlat, nlon), (1,), comm)`.
    Each rank then integrates Legendre over its local θ band and only a small
    `(lmax+1, mmax+1)` spectral `Allreduce!` remains — this scales near-linearly.
    Measured (lmax=511): analysis 1.93×@2 ranks, 3.31×@4; synthesis 1.94×/3.45×.
    Passing a φ-distributed array to `dist_analysis` emits a one-time warning.

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
pen = Pencil((nlat, nlon), (1,), comm)

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
pen = Pencil((nlat, nlon), (1,), comm)
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

## Transpose-based distributed transforms (scalable)

The gather path (`dist_analysis` / `dist_synthesis`) Allgathers the full spectrum
onto every rank, so spectral memory is replicated (`O(lmax²)` per rank regardless of
`p`). For large `lmax` or high rank counts this becomes a bottleneck.

`DistTransposePlan` is a fully distributed alternative: the spectral array is
**distributed over m** (each rank owns a disjoint set of m-columns), giving
`O(lmax² / p)` spectral memory per rank. Internally, PencilFFTs performs the
longitude FFT and a global transpose (one Alltoall per slab/batch), and the Legendre
transform is purely local per rank. The result is measured **5–7× faster than the
gather path at 4 ranks** for `lmax = 256`.

!!! tip "When to use `DistTransposePlan`"
    - High rank counts (≥ 4) where the Allreduce of the gather path becomes a bottleneck.
    - Applications that want distributed spectral memory (`O(lmax² / p)` per rank).
    - Batched transforms over `nlev` radial levels — all levels are processed in a
      single call, amortising the Alltoall cost.

### Data layout

- **Spatial** (`allocate_spatial`): real `PencilArray`, logical shape `(nlon, nlat_local, nlev)`.
  φ is the *fast* (innermost) index; the FFT is taken along φ. θ is distributed across
  ranks; `nlev` is a local batch dimension (radial levels, e.g. depth slabs).
- **Spectral** (`allocate_spectral`): complex `PencilArray`, logical shape
  `(lmax+1, nm_local, nlev)`. m is distributed; for each owned m, all l ≥ m are stored
  (l is the slow index).

### Runnable example

```julia
using MPI, SHTnsKit, PencilArrays, PencilFFTs
MPI.Init()
comm = MPI.COMM_WORLD

cfg = create_gauss_config(256, 258; nlon=513)

# Build a transpose plan: 8 radial levels batched per Alltoall
plan = DistTransposePlan(cfg; comm=comm, nlev=8, use_rfft=true)

f   = allocate_spatial(plan)    # real PencilArray: (nlon, nlat_local, 8)
Alm = allocate_spectral(plan)   # complex PencilArray: (lmax+1, nm_local, 8)

# Fill parent(f) with your data: parent(f)[φ_local, θ_local, lev]
r = PencilArrays.range_local(pencil(f))   # r[1]=φ range, r[2]=θ range
# for (il, ig) in enumerate(r[2]), (jl, jg) in enumerate(r[1])
#     parent(f)[jl, il, lev] = my_field[ig, jg, lev]

dist_analysis!(plan, Alm, f)    # spatial → distributed spectral (m-pencil)
dist_synthesis!(plan, f, Alm)   # distributed spectral → spatial

# Vector (toroidal/poloidal) transforms
Slm = allocate_spectral(plan); Tlm = allocate_spectral(plan)
Vt  = allocate_spatial(plan);  Vp  = allocate_spatial(plan)
plan_v = DistTransposePlan(cfg; comm=comm, nlev=8, use_rfft=true, with_vector=true)
dist_analysis_sphtor!(plan_v, Slm, Tlm, Vt, Vp)
dist_synthesis_sphtor!(plan_v, Vt, Vp, Slm, Tlm)

# QST (radial + toroidal/poloidal) transforms
Qlm = allocate_spectral(plan_v)
Vr  = allocate_spatial(plan_v)
dist_analysis_qst!(plan_v, Qlm, Slm, Tlm, Vr, Vt, Vp)
dist_synthesis_qst!(plan_v, Vr, Vt, Vp, Qlm, Slm, Tlm)

MPI.Finalize()
```

### API summary

| Function | Description |
|----------|-------------|
| `DistTransposePlan(cfg; comm, nlev, use_rfft, with_vector)` | Build plan (Legendre tables + PencilFFTs plan) |
| `allocate_spatial(plan)` | Real `PencilArray` `(nlon, nlat_local, nlev)` |
| `allocate_spectral(plan)` | Complex `PencilArray` `(lmax+1, nm_local, nlev)` |
| `dist_analysis!(plan, Alm, f)` | Scalar spatial → spectral |
| `dist_synthesis!(plan, f, Alm)` | Scalar spectral → spatial |
| `dist_analysis_sphtor!(plan, Slm, Tlm, Vt, Vp)` | Vector (sphtor) analysis |
| `dist_synthesis_sphtor!(plan, Vt, Vp, Slm, Tlm)` | Vector (sphtor) synthesis |
| `dist_analysis_qst!(plan, Qlm, Slm, Tlm, Vr, Vt, Vp)` | QST analysis |
| `dist_synthesis_qst!(plan, Vr, Vt, Vp, Qlm, Slm, Tlm)` | QST synthesis |

!!! note "Relationship to the gather path"
    `DistTransposePlan` **complements** (does not replace) the dense `dist_analysis` /
    `dist_synthesis` gather path. The gather path produces a fully replicated dense
    `(lmax+1, mmax+1)` spectrum on every rank, which is convenient for post-processing
    and operator applications that need global spectral access. The transpose path keeps
    the spectrum distributed (each rank sees only its m-columns) and is the right choice
    when spectral memory or Allreduce cost is a bottleneck.

    **Known follow-up:** at ≥ 4 ranks the Legendre step has a ~1.7× load imbalance
    because low-m columns (owned by rank 0) have more non-zero (l, m) pairs than high-m
    columns. Interleaved-m ownership (round-robin assignment of m to ranks) will
    equalise this and is planned for a future release.

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

