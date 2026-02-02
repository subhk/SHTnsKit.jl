## Distributed Round-trip Example (MPI)

This sketch shows how to run a distributed scalar round-trip using PencilArrays
and PencilFFTs. It assumes you have constructed a `(θ,φ)` prototype PencilArray
on your MPI communicator.

```julia
using MPI, PencilArrays, PencilFFTs, SHTnsKit

MPI.Init()
comm = COMM_WORLD

# Build config and prototype (θ,φ) pencil
cfg = create_gauss_config(32, 40; nlon=65)

# Assume you created a prototype `(θ,φ)` pencil named proto_θφ.
# For example (API depends on your PencilArrays version):
# proto_θφ = allocate(comm; dims=(:θ,:φ), sizes=(cfg.nlat, cfg.nlon), eltype=Float64)

# Fill a test field
fθφ = similar(proto_θφ)
foreachindex(fθφ) do I
    fθφ[I] = rand()
end

# Distributed analysis -> Alm (dense for now)
Alm = dist_analysis(cfg, fθφ)

# Distributed synthesis using prototype
fθφ_out = dist_synthesis(cfg, Alm; prototype_θφ=fθφ)

# Compute local/global relative errors
rel_local, rel_global = dist_scalar_roundtrip!(cfg, fθφ)
if Comm_rank(comm) == 0
    @info "Scalar round-trip rel error" rel_local rel_global
end

MPI.Finalize()
```

For vector fields, use `dist_analysis_sphtor` and `dist_synthesis_sphtor` with the
same `prototype_θφ`. Precomputing Legendre tables via `prepare_plm_tables!(cfg)` on
regular grids typically improves performance.
