#!/usr/bin/env julia

# Parallel roundtrip demo with safe PencilArrays allocation
#
# Run (2 processes):
#   mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl
#
# What it does:
# - Initializes MPI and reports ranks
# - Runs a spherical-harmonic analysis+synthesis roundtrip on each rank
# - Reduces the max error across ranks and prints it on rank 0
# - Demonstrates how to safely allocate arrays from a Pencil using
#   PencilArrays.zeros(T, pencil) or similar(pencil, T) + fill!

using SHTnsKit
using Random

# Load MPI; keep the example usable even if MPI is not present
try
    using MPI
catch e
    @error "MPI.jl is not available in this environment" exception=(e, catch_backtrace())
    exit(1)
end

# Load PencilArrays/PencilFFTs optionally. The SHT roundtrip below does not
# depend on them, but we demonstrate safe allocation from a Pencil when present.
const HAVE_PENCIL = try
    @eval using PencilArrays
    true
catch
    false
end

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const SIZE = MPI.Comm_size(COMM)

if RANK == 0
    println("SHTnsKit parallel roundtrip (each rank runs a serial transform)")
    println("MPI processes: $SIZE")
end

# Create an SHT configuration (same on all ranks)
let
    # Problem size – modest so it runs fast under multiple ranks
    lmax = 24
    nlat = 32
    nlon = 64
    cfg = create_gauss_config(lmax, nlat; mmax=lmax, nlon=nlon)

    # Make deterministic across ranks
    Random.seed!(1234)

    # Create a real spatial field on the Gauss×equiangular grid
    f = randn(Float64, cfg.nlat, cfg.nlon)

    # Roundtrip: spatial -> spectral (full) -> spatial
    alm = analysis(cfg, f)
    f2 = synthesis(cfg, alm; real_output=true)

    # Compute local max abs error and reduce to global
    local_err = maximum(abs.(f2 .- f))
    global_err = Ref(0.0)
    MPI.Allreduce!(Ref(local_err), global_err, MPI.MAX, COMM)

    if RANK == 0
        println("Roundtrip complete. max|f̂−f| = $(global_err[])")
    end

    # Optional: demonstrate safe PencilArrays allocation patterns
    if HAVE_PENCIL
        if RANK == 0
            println("PencilArrays detected. Demonstrating safe allocation…")
        end
        MPI.Barrier(COMM)
        try
            # Construct a 2D pencil decomposition matching the spatial grid
            # Note: API varies slightly across versions; the following pattern
            # works with recent PencilArrays:
            pencil = PencilArrays.Pencil((cfg.nlat, cfg.nlon), COMM)

            # Try multiple variants for cross-version compatibility
            A = try
                PencilArrays.zeros(Float64, pencil)
            catch
                try
                    PencilArrays.zeros(pencil; eltype=Float64)
                catch
                    tmp = similar(pencil, Float64); fill!(tmp, 0.0); tmp
                end
            end

            # Safe fallback that works across versions
            B = similar(pencil, ComplexF64)
            fill!(B, zero(ComplexF64))

            # Avoid the problematic pattern: zeros(pencil; eltype=…)
            if RANK == 0
                println("Allocated A::$(typeof(A)) and B::$(typeof(B)) safely")
            end
        catch e
            if RANK == 0
                @warn "PencilArrays allocation demo failed (version/API mismatch)" exception=(e, catch_backtrace())
            end
        end
    elseif RANK == 0
        println("PencilArrays not available; skipping pencil allocation demo.")
    end
end

MPI.Barrier(COMM)
if RANK == 0
    println("Done.")
end
MPI.Finalize()
