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

    # Roundtrip diagnostics: fused vs non-fused
    alm_fused = analysis(cfg, f; use_fused_loops=true)
    f2_fused = synthesis(cfg, alm_fused; real_output=true, use_fused_loops=true)
    alm_nf = analysis(cfg, f; use_fused_loops=false)
    f2_nf = synthesis(cfg, alm_nf; real_output=true, use_fused_loops=false)
    # Local errors
    lmax_f = maximum(abs.(f2_fused .- f))
    lrel_f = sqrt(sum(abs2, f2_fused .- f) / (sum(abs2, f) + eps()))
    lmax_nf = maximum(abs.(f2_nf .- f))
    lrel_nf = sqrt(sum(abs2, f2_nf .- f) / (sum(abs2, f) + eps()))
    # Reduce max across ranks
    gmax_f = Ref(0.0); grel_f = Ref(0.0); gmax_nf = Ref(0.0); grel_nf = Ref(0.0)
    MPI.Allreduce!(Ref(lmax_f), gmax_f, MPI.MAX, COMM)
    MPI.Allreduce!(Ref(lrel_f), grel_f, MPI.MAX, COMM)
    MPI.Allreduce!(Ref(lmax_nf), gmax_nf, MPI.MAX, COMM)
    MPI.Allreduce!(Ref(lrel_nf), grel_nf, MPI.MAX, COMM)

    if RANK == 0
        println("Roundtrip (fused):    max|f̂−f|=$(gmax_f[]), rel=$(grel_f[])")
        println("Roundtrip (nonfused): max|f̂−f|=$(gmax_nf[]), rel=$(grel_nf[])\n")
    end

    # Optional: demonstrate safe PencilArrays allocation patterns
    if HAVE_PENCIL
        if RANK == 0
            println("PencilArrays detected. Demonstrating safe allocation…")
        end
        MPI.Barrier(COMM)
        try
            # Construct a 2D pencil decomposition matching the spatial grid
            # Prefer named-axes API, fall back to older positional signature
            pencil = try
                PencilArrays.Pencil((:θ,:φ), (cfg.nlat, cfg.nlon); comm=COMM)
            catch
                PencilArrays.Pencil((cfg.nlat, cfg.nlon), COMM)
            end

            # Try multiple variants for cross-version compatibility
            A = try
                PencilArrays.zeros(Float64, pencil)
            catch
                try
                    PencilArrays.zeros(pencil; eltype=Float64)
                catch
                    try
                        PencilArrays.PencilArray(undef, Float64, pencil) |> x -> (fill!(x, 0.0); x)
                    catch
                        try
                            PencilArrays.PencilArray{Float64}(undef, pencil) |> x -> (fill!(x, 0.0); x)
                        catch
                            error("No compatible PencilArrays zeros/similar/PencilArray constructor found")
                        end
                    end
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

        # Optional: distributed packed roundtrip (uses extension helpers)
        try
            MPI.Barrier(COMM)
            if RANK == 0
                println("Distributed packed roundtrip demo…")
            end
            pencil = try
                PencilArrays.Pencil((:θ,:φ), (cfg.nlat, cfg.nlon); comm=COMM)
            catch
                PencilArrays.Pencil((cfg.nlat, cfg.nlon), COMM)
            end
            fθφ = try
                PencilArrays.zeros(Float64, pencil)
            catch
                try
                    PencilArrays.zeros(pencil; eltype=Float64)
                catch
                    PencilArrays.PencilArray(undef, Float64, pencil) |> x -> (fill!(x, 0.0); x)
                end
            end
            # Local fill (deterministic pattern)
            for (iθ, iφ) in Iterators.product(axes(fθφ,1), axes(fθφ,2))
                fθφ[iθ, iφ] = sin(0.11*Int(iθ)) + cos(0.07*Int(iφ))
            end
            Qlm = SHTnsKit.dist_spat_to_SH_packed(cfg, fθφ)
            fθφ_rt = SHTnsKit.dist_SH_packed_to_spat(cfg, Qlm; prototype_θφ=fθφ, real_output=true)
            # Relative error across ranks
            lrel = sqrt(sum(abs2, Array(fθφ_rt) .- Array(fθφ)) / (sum(abs2, Array(fθφ)) + eps()))
            grel = Ref(0.0)
            MPI.Allreduce!(Ref(lrel), grel, MPI.MAX, COMM)
            if RANK == 0
                println("Distributed packed roundtrip rel error: $(grel[])\n")
            end
        catch e
            if RANK == 0
                @warn "Distributed packed roundtrip demo skipped" exception=(e, catch_backtrace())
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
