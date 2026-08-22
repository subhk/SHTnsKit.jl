#!/usr/bin/env julia

# Parallel roundtrip demo with safe PencilArrays allocation
#
# Run (2 processes):
#   mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl
#
# What it does:
# - Initializes MPI and reports ranks
# - Runs accurate serial and distributed spherical-harmonic roundtrips
# - Reduces the max error across ranks and prints it on rank 0
# - Demonstrates direct PencilArrays 0.19 construction and allocation

# Load MPI and PencilArrays FIRST so that SHTnsKit's parallel extension is triggered
try
    using MPI
catch e
    @error "MPI.jl is not available in this environment" exception=(e, catch_backtrace())
    exit(1)
end

# Load PencilArrays/PencilFFTs to enable the parallel extension
try
    @eval using PencilArrays
    @eval using PencilFFTs
catch e
    @error "This example requires PencilArrays.jl and PencilFFTs.jl" exception=(e, catch_backtrace())
    exit(1)
end

# Now load SHTnsKit - the parallel extension will be loaded if MPI+PencilArrays+PencilFFTs are available
using SHTnsKit

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const SIZE = MPI.Comm_size(COMM)

if RANK == 0
    println("SHTnsKit serial and distributed roundtrips")
    println("MPI processes: $SIZE")
end

# Create an SHT configuration (same on all ranks)
let
    # Problem size – modest so it runs fast under multiple ranks
    lmax = 24
    nlat = 32
    nlon = 64
    cfg = create_gauss_config(lmax, nlat; mmax=lmax, nlon=nlon)

    # Analytic, band-limited field: a constant plus l=1,m=1 and l=2,m=0 modes.
    # A truncated transform cannot exactly reconstruct arbitrary random grid data.
    field_value(iθ, iφ) = begin
        x = cfg.x[iθ]
        sinθ = sqrt(max(0.0, 1 - x^2))
        1 + 0.25 * (3x^2 - 1) + 0.1 * sinθ * cos(cfg.φ[iφ])
    end
    f = [field_value(iθ, iφ) for iθ in 1:cfg.nlat, iφ in 1:cfg.nlon]

    # Roundtrip diagnostics
    alm = analysis(cfg, f)
    f2 = synthesis(cfg, alm; real_output=true)
    # Local errors
    lmax_err = maximum(abs.(f2 .- f))
    lrel_err = sqrt(sum(abs2, f2 .- f) / (sum(abs2, f) + eps()))
    # Reduce max across ranks
    gmax = Ref(0.0); grel = Ref(0.0)
    MPI.Allreduce!(Ref(lmax_err), gmax, MPI.MAX, COMM)
    MPI.Allreduce!(Ref(lrel_err), grel, MPI.MAX, COMM)

    if RANK == 0
        println("Roundtrip: max|f̂−f|=$(gmax[]), rel=$(grel[])\n")
    end
    gmax[] < 1e-10 && grel[] < 1e-10 || error("Serial roundtrip accuracy check failed")

    if RANK == 0
        println("PencilArrays detected. Demonstrating safe allocation…")
    end
    MPI.Barrier(COMM)

    # Decompose latitude and keep each longitude row local.
    pencil = PencilArrays.Pencil((cfg.nlat, cfg.nlon), (1,), COMM)
    local_dims = PencilArrays.size_local(pencil)
    A = PencilArray(pencil, zeros(Float64, local_dims...))
    B = PencilArray(pencil, zeros(ComplexF64, local_dims...))
    if RANK == 0
        println("Allocated A::$(typeof(A)) and B::$(typeof(B)) safely")
        println("Distributed packed roundtrip demo…")
    end

    fθφ = PencilArray(pencil, zeros(Float64, local_dims...))
    local_field = parent(fθφ)
    ranges = PencilArrays.range_local(pencil)
    for (i_local, i_global) in enumerate(ranges[1]),
        (j_local, j_global) in enumerate(ranges[2])
        local_field[i_local, j_local] = field_value(i_global, j_global)
    end

    Qlm = SHTnsKit.dist_analysis_packed(cfg, fθφ)
    fθφ_rt = SHTnsKit.dist_synthesis_packed(
        cfg, Qlm; prototype_θφ=fθφ, real_output=true
    )

    local_recovered = parent(fθφ_rt)
    local_max = maximum(abs.(local_recovered .- local_field))
    local_error2 = sum(abs2, local_recovered .- local_field)
    local_norm2 = sum(abs2, local_field)
    global_max = MPI.Allreduce(local_max, MPI.MAX, COMM)
    global_error2 = MPI.Allreduce(local_error2, MPI.SUM, COMM)
    global_norm2 = MPI.Allreduce(local_norm2, MPI.SUM, COMM)
    global_rel = sqrt(global_error2 / (global_norm2 + eps()))

    if RANK == 0
        println("Distributed packed roundtrip: max error=$global_max, rel=$global_rel\n")
    end
    global_max < 1e-10 && global_rel < 1e-10 ||
        error("Distributed packed roundtrip accuracy check failed")

    destroy_config(cfg)
end

MPI.Barrier(COMM)
if RANK == 0
    println("Done.")
end
MPI.Finalize()
