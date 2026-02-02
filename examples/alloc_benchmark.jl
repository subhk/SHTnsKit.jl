#!/usr/bin/env julia

using SHTnsKit
using Random

function bench_serial(lmax::Int)
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(42)
    f = randn(rng, nlat, nlon)
    alm = zeros(ComplexF64, lmax+1, lmax+1)
    # Baseline analysis
    a_alloc = @allocated analysis(cfg, f)
    # Plan-based analysis!
    plan = SHTPlan(cfg)
    a_alloc_plan = @allocated analysis!(plan, alm, f)
    # Baseline synthesis
    alm_rnd = randn(rng, ComplexF64, lmax+1, lmax+1)
    s_alloc = @allocated synthesis(cfg, alm_rnd)
    # Plan-based synthesis!
    fout = similar(f)
    s_alloc_plan = @allocated synthesis!(plan, fout, alm_rnd)
    return (; a_alloc, a_alloc_plan, s_alloc, s_alloc_plan)
end

function try_require(pkgs...)
    for p in pkgs
        try
            Base.require(Base.PkgId(Base.UUID("00000000-0000-0000-0000-000000000000"), String(p)))
        catch
        end
    end
end

function bench_distributed(lmax::Int)
    try
        @eval using MPI, PencilArrays, PencilFFTs
    catch
        println("Distributed deps not found; skipping distributed benchmark")
        return nothing
    end
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    Pθφ = Pencil((nlat, nlon), (MPI.Comm_size(comm), 1), comm)
    paalloc(t; eltype=Float64) = (try PencilArrays.zeros(t; eltype=eltype) catch; try PencilArrays.zeros(t) catch; try SHTnsKitParallelExt.allocate(t; eltype=eltype) catch; SHTnsKitParallelExt.allocate(t) end end end)
    fθφ = paalloc(Pθφ; eltype=Float64)
    fill!(fθφ, 0)
    # Fill deterministic content
    for (iθ, iφ) in zip(eachindex(axes(fθφ,1)), eachindex(axes(fθφ,2)))
        fθφ[iθ, iφ] = sin(0.3 * (iθ + rank + 1)) + cos(0.2 * (iφ + 2))
    end
    # Baseline: dist_analysis (alloc heavy) vs plan-based dist_analysis!
    a_alloc = @allocated begin
        Alm = SHTnsKit.dist_analysis(cfg, fθφ)
        nothing
    end
    Alm_out = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    aplan = SHTnsKit.DistAnalysisPlan(cfg, fθφ)
    a_alloc_plan = @allocated SHTnsKit.dist_analysis!(aplan, Alm_out, fθφ)
    # Reduce max allocation across ranks
    a_alloc_max = MPI.Allreduce(a_alloc, MPI.MAX, comm)
    a_alloc_plan_max = MPI.Allreduce(a_alloc_plan, MPI.MAX, comm)
    if rank == 0
        println("[dist scalar] analysis alloc bytes: baseline=$(a_alloc_max), plan=$(a_alloc_plan_max)")
    end
    # Vector: plan vs baseline
    Vtθφ = paalloc(Pθφ; eltype=Float64); fill!(Vtθφ, 0)
    Vpθφ = paalloc(Pθφ; eltype=Float64); fill!(Vpθφ, 0)
    for (iθ, iφ) in zip(eachindex(axes(Vtθφ,1)), eachindex(axes(Vtθφ,2)))
        Vtθφ[iθ, iφ] = 0.1*(iθ+1) + 0.05*(iφ+1)
        Vpθφ[iθ, iφ] = 0.2*sin(0.1*(iθ+rank+1))
    end
    v_alloc = @allocated begin
        Slm, Tlm = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ)
        nothing
    end
    Slm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Tlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    vplan = SHTnsKit.DistSphtorPlan(cfg, Vtθφ)
    v_alloc_plan = @allocated SHTnsKit.dist_analysis_sphtor!(vplan, Slm, Tlm, Vtθφ, Vpθφ)
    v_alloc_max = MPI.Allreduce(v_alloc, MPI.MAX, comm)
    v_alloc_plan_max = MPI.Allreduce(v_alloc_plan, MPI.MAX, comm)
    if rank == 0
        println("[dist vector] analysis alloc bytes: baseline=$(v_alloc_max), plan=$(v_alloc_plan_max)")
    end
    MPI.Finalize()
    return nothing
end

function main()
    lmax = parse(Int, get(ARGS, 1, "16"))
    println("=== Serial allocations (lmax=$(lmax)) ===")
    r = bench_serial(lmax)
    println(r)
    println("\n=== Distributed allocations (if MPI available) ===")
    bench_distributed(lmax)
end

main()
