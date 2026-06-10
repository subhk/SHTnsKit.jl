# DistAnalysisPlan in-place analysis: correctness + per-call allocation budget.
# Run with:
#   mpiexec -n 1 julia --project test/parallel/test_dist_plan_alloc.jl
#   mpiexec -n 2 julia --project test/parallel/test_dist_plan_alloc.jl
using MPI; MPI.Init()
using SHTnsKit, PencilArrays, PencilFFTs, Test
using Random

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const ParExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)

function band_limited_field(cfg, pen)
    lmax, mmax = cfg.lmax, cfg.mmax
    rng = MersenneTwister(7)
    alm = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        alm[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    alm[:, 1] .= real.(alm[:, 1])
    MPI.Bcast!(alm, 0, comm)
    f_full = SHTnsKit.synthesis(cfg, alm; real_output=true)
    ranges = PencilArrays.range_local(pen)
    flocal = zeros(Float64, PencilArrays.size_local(pen)...)
    for (i_local, i_global) in enumerate(ranges[1])
        for (j_local, j_global) in enumerate(ranges[2])
            flocal[i_local, j_local] = f_full[i_global, j_global]
        end
    end
    return PencilArray(pen, flocal), f_full
end

@testset "DistAnalysisPlan θ-decomposed (scaling layout)" begin
    lmax = 24; nlat = 32; nlon = 72
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    pen = Pencil((nlat, nlon), (1,), comm)   # decompose θ — the scaling layout
    f_pa, f_full = band_limited_field(cfg, pen)

    a_ref = SHTnsKit.analysis(cfg, f_full)

    for use_rfft in (false, true)
        plan = ParExt.DistAnalysisPlan(cfg, f_pa; use_rfft)
        Alm_out = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
        SHTnsKit.dist_analysis!(plan, Alm_out, f_pa)
        @test maximum(abs.(Alm_out .- a_ref)) < 1e-10

        # match the cfg-form distributed result too
        Alm_direct = SHTnsKit.dist_analysis(cfg, f_pa; use_rfft)
        @test maximum(abs.(Alm_out .- Alm_direct)) < 1e-12

        # Allocation budget: plan owns FFT buffer, Legendre scratch, work
        # matrix, index/weight caches, and the cached reduction subcomm —
        # per-call allocations must stay far below the ~30 KB the unplanned
        # path burns (Fθm + Alm zeros + collects + Comm_split).
        SHTnsKit.dist_analysis!(plan, Alm_out, f_pa)  # warmup
        a = @allocated SHTnsKit.dist_analysis!(plan, Alm_out, f_pa)
        rank == 0 && println("dist_analysis! (use_rfft=$use_rfft): $a B/call")
        @test a < 8192
    end
end

@testset "DistAnalysisPlan φ-decomposed (fallback layout)" begin
    lmax = 16; nlat = 20; nlon = 48
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    pen = Pencil((nlat, nlon), comm)   # default: decompose last dim (φ)
    f_pa, f_full = band_limited_field(cfg, pen)

    a_ref = SHTnsKit.analysis(cfg, f_full)
    plan = ParExt.DistAnalysisPlan(cfg, f_pa)
    Alm_out = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
    SHTnsKit.dist_analysis!(plan, Alm_out, f_pa)
    @test maximum(abs.(Alm_out .- a_ref)) < 1e-10
end

rank == 0 && println("test_dist_plan_alloc: all testsets done on $(MPI.Comm_size(comm)) rank(s)")
