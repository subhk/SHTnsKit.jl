using MPI; MPI.Init()
using SHTnsKit, PencilArrays, PencilFFTs, Test

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)

@testset "DistTransposePlan construction" begin
    lmax = 32; nlat = lmax + 2; nlon = 2 * lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon = nlon)

    plan = DistTransposePlan(cfg; comm = comm, nlev = 1, use_rfft = true)

    @test plan.nlat == nlat
    @test plan.nlon == nlon
    @test plan.lmax == lmax
    @test plan.mmax == lmax

    # Each rank must own at least one m and all must be in range
    @test !isempty(plan.m_local)
    @test all(0 .<= plan.m_local .<= cfg.mmax)

    # m ownership partitions 0:mmax exactly once across all ranks
    owned = MPI.Allreduce(length(plan.m_local), +, comm)
    @test owned == cfg.mmax + 1

    # Legendre tables: one (lmax+1, nlat) matrix per local m
    @test length(plan.NP) == length(plan.m_local)
    @test size(plan.NP[1]) == (lmax + 1, nlat)

    # allocate_spatial returns a real PencilArray
    fsp = allocate_spatial(plan)
    @test fsp isa PencilArray

    # allocate_spectral returns a complex PencilArray
    Alm = allocate_spectral(plan)
    @test Alm isa PencilArray

    rank == 0 && println("DistTransposePlan construction OK on $(MPI.Comm_size(comm)) rank(s)")
end

MPI.Finalize()
