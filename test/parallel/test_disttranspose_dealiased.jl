# SHTnsKit.jl - DistTransposePlan dealiased-grid round-trip tests
#
# Dealiased φ-grids use nlon > 2*mmax+1 (e.g. the 3/2 rule, nlon ≈ 3*mmax).
# These are standard for nonlinear spectral codes. This test exercises the
# transpose-based distributed transform on such a grid (nlon=24, mmax=8 →
# 2*mmax+1 = 17 < 24). It must build the plan and round-trip scalar + vector
# fields to machine precision at 1 and 2 ranks.
#
# Run with:
#   mpiexec -n 1 julia --project test/parallel/test_disttranspose_dealiased.jl
#   mpiexec -n 2 julia --project test/parallel/test_disttranspose_dealiased.jl

using MPI; MPI.Init()
using SHTnsKit, PencilArrays, PencilFFTs, Test

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)

@testset "DistTransposePlan dealiased construction (nlon>2*mmax+1)" begin
    lmax = 8; nlat = 12; nlon = 24    # dealiased: 2*mmax+1 = 17 < 24
    cfg = create_gauss_config(lmax, nlat; nlon = nlon)
    @test cfg.nlon > 2 * cfg.mmax + 1   # confirm dealiased

    plan = DistTransposePlan(cfg; comm = comm, nlev = 2, use_rfft = true, with_vector = true)
    @test plan.nlat == nlat
    @test plan.nlon == nlon
    @test plan.mmax == lmax

    # m ownership must still partition 0:mmax exactly once across all ranks
    @test !isempty(plan.m_local)
    @test all(0 .<= plan.m_local .<= cfg.mmax)
    owned = MPI.Allreduce(length(plan.m_local), +, comm)
    @test owned == cfg.mmax + 1

    fsp = allocate_spatial(plan)
    @test fsp isa PencilArray
    Alm = allocate_spectral(plan)
    @test Alm isa PencilArray
    # Alm's local m-columns must align with plan.m_local (the invariant the
    # dist_* kernels rely on: they index Alm[:, mi, :] for mi=1..length(m_local)).
    @test size(parent(Alm), 2) >= length(plan.m_local)

    rank == 0 && println("dealiased construction OK on $(MPI.Comm_size(comm)) rank(s)")
end

@testset "transpose dist_synthesis! round-trip (scalar, dealiased)" begin
    lmax = 8; nlat = 12; nlon = 24; nlev = 2
    cfg = create_gauss_config(lmax, nlat; nlon = nlon)

    # Distinct band-limited field per level
    a0 = [zeros(ComplexF64, lmax + 1, lmax + 1) for _ in 1:nlev]
    ffull = Vector{Matrix{Float64}}(undef, nlev)
    for lev in 1:nlev
        for m in 0:lmax, l in m:lmax
            sc = (lev + 0.3) / (1 + l)^2
            a0[lev][l+1, m+1] = m == 0 ? complex(sc) : complex(sc, 0.5sc)
        end
        ffull[lev] = SHTnsKit.synthesis(cfg, a0[lev]; real_output = true)  # (nlat, nlon)
    end

    plan = DistTransposePlan(cfg; comm = comm, nlev = nlev, use_rfft = true)
    f = allocate_spatial(plan)
    r = PencilArrays.range_local(pencil(f))   # r[1]=φ range, r[2]=θ range
    for lev in 1:nlev, (il, ig) in enumerate(r[2]), (jl, jg) in enumerate(r[1])
        parent(f)[jl, il, lev] = ffull[lev][ig, jg]
    end

    # Forward analysis then check vs serial reference per level
    Alm = allocate_spectral(plan)
    dist_analysis!(plan, Alm, f)
    aerr = 0.0
    for lev in 1:nlev
        a_ref = SHTnsKit.analysis(cfg, ffull[lev])
        for (mi, m) in enumerate(plan.m_local), l in m:lmax
            aerr = max(aerr, abs(parent(Alm)[l+1, mi, lev] - a_ref[l+1, m+1]))
        end
    end
    gaerr = MPI.Allreduce(aerr, MPI.MAX, comm)
    rank == 0 && println("dealiased scalar analysis gerr=$gaerr")
    @test gaerr < 1e-10

    # Reverse synthesis — round-trip
    f2 = allocate_spatial(plan)
    dist_synthesis!(plan, f2, Alm)
    lerr = maximum(abs.(parent(f2) .- parent(f)))
    gerr = MPI.Allreduce(lerr, MPI.MAX, comm)
    rank == 0 && println("dealiased scalar synthesis round-trip gerr=$gerr")
    @test gerr < 1e-10
end

@testset "transpose dist sphtor round-trip (vector, dealiased)" begin
    lmax = 8; nlat = 12; nlon = 24
    cfg = create_gauss_config(lmax, nlat; nlon = nlon)

    S0 = zeros(ComplexF64, lmax + 1, lmax + 1); T0 = copy(S0)
    for m in 0:lmax, l in max(1, m):lmax
        sc = 1 / (1 + l)^2
        S0[l+1, m+1] = complex(sc, m == 0 ? 0.0 : 0.5sc)
        T0[l+1, m+1] = complex(0.7sc, m == 0 ? 0.0 : -0.3sc)
    end
    Vt_full, Vp_full = SHTnsKit.synthesis_sphtor(cfg, S0, T0; real_output = true)
    S_ref, T_ref = SHTnsKit.analysis_sphtor(cfg, Vt_full, Vp_full)

    plan = DistTransposePlan(cfg; comm = comm, nlev = 1, use_rfft = true, with_vector = true)
    Vt = allocate_spatial(plan); Vp = allocate_spatial(plan)
    r = PencilArrays.range_local(pencil(Vt))
    for (il, ig) in enumerate(r[2]), (jl, jg) in enumerate(r[1])
        parent(Vt)[jl, il, 1] = Vt_full[ig, jg]
        parent(Vp)[jl, il, 1] = Vp_full[ig, jg]
    end

    Slm = allocate_spectral(plan); Tlm = allocate_spectral(plan)
    dist_analysis_sphtor!(plan, Slm, Tlm, Vt, Vp)
    aerr = 0.0
    for (mi, m) in enumerate(plan.m_local), l in max(1, m):lmax
        aerr = max(aerr,
            abs(parent(Slm)[l+1, mi, 1] - S_ref[l+1, m+1]),
            abs(parent(Tlm)[l+1, mi, 1] - T_ref[l+1, m+1]))
    end
    gaerr = MPI.Allreduce(aerr, MPI.MAX, comm)
    rank == 0 && println("dealiased sphtor analysis gerr=$gaerr")
    @test gaerr < 1e-10

    Vt2 = allocate_spatial(plan); Vp2 = allocate_spatial(plan)
    dist_synthesis_sphtor!(plan, Vt2, Vp2, Slm, Tlm)
    rt = MPI.Allreduce(
        max(maximum(abs.(parent(Vt2) .- parent(Vt))),
            maximum(abs.(parent(Vp2) .- parent(Vp)))),
        MPI.MAX, comm)
    rank == 0 && println("dealiased sphtor round-trip gerr=$rt")
    @test rt < 1e-10
end

MPI.Finalize()
