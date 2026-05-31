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

@testset "transpose dist_analysis! vs serial (scalar)" begin
    for lmax in (32, 64)
        nlat = lmax + 2; nlon = 2 * lmax + 1
        cfg  = create_gauss_config(lmax, nlat; nlon = nlon)

        # Build reference: random spectral coefficients → serial synthesis → serial analysis
        a0 = zeros(ComplexF64, lmax + 1, lmax + 1)
        for m in 0:lmax, l in m:lmax
            sc = 1.0 / (1 + l)^2
            a0[l+1, m+1] = m == 0 ? complex(sc) : complex(sc, 0.5sc)
        end
        f_full = SHTnsKit.synthesis(cfg, a0; real_output = true)  # (nlat, nlon)
        a_ref  = SHTnsKit.analysis(cfg, f_full)                   # (lmax+1, mmax+1) reference

        # Build transpose plan and fill spatial PencilArray from f_full
        plan = DistTransposePlan(cfg; comm = comm, nlev = 1, use_rfft = true)
        f    = allocate_spatial(plan)   # real PencilArray, global (nlon, nlat), θ-distributed

        # range_local gives global index ranges for each logical dim:
        #   r[1] = φ range (1-based global column indices, typically 1:nlon on every rank)
        #   r[2] = θ range (1-based global row indices for this rank's slice)
        r = PencilArrays.range_local(pencil(f))
        # parent(f) layout: (n_phi_local, n_theta_local, nlev)
        for (il, ig) in enumerate(r[2]), (jl, jg) in enumerate(r[1])
            # f_full[ig, jg] = f_full[θ_global, φ_global]
            parent(f)[jl, il, 1] = f_full[ig, jg]
        end

        Alm = allocate_spectral(plan)
        dist_analysis!(plan, Alm, f)

        # Compare each rank's owned m-columns against the dense serial reference
        err = 0.0
        for (mi, m) in enumerate(plan.m_local), l in m:lmax
            err = max(err, abs(parent(Alm)[l+1, mi, 1] - a_ref[l+1, m+1]))
        end
        gerr = MPI.Allreduce(err, MPI.MAX, comm)
        rank == 0 && println("lmax=$lmax scalar analysis gerr=$gerr")
        @test gerr < 1e-8
    end
end

@testset "transpose dist_synthesis! round-trip (scalar)" begin
    for lmax in (32, 64)
        nlat = lmax + 2; nlon = 2 * lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon = nlon)

        # Build band-limited field from known spectral coefficients
        a0 = zeros(ComplexF64, lmax + 1, lmax + 1)
        for m in 0:lmax, l in m:lmax
            sc = 1.0 / (1 + l)^2
            a0[l+1, m+1] = m == 0 ? complex(sc) : complex(sc, 0.5sc)
        end
        f_full = SHTnsKit.synthesis(cfg, a0; real_output = true)  # (nlat, nlon) reference

        plan = DistTransposePlan(cfg; comm = comm, nlev = 1, use_rfft = true)

        # Fill distributed spatial field from serial reference
        f = allocate_spatial(plan)
        r = PencilArrays.range_local(pencil(f))   # r[1]=φ range, r[2]=θ range
        for (il, ig) in enumerate(r[2]), (jl, jg) in enumerate(r[1])
            parent(f)[jl, il, 1] = f_full[ig, jg]
        end

        # Forward: analysis
        Alm = allocate_spectral(plan)
        dist_analysis!(plan, Alm, f)

        # Reverse: synthesis — should recover f up to machine precision
        f2 = allocate_spatial(plan)
        dist_synthesis!(plan, f2, Alm)

        lerr = maximum(abs.(parent(f2) .- parent(f)))
        gerr = MPI.Allreduce(lerr, MPI.MAX, comm)
        rank == 0 && println("lmax=$lmax synthesis round-trip gerr=$gerr")
        @test gerr < 1e-8
    end
end

MPI.Finalize()
