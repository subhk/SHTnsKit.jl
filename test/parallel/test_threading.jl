# SHTnsKit.jl - Threading / parallel CPU tests (no MPI).
# Verifies correctness under Julia's native multithreading:
#  - Batch transforms give the same result with 1 vs N threads in user-level loops
#  - Per-thread SHTPlan pattern (recommended usage) is race-free
#  - analysis_batch / synthesis_batch are thread-consistent
#
# Run in a Julia session launched with e.g. `julia -t 4 --project`.

using Test
using Random
using Base.Threads
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

function _rand_real_alm(rng, lmax, mmax)
    alm = randn(rng, ComplexF64, lmax + 1, mmax + 1)
    alm[:, 1] .= real.(alm[:, 1])
    for m in 0:mmax, l in 0:(m - 1)
        alm[l + 1, m + 1] = 0
    end
    return alm
end

@testset "Threading / parallel CPU" begin
    nt = Threads.nthreads()
    VERBOSE && @info "Threads available" nt

    @testset "Per-thread SHTPlan: concurrent synthesis" begin
        # Canonical multi-thread pattern: one plan per thread.
        lmax = 8
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plans = [SHTPlan(cfg) for _ in 1:Threads.maxthreadid()]

        ntasks = 16
        rngs = [MersenneTwister(1000 + k) for k in 1:ntasks]
        alms = [_rand_real_alm(rngs[k], lmax, lmax) for k in 1:ntasks]
        outs = [zeros(cfg.nlat, cfg.nlon) for _ in 1:ntasks]

        @threads :static for k in 1:ntasks
            plan = plans[Threads.threadid()]
            synthesis!(plan, outs[k], alms[k])
        end

        # Each output must match the serial synthesis of its own coefficients
        for k in 1:ntasks
            ref = synthesis(cfg, alms[k]; real_output=true)
            @test isapprox(outs[k], ref; rtol=1e-12, atol=1e-14)
        end
    end

    @testset "Per-thread SHTPlan: concurrent analysis" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plans = [SHTPlan(cfg) for _ in 1:Threads.maxthreadid()]

        ntasks = 12
        rng = MersenneTwister(1100)
        fields = [randn(rng, cfg.nlat, cfg.nlon) for _ in 1:ntasks]
        outs = [zeros(ComplexF64, lmax + 1, lmax + 1) for _ in 1:ntasks]

        @threads :static for k in 1:ntasks
            plan = plans[Threads.threadid()]
            analysis!(plan, outs[k], fields[k])
        end

        for k in 1:ntasks
            ref = analysis(cfg, fields[k])
            @test isapprox(outs[k], ref; rtol=1e-12, atol=1e-14)
        end
    end

    @testset "Batch analysis: result independent of thread count" begin
        # analysis_batch uses @threads :static internally over m; the result must
        # match a reference sequential run regardless of Threads.nthreads().
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(1200)
        nfields = 8
        fields = randn(rng, cfg.nlat, cfg.nlon, nfields)

        alm_batch = analysis_batch(cfg, fields)
        # Single-field reference
        for k in 1:nfields
            ref = analysis(cfg, fields[:, :, k])
            @test isapprox(alm_batch[:, :, k], ref; rtol=1e-11, atol=1e-13)
        end
    end

    @testset "Batch synthesis: result independent of thread count" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(1201)
        nfields = 8
        alm_batch = randn(rng, ComplexF64, lmax + 1, lmax + 1, nfields)
        for k in 1:nfields
            alm_batch[:, 1, k] .= real.(alm_batch[:, 1, k])
            for m in 0:lmax, l in 0:(m - 1)
                alm_batch[l + 1, m + 1, k] = 0
            end
        end

        fields = synthesis_batch(cfg, alm_batch; real_output=true)
        for k in 1:nfields
            ref = synthesis(cfg, alm_batch[:, :, k]; real_output=true)
            @test isapprox(fields[:, :, k], ref; rtol=1e-11, atol=1e-13)
        end
    end

    @testset "Determinism: repeated multi-threaded runs agree bitwise-close" begin
        lmax = 7
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(1300)
        nfields = 6
        fields = randn(rng, cfg.nlat, cfg.nlon, nfields)

        a1 = analysis_batch(cfg, fields)
        a2 = analysis_batch(cfg, fields)
        a3 = analysis_batch(cfg, fields)
        @test a1 == a2
        @test a1 == a3
    end

    @testset "Concurrent unrelated configs don't interfere" begin
        # Two different configs used in parallel tasks. Each task must produce the
        # same result as its serial counterpart.
        lmax_a, lmax_b = 5, 7
        cfg_a = create_gauss_config(lmax_a, lmax_a + 2; nlon=2*lmax_a + 1)
        cfg_b = create_gauss_config(lmax_b, lmax_b + 2; nlon=2*lmax_b + 1)

        rng = MersenneTwister(1400)
        f_a = randn(rng, cfg_a.nlat, cfg_a.nlon)
        f_b = randn(rng, cfg_b.nlat, cfg_b.nlon)

        ref_a = analysis(cfg_a, f_a)
        ref_b = analysis(cfg_b, f_b)

        results = Vector{Any}(undef, 2)
        @sync begin
            Threads.@spawn results[1] = analysis(cfg_a, f_a)
            Threads.@spawn results[2] = analysis(cfg_b, f_b)
        end

        @test isapprox(results[1], ref_a; rtol=1e-12, atol=1e-14)
        @test isapprox(results[2], ref_b; rtol=1e-12, atol=1e-14)
    end

    @testset "Per-thread sphtor plan: concurrent vector synthesis" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plans = [SHTPlan(cfg) for _ in 1:Threads.maxthreadid()]

        ntasks = 8
        rng = MersenneTwister(1500)
        Ss = [_rand_real_alm(rng, lmax, lmax) for _ in 1:ntasks]
        Ts = [_rand_real_alm(rng, lmax, lmax) for _ in 1:ntasks]
        for k in 1:ntasks
            Ss[k][1, 1] = 0; Ts[k][1, 1] = 0
        end
        Vts = [zeros(cfg.nlat, cfg.nlon) for _ in 1:ntasks]
        Vps = [zeros(cfg.nlat, cfg.nlon) for _ in 1:ntasks]

        @threads :static for k in 1:ntasks
            plan = plans[Threads.threadid()]
            synthesis_sphtor!(plan, Vts[k], Vps[k], Ss[k], Ts[k]; real_output=true)
        end

        for k in 1:ntasks
            Vt_ref, Vp_ref = synthesis_sphtor(cfg, Ss[k], Ts[k]; real_output=true)
            @test isapprox(Vts[k], Vt_ref; rtol=1e-11, atol=1e-13)
            @test isapprox(Vps[k], Vp_ref; rtol=1e-11, atol=1e-13)
        end
    end
end
