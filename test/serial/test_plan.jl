# SHTnsKit.jl - SHTPlan tests
# Exercises src/plan.jl: planned scalar/vector transforms, validation,
# normalization conversion paths, and robert_form handling.

using Test
using Random
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

@testset "SHTPlan" begin
    @testset "Constructor: use_rfft=true throws" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        @test_throws ArgumentError SHTPlan(cfg; use_rfft=true)
    end

    @testset "Planned scalar matches non-planned" begin
        lmax = 8
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plan = SHTPlan(cfg)
        rng = MersenneTwister(201)

        alm = _rand_real_alm(rng, lmax, lmax)
        f = synthesis(cfg, alm; real_output=true)
        alm_ref = analysis(cfg, f)

        f_out = zeros(cfg.nlat, cfg.nlon)
        synthesis!(plan, f_out, alm)
        @test isapprox(f_out, f; rtol=1e-12, atol=1e-14)

        alm_out = zeros(ComplexF64, lmax + 1, lmax + 1)
        analysis!(plan, alm_out, f_out)
        @test isapprox(alm_out, alm_ref; rtol=1e-12, atol=1e-14)
        @test isapprox(alm_out, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "Planned scalar: reuse across calls (no state leakage)" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plan = SHTPlan(cfg)
        rng = MersenneTwister(202)

        alm_a = _rand_real_alm(rng, lmax, lmax)
        alm_b = _rand_real_alm(rng, lmax, lmax)

        f_a = zeros(cfg.nlat, cfg.nlon); f_b = zeros(cfg.nlat, cfg.nlon)
        synthesis!(plan, f_a, alm_a)
        synthesis!(plan, f_b, alm_b)
        # Running B must not pollute A's previous result
        @test isapprox(f_a, synthesis(cfg, alm_a; real_output=true); rtol=1e-12, atol=1e-14)
        @test isapprox(f_b, synthesis(cfg, alm_b; real_output=true); rtol=1e-12, atol=1e-14)

        # Run A again → identical to first call
        f_a2 = zeros(cfg.nlat, cfg.nlon)
        synthesis!(plan, f_a2, alm_a)
        @test f_a2 == f_a
    end

    @testset "Planned scalar: dimension mismatch throws" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        plan = SHTPlan(cfg)
        alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        f = zeros(cfg.nlat, cfg.nlon)

        @test_throws DimensionMismatch synthesis!(plan, zeros(1, cfg.nlon), alm)
        @test_throws DimensionMismatch synthesis!(plan, zeros(cfg.nlat, 1), alm)
        @test_throws DimensionMismatch synthesis!(plan, f, zeros(ComplexF64, 1, cfg.mmax + 1))
        @test_throws DimensionMismatch synthesis!(plan, f, zeros(ComplexF64, cfg.lmax + 1, 1))

        @test_throws DimensionMismatch analysis!(plan, alm, zeros(1, cfg.nlon))
        @test_throws DimensionMismatch analysis!(plan, alm, zeros(cfg.nlat, 1))
        @test_throws DimensionMismatch analysis!(plan, zeros(ComplexF64, 1, cfg.mmax + 1), f)
    end

    @testset "Planned vector (sphtor) matches non-planned" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plan = SHTPlan(cfg)
        rng = MersenneTwister(203)

        Slm = _rand_real_alm(rng, lmax, lmax); Slm[1, 1] = 0
        Tlm = _rand_real_alm(rng, lmax, lmax); Tlm[1, 1] = 0

        Vt = zeros(cfg.nlat, cfg.nlon); Vp = zeros(cfg.nlat, cfg.nlon)
        synthesis_sphtor!(plan, Vt, Vp, Slm, Tlm; real_output=true)

        Vt_ref, Vp_ref = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        @test isapprox(Vt, Vt_ref; rtol=1e-11, atol=1e-13)
        @test isapprox(Vp, Vp_ref; rtol=1e-11, atol=1e-13)

        Slm_out = zeros(ComplexF64, lmax + 1, lmax + 1)
        Tlm_out = zeros(ComplexF64, lmax + 1, lmax + 1)
        analysis_sphtor!(plan, Slm_out, Tlm_out, Vt, Vp)

        Slm_ref, Tlm_ref = analysis_sphtor(cfg, Vt, Vp)
        @test isapprox(Slm_out, Slm_ref; rtol=1e-11, atol=1e-13)
        @test isapprox(Tlm_out, Tlm_ref; rtol=1e-11, atol=1e-13)
    end

    @testset "Planned vector dimension checks" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        plan = SHTPlan(cfg)
        S = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        T = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        Vt = zeros(cfg.nlat, cfg.nlon); Vp = zeros(cfg.nlat, cfg.nlon)

        @test_throws DimensionMismatch analysis_sphtor!(plan, S, T, zeros(1, cfg.nlon), Vp)
        @test_throws DimensionMismatch analysis_sphtor!(plan, S, T, Vt, zeros(cfg.nlat, 1))
        @test_throws DimensionMismatch analysis_sphtor!(plan, zeros(ComplexF64, 1, 1), T, Vt, Vp)
        @test_throws DimensionMismatch analysis_sphtor!(plan, S, zeros(ComplexF64, 1, 1), Vt, Vp)

        @test_throws DimensionMismatch synthesis_sphtor!(plan, zeros(1, cfg.nlon), Vp, S, T)
        @test_throws DimensionMismatch synthesis_sphtor!(plan, Vt, zeros(cfg.nlat, 1), S, T)
    end

    @testset "Planned scalar: non-orthonormal normalization roundtrip" begin
        # The planned path is not (yet) guaranteed to match the non-planned path
        # under non-orthonormal normalizations, but its own analysis∘synthesis
        # roundtrip must still recover the input coefficients.
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1,
                                  norm=:schmidt, cs_phase=false)
        plan = SHTPlan(cfg)
        rng = MersenneTwister(204)

        alm = _rand_real_alm(rng, lmax, lmax)
        f_plan = zeros(cfg.nlat, cfg.nlon)
        synthesis!(plan, f_plan, alm)
        @test all(isfinite, f_plan)

        alm_back = zeros(ComplexF64, lmax + 1, lmax + 1)
        analysis!(plan, alm_back, f_plan)
        @test isapprox(alm_back, alm; rtol=1e-9, atol=1e-11)
    end

    @testset "Planned sphtor: robert_form path" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1, robert_form=true)
        plan = SHTPlan(cfg)
        rng = MersenneTwister(205)

        Slm = _rand_real_alm(rng, lmax, lmax); Slm[1, 1] = 0
        Tlm = _rand_real_alm(rng, lmax, lmax); Tlm[1, 1] = 0

        Vt_ref, Vp_ref = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        Vt = zeros(cfg.nlat, cfg.nlon); Vp = zeros(cfg.nlat, cfg.nlon)
        synthesis_sphtor!(plan, Vt, Vp, Slm, Tlm; real_output=true)
        @test isapprox(Vt, Vt_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp, Vp_ref; rtol=1e-10, atol=1e-12)

        Slm_back = zeros(ComplexF64, lmax + 1, lmax + 1)
        Tlm_back = zeros(ComplexF64, lmax + 1, lmax + 1)
        analysis_sphtor!(plan, Slm_back, Tlm_back, Vt, Vp)
        Slm_ref, Tlm_ref = analysis_sphtor(cfg, Vt, Vp)
        @test isapprox(Slm_back, Slm_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Tlm_back, Tlm_ref; rtol=1e-10, atol=1e-12)
    end

    @testset "Planned scalar: complex output path (real_output=false) runs" begin
        # real_output=false skips Hermitian symmetry enforcement, producing a
        # genuinely complex spatial field — just sanity-check that the path is
        # exercised and produces finite output.
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plan = SHTPlan(cfg)
        rng = MersenneTwister(206)

        alm = _rand_real_alm(rng, lmax, lmax)
        f_cplx = zeros(ComplexF64, cfg.nlat, cfg.nlon)
        synthesis!(plan, f_cplx, alm; real_output=false)
        @test all(isfinite, f_cplx)
        @test eltype(f_cplx) <: Complex
    end
end
