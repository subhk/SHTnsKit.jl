# SHTnsKit.jl - Per-(m, ℓ)-mode transform tests
# Exercises `analysis_packed_ml` and `synthesis_packed_ml`, which operate on one
# azimuthal mode at a time. Verifies roundtrip and consistency with full transform.

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Per-mode scalar transforms" begin
    @testset "analysis_packed_ml ∘ synthesis_packed_ml roundtrip" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(900)

        for im in 0:cfg.mmax
            ltr = lmax
            Ql = randn(rng, ComplexF64, ltr - im + 1)
            Vr_m = synthesis_packed_ml(cfg, im, Ql, ltr)
            @test length(Vr_m) == cfg.nlat

            Ql_back = analysis_packed_ml(cfg, im, Vr_m, ltr)
            @test length(Ql_back) == ltr - im + 1
            @test isapprox(Ql_back, Ql; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "synthesis_packed_ml: truncation respected" begin
        lmax = 8
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(901)

        im = 2
        ltr = lmax - 3
        Ql = randn(rng, ComplexF64, ltr - im + 1)
        Vr_m = synthesis_packed_ml(cfg, im, Ql, ltr)

        # Synthesising with the full ltr=lmax and zero-padding high modes should agree
        Ql_full = zeros(ComplexF64, lmax - im + 1)
        Ql_full[1:length(Ql)] = Ql
        Vr_m_full = synthesis_packed_ml(cfg, im, Ql_full, lmax)
        @test isapprox(Vr_m, Vr_m_full; rtol=1e-12, atol=1e-14)
    end

    @testset "Dimension / range checks" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        im = 1; ltr = 3

        # Wrong Vr_m length
        @test_throws DimensionMismatch analysis_packed_ml(cfg, im,
            zeros(ComplexF64, cfg.nlat + 1), ltr)

        # im out of range
        @test_throws ArgumentError analysis_packed_ml(cfg, -1,
            zeros(ComplexF64, cfg.nlat), ltr)
        @test_throws ArgumentError analysis_packed_ml(cfg, cfg.mmax + 1,
            zeros(ComplexF64, cfg.nlat), ltr)

        # ltr out of range
        @test_throws ArgumentError analysis_packed_ml(cfg, im,
            zeros(ComplexF64, cfg.nlat), cfg.lmax + 1)
        @test_throws ArgumentError analysis_packed_ml(cfg, 2,
            zeros(ComplexF64, cfg.nlat), 1)  # ltr < im

        # synthesis: Ql length
        @test_throws DimensionMismatch synthesis_packed_ml(cfg, im,
            zeros(ComplexF64, ltr - im), ltr)
    end
end
