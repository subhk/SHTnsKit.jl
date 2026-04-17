# SHTnsKit.jl - Batch transforms: fft_batch scratch buffer coverage
# Exercises the fft_batch keyword argument added to synthesis_batch! et al.

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Batch transforms: fft_batch scratch" begin
    @testset "synthesis_batch! with caller-provided fft_batch" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(500)
        nfields = 3

        alm_batch = randn(rng, ComplexF64, lmax + 1, lmax + 1, nfields)
        for k in 1:nfields
            alm_batch[:, 1, k] .= real.(alm_batch[:, 1, k])
            for m in 0:lmax, l in 0:(m - 1)
                alm_batch[l + 1, m + 1, k] = 0
            end
        end

        f_ref = Array{Float64, 3}(undef, cfg.nlat, cfg.nlon, nfields)
        synthesis_batch!(cfg, f_ref, alm_batch; real_output=true)

        # With caller scratch
        scratch = Array{ComplexF64, 3}(undef, cfg.nlat, cfg.nlon, nfields)
        f_out = Array{Float64, 3}(undef, cfg.nlat, cfg.nlon, nfields)
        synthesis_batch!(cfg, f_out, alm_batch; real_output=true, fft_batch=scratch)
        @test f_out ≈ f_ref
    end

    @testset "synthesis_batch! fft_batch dimension mismatch" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        nfields = 2

        alm_batch = zeros(ComplexF64, lmax + 1, lmax + 1, nfields)
        f_out = zeros(cfg.nlat, cfg.nlon, nfields)
        wrong = zeros(ComplexF64, cfg.nlat, cfg.nlon, nfields + 1)
        @test_throws DimensionMismatch synthesis_batch!(cfg, f_out, alm_batch;
                                                        fft_batch=wrong)

        wrong2 = zeros(ComplexF64, cfg.nlat + 1, cfg.nlon, nfields)
        @test_throws DimensionMismatch synthesis_batch!(cfg, f_out, alm_batch;
                                                        fft_batch=wrong2)
    end

    @testset "Repeated synthesis with same scratch stays correct" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        nfields = 2
        rng = MersenneTwister(501)

        alm_a = randn(rng, ComplexF64, lmax + 1, lmax + 1, nfields)
        alm_b = randn(rng, ComplexF64, lmax + 1, lmax + 1, nfields)
        for arr in (alm_a, alm_b), k in 1:nfields
            arr[:, 1, k] .= real.(arr[:, 1, k])
            for m in 0:lmax, l in 0:(m - 1)
                arr[l + 1, m + 1, k] = 0
            end
        end

        scratch = Array{ComplexF64, 3}(undef, cfg.nlat, cfg.nlon, nfields)
        fa1 = zeros(cfg.nlat, cfg.nlon, nfields)
        fa2 = zeros(cfg.nlat, cfg.nlon, nfields)
        fb  = zeros(cfg.nlat, cfg.nlon, nfields)

        synthesis_batch!(cfg, fa1, alm_a; fft_batch=scratch)
        synthesis_batch!(cfg, fb,  alm_b; fft_batch=scratch)
        synthesis_batch!(cfg, fa2, alm_a; fft_batch=scratch)
        @test fa2 == fa1
    end
end
