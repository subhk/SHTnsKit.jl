# SHTnsKit.jl - Basic Scalar Transform Tests
# Tests for analysis, synthesis, and SHTPlan

using Test
using Random
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "Basic Scalar Transforms" begin
    @testset "Analysis-synthesis roundtrip" begin
        lmax = 10
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(42)

        # Create random spectral coefficients
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])  # m=0 must be real for real fields
        # Zero invalid entries (l < m)
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0.0
        end

        # Roundtrip: spectral -> spatial -> spectral
        f = synthesis(cfg, alm; real_output=true)
        alm_recovered = analysis(cfg, f)

        @test isapprox(alm_recovered, alm; rtol=1e-10, atol=1e-12)
        VERBOSE && @info "Scalar roundtrip" max_err=maximum(abs.(alm_recovered - alm))
    end

    @testset "SHTPlan optimized transforms" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(43)

        # Create plan
        plan = SHTPlan(cfg)

        # Test with random field
        f = randn(rng, nlat, nlon)
        alm = zeros(ComplexF64, lmax+1, lmax+1)

        # In-place analysis
        analysis!(plan, alm, f)

        # Compare with non-planned version
        alm_ref = analysis(cfg, f)
        @test isapprox(alm, alm_ref; rtol=1e-10, atol=1e-12)

        # In-place synthesis
        f_back = zeros(nlat, nlon)
        synthesis!(plan, f_back, alm)

        @test isapprox(f_back, f; rtol=1e-10, atol=1e-12)
    end

    @testset "Packed vector format" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(44)

        # Random spatial field
        f = randn(rng, nlat, nlon)

        # Packed analysis
        Qlm = spat_to_SH(cfg, vec(f))
        @test length(Qlm) == cfg.nlm

        # Packed synthesis
        f_back = SH_to_spat(cfg, Qlm)
        @test length(f_back) == nlat * nlon

        # Verify roundtrip
        @test isapprox(f_back, vec(f); rtol=1e-10, atol=1e-12)
    end

    @testset "Single spherical harmonic modes" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Test individual modes
        for l in 0:lmax
            for m in 0:l
                alm = zeros(ComplexF64, lmax+1, lmax+1)
                alm[l+1, m+1] = m == 0 ? 1.0 : 1.0 + 0.5im

                f = synthesis(cfg, alm; real_output=true)
                alm_rec = analysis(cfg, f)

                @test isapprox(alm_rec[l+1, m+1], alm[l+1, m+1]; rtol=1e-10, atol=1e-12)
            end
        end
    end

    @testset "Real field consistency" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(45)

        # Start with real spatial field
        f_real = randn(rng, nlat, nlon)

        # Analyze
        alm = analysis(cfg, f_real)

        # m=0 coefficients should be real
        @test all(abs.(imag.(alm[:, 1])) .< 1e-12)

        # Synthesize back
        f_back = synthesis(cfg, alm; real_output=true)

        # Should remain real
        @test isapprox(f_back, f_real; rtol=1e-10, atol=1e-12)
    end
end
