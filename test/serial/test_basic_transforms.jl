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

        # Start with spectral coefficients for reliable roundtrip
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])  # m=0 real
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # In-place synthesis
        f = zeros(nlat, nlon)
        synthesis!(plan, f, alm)

        # In-place analysis
        alm_back = zeros(ComplexF64, lmax+1, lmax+1)
        analysis!(plan, alm_back, f)

        # Compare recovered coefficients
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)

        # Also verify plan matches non-planned version
        alm_ref = analysis(cfg, f)
        @test isapprox(alm_back, alm_ref; rtol=1e-10, atol=1e-12)
    end

    @testset "Packed vector format" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(44)

        # Start with packed spectral coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])  # m=0 real (first lmax+1 indices)

        # Packed synthesis
        f = SH_to_spat(cfg, Qlm)
        @test length(f) == nlat * nlon

        # Packed analysis
        Qlm_back = spat_to_SH(cfg, f)
        @test length(Qlm_back) == cfg.nlm

        # Verify roundtrip
        @test isapprox(Qlm_back, Qlm; rtol=1e-10, atol=1e-12)
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

        # Start with spectral coefficients, real for m=0
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])  # m=0 real
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Synthesize to real field
        f = synthesis(cfg, alm; real_output=true)

        # Analyze back
        alm_back = analysis(cfg, f)

        # m=0 coefficients should remain real
        @test all(abs.(imag.(alm_back[:, 1])) .< 1e-12)

        # Roundtrip should preserve coefficients
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
    end
end
