# SHTnsKit.jl - Spectral Diagnostics Tests
# Tests for energy spectrum functions

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Spectral Diagnostics" begin
    lmax = 8
    nlat = lmax + 2
    nlon = 2 * lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(99)

    @testset "energy_scalar_l_spectrum dimensions" begin
        alm = zeros(ComplexF64, lmax+1, lmax+1)
        El = SHTnsKit.energy_scalar_l_spectrum(cfg, alm)
        @test length(El) == lmax + 1
        @test all(El .>= 0)
    end

    @testset "energy_scalar_l_spectrum single mode" begin
        alm = zeros(ComplexF64, lmax+1, lmax+1)
        alm[4, 1] = 1.0 + 0im  # l=3, m=0
        El = SHTnsKit.energy_scalar_l_spectrum(cfg, alm)

        # Only l=3 should have energy
        for l in 0:lmax
            if l == 3
                @test El[l+1] > 0
            else
                @test El[l+1] ≈ 0.0 atol=1e-14
            end
        end
    end

    @testset "energy_scalar_m_spectrum dimensions" begin
        alm = zeros(ComplexF64, lmax+1, lmax+1)
        Em = SHTnsKit.energy_scalar_m_spectrum(cfg, alm)
        @test length(Em) == lmax + 1
        @test all(Em .>= 0)
    end

    @testset "energy_scalar_m_spectrum single mode" begin
        alm = zeros(ComplexF64, lmax+1, lmax+1)
        alm[5, 3] = 1.0 + 1im  # l=4, m=2
        Em = SHTnsKit.energy_scalar_m_spectrum(cfg, alm)

        # Only m=2 should have energy
        for m in 0:lmax
            if m == 2
                @test Em[m+1] > 0
            else
                @test Em[m+1] ≈ 0.0 atol=1e-14
            end
        end
    end

    @testset "energy_vector_l_spectrum" begin
        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)

        # Put energy in l=3, m=1 toroidal mode
        Tlm[4, 2] = 1.0 + 0im
        El = SHTnsKit.energy_vector_l_spectrum(cfg, Slm, Tlm)

        @test length(El) == lmax + 1
        @test El[1] ≈ 0.0  # l=0 always zero for vector fields
        @test El[4] > 0    # l=3 should have energy
    end

    @testset "energy_vector_m_spectrum" begin
        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)

        Slm[3, 2] = 1.0 + 0im  # l=2, m=1
        Em = SHTnsKit.energy_vector_m_spectrum(cfg, Slm, Tlm)

        @test length(Em) == lmax + 1
        @test Em[2] > 0  # m=1 should have energy
    end

    @testset "energy_scalar_lm" begin
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 1:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0.0
        end

        Elm = SHTnsKit.energy_scalar_lm(cfg, alm)
        @test size(Elm) == (lmax+1, lmax+1)
        @test all(Elm .>= 0)

        # Sum over m should give l-spectrum
        El = SHTnsKit.energy_scalar_l_spectrum(cfg, alm)
        for l in 0:lmax
            @test isapprox(sum(Elm[l+1, :]), El[l+1]; rtol=1e-10)
        end
    end

    @testset "energy_vector_lm" begin
        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        for arr in [Slm, Tlm]
            arr[:, 1] .= real.(arr[:, 1])
            for m in 1:lmax, l in 0:(m-1)
                arr[l+1, m+1] = 0.0
            end
        end

        Elm = SHTnsKit.energy_vector_lm(cfg, Slm, Tlm)
        @test size(Elm) == (lmax+1, lmax+1)
        @test all(Elm .>= 0)

        # l=0 row should be zero
        @test all(Elm[1, :] .≈ 0.0)

        # Sum over m should give l-spectrum
        El = SHTnsKit.energy_vector_l_spectrum(cfg, Slm, Tlm)
        for l in 0:lmax
            @test isapprox(sum(Elm[l+1, :]), El[l+1]; rtol=1e-10)
        end
    end

    @testset "zero coefficients give zero energy" begin
        alm = zeros(ComplexF64, lmax+1, lmax+1)
        @test all(SHTnsKit.energy_scalar_l_spectrum(cfg, alm) .== 0)
        @test all(SHTnsKit.energy_scalar_m_spectrum(cfg, alm) .== 0)

        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)
        @test all(SHTnsKit.energy_vector_l_spectrum(cfg, Slm, Tlm) .== 0)
        @test all(SHTnsKit.energy_vector_m_spectrum(cfg, Slm, Tlm) .== 0)
    end

    @testset "energy spectrum total consistency" begin
        # Total energy from l-spectrum should equal total from m-spectrum
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 1:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0.0
        end

        El = SHTnsKit.energy_scalar_l_spectrum(cfg, alm)
        Em = SHTnsKit.energy_scalar_m_spectrum(cfg, alm)
        @test isapprox(sum(El), sum(Em); rtol=1e-10)
    end
end
