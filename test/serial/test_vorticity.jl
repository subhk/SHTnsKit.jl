# SHTnsKit.jl - Vorticity and Enstrophy Tests
# Tests for vorticity computation and enstrophy diagnostics

using Test
using Random
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "Vorticity and Enstrophy" begin
    @testset "Spectral vorticity computation" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(110)

        # Toroidal coefficients only (pure rotational flow)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Tlm[1, :] .= 0

        # Spectral vorticity
        zeta_spec = vorticity_spectral(cfg, Tlm)
        @test size(zeta_spec) == size(Tlm)

        # Vorticity coefficients should be non-zero for non-zero Tlm
        @test any(zeta_spec .!= 0)
    end

    @testset "Grid vorticity computation" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(111)

        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Tlm[1, :] .= 0

        # Grid vorticity from spectral toroidal coefficients
        zeta_grid = vorticity_grid(cfg, Tlm)
        @test size(zeta_grid) == (nlat, nlon)
    end

    @testset "Spectral-grid vorticity consistency" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(112)

        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Tlm[1, :] .= 0

        # Spectral vorticity
        zeta_spec = vorticity_spectral(cfg, Tlm)

        # Grid vorticity from spectral Tlm
        zeta_grid = vorticity_grid(cfg, Tlm)

        # Synthesize spectral vorticity to grid and compare
        zeta_synth = synthesis(cfg, zeta_spec; real_output=true)
        @test isapprox(zeta_synth, zeta_grid; rtol=1e-8, atol=1e-10)
    end

    @testset "Enstrophy spectral computation" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(113)

        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Tlm[1, :] .= 0

        # Spectral enstrophy
        ens_spec = enstrophy(cfg, Tlm)
        @test ens_spec >= 0  # Enstrophy is non-negative
    end

    @testset "Enstrophy Parseval identity" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(114)

        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Tlm[1, :] .= 0

        # Spectral enstrophy
        ens_spec = enstrophy(cfg, Tlm)

        # Grid enstrophy: compute vorticity grid first, then integrate
        zeta_grid = vorticity_grid(cfg, Tlm)
        ens_grid = grid_enstrophy(cfg, zeta_grid)

        @test isapprox(ens_spec, ens_grid; rtol=1e-8, atol=1e-10)
    end

    @testset "Enstrophy l-spectrum" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(115)

        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Tlm[1, :] .= 0

        ens_total = enstrophy(cfg, Tlm)

        Zl = enstrophy_l_spectrum(cfg, Tlm)
        @test length(Zl) == lmax + 1
        @test isapprox(sum(Zl), ens_total; rtol=1e-10)
        @test all(Zl .>= 0)
    end

    @testset "Enstrophy m-spectrum" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(116)

        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Tlm[1, :] .= 0

        ens_total = enstrophy(cfg, Tlm)

        Zm = enstrophy_m_spectrum(cfg, Tlm)
        @test isapprox(sum(Zm), ens_total; rtol=1e-10)
        @test all(Zm .>= 0)
    end

    @testset "Enstrophy per-mode" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(117)

        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Tlm[1, :] .= 0

        ens_total = enstrophy(cfg, Tlm)
        Zlm = enstrophy_lm(cfg, Tlm)

        @test size(Zlm) == (lmax+1, cfg.mmax+1)
        @test isapprox(sum(Zlm), ens_total; rtol=1e-10)
    end

    @testset "Zero vorticity for irrotational flow" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(118)

        # Pure spheroidal (irrotational) flow
        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Slm[:, 1] .= real.(Slm[:, 1])
        Slm[1, :] .= 0
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)

        # Vorticity should be zero
        zeta_spec = vorticity_spectral(cfg, Tlm)
        @test all(abs.(zeta_spec) .< 1e-14)

        # Enstrophy should be zero
        ens = enstrophy(cfg, Tlm)
        @test ens < 1e-20
    end

    @testset "Single mode enstrophy" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Single toroidal mode
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm[4, 2] = 1.0 + 0im  # l=3, m=1

        ens_spec = enstrophy(cfg, Tlm)
        @test ens_spec > 0

        # l-spectrum should have single non-zero entry
        Zl = enstrophy_l_spectrum(cfg, Tlm)
        @test Zl[4] > 0
        @test sum(Zl[1:3]) < 1e-14
        @test sum(Zl[5:end]) < 1e-14
    end
end
