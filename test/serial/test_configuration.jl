# SHTnsKit.jl - Configuration and Setup Tests
# Tests for grid configuration, indexing, and normalization

using Test
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "Configuration and Setup" begin
    @testset "Gauss grid configuration" begin
        # Basic configuration
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        @test cfg.lmax == lmax
        @test cfg.mmax == lmax
        @test cfg.nlat == nlat
        @test cfg.nlon == nlon
        @test cfg.grid_type == :gauss

        # Check Gauss weights sum to 2 (integral of 1 over sphere / 2π)
        @test isapprox(sum(cfg.w), 2.0; rtol=1e-12)

        # Check cos(θ) values are in [-1, 1] and sorted
        @test all(-1 .<= cfg.x .<= 1)
        @test issorted(cfg.x; rev=true) || issorted(cfg.x)

        # Check θ values match x = cos(θ)
        @test isapprox(cos.(cfg.θ), cfg.x; rtol=1e-12)

        # Check φ values span [0, 2π)
        @test cfg.φ[1] ≈ 0.0
        @test cfg.φ[end] < 2π
        @test length(cfg.φ) == nlon
    end

    @testset "Regular grid configuration" begin
        lmax = 8
        nlat = 2 * (lmax + 1)
        nlon = 2 * (2 * lmax + 1)

        # Regular grid with Driscoll-Healy weights
        cfg_dh = create_regular_config(lmax, nlat; nlon=nlon, include_poles=true, use_dh_weights=true)
        @test cfg_dh.grid_type in (:regular, :regular_poles, :driscoll_healy)

        # Regular grid without poles
        cfg_nopole = create_regular_config(lmax, nlat; nlon=nlon, include_poles=false)
        @test cfg_nopole.nlat == nlat
    end

    @testset "On-the-fly vs table mode" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        # Test toggling on-the-fly mode
        initial_mode = is_on_the_fly(cfg)
        set_on_the_fly!(cfg, true)
        @test is_on_the_fly(cfg) == true
        set_on_the_fly!(cfg, false)
        @test is_on_the_fly(cfg) == false
        set_on_the_fly!(cfg, initial_mode)
    end

    @testset "Normalization matrix" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2)

        # Nlm should be positive for valid (l,m) pairs
        for m in 0:lmax
            for l in m:lmax
                @test cfg.Nlm[l+1, m+1] > 0
            end
        end
    end

    @testset "Index calculations" begin
        lmax = 5
        mmax = 5
        mres = 1

        # Test nlm_calc
        nlm = nlm_calc(lmax, mmax, mres)
        @test nlm == (lmax + 1) * (lmax + 2) ÷ 2  # triangular number

        # Test LM_index roundtrip
        for m in 0:mmax
            for l in m:lmax
                idx = LM_index(lmax, mres, l, m)
                @test idx >= 0
                @test idx < nlm
            end
        end

        # Test nlm_cplx_calc
        nlm_cplx = nlm_cplx_calc(lmax, mmax, mres)
        @test nlm_cplx == (lmax + 1)^2  # full square for complex
    end

    @testset "Various lmax configurations" begin
        for lmax in [4, 8, 16, 32]
            nlat = lmax + 2
            nlon = 2*lmax + 1
            cfg = create_gauss_config(lmax, nlat; nlon=nlon)

            @test cfg.lmax == lmax
            @test cfg.nlm == (lmax + 1) * (lmax + 2) ÷ 2
            @test length(cfg.x) == nlat
            @test length(cfg.w) == nlat
            @test length(cfg.θ) == nlat
            @test length(cfg.φ) == nlon
        end
    end
end
