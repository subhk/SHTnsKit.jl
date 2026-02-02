# SHTnsKit.jl - Truncated and Mode-Limited Transform Tests
# Tests for degree-limited and single-mode transforms

using Test
using Random
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "Truncated and Mode-Limited Transforms" begin
    @testset "Degree-limited analysis" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(45)

        # Random spatial field
        f = randn(rng, nlat, nlon)
        Qlm_full = spat_to_SH(cfg, vec(f))

        # Truncated analysis
        ltr = lmax - 2
        Qlm_trunc = spat_to_SH_l(cfg, vec(f), ltr)

        # Verify truncation: modes l > ltr should be zero
        for m in 0:cfg.mmax
            for l in m:cfg.lmax
                idx = LM_index(cfg.lmax, cfg.mres, l, m) + 1
                if l <= ltr
                    @test isapprox(Qlm_trunc[idx], Qlm_full[idx]; rtol=1e-10, atol=1e-12)
                else
                    @test Qlm_trunc[idx] == 0
                end
            end
        end
    end

    @testset "Degree-limited synthesis" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(46)

        # Random spatial field
        f = randn(rng, nlat, nlon)
        Qlm_full = spat_to_SH(cfg, vec(f))
        ltr = lmax - 2

        # Truncated synthesis
        f_trunc = SH_to_spat_l(cfg, Qlm_full, ltr)

        # Should match synthesis of zeroed-out high modes
        Qlm_zeroed = copy(Qlm_full)
        for m in 0:cfg.mmax
            for l in (ltr+1):cfg.lmax
                idx = LM_index(cfg.lmax, cfg.mres, l, m) + 1
                Qlm_zeroed[idx] = 0
            end
        end
        f_zeroed = SH_to_spat(cfg, Qlm_zeroed)
        @test isapprox(f_trunc, f_zeroed; rtol=1e-10, atol=1e-12)
    end

    @testset "Point evaluation" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Single spherical harmonic: Y_2^0 ∝ (3cos²θ - 1)/2
        alm = zeros(ComplexF64, lmax+1, lmax+1)
        alm[3, 1] = 1.0  # l=2, m=0

        # Evaluate at specific point
        cost = 0.5  # cos(θ)
        phi = 0.0
        val = SH_to_point(cfg, alm, cost, phi)

        # Should be finite and well-defined
        @test !isnan(val) && !isinf(val)

        # Test at multiple points
        for cost in [-0.8, -0.3, 0.0, 0.3, 0.8]
            for phi in [0.0, π/4, π/2, π, 3π/2]
                val = SH_to_point(cfg, alm, cost, phi)
                @test !isnan(val) && !isinf(val)
            end
        end
    end

    @testset "Latitude evaluation" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(47)

        # Random coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])  # m=0 real

        cost = 0.3
        lat_vals = SH_to_lat(cfg, Qlm, cost)

        @test length(lat_vals) == nlon

        # Verify consistency with full synthesis at this latitude
        # The latitude values should match a row of the full synthesis
        f_full = reshape(SH_to_spat(cfg, Qlm), nlat, nlon)

        # Find closest latitude in grid
        idx = argmin(abs.(cfg.x .- cost))
        if abs(cfg.x[idx] - cost) < 1e-10
            @test isapprox(lat_vals, f_full[idx, :]; rtol=1e-9, atol=1e-11)
        end
    end

    @testset "Mode-limited transforms" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(48)

        # Test for different m values
        for im in 0:3
            ltr = lmax - 1
            len = ltr - im + 1

            if len > 0
                # Random coefficients for this m mode
                Ql = randn(rng, ComplexF64, len)

                # Mode synthesis
                f_ml = SH_to_spat_ml(cfg, im, Ql, ltr)
                @test length(f_ml) == nlat

                # Mode analysis roundtrip
                Ql_back = spat_to_SH_ml(cfg, im, f_ml, ltr)
                @test isapprox(Ql_back, Ql; rtol=1e-9, atol=1e-11)
            end
        end
    end
end
