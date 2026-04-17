# SHTnsKit.jl - South-pole-first grid ordering tests
# Exercises set_south_pole_first! / set_north_pole_first! and verifies that
# transforms remain correct under the reversed latitude ordering.

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

function _real_field_alm(rng, lmax)
    alm = randn(rng, ComplexF64, lmax + 1, lmax + 1)
    alm[:, 1] .= real.(alm[:, 1])
    for m in 0:lmax, l in 0:(m - 1)
        alm[l + 1, m + 1] = 0
    end
    return alm
end

@testset "South-pole-first ordering" begin
    @testset "set_south_pole_first! toggles flag and reverses x" begin
        cfg = create_gauss_config(6, 8; nlon=13)
        @test cfg.south_pole_first == false
        @test is_south_pole_first(cfg) == false
        x_orig = copy(cfg.x)
        θ_orig = copy(cfg.θ)
        w_orig = copy(cfg.w)

        set_south_pole_first!(cfg)
        @test cfg.south_pole_first == true
        @test is_south_pole_first(cfg) == true
        @test cfg.x == reverse(x_orig)
        @test cfg.θ == reverse(θ_orig)
        @test cfg.w == reverse(w_orig)

        # Re-toggle: no-op when already SPF
        set_south_pole_first!(cfg)
        @test cfg.x == reverse(x_orig)

        set_north_pole_first!(cfg)
        @test cfg.south_pole_first == false
        @test cfg.x == x_orig
        @test cfg.w == w_orig

        # Re-toggle: no-op when already NPF
        set_north_pole_first!(cfg)
        @test cfg.x == x_orig
    end

    @testset "Roundtrip under south-pole-first" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        set_south_pole_first!(cfg)

        rng = MersenneTwister(800)
        alm = _real_field_alm(rng, lmax)
        f = synthesis(cfg, alm; real_output=true)
        alm_back = analysis(cfg, f)
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "SPF spatial field is NPF result reversed in θ" begin
        lmax = 5
        rng = MersenneTwister(801)
        alm = _real_field_alm(rng, lmax)

        cfg_n = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        f_n = synthesis(cfg_n, alm; real_output=true)

        cfg_s = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        set_south_pole_first!(cfg_s)
        f_s = synthesis(cfg_s, alm; real_output=true)

        # The south-pole-first field is the north-pole-first field reversed
        # along the latitude axis (same physical field, different storage order)
        @test isapprox(f_s, reverse(f_n; dims=1); rtol=1e-10, atol=1e-12)
    end

    @testset "Energy is ordering-invariant" begin
        lmax = 5
        cfg_n = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        cfg_s = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        set_south_pole_first!(cfg_s)

        rng = MersenneTwister(802)
        alm = _real_field_alm(rng, lmax)

        E_n = energy_scalar(cfg_n, alm)
        E_s = energy_scalar(cfg_s, alm)
        @test isapprox(E_n, E_s; rtol=1e-12, atol=1e-14)
    end

    @testset "PLM tables survive ordering flip (recomputed)" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        prepare_plm_tables!(cfg)
        @test cfg.use_plm_tables
        set_south_pole_first!(cfg)
        @test cfg.use_plm_tables
        # After flip, a transform still roundtrips
        rng = MersenneTwister(803)
        alm = _real_field_alm(rng, lmax)
        f = synthesis(cfg, alm; real_output=true)
        alm_back = analysis(cfg, f)
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
    end
end
