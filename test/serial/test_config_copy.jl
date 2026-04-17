# SHTnsKit.jl - SHTConfig copy / mutation-isolation tests
# Ensures Base.copy(cfg) deep-copies mutable state so modifying one config does
# not bleed into another, and flags/tables survive independently.

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "SHTConfig copy" begin
    @testset "Base.copy yields independent arrays" begin
        cfg = create_gauss_config(6, 8; nlon=13)
        cfg2 = copy(cfg)

        @test cfg2 !== cfg
        @test cfg2.lmax == cfg.lmax
        @test cfg2.mmax == cfg.mmax
        @test cfg2.nlat == cfg.nlat
        @test cfg2.nlon == cfg.nlon
        @test cfg2.θ == cfg.θ
        @test cfg2.w == cfg.w
        @test cfg2.x == cfg.x

        # Mutating the copy must not alter the original
        cfg2.θ[1] = 99.0
        @test cfg.θ[1] != 99.0
        cfg2.w[end] = -1.0
        @test cfg.w[end] != -1.0
    end

    @testset "PLM tables on original are independent of copy" begin
        cfg = create_gauss_config(5, 7; nlon=11)
        prepare_plm_tables!(cfg)
        cfg2 = copy(cfg)

        @test cfg2.use_plm_tables == true
        @test length(cfg2.plm_tables) == length(cfg.plm_tables)
        # Mutate copy's tables
        cfg2.plm_tables[1][1, 1] = 7.5
        @test cfg.plm_tables[1][1, 1] != 7.5
    end

    @testset "south_pole_first flag on copy independent of original" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        cfg2 = copy(cfg)
        set_south_pole_first!(cfg2)
        @test cfg2.south_pole_first == true
        @test cfg.south_pole_first == false
        # x arrays must not be entangled
        @test cfg.x != cfg2.x
    end

    @testset "padding flag / nlat_padded independent" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        cfg2 = copy(cfg)
        set_allow_padding!(cfg2)
        @test cfg2.allow_padding
        @test cfg.allow_padding == false
    end

    @testset "Copy remains transform-capable" begin
        cfg = create_gauss_config(5, 7; nlon=11)
        cfg2 = copy(cfg)
        alm = zeros(ComplexF64, cfg2.lmax + 1, cfg2.mmax + 1)
        alm[3, 2] = 1.0 + 0im  # l=2, m=1
        f = synthesis(cfg2, alm; real_output=true)
        alm_back = analysis(cfg2, f)
        @test isapprox(alm_back[3, 2], alm[3, 2]; rtol=1e-10, atol=1e-12)
    end
end
