# SHTnsKit.jl - Padded spatial allocation tests
# Exercises set_allow_padding!, allocate_padded_spatial, allocate_padded_spatial_batch,
# copy_to_padded! / copy_from_padded!, estimate_padding_overhead, NSPAT_ALLOC.

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Padding helpers" begin
    @testset "Default state: padding disabled" begin
        cfg = create_gauss_config(6, 8; nlon=13)
        @test cfg.allow_padding == false
        @test is_padding_enabled(cfg) == false
        @test get_nlat_padded(cfg) == cfg.nlat
        @test estimate_padding_overhead(cfg) == 0.0
    end

    @testset "Enable padding" begin
        cfg = create_gauss_config(6, 8; nlon=13)
        set_allow_padding!(cfg)
        @test cfg.allow_padding == true
        @test is_padding_enabled(cfg) == true

        nlat_p = get_nlat_padded(cfg)
        @test nlat_p ≥ cfg.nlat

        # NSPAT_ALLOC tracks padded spat_dist if > 0
        n_spat = NSPAT_ALLOC(cfg)
        @test n_spat ≥ cfg.nspat

        # Overhead ≥ 0%
        @test estimate_padding_overhead(cfg) ≥ 0.0
    end

    @testset "allocate_padded_spatial shape" begin
        cfg = create_gauss_config(6, 8; nlon=13)
        set_allow_padding!(cfg)

        buf = allocate_padded_spatial(cfg)
        @test size(buf, 1) == get_nlat_padded(cfg)
        @test size(buf, 2) == cfg.nlon
        @test all(iszero, buf)

        buf_f32 = allocate_padded_spatial(cfg, Float32)
        @test eltype(buf_f32) == Float32
    end

    @testset "allocate_padded_spatial_batch shape" begin
        cfg = create_gauss_config(5, 7; nlon=11)
        set_allow_padding!(cfg)

        nfields = 4
        buf = allocate_padded_spatial_batch(cfg, nfields)
        @test size(buf) == (get_nlat_padded(cfg), cfg.nlon, nfields)
        @test eltype(buf) == Float64
    end

    @testset "copy_to_padded! / copy_from_padded! roundtrip" begin
        cfg = create_gauss_config(5, 7; nlon=11)
        set_allow_padding!(cfg)

        src = reshape(collect(1.0:(cfg.nlat * cfg.nlon)), cfg.nlat, cfg.nlon)
        padded = allocate_padded_spatial(cfg)

        copy_to_padded!(padded, src, cfg)
        @test padded[1:cfg.nlat, :] == src
        # Padded rows past nlat stay zero
        if size(padded, 1) > cfg.nlat
            @test all(iszero, padded[cfg.nlat + 1:end, :])
        end

        back = zeros(cfg.nlat, cfg.nlon)
        copy_from_padded!(back, padded, cfg)
        @test back == src
    end

    @testset "Padded copy: dimension checks" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        set_allow_padding!(cfg)
        padded = allocate_padded_spatial(cfg)

        # src wrong rows
        @test_throws DimensionMismatch copy_to_padded!(padded,
            zeros(cfg.nlat + 1, cfg.nlon), cfg)
        # src wrong cols
        @test_throws DimensionMismatch copy_to_padded!(padded,
            zeros(cfg.nlat, cfg.nlon + 1), cfg)
        # dest too short
        @test_throws DimensionMismatch copy_to_padded!(zeros(cfg.nlat - 1, cfg.nlon),
            zeros(cfg.nlat, cfg.nlon), cfg)

        # copy_from_padded!: dest must equal nlat × nlon
        @test_throws DimensionMismatch copy_from_padded!(zeros(cfg.nlat + 1, cfg.nlon),
            padded, cfg)
        @test_throws DimensionMismatch copy_from_padded!(zeros(cfg.nlat, cfg.nlon - 1),
            padded, cfg)
    end

    @testset "disable_padding! resets state" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        set_allow_padding!(cfg)
        @test cfg.allow_padding
        disable_padding!(cfg)
        @test cfg.allow_padding == false
        @test get_nlat_padded(cfg) == cfg.nlat
    end
end
