# SHTnsKit.jl - Buffer Utilities Tests
# Tests for buffer allocation, validation, and management

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Buffer Utilities" begin
    lmax = 8
    nlat = lmax + 2
    nlon = 2 * lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)

    @testset "allocate_spectral_pair" begin
        t1 = zeros(ComplexF64, lmax+1, lmax+1)
        t2 = zeros(ComplexF64, lmax+1, lmax+1)
        b1, b2 = SHTnsKit.allocate_spectral_pair(t1, t2)
        @test size(b1) == size(t1)
        @test size(b2) == size(t2)
        @test eltype(b1) == ComplexF64
    end

    @testset "allocate_spectral_triple" begin
        t = zeros(ComplexF64, lmax+1, lmax+1)
        b1, b2, b3 = SHTnsKit.allocate_spectral_triple(t, t, t)
        @test size(b1) == size(t)
        @test size(b2) == size(t)
        @test size(b3) == size(t)
    end

    @testset "copy_spectral_pair" begin
        s1 = randn(ComplexF64, lmax+1, lmax+1)
        s2 = randn(ComplexF64, lmax+1, lmax+1)
        c1, c2 = SHTnsKit.copy_spectral_pair(s1, s2)
        @test c1 == s1
        @test c2 == s2
        # Should be independent copies
        c1[1, 1] = 999.0
        @test s1[1, 1] != 999.0
    end

    @testset "copy_spectral_triple" begin
        s1 = randn(ComplexF64, lmax+1, lmax+1)
        s2 = randn(ComplexF64, lmax+1, lmax+1)
        s3 = randn(ComplexF64, lmax+1, lmax+1)
        c1, c2, c3 = SHTnsKit.copy_spectral_triple(s1, s2, s3)
        @test c1 == s1
        @test c2 == s2
        @test c3 == s3
    end

    @testset "allocate_spatial_pair" begin
        t1 = zeros(nlat, nlon)
        t2 = zeros(nlat, nlon)
        b1, b2 = SHTnsKit.allocate_spatial_pair(t1, t2)
        @test size(b1) == (nlat, nlon)
        @test size(b2) == (nlat, nlon)
    end

    @testset "allocate_spatial_triple" begin
        t = zeros(nlat, nlon)
        b1, b2, b3 = SHTnsKit.allocate_spatial_triple(t, t, t)
        @test size(b1) == (nlat, nlon)
        @test size(b2) == (nlat, nlon)
        @test size(b3) == (nlat, nlon)
    end

    @testset "create_zero_coefficients" begin
        arr = SHTnsKit.create_zero_coefficients(cfg)
        @test size(arr) == (lmax + 1, lmax + 1)
        @test eltype(arr) == ComplexF64
        @test all(arr .== 0)

        # With different type
        arr_f = SHTnsKit.create_zero_coefficients(cfg, Float64)
        @test eltype(arr_f) == Float64
        @test all(arr_f .== 0)
    end

    @testset "scratch_spatial" begin
        buf = SHTnsKit.scratch_spatial(cfg)
        @test size(buf) == (nlat, nlon)
        @test eltype(buf) == Float64
        @test all(buf .== 0)
    end

    @testset "scratch_fft" begin
        buf = SHTnsKit.scratch_fft(cfg)
        @test size(buf) == (nlat, nlon)
        @test eltype(buf) <: Complex
        @test all(buf .== 0)
    end

    @testset "validate_spectral_dimensions" begin
        good = zeros(ComplexF64, lmax+1, lmax+1)
        @test SHTnsKit.validate_spectral_dimensions(good, cfg) === good

        bad_rows = zeros(ComplexF64, lmax+2, lmax+1)
        @test_throws DimensionMismatch SHTnsKit.validate_spectral_dimensions(bad_rows, cfg)

        bad_cols = zeros(ComplexF64, lmax+1, lmax+2)
        @test_throws DimensionMismatch SHTnsKit.validate_spectral_dimensions(bad_cols, cfg)
    end

    @testset "validate_spatial_dimensions" begin
        good = zeros(nlat, nlon)
        @test SHTnsKit.validate_spatial_dimensions(good, cfg) === good

        bad = zeros(nlat + 1, nlon)
        @test_throws DimensionMismatch SHTnsKit.validate_spatial_dimensions(bad, cfg)
    end

    @testset "validate_spectral_pair_dimensions" begin
        a1 = zeros(ComplexF64, lmax+1, lmax+1)
        a2 = zeros(ComplexF64, lmax+1, lmax+1)
        r1, r2 = SHTnsKit.validate_spectral_pair_dimensions(a1, a2, cfg)
        @test r1 === a1
        @test r2 === a2

        bad = zeros(ComplexF64, lmax+2, lmax+1)
        @test_throws DimensionMismatch SHTnsKit.validate_spectral_pair_dimensions(bad, a2, cfg)
    end

    @testset "validate_spatial_pair_dimensions" begin
        a1 = zeros(nlat, nlon)
        a2 = zeros(nlat, nlon)
        r1, r2 = SHTnsKit.validate_spatial_pair_dimensions(a1, a2, cfg)
        @test r1 === a1

        bad = zeros(nlat + 1, nlon)
        @test_throws DimensionMismatch SHTnsKit.validate_spatial_pair_dimensions(bad, a2, cfg)
    end

    @testset "validate_qst_dimensions" begin
        Q = zeros(ComplexF64, lmax+1, lmax+1)
        S = zeros(ComplexF64, lmax+1, lmax+1)
        T = zeros(ComplexF64, lmax+1, lmax+1)
        rQ, rS, rT = SHTnsKit.validate_qst_dimensions(Q, S, T, cfg)
        @test rQ === Q

        bad = zeros(ComplexF64, lmax+2, lmax+1)
        @test_throws DimensionMismatch SHTnsKit.validate_qst_dimensions(bad, S, T, cfg)
    end

    @testset "validate_vector_spatial_dimensions" begin
        Vr = zeros(nlat, nlon)
        Vt = zeros(nlat, nlon)
        Vp = zeros(nlat, nlon)
        r1, r2, r3 = SHTnsKit.validate_vector_spatial_dimensions(Vr, Vt, Vp, cfg)
        @test r1 === Vr

        bad = zeros(nlat + 1, nlon)
        @test_throws DimensionMismatch SHTnsKit.validate_vector_spatial_dimensions(bad, Vt, Vp, cfg)
    end

    @testset "zero_high_degree_modes!" begin
        arr = ones(ComplexF64, lmax+1, lmax+1)
        ltr = 4
        SHTnsKit.zero_high_degree_modes!(arr, cfg, ltr)

        # Modes with l <= ltr should be preserved
        for m in 0:lmax, l in m:min(ltr, lmax)
            @test arr[l+1, m+1] ≈ 1.0
        end

        # Modes with l > ltr should be zeroed
        for m in 0:lmax, l in (ltr+1):lmax
            if l >= m
                @test arr[l+1, m+1] ≈ 0.0
            end
        end
    end

    @testset "zero_high_degree_modes! tuple" begin
        a1 = ones(ComplexF64, lmax+1, lmax+1)
        a2 = ones(ComplexF64, lmax+1, lmax+1)
        ltr = 5
        SHTnsKit.zero_high_degree_modes!((a1, a2), cfg, ltr)

        for m in 0:lmax, l in (ltr+1):lmax
            if l >= m
                @test a1[l+1, m+1] ≈ 0.0
                @test a2[l+1, m+1] ≈ 0.0
            end
        end
    end

    @testset "thread_local_legendre_buffers" begin
        bufs = SHTnsKit.thread_local_legendre_buffers(10)
        @test length(bufs) == Threads.maxthreadid()
        @test length(bufs[1]) == 11

        bufs4 = SHTnsKit.thread_local_legendre_buffers(10, 4)
        @test length(bufs4) == 4
        @test length(bufs4[1]) == 11
    end
end
