# SHTnsKit.jl - Normalization and Phase Convention Tests
# Tests for norm_scale_from_orthonormal, cs_phase_factor, convert_alm_norm!

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Normalization and Phase Conventions" begin
    @testset "norm_scale_from_orthonormal" begin
        # Orthonormal → orthonormal is identity
        for l in 0:10, m in 0:l
            @test SHTnsKit.norm_scale_from_orthonormal(l, m, :orthonormal) ≈ 1.0
        end

        # Orthonormal → fourpi gives sqrt(4π) for all (l,m)
        for l in 0:10, m in 0:l
            @test SHTnsKit.norm_scale_from_orthonormal(l, m, :fourpi) ≈ sqrt(4π)
        end

        # Schmidt semi-normalized: m=0 case
        for l in 0:10
            expected = sqrt(4π / (2l + 1))
            @test SHTnsKit.norm_scale_from_orthonormal(l, 0, :schmidt) ≈ expected
        end

        # Schmidt semi-normalized: m>0 case
        for l in 1:10, m in 1:l
            expected = sqrt(2.0 * 4π / (2l + 1))
            @test SHTnsKit.norm_scale_from_orthonormal(l, m, :schmidt) ≈ expected
        end

        # Unsupported normalization should throw
        @test_throws ArgumentError SHTnsKit.norm_scale_from_orthonormal(2, 1, :unknown)
    end

    @testset "cs_phase_factor" begin
        # Same convention → factor is 1
        for m in 0:10
            @test SHTnsKit.cs_phase_factor(m, true, true) ≈ 1.0
            @test SHTnsKit.cs_phase_factor(m, false, false) ≈ 1.0
        end

        # Different conventions → factor is (-1)^m
        for m in 0:10
            expected = (-1.0)^m
            @test SHTnsKit.cs_phase_factor(m, true, false) ≈ expected
            @test SHTnsKit.cs_phase_factor(m, false, true) ≈ expected
        end

        # Even m: phase factor is +1 when switching
        @test SHTnsKit.cs_phase_factor(0, true, false) ≈ 1.0
        @test SHTnsKit.cs_phase_factor(2, true, false) ≈ 1.0
        @test SHTnsKit.cs_phase_factor(4, true, false) ≈ 1.0

        # Odd m: phase factor is -1 when switching
        @test SHTnsKit.cs_phase_factor(1, true, false) ≈ -1.0
        @test SHTnsKit.cs_phase_factor(3, true, false) ≈ -1.0
        @test SHTnsKit.cs_phase_factor(5, true, false) ≈ -1.0
    end

    @testset "convert_alm_norm! roundtrip" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        # Create random coefficients
        src = zeros(ComplexF64, lmax+1, lmax+1)
        for m in 0:lmax, l in m:lmax
            src[l+1, m+1] = randn(ComplexF64)
        end
        src[:, 1] .= real.(src[:, 1])  # m=0 real

        dest = similar(src)
        back = similar(src)

        # Internal → external → internal should roundtrip
        SHTnsKit.convert_alm_norm!(dest, src, cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(back, dest, cfg; to_internal=true)
        @test isapprox(back, src; rtol=1e-12)
    end

    @testset "convert_alm_norm! fourpi convention" begin
        lmax = 4
        nlat = lmax + 2
        cfg_fourpi = create_gauss_config(lmax, nlat; norm=:fourpi, cs_phase=true)

        src = zeros(ComplexF64, lmax+1, lmax+1)
        src[1, 1] = 1.0 + 0im  # l=0, m=0
        src[2, 1] = 2.0 + 0im  # l=1, m=0

        dest = similar(src)
        SHTnsKit.convert_alm_norm!(dest, src, cfg_fourpi; to_internal=false)

        # For fourpi norm, conversion from internal divides by sqrt(4π)
        @test isapprox(dest[1, 1], src[1, 1] / sqrt(4π); rtol=1e-12)
    end

    @testset "convert_alm_norm! dimension mismatch" begin
        lmax = 4
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        src = zeros(ComplexF64, lmax+1, lmax+1)
        dest_wrong = zeros(ComplexF64, lmax+2, lmax+1)

        @test_throws DimensionMismatch SHTnsKit.convert_alm_norm!(dest_wrong, src, cfg)
    end
end
