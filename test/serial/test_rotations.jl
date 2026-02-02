# SHTnsKit.jl - Rotation Tests
# Tests for spectral rotations and Wigner-d matrices

using Test
using Random
using LinearAlgebra
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

@testset "Rotations" begin
    @testset "Z-axis rotation phase" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(90)

        # Random coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])  # m=0 real

        # Rotation angle
        α = π/4
        Rlm = similar(Qlm)
        SH_Zrotate(cfg, Qlm, α, Rlm)

        # Z-rotation multiplies by exp(imα)
        for m in 0:cfg.mmax
            for l in m:cfg.lmax
                idx = LM_index(cfg.lmax, cfg.mres, l, m) + 1
                expected = Qlm[idx] * cis(m * α)
                @test isapprox(Rlm[idx], expected; rtol=1e-12, atol=1e-14)
            end
        end
    end

    @testset "Z-axis rotation in-place" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(90)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        α = π/4
        Rlm = similar(Qlm)
        SH_Zrotate(cfg, Qlm, α, Rlm)

        # In-place rotation
        Qlm_copy = copy(Qlm)
        SH_Zrotate(cfg, Qlm_copy, α, Qlm_copy)
        @test isapprox(Qlm_copy, Rlm; rtol=1e-12, atol=1e-14)
    end

    @testset "Z-rotation periodicity" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(91)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        # Rotation by 2π should return original (for m-integer modes)
        Rlm = similar(Qlm)
        SH_Zrotate(cfg, Qlm, 2π, Rlm)
        @test isapprox(Rlm, Qlm; rtol=1e-10, atol=1e-12)
    end

    @testset "Y-axis rotation energy preservation" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(92)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        α = π/3
        Rlm = similar(Qlm)
        SH_Yrotate(cfg, Qlm, α, Rlm)

        # Y-rotation mixes m modes within same l
        # Check that total energy is preserved (rotation is unitary)
        E_before = sum(abs2, Qlm)
        E_after = sum(abs2, Rlm)
        @test isapprox(E_before, E_after; rtol=1e-10, atol=1e-12)
    end

    @testset "90-degree rotations" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(93)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        # Y-rotation by 90°
        Rlm_y90 = similar(Qlm)
        SH_Yrotate90(cfg, Qlm, Rlm_y90)

        # X-rotation by 90°
        Rlm_x90 = similar(Qlm)
        SH_Xrotate90(cfg, Qlm, Rlm_x90)

        # Both should preserve energy
        @test isapprox(sum(abs2, Qlm), sum(abs2, Rlm_y90); rtol=1e-10)
        @test isapprox(sum(abs2, Qlm), sum(abs2, Rlm_x90); rtol=1e-10)
    end

    @testset "Wigner-d matrix orthogonality" begin
        # Test orthogonality: d^T d = I for real orthogonal case
        for l in 0:4
            β = 0.7
            d = SHTnsKit.wigner_d_matrix(l, β)
            n = 2l + 1
            @test size(d) == (n, n)

            # Orthogonality check
            prod = d' * d
            @test isapprox(prod, I(n); rtol=1e-10, atol=1e-12)
        end
    end

    @testset "Wigner-d matrix identity at β=0" begin
        for l in 0:4
            d0 = SHTnsKit.wigner_d_matrix(l, 0.0)
            @test isapprox(d0, I(2l+1); rtol=1e-12)
        end
    end

    @testset "Wigner-d matrix β=π inversion" begin
        # At β=π, d^l_{mm'}(π) = (-1)^(l+m) δ_{m,-m'}
        for l in 0:3
            d_pi = SHTnsKit.wigner_d_matrix(l, π)
            n = 2l + 1
            for m in -l:l
                for mp in -l:l
                    expected = (mp == -m) ? (-1.0)^(l + m) : 0.0
                    @test isapprox(d_pi[m+l+1, mp+l+1], expected; rtol=1e-10, atol=1e-12)
                end
            end
        end
    end

    @testset "SHTRotation struct" begin
        lmax = 4
        mmax = 4

        # Create rotation
        rot = SHTRotation(lmax, mmax; α=0.1, β=0.2, γ=0.3)
        @test rot.lmax == lmax
        @test rot.mmax == mmax
        @test rot.α ≈ 0.1
        @test rot.β ≈ 0.2
        @test rot.γ ≈ 0.3
        @test rot.conv == :ZYZ

        # Set angles ZYZ
        shtns_rotation_set_angles_ZYZ(rot, 0.5, 0.6, 0.7)
        @test rot.α ≈ 0.5
        @test rot.β ≈ 0.6
        @test rot.γ ≈ 0.7
        @test rot.conv == :ZYZ

        # Set angles ZXZ
        shtns_rotation_set_angles_ZXZ(rot, 0.8, 0.9, 1.0)
        @test rot.α ≈ 0.8
        @test rot.β ≈ 0.9
        @test rot.γ ≈ 1.0
        @test rot.conv == :ZXZ
    end

    @testset "General rotation application (real)" begin
        lmax = 3
        mmax = 3
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(94)

        # Random coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        # Create and apply rotation
        rot = SHTRotation(lmax, mmax)
        shtns_rotation_set_angles_ZYZ(rot, π/6, π/4, π/3)

        Rlm = similar(Qlm)
        shtns_rotation_apply_real(rot, Qlm, Rlm)

        # Energy preserved
        @test isapprox(sum(abs2, Qlm), sum(abs2, Rlm); rtol=1e-9)
    end

    @testset "Complex rotation application" begin
        lmax = 3
        mmax = 3
        rng = MersenneTwister(95)

        # Complex coefficients (full m range)
        nlm_cplx = nlm_cplx_calc(lmax, mmax, 1)
        Zlm = randn(rng, ComplexF64, nlm_cplx)

        rot = SHTRotation(lmax, mmax)
        shtns_rotation_set_angles_ZYZ(rot, 0.3, 0.5, 0.2)

        Rlm = similar(Zlm)
        shtns_rotation_apply_cplx(rot, Zlm, Rlm)

        # Energy preserved
        @test isapprox(sum(abs2, Zlm), sum(abs2, Rlm); rtol=1e-9)
    end

    @testset "Rotation composition" begin
        lmax = 3
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(96)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        # Two Z-rotations should compose additively
        α1 = π/6
        α2 = π/4
        Rlm1 = similar(Qlm)
        Rlm2 = similar(Qlm)
        Rlm_direct = similar(Qlm)

        SH_Zrotate(cfg, Qlm, α1, Rlm1)
        SH_Zrotate(cfg, Rlm1, α2, Rlm2)
        SH_Zrotate(cfg, Qlm, α1 + α2, Rlm_direct)

        @test isapprox(Rlm2, Rlm_direct; rtol=1e-10, atol=1e-12)
    end

    @testset "Rotation inverse" begin
        lmax = 3
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(97)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        α = π/3
        Rlm = similar(Qlm)
        Qlm_back = similar(Qlm)

        # Rotate forward then backward
        SH_Zrotate(cfg, Qlm, α, Rlm)
        SH_Zrotate(cfg, Rlm, -α, Qlm_back)

        @test isapprox(Qlm_back, Qlm; rtol=1e-10, atol=1e-12)
    end
end
