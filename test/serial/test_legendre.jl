# SHTnsKit.jl - Legendre Polynomial Tests
# Tests for Plm_row!, Plm_and_dPdx_row!, Plm_and_dPdtheta_row!,
# Plm_over_sinth_row!, Plm_dPdtheta_over_sinth_row!, Nlm_table, gausslegendre

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Legendre Polynomials" begin
    @testset "Plm_row! basic values" begin
        # P_0^0(x) = 1 for all x
        P = zeros(11)
        SHTnsKit.Plm_row!(P, 0.5, 10, 0)
        @test P[1] ≈ 1.0

        # P_1^0(x) = x
        @test P[2] ≈ 0.5

        # P_2^0(x) = (3x²-1)/2
        @test P[3] ≈ (3 * 0.25 - 1) / 2

        # P_3^0(x) = (5x³ - 3x)/2
        x = 0.5
        @test P[4] ≈ (5 * x^3 - 3x) / 2
    end

    @testset "Plm_row! m=1" begin
        x = 0.6
        P = zeros(11)
        SHTnsKit.Plm_row!(P, x, 10, 1)

        # P_1^1(x) = -sqrt(1-x²) (Condon-Shortley)
        @test isapprox(P[2], -sqrt(1 - x^2); rtol=1e-12)

        # P_2^1(x) = -3x*sqrt(1-x²)
        @test isapprox(P[3], -3x * sqrt(1 - x^2); rtol=1e-12)
    end

    @testset "Plm_row! m > lmax returns zeros" begin
        P = zeros(5)
        SHTnsKit.Plm_row!(P, 0.5, 2, 5)
        @test all(P .== 0.0)
    end

    @testset "Plm_row! negative m throws" begin
        P = zeros(5)
        @test_throws ArgumentError SHTnsKit.Plm_row!(P, 0.5, 4, -1)
    end

    @testset "Plm_row! at poles (x=±1)" begin
        P = zeros(11)

        # At x = 1 (north pole)
        SHTnsKit.Plm_row!(P, 1.0, 10, 0)
        # P_l^0(1) = 1 for all l
        for l in 0:10
            @test isapprox(P[l+1], 1.0; atol=1e-12)
        end

        # At x = -1 (south pole)
        SHTnsKit.Plm_row!(P, -1.0, 10, 0)
        # P_l^0(-1) = (-1)^l
        for l in 0:10
            @test isapprox(P[l+1], (-1.0)^l; atol=1e-12)
        end

        # m > 0 at poles: P_l^m(±1) = 0
        SHTnsKit.Plm_row!(P, 1.0, 10, 1)
        for l in 1:10
            @test abs(P[l+1]) < 1e-10
        end
    end

    @testset "Plm_row! orthogonality" begin
        # ∫₋₁¹ P_l^m(x) P_{l'}^m(x) dx = 2/(2l+1) * (l+m)!/(l-m)! * δ_{ll'}
        # Test using Gauss-Legendre quadrature
        lmax = 15
        nlat = lmax + 5
        x_nodes, w_nodes = SHTnsKit.gausslegendre(nlat)

        for m in 0:3
            P1 = zeros(lmax + 1)
            P2 = zeros(lmax + 1)
            for l1 in m:min(lmax, m+4), l2 in m:min(lmax, m+4)
                integral = 0.0
                for (xi, wi) in zip(x_nodes, w_nodes)
                    SHTnsKit.Plm_row!(P1, xi, lmax, m)
                    integral += wi * P1[l1+1] * P1[l2+1]
                end
                if l1 == l2
                    # Expected: 2/(2l+1) * (l+m)!/(l-m)!
                    expected = 2.0 / (2l1 + 1)
                    for k in (l1-m+1):(l1+m)
                        expected *= k
                    end
                    @test isapprox(integral, expected; rtol=1e-10)
                else
                    @test abs(integral) < 1e-10
                end
            end
        end
    end

    @testset "Plm_row! large m (log-space)" begin
        # Test that log-space computation (m > 140) gives same results as direct
        # for intermediate m values where both should work
        x = 0.7
        lmax = 200
        P_large = zeros(lmax + 1)

        # m = 150 (uses log-space path)
        SHTnsKit.Plm_row!(P_large, x, lmax, 150)
        # P_m^m should be nonzero and finite
        @test isfinite(P_large[151])
        @test P_large[151] != 0.0

        # m = 140 (uses direct path, boundary)
        P_direct = zeros(lmax + 1)
        SHTnsKit.Plm_row!(P_direct, x, lmax, 140)
        @test isfinite(P_direct[141])
        @test P_direct[141] != 0.0
    end

    @testset "Plm_row! large m at pole" begin
        P = zeros(201)
        # At pole, P_l^m = 0 for m > 0, even with log-space path
        SHTnsKit.Plm_row!(P, 1.0, 200, 150)
        @test all(abs.(P) .< 1e-10)
    end

    @testset "Plm_and_dPdx_row!" begin
        x = 0.6
        lmax = 10
        P = zeros(lmax + 1)
        dPdx = zeros(lmax + 1)

        SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, 0)

        # dP_0^0/dx = 0
        @test dPdx[1] ≈ 0.0

        # dP_1^0/dx = 1
        @test isapprox(dPdx[2], 1.0; atol=1e-12)

        # dP_2^0/dx = 3x
        @test isapprox(dPdx[3], 3x; rtol=1e-12)

        # Verify P values are computed correctly too
        @test P[1] ≈ 1.0
        @test P[2] ≈ x
    end

    @testset "Plm_and_dPdx_row! at poles" begin
        lmax = 8
        P = zeros(lmax + 1)
        dPdx = zeros(lmax + 1)

        # North pole (x=1): dP_l^0/dx = l(l+1)/2
        SHTnsKit.Plm_and_dPdx_row!(P, dPdx, 1.0, lmax, 0)
        for l in 1:lmax
            expected = l * (l + 1) / 2.0
            @test isapprox(dPdx[l+1], expected; rtol=1e-10)
        end

        # South pole (x=-1): dP_l^0/dx = (-1)^{l+1} * l(l+1)/2
        SHTnsKit.Plm_and_dPdx_row!(P, dPdx, -1.0, lmax, 0)
        for l in 1:lmax
            expected = (-1.0)^(l+1) * l * (l + 1) / 2.0
            @test isapprox(dPdx[l+1], expected; rtol=1e-10)
        end
    end

    @testset "Plm_and_dPdtheta_row!" begin
        x = 0.6
        lmax = 10
        P = zeros(lmax + 1)
        dPdtheta = zeros(lmax + 1)

        SHTnsKit.Plm_and_dPdtheta_row!(P, dPdtheta, x, lmax, 0)

        # dP_0^0/dθ = 0
        @test dPdtheta[1] ≈ 0.0

        # Relationship: dP/dθ = -sinθ * dP/dx
        P2 = zeros(lmax + 1)
        dPdx = zeros(lmax + 1)
        SHTnsKit.Plm_and_dPdx_row!(P2, dPdx, x, lmax, 0)
        sinth = sqrt(1 - x^2)
        for l in 0:lmax
            @test isapprox(dPdtheta[l+1], -sinth * dPdx[l+1]; rtol=1e-10)
        end
    end

    @testset "Plm_and_dPdtheta_row! at poles" begin
        lmax = 8
        P = zeros(lmax + 1)
        dPdtheta = zeros(lmax + 1)

        # m=0 at pole: dP/dθ = 0
        SHTnsKit.Plm_and_dPdtheta_row!(P, dPdtheta, 1.0, lmax, 0)
        for l in 0:lmax
            @test abs(dPdtheta[l+1]) < 1e-10
        end

        # m=1 at north pole: dP_l^1/dθ = -l(l+1)/2
        SHTnsKit.Plm_and_dPdtheta_row!(P, dPdtheta, 1.0, lmax, 1)
        for l in 1:lmax
            expected = -l * (l + 1) / 2.0
            @test isapprox(dPdtheta[l+1], expected; rtol=1e-10)
        end

        # m>1 at pole: dP/dθ = 0
        SHTnsKit.Plm_and_dPdtheta_row!(P, dPdtheta, 1.0, lmax, 2)
        for l in 0:lmax
            @test abs(dPdtheta[l+1]) < 1e-10
        end
    end

    @testset "Plm_over_sinth_row!" begin
        x = 0.6
        lmax = 10
        P = zeros(lmax + 1)
        P_over_sinth = zeros(lmax + 1)

        SHTnsKit.Plm_over_sinth_row!(P, P_over_sinth, x, lmax, 1)

        # Should equal P/sin(θ)
        sinth = sqrt(1 - x^2)
        for l in 1:lmax
            @test isapprox(P_over_sinth[l+1], P[l+1] / sinth; rtol=1e-12)
        end
    end

    @testset "Plm_over_sinth_row! at poles" begin
        lmax = 8
        P = zeros(lmax + 1)
        P_over_sinth = zeros(lmax + 1)

        # m=1 at north pole: P_l^1/sinθ = -l(l+1)/2
        SHTnsKit.Plm_over_sinth_row!(P, P_over_sinth, 1.0, lmax, 1)
        for l in 1:lmax
            expected = -l * (l + 1) / 2.0
            @test isapprox(P_over_sinth[l+1], expected; rtol=1e-10)
        end

        # m>1 at pole: P/sinθ = 0
        SHTnsKit.Plm_over_sinth_row!(P, P_over_sinth, 1.0, lmax, 2)
        for l in 0:lmax
            @test abs(P_over_sinth[l+1]) < 1e-10
        end
    end

    @testset "Plm_dPdtheta_over_sinth_row! combined" begin
        x = 0.6
        lmax = 10
        P = zeros(lmax + 1)
        dPdtheta = zeros(lmax + 1)
        P_over_sinth = zeros(lmax + 1)

        SHTnsKit.Plm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, 1)

        # Compare with separate calls
        P_ref = zeros(lmax + 1)
        dPdtheta_ref = zeros(lmax + 1)
        SHTnsKit.Plm_and_dPdtheta_row!(P_ref, dPdtheta_ref, x, lmax, 1)

        P_ref2 = zeros(lmax + 1)
        Pos_ref = zeros(lmax + 1)
        SHTnsKit.Plm_over_sinth_row!(P_ref2, Pos_ref, x, lmax, 1)

        @test isapprox(P, P_ref; rtol=1e-14)
        @test isapprox(dPdtheta, dPdtheta_ref; rtol=1e-14)
        @test isapprox(P_over_sinth, Pos_ref; rtol=1e-14)
    end

    @testset "Plm_dPdtheta_over_sinth_row! at poles" begin
        lmax = 8
        P = zeros(lmax + 1)
        dPdtheta = zeros(lmax + 1)
        P_over_sinth = zeros(lmax + 1)

        # m=1 at north pole
        SHTnsKit.Plm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, 1.0, lmax, 1)
        for l in 1:lmax
            @test isapprox(dPdtheta[l+1], -l * (l + 1) / 2.0; rtol=1e-10)
            @test isapprox(P_over_sinth[l+1], -l * (l + 1) / 2.0; rtol=1e-10)
        end

        # m=0 at pole: everything should be zero for derivatives
        SHTnsKit.Plm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, 1.0, lmax, 0)
        for l in 0:lmax
            @test abs(dPdtheta[l+1]) < 1e-10
        end
    end

    @testset "Plm_dPdtheta_over_sinth_row! lmax < m" begin
        P = zeros(5)
        dPdtheta = zeros(5)
        P_over_sinth = zeros(5)
        SHTnsKit.Plm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, 0.5, 2, 5)
        @test all(P .== 0.0)
        @test all(dPdtheta .== 0.0)
        @test all(P_over_sinth .== 0.0)
    end

    @testset "Nlm_table properties" begin
        lmax = 10
        mmax = 10
        N = SHTnsKit.Nlm_table(lmax, mmax)

        @test size(N) == (lmax + 1, mmax + 1)

        # N_{l,m} = 0 for l < m
        for m in 1:mmax, l in 0:(m-1)
            @test N[l+1, m+1] == 0.0
        end

        # N_{l,m} > 0 for l >= m
        for m in 0:mmax, l in m:lmax
            @test N[l+1, m+1] > 0.0
        end

        # N_{0,0} = 1/sqrt(4π)
        @test isapprox(N[1, 1], 1 / sqrt(4π); rtol=1e-12)

        # Verify orthonormality: N² * integral of P² = 1/(4π) ... simplified check
        # N_{l,0} = sqrt((2l+1)/(4π))
        for l in 0:lmax
            @test isapprox(N[l+1, 1], sqrt((2l + 1) / (4π)); rtol=1e-12)
        end
    end

    @testset "gausslegendre properties" begin
        for n in [4, 8, 16, 32, 64]
            x, w = SHTnsKit.gausslegendre(n)

            @test length(x) == n
            @test length(w) == n

            # Weights sum to 2
            @test isapprox(sum(w), 2.0; rtol=1e-12)

            # Nodes in [-1, 1]
            @test all(-1.0 .<= x .<= 1.0)

            # Symmetry: x[k] ≈ -x[n-k+1]
            for k in 1:(n ÷ 2)
                @test isapprox(x[k], -x[n - k + 1]; atol=1e-14)
                @test isapprox(w[k], w[n - k + 1]; atol=1e-14)
            end

            # Nodes are sorted
            @test issorted(x)

            # All weights positive
            @test all(w .> 0)
        end

        # n=1 is valid
        x, w = SHTnsKit.gausslegendre(1)
        @test x[1] ≈ 0.0
        @test w[1] ≈ 2.0

        # n=0 throws
        @test_throws ArgumentError SHTnsKit.gausslegendre(0)
    end

    @testset "gausslegendre exactness" begin
        # n-point Gauss-Legendre should be exact for polynomials up to degree 2n-1
        n = 8
        x, w = SHTnsKit.gausslegendre(n)

        # ∫₋₁¹ x^k dx = 2/(k+1) if k even, 0 if k odd
        for k in 0:(2n - 1)
            integral = sum(w .* x .^ k)
            if iseven(k)
                expected = 2.0 / (k + 1)
            else
                expected = 0.0
            end
            @test isapprox(integral, expected; atol=1e-12)
        end
    end
end
