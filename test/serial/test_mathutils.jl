# SHTnsKit.jl - Math Utilities Tests
# Tests for logfactorial, loggamma, driscoll_healy_weights

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Math Utilities" begin
    @testset "logfactorial basic values" begin
        @test SHTnsKit.logfactorial(0) ≈ 0.0          # log(0!) = 0
        @test SHTnsKit.logfactorial(1) ≈ 0.0          # log(1!) = 0
        @test SHTnsKit.logfactorial(2) ≈ log(2.0)     # log(2!) = log(2)
        @test SHTnsKit.logfactorial(3) ≈ log(6.0)     # log(3!) = log(6)
        @test SHTnsKit.logfactorial(4) ≈ log(24.0)    # log(4!) = log(24)
        @test SHTnsKit.logfactorial(5) ≈ log(120.0)   # log(5!) = log(120)
        @test SHTnsKit.logfactorial(10) ≈ log(3628800.0)
    end

    @testset "logfactorial large values" begin
        # For large n, exp(logfactorial(n)) should give n!
        # Check consistency: logfactorial(n) = logfactorial(n-1) + log(n)
        for n in 2:200
            @test isapprox(SHTnsKit.logfactorial(n),
                           SHTnsKit.logfactorial(n - 1) + log(n);
                           rtol=1e-12)
        end
    end

    @testset "logfactorial negative throws" begin
        @test_throws DomainError SHTnsKit.logfactorial(-1)
    end

    @testset "logfactorial thread safety" begin
        # Access from multiple tasks shouldn't error
        results = Vector{Float64}(undef, 50)
        Threads.@threads for i in 1:50
            results[i] = SHTnsKit.logfactorial(i + 100)
        end
        # Verify consistency
        for i in 1:50
            @test results[i] ≈ SHTnsKit.logfactorial(i + 100)
        end
    end

    @testset "loggamma integers" begin
        # Γ(n) = (n-1)! for positive integers
        @test SHTnsKit.loggamma(1) ≈ 0.0          # log(Γ(1)) = log(0!) = 0
        @test SHTnsKit.loggamma(2) ≈ 0.0          # log(Γ(2)) = log(1!) = 0
        @test SHTnsKit.loggamma(3) ≈ log(2.0)     # log(Γ(3)) = log(2!) = log(2)
        @test SHTnsKit.loggamma(5) ≈ log(24.0)    # log(Γ(5)) = log(4!) = log(24)
    end

    @testset "loggamma domain errors" begin
        @test_throws DomainError SHTnsKit.loggamma(0)
        @test_throws DomainError SHTnsKit.loggamma(-1)
    end

    @testset "loggamma real integer-valued" begin
        # Integer-valued reals should work
        @test SHTnsKit.loggamma(5.0) ≈ log(24.0)
        @test SHTnsKit.loggamma(3.0) ≈ log(2.0)
    end

    @testset "loggamma non-integer real throws" begin
        @test_throws ArgumentError SHTnsKit.loggamma(1.5)
        @test_throws ArgumentError SHTnsKit.loggamma(0.5)
    end

    @testset "driscoll_healy_weights basic properties" begin
        for n in [4, 8, 16, 32]
            w = SHTnsKit.driscoll_healy_weights(n)
            @test length(w) == n

            # Weights should sum to approximately 2
            @test isapprox(sum(w), 2.0; rtol=1e-4)

            # First weight (north pole, j=0) should be 0
            @test w[1] ≈ 0.0 atol=1e-14

            # All weights should be non-negative
            @test all(w .>= -1e-14)
        end
    end

    @testset "driscoll_healy_weights validation" begin
        @test_throws ArgumentError SHTnsKit.driscoll_healy_weights(1)
        @test_throws ArgumentError SHTnsKit.driscoll_healy_weights(3)  # must be even
    end

    @testset "driscoll_healy_weights 4π normalization" begin
        n = 8
        w_no = SHTnsKit.driscoll_healy_weights(n; apply_4pi_normalization=false)
        w_4pi = SHTnsKit.driscoll_healy_weights(n; apply_4pi_normalization=true)
        @test isapprox(w_4pi, w_no .* sqrt(4π); rtol=1e-14)
    end
end
