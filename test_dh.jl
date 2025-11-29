# Test script for Driscoll-Healy weights implementation
using Test

# Include the mathutils file to test the DH weight function
include("src/mathutils.jl")

# Test the DH weights function
@testset "Driscoll-Healy Weights" begin
    # Test with n=4
    w4 = driscoll_healy_weights(4)
    println("Weights for n=4: ", w4)
    @test length(w4) == 4
    @test w4[1] ≈ 0.0  # North pole should be zero
    @test w4[end] ≈ 0.0  # South pole should be zero
    @test all(w4 .>= 0.0)  # All weights should be non-negative

    # Test with n=18 (for lmax=8)
    lmax = 8
    n = 2*(lmax + 1)
    w = driscoll_healy_weights(n)
    println("Weights for n=$n (lmax=$lmax): ", w)
    @test length(w) == n
    @test w[1] ≈ 0.0
    @test w[end] ≈ 0.0

    # Test that odd n throws an error
    @test_throws ArgumentError driscoll_healy_weights(5)

    # Test that n < 2 throws an error
    @test_throws ArgumentError driscoll_healy_weights(1)
end

println("\nAll tests passed!")
