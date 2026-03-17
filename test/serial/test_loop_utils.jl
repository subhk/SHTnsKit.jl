# SHTnsKit.jl - Loop Utilities Tests
# Tests for @sht_loop macro, loop backend, helper functions

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Loop Utilities" begin
    @testset "loop_backend" begin
        @test SHTnsKit.loop_backend() isa String
        @test SHTnsKit.loop_backend() in ("auto", "SIMD")
    end

    @testset "set_loop_backend" begin
        old = SHTnsKit.loop_backend()

        SHTnsKit.set_loop_backend("SIMD")
        @test SHTnsKit.loop_backend() == "SIMD"

        SHTnsKit.set_loop_backend("auto")
        @test SHTnsKit.loop_backend() == "auto"

        @test_throws ArgumentError SHTnsKit.set_loop_backend("invalid")

        SHTnsKit.set_loop_backend(old)
    end

    @testset "_is_cpu_array" begin
        @test SHTnsKit._is_cpu_array(rand(5)) == true
        @test SHTnsKit._is_cpu_array(zeros(3, 4)) == true
        @test SHTnsKit._is_cpu_array(ones(ComplexF64, 2, 3)) == true
    end

    @testset "_is_pencil_array" begin
        # Regular arrays should not be PencilArrays
        @test SHTnsKit._is_pencil_array(rand(5)) == false
        @test SHTnsKit._is_pencil_array(zeros(3, 4)) == false
    end

    @testset "_get_local_data" begin
        arr = rand(5)
        @test SHTnsKit._get_local_data(arr) === arr
    end

    @testset "spectral_range" begin
        r = SHTnsKit.spectral_range(4, 4)
        @test r isa CartesianIndices
        @test size(r) == (5, 5)
    end

    @testset "spatial_range" begin
        r = SHTnsKit.spatial_range(8, 16)
        @test r isa CartesianIndices
        @test size(r) == (8, 16)
    end

    @testset "latitude_range" begin
        r = SHTnsKit.latitude_range(10)
        @test r == 1:10
    end

    @testset "mode_range" begin
        r = SHTnsKit.mode_range(5)
        @test r == 0:5
    end

    @testset "local_range" begin
        arr = rand(4, 8)
        r = SHTnsKit.local_range(arr)
        @test r == CartesianIndices(arr)
    end

    @testset "local_size" begin
        arr = rand(4, 8)
        @test SHTnsKit.local_size(arr) == (4, 8)
    end

    @testset "CI shorthand" begin
        idx = SHTnsKit.CI(2, 3)
        @test idx == CartesianIndex(2, 3)
    end

    @testset "δ unit index" begin
        d1 = SHTnsKit.δ(1, CartesianIndex(2, 3))
        @test d1 == CartesianIndex(1, 0)

        d2 = SHTnsKit.δ(2, CartesianIndex(2, 3))
        @test d2 == CartesianIndex(0, 1)
    end

    @testset "inside" begin
        arr = zeros(6, 8)
        r = SHTnsKit.inside(arr)
        # Default buff=1, so interior is [2:5, 2:7]
        @test size(r) == (4, 6)

        r2 = SHTnsKit.inside(arr; buff=2)
        @test size(r2) == (2, 4)
    end

    @testset "_loop_index_symbols" begin
        @test SHTnsKit._loop_index_symbols(:I) == [:I]
        @test SHTnsKit._loop_index_symbols(Expr(:tuple, :i, :j)) == [:i, :j]
    end

    @testset "@sht_loop basic" begin
        # Test that @sht_loop writes correct values
        dest = zeros(4, 8)
        src = rand(4, 8)
        SHTnsKit.@sht_loop dest[I] = src[I] over I ∈ CartesianIndices(dest)
        @test dest ≈ src
    end
end
