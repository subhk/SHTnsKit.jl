# SHTnsKit.jl - Parallel Grid Resolution Tests
# Run all parametric tests across multiple lat/lon configurations

using Test

@testset "SHTnsKit Parallel Grid Tests" begin
    include("test_scalar_parametric.jl")
    include("test_vector_parametric.jl")
    include("test_qst_parametric.jl")
end
