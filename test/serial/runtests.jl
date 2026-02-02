# SHTnsKit.jl - Serial Test Runner
# Run all serial (single-processor) tests

using Test

@testset "SHTnsKit Serial Tests" begin
    include("test_configuration.jl")
    include("test_indexing.jl")
    include("test_basic_transforms.jl")
    include("test_truncated_transforms.jl")
    include("test_vector_transforms.jl")
    include("test_qst_transforms.jl")
    include("test_batch_transforms.jl")
    include("test_operators.jl")
    include("test_rotations.jl")
    include("test_energy_diagnostics.jl")
    include("test_vorticity.jl")
    include("test_gradients.jl")
    include("test_complex_packed.jl")
end
