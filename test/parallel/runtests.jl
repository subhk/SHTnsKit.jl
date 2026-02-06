# SHTnsKit.jl - Parallel Grid Resolution Tests
# Run all parametric tests across multiple lat/lon configurations
#
# This folder contains:
# - test_scalar_parametric.jl  : Scalar transforms across multiple grid sizes
# - test_vector_parametric.jl  : Vector (sphtor) transforms across multiple grid sizes
# - test_qst_parametric.jl     : QST (3D vector) transforms across multiple grid sizes
# - test_packed_storage.jl     : Non-MPI tests for packed storage utilities
# - test_mpi_comprehensive.jl  : MPI distributed tests (run separately with mpiexec)
#
# To run MPI tests:
#   mpiexec -n 4 julia --project test/parallel/test_mpi_comprehensive.jl

using Test

@testset "SHTnsKit Parallel Grid Tests" begin
    include("test_scalar_parametric.jl")
    include("test_vector_parametric.jl")
    include("test_qst_parametric.jl")
    include("test_packed_storage.jl")
end

# Note: MPI tests are not included here as they require mpiexec to run.
# Run them separately with: mpiexec -n 4 julia --project test/parallel/test_mpi_comprehensive.jl
