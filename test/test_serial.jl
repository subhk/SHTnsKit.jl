# SHTnsKit.jl Comprehensive Single-Processor Test Suite
#
# This file runs all single-processor (non-MPI, non-GPU) tests for SHTnsKit.jl.
# Tests are organized into separate files by category in the serial/ directory.
#
# Run with: julia --project test/test_serial.jl
# Or include from runtests.jl

using Test
using SHTnsKit

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"
VERBOSE && @info "Running SHTnsKit serial test suite..."

# Get the directory containing this file
const TEST_DIR = @__DIR__

@testset "SHTnsKit Serial Tests" begin
    # Section 1: Configuration and Setup
    include(joinpath(TEST_DIR, "serial", "test_configuration.jl"))

    # Section 2: Basic Scalar Transforms
    include(joinpath(TEST_DIR, "serial", "test_basic_transforms.jl"))

    # Section 3: Truncated and Mode-Limited Transforms
    include(joinpath(TEST_DIR, "serial", "test_truncated_transforms.jl"))

    # Section 4: Vector Transforms (Spheroidal-Toroidal)
    include(joinpath(TEST_DIR, "serial", "test_vector_transforms.jl"))

    # Section 5: QST (3D Vector) Transforms
    include(joinpath(TEST_DIR, "serial", "test_qst_transforms.jl"))

    # Section 6: Batch Transforms
    include(joinpath(TEST_DIR, "serial", "test_batch_transforms.jl"))

    # Section 7: Spectral Operators
    include(joinpath(TEST_DIR, "serial", "test_operators.jl"))

    # Section 8: Rotations
    include(joinpath(TEST_DIR, "serial", "test_rotations.jl"))

    # Section 9: Energy Diagnostics
    include(joinpath(TEST_DIR, "serial", "test_energy_diagnostics.jl"))

    # Section 10: Vorticity and Enstrophy
    include(joinpath(TEST_DIR, "serial", "test_vorticity.jl"))

    # Section 11: Energy Gradients
    include(joinpath(TEST_DIR, "serial", "test_gradients.jl"))

    # Section 12: Complex and Packed Format
    include(joinpath(TEST_DIR, "serial", "test_complex_packed.jl"))

    # Section 13: Indexing Utilities
    include(joinpath(TEST_DIR, "serial", "test_indexing.jl"))
end

VERBOSE && @info "SHTnsKit serial test suite complete!"
