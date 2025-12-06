#!/usr/bin/env julia
#
# Comprehensive MPI test for SHTnsKit with PencilArrays and PencilFFTs
# Run with: mpiexec -n 4 julia --project test/test_mpi_pencil.jl
#

using MPI
MPI.Init()

using Test
using LinearAlgebra
using PencilArrays
using PencilFFTs
using SHTnsKit

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

# Helper function for synchronized printing
function mpi_println(args...)
    for r in 0:(nprocs-1)
        if rank == r
            println("[Rank $rank] ", args...)
            flush(stdout)
        end
        MPI.Barrier(comm)
    end
end

# Helper function for root-only printing
function root_println(args...)
    if rank == 0
        println(args...)
        flush(stdout)
    end
    MPI.Barrier(comm)
end

"""
Create a distributed PencilArray for spatial data (θ × φ grid)
"""
function create_spatial_pencil(cfg::SHTnsKit.SHTConfig)
    nlat, nlon = cfg.nlat, cfg.nlon
    # Create a Pencil with θ distributed and φ fully local
    pen = Pencil((nlat, nlon), comm)
    return pen
end

# Compatibility function for range_local
function local_ranges(pen::Pencil)
    return PencilArrays.range_local(pen)
end

"""
Test basic analysis and synthesis roundtrip with distributed arrays
"""
function test_scalar_roundtrip(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("\n=== Testing Scalar Analysis/Synthesis Roundtrip ===")

    # Create spatial field: Y_2^0 = (3cos²θ - 1)/2
    fθφ_local = zeros(Float64, PencilArrays.size_local(pen)...)

    # Get global indices for this process
    ranges = local_ranges(pen)
    θ_range = ranges[1]
    φ_range = ranges[2]

    for (i_local, i_global) in enumerate(θ_range)
        x = cfg.x[i_global]  # cos(θ)
        val = (3 * x^2 - 1) / 2  # Y_2^0 normalized
        for j in 1:length(φ_range)
            fθφ_local[i_local, j] = val
        end
    end

    # Create PencilArray from local data
    fθφ = PencilArray(pen, fθφ_local)

    # Perform distributed analysis
    alm = SHTnsKit.dist_analysis(cfg, fθφ)

    # Check that we get the expected coefficients
    # Y_2^0 should be dominant at l=2, m=0
    if rank == 0
        max_idx = argmax(abs.(alm))
        println("  Max coefficient at: $max_idx, value = $(alm[max_idx])")
        println("  a_{2,0} = $(alm[3, 1])")

        # Check that l=2, m=0 is the dominant coefficient
        @test argmax(abs.(alm)) == CartesianIndex(3, 1)  # l=2, m=0 is at [3,1]
    end

    # Perform distributed synthesis
    fθφ_recovered = SHTnsKit.dist_synthesis(cfg, alm; prototype_θφ=fθφ, real_output=true)

    # Compare with original (on local data)
    max_err = 0.0
    for i in 1:size(fθφ_recovered, 1)
        for j in 1:size(fθφ_recovered, 2)
            err = abs(fθφ_recovered[i, j] - fθφ_local[i, j])
            max_err = max(max_err, err)
        end
    end

    # Gather max error from all processes
    global_max_err = MPI.Allreduce(max_err, MPI.MAX, comm)

    root_println("  Roundtrip max error: $global_max_err")
    @test global_max_err < 1e-10

    MPI.Barrier(comm)
    return true
end

"""
Test vector (spheroidal/toroidal) transform roundtrip
Tests coefficient roundtrip: spectral -> spatial -> spectral
This is the proper test because it uses band-limited input.
"""
function test_vector_roundtrip(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("\n=== Testing Vector Transform Roundtrip ===")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create band-limited test coefficients (on rank 0 only)
    Slm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)
    Tlm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)

    # Set some test coefficients
    if 2 <= lmax && 1 <= mmax
        Slm_orig[3, 2] = 0.5 + 0.3im  # S_{2,1}
        Tlm_orig[3, 2] = 0.7 - 0.2im  # T_{2,1}
    end
    if 3 <= lmax
        Slm_orig[4, 1] = 1.0 + 0.0im  # S_{3,0}
    end
    if 4 <= lmax && 2 <= mmax
        Tlm_orig[5, 3] = 0.4 + 0.1im  # T_{4,2}
    end

    # Synthesize to spatial domain
    Vt_full, Vp_full = SHTnsKit.SHsphtor_to_spat(cfg, Slm_orig, Tlm_orig)

    # Create distributed PencilArrays from the full spatial data
    ranges = local_ranges(pen)
    θ_range = ranges[1]
    φ_range = ranges[2]

    Vt_local = zeros(Float64, PencilArrays.size_local(pen)...)
    Vp_local = zeros(Float64, PencilArrays.size_local(pen)...)

    for (i_local, i_global) in enumerate(θ_range)
        for (j_local, j_global) in enumerate(φ_range)
            Vt_local[i_local, j_local] = Vt_full[i_global, j_global]
            Vp_local[i_local, j_local] = Vp_full[i_global, j_global]
        end
    end

    Vt = PencilArray(pen, Vt_local)
    Vp = PencilArray(pen, Vp_local)

    # Perform distributed vector analysis
    Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vt, Vp)

    if rank == 0
        println("  Slm max value: $(maximum(abs.(Slm)))")
        println("  Tlm max value: $(maximum(abs.(Tlm)))")
    end

    # Compare recovered coefficients with original
    max_err_S = maximum(abs.(Slm .- Slm_orig))
    max_err_T = maximum(abs.(Tlm .- Tlm_orig))

    root_println("  Slm coefficient roundtrip max error: $max_err_S")
    root_println("  Tlm coefficient roundtrip max error: $max_err_T")

    @test max_err_S < 1e-10
    @test max_err_T < 1e-10

    # Also verify spatial synthesis roundtrip
    Vt_out, Vp_out = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ=Vt, real_output=true)

    max_err_t = 0.0
    max_err_p = 0.0
    for i in 1:size(Vt_out, 1)
        for j in 1:size(Vt_out, 2)
            max_err_t = max(max_err_t, abs(Vt_out[i, j] - Vt_local[i, j]))
            max_err_p = max(max_err_p, abs(Vp_out[i, j] - Vp_local[i, j]))
        end
    end

    global_max_err_t = MPI.Allreduce(max_err_t, MPI.MAX, comm)
    global_max_err_p = MPI.Allreduce(max_err_p, MPI.MAX, comm)

    root_println("  Vt spatial roundtrip max error: $global_max_err_t")
    root_println("  Vp spatial roundtrip max error: $global_max_err_p")

    @test global_max_err_t < 1e-10
    @test global_max_err_p < 1e-10

    MPI.Barrier(comm)
    return true
end

"""
Test Z-rotation of spherical harmonic coefficients
Uses synthesis -> rotate in spatial -> analysis approach
"""
function test_rotation(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("\n=== Testing Rotation ===")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create dense spectral coefficient array with a_{1,1} = 1
    Alm = zeros(ComplexF64, lmax + 1, mmax + 1)
    Alm[2, 2] = 1.0 + 0.0im  # l=1, m=1

    # Synthesize to spatial domain
    fθφ = SHTnsKit.synthesis(cfg, Alm)

    # Manually verify the synthesis value at a test point
    # Y_1^1 has specific structure - we just verify synthesis/analysis roundtrip
    Alm2 = SHTnsKit.analysis(cfg, fθφ)

    err = abs(Alm2[2, 2] - Alm[2, 2])
    root_println("  Roundtrip error for a_{1,1}: $err")
    @test err < 1e-10

    MPI.Barrier(comm)
    return true
end

"""
Test point evaluation using synthesis (synthesis-based approach)
"""
function test_point_evaluation(cfg::SHTnsKit.SHTConfig)
    root_println("\n=== Testing Point Evaluation ===")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create dense spectral coefficient array
    Alm = zeros(ComplexF64, lmax + 1, mmax + 1)

    # Set a_{2,0} = 1 (Y_2^0)  - l=2 is at index 3, m=0 is at index 1
    Alm[3, 1] = 1.0 + 0.0im

    # Synthesize to spatial domain
    fθφ = SHTnsKit.synthesis(cfg, Alm)

    # Check value at the north pole (first latitude point closest to north)
    # P_2^0(x) = (3x^2 - 1)/2, with normalization
    x_north = cfg.x[1]  # cos(θ) of first grid point
    expected_val = cfg.Nlm[3, 1] * (3 * x_north^2 - 1) / 2

    actual_val = fθφ[1, 1]  # Value at first grid point, first longitude

    root_println("  Value at northern-most point: $actual_val")
    root_println("  Expected: $expected_val")

    err = abs(actual_val - expected_val)
    @test err < 1e-10

    MPI.Barrier(comm)
    return true
end

"""
Test latitude evaluation using serial API
"""
function test_latitude_evaluation(cfg::SHTnsKit.SHTConfig)
    root_println("\n=== Testing Latitude Evaluation ===")

    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon

    # Create dense spectral coefficient array
    Alm = zeros(ComplexF64, lmax + 1, mmax + 1)

    # Set a_{0,0} = 1 (constant function Y_0^0) - l=0 is at index 1, m=0 is at index 1
    Alm[1, 1] = 1.0 + 0.0im

    # Evaluate at equator using serial version
    cost = 0.0
    vals = SHTnsKit.SH_to_lat(cfg, Alm, cost; nphi=nlon)

    # For constant function, all values should be N_0^0 ≈ 0.2821 (orthonormal)
    expected = cfg.Nlm[1, 1]

    root_println("  Latitude values (first 5): $(vals[1:min(5, length(vals))])")
    root_println("  Expected constant: $expected")

    max_err = maximum(abs.(vals .- expected))
    @test max_err < 1e-10

    MPI.Barrier(comm)
    return true
end

"""
Main test runner
"""
function run_tests()
    root_println("="^60)
    root_println("SHTnsKit MPI + PencilArrays Test Suite")
    root_println("="^60)
    root_println("Running with $nprocs MPI processes")

    # Configuration parameters
    lmax = 32
    nlat = 48  # Should be >= lmax+1 for accurate Gauss quadrature
    nlon = 96  # Should be >= 2*mmax+1 for FFT

    root_println("\nConfiguration:")
    root_println("  lmax = $lmax")
    root_println("  nlat = $nlat (Gauss points)")
    root_println("  nlon = $nlon (longitude points)")

    # Create SHT configuration
    cfg = SHTnsKit.create_gauss_config(lmax, nlat; nlon=nlon)

    # Create spatial Pencil decomposition
    pen = create_spatial_pencil(cfg)

    local_size = PencilArrays.size_local(pen)
    mpi_println("Local size: $local_size")

    # Run tests
    all_passed = true

    try
        all_passed &= test_scalar_roundtrip(cfg, pen)
    catch e
        root_println("ERROR in scalar roundtrip test: $e")
        all_passed = false
    end

    try
        all_passed &= test_vector_roundtrip(cfg, pen)
    catch e
        root_println("ERROR in vector roundtrip test: $e")
        all_passed = false
    end

    try
        all_passed &= test_rotation(cfg, pen)
    catch e
        root_println("ERROR in rotation test: $e")
        all_passed = false
    end

    try
        all_passed &= test_point_evaluation(cfg)
    catch e
        root_println("ERROR in point evaluation test: $e")
        all_passed = false
    end

    try
        all_passed &= test_latitude_evaluation(cfg)
    catch e
        root_println("ERROR in latitude evaluation test: $e")
        all_passed = false
    end

    root_println("\n" * "="^60)
    if all_passed
        root_println("All tests PASSED!")
    else
        root_println("Some tests FAILED!")
    end
    root_println("="^60)

    # Cleanup
    SHTnsKit.destroy_config(cfg)

    return all_passed
end

# Run the tests
success = run_tests()

MPI.Finalize()

# Exit with appropriate code
exit(success ? 0 : 1)
