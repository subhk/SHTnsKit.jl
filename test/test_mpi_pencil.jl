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

"""
Test basic analysis and synthesis roundtrip with distributed arrays
"""
function test_scalar_roundtrip(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("\n=== Testing Scalar Analysis/Synthesis Roundtrip ===")

    # Create spatial field: Y_2^0 = (3cos²θ - 1)/2
    fθφ_local = zeros(Float64, size_local(pen)...)

    # Get global indices for this process
    ranges = range_local(pen)
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
"""
function test_vector_roundtrip(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("\n=== Testing Vector Transform Roundtrip ===")

    # Create vector field components
    Vt_local = zeros(Float64, size_local(pen)...)
    Vp_local = zeros(Float64, size_local(pen)...)

    ranges = range_local(pen)
    θ_range = ranges[1]
    φ_range = ranges[2]

    for (i_local, i_global) in enumerate(θ_range)
        x = cfg.x[i_global]
        sθ = sqrt(max(0.0, 1 - x^2))
        for (j_local, j_global) in enumerate(φ_range)
            φ = cfg.φ[j_global]
            # Create a simple test vector field
            Vt_local[i_local, j_local] = sθ * cos(φ)
            Vp_local[i_local, j_local] = sin(φ)
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

    # Perform distributed vector synthesis
    Vt_out, Vp_out = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ=Vt, real_output=true)

    # Compare with original
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

    root_println("  Vt roundtrip max error: $global_max_err_t")
    root_println("  Vp roundtrip max error: $global_max_err_p")

    @test global_max_err_t < 1e-10
    @test global_max_err_p < 1e-10

    MPI.Barrier(comm)
    return true
end

"""
Test Z-rotation of spherical harmonic coefficients
"""
function test_rotation(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("\n=== Testing Distributed Rotation ===")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create PencilArray for spectral coefficients
    spec_pen = Pencil((lmax + 1, mmax + 1), comm)
    Alm = PencilArray{ComplexF64}(undef, spec_pen)

    # Initialize with a simple pattern: only l=1, m=1 nonzero
    fill!(parent(Alm), 0.0 + 0.0im)
    spec_ranges = range_local(spec_pen)
    l_range = spec_ranges[1]
    m_range = spec_ranges[2]

    # Set a_{1,1} = 1
    if 2 in l_range && 2 in m_range
        local_l = findfirst(==(2), l_range)
        local_m = findfirst(==(2), m_range)
        parent(Alm)[local_l, local_m] = 1.0 + 0.0im
    end

    # Test Z-rotation by π/4
    alpha = π / 4
    R_p = similar(Alm)
    SHTnsKit.dist_SH_Zrotate(cfg, Alm, alpha, R_p)

    # Z-rotation should multiply by exp(i*m*alpha)
    expected_phase = exp(1im * 1 * alpha)  # m=1

    if 2 in l_range && 2 in m_range
        local_l = findfirst(==(2), l_range)
        local_m = findfirst(==(2), m_range)
        actual = parent(R_p)[local_l, local_m]
        expected = expected_phase
        err = abs(actual - expected)
        println("  [Rank $rank] Z-rotation error: $err")
        @test err < 1e-14
    end

    MPI.Barrier(comm)
    return true
end

"""
Test point evaluation from distributed coefficients
"""
function test_point_evaluation(cfg::SHTnsKit.SHTConfig)
    root_println("\n=== Testing Point Evaluation ===")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create spectral coefficient array
    spec_pen = Pencil((lmax + 1, mmax + 1), comm)
    Alm_p = PencilArray{ComplexF64}(undef, spec_pen)

    # Set a_{2,0} = 1 (Y_2^0)
    fill!(parent(Alm_p), 0.0 + 0.0im)
    spec_ranges = range_local(spec_pen)
    l_range = spec_ranges[1]
    m_range = spec_ranges[2]

    if 3 in l_range && 1 in m_range  # l=2 is at index 3, m=0 is at index 1
        local_l = findfirst(==(3), l_range)
        local_m = findfirst(==(1), m_range)
        parent(Alm_p)[local_l, local_m] = 1.0 + 0.0im
    end

    # Evaluate at the north pole (θ=0, x=1)
    cost = 1.0
    phi = 0.0

    val = SHTnsKit.dist_SH_to_point(cfg, Alm_p, cost, phi)

    # Y_2^0(θ=0) = N * (3*1 - 1)/2 = N where N is normalization
    # With orthonormal normalization: N_2^0 ≈ 0.6307
    if rank == 0
        println("  Value at north pole: $val")
        println("  Expected: $(cfg.Nlm[3, 1] * 1.0)")  # P_2^0(1) = 1
    end

    @test abs(val - cfg.Nlm[3, 1]) < 1e-10

    MPI.Barrier(comm)
    return true
end

"""
Test latitude evaluation from distributed coefficients
"""
function test_latitude_evaluation(cfg::SHTnsKit.SHTConfig)
    root_println("\n=== Testing Latitude Evaluation ===")

    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon

    # Create spectral coefficient array
    spec_pen = Pencil((lmax + 1, mmax + 1), comm)
    Alm_p = PencilArray{ComplexF64}(undef, spec_pen)

    # Set a_{0,0} = 1 (constant function Y_0^0)
    fill!(parent(Alm_p), 0.0 + 0.0im)
    spec_ranges = range_local(spec_pen)
    l_range = spec_ranges[1]
    m_range = spec_ranges[2]

    if 1 in l_range && 1 in m_range  # l=0 is at index 1, m=0 is at index 1
        local_l = findfirst(==(1), l_range)
        local_m = findfirst(==(1), m_range)
        parent(Alm_p)[local_l, local_m] = 1.0 + 0.0im
    end

    # Evaluate at equator
    cost = 0.0
    vals = SHTnsKit.dist_SH_to_lat(cfg, Alm_p, cost; nphi=nlon)

    # For constant function, all values should be N_0^0 ≈ 0.2821 (orthonormal)
    expected = cfg.Nlm[1, 1]

    if rank == 0
        println("  Latitude values (first 5): $(vals[1:min(5, length(vals))])")
        println("  Expected constant: $expected")
    end

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

    local_size = size_local(pen)
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
