#!/usr/bin/env julia
#
# Comprehensive MPI test for SHTnsKit with PencilArrays and PencilFFTs
# Tests all features across multiple grid configurations
# Run with: mpiexec -n 4 julia --project test/parallel/test_mpi_comprehensive.jl
#

using MPI
MPI.Init()

using Test
using LinearAlgebra
using Random
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

# Compatibility function for range_local
function local_ranges(pen::Pencil)
    return PencilArrays.range_local(pen)
end

"""
Create a distributed PencilArray for spatial data
"""
function create_spatial_pencil(cfg::SHTnsKit.SHTConfig)
    nlat, nlon = cfg.nlat, cfg.nlon
    pen = Pencil((nlat, nlon), comm)
    return pen
end

# ======================= SCALAR TRANSFORM TESTS =======================

"""
Test scalar analysis/synthesis roundtrip
"""
function test_scalar_roundtrip(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing scalar analysis/synthesis roundtrip...")

    # Create spatial field: Y_2^0 = (3cos^2 theta - 1)/2
    flocal = zeros(Float64, PencilArrays.size_local(pen)...)
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]

    for (i_local, i_global) in enumerate(theta_range)
        x = cfg.x[i_global]
        val = (3 * x^2 - 1) / 2
        for j in 1:length(phi_range)
            flocal[i_local, j] = val
        end
    end

    f_pa = PencilArray(pen, flocal)

    # Analysis
    alm = SHTnsKit.dist_analysis(cfg, f_pa)

    # Synthesis
    f_rec = SHTnsKit.dist_synthesis(cfg, alm; prototype_θφ=f_pa, real_output=true)

    # Compare
    max_err = 0.0
    for i in 1:size(f_rec, 1)
        for j in 1:size(f_rec, 2)
            max_err = max(max_err, abs(f_rec[i, j] - flocal[i, j]))
        end
    end
    global_max_err = MPI.Allreduce(max_err, MPI.MAX, comm)

    @test global_max_err < 1e-10
    return global_max_err
end

"""
Test scalar transforms with random coefficients
"""
function test_scalar_random_roundtrip(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing scalar transforms with random coefficients...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create band-limited coefficients on rank 0
    rng = MersenneTwister(42)
    alm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)

    if rank == 0
        for m in 0:mmax
            for l in m:lmax
                alm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
            end
        end
        alm_orig[:, 1] .= real.(alm_orig[:, 1])
    end

    # Broadcast to all ranks
    MPI.Bcast!(alm_orig, 0, comm)

    # Synthesize to spatial domain
    f_full = SHTnsKit.synthesis(cfg, alm_orig; real_output=true)

    # Create distributed PencilArray
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]
    flocal = zeros(Float64, PencilArrays.size_local(pen)...)

    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            flocal[i_local, j_local] = f_full[i_global, j_global]
        end
    end

    f_pa = PencilArray(pen, flocal)

    # Distributed analysis
    alm_rec = SHTnsKit.dist_analysis(cfg, f_pa)

    # Compare coefficients
    max_err = maximum(abs.(alm_rec - alm_orig))
    @test max_err < 1e-10
    return max_err
end

# ======================= VECTOR TRANSFORM TESTS =======================

"""
Test vector (spheroidal/toroidal) transform roundtrip
"""
function test_vector_roundtrip(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing vector sphtor analysis/synthesis roundtrip...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create band-limited test coefficients
    rng = MersenneTwister(100)
    Slm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)
    Tlm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)

    for m in 0:mmax
        for l in max(1, m):lmax
            Slm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
            Tlm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
        end
    end
    Slm_orig[:, 1] .= real.(Slm_orig[:, 1])
    Tlm_orig[:, 1] .= real.(Tlm_orig[:, 1])

    # Synthesize to spatial domain
    Vt_full, Vp_full = SHTnsKit.synthesis_sphtor(cfg, Slm_orig, Tlm_orig; real_output=true)

    # Create distributed PencilArrays
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]

    Vt_local = zeros(Float64, PencilArrays.size_local(pen)...)
    Vp_local = zeros(Float64, PencilArrays.size_local(pen)...)

    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            Vt_local[i_local, j_local] = Vt_full[i_global, j_global]
            Vp_local[i_local, j_local] = Vp_full[i_global, j_global]
        end
    end

    Vt = PencilArray(pen, Vt_local)
    Vp = PencilArray(pen, Vp_local)

    # Distributed vector analysis
    Slm_rec, Tlm_rec = SHTnsKit.dist_analysis_sphtor(cfg, Vt, Vp)

    # Compare coefficients
    max_err_S = maximum(abs.(Slm_rec - Slm_orig))
    max_err_T = maximum(abs.(Tlm_rec - Tlm_orig))

    @test max_err_S < 1e-9
    @test max_err_T < 1e-9

    # Also test synthesis roundtrip
    Vt_out, Vp_out = SHTnsKit.dist_synthesis_sphtor(cfg, Slm_rec, Tlm_rec; prototype_θφ=Vt, real_output=true)

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

    @test global_max_err_t < 1e-10
    @test global_max_err_p < 1e-10

    return (max_err_S, max_err_T, global_max_err_t, global_max_err_p)
end

# ======================= QST (3D VECTOR) TRANSFORM TESTS =======================

"""
Test QST (3D vector) transform roundtrip
"""
function test_qst_roundtrip(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing QST (3D vector) analysis/synthesis roundtrip...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create band-limited QST coefficients
    rng = MersenneTwister(200)
    Qlm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)
    Slm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)
    Tlm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)

    for m in 0:mmax
        for l in m:lmax
            Qlm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
        end
        for l in max(1, m):lmax
            Slm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
            Tlm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
        end
    end
    Qlm_orig[:, 1] .= real.(Qlm_orig[:, 1])
    Slm_orig[:, 1] .= real.(Slm_orig[:, 1])
    Tlm_orig[:, 1] .= real.(Tlm_orig[:, 1])

    # Synthesize to spatial domain
    Vr_full, Vt_full, Vp_full = SHTnsKit.synthesis_qst(cfg, Qlm_orig, Slm_orig, Tlm_orig; real_output=true)

    # Create distributed PencilArrays
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]

    Vr_local = zeros(Float64, PencilArrays.size_local(pen)...)
    Vt_local = zeros(Float64, PencilArrays.size_local(pen)...)
    Vp_local = zeros(Float64, PencilArrays.size_local(pen)...)

    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            Vr_local[i_local, j_local] = Vr_full[i_global, j_global]
            Vt_local[i_local, j_local] = Vt_full[i_global, j_global]
            Vp_local[i_local, j_local] = Vp_full[i_global, j_global]
        end
    end

    Vr = PencilArray(pen, Vr_local)
    Vt = PencilArray(pen, Vt_local)
    Vp = PencilArray(pen, Vp_local)

    # Distributed QST analysis
    Qlm_rec, Slm_rec, Tlm_rec = SHTnsKit.dist_analysis_qst(cfg, Vr, Vt, Vp)

    # Compare coefficients
    max_err_Q = maximum(abs.(Qlm_rec - Qlm_orig))
    max_err_S = maximum(abs.(Slm_rec - Slm_orig))
    max_err_T = maximum(abs.(Tlm_rec - Tlm_orig))

    @test max_err_Q < 1e-9
    @test max_err_S < 1e-9
    @test max_err_T < 1e-9

    return (max_err_Q, max_err_S, max_err_T)
end

# ======================= ENERGY DIAGNOSTIC TESTS =======================

"""
Test Parseval's identity for distributed scalar fields
"""
function test_scalar_parseval(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing scalar Parseval's identity...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create random coefficients
    rng = MersenneTwister(300)
    alm = randn(rng, ComplexF64, lmax + 1, mmax + 1)
    alm[:, 1] .= real.(alm[:, 1])
    for m in 0:mmax, l in 0:(m-1)
        alm[l+1, m+1] = 0
    end

    # Synthesize to spatial
    f = SHTnsKit.synthesis(cfg, alm; real_output=true)

    # Create distributed array
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]
    flocal = zeros(Float64, PencilArrays.size_local(pen)...)

    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            flocal[i_local, j_local] = f[i_global, j_global]
        end
    end

    f_pa = PencilArray(pen, flocal)

    # Compute energies
    E_spec = SHTnsKit.energy_scalar(cfg, alm)
    E_grid = SHTnsKit.grid_energy_scalar(cfg, f_pa)

    rel_err = abs(E_spec - E_grid) / (abs(E_grid) + eps())
    @test rel_err < 1e-9

    return (E_spec, E_grid, rel_err)
end

"""
Test Parseval's identity for distributed vector fields
"""
function test_vector_parseval(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing vector Parseval's identity...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create random coefficients
    rng = MersenneTwister(400)
    Slm = zeros(ComplexF64, lmax + 1, mmax + 1)
    Tlm = zeros(ComplexF64, lmax + 1, mmax + 1)

    for m in 0:mmax
        for l in max(1, m):lmax
            Slm[l+1, m+1] = randn(rng) + im * randn(rng)
            Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
        end
    end
    Slm[:, 1] .= real.(Slm[:, 1])
    Tlm[:, 1] .= real.(Tlm[:, 1])

    # Synthesize to spatial
    Vt, Vp = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm; real_output=true)

    # Create distributed arrays
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]

    Vt_local = zeros(Float64, PencilArrays.size_local(pen)...)
    Vp_local = zeros(Float64, PencilArrays.size_local(pen)...)

    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            Vt_local[i_local, j_local] = Vt[i_global, j_global]
            Vp_local[i_local, j_local] = Vp[i_global, j_global]
        end
    end

    Vt_pa = PencilArray(pen, Vt_local)
    Vp_pa = PencilArray(pen, Vp_local)

    # Compute energies
    E_spec = SHTnsKit.energy_vector(cfg, Slm, Tlm)
    E_grid = SHTnsKit.grid_energy_vector(cfg, Vt_pa, Vp_pa)

    rel_err = abs(E_spec - E_grid) / (abs(E_grid) + eps())
    @test rel_err < 1e-8

    return (E_spec, E_grid, rel_err)
end

# ======================= DIVERGENCE/VORTICITY TESTS =======================

"""
Test divergence and vorticity operations
"""
function test_div_vort(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing divergence/vorticity operations...")

    lmax, mmax = cfg.lmax, cfg.mmax

    rng = MersenneTwister(500)
    Slm = zeros(ComplexF64, lmax + 1, mmax + 1)
    Tlm = zeros(ComplexF64, lmax + 1, mmax + 1)

    for m in 0:mmax
        for l in max(1, m):lmax
            Slm[l+1, m+1] = randn(rng) + im * randn(rng)
            Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
        end
    end
    Slm[:, 1] .= real.(Slm[:, 1])
    Tlm[:, 1] .= real.(Tlm[:, 1])

    # Compute divergence from spheroidal
    div_lm = SHTnsKit.divergence_from_spheroidal(cfg, Slm)

    # Recover spheroidal from divergence
    Slm_rec = SHTnsKit.spheroidal_from_divergence(cfg, div_lm)

    @test isapprox(Slm_rec, Slm; rtol=1e-9, atol=1e-11)

    # Compute vorticity from toroidal
    vort_lm = SHTnsKit.vorticity_from_toroidal(cfg, Tlm)

    # Recover toroidal from vorticity
    Tlm_rec = SHTnsKit.toroidal_from_vorticity(cfg, vort_lm)

    @test isapprox(Tlm_rec, Tlm; rtol=1e-9, atol=1e-11)

    return true
end

# ======================= ROTATION TESTS =======================

"""
Test Z-rotation of spherical harmonic coefficients
"""
function test_z_rotation(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing Z-rotation...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create coefficients
    alm = zeros(ComplexF64, lmax + 1, mmax + 1)
    alm[2, 2] = 1.0 + 0.0im  # l=1, m=1

    alpha = 0.5

    # Z-rotation (use dist_SH_Zrotate for matrix form)
    alm_rot = similar(alm)
    SHTnsKit.dist_SH_Zrotate(cfg, alm, alpha, alm_rot)

    # For Z-rotation, |coefficient| should be preserved
    @test isapprox(abs(alm_rot[2, 2]), abs(alm[2, 2]); rtol=1e-10)

    return true
end

# ======================= MULTI-RESOLUTION TESTS =======================

"""
Run all tests for a given grid configuration
"""
function run_tests_for_config(lmax::Int, nlat::Int, nlon::Int)
    root_println("\n" * "="^60)
    root_println("Testing configuration: lmax=$lmax, nlat=$nlat, nlon=$nlon")
    root_println("="^60)

    cfg = SHTnsKit.create_gauss_config(lmax, nlat; nlon=nlon)
    pen = create_spatial_pencil(cfg)

    local_size = PencilArrays.size_local(pen)
    root_println("Local size on rank 0: $local_size")

    all_passed = true

    # Scalar tests
    try
        test_scalar_roundtrip(cfg, pen)
        root_println("    [PASS] Scalar roundtrip")
    catch e
        root_println("    [FAIL] Scalar roundtrip: $e")
        all_passed = false
    end

    try
        test_scalar_random_roundtrip(cfg, pen)
        root_println("    [PASS] Scalar random roundtrip")
    catch e
        root_println("    [FAIL] Scalar random roundtrip: $e")
        all_passed = false
    end

    # Vector tests
    try
        test_vector_roundtrip(cfg, pen)
        root_println("    [PASS] Vector sphtor roundtrip")
    catch e
        root_println("    [FAIL] Vector sphtor roundtrip: $e")
        all_passed = false
    end

    # QST tests (only for larger grids)
    if lmax >= 4
        try
            test_qst_roundtrip(cfg, pen)
            root_println("    [PASS] QST roundtrip")
        catch e
            root_println("    [FAIL] QST roundtrip: $e")
            all_passed = false
        end
    end

    # Energy tests
    try
        test_scalar_parseval(cfg, pen)
        root_println("    [PASS] Scalar Parseval")
    catch e
        root_println("    [FAIL] Scalar Parseval: $e")
        all_passed = false
    end

    try
        test_vector_parseval(cfg, pen)
        root_println("    [PASS] Vector Parseval")
    catch e
        root_println("    [FAIL] Vector Parseval: $e")
        all_passed = false
    end

    # Divergence/vorticity
    try
        test_div_vort(cfg, pen)
        root_println("    [PASS] Divergence/vorticity")
    catch e
        root_println("    [FAIL] Divergence/vorticity: $e")
        all_passed = false
    end

    # Rotation
    try
        test_z_rotation(cfg, pen)
        root_println("    [PASS] Z-rotation")
    catch e
        root_println("    [FAIL] Z-rotation: $e")
        all_passed = false
    end

    SHTnsKit.destroy_config(cfg)

    return all_passed
end

"""
Main test runner
"""
function run_all_tests()
    root_println("="^60)
    root_println("SHTnsKit MPI Comprehensive Test Suite")
    root_println("="^60)
    root_println("Running with $nprocs MPI processes")

    # Test multiple grid configurations
    grid_configs = [
        (lmax=8,  nlat=12, nlon=17),   # Small
        (lmax=16, nlat=20, nlon=33),   # Medium
        (lmax=32, nlat=48, nlon=65),   # Large
    ]

    all_passed = true

    for (lmax, nlat, nlon) in grid_configs
        passed = run_tests_for_config(lmax, nlat, nlon)
        all_passed = all_passed && passed
    end

    root_println("\n" * "="^60)
    if all_passed
        root_println("All tests PASSED!")
    else
        root_println("Some tests FAILED!")
    end
    root_println("="^60)

    return all_passed
end

# Run the tests
success = run_all_tests()

MPI.Finalize()

# Exit with appropriate code
exit(success ? 0 : 1)
