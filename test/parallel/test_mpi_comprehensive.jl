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

# Get the parallel extension module for functions only exported there
const ParExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)

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

# ======================= PLAN-BASED TRANSFORM TESTS =======================

"""
Test DistAnalysisPlan + dist_analysis! (in-place scalar analysis)
"""
function test_plan_analysis(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing DistAnalysisPlan + dist_analysis! ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create band-limited test coefficients
    rng = MersenneTwister(600)
    alm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        alm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    alm_orig[:, 1] .= real.(alm_orig[:, 1])
    MPI.Bcast!(alm_orig, 0, comm)

    # Synthesize to get spatial field
    f_full = SHTnsKit.synthesis(cfg, alm_orig; real_output=true)

    # Distribute
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]
    flocal = zeros(Float64, PencilArrays.size_local(pen)...)
    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            flocal[i_local, j_local] = f_full[i_global, j_global]
        end
    end
    f_pa = PencilArray(pen, flocal)

    # Create plan and run in-place analysis
    plan = SHTnsKit.DistAnalysisPlan(cfg, f_pa)
    Alm_out = zeros(ComplexF64, lmax + 1, mmax + 1)
    SHTnsKit.dist_analysis!(plan, Alm_out, f_pa)

    # Compare with direct analysis
    Alm_direct = SHTnsKit.dist_analysis(cfg, f_pa)

    max_err = maximum(abs.(Alm_out - Alm_direct))
    @test max_err < 1e-12
    return max_err
end

"""
Test DistPlan + dist_synthesis! (in-place scalar synthesis)
"""
function test_plan_synthesis(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing DistPlan + dist_synthesis! ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create test coefficients on a spectral pencil
    rng = MersenneTwister(601)
    alm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        alm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    alm_orig[:, 1] .= real.(alm_orig[:, 1])
    MPI.Bcast!(alm_orig, 0, comm)

    # Create a spatial prototype
    f_full = SHTnsKit.synthesis(cfg, alm_orig; real_output=true)
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]
    flocal = zeros(Float64, PencilArrays.size_local(pen)...)
    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            flocal[i_local, j_local] = f_full[i_global, j_global]
        end
    end
    f_pa = PencilArray(pen, flocal)

    # Create spectral pencil for Alm
    spec_pen = SHTnsKit.create_spectral_pencil(cfg; comm=comm)
    Alm_pencil = PencilArray{ComplexF64}(undef, spec_pen)
    # Fill in spectral data
    for (ii, il) in enumerate(axes(Alm_pencil, 1))
        for (jj, jm) in enumerate(axes(Alm_pencil, 2))
            lval = globalindices(Alm_pencil, 1)[ii] - 1
            mval = globalindices(Alm_pencil, 2)[jj] - 1
            if lval >= mval && mval <= mmax && lval <= lmax
                Alm_pencil[il, jm] = alm_orig[lval+1, mval+1]
            else
                Alm_pencil[il, jm] = 0.0 + 0.0im
            end
        end
    end

    # Create plan and run in-place synthesis
    plan = SHTnsKit.DistPlan(cfg, f_pa)
    fθφ_out = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))
    SHTnsKit.dist_synthesis!(plan, fθφ_out, Alm_pencil; real_output=true)

    # Compare with reference spatial data
    max_err = 0.0
    for i in 1:size(fθφ_out, 1)
        for j in 1:size(fθφ_out, 2)
            max_err = max(max_err, abs(fθφ_out[i, j] - flocal[i, j]))
        end
    end
    global_max_err = MPI.Allreduce(max_err, MPI.MAX, comm)
    @test global_max_err < 1e-10
    return global_max_err
end

"""
Test DistSphtorPlan + dist_analysis_sphtor! / dist_synthesis_sphtor! (in-place vector plan transforms)
"""
function test_plan_sphtor(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing DistSphtorPlan in-place vector transforms ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create test coefficients
    rng = MersenneTwister(602)
    Slm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)
    Tlm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in max(1, m):lmax
        Slm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
        Tlm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    Slm_orig[:, 1] .= real.(Slm_orig[:, 1])
    Tlm_orig[:, 1] .= real.(Tlm_orig[:, 1])

    # Synthesize to spatial
    Vt_full, Vp_full = SHTnsKit.synthesis_sphtor(cfg, Slm_orig, Tlm_orig; real_output=true)

    # Distribute
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
    Vt_pa = PencilArray(pen, Vt_local)
    Vp_pa = PencilArray(pen, Vp_local)

    # Create sphtor plan and test in-place analysis
    plan = SHTnsKit.DistSphtorPlan(cfg, Vt_pa)
    Slm_out = zeros(ComplexF64, lmax + 1, mmax + 1)
    Tlm_out = zeros(ComplexF64, lmax + 1, mmax + 1)
    SHTnsKit.dist_analysis_sphtor!(plan, Slm_out, Tlm_out, Vt_pa, Vp_pa)

    max_err_S = maximum(abs.(Slm_out - Slm_orig))
    max_err_T = maximum(abs.(Tlm_out - Tlm_orig))
    @test max_err_S < 1e-9
    @test max_err_T < 1e-9

    # Test in-place synthesis roundtrip
    Vt_out = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))
    Vp_out = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))
    SHTnsKit.dist_synthesis_sphtor!(plan, Vt_out, Vp_out, Slm_out, Tlm_out; real_output=true)

    max_err_vt = 0.0
    max_err_vp = 0.0
    for i in 1:size(Vt_out, 1)
        for j in 1:size(Vt_out, 2)
            max_err_vt = max(max_err_vt, abs(Vt_out[i, j] - Vt_local[i, j]))
            max_err_vp = max(max_err_vp, abs(Vp_out[i, j] - Vp_local[i, j]))
        end
    end
    global_max_err_vt = MPI.Allreduce(max_err_vt, MPI.MAX, comm)
    global_max_err_vp = MPI.Allreduce(max_err_vp, MPI.MAX, comm)
    @test global_max_err_vt < 1e-10
    @test global_max_err_vp < 1e-10

    return (max_err_S, max_err_T, global_max_err_vt, global_max_err_vp)
end

# ======================= SCALAR LAPLACIAN TESTS =======================

"""
Test dist_scalar_laplacian: apply Laplacian to a known field
Y_l^m should give -l(l+1)*Y_l^m after Laplacian.
"""
function test_scalar_laplacian(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing dist_scalar_laplacian ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create coefficients with a single mode Y_l^m for known eigenvalue
    rng = MersenneTwister(700)
    alm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        alm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    alm_orig[:, 1] .= real.(alm_orig[:, 1])
    MPI.Bcast!(alm_orig, 0, comm)

    # Synthesize to spatial
    f_full = SHTnsKit.synthesis(cfg, alm_orig; real_output=true)

    # Distribute
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]
    flocal = zeros(Float64, PencilArrays.size_local(pen)...)
    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            flocal[i_local, j_local] = f_full[i_global, j_global]
        end
    end
    f_pa = PencilArray(pen, flocal)

    # Distributed Laplacian
    lap_dist = SHTnsKit.dist_scalar_laplacian(cfg, f_pa; prototype_θφ=f_pa, real_output=true)

    # Serial Laplacian: apply -l(l+1) to coefficients then synthesize
    alm_lap = copy(alm_orig)
    SHTnsKit.dist_apply_laplacian!(cfg, alm_lap)
    lap_serial_full = SHTnsKit.synthesis(cfg, alm_lap; real_output=true)

    # Compare
    max_err = 0.0
    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            max_err = max(max_err, abs(lap_dist[i_local, j_local] - lap_serial_full[i_global, j_global]))
        end
    end
    global_max_err = MPI.Allreduce(max_err, MPI.MAX, comm)
    @test global_max_err < 1e-9
    return global_max_err
end

# ======================= SPATIAL DIVERGENCE/VORTICITY TESTS =======================

"""
Test dist_spatial_divergence: compare with serial divergence_from_spheroidal
"""
function test_spatial_divergence(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing dist_spatial_divergence ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create vector field coefficients
    rng = MersenneTwister(800)
    Slm = zeros(ComplexF64, lmax + 1, mmax + 1)
    Tlm = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in max(1, m):lmax
        Slm[l+1, m+1] = randn(rng) + im * randn(rng)
        Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    Slm[:, 1] .= real.(Slm[:, 1])
    Tlm[:, 1] .= real.(Tlm[:, 1])

    # Synthesize to spatial
    Vt_full, Vp_full = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm; real_output=true)

    # Distribute
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
    Vt_pa = PencilArray(pen, Vt_local)
    Vp_pa = PencilArray(pen, Vp_local)

    # Distributed divergence
    div_dist = SHTnsKit.dist_spatial_divergence(cfg, Vt_pa, Vp_pa; prototype_θφ=Vt_pa, real_output=true)

    # Serial divergence: divergence_from_spheroidal then synthesize
    div_lm = SHTnsKit.divergence_from_spheroidal(cfg, Slm)
    div_serial_full = SHTnsKit.synthesis(cfg, div_lm; real_output=true)

    # Compare
    max_err = 0.0
    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            max_err = max(max_err, abs(div_dist[i_local, j_local] - div_serial_full[i_global, j_global]))
        end
    end
    global_max_err = MPI.Allreduce(max_err, MPI.MAX, comm)
    @test global_max_err < 1e-9
    return global_max_err
end

"""
Test dist_spatial_vorticity: compare with serial vorticity_from_toroidal
"""
function test_spatial_vorticity(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing dist_spatial_vorticity ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create vector field coefficients
    rng = MersenneTwister(801)
    Slm = zeros(ComplexF64, lmax + 1, mmax + 1)
    Tlm = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in max(1, m):lmax
        Slm[l+1, m+1] = randn(rng) + im * randn(rng)
        Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    Slm[:, 1] .= real.(Slm[:, 1])
    Tlm[:, 1] .= real.(Tlm[:, 1])

    # Synthesize to spatial
    Vt_full, Vp_full = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm; real_output=true)

    # Distribute
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
    Vt_pa = PencilArray(pen, Vt_local)
    Vp_pa = PencilArray(pen, Vp_local)

    # Distributed vorticity
    vort_dist = SHTnsKit.dist_spatial_vorticity(cfg, Vt_pa, Vp_pa; prototype_θφ=Vt_pa, real_output=true)

    # Serial vorticity: vorticity_from_toroidal then synthesize
    vort_lm = SHTnsKit.vorticity_from_toroidal(cfg, Tlm)
    vort_serial_full = SHTnsKit.synthesis(cfg, vort_lm; real_output=true)

    # Compare
    max_err = 0.0
    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            max_err = max(max_err, abs(vort_dist[i_local, j_local] - vort_serial_full[i_global, j_global]))
        end
    end
    global_max_err = MPI.Allreduce(max_err, MPI.MAX, comm)
    @test global_max_err < 1e-9
    return global_max_err
end

# ======================= POINT/LATITUDE EVALUATION TESTS =======================

"""
Test dist_SH_to_point: evaluate at a random point, compare with serial synthesis_point
"""
function test_point_evaluation(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing dist_SH_to_point ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create coefficients
    rng = MersenneTwister(900)
    alm = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        alm[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    alm[:, 1] .= real.(alm[:, 1])
    MPI.Bcast!(alm, 0, comm)

    # Create distributed spectral PencilArray
    spec_pen = SHTnsKit.create_spectral_pencil(cfg; comm=comm)
    Alm_pencil = PencilArray{ComplexF64}(undef, spec_pen)
    for (ii, il) in enumerate(axes(Alm_pencil, 1))
        for (jj, jm) in enumerate(axes(Alm_pencil, 2))
            lval = globalindices(Alm_pencil, 1)[ii] - 1
            mval = globalindices(Alm_pencil, 2)[jj] - 1
            if lval >= mval && mval <= mmax && lval <= lmax
                Alm_pencil[il, jm] = alm[lval+1, mval+1]
            else
                Alm_pencil[il, jm] = 0.0 + 0.0im
            end
        end
    end

    # Evaluate at a random point
    cost = 0.3   # cos(theta)
    phi = 1.2    # azimuthal angle

    # Distributed evaluation
    val_dist = SHTnsKit.dist_SH_to_point(cfg, Alm_pencil, cost, phi)

    # Serial evaluation
    val_serial = SHTnsKit.synthesis_point(cfg, alm, cost, phi)

    # Compare (dist_SH_to_point returns complex for m>0 terms combined differently)
    # Use real parts for real field comparison
    err = abs(real(val_dist) - real(val_serial))
    @test err < 1e-9
    return err
end

"""
Test dist_SH_to_lat: evaluate along a latitude, compare with serial SH_to_lat
"""
function test_latitude_evaluation(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing dist_SH_to_lat ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create coefficients
    rng = MersenneTwister(901)
    alm = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        alm[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    alm[:, 1] .= real.(alm[:, 1])
    MPI.Bcast!(alm, 0, comm)

    # Create distributed spectral PencilArray
    spec_pen = SHTnsKit.create_spectral_pencil(cfg; comm=comm)
    Alm_pencil = PencilArray{ComplexF64}(undef, spec_pen)
    for (ii, il) in enumerate(axes(Alm_pencil, 1))
        for (jj, jm) in enumerate(axes(Alm_pencil, 2))
            lval = globalindices(Alm_pencil, 1)[ii] - 1
            mval = globalindices(Alm_pencil, 2)[jj] - 1
            if lval >= mval && mval <= mmax && lval <= lmax
                Alm_pencil[il, jm] = alm[lval+1, mval+1]
            else
                Alm_pencil[il, jm] = 0.0 + 0.0im
            end
        end
    end

    # Pick a latitude
    cost = 0.5

    # Distributed evaluation
    lat_dist = SHTnsKit.dist_SH_to_lat(cfg, Alm_pencil, cost; real_output=true)

    # Serial evaluation (needs packed format)
    mres = cfg.mres
    alm_packed = zeros(ComplexF64, cfg.nlm)
    for m in 0:mmax, l in m:lmax
        idx = SHTnsKit.LM_index(lmax, mres, l, m) + 1
        alm_packed[idx] = alm[l+1, m+1]
    end
    lat_serial = SHTnsKit.SH_to_lat(cfg, alm_packed, cost)

    # Compare
    max_err = maximum(abs.(lat_dist - lat_serial))
    @test max_err < 1e-9
    return max_err
end

# ======================= PACKED DISTRIBUTED TRANSFORMS =======================

"""
Test dist_analysis_packed / dist_synthesis_packed roundtrip
"""
function test_packed_roundtrip(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing packed distributed transforms roundtrip ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create band-limited test coefficients in packed form
    rng = MersenneTwister(1000)
    alm_packed_orig = zeros(ComplexF64, cfg.nlm)
    for m in 0:mmax, l in m:lmax
        idx = SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1
        if m == 0
            alm_packed_orig[idx] = randn(rng)
        else
            alm_packed_orig[idx] = randn(rng) + im * randn(rng)
        end
    end
    MPI.Bcast!(alm_packed_orig, 0, comm)

    # Convert to dense for synthesis
    alm_dense = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        alm_dense[l+1, m+1] = alm_packed_orig[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1]
    end

    # Synthesize to spatial
    f_full = SHTnsKit.synthesis(cfg, alm_dense; real_output=true)

    # Distribute
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]
    flocal = zeros(Float64, PencilArrays.size_local(pen)...)
    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            flocal[i_local, j_local] = f_full[i_global, j_global]
        end
    end
    f_pa = PencilArray(pen, flocal)

    # Packed analysis
    alm_packed_rec = SHTnsKit.dist_analysis_packed(cfg, f_pa)

    # Packed synthesis roundtrip
    f_rec = SHTnsKit.dist_synthesis_packed(cfg, alm_packed_rec; prototype_θφ=f_pa, real_output=true)

    # Compare spatial fields
    max_err = 0.0
    for i in 1:size(f_rec, 1)
        for j in 1:size(f_rec, 2)
            max_err = max(max_err, abs(f_rec[i, j] - flocal[i, j]))
        end
    end
    global_max_err = MPI.Allreduce(max_err, MPI.MAX, comm)
    @test global_max_err < 1e-10

    # Also compare packed coefficients
    coeff_err = maximum(abs.(alm_packed_rec - alm_packed_orig))
    @test coeff_err < 1e-10

    return (global_max_err, coeff_err)
end

# ======================= ROUNDTRIP CONVENIENCE FUNCTIONS =======================

"""
Test dist_scalar_roundtrip! convenience function
"""
function test_scalar_roundtrip_convenience(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing dist_scalar_roundtrip! ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create band-limited field
    rng = MersenneTwister(1100)
    alm = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        alm[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    alm[:, 1] .= real.(alm[:, 1])
    MPI.Bcast!(alm, 0, comm)

    f_full = SHTnsKit.synthesis(cfg, alm; real_output=true)

    # Distribute
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]
    flocal = zeros(Float64, PencilArrays.size_local(pen)...)
    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            flocal[i_local, j_local] = f_full[i_global, j_global]
        end
    end
    f_pa = PencilArray(pen, flocal)

    # Roundtrip
    rel_local, rel_global = SHTnsKit.dist_scalar_roundtrip!(cfg, f_pa)
    @test rel_global < 1e-10
    return rel_global
end

"""
Test dist_vector_roundtrip! convenience function
"""
function test_vector_roundtrip_convenience(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing dist_vector_roundtrip! ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create band-limited vector field
    rng = MersenneTwister(1101)
    Slm = zeros(ComplexF64, lmax + 1, mmax + 1)
    Tlm = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in max(1, m):lmax
        Slm[l+1, m+1] = randn(rng) + im * randn(rng)
        Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    Slm[:, 1] .= real.(Slm[:, 1])
    Tlm[:, 1] .= real.(Tlm[:, 1])

    Vt_full, Vp_full = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm; real_output=true)

    # Distribute
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
    Vt_pa = PencilArray(pen, Vt_local)
    Vp_pa = PencilArray(pen, Vp_local)

    # Roundtrip
    (rl_t, rg_t), (rl_p, rg_p) = SHTnsKit.dist_vector_roundtrip!(cfg, Vt_pa, Vp_pa)
    @test rg_t < 1e-10
    @test rg_p < 1e-10
    return max(rg_t, rg_p)
end

# ======================= DISTRIBUTED SPECTRAL STORAGE TESTS =======================

"""
Test DistributedSpectralPlan + DistributedSpectralArray scatter/gather roundtrip
"""
function test_distributed_spectral_storage(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing DistributedSpectralPlan scatter/gather roundtrip ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create the distributed spectral plan
    dsp = ParExt.create_distributed_spectral_plan(lmax, mmax, comm)

    # Create dense coefficients
    rng = MersenneTwister(1200)
    alm_dense = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        alm_dense[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    alm_dense[:, 1] .= real.(alm_dense[:, 1])
    MPI.Bcast!(alm_dense, 0, comm)

    # Create distributed spectral array and scatter
    dsa = ParExt.create_distributed_spectral_array(dsp)
    ParExt.scatter_from_dense!(dsa, alm_dense)

    # Gather back and compare
    alm_gathered = ParExt.gather_to_dense(dsa)

    max_err = maximum(abs.(alm_gathered - alm_dense))
    @test max_err < 1e-14  # Should be exact (no floating point operations)
    return max_err
end

"""
Test dist_analysis_distributed / dist_synthesis_distributed full roundtrip
"""
function test_distributed_spectral_roundtrip(cfg::SHTnsKit.SHTConfig, pen::Pencil)
    root_println("  Testing dist_analysis_distributed / dist_synthesis_distributed roundtrip ...")

    lmax, mmax = cfg.lmax, cfg.mmax

    # Create test coefficients
    rng = MersenneTwister(1201)
    alm_orig = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        alm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    alm_orig[:, 1] .= real.(alm_orig[:, 1])
    MPI.Bcast!(alm_orig, 0, comm)

    # Synthesize to spatial
    f_full = SHTnsKit.synthesis(cfg, alm_orig; real_output=true)

    # Distribute spatially
    ranges = local_ranges(pen)
    theta_range, phi_range = ranges[1], ranges[2]
    flocal = zeros(Float64, PencilArrays.size_local(pen)...)
    for (i_local, i_global) in enumerate(theta_range)
        for (j_local, j_global) in enumerate(phi_range)
            flocal[i_local, j_local] = f_full[i_global, j_global]
        end
    end
    f_pa = PencilArray(pen, flocal)

    # Create distributed spectral plan
    dsp = ParExt.create_distributed_spectral_plan(lmax, mmax, comm)

    # Distributed analysis → distributed spectral array
    dsa = ParExt.dist_analysis_distributed(cfg, f_pa; plan=dsp)

    # Gather and compare with original coefficients
    alm_gathered = ParExt.gather_to_dense(dsa)
    coeff_err = maximum(abs.(alm_gathered - alm_orig))
    @test coeff_err < 1e-10

    # Distributed synthesis from distributed spectral array
    f_rec = ParExt.dist_synthesis_distributed(cfg, dsa; prototype_θφ=f_pa, real_output=true)

    # Compare spatial fields
    max_err = 0.0
    for i in 1:size(f_rec, 1)
        for j in 1:size(f_rec, 2)
            max_err = max(max_err, abs(f_rec[i, j] - flocal[i, j]))
        end
    end
    global_max_err = MPI.Allreduce(max_err, MPI.MAX, comm)
    @test global_max_err < 1e-10

    return (coeff_err, global_max_err)
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

    # Plan-based transforms
    try
        test_plan_analysis(cfg, pen)
        root_println("    [PASS] Plan-based analysis")
    catch e
        root_println("    [FAIL] Plan-based analysis: $e")
        all_passed = false
    end

    try
        test_plan_synthesis(cfg, pen)
        root_println("    [PASS] Plan-based synthesis")
    catch e
        root_println("    [FAIL] Plan-based synthesis: $e")
        all_passed = false
    end

    try
        test_plan_sphtor(cfg, pen)
        root_println("    [PASS] Plan-based sphtor")
    catch e
        root_println("    [FAIL] Plan-based sphtor: $e")
        all_passed = false
    end

    # Scalar Laplacian
    try
        test_scalar_laplacian(cfg, pen)
        root_println("    [PASS] Scalar Laplacian")
    catch e
        root_println("    [FAIL] Scalar Laplacian: $e")
        all_passed = false
    end

    # Spatial divergence/vorticity
    try
        test_spatial_divergence(cfg, pen)
        root_println("    [PASS] Spatial divergence")
    catch e
        root_println("    [FAIL] Spatial divergence: $e")
        all_passed = false
    end

    try
        test_spatial_vorticity(cfg, pen)
        root_println("    [PASS] Spatial vorticity")
    catch e
        root_println("    [FAIL] Spatial vorticity: $e")
        all_passed = false
    end

    # Point/latitude evaluation
    try
        test_point_evaluation(cfg, pen)
        root_println("    [PASS] Point evaluation")
    catch e
        root_println("    [FAIL] Point evaluation: $e")
        all_passed = false
    end

    try
        test_latitude_evaluation(cfg, pen)
        root_println("    [PASS] Latitude evaluation")
    catch e
        root_println("    [FAIL] Latitude evaluation: $e")
        all_passed = false
    end

    # Packed distributed transforms
    try
        test_packed_roundtrip(cfg, pen)
        root_println("    [PASS] Packed roundtrip")
    catch e
        root_println("    [FAIL] Packed roundtrip: $e")
        all_passed = false
    end

    # Roundtrip convenience functions
    try
        test_scalar_roundtrip_convenience(cfg, pen)
        root_println("    [PASS] Scalar roundtrip convenience")
    catch e
        root_println("    [FAIL] Scalar roundtrip convenience: $e")
        all_passed = false
    end

    try
        test_vector_roundtrip_convenience(cfg, pen)
        root_println("    [PASS] Vector roundtrip convenience")
    catch e
        root_println("    [FAIL] Vector roundtrip convenience: $e")
        all_passed = false
    end

    # Distributed spectral storage
    try
        test_distributed_spectral_storage(cfg, pen)
        root_println("    [PASS] Distributed spectral storage")
    catch e
        root_println("    [FAIL] Distributed spectral storage: $e")
        all_passed = false
    end

    try
        test_distributed_spectral_roundtrip(cfg, pen)
        root_println("    [PASS] Distributed spectral roundtrip")
    catch e
        root_println("    [FAIL] Distributed spectral roundtrip: $e")
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
