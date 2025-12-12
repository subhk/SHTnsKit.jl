#!/usr/bin/env julia

# Parallel Y-rotation example demonstrating distributed spherical harmonic rotations
#
# Run with: mpiexec -n 2 julia --project examples/rotate_y_parallel.jl

using MPI
using PencilArrays
using PencilFFTs
using SHTnsKit

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if rank == 0
        println("Parallel Y-rotation example with $nprocs MPI processes")
    end

    lmax = 16
    nlat = lmax + 2
    nlon = 2*lmax + 1
    β = 0.35  # Rotation angle around Y-axis (radians)

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)

    # Build a balanced 2D processor grid
    function _procgrid(p)
        best = (1, p); diff = p - 1
        for d in 1:p
            if p % d == 0
                d2 = div(p, d)
                if abs(d - d2) < diff
                    best = (d, d2); diff = abs(d - d2)
                end
            end
        end
        return best
    end

    pθ, pφ = _procgrid(nprocs)
    topo = Pencil((nlat, nlon), (pθ, pφ), comm)

    # Create distributed spatial field using PencilArrays v0.19+ API
    local_dims = PencilArrays.size_local(topo)
    fθφ = PencilArray(topo, zeros(Float64, local_dims...))

    # Fill with test pattern
    local_data = parent(fθφ)
    for iθ in axes(local_data, 1), iφ in axes(local_data, 2)
        local_data[iθ, iφ] = sin(0.3*(iθ+1)) + 0.7*cos(0.2*(iφ+1))
    end

    # Distributed analysis: spatial -> spectral
    Alm = SHTnsKit.dist_analysis(cfg, fθφ)

    if rank == 0
        println("Distributed analysis complete. Alm size: $(size(Alm))")
    end

    # Apply Y-rotation in spectral space
    # Y-rotation mixes coefficients with different m values at the same l
    Alm_rot = zeros(ComplexF64, size(Alm)...)
    SHTnsKit.dist_SH_Yrotate(cfg, Alm, β, Alm_rot)

    if rank == 0
        println("Y-rotation by β=$(β) applied in spectral space")
        println("Max |Alm_rot|: $(maximum(abs.(Alm_rot)))")
    end

    # Synthesize rotated field back to spatial domain
    fθφ_rot = SHTnsKit.dist_synthesis(cfg, Alm_rot; prototype_θφ=fθφ, real_output=true)

    if rank == 0
        println("Synthesis of rotated field complete")
    end

    # Verify that rotation preserved energy (Parseval's theorem)
    # Energy should be conserved: sum(|Alm|^2) ≈ sum(|Alm_rot|^2)
    energy_orig = sum(abs2, Alm)
    energy_rot = sum(abs2, Alm_rot)
    energy_ratio = energy_rot / energy_orig

    if rank == 0
        println("Energy conservation check:")
        println("  Original energy: $energy_orig")
        println("  Rotated energy: $energy_rot")
        println("  Ratio: $energy_ratio (should be ≈ 1.0)")
        if abs(energy_ratio - 1.0) < 1e-10
            println("✓ Energy conservation PASSED")
        else
            println("⚠ Energy conservation check: ratio = $energy_ratio")
        end
    end

    MPI.Barrier(comm)
    if rank == 0
        println("Done.")
    end

    MPI.Finalize()
end

main()
