#!/usr/bin/env julia

# Parallel operator example demonstrating distributed spherical harmonic operations
#
# Run with: mpiexec -n 2 julia --project examples/operator_parallel.jl

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
        println("Parallel operator example with $nprocs MPI processes")
    end

    lmax = 16
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)

    # Build a balanced 2D processor grid
    function _procgrid(p)
        best = (1,p); diff = p-1
        for d in 1:p
            if p % d == 0
                d2 = div(p,d)
                if abs(d-d2) < diff
                    best = (d,d2); diff = abs(d-d2)
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
        local_data[iθ, iφ] = sin(0.3*(iθ+1)) * cos(0.2*(iφ+1))
    end

    # Distributed analysis: spatial -> spectral
    Alm = SHTnsKit.dist_analysis(cfg, fθφ)

    if rank == 0
        println("Distributed analysis complete. Alm size: $(size(Alm))")
        println("Max |Alm|: $(maximum(abs.(Alm)))")
    end

    # Apply Laplacian operator in spectral space: -l(l+1) multiplication
    Alm_lap = copy(Alm)
    SHTnsKit.dist_apply_laplacian!(cfg, Alm_lap)

    # Synthesize back to spatial domain
    fθφ_lap = SHTnsKit.dist_synthesis(cfg, Alm_lap; prototype_θφ=fθφ, real_output=true)

    if rank == 0
        println("Laplacian operator applied in spectral space")
    end

    # Reference: compute Laplacian via analysis -> -l(l+1) -> synthesis
    # For comparison, compute the same thing via serial transforms if possible
    if nprocs == 1
        # Verify against serial computation
        Alm_serial = SHTnsKit.analysis(cfg, local_data)
        Alm_lap_serial = copy(Alm_serial)
        SHTnsKit.dist_apply_laplacian!(cfg, Alm_lap_serial)
        fθφ_lap_serial = SHTnsKit.synthesis(cfg, Alm_lap_serial; real_output=true)

        err = maximum(abs.(parent(fθφ_lap) .- fθφ_lap_serial))
        println("Roundtrip error (distributed vs serial): $err")
    end

    # Check that the Laplacian was applied correctly
    # For a simple test: verify that Alm coefficients were scaled by -l(l+1)
    check_scaling = true
    for l in 0:cfg.lmax
        for m in 0:min(l, cfg.mmax)
            expected_scale = -l * (l + 1)
            if abs(Alm[l+1, m+1]) > 1e-14
                actual_scale = Alm_lap[l+1, m+1] / Alm[l+1, m+1]
                if abs(actual_scale - expected_scale) > 1e-10
                    check_scaling = false
                end
            end
        end
    end

    if rank == 0
        println("Laplacian scaling check: $(check_scaling ? "PASSED" : "FAILED")")
    end

    MPI.Barrier(comm)
    if rank == 0
        println("Done.")
    end

    MPI.Finalize()
end

main()
