##########
# Unified dispatch helpers for PencilArray inputs
##########

"""
    create_spectral_pencil(cfg; comm=MPI.COMM_WORLD)

Create a Pencil configuration for distributed spectral coefficients.
The spectral array has dimensions (lmax+1, mmax+1) and is distributed
along the m dimension (dimension 2) for optimal SHT performance.
"""
function SHTnsKit.create_spectral_pencil(cfg::SHTnsKit.SHTConfig; comm=MPI.COMM_WORLD)
    # Distribute along m (dimension 2) - each rank owns all l values for its m subset
    # This is optimal because each m-column is independent in Legendre transforms
    return Pencil((cfg.lmax + 1, cfg.mmax + 1), comm; decomp_dims=(2,))
end

"""
    create_spectral_array(cfg; comm=MPI.COMM_WORLD)

Create an uninitialized distributed PencilArray for spectral coefficients.
"""
function SHTnsKit.create_spectral_array(cfg::SHTnsKit.SHTConfig; comm=MPI.COMM_WORLD)
    pen = SHTnsKit.create_spectral_pencil(cfg; comm)
    return PencilArray{ComplexF64}(undef, pen)
end

"""
    matrix_to_spectral_pencil(cfg, Alm::AbstractMatrix; comm=MPI.COMM_WORLD)

Convert a dense spectral coefficient matrix to a distributed PencilArray.
Each rank receives its local portion of the m-distributed array.
"""
function SHTnsKit.matrix_to_spectral_pencil(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix; comm=MPI.COMM_WORLD)
    size(Alm) == (cfg.lmax + 1, cfg.mmax + 1) || throw(DimensionMismatch(
        "Alm size $(size(Alm)) does not match expected ($((cfg.lmax + 1, cfg.mmax + 1)))"))

    pen = SHTnsKit.create_spectral_pencil(cfg; comm)
    Alm_p = PencilArray{ComplexF64}(undef, pen)

    # Copy only the local portion
    lloc = axes(Alm_p, 1)
    mloc = axes(Alm_p, 2)
    gl_l = globalindices(Alm_p, 1)
    gl_m = globalindices(Alm_p, 2)

    for (jj, jm) in enumerate(mloc)
        mglob = gl_m[jj]
        for (ii, il) in enumerate(lloc)
            lglob = gl_l[ii]
            Alm_p[il, jm] = Alm[lglob, mglob]
        end
    end

    return Alm_p
end

"""
    spectral_pencil_to_matrix(cfg, Alm_p::PencilArray; comm=MPI.COMM_WORLD)

Gather a distributed spectral PencilArray to a dense matrix on all ranks.
"""
function SHTnsKit.spectral_pencil_to_matrix(cfg::SHTnsKit.SHTConfig, Alm_p::PencilArray; comm=nothing)
    if comm === nothing
        comm = communicator(Alm_p)
    end

    Alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)

    # Copy local portion
    lloc = axes(Alm_p, 1)
    mloc = axes(Alm_p, 2)
    gl_l = globalindices(Alm_p, 1)
    gl_m = globalindices(Alm_p, 2)

    for (jj, jm) in enumerate(mloc)
        mglob = gl_m[jj]
        for (ii, il) in enumerate(lloc)
            lglob = gl_l[ii]
            Alm[lglob, mglob] = Alm_p[il, jm]
        end
    end

    # Allreduce to combine contributions from all ranks
    MPI.Allreduce!(Alm, +, comm)

    return Alm
end

##########
# Scalar transform dispatch
##########

function SHTnsKit.synthesis(cfg::SHTnsKit.SHTConfig, Alm::PencilArray;
                            prototype_θφ::PencilArray, real_output::Bool=true)
    return SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output)
end

function SHTnsKit.analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                           use_rfft::Bool=false, return_pencil::Bool=false)
    Alm = SHTnsKit.dist_analysis(cfg, fθφ; use_rfft)

    if return_pencil
        return SHTnsKit.matrix_to_spectral_pencil(cfg, Alm; comm=communicator(fθφ))
    else
        return Alm
    end
end

##########
# Vector/QST dispatch for PencilArrays
##########

function SHTnsKit.analysis_sphtor(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArray, Vpθφ::PencilArray;
                                   use_tables=cfg.use_plm_tables, return_pencil::Bool=false)
    Slm, Tlm = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ; use_tables)

    if return_pencil
        comm = communicator(Vtθφ)
        return SHTnsKit.matrix_to_spectral_pencil(cfg, Slm; comm),
               SHTnsKit.matrix_to_spectral_pencil(cfg, Tlm; comm)
    else
        return Slm, Tlm
    end
end

function SHTnsKit.analysis_qst(cfg::SHTnsKit.SHTConfig, Vrθφ::PencilArray, Vtθφ::PencilArray, Vpθφ::PencilArray;
                                return_pencil::Bool=false)
    Qlm, Slm, Tlm = SHTnsKit.dist_analysis_qst(cfg, Vrθφ, Vtθφ, Vpθφ)

    if return_pencil
        comm = communicator(Vrθφ)
        return SHTnsKit.matrix_to_spectral_pencil(cfg, Qlm; comm),
               SHTnsKit.matrix_to_spectral_pencil(cfg, Slm; comm),
               SHTnsKit.matrix_to_spectral_pencil(cfg, Tlm; comm)
    else
        return Qlm, Slm, Tlm
    end
end
