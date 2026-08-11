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
    return Pencil((cfg.lmax + 1, cfg.mmax + 1), comm)
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
    create_spatial_pencil(cfg; comm=MPI.COMM_WORLD)

Create a Pencil for distributed spatial data `(nlat, nlon)` decomposed along
**latitude (θ, dimension 1)** — the scaling-friendly axis for SHTnsKit.

Prefer this over the bare `Pencil((nlat, nlon), comm)`: PencilArrays splits the
*last* dimension by default, which decomposes longitude (φ). The φ-distributed
analysis path `Allgatherv!`s the full longitude onto every rank and then
replicates the Legendre transform, so it does NOT scale (it usually gets slower
with more ranks). Latitude decomposition instead divides the Legendre work per
rank. Ordinary `analysis(cfg, pencil)` reduces only each destination rank's
owned m-columns; pass `return_pencil=false` explicitly for the replicated dense
compatibility result.
"""
function SHTnsKit.create_spatial_pencil(cfg::SHTnsKit.SHTConfig; comm=MPI.COMM_WORLD)
    # Decompose dimension 1 (θ / latitude). The (1,) tuple selects which global
    # dimension to split; omitting it would default to the last dim (φ).
    return Pencil((cfg.nlat, cfg.nlon), (1,), comm)
end

"""
    create_spatial_array(cfg; comm=MPI.COMM_WORLD)

Create a zero-initialized distributed `PencilArray{Float64}` for spatial data,
decomposed along latitude (θ). See [`create_spatial_pencil`](@ref).
"""
function SHTnsKit.create_spatial_array(cfg::SHTnsKit.SHTConfig; comm=MPI.COMM_WORLD)
    pen = SHTnsKit.create_spatial_pencil(cfg; comm)
    arr = PencilArray{Float64}(undef, pen)
    fill!(parent(arr), 0.0)
    return arr
end

"""
    matrix_to_spectral_pencil(cfg, Alm::AbstractMatrix; comm=MPI.COMM_WORLD)

Convert a dense spectral coefficient matrix to a distributed PencilArray.
Each rank receives its local portion of the m-distributed array.
"""
function SHTnsKit.matrix_to_spectral_pencil(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix; comm=MPI.COMM_WORLD)
    _record_pencil_scalar_stat!(:full_matrix_helper_calls, 1)
    pen = SHTnsKit.create_spectral_pencil(cfg; comm)
    known_comm = PencilArrays.get_comm(pen)
    _validate_explicit_comm!(known_comm, comm, :matrix_to_spectral_pencil)
    _validate_cfg_replicated(cfg, known_comm)
    _validate_spectral_pencil_plan!(cfg, pen, known_comm, :matrix_to_spectral_pencil)
    _validate_dense_spectral_matrix!(cfg, Alm, known_comm, :matrix_to_spectral_pencil)

    Alm_p = PencilArray{eltype(Alm)}(undef, pen)

    # Copy only the local portion
    lloc = axes(Alm_p, 1)
    mloc = axes(Alm_p, 2)
    gl_l = collect(Int, globalindices(Alm_p, 1))
    gl_m = collect(Int, globalindices(Alm_p, 2))

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
    spectral_pencil_to_matrix(cfg, Alm_p::PencilArray; comm=nothing)

Gather a distributed spectral PencilArray to a dense matrix on all ranks.
"""
function SHTnsKit.spectral_pencil_to_matrix(cfg::SHTnsKit.SHTConfig, Alm_p::PencilArray; comm=nothing)
    _record_pencil_scalar_stat!(:full_matrix_helper_calls, 1)
    known_comm = communicator(Alm_p)
    _validate_cfg_replicated(cfg, known_comm)
    _validate_explicit_comm!(known_comm, comm, :spectral_pencil_to_matrix)
    _validate_scalar_pencil!(
        cfg, Alm_p, (cfg.lmax + 1, cfg.mmax + 1),
        :spectral_pencil_to_matrix;
        comm=known_comm, require_full_first_dim=true,
        required_decomposition=(2,), require_complex_input=true,
    )

    Alm = zeros(eltype(Alm_p), cfg.lmax + 1, cfg.mmax + 1)

    # Copy local portion
    lloc = axes(Alm_p, 1)
    mloc = axes(Alm_p, 2)
    gl_l = collect(Int, globalindices(Alm_p, 1))
    gl_m = collect(Int, globalindices(Alm_p, 2))

    for (jj, jm) in enumerate(mloc)
        mglob = gl_m[jj]
        for (ii, il) in enumerate(lloc)
            lglob = gl_l[ii]
            Alm[lglob, mglob] = Alm_p[il, jm]
        end
    end

    # Allreduce to combine contributions from all ranks
    # Always use the Pencil's communicator. An explicitly supplied congruent
    # communicator is accepted above, but may be a different duplicate on each
    # rank; using the known communicator keeps collective context ordering safe.
    MPI.Allreduce!(Alm, +, known_comm)

    return Alm
end

##########
# Scalar transform dispatch
##########

function SHTnsKit.synthesis(cfg::SHTnsKit.SHTConfig, Alm::PencilArray;
                            prototype_θφ::PencilArray, real_output::Bool=true,
                            use_rfft::Bool=false)
    local_result = SHTnsKit.dist_synthesis(
        cfg, Alm; prototype_θφ, real_output, use_rfft,
    )
    result = PencilArray{eltype(local_result)}(undef, pencil(prototype_θφ))
    copyto!(parent(result), local_result)
    return result
end

function SHTnsKit.synthesis_cplx(cfg::SHTnsKit.SHTConfig, Alm::PencilArray;
                                 prototype_θφ::PencilArray)
    return SHTnsKit.synthesis(
        cfg, Alm; prototype_θφ, real_output=false,
    )
end

"""
Analyze a distributed field. The ordinary/default result is a spectral
`PencilArray`; `return_pencil=false` preserves the replicated dense
compatibility result from `dist_analysis`.
"""
function SHTnsKit.analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                           use_rfft::Bool=false, return_pencil::Bool=true)
    if return_pencil
        return dist_analysis_pencil(cfg, fθφ; use_rfft)
    else
        # Dense return is the preserved compatibility path.
        return SHTnsKit.dist_analysis(cfg, fθφ; use_rfft)
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
