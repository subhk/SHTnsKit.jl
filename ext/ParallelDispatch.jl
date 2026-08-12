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
                                   use_tables=cfg.use_plm_tables,
                                   use_rfft::Bool=false,
                                   return_pencil::Bool=true)
    comm = communicator(Vtθφ)
    return_pencil = _validate_collective_bool_option!(
        comm, return_pencil, :analysis_sphtor, UInt32(0x1000),
    )
    if return_pencil
        return dist_analysis_sphtor_pencil(cfg, Vtθφ, Vpθφ; use_rfft)
    else
        return SHTnsKit.dist_analysis_sphtor(
            cfg, Vtθφ, Vpθφ; use_tables, use_rfft,
        )
    end
end

function SHTnsKit.analysis_sphtor_cplx(cfg::SHTnsKit.SHTConfig,
                                        Vtθφ::PencilArray,
                                        Vpθφ::PencilArray;
                                        return_pencil::Bool=true)
    return SHTnsKit.analysis_sphtor(
        cfg, Vtθφ, Vpθφ; return_pencil, use_rfft=false,
    )
end

function SHTnsKit.synthesis_sphtor(cfg::SHTnsKit.SHTConfig,
                                    Slm::PencilArray, Tlm::PencilArray;
                                    prototype_θφ::PencilArray,
                                    real_output::Bool=true,
                                    use_rfft::Bool=false)
    Vt_local, Vp_local = dist_synthesis_sphtor_pencil(
        cfg, Slm, Tlm; prototype_θφ, real_output, use_rfft,
    )
    Vt = PencilArray{eltype(Vt_local)}(undef, pencil(prototype_θφ))
    Vp = PencilArray{eltype(Vp_local)}(undef, pencil(prototype_θφ))
    copyto!(parent(Vt), Vt_local); copyto!(parent(Vp), Vp_local)
    return Vt, Vp
end


function SHTnsKit.synthesis_sphtor_cplx(cfg::SHTnsKit.SHTConfig,
                                         Slm::PencilArray,
                                         Tlm::PencilArray;
                                         prototype_θφ::PencilArray)
    return SHTnsKit.synthesis_sphtor(
        cfg, Slm, Tlm; prototype_θφ, real_output=false,
    )
end

function _zero_spectral_pencil_like(reference::PencilArray)
    result = PencilArray{eltype(reference)}(undef, pencil(reference))
    fill!(parent(result), zero(eltype(result)))
    return result
end

function SHTnsKit.synthesis_sph(cfg::SHTnsKit.SHTConfig,
                                Slm::PencilArray;
                                prototype_θφ::PencilArray,
                                real_output::Bool=true)
    return SHTnsKit.synthesis_sphtor(
        cfg, Slm, _zero_spectral_pencil_like(Slm);
        prototype_θφ, real_output,
    )
end


function SHTnsKit.synthesis_sph_cplx(cfg::SHTnsKit.SHTConfig,
                                     Slm::PencilArray;
                                     prototype_θφ::PencilArray)
    return SHTnsKit.synthesis_sph(
        cfg, Slm; prototype_θφ, real_output=false,
    )
end

function SHTnsKit.synthesis_tor(cfg::SHTnsKit.SHTConfig,
                                Tlm::PencilArray;
                                prototype_θφ::PencilArray,
                                real_output::Bool=true)
    return SHTnsKit.synthesis_sphtor(
        cfg, _zero_spectral_pencil_like(Tlm), Tlm;
        prototype_θφ, real_output,
    )
end


function SHTnsKit.synthesis_tor_cplx(cfg::SHTnsKit.SHTConfig,
                                     Tlm::PencilArray;
                                     prototype_θφ::PencilArray)
    return SHTnsKit.synthesis_tor(
        cfg, Tlm; prototype_θφ, real_output=false,
    )
end

function _pencil_vector_diagonal!(output::PencilArray,
                                  cfg::SHTnsKit.SHTConfig,
                                  input::PencilArray;
                                  inverse::Bool,
                                  operation::Symbol)
    comm = communicator(input)
    _validate_cfg_replicated(cfg, comm)
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    _validate_scalar_pencil!(
        cfg, input, expected, operation; comm, peer=output,
        require_full_first_dim=true, required_decomposition=(2,),
    )
    _validate_scalar_pencil!(
        cfg, output, expected, operation; comm, peer=input,
        require_full_first_dim=true, required_decomposition=(2,),
    )
    _validate_identical_pencil_layout!(input, output, operation; comm)
    flags = eltype(input) === eltype(output) ? UInt32(0) : UInt32(0x0004)
    _collective_validation_error(comm, flags, operation)

    l_globals = collect(Int, globalindices(input, 1))
    m_globals = collect(Int, globalindices(input, 2))
    source = parent(input); destination = parent(output)
    @inbounds for (local_m, m_index) in pairs(m_globals),
                  (local_l, l_index) in pairs(l_globals)
        l = l_index - 1; m = m_index - 1
        if l >= max(1, m) && m % cfg.mres == 0
            ll1 = l * (l + 1)
            destination[local_l, local_m] = inverse ?
                -(source[local_l, local_m] / ll1) :
                -ll1 * source[local_l, local_m]
        else
            destination[local_l, local_m] = zero(eltype(destination))
        end
    end
    return output
end

function _pencil_vector_diagonal(cfg::SHTnsKit.SHTConfig,
                                 input::PencilArray;
                                 inverse::Bool,
                                 operation::Symbol)
    output = PencilArray{eltype(input)}(undef, pencil(input))
    return _pencil_vector_diagonal!(
        output, cfg, input; inverse, operation,
    )
end

SHTnsKit.divergence_from_spheroidal(cfg::SHTnsKit.SHTConfig,
                                    input::PencilArray) =
    _pencil_vector_diagonal(
        cfg, input; inverse=false, operation=:divergence_from_spheroidal,
    )
SHTnsKit.divergence_from_spheroidal!(cfg::SHTnsKit.SHTConfig,
                                     output::PencilArray,
                                     input::PencilArray) =
    _pencil_vector_diagonal!(
        output, cfg, input;
        inverse=false, operation=:divergence_from_spheroidal!,
    )
SHTnsKit.spheroidal_from_divergence(cfg::SHTnsKit.SHTConfig,
                                    input::PencilArray) =
    _pencil_vector_diagonal(
        cfg, input; inverse=true, operation=:spheroidal_from_divergence,
    )
SHTnsKit.spheroidal_from_divergence!(cfg::SHTnsKit.SHTConfig,
                                     output::PencilArray,
                                     input::PencilArray) =
    _pencil_vector_diagonal!(
        output, cfg, input;
        inverse=true, operation=:spheroidal_from_divergence!,
    )
SHTnsKit.vorticity_from_toroidal(cfg::SHTnsKit.SHTConfig,
                                 input::PencilArray) =
    _pencil_vector_diagonal(
        cfg, input; inverse=false, operation=:vorticity_from_toroidal,
    )
SHTnsKit.vorticity_from_toroidal!(cfg::SHTnsKit.SHTConfig,
                                  output::PencilArray,
                                  input::PencilArray) =
    _pencil_vector_diagonal!(
        output, cfg, input;
        inverse=false, operation=:vorticity_from_toroidal!,
    )
SHTnsKit.toroidal_from_vorticity(cfg::SHTnsKit.SHTConfig,
                                 input::PencilArray) =
    _pencil_vector_diagonal(
        cfg, input; inverse=true, operation=:toroidal_from_vorticity,
    )
SHTnsKit.toroidal_from_vorticity!(cfg::SHTnsKit.SHTConfig,
                                  output::PencilArray,
                                  input::PencilArray) =
    _pencil_vector_diagonal!(
        output, cfg, input;
        inverse=true, operation=:toroidal_from_vorticity!,
    )

function SHTnsKit.analysis_qst(cfg::SHTnsKit.SHTConfig,
                               Vrθφ::PencilArray,
                               Vtθφ::PencilArray,
                               Vpθφ::PencilArray;
                               use_rfft::Bool=false,
                               return_pencil::Bool=true)
    comm = _validate_qst_spatial_inputs!(
        cfg, Vrθφ, Vtθφ, Vpθφ; use_rfft,
    )
    return_pencil = _validate_collective_bool_option!(
        comm, return_pencil, :analysis_qst, UInt32(0x1000),
    )
    if return_pencil
        return dist_analysis_pencil(cfg, Vrθφ; use_rfft),
               dist_analysis_sphtor_pencil(cfg, Vtθφ, Vpθφ; use_rfft)...
    end
    return SHTnsKit.dist_analysis_qst(
        cfg, Vrθφ, Vtθφ, Vpθφ; use_rfft,
    )
end

function SHTnsKit.analysis_qst_cplx(cfg::SHTnsKit.SHTConfig,
                                    Vrθφ::PencilArray,
                                    Vtθφ::PencilArray,
                                    Vpθφ::PencilArray;
                                    return_pencil::Bool=true)
    return SHTnsKit.analysis_qst(
        cfg, Vrθφ, Vtθφ, Vpθφ; return_pencil,
    )
end

function SHTnsKit.synthesis_qst(cfg::SHTnsKit.SHTConfig,
                                Qlm::PencilArray,
                                Slm::PencilArray,
                                Tlm::PencilArray;
                                prototype_θφ::PencilArray,
                                real_output::Bool=true,
                                use_rfft::Bool=false)
    _validate_qst_synthesis_inputs!(
        cfg, Qlm, Slm, Tlm, prototype_θφ; real_output, use_rfft,
    )
    Vr_local = SHTnsKit.dist_synthesis(
        cfg, Qlm; prototype_θφ, real_output, use_rfft,
    )
    Vt_local, Vp_local = dist_synthesis_sphtor_pencil(
        cfg, Slm, Tlm; prototype_θφ, real_output, use_rfft,
    )
    outputs = map((Vr_local, Vt_local, Vp_local)) do local_value
        result = PencilArray{eltype(local_value)}(undef, pencil(prototype_θφ))
        copyto!(parent(result), local_value)
        result
    end
    return outputs
end

function SHTnsKit.synthesis_qst_cplx(cfg::SHTnsKit.SHTConfig,
                                     Qlm::PencilArray,
                                     Slm::PencilArray,
                                     Tlm::PencilArray;
                                     prototype_θφ::PencilArray)
    return SHTnsKit.synthesis_qst(
        cfg, Qlm, Slm, Tlm; prototype_θφ, real_output=false,
    )
end
