##########
# PencilArray operators
##########

const _OPERATOR_STATS_LOCK = ReentrantLock()
const _OPERATOR_STATS = Dict{Symbol,Int}(
    :diagonal_payload_sent_elements => 0,
    :narrow_payload_sent_elements => 0,
)

function _reset_operator_stats!()
    lock(_OPERATOR_STATS_LOCK) do
        for key in keys(_OPERATOR_STATS)
            _OPERATOR_STATS[key] = 0
        end
    end
    return nothing
end

function _operator_stats()
    lock(_OPERATOR_STATS_LOCK) do
        return (
            diagonal_payload_sent_elements=
                _OPERATOR_STATS[:diagonal_payload_sent_elements],
            narrow_payload_sent_elements=
                _OPERATOR_STATS[:narrow_payload_sent_elements],
        )
    end
end

function _validate_operator_pencils!(cfg, input::PencilArray,
                                     output::PencilArray,
                                     operation::Symbol;
                                     comm=MPI.COMM_WORLD)
    _validate_qst_pencil_communicators!(comm, (input, output), operation)
    _validate_cfg_replicated(cfg, comm)
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    _validate_scalar_pencil!(
        cfg, input, expected, operation; comm, peer=output,
        require_full_first_dim=true, required_decomposition=(2,),
        require_complex_input=true,
    )
    _validate_scalar_pencil!(
        cfg, output, expected, operation; comm, peer=input,
        require_full_first_dim=true, required_decomposition=(2,),
        require_complex_input=true,
    )
    _validate_identical_pencil_layout!(input, output, operation; comm)
    flags = eltype(input) === eltype(output) ? UInt32(0) : UInt32(0x0004)
    Base.mightalias(parent(input), parent(output)) && (flags |= 0x2000)
    _collective_validation_error(comm, flags, operation)
    return comm
end

function _validate_operator_matrix!(mx::AbstractVector{<:Real}, cfg,
                                    comm, operation::Symbol)
    code = _scalar_precision_code(eltype(mx))
    flags = UInt32(0)
    expected = 2cfg.nlm
    length(mx) == expected || (flags |= 0x0001)
    code == 0 && (flags |= 0x0004)
    MPI.Allreduce(code, min, comm) == MPI.Allreduce(code, max, comm) ||
        (flags |= 0x0004)

    # Convert every valid representation to the same exact carrier before the
    # collective: Float32 and Float64 values are represented exactly by
    # Float64.  Invalid ranks still participate with the fixed configured
    # length, so no rank can enter a differently shaped candidate collective.
    values = zeros(Float64, expected)
    if code != 0 && length(mx) == expected
        @inbounds for k in eachindex(values)
            values[k] = Float64(mx[k])
            isfinite(values[k]) || (flags |= 0x4000)
        end
    end
    reference = copy(values)
    MPI.Bcast!(reference, 0, comm)
    values == reference || (flags |= 0x4000)
    _collective_validation_error(comm, flags, operation)
    return nothing
end

"""
    dist_apply_laplacian!(cfg, Alm_pencil::PencilArray)

In-place multiply by -l(l+1) for distributed Alm with dims (:l,:m). No communication.
"""
function SHTnsKit.dist_apply_laplacian!(cfg::SHTnsKit.SHTConfig, Alm_pencil::PencilArray)
    comm = MPI.COMM_WORLD
    _validate_qst_pencil_communicators!(comm, (Alm_pencil,), :dist_apply_laplacian!)
    _validate_cfg_replicated(cfg, comm)
    _validate_scalar_pencil!(
        cfg, Alm_pencil, (cfg.lmax + 1, cfg.mmax + 1),
        :dist_apply_laplacian!; comm, require_full_first_dim=true,
        required_decomposition=(2,), require_complex_input=true,
    )
    l_globals = collect(Int, globalindices(Alm_pencil, 1))
    m_globals = collect(Int, globalindices(Alm_pencil, 2))
    values = parent(Alm_pencil)
    @inbounds for (local_m, m_index) in pairs(m_globals),
                  (local_l, l_index) in pairs(l_globals)
        l = l_index - 1; m = m_index - 1
        values[local_l, local_m] = m % cfg.mres == 0 && l >= max(1, m) ?
            -(l * (l + 1)) * values[local_l, local_m] : zero(eltype(values))
    end
    return Alm_pencil
end

"""
    dist_SH_mul_mx!(cfg, mx, Alm_pencil::PencilArray, R_pencil::PencilArray)

Apply a three-diagonal operator to m-decomposed spectral pencils.  Every rank
owns complete l columns for its local m orders, so l±1 neighbours are read from
local storage and no payload communication is required.
Forward pass: R[l,m] = mx[2*lm_prev+2]*Q[l-1,m] + mx[2*lm_next+1]*Q[l+1,m]
where lm_prev = LM_index(l-1,m) and lm_next = LM_index(l+1,m).
"""
function SHTnsKit.SH_mul_mx(::SHTnsKit.CPU, cfg::SHTnsKit.SHTConfig,
                            mx::AbstractVector{<:Real},
                            input::PencilArray, output::PencilArray)
    comm = _validate_operator_pencils!(cfg, input, output, :SH_mul_mx)
    _validate_operator_matrix!(mx, cfg, comm, :SH_mul_mx)
    l_globals = collect(Int, globalindices(input, 1))
    m_globals = collect(Int, globalindices(input, 2))
    source = parent(input); destination = parent(output)
    CT = promote_type(eltype(input), eltype(output), complex(eltype(mx)))
    @inbounds for (local_m, m_index) in pairs(m_globals),
                  (local_l, l_index) in pairs(l_globals)
        l = l_index - 1; m = m_index - 1
        acc = zero(CT)
        if m % cfg.mres == 0 && l >= m
            if l > m
                below = SHTnsKit.LM_index(cfg.lmax, cfg.mres, l - 1, m)
                acc += mx[2below + 2] * source[local_l - 1, local_m]
            end
            if l < cfg.lmax
                above = SHTnsKit.LM_index(cfg.lmax, cfg.mres, l + 1, m)
                acc += mx[2above + 1] * source[local_l + 1, local_m]
            end
        end
        destination[local_l, local_m] = acc
    end
    return output
end

SHTnsKit.SH_mul_mx(cfg::SHTnsKit.SHTConfig, mx::AbstractVector{<:Real},
                   input::PencilArray, output::PencilArray) =
    SHTnsKit.SH_mul_mx(SHTnsKit.CPU(), cfg, mx, input, output)

SHTnsKit.dist_SH_mul_mx!(cfg::SHTnsKit.SHTConfig,
                         mx::AbstractVector{<:Real}, input::PencilArray,
                         output::PencilArray) =
    SHTnsKit.SH_mul_mx(SHTnsKit.CPU(), cfg, mx, input, output)

"""
    dist_spatial_divergence(cfg, Vtθφ, Vpθφ; prototype_θφ=Vtθφ, use_rfft=false, real_output=true)

Compute ∇·V for a distributed horizontal vector field using spectral decomposition.
"""
function SHTnsKit.dist_spatial_divergence(cfg::SHTnsKit.SHTConfig,
                                          Vtθφ::PencilArray, Vpθφ::PencilArray;
                                          prototype_θφ::PencilArray=Vtθφ,
                                          use_rfft::Bool=false,
                                          real_output::Bool=true)
    Slm, _ = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ; use_rfft)
    δlm = SHTnsKit.divergence_from_spheroidal(cfg, Slm)
    return SHTnsKit.dist_synthesis(cfg, δlm; prototype_θφ=prototype_θφ,
                                   real_output=real_output, use_rfft=use_rfft)
end

"""
    dist_spatial_vorticity(cfg, Vtθφ, Vpθφ; prototype_θφ=Vtθφ, use_rfft=false, real_output=true)

Compute vertical vorticity (∇×V)·r̂ for a distributed horizontal vector field.
"""
function SHTnsKit.dist_spatial_vorticity(cfg::SHTnsKit.SHTConfig,
                                         Vtθφ::PencilArray, Vpθφ::PencilArray;
                                         prototype_θφ::PencilArray=Vtθφ,
                                         use_rfft::Bool=false,
                                         real_output::Bool=true)
    _, Tlm = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ; use_rfft)
    ζlm = SHTnsKit.vorticity_from_toroidal(cfg, Tlm)
    return SHTnsKit.dist_synthesis(cfg, ζlm; prototype_θφ=prototype_θφ,
                                   real_output=real_output, use_rfft=use_rfft)
end

"""
    dist_scalar_laplacian(cfg, fθφ; prototype_θφ=fθφ, use_rfft=false, real_output=true)

Apply spherical Laplacian to a distributed scalar field by transforming to spectral
space, scaling by −l(l+1), and synthesizing back.
"""
function SHTnsKit.dist_scalar_laplacian(cfg::SHTnsKit.SHTConfig,
                                        fθφ::PencilArray;
                                        prototype_θφ::PencilArray=fθφ,
                                        use_rfft::Bool=false,
                                        real_output::Bool=true)
    Alm = SHTnsKit.dist_analysis(cfg, fθφ; use_rfft)
    SHTnsKit.dist_apply_laplacian!(cfg, Alm)
    return SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=prototype_θφ,
                                   real_output=real_output, use_rfft=use_rfft)
end

"""
    dist_scalar_laplacian!(cfg, outθφ, inθφ; use_rfft=false, real_output=true)

In-place version that writes the Laplacian of `inθφ` into `outθφ`.
"""
function SHTnsKit.dist_scalar_laplacian!(cfg::SHTnsKit.SHTConfig,
                                         outθφ::PencilArray,
                                         inθφ::PencilArray;
                                         use_rfft::Bool=false,
                                         real_output::Bool=true)
    result = SHTnsKit.dist_scalar_laplacian(cfg, inθφ; prototype_θφ=outθφ,
                                            use_rfft=use_rfft, real_output=real_output)
    copyto!(outθφ, result)
    return outθφ
end
