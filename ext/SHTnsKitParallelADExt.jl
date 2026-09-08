module SHTnsKitParallelADExt

#=
================================================================================
SHTnsKitParallelADExt — ChainRules rrules for distributed transforms
================================================================================

Loads only when ChainRulesCore + MPI + PencilArrays + PencilFFTs are all
present. Provides backward-pass rules for `dist_analysis` and `dist_synthesis`
so Zygote/ChainRules-based AD pipelines get accurate gradients through the
distributed spatial↔spectral path without falling back to source-level tracing
of MPI collectives.

Math summary
------------
Forward `dist_analysis(cfg, fθφ)` produces a fully-reduced `Alm` that is
identical on every rank (enforced by the contract upheld in `dist_synthesis`).
Its adjoint operator maps an `Alm̄` (also replicated) to a spatial cotangent
`f̄θφ` localized per-rank — this is exactly the local `_adjoint_analysis`
already implemented in `SHTnsKitAdvancedADExt` restricted to the rank's θ
slab. No inter-rank communication is needed for the backward pass: every
rank's θ rows are independent in the adjoint.

Forward `dist_synthesis(cfg, Alm; prototype_θφ)` maps a replicated `Alm` to a
distributed spatial field. Its adjoint maps a distributed spatial cotangent
`f̄_local` to a replicated `Ālm` — an Allreduce across ranks sums per-rank
contributions, which matches the adjoint of the implicit "broadcast Alm"
operation on the forward side.
================================================================================
=#

using ChainRulesCore
using MPI
using PencilArrays
using PencilArrays: PencilArray
using SHTnsKit
using FFTW

# Distributed reverse rules in this extension use CPU FFTW/Legendre adjoints.
# Runtime storage classification accepts CPU wrappers (views, shared arrays,
# and custom host arrays) while preventing a vendor PencilArray from reaching
# an implicit host conversion. GPU-backed distributed AD requires a
# vendor-native compound rule; until one is available, fail at the storage
# boundary before running the forward transform.

@inline function _require_host_pencil(operation::Symbol, value::PencilArray,
                                      comm=communicator(value))
    local_ok = try
        SHTnsKit.on_device(parent(value)) isa SHTnsKit.CPU
    catch
        false
    end
    MPI.Allreduce(local_ok, &, comm) && return value
    throw(SHTnsKit.BackendUnavailableError(
        operation,
        "distributed reverse-mode AD for GPU-backed PencilArray storage requires a vendor-native compound rule",
    ))
end

@inline function _materialize_host_coefficient(operation::Symbol, value, cfg)
    value isa ChainRulesCore.AbstractZero &&
        return zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    SHTnsKit.on_device(value) isa SHTnsKit.CPU || throw(
        SHTnsKit.BackendUnavailableError(
            operation,
            "distributed reverse-mode AD coefficient cotangents must remain on the CPU for a host-backed PencilArray",
        ),
    )
    return ComplexF64.(value)
end

# ----- PencilArrays helpers -------------------------------------------------
# `communicator` and `globalindices` are internal helpers of the sibling
# SHTnsKitParallelExt module and are NOT exported by PencilArrays (verified
# across 0.19.8–0.19.11). This is a SEPARATE extension module, so without local
# definitions every rrule below throws `UndefVarError` the moment it fires.
# These mirror the primary 0.19 API used by the main extension.
@inline communicator(A) = PencilArrays.get_comm(A)

@inline globalindices(A, dim) = PencilArrays.range_local(PencilArrays.pencil(A))[dim]

# ----- helpers ---------------------------------------------------------------

"""
    _phi_window(φ_globals, nlon_local, cfg_nlon) -> (φ_is_local, φ_window)

The rank's global φ slice as a range, or `nothing` when φ is replicated.

A rank can legitimately own ZERO φ columns — a pencil with more partitions than
the dimension has points, e.g. `nlon = 4` on 5 ranks. The raw
`first(φ_globals)` throws `BoundsError` there, and because these rrules run
inside a collective region (the pullbacks `MPI.Allreduce` below) that kills one
rank while the others block forever, so the job hangs instead of failing. An
empty window is what the zero-pad path already wants: nothing is copied into
`f̄_full`, so this rank contributes an all-zero partial to the Allreduce.

Mirrors `_owned_range` in ext/ParallelTransforms.jl — duplicated because this is
a separate extension module and cannot see it.
"""
@inline function _phi_window(φ_globals, nlon_local::Int, cfg_nlon::Int)
    φ_is_local = (nlon_local == cfg_nlon)
    φ_is_local && return true, nothing
    isempty(φ_globals) && return false, 1:0
    φ_start = Int(first(φ_globals))
    return false, φ_start:(φ_start + nlon_local - 1)
end


# The rank-local adjoint is just the parametrized `SHTnsKit._adjoint_analysis`
# called with a restricted θ subset and optional φ-window.
@inline function _local_adjoint_analysis(cfg::SHTnsKit.SHTConfig, Alm̄,
                                          θ_globals::AbstractVector{<:Integer},
                                          φ_window)
    return SHTnsKit._adjoint_analysis(cfg, Alm̄; θ_globals=θ_globals, φ_window=φ_window)
end

"""Materialize and collectively verify one logically replicated cotangent."""
function _replicated_coeff_cotangent(cfg::SHTnsKit.SHTConfig, ȳ, comm;
                                     packed::Bool=false,
                                     operation::Symbol=:dist_analysis_pullback)
    ȳ = ChainRulesCore.unthunk(ȳ)
    is_zero = ȳ isa ChainRulesCore.AbstractZero

    # Every rank must take the same validation collectives. In particular, a
    # rank-local AbstractZero must not skip an Allreduce taken by nonzero peers:
    # that would mismatch this validation with the later Bcast and deadlock.
    local_host_ok = is_zero || try
        SHTnsKit.on_device(ȳ) isa SHTnsKit.CPU
    catch
        false
    end
    MPI.Allreduce(local_host_ok, &, comm) || throw(
        SHTnsKit.BackendUnavailableError(
            operation,
            "distributed reverse-mode AD coefficient cotangents must remain on the CPU for a host-backed PencilArray",
        ),
    )

    local_shape_ok = is_zero || try
        packed ? length(ȳ) == cfg.nlm :
                 (ȳ isa AbstractMatrix &&
                  size(ȳ) == (cfg.lmax + 1, cfg.mmax + 1))
    catch
        false
    end
    MPI.Allreduce(local_shape_ok, &, comm) || throw(DimensionMismatch(
        packed ? "distributed packed coefficient cotangent must have length $(cfg.nlm)" :
                 "distributed coefficient cotangent must have size $((cfg.lmax + 1, cfg.mmax + 1))",
    ))

    Alm̄ = nothing
    local_materialization_ok = true
    if is_zero
        Alm̄ = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    else
        Alm̄ = try
            converted = ComplexF64.(ȳ)
            packed ? SHTnsKit.unpack_lm(cfg, converted) : converted
        catch
            local_materialization_ok = false
            nothing
        end
    end
    MPI.Allreduce(local_materialization_ok, &, comm) || throw(ArgumentError(
        "distributed coefficient cotangent elements must be convertible to ComplexF64",
    ))

    if MPI.Comm_size(comm) > 1
        root_value = similar(Alm̄)
        if MPI.Comm_rank(comm) == 0
            copyto!(root_value, Alm̄)
        else
            fill!(root_value, zero(eltype(root_value)))
        end
        MPI.Bcast!(root_value, comm; root=0)
        same = Alm̄ == root_value
        MPI.Allreduce(same, &, comm) || throw(ArgumentError(
            "the cotangent of replicated distributed-analysis output must be identical on every rank; rank-varying partial cotangents are unsupported",
        ))
    end
    return Alm̄
end

@inline _cotangent_eltype_code(::Type{Float32}) = 1
@inline _cotangent_eltype_code(::Type{Float64}) = 2
@inline _cotangent_eltype_code(::Type{ComplexF32}) = 3
@inline _cotangent_eltype_code(::Type{ComplexF64}) = 4
@inline _cotangent_eltype_code(::Type) = 0

"""Collectively validate and materialize one rank-local spatial cotangent."""
function _local_spatial_cotangent(ȳ, prototype::PencilArray, comm;
                                   zero_eltype::Type,
                                   operation::Symbol)
    ȳ = ChainRulesCore.unthunk(ȳ)
    is_zero = ȳ isa ChainRulesCore.AbstractZero
    is_pencil = ȳ isa PencilArray
    local_flags = UInt32(0)
    local_value = nothing
    materialized = nothing

    # The adjoint result is subsequently reduced by MPI, so even otherwise
    # valid rank-local cotangents must be converted to one common element type.
    target_code = _cotangent_eltype_code(zero_eltype)
    root_target_code = MPI.bcast(target_code, 0, comm)
    (target_code != root_target_code || target_code == 0) &&
        (local_flags |= 0x0020)

    if !is_zero
        if is_pencil
            local_value = try
                parent(ȳ)
            catch
                local_flags |= 0x0001
                nothing
            end
        elseif ȳ isa AbstractMatrix
            local_value = ȳ
        else
            local_flags |= 0x0001
        end

        if local_value !== nothing
            host_ok = try
                SHTnsKit.on_device(local_value) isa SHTnsKit.CPU
            catch
                false
            end
            host_ok || (local_flags |= 0x0002)

            size_ok = try
                size(local_value) == size(parent(prototype))
            catch
                false
            end
            size_ok || (local_flags |= 0x0004)

            eltype_ok = try
                float(eltype(local_value))
                true
            catch
                false
            end
            eltype_ok || (local_flags |= 0x0010)

            if local_flags == 0
                materialized = try
                    eltype(local_value) === zero_eltype ? local_value :
                        Matrix{zero_eltype}(local_value)
                catch
                    local_flags |= 0x0010
                    nothing
                end
            end
        end

        if is_pencil
            layout_ok = try
                reference_pen = PencilArrays.pencil(prototype)
                candidate_pen = PencilArrays.pencil(ȳ)
                candidate_comm = communicator(ȳ)
                MPI.Comm_size(candidate_comm) == MPI.Comm_size(comm) &&
                    MPI.Comm_compare(candidate_comm, comm) in
                        (MPI.IDENT, MPI.CONGRUENT) &&
                    PencilArrays.size_global(ȳ) ==
                        PencilArrays.size_global(prototype) &&
                    PencilArrays.decomposition(candidate_pen) ==
                        PencilArrays.decomposition(reference_pen) &&
                    size(PencilArrays.topology(candidate_pen)) ==
                        size(PencilArrays.topology(reference_pen)) &&
                    PencilArrays.range_local(candidate_pen) ==
                        PencilArrays.range_local(reference_pen) &&
                    PencilArrays.permutation(ȳ) ==
                        PencilArrays.permutation(prototype)
            catch
                false
            end
            layout_ok || (local_flags |= 0x0008)
        end
    end

    # A single bitmask reduction gives every rank the same verdict before any
    # rank touches the cotangent's shape, eltype, or storage in the adjoint.
    flags = MPI.Allreduce(local_flags, |, comm)
    flags == 0 || begin
        flags & 0x0001 != 0 && throw(ArgumentError(
            "$operation spatial cotangent must be an AbstractMatrix, " *
            "PencilArray, or AbstractZero",
        ))
        flags & 0x0002 != 0 && throw(SHTnsKit.BackendUnavailableError(
            operation,
            "distributed reverse-mode AD spatial cotangents must remain on the CPU",
        ))
        flags & 0x0008 != 0 && throw(ArgumentError(
            "$operation PencilArray cotangent must match the spatial output layout and communicator",
        ))
        flags & 0x0004 != 0 && throw(DimensionMismatch(
            "$operation spatial cotangent must have rank-local size $(size(parent(prototype)))",
        ))
        throw(ArgumentError(
            "$operation spatial cotangent must have a floating-point-compatible element type",
        ))
    end

    return is_zero ? zeros(zero_eltype, size(parent(prototype))) : materialized
end

"""Collectively unpack a two-component spatial cotangent."""
function _cotangent_pair(ȳ, comm, operation::Symbol)
    ȳ = ChainRulesCore.unthunk(ȳ)
    if ȳ isa ChainRulesCore.AbstractZero
        first_component = ȳ
        second_component = ȳ
        local_ok = true
    else
        local_ok = true
        first_component = second_component = ChainRulesCore.ZeroTangent()
        try
            first_component = ChainRulesCore.unthunk(ȳ[1])
            second_component = ChainRulesCore.unthunk(ȳ[2])
        catch
            local_ok = false
        end
    end
    MPI.Allreduce(local_ok, &, comm) || throw(ArgumentError(
        "$operation cotangent must contain two spatial components",
    ))
    return first_component, second_component
end

function _require_ad_communicator_match(spectral::PencilArray,
                                        spatial::PencilArray)
    # Reduce the verdict on the spatial prototype's communicator. Throwing on
    # only the rank whose spectral operand uses COMM_SELF would strand peers.
    comm = communicator(spatial)
    local_ok = try
        MPI.Comm_compare(communicator(spectral), comm) in
            (MPI.IDENT, MPI.CONGRUENT)
    catch
        false
    end
    MPI.Allreduce(local_ok, &, comm) || throw(ArgumentError(
        "spectral and spatial PencilArrays must use communicators with the same process group and rank order",
    ))
    return nothing
end

"""Scatter a dense coefficient cotangent into a primal spectral pencil."""
function _scatter_spectral_tangent(primal::PencilArray, dense::AbstractMatrix)
    lr = collect(globalindices(primal, 1))
    mr = collect(globalindices(primal, 2))
    raw = Matrix{eltype(dense)}(undef, length(lr), length(mr))
    @inbounds for (jj, gm) in enumerate(mr), (ii, gl) in enumerate(lr)
        raw[ii, jj] = dense[gl, gm]
    end
    local_parent = ProjectTo(parent(primal))(raw)
    return PencilArray(PencilArrays.pencil(primal), local_parent)
end

# ----- dist_analysis rrule ---------------------------------------------------

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_analysis),
                              cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                              use_tables=cfg.use_plm_tables,
                              use_rfft::Bool=false,
                              use_packed_storage::Bool=false)
    comm = communicator(fθφ)
    _require_host_pencil(:dist_analysis_pullback, fθφ, comm)
    y = SHTnsKit.dist_analysis(cfg, fθφ;
                               use_tables, use_rfft, use_packed_storage)
    θ_globals = collect(globalindices(fθφ, 1))
    φ_globals = collect(globalindices(fθφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)
    project_f_parent = ProjectTo(parent(fθφ))

    function dist_analysis_pullback(ȳ)
        Alm̄ = _replicated_coeff_cotangent(
            cfg, ȳ, comm; packed=use_packed_storage,
            operation=:dist_analysis_pullback)
        f̄_parent = _local_adjoint_analysis(cfg, Alm̄, θ_globals, φ_window)
        # Wrap in a PencilArray sharing fθφ's pencil so downstream grads stay distributed.
        f̄ = PencilArray(PencilArrays.pencil(fθφ), project_f_parent(f̄_parent))
        return NoTangent(), NoTangent(), f̄
    end
    return y, dist_analysis_pullback
end

# ----- dist_synthesis rrule --------------------------------------------------

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis),
                              cfg::SHTnsKit.SHTConfig, Alm::PencilArray;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false)
    comm = communicator(prototype_θφ)
    _require_host_pencil(:dist_synthesis_pullback, Alm, comm)
    _require_host_pencil(:dist_synthesis_pullback, prototype_θφ, comm)
    _require_ad_communicator_match(Alm, prototype_θφ)
    y = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output, use_rfft)
    θ_globals = collect(globalindices(prototype_θφ, 1))
    φ_globals = collect(globalindices(prototype_θφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)

    function dist_synthesis_pencil_pullback(ȳ)
        nθ_local = length(θ_globals)
        ȳ_loc = _local_spatial_cotangent(
            ȳ, prototype_θφ, comm;
            zero_eltype=eltype(y),
            operation=:dist_synthesis_pullback,
        )
        f̄_full = zeros(float(eltype(ȳ_loc)), nθ_local, cfg.nlon)
        if φ_is_local
            f̄_full .= ȳ_loc
        else
            @views f̄_full[:, φ_window] .= ȳ_loc
        end
        Āpartial = SHTnsKit._adjoint_synthesis(
            cfg, f̄_full; θ_globals=θ_globals, real_output=real_output)
        Ādense = MPI.Allreduce(Āpartial, +, comm)
        Ā = _scatter_spectral_tangent(Alm, Ādense)
        return NoTangent(), NoTangent(), Ā
    end
    return y, dist_synthesis_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis),
                              cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false)
    comm = communicator(prototype_θφ)
    _require_host_pencil(:dist_synthesis_pullback, prototype_θφ, comm)
    y = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output, use_rfft)
    project_Alm = ProjectTo(Alm)
    θ_globals = collect(globalindices(prototype_θφ, 1))
    φ_globals = collect(globalindices(prototype_θφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)

    function dist_synthesis_pullback(ȳ)
        # Adjoint of synthesis is `_adjoint_synthesis` (NO quadrature weights /
        # cphi — those belong to analysis), applied on this rank's θ slab over a
        # FULL-nlon-width cotangent, then Allreduce-summed across ranks to build
        # the replicated Ālm. Using `dist_analysis` here would wrongly inject the
        # Gauss weights `w[θ]·cphi`. The forward ifft is over the full φ width and
        # only then sliced, so we zero-pad the local φ window back to full nlon;
        # FFT linearity makes Σ_ranks fft(padded window) = fft(full field).
        nθ_local = length(θ_globals)
        ȳ_loc = _local_spatial_cotangent(
            ȳ, prototype_θφ, comm;
            zero_eltype=eltype(y),
            operation=:dist_synthesis_pullback,
        )
        ET = float(eltype(ȳ_loc))  # real for real_output, complex otherwise
        f̄_full = zeros(ET, nθ_local, cfg.nlon)
        if φ_is_local
            f̄_full .= ȳ_loc
        else
            @views f̄_full[:, φ_window] .= ȳ_loc
        end
        Ālm_partial = SHTnsKit._adjoint_synthesis(cfg, f̄_full;
                                                  θ_globals=θ_globals,
                                                  real_output=real_output)
        Ālm = project_Alm(MPI.Allreduce(Ālm_partial, +, comm))
        return NoTangent(), NoTangent(), Ālm
    end
    return y, dist_synthesis_pullback
end

# ----- dist_analysis_sphtor rrule -------------------------------------------
# Adjoint: analogous to scalar dist_analysis. (Slm̄, Tlm̄) arrive replicated;
# each rank reconstructs its own (V̄t, V̄p) θ rows × φ window locally using
# the shared `_adjoint_analysis_sphtor` primitive, no inter-rank comms needed.

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_analysis_sphtor),
                              cfg::SHTnsKit.SHTConfig,
                              Vtθφ::PencilArray, Vpθφ::PencilArray;
                              kwargs...)
    comm = communicator(Vtθφ)
    _require_host_pencil(:dist_analysis_sphtor_pullback, Vtθφ, comm)
    _require_host_pencil(:dist_analysis_sphtor_pullback, Vpθφ, comm)
    y = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ; kwargs...)
    θ_globals = collect(globalindices(Vtθφ, 1))
    φ_globals = collect(globalindices(Vtθφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)
    project_Vt_parent = ProjectTo(parent(Vtθφ))
    project_Vp_parent = ProjectTo(parent(Vpθφ))

    function dist_analysis_sphtor_pullback(ȳ)
        # Unthunk each component and materialise only after the host-storage
        # guard. This preserves CPU AD without making a hidden vendor→host copy.
        Slm̄, Tlm̄ = _cotangent_pair(
            ȳ, comm, :dist_analysis_sphtor_pullback,
        )
        S̄in = _replicated_coeff_cotangent(
            cfg, Slm̄, comm; operation=:dist_analysis_sphtor_pullback)
        T̄in = _replicated_coeff_cotangent(
            cfg, Tlm̄, comm; operation=:dist_analysis_sphtor_pullback)
        V̄t_parent, V̄p_parent = SHTnsKit._adjoint_analysis_sphtor(
            cfg, S̄in, T̄in;
            θ_globals=θ_globals, φ_window=φ_window)
        V̄t = PencilArray(PencilArrays.pencil(Vtθφ), project_Vt_parent(V̄t_parent))
        V̄p = PencilArray(PencilArrays.pencil(Vpθφ), project_Vp_parent(V̄p_parent))
        return NoTangent(), NoTangent(), V̄t, V̄p
    end
    return y, dist_analysis_sphtor_pullback
end

# ----- dist_synthesis_sphtor rrule ------------------------------------------
# Adjoint: analogous to scalar. dist_analysis_sphtor on the spatial cotangents
# performs the per-rank analysis + Allreduce to produce replicated (Ālm_S, Ālm_T).

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis_sphtor),
                              cfg::SHTnsKit.SHTConfig,
                              Slm::PencilArray, Tlm::PencilArray;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false)
    comm = communicator(prototype_θφ)
    _require_host_pencil(:dist_synthesis_sphtor_pullback, Slm, comm)
    _require_host_pencil(:dist_synthesis_sphtor_pullback, Tlm, comm)
    _require_host_pencil(
        :dist_synthesis_sphtor_pullback, prototype_θφ, comm,
    )
    _require_ad_communicator_match(Slm, prototype_θφ)
    _require_ad_communicator_match(Tlm, prototype_θφ)
    y = SHTnsKit.dist_synthesis_sphtor(
        cfg, Slm, Tlm; prototype_θφ, real_output, use_rfft)
    θ_globals = collect(globalindices(prototype_θφ, 1))
    φ_globals = collect(globalindices(prototype_θφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)

    function dist_synthesis_sphtor_pencil_pullback(ȳ)
        V̄t, V̄p = _cotangent_pair(
            ȳ, comm, :dist_synthesis_sphtor_pullback,
        )
        nθ_local = length(θ_globals)
        V̄t_loc = _local_spatial_cotangent(
            V̄t, prototype_θφ, comm; zero_eltype=eltype(y[1]),
            operation=:dist_synthesis_sphtor_pullback,
        )
        V̄p_loc = _local_spatial_cotangent(
            V̄p, prototype_θφ, comm; zero_eltype=eltype(y[2]),
            operation=:dist_synthesis_sphtor_pullback,
        )
        V̄t_full = zeros(float(eltype(V̄t_loc)), nθ_local, cfg.nlon)
        V̄p_full = zeros(float(eltype(V̄p_loc)), nθ_local, cfg.nlon)
        if φ_is_local
            V̄t_full .= V̄t_loc
            V̄p_full .= V̄p_loc
        else
            @views V̄t_full[:, φ_window] .= V̄t_loc
            @views V̄p_full[:, φ_window] .= V̄p_loc
        end
        S̄partial, T̄partial = SHTnsKit._adjoint_synthesis_sphtor(
            cfg, V̄t_full, V̄p_full;
            θ_globals=θ_globals, real_output=real_output)
        n = length(S̄partial)
        combined = MPI.Allreduce!(vcat(vec(S̄partial), vec(T̄partial)), +, comm)
        copyto!(S̄partial, 1, combined, 1, n)
        copyto!(T̄partial, 1, combined, n + 1, length(combined) - n)
        S̄ = _scatter_spectral_tangent(Slm, S̄partial)
        T̄ = _scatter_spectral_tangent(Tlm, T̄partial)
        return NoTangent(), NoTangent(), S̄, T̄
    end
    return y, dist_synthesis_sphtor_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis_sphtor),
                              cfg::SHTnsKit.SHTConfig,
                              Slm::AbstractMatrix, Tlm::AbstractMatrix;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false)
    comm = communicator(prototype_θφ)
    _require_host_pencil(
        :dist_synthesis_sphtor_pullback, prototype_θφ, comm,
    )
    y = SHTnsKit.dist_synthesis_sphtor(cfg, Slm, Tlm;
                                        prototype_θφ=prototype_θφ,
                                        real_output=real_output,
                                        use_rfft=use_rfft)
    project_Slm = ProjectTo(Slm)
    project_Tlm = ProjectTo(Tlm)
    θ_globals = collect(globalindices(prototype_θφ, 1))
    φ_globals = collect(globalindices(prototype_θφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)

    function dist_synthesis_sphtor_pullback(ȳ)
        # Unthunk each component (a Tangent tuple of thunks would otherwise
        # slip through), while making malformed tuples a collective error.
        V̄t, V̄p = _cotangent_pair(
            ȳ, comm, :dist_synthesis_sphtor_pullback,
        )
        # Adjoint of vector synthesis is `_adjoint_synthesis_sphtor` (no quadrature
        # weights), applied per-rank on the local θ slab over a full-nlon-width
        # (zero-padded) cotangent, then Allreduce-summed. Previously this called
        # `dist_analysis_sphtor`, which injects Gauss weights `w[θ]·scaleφ/(l(l+1))`
        # that the synthesis adjoint must NOT carry.
        # Zero spatial cotangents likewise: materialise before touching eltype.
        V̄t_loc = _local_spatial_cotangent(
            V̄t, prototype_θφ, comm; zero_eltype=eltype(y[1]),
            operation=:dist_synthesis_sphtor_pullback,
        )
        V̄p_loc = _local_spatial_cotangent(
            V̄p, prototype_θφ, comm; zero_eltype=eltype(y[2]),
            operation=:dist_synthesis_sphtor_pullback,
        )
        nθ_local = length(θ_globals)
        ETt = float(eltype(V̄t_loc)); ETp = float(eltype(V̄p_loc))
        V̄t_full = zeros(ETt, nθ_local, cfg.nlon)
        V̄p_full = zeros(ETp, nθ_local, cfg.nlon)
        if φ_is_local
            V̄t_full .= V̄t_loc
            V̄p_full .= V̄p_loc
        else
            @views V̄t_full[:, φ_window] .= V̄t_loc
            @views V̄p_full[:, φ_window] .= V̄p_loc
        end
        S̄p, T̄p = SHTnsKit._adjoint_synthesis_sphtor(cfg, V̄t_full, V̄p_full;
                                                    θ_globals=θ_globals,
                                                    real_output=real_output)
        # One batched Allreduce over stacked (S̄,T̄) instead of two round-trips.
        n = length(S̄p)
        combined = MPI.Allreduce!(vcat(vec(S̄p), vec(T̄p)), +, comm)
        # Scatter back into the matrices `_adjoint_synthesis_sphtor` already
        # allocated. `combined[1:n]` would allocate two more full-size arrays on
        # top of the vcat (~12 MB per backward pass at lmax=511), and returning
        # `reshape(view(combined, …))` avoids that but hands back a ReshapedArray
        # rather than a Matrix — a downstream distributed pullback that feeds the
        # tangent straight to `MPI.Allreduce!` then fails buffer conversion.
        # copyto! reuses S̄p/T̄p, so this is both allocation-free and an Array.
        copyto!(S̄p, 1, combined, 1, n)
        copyto!(T̄p, 1, combined, n + 1, length(combined) - n)
        return NoTangent(), NoTangent(), project_Slm(S̄p), project_Tlm(T̄p)
    end
    return y, dist_synthesis_sphtor_pullback
end

end # module
