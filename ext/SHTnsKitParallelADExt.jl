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

# ----- helpers ---------------------------------------------------------------

# The rank-local adjoint is just the parametrized `SHTnsKit._adjoint_analysis`
# called with a restricted θ subset and optional φ-window.
@inline function _local_adjoint_analysis(cfg::SHTnsKit.SHTConfig, Alm̄,
                                          θ_globals::AbstractVector{<:Integer},
                                          nlon_local::Int, φ_start::Int, φ_is_local::Bool)
    φ_window = φ_is_local ? nothing : (φ_start:(φ_start + nlon_local - 1))
    return SHTnsKit._adjoint_analysis(cfg, Alm̄; θ_globals=θ_globals, φ_window=φ_window)
end

# ----- dist_analysis rrule ---------------------------------------------------

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_analysis),
                              cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                              kwargs...)
    y = SHTnsKit.dist_analysis(cfg, fθφ; kwargs...)
    θ_globals = collect(globalindices(fθφ, 1))
    φ_globals = collect(globalindices(fθφ, 2))
    nlon_local = length(φ_globals)
    φ_start = first(φ_globals)
    φ_is_local = (nlon_local == cfg.nlon)

    function dist_analysis_pullback(ȳ)
        # Alm̄ may arrive replicated (standard) or as a partial tangent.
        Alm̄ = ȳ isa AbstractMatrix ? ȳ : collect(ȳ)
        f̄_parent = _local_adjoint_analysis(cfg, Alm̄, θ_globals,
                                            nlon_local, φ_start, φ_is_local)
        # Wrap in a PencilArray sharing fθφ's pencil so downstream grads stay distributed.
        f̄ = PencilArray(fθφ.pencil, f̄_parent)
        return NoTangent(), NoTangent(), f̄
    end
    return y, dist_analysis_pullback
end

# ----- dist_synthesis rrule --------------------------------------------------

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis),
                              cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false)
    y = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output, use_rfft)
    comm = communicator(prototype_θφ)

    function dist_synthesis_pullback(ȳ)
        # ȳ: spatial cotangent (PencilArray or Array). Adjoint of synthesis is
        # analysis of the cotangent summed across ranks to reconstruct the
        # replicated Ālm.
        ȳ_parent = ȳ isa PencilArray ? parent(ȳ) : Matrix{eltype(Alm)}(ȳ)
        # Per-rank partial analysis restricted to local θ rows & full φ
        # (dist_analysis handles both single-rank and distributed inputs).
        ȳ_pa = ȳ isa PencilArray ? ȳ : PencilArray(prototype_θφ.pencil, ȳ_parent)
        Ālm = SHTnsKit.dist_analysis(cfg, ȳ_pa; use_rfft=use_rfft)
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
    y = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ; kwargs...)
    θ_globals = collect(globalindices(Vtθφ, 1))
    φ_globals = collect(globalindices(Vtθφ, 2))
    nlon_local = length(φ_globals)
    φ_start = first(φ_globals)
    φ_is_local = (nlon_local == cfg.nlon)
    φ_window = φ_is_local ? nothing : (φ_start:(φ_start + nlon_local - 1))

    function dist_analysis_sphtor_pullback(ȳ)
        Slm̄, Tlm̄ = ȳ isa Tuple ? ȳ : (ȳ[1], ȳ[2])
        V̄t_parent, V̄p_parent = SHTnsKit._adjoint_analysis_sphtor(
            cfg, Matrix{ComplexF64}(Slm̄), Matrix{ComplexF64}(Tlm̄);
            θ_globals=θ_globals, φ_window=φ_window)
        V̄t = PencilArray(Vtθφ.pencil, V̄t_parent)
        V̄p = PencilArray(Vpθφ.pencil, V̄p_parent)
        return NoTangent(), NoTangent(), V̄t, V̄p
    end
    return y, dist_analysis_sphtor_pullback
end

# ----- dist_synthesis_sphtor rrule ------------------------------------------
# Adjoint: analogous to scalar. dist_analysis_sphtor on the spatial cotangents
# performs the per-rank analysis + Allreduce to produce replicated (Ālm_S, Ālm_T).

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis_sphtor),
                              cfg::SHTnsKit.SHTConfig,
                              Slm::AbstractMatrix, Tlm::AbstractMatrix;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false)
    y = SHTnsKit.dist_synthesis_sphtor(cfg, Slm, Tlm;
                                        prototype_θφ=prototype_θφ,
                                        real_output=real_output,
                                        use_rfft=use_rfft)

    function dist_synthesis_sphtor_pullback(ȳ)
        V̄t, V̄p = ȳ isa Tuple ? ȳ : (ȳ[1], ȳ[2])
        V̄t_pa = V̄t isa PencilArray ? V̄t :
                PencilArray(prototype_θφ.pencil, Matrix{Float64}(V̄t))
        V̄p_pa = V̄p isa PencilArray ? V̄p :
                PencilArray(prototype_θφ.pencil, Matrix{Float64}(V̄p))
        S̄, T̄ = SHTnsKit.dist_analysis_sphtor(cfg, V̄t_pa, V̄p_pa; use_rfft=use_rfft)
        return NoTangent(), NoTangent(), S̄, T̄
    end
    return y, dist_synthesis_sphtor_pullback
end

end # module
