module SHTnsKitAMDGPUExt

using SHTnsKit
using AMDGPU
using GPUArrays
using GPUArraysCore
using KernelAbstractions
using FFTW

include("GPUCommon.jl")
using .GPUCommon: legendre_table_kernel!, scalar_analysis_kernel!,
                  scalar_synthesis_kernel!, coefficient_conversion_kernel!,
                  coefficient_batch_conversion_kernel!,
                  real_pack_kernel!, real_unpack_kernel!,
                  mode_analysis_kernel!, mode_synthesis_kernel!,
                  scalar_batch_analysis_kernel!, scalar_batch_synthesis_kernel!,
                  complex_packed_analysis_kernel!, complex_packed_synthesis_kernel!,
                  scalar_config_signature, scalar_host_tables,
                  ScalarTableCache, scalar_cache_lookup, scalar_cache_insert!,
                  scalar_cache_clear!, scalar_cache_size
using .GPUCommon: ScalarWorkspaceCache, scalar_workspace_use!,
                  scalar_workspace_clear!, scalar_workspace_size

import SHTnsKit: analysis, synthesis, synthesis_cplx, on_device,
                 analysis_packed, synthesis_packed,
                 analysis_packed_l, synthesis_packed_l,
                 analysis_axisym, synthesis_axisym,
                 analysis_axisym_l, synthesis_axisym_l,
                 analysis_packed_ml, synthesis_packed_ml,
                 analysis_packed_cplx, synthesis_packed_cplx,
                 analysis_packed_cplx_l, synthesis_packed_cplx_l,
                 analysis_batch, analysis_batch!,
                 synthesis_batch, synthesis_batch!, synthesis_batch_cplx,
                 analysis!, synthesis!,
                 _register_gpu_adapter!, _gpu_adapter_functional,
                 _gpu_adapter_matches, _gpu_adapter_adapt,
                 _gpu_adapter_analysis, _gpu_adapter_synthesis,
                 _gpu_adapter_clear_cache!

mutable struct AMDGPUAdapter end
const AMDGPU_ADAPTER = AMDGPUAdapter()

function __init__()
    _register_gpu_adapter!(:amdgpu, AMDGPU_ADAPTER)
    return nothing
end

_gpu_adapter_functional(::AMDGPUAdapter) = AMDGPU.functional()
_gpu_adapter_matches(::AMDGPUAdapter, ::AMDGPU.AnyROCArray) = true

function _gpu_adapter_adapt(::AMDGPUAdapter, value)
    AMDGPU.functional() || throw(SHTnsKit.BackendUnavailableError(
        :to_device,
        "AMDGPU.jl is loaded but AMDGPU.functional() is false",
    ))
    return ROCArray(value)
end

function _require_amdgpu(operation::Symbol)
    AMDGPU.functional() || throw(SHTnsKit.BackendUnavailableError(
        operation,
        "AMDGPU.jl is loaded but AMDGPU.functional() is false",
    ))
    AMDGPU.functional(:rocfft) || throw(SHTnsKit.BackendUnavailableError(
        operation,
        "AMDGPU is functional but rocFFT is unavailable",
    ))
    return nothing
end

struct AMDGPUScalarTables{TX,TW,TP,TS}
    x::TX
    weights::TW
    Plm::TP
    scales::TS
end

const _AMDGPU_SCALAR_CACHE = ScalarTableCache(8)
const _AMDGPU_WORKSPACE_CACHE = ScalarWorkspaceCache(8)

function _amdgpu_scalar_tables(cfg::SHTConfig, ::Type{T}) where {T<:AbstractFloat}
    device = AMDGPU.device_id()
    identity = objectid(cfg)
    signature = scalar_config_signature(cfg)
    cached = scalar_cache_lookup(
        _AMDGPU_SCALAR_CACHE, device, identity, T, signature,
    )
    cached === nothing || return cached

    x_host, weights_host, scales_host = scalar_host_tables(cfg, T)
    x = ROCArray(x_host)
    weights = ROCArray(weights_host)
    scales = ROCArray(scales_host)
    Plm = AMDGPU.zeros(T, cfg.nlat, cfg.lmax + 1, cfg.mmax + 1)
    backend = ROCBackend()
    kernel! = legendre_table_kernel!(backend)
    kernel!(Plm, x, cfg.lmax, cfg.mmax;
            ndrange=(cfg.nlat, cfg.mmax + 1))
    AMDGPU.synchronize()
    built = AMDGPUScalarTables(x, weights, Plm, scales)

    return scalar_cache_insert!(
        _AMDGPU_SCALAR_CACHE, device, identity, T, signature, built,
    )
end

function _amdgpu_clear_scalar_cache!(; device=nothing)
    scalar_cache_clear!(_AMDGPU_SCALAR_CACHE; device)
    scalar_workspace_clear!(_AMDGPU_WORKSPACE_CACHE; device)
    return nothing
end

function _gpu_adapter_clear_cache!(::AMDGPUAdapter)
    _amdgpu_clear_scalar_cache!()
    return nothing
end

function _amdgpu_workspace_builder(cfg::SHTConfig, ::Type{RT}, nfields::Int,
                                   use_rfft::Bool) where {RT<:AbstractFloat}
    CT = Complex{RT}
    spatial_shape = nfields == 0 ? (cfg.nlat, cfg.nlon) :
                                   (cfg.nlat, cfg.nlon, nfields)
    spectral_shape = nfields == 0 ? (cfg.lmax + 1, cfg.mmax + 1) :
                                    (cfg.lmax + 1, cfg.mmax + 1, nfields)
    canonical = AMDGPU.zeros(CT, spectral_shape)
    if use_rfft
        real_buffer = AMDGPU.zeros(RT, spatial_shape)
        half_shape = Base.setindex(spatial_shape, cfg.nlon ÷ 2 + 1, 2)
        fourier = AMDGPU.zeros(CT, half_shape)
        forward = FFTW.plan_rfft(real_buffer, 2)
        inverse = FFTW.plan_irfft(fourier, cfg.nlon, 2)
        return (; canonical, fourier, real_buffer, forward, inverse)
    end
    fourier = AMDGPU.zeros(CT, spatial_shape)
    forward = FFTW.plan_fft!(fourier, 2)
    inverse = FFTW.plan_ifft!(fourier, 2)
    return (; canonical, fourier, real_buffer=nothing, forward, inverse)
end

function _with_amdgpu_workspace(f, owner, cfg::SHTConfig, ::Type{RT},
                                nfields::Int, use_rfft::Bool) where {RT<:AbstractFloat}
    device = AMDGPU.device_id()
    kind = nfields == 0 ? :scalar : :batch
    shape = (cfg.nlat, cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres,
             nfields, use_rfft)
    signature = hash((scalar_config_signature(cfg), use_rfft, nfields))
    builder = () -> _amdgpu_workspace_builder(cfg, RT, nfields, use_rfft)
    return scalar_workspace_use!(
        f, builder, _AMDGPU_WORKSPACE_CACHE, device, owner, RT,
        kind, shape, signature,
    )
end

function _amdgpu_batch_scratch(cfg::SHTConfig, fft_batch, ::Type{CT},
                               nfields::Int, use_rfft::Bool, operands...) where {CT<:Complex}
    fft_batch === nothing && return nothing
    fft_batch isa AMDGPU.AnyROCArray || throw(ArgumentError(
        "fft_batch must use AMDGPU storage",
    ))
    nbins = use_rfft ? cfg.nlon ÷ 2 + 1 : cfg.nlon
    size(fft_batch) == (cfg.nlat, nbins, nfields) || throw(DimensionMismatch(
        "fft_batch size must be ($(cfg.nlat), $nbins, $nfields)",
    ))
    eltype(fft_batch) === CT || throw(ArgumentError(
        "fft_batch must have element type $CT",
    ))
    any(value -> Base.mightalias(fft_batch, value), operands) &&
        throw(ArgumentError("fft_batch must not alias transform input or output"))
    return fft_batch
end

function _amdgpu_scalar_analysis_direct!(owner, cfg::SHTConfig,
                                         output::AMDGPU.AnyROCArray,
                                         field::AMDGPU.AnyROCArray;
                                         use_rfft::Bool=false)
    _require_amdgpu(:analysis!)
    size(field) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "field must have size ($(cfg.nlat), $(cfg.nlon))",
    ))
    size(output) == (cfg.lmax + 1, cfg.mmax + 1) ||
        throw(DimensionMismatch("plan analysis output shape mismatch"))
    use_rfft && !(eltype(field) <: Real) && throw(ArgumentError(
        "use_rfft=true requires a real-valued input",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2",
    ))
    RT = typeof(float(real(zero(eltype(field)))))
    eltype(output) <: Complex || throw(ArgumentError("analysis output must be complex"))
    tables = _amdgpu_scalar_tables(cfg, RT)
    return _with_amdgpu_workspace(owner, cfg, RT, 0, use_rfft) do workspace
        fourier = workspace.fourier
        if use_rfft
            copyto!(workspace.real_buffer, field)
            mul!(fourier, workspace.forward, workspace.real_buffer)
        else
            copyto!(fourier, field)
            mul!(fourier, workspace.forward, fourier)
        end
        backend = ROCBackend()
        scalar_analysis_kernel!(backend)(
            output, fourier, tables.Plm, tables.weights, RT(cfg.cphi),
            cfg.lmax, cfg.mmax, cfg.mres, cfg.lmax;
            ndrange=(cfg.lmax + 1, cfg.mmax + 1),
        )
        coefficient_conversion_kernel!(backend)(
            output, output, tables.scales, cfg.lmax, cfg.mmax, false;
            ndrange=(cfg.lmax + 1, cfg.mmax + 1),
        )
        AMDGPU.synchronize()
        output
    end
end

function _amdgpu_scalar_synthesis_direct!(owner, cfg::SHTConfig,
                                          output::AMDGPU.AnyROCArray,
                                          coefficients::AMDGPU.AnyROCArray;
                                          real_output::Bool=true,
                                          use_rfft::Bool=false)
    _require_amdgpu(:synthesis!)
    size(coefficients) == (cfg.lmax + 1, cfg.mmax + 1) ||
        throw(DimensionMismatch("plan coefficient shape mismatch"))
    size(output) == (cfg.nlat, cfg.nlon) ||
        throw(DimensionMismatch("plan synthesis output shape mismatch"))
    !real_output && eltype(output) <: Real && throw(ArgumentError(
        "real plan output storage requires real_output=true",
    ))
    use_rfft && (!real_output || !(eltype(output) <: Real)) && throw(ArgumentError(
        "use_rfft=true requires real-valued output",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2",
    ))
    RT = typeof(float(real(zero(eltype(coefficients)))))
    tables = _amdgpu_scalar_tables(cfg, RT)
    return _with_amdgpu_workspace(owner, cfg, RT, 0, use_rfft) do workspace
        backend = ROCBackend()
        coefficient_conversion_kernel!(backend)(
            workspace.canonical, coefficients, tables.scales,
            cfg.lmax, cfg.mmax, true;
            ndrange=(cfg.lmax + 1, cfg.mmax + 1),
        )
        fill!(workspace.fourier, zero(eltype(workspace.fourier)))
        scalar_synthesis_kernel!(backend)(
            workspace.fourier, workspace.canonical, tables.Plm,
            RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon,
            cfg.lmax, cfg.mmax, cfg.mres, real_output && !use_rfft;
            ndrange=(cfg.nlat, cfg.mmax + 1),
        )
        AMDGPU.synchronize()
        if use_rfft
            mul!(workspace.real_buffer, workspace.inverse, workspace.fourier)
            copyto!(output, workspace.real_buffer)
        else
            mul!(workspace.fourier, workspace.inverse, workspace.fourier)
            real_output ? (output .= real.(workspace.fourier)) :
                          copyto!(output, workspace.fourier)
        end
        AMDGPU.synchronize()
        output
    end
end

function _amdgpu_scalar_analysis(cfg::SHTConfig, field::AMDGPU.AnyROCArray;
                                 use_rfft::Bool=false, fft_scratch=nothing,
                                 lcap::Int=cfg.lmax)
    0 ≤ lcap ≤ cfg.lmax || throw(ArgumentError(
        "lcap must satisfy 0 ≤ lcap ≤ lmax=$(cfg.lmax)",
    ))
    _require_amdgpu(:analysis)
    size(field) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "field must have size ($(cfg.nlat), $(cfg.nlon)), got $(size(field))",
    ))
    fft_scratch === nothing || throw(ArgumentError(
        "AMDGPU scalar transforms do not accept a host fft_scratch",
    ))
    use_rfft && !(eltype(field) <: Real) && throw(ArgumentError(
        "use_rfft=true requires a real-valued input",
    ))
    RT = typeof(float(real(zero(eltype(field)))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    # `use_rfft` is a performance hint. The current rocFFT adapter deliberately
    # shares the complex pipeline for valid real transforms.
    fourier = CT.(field)
    FFTW.fft!(fourier, 2)

    backend = ROCBackend()
    canonical = AMDGPU.zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    analyze! = scalar_analysis_kernel!(backend)
    analyze!(canonical, fourier, tables.Plm, tables.weights, RT(cfg.cphi),
             cfg.lmax, cfg.mmax, cfg.mres, lcap;
             ndrange=(lcap + 1, min(cfg.mmax, lcap) + 1))

    configured = similar(canonical)
    convert! = coefficient_conversion_kernel!(backend)
    convert!(configured, canonical, tables.scales, cfg.lmax, cfg.mmax, false;
             ndrange=(cfg.lmax + 1, cfg.mmax + 1))
    AMDGPU.synchronize()
    return configured
end

function _amdgpu_scalar_synthesis(cfg::SHTConfig,
                                  coefficients::AMDGPU.AnyROCArray;
                                  real_output::Bool=true, use_rfft::Bool=false,
                                  fft_scratch=nothing, lcap::Int=cfg.lmax)
    0 ≤ lcap ≤ cfg.lmax || throw(ArgumentError(
        "lcap must satisfy 0 ≤ lcap ≤ lmax=$(cfg.lmax)",
    ))
    _require_amdgpu(:synthesis)
    size(coefficients) == (cfg.lmax + 1, cfg.mmax + 1) || throw(DimensionMismatch(
        "coefficients must have size ($(cfg.lmax + 1), $(cfg.mmax + 1)), got $(size(coefficients))",
    ))
    fft_scratch === nothing || throw(ArgumentError(
        "AMDGPU scalar transforms do not accept a host fft_scratch",
    ))
    use_rfft && !real_output && throw(ArgumentError(
        "use_rfft=true implies real_output",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2, got mmax=$(cfg.mmax), nlon=$(cfg.nlon)",
    ))
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    configured = CT.(coefficients)
    canonical = similar(configured)
    backend = ROCBackend()
    convert! = coefficient_conversion_kernel!(backend)
    convert!(canonical, configured, tables.scales, cfg.lmax, cfg.mmax, true;
             ndrange=(cfg.lmax + 1, cfg.mmax + 1))

    # The complex rocFFT route is mathematically identical for this valid hint.
    fourier = AMDGPU.zeros(CT, cfg.nlat, cfg.nlon)
    synthesize! = scalar_synthesis_kernel!(backend)
    synthesize!(fourier, canonical, tables.Plm, RT(SHTnsKit.phi_inv_scale(cfg)),
                cfg.nlon, lcap, min(cfg.mmax, lcap), cfg.mres, real_output;
                ndrange=(cfg.nlat, min(cfg.mmax, lcap) + 1))
    AMDGPU.synchronize()
    FFTW.ifft!(fourier, 2)
    return real_output ? real.(fourier) : fourier
end

function _gpu_adapter_analysis(::AMDGPUAdapter, cfg::SHTConfig,
                               field::AMDGPU.AnyROCArray; kwargs...)
    return _amdgpu_scalar_analysis(cfg, field; kwargs...)
end

function _gpu_adapter_synthesis(::AMDGPUAdapter, cfg::SHTConfig,
                                coefficients::AMDGPU.AnyROCArray; kwargs...)
    return _amdgpu_scalar_synthesis(cfg, coefficients; kwargs...)
end

on_device(::AMDGPU.AnyROCArray) = SHTnsKit.GPU()

analysis(cfg::SHTConfig, field::AMDGPU.AnyROCArray{T,2}; kwargs...) where {T} =
    analysis(SHTnsKit.GPU(), cfg, field; kwargs...)
synthesis(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{T,2}; kwargs...) where {T} =
    synthesis(SHTnsKit.GPU(), cfg, coefficients; kwargs...)
synthesis_cplx(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{T,2}) where {T} =
    synthesis_cplx(SHTnsKit.GPU(), cfg, coefficients)

@inline function _amdgpu_lcap(cfg::SHTConfig, ltr::Integer)
    return SHTnsKit._validate_degree_limit(cfg, ltr)
end

function _amdgpu_pack_lm(cfg::SHTConfig, dense::AMDGPU.AnyROCArray, lcap::Int)
    packed = AMDGPU.zeros(eltype(dense), cfg.nlm)
    kernel! = real_pack_kernel!(ROCBackend())
    kernel!(packed, dense, cfg.lmax, cfg.mmax, cfg.mres, lcap;
            ndrange=(cfg.lmax + 1, cfg.mmax ÷ cfg.mres + 1))
    AMDGPU.synchronize()
    return packed
end


function _amdgpu_unpack_lm(cfg::SHTConfig, packed::AMDGPU.AnyROCArray, lcap::Int)
    length(packed) == cfg.nlm || throw(DimensionMismatch(
        "Qlm must have length $(cfg.nlm)",
    ))
    dense = AMDGPU.zeros(eltype(packed), cfg.lmax + 1, cfg.mmax + 1)
    kernel! = real_unpack_kernel!(ROCBackend())
    kernel!(dense, packed, cfg.lmax, cfg.mmax, cfg.mres, lcap;
            ndrange=size(dense))
    AMDGPU.synchronize()
    return dense
end

function analysis_packed(::SHTnsKit.GPU, cfg::SHTConfig,
                         field::AMDGPU.AnyROCArray{T,1}) where {T<:Real}
    _require_amdgpu(:analysis_packed)
    length(field) == cfg.nspat || throw(DimensionMismatch(
        "field must have length $(cfg.nspat)",
    ))
    return _amdgpu_pack_lm(
        cfg, _amdgpu_scalar_analysis(cfg, reshape(field, cfg.nlat, cfg.nlon)), cfg.lmax,
    )
end
analysis_packed(cfg::SHTConfig, field::AMDGPU.AnyROCArray{T,1}) where {T<:Real} =
    analysis_packed(SHTnsKit.GPU(), cfg, field)

function synthesis_packed(::SHTnsKit.GPU, cfg::SHTConfig,
                          coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex}
    _require_amdgpu(:synthesis_packed)
    return vec(_amdgpu_scalar_synthesis(
        cfg, _amdgpu_unpack_lm(cfg, coefficients, cfg.lmax); real_output=true,
    ))
end
synthesis_packed(cfg::SHTConfig,
                 coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex} =
    synthesis_packed(SHTnsKit.GPU(), cfg, coefficients)

function analysis_packed_l(::SHTnsKit.GPU, cfg::SHTConfig,
                           field::AMDGPU.AnyROCArray{T,1}, ltr::Integer) where {T<:Real}
    lcap = _amdgpu_lcap(cfg, ltr)
    length(field) == cfg.nspat || throw(DimensionMismatch(
        "field must have length $(cfg.nspat)",
    ))
    return _amdgpu_pack_lm(
        cfg, _amdgpu_scalar_analysis(
            cfg, reshape(field, cfg.nlat, cfg.nlon); lcap,
        ), lcap,
    )
end
analysis_packed_l(cfg::SHTConfig, field::AMDGPU.AnyROCArray{T,1},
                  ltr::Integer) where {T<:Real} =
    analysis_packed_l(SHTnsKit.GPU(), cfg, field, ltr)

function synthesis_packed_l(::SHTnsKit.GPU, cfg::SHTConfig,
                            coefficients::AMDGPU.AnyROCArray{T,1},
                            ltr::Integer) where {T<:Complex}
    lcap = _amdgpu_lcap(cfg, ltr)
    return vec(_amdgpu_scalar_synthesis(
        cfg, _amdgpu_unpack_lm(cfg, coefficients, lcap);
        real_output=true, lcap,
    ))
end
synthesis_packed_l(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    synthesis_packed_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _amdgpu_mode_analysis(cfg::SHTConfig, physical_m::Int,
                               mode::AMDGPU.AnyROCArray, lcap::Int, scale)
    RT = typeof(float(real(zero(eltype(mode)))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    canonical = AMDGPU.zeros(CT, lcap - physical_m + 1)
    kernel! = mode_analysis_kernel!(ROCBackend())
    kernel!(canonical, mode, tables.Plm, tables.weights, RT(scale), physical_m, lcap;
            ndrange=length(canonical))
    configured = canonical ./ @view(tables.scales[(physical_m + 1):(lcap + 1), physical_m + 1])
    AMDGPU.synchronize()
    return configured
end

function _amdgpu_mode_synthesis(cfg::SHTConfig, physical_m::Int,
                                coefficients::AMDGPU.AnyROCArray, lcap::Int, scale)
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    canonical = CT.(coefficients) .*
        @view(tables.scales[(physical_m + 1):(lcap + 1), physical_m + 1])
    mode = AMDGPU.zeros(CT, cfg.nlat)
    kernel! = mode_synthesis_kernel!(ROCBackend())
    kernel!(mode, canonical, tables.Plm, RT(scale), physical_m, lcap;
            ndrange=cfg.nlat)
    AMDGPU.synchronize()
    return mode
end

function analysis_axisym(::SHTnsKit.GPU, cfg::SHTConfig,
                         field::AMDGPU.AnyROCArray{T,1}) where {T<:Real}
    length(field) == cfg.nlat || throw(DimensionMismatch(
        "field must have length nlat=$(cfg.nlat)",
    ))
    return _amdgpu_mode_analysis(cfg, 0, field, cfg.lmax, cfg.cphi * cfg.nlon)
end
analysis_axisym(cfg::SHTConfig, field::AMDGPU.AnyROCArray{T,1}) where {T<:Real} =
    analysis_axisym(SHTnsKit.GPU(), cfg, field)

function synthesis_axisym(::SHTnsKit.GPU, cfg::SHTConfig,
                          coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex}
    length(coefficients) == cfg.lmax + 1 || throw(DimensionMismatch(
        "coefficients must have length lmax+1=$(cfg.lmax + 1)",
    ))
    return real.(_amdgpu_mode_synthesis(cfg, 0, coefficients, cfg.lmax, 1))
end
synthesis_axisym(cfg::SHTConfig,
                 coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex} =
    synthesis_axisym(SHTnsKit.GPU(), cfg, coefficients)

function analysis_axisym_l(::SHTnsKit.GPU, cfg::SHTConfig,
                           field::AMDGPU.AnyROCArray{T,1}, ltr::Integer) where {T<:Real}
    lcap = _amdgpu_lcap(cfg, ltr)
    length(field) == cfg.nlat || throw(DimensionMismatch(
        "field must have length nlat=$(cfg.nlat)",
    ))
    return _amdgpu_mode_analysis(cfg, 0, field, lcap, cfg.cphi * cfg.nlon)
end
analysis_axisym_l(cfg::SHTConfig, field::AMDGPU.AnyROCArray{T,1},
                  ltr::Integer) where {T<:Real} =
    analysis_axisym_l(SHTnsKit.GPU(), cfg, field, ltr)

function synthesis_axisym_l(::SHTnsKit.GPU, cfg::SHTConfig,
                            coefficients::AMDGPU.AnyROCArray{T,1},
                            ltr::Integer) where {T<:Complex}
    lcap = _amdgpu_lcap(cfg, ltr)
    length(coefficients) >= lcap + 1 || throw(DimensionMismatch(
        "coefficients must contain degrees 0:ltr",
    ))
    return real.(_amdgpu_mode_synthesis(
        cfg, 0, @view(coefficients[1:(lcap + 1)]), lcap, 1,
    ))
end
synthesis_axisym_l(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    synthesis_axisym_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _amdgpu_fixed_order(cfg::SHTConfig, im::Int, ltr::Integer)
    im >= 0 || throw(ArgumentError("im must be >= 0"))
    im <= cfg.mmax ÷ cfg.mres || throw(ArgumentError(
        "im must be <= mmax/mres=$(cfg.mmax ÷ cfg.mres)",
    ))
    lcap = _amdgpu_lcap(cfg, ltr)
    physical_m = im * cfg.mres
    lcap >= physical_m || throw(ArgumentError(
        "ltr must be >= im*mres=$(physical_m)",
    ))
    return physical_m, lcap
end

function analysis_packed_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Int,
                            mode::AMDGPU.AnyROCArray{T,1}, ltr::Integer) where {T<:Complex}
    physical_m, lcap = _amdgpu_fixed_order(cfg, im, ltr)
    length(mode) == cfg.nlat || throw(DimensionMismatch(
        "mode must have length nlat=$(cfg.nlat)",
    ))
    return _amdgpu_mode_analysis(cfg, physical_m, mode, lcap, cfg.cphi)
end
analysis_packed_ml(cfg::SHTConfig, im::Int, mode::AMDGPU.AnyROCArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    analysis_packed_ml(SHTnsKit.GPU(), cfg, im, mode, ltr)

function synthesis_packed_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Int,
                             coefficients::AMDGPU.AnyROCArray{T,1},
                             ltr::Integer) where {T<:Complex}
    physical_m, lcap = _amdgpu_fixed_order(cfg, im, ltr)
    length(coefficients) == lcap - physical_m + 1 || throw(DimensionMismatch(
        "coefficients have the wrong fixed-order length",
    ))
    return _amdgpu_mode_synthesis(
        cfg, physical_m, coefficients, lcap, SHTnsKit.phi_inv_scale(cfg),
    )
end
synthesis_packed_ml(cfg::SHTConfig, im::Int,
                    coefficients::AMDGPU.AnyROCArray{T,1},
                    ltr::Integer) where {T<:Complex} =
    synthesis_packed_ml(SHTnsKit.GPU(), cfg, im, coefficients, ltr)

function _amdgpu_analysis_packed_cplx(cfg::SHTConfig,
                                      field::AMDGPU.AnyROCArray{T,2},
                                      lcap::Int) where {T<:Complex}
    cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    size(field) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "field must have size ($(cfg.nlat), $(cfg.nlon))",
    ))
    RT = typeof(float(real(zero(T))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    fourier = CT.(field)
    FFTW.fft!(fourier, 2)
    packed = AMDGPU.zeros(CT, SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1))
    mcap = min(cfg.mmax, lcap)
    kernel! = complex_packed_analysis_kernel!(ROCBackend())
    kernel!(packed, fourier, tables.Plm, tables.weights, tables.scales,
            RT(cfg.cphi), cfg.nlon, lcap, cfg.mmax, mcap;
            ndrange=(lcap + 1, 2mcap + 1))
    AMDGPU.synchronize()
    return packed
end
function analysis_packed_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                              field::AMDGPU.AnyROCArray{T,2}) where {T<:Complex}
    return _amdgpu_analysis_packed_cplx(cfg, field, cfg.lmax)
end
analysis_packed_cplx(cfg::SHTConfig,
                     field::AMDGPU.AnyROCArray{T,2}) where {T<:Complex} =
    analysis_packed_cplx(SHTnsKit.GPU(), cfg, field)

function analysis_packed_cplx_l(::SHTnsKit.GPU, cfg::SHTConfig,
                                field::AMDGPU.AnyROCArray{T,2},
                                ltr::Integer) where {T<:Complex}
    return _amdgpu_analysis_packed_cplx(
        cfg, field, _amdgpu_lcap(cfg, ltr),
    )
end
analysis_packed_cplx_l(cfg::SHTConfig,
                       field::AMDGPU.AnyROCArray{T,2},
                       ltr::Integer) where {T<:Complex} =
    analysis_packed_cplx_l(SHTnsKit.GPU(), cfg, field, ltr)

function _amdgpu_synthesis_packed_cplx(cfg::SHTConfig,
                                       coefficients::AMDGPU.AnyROCArray{T,1},
                                       lcap::Int) where {T<:Complex}
    cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    expected = SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    length(coefficients) == expected || throw(DimensionMismatch(
        "coefficients must have length $expected",
    ))
    RT = typeof(float(real(zero(T))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    fourier = AMDGPU.zeros(CT, cfg.nlat, cfg.nlon)
    mcap = min(cfg.mmax, lcap)
    kernel! = complex_packed_synthesis_kernel!(ROCBackend())
    kernel!(fourier, coefficients, tables.Plm, tables.scales,
            RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon, lcap, cfg.mmax, mcap;
            ndrange=(cfg.nlat, 2mcap + 1))
    AMDGPU.synchronize()
    FFTW.ifft!(fourier, 2)
    return fourier
end
function synthesis_packed_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                               coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex}
    return _amdgpu_synthesis_packed_cplx(cfg, coefficients, cfg.lmax)
end
synthesis_packed_cplx(cfg::SHTConfig,
                      coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex} =
    synthesis_packed_cplx(SHTnsKit.GPU(), cfg, coefficients)

function synthesis_packed_cplx_l(::SHTnsKit.GPU, cfg::SHTConfig,
                                 coefficients::AMDGPU.AnyROCArray{T,1},
                                 ltr::Integer) where {T<:Complex}
    return _amdgpu_synthesis_packed_cplx(
        cfg, coefficients, _amdgpu_lcap(cfg, ltr),
    )
end
synthesis_packed_cplx_l(cfg::SHTConfig,
                        coefficients::AMDGPU.AnyROCArray{T,1},
                        ltr::Integer) where {T<:Complex} =
    synthesis_packed_cplx_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _amdgpu_batch_analysis(cfg::SHTConfig, fields::AMDGPU.AnyROCArray;
                                use_rfft::Bool=false, fft_batch=nothing)
    fft_batch === nothing || throw(ArgumentError(
        "AMDGPU scalar batches do not accept caller-provided fft_batch scratch",
    ))
    size(fields, 1) == cfg.nlat && size(fields, 2) == cfg.nlon ||
        throw(DimensionMismatch("fields must start with (nlat, nlon)"))
    size(fields, 3) > 0 || throw(ArgumentError(
        "analysis_batch requires at least one field",
    ))
    use_rfft && !(eltype(fields) <: Real) && throw(ArgumentError(
        "use_rfft=true requires real-valued fields",
    ))
    RT = typeof(float(real(zero(eltype(fields)))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    fourier = CT.(fields)
    FFTW.fft!(fourier, 2)
    canonical = AMDGPU.zeros(CT, cfg.lmax + 1, cfg.mmax + 1, size(fields, 3))
    kernel! = scalar_batch_analysis_kernel!(ROCBackend())
    kernel!(canonical, fourier, tables.Plm, tables.weights, RT(cfg.cphi),
            cfg.lmax, cfg.mmax, cfg.mres; ndrange=size(canonical))
    configured = canonical ./ reshape(
        tables.scales, cfg.lmax + 1, cfg.mmax + 1, 1,
    )
    AMDGPU.synchronize()
    return configured
end

function _amdgpu_batch_analysis_direct!(cfg::SHTConfig,
                                        output::AMDGPU.AnyROCArray{<:Complex,3},
                                        fields::AMDGPU.AnyROCArray{<:Real,3};
                                        use_rfft::Bool=false, fft_batch=nothing)
    _require_amdgpu(:analysis_batch!)
    size(fields, 1) == cfg.nlat && size(fields, 2) == cfg.nlon ||
        throw(DimensionMismatch("fields must start with (nlat, nlon)"))
    nfields = size(fields, 3)
    nfields > 0 || throw(ArgumentError(
        "analysis_batch! requires at least one field",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2",
    ))
    size(output) == (cfg.lmax + 1, cfg.mmax + 1, nfields) ||
        throw(DimensionMismatch("output batch shape mismatch"))
    RT = typeof(float(eltype(fields)))
    CT = Complex{RT}
    scratch = _amdgpu_batch_scratch(
        cfg, fft_batch, CT, nfields, use_rfft, output, fields,
    )
    tables = _amdgpu_scalar_tables(cfg, RT)
    return _with_amdgpu_workspace(cfg, cfg, RT, nfields, use_rfft) do workspace
        fourier = scratch === nothing ? workspace.fourier : scratch
        if use_rfft
            copyto!(workspace.real_buffer, fields)
            mul!(fourier, workspace.forward, workspace.real_buffer)
        else
            copyto!(fourier, fields)
            mul!(fourier, workspace.forward, fourier)
        end
        backend = ROCBackend()
        scalar_batch_analysis_kernel!(backend)(
            output, fourier, tables.Plm, tables.weights, RT(cfg.cphi),
            cfg.lmax, cfg.mmax, cfg.mres; ndrange=size(output),
        )
        coefficient_batch_conversion_kernel!(backend)(
            output, output, tables.scales, cfg.lmax, cfg.mmax, false;
            ndrange=size(output),
        )
        AMDGPU.synchronize()
        output
    end
end

analysis_batch(::SHTnsKit.GPU, cfg::SHTConfig,
               fields::AMDGPU.AnyROCArray{T,3}; use_rfft::Bool=false) where {T<:Real} =
    _amdgpu_batch_analysis(cfg, fields; use_rfft)
analysis_batch(cfg::SHTConfig, fields::AMDGPU.AnyROCArray{T,3}; kwargs...) where {T<:Real} =
    analysis_batch(SHTnsKit.GPU(), cfg, fields; kwargs...)

function analysis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                         output::AMDGPU.AnyROCArray{T,3},
                         fields::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Complex,R<:Real}
    return _amdgpu_batch_analysis_direct!(cfg, output, fields; kwargs...)
end
analysis_batch!(cfg::SHTConfig, output::AMDGPU.AnyROCArray{T,3},
                fields::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Complex,R<:Real} =
    analysis_batch!(SHTnsKit.GPU(), cfg, output, fields; kwargs...)

function _amdgpu_batch_synthesis(cfg::SHTConfig,
                                 coefficients::AMDGPU.AnyROCArray;
                                 real_output::Bool=true, use_rfft::Bool=false,
                                 fft_batch=nothing)
    fft_batch === nothing || throw(ArgumentError(
        "AMDGPU scalar batches do not accept caller-provided fft_batch scratch",
    ))
    size(coefficients, 1) == cfg.lmax + 1 &&
        size(coefficients, 2) == cfg.mmax + 1 ||
        throw(DimensionMismatch("coefficient batch has the wrong spectral shape"))
    size(coefficients, 3) > 0 || throw(ArgumentError(
        "synthesis_batch requires at least one coefficient field",
    ))
    use_rfft && !real_output && throw(ArgumentError("use_rfft=true implies real_output"))
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    canonical = CT.(coefficients) .* reshape(
        tables.scales, cfg.lmax + 1, cfg.mmax + 1, 1,
    )
    fourier = AMDGPU.zeros(CT, cfg.nlat, cfg.nlon, size(coefficients, 3))
    kernel! = scalar_batch_synthesis_kernel!(ROCBackend())
    kernel!(fourier, canonical, tables.Plm, RT(SHTnsKit.phi_inv_scale(cfg)),
            cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres, real_output;
            ndrange=(cfg.nlat, cfg.mmax + 1, size(coefficients, 3)))
    AMDGPU.synchronize()
    FFTW.ifft!(fourier, 2)
    return real_output ? real.(fourier) : fourier
end

function _amdgpu_batch_synthesis_direct!(cfg::SHTConfig,
                                         output::AMDGPU.AnyROCArray{<:Number,3},
                                         coefficients::AMDGPU.AnyROCArray{<:Complex,3};
                                         real_output::Bool=true,
                                         use_rfft::Bool=false, fft_batch=nothing)
    _require_amdgpu(:synthesis_batch!)
    size(coefficients, 1) == cfg.lmax + 1 &&
        size(coefficients, 2) == cfg.mmax + 1 ||
        throw(DimensionMismatch("coefficient batch has the wrong spectral shape"))
    nfields = size(coefficients, 3)
    nfields > 0 || throw(ArgumentError(
        "synthesis_batch! requires at least one coefficient field",
    ))
    size(output) == (cfg.nlat, cfg.nlon, nfields) ||
        throw(DimensionMismatch("output batch shape mismatch"))
    !real_output && eltype(output) <: Real && throw(ArgumentError(
        "real batch output storage requires real_output=true",
    ))
    use_rfft && (!real_output || !(eltype(output) <: Real)) && throw(ArgumentError(
        "use_rfft=true requires real-valued output",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2",
    ))
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    scratch = _amdgpu_batch_scratch(
        cfg, fft_batch, CT, nfields, use_rfft, output, coefficients,
    )
    tables = _amdgpu_scalar_tables(cfg, RT)
    return _with_amdgpu_workspace(cfg, cfg, RT, nfields, use_rfft) do workspace
        fourier = scratch === nothing ? workspace.fourier : scratch
        backend = ROCBackend()
        coefficient_batch_conversion_kernel!(backend)(
            workspace.canonical, coefficients, tables.scales,
            cfg.lmax, cfg.mmax, true; ndrange=size(coefficients),
        )
        fill!(fourier, zero(eltype(fourier)))
        scalar_batch_synthesis_kernel!(backend)(
            fourier, workspace.canonical, tables.Plm,
            RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon,
            cfg.lmax, cfg.mmax, cfg.mres, real_output && !use_rfft;
            ndrange=(cfg.nlat, cfg.mmax + 1, nfields),
        )
        AMDGPU.synchronize()
        if use_rfft
            mul!(workspace.real_buffer, workspace.inverse, fourier)
            copyto!(output, workspace.real_buffer)
        else
            mul!(fourier, workspace.inverse, fourier)
            real_output ? (output .= real.(fourier)) : copyto!(output, fourier)
        end
        AMDGPU.synchronize()
        output
    end
end

synthesis_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                coefficients::AMDGPU.AnyROCArray{T,3};
                real_output::Bool=true, use_rfft::Bool=false) where {T<:Complex} =
    _amdgpu_batch_synthesis(cfg, coefficients; real_output, use_rfft)
synthesis_batch(cfg::SHTConfig,
                coefficients::AMDGPU.AnyROCArray{T,3}; kwargs...) where {T<:Complex} =
    synthesis_batch(SHTnsKit.GPU(), cfg, coefficients; kwargs...)
synthesis_batch_cplx(cfg::SHTConfig,
                     coefficients::AMDGPU.AnyROCArray{T,3}) where {T<:Complex} =
    _amdgpu_batch_synthesis(cfg, coefficients; real_output=false)
synthesis_batch_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                     coefficients::AMDGPU.AnyROCArray{T,3}) where {T<:Complex} =
    _amdgpu_batch_synthesis(cfg, coefficients; real_output=false)

function synthesis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                          output::AMDGPU.AnyROCArray{T,3},
                          coefficients::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T,R<:Complex}
    return _amdgpu_batch_synthesis_direct!(cfg, output, coefficients; kwargs...)
end
function synthesis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                          output::AMDGPU.AnyROCArray{T,3},
                          coefficients::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Real,R<:Complex}
    return _amdgpu_batch_synthesis_direct!(cfg, output, coefficients; kwargs...)
end
synthesis_batch!(cfg::SHTConfig, output::AMDGPU.AnyROCArray{T,3},
                 coefficients::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T,R<:Complex} =
    synthesis_batch!(SHTnsKit.GPU(), cfg, output, coefficients; kwargs...)

function analysis!(::SHTnsKit.GPU, plan::SHTPlan,
                   output::AMDGPU.AnyROCArray{T,2},
                   field::AMDGPU.AnyROCArray{R,2}) where {T<:Complex,R<:Number}
    return _amdgpu_scalar_analysis_direct!(
        plan, plan.cfg, output, field; use_rfft=plan.use_rfft,
    )
end
analysis!(plan::SHTPlan, output::AMDGPU.AnyROCArray{T,2},
          field::AMDGPU.AnyROCArray{R,2}) where {T<:Complex,R<:Number} =
    analysis!(SHTnsKit.GPU(), plan, output, field)

function synthesis!(::SHTnsKit.GPU, plan::SHTPlan,
                    output::AMDGPU.AnyROCArray{T,2},
                    coefficients::AMDGPU.AnyROCArray{R,2};
                    real_output::Bool=true) where {T<:Number,R<:Complex}
    return _amdgpu_scalar_synthesis_direct!(
        plan, plan.cfg, output, coefficients;
        real_output, use_rfft=plan.use_rfft,
    )
end
synthesis!(plan::SHTPlan, output::AMDGPU.AnyROCArray{T,2},
           coefficients::AMDGPU.AnyROCArray{R,2};
           real_output::Bool=true) where {T<:Number,R<:Complex} =
    synthesis!(SHTnsKit.GPU(), plan, output, coefficients; real_output)
synthesis_batch!(cfg::SHTConfig, output::AMDGPU.AnyROCArray{T,3},
                 coefficients::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Real,R<:Complex} =
    synthesis_batch!(SHTnsKit.GPU(), cfg, output, coefficients; kwargs...)

end # module SHTnsKitAMDGPUExt
