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
                  real_pack_kernel!, real_unpack_kernel!,
                  mode_analysis_kernel!, mode_synthesis_kernel!,
                  scalar_batch_analysis_kernel!, scalar_batch_synthesis_kernel!,
                  complex_packed_analysis_kernel!, complex_packed_synthesis_kernel!,
                  scalar_config_signature, scalar_host_tables,
                  ScalarTableCache, scalar_cache_lookup, scalar_cache_insert!,
                  scalar_cache_clear!, scalar_cache_size

import SHTnsKit: analysis, synthesis, synthesis_cplx, on_device,
                 analysis_packed, synthesis_packed,
                 analysis_packed_l, synthesis_packed_l,
                 analysis_axisym, synthesis_axisym,
                 analysis_axisym_l, synthesis_axisym_l,
                 analysis_packed_ml, synthesis_packed_ml,
                 analysis_packed_cplx, synthesis_packed_cplx,
                 analysis_batch, analysis_batch!,
                 synthesis_batch, synthesis_batch!, synthesis_batch_cplx,
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
    return nothing
end

function _gpu_adapter_clear_cache!(::AMDGPUAdapter)
    _amdgpu_clear_scalar_cache!()
    return nothing
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
    lcap = Int(ltr)
    0 ≤ lcap ≤ cfg.lmax || throw(ArgumentError(
        "ltr must satisfy 0 ≤ ltr ≤ lmax=$(cfg.lmax)",
    ))
    return lcap
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
                           field::AMDGPU.AnyROCArray{T,1}, ltr::Int) where {T<:Real}
    lcap = _amdgpu_lcap(cfg, ltr)
    length(field) == cfg.nlat || throw(DimensionMismatch(
        "field must have length nlat=$(cfg.nlat)",
    ))
    return _amdgpu_mode_analysis(cfg, 0, field, lcap, cfg.cphi * cfg.nlon)
end
analysis_axisym_l(cfg::SHTConfig, field::AMDGPU.AnyROCArray{T,1},
                  ltr::Int) where {T<:Real} =
    analysis_axisym_l(SHTnsKit.GPU(), cfg, field, ltr)

function synthesis_axisym_l(::SHTnsKit.GPU, cfg::SHTConfig,
                            coefficients::AMDGPU.AnyROCArray{T,1},
                            ltr::Int) where {T<:Complex}
    lcap = _amdgpu_lcap(cfg, ltr)
    length(coefficients) >= lcap + 1 || throw(DimensionMismatch(
        "coefficients must contain degrees 0:ltr",
    ))
    return real.(_amdgpu_mode_synthesis(
        cfg, 0, @view(coefficients[1:(lcap + 1)]), lcap, 1,
    ))
end
synthesis_axisym_l(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{T,1},
                   ltr::Int) where {T<:Complex} =
    synthesis_axisym_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _amdgpu_fixed_order(cfg::SHTConfig, im::Int, ltr::Int)
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
                            mode::AMDGPU.AnyROCArray{T,1}, ltr::Int) where {T<:Complex}
    physical_m, lcap = _amdgpu_fixed_order(cfg, im, ltr)
    length(mode) == cfg.nlat || throw(DimensionMismatch(
        "mode must have length nlat=$(cfg.nlat)",
    ))
    return _amdgpu_mode_analysis(cfg, physical_m, mode, lcap, cfg.cphi)
end
analysis_packed_ml(cfg::SHTConfig, im::Int, mode::AMDGPU.AnyROCArray{T,1},
                   ltr::Int) where {T<:Complex} =
    analysis_packed_ml(SHTnsKit.GPU(), cfg, im, mode, ltr)

function synthesis_packed_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Int,
                             coefficients::AMDGPU.AnyROCArray{T,1},
                             ltr::Int) where {T<:Complex}
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
                    ltr::Int) where {T<:Complex} =
    synthesis_packed_ml(SHTnsKit.GPU(), cfg, im, coefficients, ltr)

function analysis_packed_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                              field::AMDGPU.AnyROCArray{T,2}) where {T<:Complex}
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
    kernel! = complex_packed_analysis_kernel!(ROCBackend())
    kernel!(packed, fourier, tables.Plm, tables.weights, tables.scales,
            RT(cfg.cphi), cfg.nlon, cfg.lmax, cfg.mmax;
            ndrange=(cfg.lmax + 1, 2cfg.mmax + 1))
    AMDGPU.synchronize()
    return packed
end
analysis_packed_cplx(cfg::SHTConfig,
                     field::AMDGPU.AnyROCArray{T,2}) where {T<:Complex} =
    analysis_packed_cplx(SHTnsKit.GPU(), cfg, field)

function synthesis_packed_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                               coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex}
    cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    expected = SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    length(coefficients) == expected || throw(DimensionMismatch(
        "coefficients must have length $expected",
    ))
    RT = typeof(float(real(zero(T))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    fourier = AMDGPU.zeros(CT, cfg.nlat, cfg.nlon)
    kernel! = complex_packed_synthesis_kernel!(ROCBackend())
    kernel!(fourier, coefficients, tables.Plm, tables.scales,
            RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon, cfg.lmax, cfg.mmax;
            ndrange=(cfg.nlat, 2cfg.mmax + 1))
    AMDGPU.synchronize()
    FFTW.ifft!(fourier, 2)
    return fourier
end
synthesis_packed_cplx(cfg::SHTConfig,
                      coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex} =
    synthesis_packed_cplx(SHTnsKit.GPU(), cfg, coefficients)

function _amdgpu_batch_analysis(cfg::SHTConfig, fields::AMDGPU.AnyROCArray;
                                use_rfft::Bool=false, fft_batch=nothing)
    fft_batch === nothing || throw(ArgumentError(
        "AMDGPU scalar batches do not accept caller-provided fft_batch scratch",
    ))
    size(fields, 1) == cfg.nlat && size(fields, 2) == cfg.nlon ||
        throw(DimensionMismatch("fields must start with (nlat, nlon)"))
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

analysis_batch(::SHTnsKit.GPU, cfg::SHTConfig,
               fields::AMDGPU.AnyROCArray{T,3}; use_rfft::Bool=false) where {T<:Real} =
    _amdgpu_batch_analysis(cfg, fields; use_rfft)
analysis_batch(cfg::SHTConfig, fields::AMDGPU.AnyROCArray{T,3}; kwargs...) where {T<:Real} =
    analysis_batch(SHTnsKit.GPU(), cfg, fields; kwargs...)

function analysis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                         output::AMDGPU.AnyROCArray{T,3},
                         fields::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Complex,R<:Real}
    size(output) == (cfg.lmax + 1, cfg.mmax + 1, size(fields, 3)) ||
        throw(DimensionMismatch("output batch shape mismatch"))
    result = _amdgpu_batch_analysis(cfg, fields; kwargs...)
    copyto!(output, result)
    return output
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
    size(output) == (cfg.nlat, cfg.nlon, size(coefficients, 3)) ||
        throw(DimensionMismatch("output batch shape mismatch"))
    result = _amdgpu_batch_synthesis(cfg, coefficients; kwargs...)
    copyto!(output, result)
    return output
end
function synthesis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                          output::AMDGPU.AnyROCArray{T,3},
                          coefficients::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Real,R<:Complex}
    size(output) == (cfg.nlat, cfg.nlon, size(coefficients, 3)) ||
        throw(DimensionMismatch("output batch shape mismatch"))
    result = _amdgpu_batch_synthesis(cfg, coefficients; kwargs...)
    copyto!(output, result)
    return output
end
synthesis_batch!(cfg::SHTConfig, output::AMDGPU.AnyROCArray{T,3},
                 coefficients::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T,R<:Complex} =
    synthesis_batch!(SHTnsKit.GPU(), cfg, output, coefficients; kwargs...)
synthesis_batch!(cfg::SHTConfig, output::AMDGPU.AnyROCArray{T,3},
                 coefficients::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Real,R<:Complex} =
    synthesis_batch!(SHTnsKit.GPU(), cfg, output, coefficients; kwargs...)

end # module SHTnsKitAMDGPUExt
