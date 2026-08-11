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
                  scalar_config_signature, scalar_host_tables

import SHTnsKit: analysis, synthesis, on_device,
                 _register_gpu_adapter!, _gpu_adapter_functional,
                 _gpu_adapter_matches, _gpu_adapter_adapt,
                 _gpu_adapter_analysis, _gpu_adapter_synthesis

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

const _AMDGPU_SCALAR_CACHE = Dict{Tuple{Int,UInt,DataType},Any}()
const _AMDGPU_SCALAR_CACHE_LOCK = ReentrantLock()

function _amdgpu_scalar_tables(cfg::SHTConfig, ::Type{T}) where {T<:AbstractFloat}
    key = (AMDGPU.device_id(), scalar_config_signature(cfg), T)
    cached = lock(_AMDGPU_SCALAR_CACHE_LOCK) do
        get(_AMDGPU_SCALAR_CACHE, key, nothing)
    end
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

    return lock(_AMDGPU_SCALAR_CACHE_LOCK) do
        get!(_AMDGPU_SCALAR_CACHE, key, built)
    end
end

function _amdgpu_scalar_analysis(cfg::SHTConfig, field::AMDGPU.AnyROCArray;
                                 use_rfft::Bool=false, fft_scratch=nothing)
    _require_amdgpu(:analysis)
    size(field) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "field must have size ($(cfg.nlat), $(cfg.nlon)), got $(size(field))",
    ))
    fft_scratch === nothing || throw(ArgumentError(
        "AMDGPU scalar transforms do not accept a host fft_scratch",
    ))
    RT = typeof(float(real(zero(eltype(field)))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    fourier = CT.(field)
    FFTW.fft!(fourier, 2)

    backend = ROCBackend()
    canonical = AMDGPU.zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    analyze! = scalar_analysis_kernel!(backend)
    analyze!(canonical, fourier, tables.Plm, tables.weights, RT(cfg.cphi),
             cfg.lmax, cfg.mmax, cfg.mres;
             ndrange=(cfg.lmax + 1, cfg.mmax + 1))

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
                                  fft_scratch=nothing)
    _require_amdgpu(:synthesis)
    size(coefficients) == (cfg.lmax + 1, cfg.mmax + 1) || throw(DimensionMismatch(
        "coefficients must have size ($(cfg.lmax + 1), $(cfg.mmax + 1)), got $(size(coefficients))",
    ))
    fft_scratch === nothing || throw(ArgumentError(
        "AMDGPU scalar transforms do not accept a host fft_scratch",
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

    fourier = AMDGPU.zeros(CT, cfg.nlat, cfg.nlon)
    synthesize! = scalar_synthesis_kernel!(backend)
    synthesize!(fourier, canonical, tables.Plm, RT(SHTnsKit.phi_inv_scale(cfg)),
                cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres, real_output;
                ndrange=(cfg.nlat, cfg.mmax + 1))
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

end # module SHTnsKitAMDGPUExt
