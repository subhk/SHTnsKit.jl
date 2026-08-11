module SHTnsKitGPUExt

using SHTnsKit
using KernelAbstractions, GPUArrays, GPUArraysCore
using LinearAlgebra, FFTW

# Import CUDA
using CUDA
using CUDA.CUFFT

include("GPUCommon.jl")
using .GPUCommon: laplacian_kernel!, legendre_table_kernel!,
                  scalar_analysis_kernel!, scalar_synthesis_kernel!,
                  coefficient_conversion_kernel!,
                  coefficient_batch_conversion_kernel!, scalar_config_signature,
                  real_pack_kernel!, real_unpack_kernel!,
                  mode_analysis_kernel!, mode_synthesis_kernel!,
                  scalar_batch_analysis_kernel!, scalar_batch_synthesis_kernel!,
                  complex_packed_analysis_kernel!, complex_packed_synthesis_kernel!,
                  scalar_host_tables, ScalarTableCache, scalar_cache_lookup,
                  scalar_cache_insert!, scalar_cache_clear!, scalar_cache_size
using .GPUCommon: ScalarWorkspaceCache, scalar_workspace_use!,
                  scalar_workspace_clear!, scalar_workspace_size

# Import functions from SHTnsKit to extend them
import SHTnsKit: gpu_analysis, gpu_synthesis, gpu_analysis_safe, gpu_synthesis_safe,
                 gpu_analysis_sphtor, gpu_synthesis_sphtor,
                 gpu_apply_laplacian!,
                 gpu_memory_info, check_gpu_memory, gpu_clear_cache!,
                 estimate_memory_usage, get_available_gpus, set_gpu_device

# Import device routing functions to extend.
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

# ============================================================================
# CUDA Backend Integration with device_utils.jl
# ============================================================================

mutable struct CUDAAdapter end
const CUDA_ADAPTER = CUDAAdapter()

function __init__()
    _register_gpu_adapter!(:cuda, CUDA_ADAPTER)
    return nothing
end

_gpu_adapter_functional(::CUDAAdapter) = CUDA.functional()
_gpu_adapter_matches(::CUDAAdapter, ::CUDA.AnyCuArray) = true

function _require_cuda(operation::Symbol, device)
    CUDA.functional() || throw(SHTnsKit.BackendUnavailableError(
        operation,
        "CUDA.jl is loaded but CUDA.functional() is false; configure a working CUDA runtime (CPU fallback is available only through gpu_analysis_safe/gpu_synthesis_safe)",
    ))
    device isa SHTnsKit.GPU || throw(ArgumentError(
        "`$operation` is a strict GPU operation and does not accept CPU(); use a gpu_*_safe wrapper for explicit fallback",
    ))
    return nothing
end

function _gpu_adapter_adapt(::CUDAAdapter, value)
    CUDA.functional() || throw(SHTnsKit.BackendUnavailableError(
        :to_device,
        "CUDA.jl is loaded but CUDA.functional() is false",
    ))
    return CuArray(value)
end

struct CUDAScalarTables{TX,TW,TP,TS}
    x::TX
    weights::TW
    Plm::TP
    scales::TS
end

const _CUDA_SCALAR_CACHE = ScalarTableCache(8)
const _CUDA_WORKSPACE_CACHE = ScalarWorkspaceCache(8)

function _cuda_scalar_tables(cfg::SHTConfig, ::Type{T}) where {T<:AbstractFloat}
    device = CUDA.deviceid(CUDA.device())
    identity = objectid(cfg)
    signature = scalar_config_signature(cfg)
    cached = scalar_cache_lookup(
        _CUDA_SCALAR_CACHE, device, identity, T, signature,
    )
    cached === nothing || return cached

    x_host, weights_host, scales_host = scalar_host_tables(cfg, T)
    x = CuArray(x_host)
    weights = CuArray(weights_host)
    scales = CuArray(scales_host)
    Plm = CUDA.zeros(T, cfg.nlat, cfg.lmax + 1, cfg.mmax + 1)
    backend = CUDABackend()
    kernel! = legendre_table_kernel!(backend)
    kernel!(Plm, x, cfg.lmax, cfg.mmax;
            ndrange=(cfg.nlat, cfg.mmax + 1))
    CUDA.synchronize()
    built = CUDAScalarTables(x, weights, Plm, scales)

    return scalar_cache_insert!(
        _CUDA_SCALAR_CACHE, device, identity, T, signature, built,
    )
end

function _cuda_clear_scalar_cache!(; device=nothing)
    scalar_cache_clear!(_CUDA_SCALAR_CACHE; device)
    scalar_workspace_clear!(_CUDA_WORKSPACE_CACHE; device)
    return nothing
end

function _gpu_adapter_clear_cache!(::CUDAAdapter)
    _cuda_clear_scalar_cache!()
    return nothing
end

function _cuda_workspace_builder(cfg::SHTConfig, ::Type{RT}, nfields::Int,
                                 use_rfft::Bool) where {RT<:AbstractFloat}
    CT = Complex{RT}
    spatial_shape = nfields == 0 ? (cfg.nlat, cfg.nlon) :
                                   (cfg.nlat, cfg.nlon, nfields)
    spectral_shape = nfields == 0 ? (cfg.lmax + 1, cfg.mmax + 1) :
                                    (cfg.lmax + 1, cfg.mmax + 1, nfields)
    canonical = CUDA.zeros(CT, spectral_shape)
    if use_rfft
        real_buffer = CUDA.zeros(RT, spatial_shape)
        half_shape = Base.setindex(spatial_shape, cfg.nlon ÷ 2 + 1, 2)
        fourier = CUDA.zeros(CT, half_shape)
        forward = CUFFT.plan_rfft(real_buffer, 2)
        inverse = CUFFT.plan_irfft(fourier, cfg.nlon, 2)
        return (; canonical, fourier, real_buffer, forward, inverse)
    end
    fourier = CUDA.zeros(CT, spatial_shape)
    forward = CUFFT.plan_fft!(fourier, 2)
    inverse = CUFFT.plan_ifft!(fourier, 2)
    return (; canonical, fourier, real_buffer=nothing, forward, inverse)
end

function _with_cuda_workspace(f, owner, cfg::SHTConfig, ::Type{RT},
                              nfields::Int, use_rfft::Bool) where {RT<:AbstractFloat}
    device = CUDA.deviceid(CUDA.device())
    kind = nfields == 0 ? :scalar : :batch
    shape = (cfg.nlat, cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres,
             nfields, use_rfft)
    signature = hash((scalar_config_signature(cfg), use_rfft, nfields))
    builder = () -> _cuda_workspace_builder(cfg, RT, nfields, use_rfft)
    return scalar_workspace_use!(
        f, builder, _CUDA_WORKSPACE_CACHE, device, owner, RT,
        kind, shape, signature,
    )
end

function _cuda_batch_scratch(cfg::SHTConfig, fft_batch, ::Type{CT},
                             nfields::Int, use_rfft::Bool, operands...) where {CT<:Complex}
    fft_batch === nothing && return nothing
    fft_batch isa CUDA.AnyCuArray || throw(ArgumentError(
        "fft_batch must use CUDA storage",
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

function _cuda_scalar_analysis_direct!(owner, cfg::SHTConfig,
                                       output::CUDA.AnyCuArray,
                                       field::CUDA.AnyCuArray;
                                       use_rfft::Bool=false)
    _require_cuda(:analysis!, SHTnsKit.GPU())
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
    CT = Complex{RT}
    eltype(output) <: Complex || throw(ArgumentError("analysis output must be complex"))
    tables = _cuda_scalar_tables(cfg, RT)
    return _with_cuda_workspace(owner, cfg, RT, 0, use_rfft) do workspace
        fourier = workspace.fourier
        if use_rfft
            copyto!(workspace.real_buffer, field)
            mul!(fourier, workspace.forward, workspace.real_buffer)
        else
            copyto!(fourier, field)
            mul!(fourier, workspace.forward, fourier)
        end
        backend = CUDABackend()
        scalar_analysis_kernel!(backend)(
            output, fourier, tables.Plm, tables.weights, RT(cfg.cphi),
            cfg.lmax, cfg.mmax, cfg.mres, cfg.lmax;
            ndrange=(cfg.lmax + 1, cfg.mmax + 1),
        )
        coefficient_conversion_kernel!(backend)(
            output, output, tables.scales, cfg.lmax, cfg.mmax, false;
            ndrange=(cfg.lmax + 1, cfg.mmax + 1),
        )
        CUDA.synchronize()
        output
    end
end

function _cuda_scalar_synthesis_direct!(owner, cfg::SHTConfig,
                                        output::CUDA.AnyCuArray,
                                        coefficients::CUDA.AnyCuArray;
                                        real_output::Bool=true,
                                        use_rfft::Bool=false)
    _require_cuda(:synthesis!, SHTnsKit.GPU())
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
    tables = _cuda_scalar_tables(cfg, RT)
    return _with_cuda_workspace(owner, cfg, RT, 0, use_rfft) do workspace
        backend = CUDABackend()
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
        CUDA.synchronize()
        if use_rfft
            mul!(workspace.real_buffer, workspace.inverse, workspace.fourier)
            copyto!(output, workspace.real_buffer)
        else
            mul!(workspace.fourier, workspace.inverse, workspace.fourier)
            real_output ? (output .= real.(workspace.fourier)) :
                          copyto!(output, workspace.fourier)
        end
        CUDA.synchronize()
        output
    end
end

function _cuda_scalar_analysis(cfg::SHTConfig, field::CUDA.AnyCuArray;
                               use_rfft::Bool=false, fft_scratch=nothing,
                               lcap::Int=cfg.lmax)
    0 ≤ lcap ≤ cfg.lmax || throw(ArgumentError(
        "lcap must satisfy 0 ≤ lcap ≤ lmax=$(cfg.lmax)",
    ))
    size(field) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "field must have size ($(cfg.nlat), $(cfg.nlon)), got $(size(field))",
    ))
    fft_scratch === nothing || throw(ArgumentError(
        "CUDA scalar transforms do not accept a host fft_scratch",
    ))
    use_rfft && !(eltype(field) <: Real) && throw(ArgumentError(
        "use_rfft=true requires a real-valued input",
    ))
    RT = typeof(float(real(zero(eltype(field)))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    # `use_rfft` is a performance hint. CUDA currently uses the complex CUFFT
    # pipeline for valid real transforms; the mathematical result is identical.
    fourier = CT.(field)
    gpu_fft!(fourier, 2)

    backend = CUDABackend()
    canonical = CUDA.zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    analyze! = scalar_analysis_kernel!(backend)
    analyze!(canonical, fourier, tables.Plm, tables.weights, RT(cfg.cphi),
             cfg.lmax, cfg.mmax, cfg.mres, lcap;
             ndrange=(lcap + 1, min(cfg.mmax, lcap) + 1))

    configured = similar(canonical)
    convert! = coefficient_conversion_kernel!(backend)
    convert!(configured, canonical, tables.scales, cfg.lmax, cfg.mmax, false;
             ndrange=(cfg.lmax + 1, cfg.mmax + 1))
    CUDA.synchronize()
    return configured
end

function _cuda_scalar_synthesis(cfg::SHTConfig, coefficients::CUDA.AnyCuArray;
                                real_output::Bool=true, use_rfft::Bool=false,
                                fft_scratch=nothing, lcap::Int=cfg.lmax)
    0 ≤ lcap ≤ cfg.lmax || throw(ArgumentError(
        "lcap must satisfy 0 ≤ lcap ≤ lmax=$(cfg.lmax)",
    ))
    size(coefficients) == (cfg.lmax + 1, cfg.mmax + 1) || throw(DimensionMismatch(
        "coefficients must have size ($(cfg.lmax + 1), $(cfg.mmax + 1)), got $(size(coefficients))",
    ))
    fft_scratch === nothing || throw(ArgumentError(
        "CUDA scalar transforms do not accept a host fft_scratch",
    ))
    use_rfft && !real_output && throw(ArgumentError(
        "use_rfft=true implies real_output",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2, got mmax=$(cfg.mmax), nlon=$(cfg.nlon)",
    ))
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    configured = CT.(coefficients)
    canonical = similar(configured)
    backend = CUDABackend()
    convert! = coefficient_conversion_kernel!(backend)
    convert!(canonical, configured, tables.scales, cfg.lmax, cfg.mmax, true;
             ndrange=(cfg.lmax + 1, cfg.mmax + 1))

    # Keep one complex CUFFT implementation for both valid modes until a vendor
    # rFFT plan is measurably preferable; `use_rfft` does not alter semantics.
    fourier = CUDA.zeros(CT, cfg.nlat, cfg.nlon)
    synthesize! = scalar_synthesis_kernel!(backend)
    synthesize!(fourier, canonical, tables.Plm, RT(SHTnsKit.phi_inv_scale(cfg)),
                cfg.nlon, lcap, min(cfg.mmax, lcap), cfg.mres, real_output;
                ndrange=(cfg.nlat, min(cfg.mmax, lcap) + 1))
    CUDA.synchronize()
    gpu_ifft!(fourier, 2)
    return real_output ? real.(fourier) : fourier
end

function _gpu_adapter_analysis(::CUDAAdapter, cfg::SHTConfig, field::CUDA.AnyCuArray; kwargs...)
    _require_cuda(:analysis, SHTnsKit.GPU())
    return _cuda_scalar_analysis(cfg, field; kwargs...)
end

function _gpu_adapter_synthesis(::CUDAAdapter, cfg::SHTConfig, coefficients::CUDA.AnyCuArray; kwargs...)
    _require_cuda(:synthesis, SHTnsKit.GPU())
    return _cuda_scalar_synthesis(cfg, coefficients; kwargs...)
end

analysis(cfg::SHTConfig, field::CUDA.AnyCuArray{T,2}; kwargs...) where {T} =
    analysis(SHTnsKit.GPU(), cfg, field; kwargs...)
synthesis(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,2}; kwargs...) where {T} =
    synthesis(SHTnsKit.GPU(), cfg, coefficients; kwargs...)
synthesis_cplx(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,2}) where {T} =
    synthesis_cplx(SHTnsKit.GPU(), cfg, coefficients)

@inline function _cuda_lcap(cfg::SHTConfig, ltr::Integer)
    return SHTnsKit._validate_degree_limit(cfg, ltr)
end

function _cuda_pack_lm(cfg::SHTConfig, dense::CUDA.AnyCuArray, lcap::Int)
    packed = CUDA.zeros(eltype(dense), cfg.nlm)
    kernel! = real_pack_kernel!(CUDABackend())
    kernel!(packed, dense, cfg.lmax, cfg.mmax, cfg.mres, lcap;
            ndrange=(cfg.lmax + 1, cfg.mmax ÷ cfg.mres + 1))
    CUDA.synchronize()
    return packed
end

function _cuda_unpack_lm(cfg::SHTConfig, packed::CUDA.AnyCuArray, lcap::Int)
    length(packed) == cfg.nlm || throw(DimensionMismatch(
        "Qlm must have length $(cfg.nlm)",
    ))
    dense = CUDA.zeros(eltype(packed), cfg.lmax + 1, cfg.mmax + 1)
    kernel! = real_unpack_kernel!(CUDABackend())
    kernel!(dense, packed, cfg.lmax, cfg.mmax, cfg.mres, lcap;
            ndrange=size(dense))
    CUDA.synchronize()
    return dense
end

function analysis_packed(::SHTnsKit.GPU, cfg::SHTConfig,
                         field::CUDA.AnyCuArray{T,1}) where {T<:Real}
    _require_cuda(:analysis_packed, SHTnsKit.GPU())
    length(field) == cfg.nspat || throw(DimensionMismatch(
        "field must have length $(cfg.nspat)",
    ))
    return _cuda_pack_lm(
        cfg, _cuda_scalar_analysis(cfg, reshape(field, cfg.nlat, cfg.nlon)), cfg.lmax,
    )
end
analysis_packed(cfg::SHTConfig, field::CUDA.AnyCuArray{T,1}) where {T<:Real} =
    analysis_packed(SHTnsKit.GPU(), cfg, field)

function synthesis_packed(::SHTnsKit.GPU, cfg::SHTConfig,
                          coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex}
    _require_cuda(:synthesis_packed, SHTnsKit.GPU())
    return vec(_cuda_scalar_synthesis(
        cfg, _cuda_unpack_lm(cfg, coefficients, cfg.lmax); real_output=true,
    ))
end
synthesis_packed(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex} =
    synthesis_packed(SHTnsKit.GPU(), cfg, coefficients)

function analysis_packed_l(::SHTnsKit.GPU, cfg::SHTConfig,
                           field::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Real}
    lcap = _cuda_lcap(cfg, ltr)
    length(field) == cfg.nspat || throw(DimensionMismatch(
        "field must have length $(cfg.nspat)",
    ))
    return _cuda_pack_lm(
        cfg, _cuda_scalar_analysis(
            cfg, reshape(field, cfg.nlat, cfg.nlon); lcap,
        ), lcap,
    )
end
analysis_packed_l(cfg::SHTConfig, field::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Real} =
    analysis_packed_l(SHTnsKit.GPU(), cfg, field, ltr)

function synthesis_packed_l(::SHTnsKit.GPU, cfg::SHTConfig,
                            coefficients::CUDA.AnyCuArray{T,1},
                            ltr::Integer) where {T<:Complex}
    lcap = _cuda_lcap(cfg, ltr)
    return vec(_cuda_scalar_synthesis(
        cfg, _cuda_unpack_lm(cfg, coefficients, lcap);
        real_output=true, lcap,
    ))
end
synthesis_packed_l(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    synthesis_packed_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _cuda_mode_analysis(cfg::SHTConfig, physical_m::Int,
                             mode::CUDA.AnyCuArray, lcap::Int, scale)
    RT = typeof(float(real(zero(eltype(mode)))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    canonical = CUDA.zeros(CT, lcap - physical_m + 1)
    kernel! = mode_analysis_kernel!(CUDABackend())
    kernel!(canonical, mode, tables.Plm, tables.weights, RT(scale), physical_m, lcap;
            ndrange=length(canonical))
    configured = canonical ./ @view(tables.scales[(physical_m + 1):(lcap + 1), physical_m + 1])
    CUDA.synchronize()
    return configured
end

function _cuda_mode_synthesis(cfg::SHTConfig, physical_m::Int,
                              coefficients::CUDA.AnyCuArray, lcap::Int, scale)
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    canonical = CT.(coefficients) .*
        @view(tables.scales[(physical_m + 1):(lcap + 1), physical_m + 1])
    mode = CUDA.zeros(CT, cfg.nlat)
    kernel! = mode_synthesis_kernel!(CUDABackend())
    kernel!(mode, canonical, tables.Plm, RT(scale), physical_m, lcap;
            ndrange=cfg.nlat)
    CUDA.synchronize()
    return mode
end

function analysis_axisym(::SHTnsKit.GPU, cfg::SHTConfig,
                         field::CUDA.AnyCuArray{T,1}) where {T<:Real}
    length(field) == cfg.nlat || throw(DimensionMismatch(
        "field must have length nlat=$(cfg.nlat)",
    ))
    return _cuda_mode_analysis(cfg, 0, field, cfg.lmax, cfg.cphi * cfg.nlon)
end
analysis_axisym(cfg::SHTConfig, field::CUDA.AnyCuArray{T,1}) where {T<:Real} =
    analysis_axisym(SHTnsKit.GPU(), cfg, field)

function synthesis_axisym(::SHTnsKit.GPU, cfg::SHTConfig,
                          coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex}
    length(coefficients) == cfg.lmax + 1 || throw(DimensionMismatch(
        "coefficients must have length lmax+1=$(cfg.lmax + 1)",
    ))
    return real.(_cuda_mode_synthesis(cfg, 0, coefficients, cfg.lmax, 1))
end
synthesis_axisym(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex} =
    synthesis_axisym(SHTnsKit.GPU(), cfg, coefficients)

function analysis_axisym_l(::SHTnsKit.GPU, cfg::SHTConfig,
                           field::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Real}
    lcap = _cuda_lcap(cfg, ltr)
    length(field) == cfg.nlat || throw(DimensionMismatch(
        "field must have length nlat=$(cfg.nlat)",
    ))
    return _cuda_mode_analysis(cfg, 0, field, lcap, cfg.cphi * cfg.nlon)
end
analysis_axisym_l(cfg::SHTConfig, field::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Real} =
    analysis_axisym_l(SHTnsKit.GPU(), cfg, field, ltr)

function synthesis_axisym_l(::SHTnsKit.GPU, cfg::SHTConfig,
                            coefficients::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Complex}
    lcap = _cuda_lcap(cfg, ltr)
    length(coefficients) >= lcap + 1 || throw(DimensionMismatch(
        "coefficients must contain degrees 0:ltr",
    ))
    return real.(_cuda_mode_synthesis(
        cfg, 0, @view(coefficients[1:(lcap + 1)]), lcap, 1,
    ))
end
synthesis_axisym_l(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    synthesis_axisym_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _cuda_fixed_order(cfg::SHTConfig, im::Int, ltr::Integer)
    im >= 0 || throw(ArgumentError("im must be >= 0"))
    im <= cfg.mmax ÷ cfg.mres || throw(ArgumentError(
        "im must be <= mmax/mres=$(cfg.mmax ÷ cfg.mres)",
    ))
    lcap = _cuda_lcap(cfg, ltr)
    physical_m = im * cfg.mres
    lcap >= physical_m || throw(ArgumentError(
        "ltr must be >= im*mres=$(physical_m)",
    ))
    return physical_m, lcap
end

function analysis_packed_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Int,
                            mode::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Complex}
    physical_m, lcap = _cuda_fixed_order(cfg, im, ltr)
    length(mode) == cfg.nlat || throw(DimensionMismatch(
        "mode must have length nlat=$(cfg.nlat)",
    ))
    return _cuda_mode_analysis(cfg, physical_m, mode, lcap, cfg.cphi)
end
analysis_packed_ml(cfg::SHTConfig, im::Int, mode::CUDA.AnyCuArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    analysis_packed_ml(SHTnsKit.GPU(), cfg, im, mode, ltr)

function synthesis_packed_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Int,
                             coefficients::CUDA.AnyCuArray{T,1},
                             ltr::Integer) where {T<:Complex}
    physical_m, lcap = _cuda_fixed_order(cfg, im, ltr)
    length(coefficients) == lcap - physical_m + 1 || throw(DimensionMismatch(
        "coefficients have the wrong fixed-order length",
    ))
    return _cuda_mode_synthesis(
        cfg, physical_m, coefficients, lcap, SHTnsKit.phi_inv_scale(cfg),
    )
end
synthesis_packed_ml(cfg::SHTConfig, im::Int,
                    coefficients::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Complex} =
    synthesis_packed_ml(SHTnsKit.GPU(), cfg, im, coefficients, ltr)

function _cuda_analysis_packed_cplx(cfg::SHTConfig,
                                    field::CUDA.AnyCuArray{T,2},
                                    lcap::Int) where {T<:Complex}
    cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    size(field) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "field must have size ($(cfg.nlat), $(cfg.nlon))",
    ))
    RT = typeof(float(real(zero(T))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    fourier = CT.(field)
    gpu_fft!(fourier, 2)
    packed = CUDA.zeros(CT, SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1))
    mcap = min(cfg.mmax, lcap)
    kernel! = complex_packed_analysis_kernel!(CUDABackend())
    kernel!(packed, fourier, tables.Plm, tables.weights, tables.scales,
            RT(cfg.cphi), cfg.nlon, lcap, cfg.mmax, mcap;
            ndrange=(lcap + 1, 2mcap + 1))
    CUDA.synchronize()
    return packed
end
function analysis_packed_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                              field::CUDA.AnyCuArray{T,2}) where {T<:Complex}
    return _cuda_analysis_packed_cplx(cfg, field, cfg.lmax)
end
analysis_packed_cplx(cfg::SHTConfig, field::CUDA.AnyCuArray{T,2}) where {T<:Complex} =
    analysis_packed_cplx(SHTnsKit.GPU(), cfg, field)

function analysis_packed_cplx_l(::SHTnsKit.GPU, cfg::SHTConfig,
                                field::CUDA.AnyCuArray{T,2},
                                ltr::Integer) where {T<:Complex}
    return _cuda_analysis_packed_cplx(cfg, field, _cuda_lcap(cfg, ltr))
end
analysis_packed_cplx_l(cfg::SHTConfig, field::CUDA.AnyCuArray{T,2},
                       ltr::Integer) where {T<:Complex} =
    analysis_packed_cplx_l(SHTnsKit.GPU(), cfg, field, ltr)

function _cuda_synthesis_packed_cplx(cfg::SHTConfig,
                                     coefficients::CUDA.AnyCuArray{T,1},
                                     lcap::Int) where {T<:Complex}
    cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    expected = SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    length(coefficients) == expected || throw(DimensionMismatch(
        "coefficients must have length $expected",
    ))
    RT = typeof(float(real(zero(T))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    fourier = CUDA.zeros(CT, cfg.nlat, cfg.nlon)
    mcap = min(cfg.mmax, lcap)
    kernel! = complex_packed_synthesis_kernel!(CUDABackend())
    kernel!(fourier, coefficients, tables.Plm, tables.scales,
            RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon, lcap, cfg.mmax, mcap;
            ndrange=(cfg.nlat, 2mcap + 1))
    CUDA.synchronize()
    gpu_ifft!(fourier, 2)
    return fourier
end
function synthesis_packed_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                               coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex}
    return _cuda_synthesis_packed_cplx(cfg, coefficients, cfg.lmax)
end
synthesis_packed_cplx(cfg::SHTConfig,
                      coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex} =
    synthesis_packed_cplx(SHTnsKit.GPU(), cfg, coefficients)

function synthesis_packed_cplx_l(::SHTnsKit.GPU, cfg::SHTConfig,
                                 coefficients::CUDA.AnyCuArray{T,1},
                                 ltr::Integer) where {T<:Complex}
    return _cuda_synthesis_packed_cplx(
        cfg, coefficients, _cuda_lcap(cfg, ltr),
    )
end
synthesis_packed_cplx_l(cfg::SHTConfig,
                        coefficients::CUDA.AnyCuArray{T,1},
                        ltr::Integer) where {T<:Complex} =
    synthesis_packed_cplx_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _cuda_batch_analysis(cfg::SHTConfig, fields::CUDA.AnyCuArray;
                              use_rfft::Bool=false, fft_batch=nothing)
    fft_batch === nothing || throw(ArgumentError(
        "CUDA scalar batches do not accept caller-provided fft_batch scratch",
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
    tables = _cuda_scalar_tables(cfg, RT)
    fourier = CT.(fields)
    gpu_fft!(fourier, 2)
    canonical = CUDA.zeros(CT, cfg.lmax + 1, cfg.mmax + 1, size(fields, 3))
    kernel! = scalar_batch_analysis_kernel!(CUDABackend())
    kernel!(canonical, fourier, tables.Plm, tables.weights, RT(cfg.cphi),
            cfg.lmax, cfg.mmax, cfg.mres; ndrange=size(canonical))
    configured = canonical ./ reshape(
        tables.scales, cfg.lmax + 1, cfg.mmax + 1, 1,
    )
    CUDA.synchronize()
    return configured
end

function _cuda_batch_analysis_direct!(cfg::SHTConfig,
                                      output::CUDA.AnyCuArray{<:Complex,3},
                                      fields::CUDA.AnyCuArray{<:Real,3};
                                      use_rfft::Bool=false, fft_batch=nothing)
    _require_cuda(:analysis_batch!, SHTnsKit.GPU())
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
    scratch = _cuda_batch_scratch(
        cfg, fft_batch, CT, nfields, use_rfft, output, fields,
    )
    tables = _cuda_scalar_tables(cfg, RT)
    return _with_cuda_workspace(cfg, cfg, RT, nfields, use_rfft) do workspace
        fourier = scratch === nothing ? workspace.fourier : scratch
        if use_rfft
            copyto!(workspace.real_buffer, fields)
            mul!(fourier, workspace.forward, workspace.real_buffer)
        else
            copyto!(fourier, fields)
            mul!(fourier, workspace.forward, fourier)
        end
        backend = CUDABackend()
        scalar_batch_analysis_kernel!(backend)(
            output, fourier, tables.Plm, tables.weights, RT(cfg.cphi),
            cfg.lmax, cfg.mmax, cfg.mres; ndrange=size(output),
        )
        coefficient_batch_conversion_kernel!(backend)(
            output, output, tables.scales, cfg.lmax, cfg.mmax, false;
            ndrange=size(output),
        )
        CUDA.synchronize()
        output
    end
end

function analysis_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                        fields::CUDA.AnyCuArray{T,3}; use_rfft::Bool=false) where {T<:Real}
    return _cuda_batch_analysis(cfg, fields; use_rfft)
end
analysis_batch(cfg::SHTConfig, fields::CUDA.AnyCuArray{T,3}; kwargs...) where {T<:Real} =
    analysis_batch(SHTnsKit.GPU(), cfg, fields; kwargs...)

function analysis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                         output::CUDA.AnyCuArray{T,3},
                         fields::CUDA.AnyCuArray{R,3}; kwargs...) where {T<:Complex,R<:Real}
    return _cuda_batch_analysis_direct!(cfg, output, fields; kwargs...)
end
analysis_batch!(cfg::SHTConfig, output::CUDA.AnyCuArray{T,3},
                fields::CUDA.AnyCuArray{R,3}; kwargs...) where {T<:Complex,R<:Real} =
    analysis_batch!(SHTnsKit.GPU(), cfg, output, fields; kwargs...)

function _cuda_batch_synthesis(cfg::SHTConfig,
                               coefficients::CUDA.AnyCuArray;
                               real_output::Bool=true, use_rfft::Bool=false,
                               fft_batch=nothing)
    fft_batch === nothing || throw(ArgumentError(
        "CUDA scalar batches do not accept caller-provided fft_batch scratch",
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
    tables = _cuda_scalar_tables(cfg, RT)
    canonical = CT.(coefficients) .* reshape(
        tables.scales, cfg.lmax + 1, cfg.mmax + 1, 1,
    )
    fourier = CUDA.zeros(CT, cfg.nlat, cfg.nlon, size(coefficients, 3))
    kernel! = scalar_batch_synthesis_kernel!(CUDABackend())
    kernel!(fourier, canonical, tables.Plm, RT(SHTnsKit.phi_inv_scale(cfg)),
            cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres, real_output;
            ndrange=(cfg.nlat, cfg.mmax + 1, size(coefficients, 3)))
    CUDA.synchronize()
    gpu_ifft!(fourier, 2)
    return real_output ? real.(fourier) : fourier
end

function _cuda_batch_synthesis_direct!(cfg::SHTConfig,
                                       output::CUDA.AnyCuArray{<:Number,3},
                                       coefficients::CUDA.AnyCuArray{<:Complex,3};
                                       real_output::Bool=true,
                                       use_rfft::Bool=false, fft_batch=nothing)
    _require_cuda(:synthesis_batch!, SHTnsKit.GPU())
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
    scratch = _cuda_batch_scratch(
        cfg, fft_batch, CT, nfields, use_rfft, output, coefficients,
    )
    tables = _cuda_scalar_tables(cfg, RT)
    return _with_cuda_workspace(cfg, cfg, RT, nfields, use_rfft) do workspace
        fourier = scratch === nothing ? workspace.fourier : scratch
        backend = CUDABackend()
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
        CUDA.synchronize()
        if use_rfft
            mul!(workspace.real_buffer, workspace.inverse, fourier)
            copyto!(output, workspace.real_buffer)
        else
            mul!(fourier, workspace.inverse, fourier)
            real_output ? (output .= real.(fourier)) : copyto!(output, fourier)
        end
        CUDA.synchronize()
        output
    end
end

function synthesis_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                         coefficients::CUDA.AnyCuArray{T,3};
                         real_output::Bool=true, use_rfft::Bool=false) where {T<:Complex}
    return _cuda_batch_synthesis(cfg, coefficients; real_output, use_rfft)
end
synthesis_batch(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,3}; kwargs...) where {T<:Complex} =
    synthesis_batch(SHTnsKit.GPU(), cfg, coefficients; kwargs...)
synthesis_batch_cplx(cfg::SHTConfig,
                     coefficients::CUDA.AnyCuArray{T,3}) where {T<:Complex} =
    _cuda_batch_synthesis(cfg, coefficients; real_output=false)
synthesis_batch_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                     coefficients::CUDA.AnyCuArray{T,3}) where {T<:Complex} =
    _cuda_batch_synthesis(cfg, coefficients; real_output=false)

function synthesis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                          output::CUDA.AnyCuArray{T,3},
                          coefficients::CUDA.AnyCuArray{R,3}; kwargs...) where {T,R<:Complex}
    return _cuda_batch_synthesis_direct!(cfg, output, coefficients; kwargs...)
end
function synthesis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                          output::CUDA.AnyCuArray{T,3},
                          coefficients::CUDA.AnyCuArray{R,3}; kwargs...) where {T<:Real,R<:Complex}
    return _cuda_batch_synthesis_direct!(cfg, output, coefficients; kwargs...)
end
synthesis_batch!(cfg::SHTConfig, output::CUDA.AnyCuArray{T,3},
                 coefficients::CUDA.AnyCuArray{R,3}; kwargs...) where {T,R<:Complex} =
    synthesis_batch!(SHTnsKit.GPU(), cfg, output, coefficients; kwargs...)

function analysis!(::SHTnsKit.GPU, plan::SHTPlan,
                   output::CUDA.AnyCuArray{T,2},
                   field::CUDA.AnyCuArray{R,2}) where {T<:Complex,R<:Number}
    return _cuda_scalar_analysis_direct!(
        plan, plan.cfg, output, field; use_rfft=plan.use_rfft,
    )
end
analysis!(plan::SHTPlan, output::CUDA.AnyCuArray{T,2},
          field::CUDA.AnyCuArray{R,2}) where {T<:Complex,R<:Number} =
    analysis!(SHTnsKit.GPU(), plan, output, field)

function synthesis!(::SHTnsKit.GPU, plan::SHTPlan,
                    output::CUDA.AnyCuArray{T,2},
                    coefficients::CUDA.AnyCuArray{R,2};
                    real_output::Bool=true) where {T<:Number,R<:Complex}
    return _cuda_scalar_synthesis_direct!(
        plan, plan.cfg, output, coefficients;
        real_output, use_rfft=plan.use_rfft,
    )
end
synthesis!(plan::SHTPlan, output::CUDA.AnyCuArray{T,2},
           coefficients::CUDA.AnyCuArray{R,2};
           real_output::Bool=true) where {T<:Number,R<:Complex} =
    synthesis!(SHTnsKit.GPU(), plan, output, coefficients; real_output)
synthesis_batch!(cfg::SHTConfig, output::CUDA.AnyCuArray{T,3},
                 coefficients::CUDA.AnyCuArray{R,3}; kwargs...) where {T<:Real,R<:Complex} =
    synthesis_batch!(SHTnsKit.GPU(), cfg, output, coefficients; kwargs...)

"""
    _to_gpu_impl(arr::AbstractArray)

Transfer array to CUDA GPU. Overrides the stub in device_utils.jl.
"""
function _to_gpu_impl(arr::AbstractArray)
    if !CUDA.functional()
        error("CUDA is not functional")
    end
    return CuArray(arr)
end

# Avoid double conversion
_to_gpu_impl(arr::CuArray) = arr

"""
    on_device(arr::CuArray) -> ComputeDevice

Returns `GPU()` for CUDA arrays.
"""
on_device(::CUDA.AnyCuArray) = SHTnsKit.GPU()

@inline _is_cpu_device(::SHTnsKit.CPU) = true
@inline _is_cpu_device(::SHTnsKit.GPU) = false

"""
    get_available_gpus()

Returns a list of available CUDA GPU devices with their IDs and names.
"""
function get_available_gpus()
    gpus = []
    if CUDA.functional()
        for i = 0:(CUDA.ndevices()-1)
            push!(gpus, (device=SHTnsKit.GPU(), id=i, name=CUDA.name(CUDA.CuDevice(i))))
        end
    end
    return gpus
end

"""
    set_gpu_device(device_id::Int)

Set the active CUDA GPU device by ID.
"""
function set_gpu_device(device_id::Int)
    if CUDA.functional()
        CUDA.device!(device_id)
        return true
    end
    return false
end

set_gpu_device(::SHTnsKit.GPU, device_id::Int) = set_gpu_device(device_id)

# ============================================================================
# cuFFT-based FFT operations with pre-planned transforms
# ============================================================================

"""
    CuFFTPlan

Pre-planned cuFFT operations for efficient repeated transforms.
"""
struct CuFFTPlan
    forward_plan::CUFFT.CuFFTPlan
    inverse_plan::CUFFT.CuFFTPlan
    buffer::CuArray{ComplexF64, 2}
    nlat::Int
    nlon::Int
end

"""
    create_cufft_plan(nlat::Int, nlon::Int)

Create pre-planned cuFFT operations for a grid of size (nlat, nlon).
Forward and inverse plans are created for transforms along the longitude dimension.
"""
function create_cufft_plan(nlat::Int, nlon::Int)
    # Allocate buffer for FFT operations
    buffer = CUDA.zeros(ComplexF64, nlat, nlon)

    # Create forward FFT plan along dimension 2 (longitude)
    forward_plan = CUFFT.plan_fft!(buffer, 2)

    # Create inverse FFT plan along dimension 2 (longitude)
    inverse_plan = CUFFT.plan_ifft!(buffer, 2)

    return CuFFTPlan(forward_plan, inverse_plan, buffer, nlat, nlon)
end

"""
    gpu_fft!(plan::CuFFTPlan, data::CuArray)

Perform in-place forward FFT using pre-planned cuFFT.
"""
function gpu_fft!(plan::CuFFTPlan, data::CuArray)
    mul!(data, plan.forward_plan, data)
    return data
end

"""
    gpu_ifft!(plan::CuFFTPlan, data::CuArray)

Perform in-place inverse FFT using pre-planned cuFFT.
"""
function gpu_ifft!(plan::CuFFTPlan, data::CuArray)
    mul!(data, plan.inverse_plan, data)
    return data
end

"""
    gpu_fft!(data::CuArray, dims)

Perform FFT on CUDA array along specified dimensions (without pre-planning).
"""
function gpu_fft!(data::CuArray, dims)
    plan = CUFFT.plan_fft!(data, dims)
    plan * data  # In-place plan: mul! is called internally, modifying data
    return data
end

"""
    gpu_ifft!(data::CuArray, dims)

Perform inverse FFT on CUDA array along specified dimensions (without pre-planning).
"""
function gpu_ifft!(data::CuArray, dims)
    plan = CUFFT.plan_ifft!(data, dims)
    plan * data
    return data
end

"""
    gpu_rfft(data::CuArray, dims)

Perform real-to-complex FFT on CUDA array.
"""
function gpu_rfft(data::CuArray{<:Real}, dims)
    return CUFFT.rfft(data, dims)
end

"""
    gpu_irfft(data::CuArray, n, dims)

Perform complex-to-real inverse FFT on CUDA array.
"""
function gpu_irfft(data::CuArray, n, dims)
    return CUFFT.irfft(data, n, dims)
end

# ============================================================================
# GPU-accelerated core operations using KernelAbstractions
# ============================================================================

@kernel function legendre_associated_kernel!(Plm, x, lmax, mmax)
    """
    GPU kernel for computing ORTHONORMAL normalized associated Legendre functions P̄_l^m(x).
    Parallelized over (latitude, m) pairs for maximum GPU utilization.

    Computes P̄_l^m = Nlm·P_l^m directly via a bounded recurrence (|P̄| ≲ 1 at all l,m),
    fixing overflow for lmax ≥ ~151 that afflicted the old unnormalized approach.
    The downstream analysis/synthesis kernels must NOT multiply by Nlm (it is folded in).

    Recurrence (Condon–Shortley phase included, orthonormal convention):
      P̄_0^0 = 1/sqrt(4π)   (INV_SQRT_4PI)
      P̄_m^m = −sqrt((2m+1)/(2m)) · sinθ · P̄_{m−1}^{m−1}   (sectoral step)
      P̄_{m+1}^m = sqrt(2m+3) · x · P̄_m^m
      P̄_l^m = a·x·P̄_{l−1}^m − b·P̄_{l−2}^m   (l ≥ m+2)
        a = sqrt(((2l−1)(2l+1)) / ((l−m)(l+m)))
        b = sqrt(((2l+1)(l−1−m)(l−1+m)) / ((2l−3)(l−m)(l+m)))
    """
    i, m_idx = @index(Global, NTuple)
    nlat = length(x)
    if i <= nlat && m_idx <= mmax + 1
        m = m_idx - 1
        xi = x[i]
        sint = sqrt(max(0.0, 1.0 - xi*xi))  # sin(θ)

        # ── Sectoral term P̄_m^m ────────────────────────────────────────────────
        # Start from P̄_0^0 = 1/sqrt(4π) and apply the sectoral recurrence m times.
        # Each step: P̄_k^k = −sqrt((2k+1)/(2k)) · sinθ · P̄_{k−1}^{k−1}
        const_inv_sqrt4pi = 0.28209479177387814  # 1/sqrt(4π)
        pmm = const_inv_sqrt4pi
        @inbounds for k = 1:m
            pmm = -sqrt((2.0*k + 1.0) / (2.0*k)) * sint * pmm
        end
        Plm[i, m+1, m_idx] = pmm

        # ── P̄_{m+1}^m ──────────────────────────────────────────────────────────
        if m < lmax
            pm1m = sqrt(2.0*m + 3.0) * xi * pmm
            Plm[i, m+2, m_idx] = pm1m

            # ── Three-term recurrence for l ≥ m+2 ──────────────────────────────
            plm_prev2 = pmm    # P̄_{l−2}^m
            plm_prev1 = pm1m   # P̄_{l−1}^m
            @inbounds for l = m+2:lmax
                fl  = Float64(l)
                fm  = Float64(m)
                a = sqrt(((2.0*fl - 1.0) * (2.0*fl + 1.0)) /
                         ((fl - fm) * (fl + fm)))
                b = sqrt(((2.0*fl + 1.0) * (fl - 1.0 - fm) * (fl - 1.0 + fm)) /
                         ((2.0*fl - 3.0) * (fl - fm) * (fl + fm)))
                plm = a * xi * plm_prev1 - b * plm_prev2
                Plm[i, l+1, m_idx] = plm
                plm_prev2 = plm_prev1
                plm_prev1 = plm
            end
        end
    end
end

@kernel function legendre_and_derivative_kernel!(Plm, dPlm, x, lmax, mmax)
    """
    GPU kernel for computing ORTHONORMAL normalized P̄_l^m(x) AND their
    derivatives dP̄_l^m/dx. Required for vector spherical harmonic transforms.

    Computes P̄_l^m via the same bounded normalized sectoral recurrence as
    legendre_associated_kernel! (no overflow at high lmax).

    For dP̄_l^m/dx the standard formula adapted to the normalized functions is:
      (x²−1) · dP̄_l^m/dx = l·x·P̄_l^m − sqrt((l²−m²)(2l+1)/(2l−1)) · P̄_{l−1}^m
    At poles (x²−1 ≈ 0) the derivative is set to zero (pole handling is done by
    the pole closed-form in the downstream synthesis kernel via Nlm separately).

    The downstream kernels must NOT multiply by Nlm (it is folded into P̄ and dP̄).
    """
    i, m_idx = @index(Global, NTuple)
    nlat = length(x)
    if i <= nlat && m_idx <= mmax + 1
        m = m_idx - 1
        xi = x[i]
        sint = sqrt(max(0.0, 1.0 - xi*xi))
        x2m1 = xi*xi - 1.0
        # Guard against x = ±1 (poles)
        inv_x2m1 = abs(x2m1) < 1e-14 ? 0.0 : 1.0 / x2m1

        # ── Sectoral P̄_m^m (same as legendre_associated_kernel!) ────────────
        const_inv_sqrt4pi = 0.28209479177387814
        pmm = const_inv_sqrt4pi
        @inbounds for k = 1:m
            pmm = -sqrt((2.0*k + 1.0) / (2.0*k)) * sint * pmm
        end
        Plm[i, m+1, m_idx] = pmm

        # dP̄_m^m/dx = m·x·P̄_m^m / (x²−1)   [0 for m=0]
        if m == 0
            dPlm[i, 1, 1] = 0.0
        else
            dPlm[i, m+1, m_idx] = Float64(m) * xi * pmm * inv_x2m1
        end

        if m < lmax
            # ── P̄_{m+1}^m ───────────────────────────────────────────────────
            pm1m = sqrt(2.0*m + 3.0) * xi * pmm
            Plm[i, m+2, m_idx] = pm1m

            # dP̄_{m+1}^m/dx: l=m+1 case.
            # Numerator: (m+1)·x·P̄_{m+1}^m − sqrt(((m+1)²−m²)(2(m+1)+1)/(2(m+1)−1)) · P̄_m^m
            # = (m+1)·x·P̄_{m+1}^m − sqrt((2m+1)(2m+3)/(2m+1)) · P̄_m^m
            # = (m+1)·x·P̄_{m+1}^m − sqrt(2m+3) · P̄_m^m
            dPlm[i, m+2, m_idx] = (Float64(m+1) * xi * pm1m -
                                    sqrt(2.0*m + 3.0) * pmm) * inv_x2m1

            # ── Three-term recurrence for l ≥ m+2 ──────────────────────────
            plm_prev2 = pmm    # P̄_{l−2}^m = P̄_m^m
            plm_prev1 = pm1m   # P̄_{l−1}^m = P̄_{m+1}^m
            @inbounds for l = m+2:lmax
                fl = Float64(l)
                fm = Float64(m)
                a = sqrt(((2.0*fl - 1.0) * (2.0*fl + 1.0)) /
                         ((fl - fm) * (fl + fm)))
                b = sqrt(((2.0*fl + 1.0) * (fl - 1.0 - fm) * (fl - 1.0 + fm)) /
                         ((2.0*fl - 3.0) * (fl - fm) * (fl + fm)))
                plm = a * xi * plm_prev1 - b * plm_prev2
                Plm[i, l+1, m_idx] = plm
                # dP̄_l^m/dx = [l·x·P̄_l^m − sqrt((l²−m²)(2l+1)/(2l−1))·P̄_{l−1}^m] / (x²−1)
                scale_lm = sqrt((fl*fl - fm*fm) * (2.0*fl + 1.0) / (2.0*fl - 1.0))
                dPlm[i, l+1, m_idx] = (fl * xi * plm - scale_lm * plm_prev1) * inv_x2m1
                plm_prev2 = plm_prev1
                plm_prev1 = plm
            end
        end
    end
end

"""
    gpu_analysis(cfg::SHTConfig, spatial_data; device=GPU())

GPU-accelerated spherical harmonic analysis transform using cuFFT.

Implements: a_lm = ∫∫ f(θ,φ) Y_l^m*(θ,φ) sin(θ) dθ dφ
1. FFT along φ (dimension 2) to extract Fourier modes
2. Gauss-Legendre integration along θ (dimension 1) with P_l^m weights

Fully parallelized: all (l,m) coefficients computed in a single kernel launch.

"""
function gpu_analysis(cfg::SHTConfig, spatial_data; device=SHTnsKit.GPU(), kwargs...)
    _require_cuda(:gpu_analysis, device)
    device_data = spatial_data isa CUDA.AnyCuArray ? spatial_data : CuArray(spatial_data)
    return Array(_cuda_scalar_analysis(cfg, device_data; kwargs...))
end

"""
    gpu_synthesis(cfg::SHTConfig, coeffs; device=GPU(), real_output=true)

GPU-accelerated spherical harmonic synthesis transform using cuFFT.

Implements: f(θ,φ) = Σ_l Σ_m a_lm Y_l^m(θ,φ)
1. Legendre summation along θ: F_m(θ) = Σ_l a_lm * P_l^m(cos θ) * N_lm
2. Inverse FFT along φ (dimension 2) to reconstruct spatial field

Fully parallelized: all (θ,m) Fourier modes computed in a single kernel launch.
"""
function gpu_synthesis(cfg::SHTConfig, coeffs; device=SHTnsKit.GPU(),
                       real_output=true, kwargs...)
    _require_cuda(:gpu_synthesis, device)
    device_coefficients = coeffs isa CUDA.AnyCuArray ? coeffs : CuArray(coeffs)
    return Array(_cuda_scalar_synthesis(
        cfg, device_coefficients; real_output, kwargs...,
    ))
end

# ============================================================================
# Vector field GPU operations using proper spectral method
# ============================================================================

# Pole limits for the sphtor kernels — device twins of `_dPdtheta_at_pole` /
# `_P_over_sinth_at_pole` in src/kernels.jl.
#
# At an exact pole node (sinθ == 0, i.e. a pole-inclusive regular or
# Driscoll-Healy grid) the tables give dP̄/dθ = -sinθ·dP̄/dx = 0 and P̄/sinθ = 0
# because `inv_sθ` is guarded to 0. Both are wrong: the true limits are finite
# and non-zero for m = 1, so without this branch the GPU silently dropped the
# entire m = 1 contribution from the two pole rows and disagreed with the CPU
# for the same cfg. `(-1)^k` is written as an `isodd`/`iseven` select to stay
# allocation- and intrinsic-free inside a kernel.
const _GPU_POLE_TOL = SHTnsKit.POLE_TOLERANCE_FACTOR * eps(Float64)

@inline function _gpu_dPdtheta_at_pole(l::Int, m::Int, x::Float64, N::Float64)
    m == 1 || return 0.0
    ll1 = 0.5 * Float64(l * (l + 1))
    s = x > 0 ? -1.0 : (isodd(l) ? 1.0 : -1.0)   # north: -1;  south: (-1)^(l+1)
    return s * N * ll1
end

@inline function _gpu_P_over_sinth_at_pole(l::Int, m::Int, x::Float64, N::Float64)
    m == 1 || return 0.0
    ll1 = 0.5 * Float64(l * (l + 1))
    s = x > 0 ? -1.0 : (iseven(l) ? 1.0 : -1.0)  # north: -1;  south: (-1)^l
    return s * N * ll1
end

"""
    gpu_analysis_sphtor(cfg::SHTConfig, vθ, vφ; device=GPU())

GPU-accelerated spheroidal-toroidal decomposition of vector fields using proper spectral method.

Uses the adjoint of the synthesis formula with Gauss-Legendre quadrature:
    S_lm = Σ_i w_i * scaleφ / (l(l+1)) * (F_θ * ∂Y_l^m/∂θ + conj(im·m/sinθ·Y_l^m) * F_φ)
    T_lm = Σ_i w_i * scaleφ / (l(l+1)) * (-conj(im·m/sinθ·Y_l^m) * F_θ + ∂Y_l^m/∂θ * F_φ)

Where F_θ, F_φ are Fourier modes of Vθ, Vφ and w_i are quadrature weights.
All computation stays on GPU for maximum performance.
"""
function gpu_analysis_sphtor(cfg::SHTConfig, vθ, vφ; device=SHTnsKit.GPU())
    _require_cuda(:gpu_analysis_sphtor, device)

    backend = CUDABackend()
    nlat, nlon = cfg.nlat, cfg.nlon
    lmax, mmax = cfg.lmax, cfg.mmax

    # Transfer input to GPU and compute FFT along φ
    gpu_vθ = CuArray(ComplexF64.(vθ))
    gpu_vφ = CuArray(ComplexF64.(vφ))
    gpu_fft!(gpu_vθ, 2)
    gpu_fft!(gpu_vφ, 2)

    # Transfer config data to GPU. `Nlm` is only read on the pole branch below,
    # where the P̄/dP̄ tables collapse to 0 and the closed-form limits need N.
    x_values = CuArray(cfg.x)
    weights = CuArray(cfg.w)
    Nlm_values = CuArray(cfg.Nlm)
    scaleφ = cfg.cphi
    robert_form = cfg.robert_form

    # Compute ORTHONORMAL normalized P̄_l^m AND their dP̄/dx on GPU.
    # Both arrays already include Nlm — downstream kernels must NOT multiply by Nlm.
    Plm = CUDA.zeros(Float64, nlat, lmax+1, mmax+1)
    dPlm = CUDA.zeros(Float64, nlat, lmax+1, mmax+1)
    legendre_deriv_kernel! = legendre_and_derivative_kernel!(backend)
    legendre_deriv_kernel!(Plm, dPlm, x_values, lmax, mmax; ndrange=(nlat, mmax+1))
    CUDA.synchronize()

    # Phase 1: Compute per-latitude weighted contributions for each (l, m)
    # This produces intermediate arrays of shape (nlat, lmax+1, mmax+1)
    # Each thread handles one (latitude, l, m) triplet
    S_contrib = CUDA.zeros(ComplexF64, nlat, lmax+1, mmax+1)
    T_contrib = CUDA.zeros(ComplexF64, nlat, lmax+1, mmax+1)

    @kernel function vector_analysis_contrib_kernel!(S_out, T_out, Fθ, Fφ, Plm, dPlm,
                                                      x_vals, w_vals, Nlm_vals, scale,
                                                      nlat, lmax, mmax, do_robert)
        i_lat, l_idx, m_idx = @index(Global, NTuple)

        if i_lat <= nlat && l_idx <= lmax + 1 && m_idx <= mmax + 1
            l = l_idx - 1
            m = m_idx - 1

            # Only compute for valid (l, m) pairs where l >= max(1, m)
            if l >= max(1, m)
                x = x_vals[i_lat]
                sθ = sqrt(max(0.0, 1.0 - x * x))
                is_pole = sθ < _GPU_POLE_TOL
                inv_sθ = is_pole ? 0.0 : 1.0 / sθ
                wi = w_vals[i_lat]

                # Get Fourier modes for this latitude and m
                Fθ_val = Fθ[i_lat, m_idx]
                Fφ_val = Fφ[i_lat, m_idx]

                # Robert-form handling: input is sin(θ)*V, divide by sin(θ)
                if do_robert && !is_pole
                    Fθ_val /= sθ
                    Fφ_val /= sθ
                end

                # Get ORTHONORMAL normalized Legendre values (Nlm already folded in)
                P = Plm[i_lat, l_idx, m_idx]   # P̄_l^m
                dP = dPlm[i_lat, l_idx, m_idx]  # dP̄_l^m/dx

                # ∂Ȳ_l^m/∂θ = -sinθ * dP̄_l^m/dx, and Ȳ/sinθ = P̄ * inv_sθ — both
                # collapse to 0 at an exact pole node, so substitute the closed-form
                # limits there (mirrors `_sphtor_analysis_kernel!` on the CPU).
                dθY = -sθ * dP
                Y_over_s = P * inv_sθ
                if is_pole
                    N = Nlm_vals[l_idx, m_idx]
                    dθY = _gpu_dPdtheta_at_pole(l, m, x, N)
                    Y_over_s = _gpu_P_over_sinth_at_pole(l, m, x, N)
                end

                # Compute coefficient and term
                coeff = wi * scale / (l * (l + 1))
                term_re = 0.0
                term_im = m * Y_over_s

                # Adjoint of synthesis formulas:
                # S_lm += coeff * (Fθ * dθY + conj(term) * Fφ)
                # T_lm += coeff * (-conj(term) * Fθ + dθY * Fφ)

                # conj(term) = (term_re, -term_im) = (0, -m*Ȳ/sinθ)
                Fθ_re = real(Fθ_val)
                Fθ_im = imag(Fθ_val)
                Fφ_re = real(Fφ_val)
                Fφ_im = imag(Fφ_val)

                # Fθ * dθY (dθY is real)
                s1_re = Fθ_re * dθY
                s1_im = Fθ_im * dθY

                # conj(term) * Fφ = (0 - (-term_im)*Fφ_im, 0*Fφ_im + (-term_im)*Fφ_re)
                #                 = (term_im * Fφ_im, -term_im * Fφ_re)
                s2_re = term_im * Fφ_im
                s2_im = -term_im * Fφ_re

                S_out[i_lat, l_idx, m_idx] = coeff * ComplexF64(s1_re + s2_re, s1_im + s2_im)

                # -conj(term) * Fθ = -(term_im * Fθ_im, -term_im * Fθ_re)
                #                  = (-term_im * Fθ_im, term_im * Fθ_re)
                t1_re = -term_im * Fθ_im
                t1_im = term_im * Fθ_re

                # dθY * Fφ (dθY is real)
                t2_re = dθY * Fφ_re
                t2_im = dθY * Fφ_im

                T_out[i_lat, l_idx, m_idx] = coeff * ComplexF64(t1_re + t2_re, t1_im + t2_im)
            end
        end
    end

    contrib_kernel! = vector_analysis_contrib_kernel!(backend)
    contrib_kernel!(S_contrib, T_contrib, gpu_vθ, gpu_vφ, Plm, dPlm,
                    x_values, weights, Nlm_values, scaleφ,
                    nlat, lmax, mmax, robert_form;
                    ndrange=(nlat, lmax+1, mmax+1))
    CUDA.synchronize()

    # Phase 2: Sum over latitude dimension to get final coefficients
    # Use CUDA reduction: sum along dimension 1
    Slm_gpu = dropdims(sum(S_contrib, dims=1), dims=1)
    Tlm_gpu = dropdims(sum(T_contrib, dims=1), dims=1)

    # Transfer results back to CPU
    Slm = Array(Slm_gpu)
    Tlm = Array(Tlm_gpu)

    # Orthonormal-only, like the CPU sphtor pair.
    return Slm, Tlm
end

"""
    gpu_synthesis_sphtor(cfg::SHTConfig, sph_coeffs, tor_coeffs; device=GPU(), real_output=true)

GPU-accelerated synthesis of spheroidal-toroidal vector field components using proper spectral method.

Uses the spectral formula:
    V_θ = ∂S/∂θ - (1/sinθ) ∂T/∂φ = Σ_{l,m} [∂Y_l^m/∂θ * S_lm - im/sinθ * Y_l^m * T_lm]
    V_φ = (1/sinθ) ∂S/∂φ + ∂T/∂θ = Σ_{l,m} [im/sinθ * Y_l^m * S_lm + ∂Y_l^m/∂θ * T_lm]

Where ∂Y_l^m/∂θ = -sinθ * N_lm * dP_l^m/dx (x = cosθ)
"""
function gpu_synthesis_sphtor(cfg::SHTConfig, sph_coeffs, tor_coeffs; device=SHTnsKit.GPU(), real_output=true)
    _require_cuda(:gpu_synthesis_sphtor, device)

    backend = CUDABackend()
    nlat, nlon = cfg.nlat, cfg.nlon
    lmax, mmax = cfg.lmax, cfg.mmax

    # Orthonormal-only, like the CPU sphtor pair.
    Slm_int, Tlm_int = sph_coeffs, tor_coeffs

    # Transfer coefficients to GPU. Nlm is folded into P̄/dP̄ for the interior;
    # it is still needed for the pole-limit closed forms, where those tables are 0.
    gpu_Slm = CuArray(ComplexF64.(Slm_int))
    gpu_Tlm = CuArray(ComplexF64.(Tlm_int))
    x_values = CuArray(cfg.x)
    Nlm_values = CuArray(cfg.Nlm)

    # Compute ORTHONORMAL normalized P̄_l^m AND dP̄/dx on GPU.
    # Both arrays include Nlm — downstream kernel must NOT multiply by Nlm.
    Plm = CUDA.zeros(Float64, nlat, lmax+1, mmax+1)
    dPlm = CUDA.zeros(Float64, nlat, lmax+1, mmax+1)
    legendre_deriv_kernel! = legendre_and_derivative_kernel!(backend)
    legendre_deriv_kernel!(Plm, dPlm, x_values, lmax, mmax; ndrange=(nlat, mmax+1))
    CUDA.synchronize()

    # Fourier coefficients for vector components
    Fθ = CUDA.zeros(ComplexF64, nlat, nlon)
    Fφ = CUDA.zeros(ComplexF64, nlat, nlon)

    # sin(θ) values for each latitude
    sintheta = CuArray(cfg.st)

    # Scale factor for inverse FFT - use phi_inv_scale to match CPU (not 1/cphi!)
    inv_scaleφ = SHTnsKit.phi_inv_scale(cfg)

    # Kernel for spectral vector synthesis - compute Fourier modes for each (latitude, m).
    # Plm holds P̄_l^m and dPlm holds dP̄_l^m/dx (both orthonormal-normalized, no Nlm factor).
    @kernel function vector_spectral_synthesis_kernel!(Ftheta, Fphi, Slm, Tlm, Plm, dPlm, sintheta, x_vals, Nlm_vals, nlat, nlon, lmax, mmax, inv_scale)
        i, m_idx = @index(Global, NTuple)
        if i <= nlat && m_idx <= mmax + 1
            m = m_idx - 1
            sθ = sintheta[i]
            x = x_vals[i]
            is_pole = abs(sθ) < _GPU_POLE_TOL
            inv_sθ = is_pole ? 0.0 : 1.0 / sθ

            gθ = ComplexF64(0, 0)
            gφ = ComplexF64(0, 0)

            @inbounds for l = m:lmax
                l_idx = l + 1
                # P̄_l^m and dP̄_l^m/dx already include Nlm — no separate N factor.
                Y    = Plm[i, l_idx, m_idx]   # P̄_l^m
                dP   = dPlm[i, l_idx, m_idx]  # dP̄_l^m/dx

                # ∂Ȳ_l^m/∂θ = -sinθ * dP̄_l^m/dx and Ȳ/sinθ = P̄ * inv_sθ. At an
                # exact pole node both are 0 (inv_sθ is guarded), which drops the
                # whole m=1 term; substitute the closed-form limits as the CPU
                # `_sphtor_synthesis_kernel` does.
                dYdθ     = -sθ * dP
                Y_over_s = Y * inv_sθ
                if is_pole
                    N = Nlm_vals[l_idx, m_idx]
                    dYdθ     = _gpu_dPdtheta_at_pole(l, m, x, N)
                    Y_over_s = _gpu_P_over_sinth_at_pole(l, m, x, N)
                end

                Sl = Slm[l_idx, m_idx]
                Tl = Tlm[l_idx, m_idx]

                # V_θ = ∂S/∂θ - (im/sinθ) * T
                gθ += dYdθ * Sl - ComplexF64(0, m) * Y_over_s * Tl
                # V_φ = (im/sinθ) * S + ∂T/∂θ
                gφ += ComplexF64(0, m) * Y_over_s * Sl + dYdθ * Tl
            end

            # Store in Fourier coefficient array
            Ftheta[i, m_idx] = inv_scale * gθ
            Fphi[i, m_idx] = inv_scale * gφ
        end
    end

    synth_kernel! = vector_spectral_synthesis_kernel!(backend)
    synth_kernel!(Fθ, Fφ, gpu_Slm, gpu_Tlm, Plm, dPlm, sintheta, x_values, Nlm_values, nlat, nlon, lmax, mmax, inv_scaleφ; ndrange=(nlat, mmax+1))
    CUDA.synchronize()

    # Apply Hermitian symmetry for real output
    if real_output
        @kernel function hermitian_symmetry_kernel!(F, nlat, nlon, mmax)
            i, m_idx = @index(Global, NTuple)
            if i <= nlat && m_idx <= mmax + 1
                m = m_idx - 1
                if m > 0 && m <= nlon ÷ 2
                    conj_idx = nlon - m + 1
                    if conj_idx >= 1 && conj_idx <= nlon
                        F[i, conj_idx] = conj(F[i, m_idx])
                    end
                end
            end
        end
        herm_kernel! = hermitian_symmetry_kernel!(backend)
        herm_kernel!(Fθ, nlat, nlon, mmax; ndrange=(nlat, mmax+1))
        herm_kernel!(Fφ, nlat, nlon, mmax; ndrange=(nlat, mmax+1))
        CUDA.synchronize()
    end

    # Inverse FFT along φ
    gpu_ifft!(Fθ, 2)
    gpu_ifft!(Fφ, 2)

    result_vθ = Array(Fθ)
    result_vφ = Array(Fφ)

    # Apply Robert-form scaling if configured (multiply by sin(θ) after IFFT)
    if cfg.robert_form
        for i in 1:nlat
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
            result_vθ[i, :] .*= sθ
            result_vφ[i, :] .*= sθ
        end
    end

    if real_output
        return real(result_vθ), real(result_vφ)
    else
        return result_vθ, result_vφ
    end
end

# ============================================================================
# Laplacian operator
# ============================================================================

"""
    gpu_apply_laplacian!(cfg::SHTConfig, coeffs; device=GPU())

GPU-accelerated Laplacian operator in spectral space.
"""
function gpu_apply_laplacian!(cfg::SHTConfig, coeffs; device=SHTnsKit.GPU())
    _require_cuda(:gpu_apply_laplacian!, device)

    # Validate input dimensions
    lmax, mmax = cfg.lmax, cfg.mmax
    size(coeffs, 1) == lmax + 1 || throw(DimensionMismatch("coeffs must have $(lmax+1) rows (lmax+1), got $(size(coeffs, 1))"))
    size(coeffs, 2) == mmax + 1 || throw(DimensionMismatch("coeffs must have $(mmax+1) columns (mmax+1), got $(size(coeffs, 2))"))

    gpu_coeffs = CuArray(coeffs)
    # Zero-initialize output to handle l < m entries (which should be zero)
    output = CUDA.zeros(eltype(gpu_coeffs), size(gpu_coeffs))

    backend = CUDABackend()
    kernel! = laplacian_kernel!(backend)
    kernel!(output, gpu_coeffs, lmax, mmax; ndrange=(lmax+1, mmax+1))
    CUDA.synchronize()

    coeffs .= Array(output)
    return coeffs
end

# ============================================================================
# Memory utilities
# ============================================================================

"""
    gpu_memory_info()

Get memory information for the active CUDA device.
Returns a named tuple with `free` and `total` fields (in bytes).
"""
function gpu_memory_info()
    _require_cuda(:gpu_memory_info, SHTnsKit.GPU())
    mem = CUDA.MemoryInfo()
    return (free=mem.free_bytes, total=mem.total_bytes)
end

"""
    check_gpu_memory(required_bytes::Int)

Check if sufficient GPU memory is available.
"""
function check_gpu_memory(required_bytes::Int)
    CUDA.functional() || return false
    mem_info = gpu_memory_info()
    if mem_info.free < required_bytes
        @warn "Insufficient memory: need $(required_bytes÷(1024^3)) GB, have $(mem_info.free÷(1024^3)) GB available"
        return false
    end
    return true
end

"""
    gpu_clear_cache!()

Clear the active CUDA device's memory cache.
"""
function gpu_clear_cache!()
    _require_cuda(:gpu_clear_cache!, SHTnsKit.GPU())
    # Preserve the historical API's process-wide scalar-cache clear while
    # `CUDA.reclaim()` releases the active device's allocator cache.
    _cuda_clear_scalar_cache!()
    try
        CUDA.reclaim()
        @info "CUDA memory cache cleared"
    catch e
        @warn "Failed to clear CUDA cache: $e"
    end
    return nothing
end

"""
    estimate_memory_usage(cfg::SHTConfig, operation::Symbol)

Estimate memory usage for GPU operations.
"""
function estimate_memory_usage(cfg::SHTConfig, operation::Symbol)
    spatial_size = cfg.nlat * cfg.nlon * 16  # ComplexF64 = 16 bytes
    coeff_size = (cfg.lmax + 1) * (cfg.mmax + 1) * 16
    legendre_size = cfg.nlat * (cfg.lmax + 1) * (cfg.mmax + 1) * 8

    if operation == :analysis
        return spatial_size + coeff_size + legendre_size + spatial_size
    elseif operation == :synthesis
        return coeff_size + spatial_size + legendre_size + spatial_size
    elseif operation == :vector
        return 2 * spatial_size + 2 * coeff_size + legendre_size + 2 * spatial_size
    else
        return spatial_size + coeff_size
    end
end

function _cpu_analysis_fallback(cfg::SHTConfig, spatial_data)
    cpu_data = SHTnsKit.to_device(SHTnsKit.CPU(), spatial_data)
    return SHTnsKit.analysis(SHTnsKit.CPU(), cfg, cpu_data)
end

function _cpu_synthesis_fallback(cfg::SHTConfig, coefficients; real_output::Bool)
    cpu_coefficients = SHTnsKit.to_device(SHTnsKit.CPU(), coefficients)
    return SHTnsKit.synthesis(
        SHTnsKit.CPU(),
        cfg,
        cpu_coefficients;
        real_output=real_output,
    )
end

function _with_cuda_oom_fallback(gpu_call, cpu_call)
    try
        return gpu_call()
    catch err
        err isa CUDA.OutOfGPUMemoryError || rethrow()
        @warn "GPU out of memory, falling back to CPU" exception=(err, catch_backtrace())
        return cpu_call()
    end
end

"""
    gpu_analysis_safe(cfg::SHTConfig, spatial_data; device=get_device())

Memory-safe GPU analysis with automatic fallback to CPU.
"""
function gpu_analysis_safe(cfg::SHTConfig, spatial_data; device=get_device())
    if _is_cpu_device(device) || !CUDA.functional()
        return _cpu_analysis_fallback(cfg, spatial_data)
    end

    required_memory = estimate_memory_usage(cfg, :analysis)
    if !check_gpu_memory(required_memory)
        @info "Falling back to CPU due to memory constraints"
        return _cpu_analysis_fallback(cfg, spatial_data)
    end

    return _with_cuda_oom_fallback(
        () -> gpu_analysis(cfg, spatial_data; device=device),
        () -> _cpu_analysis_fallback(cfg, spatial_data),
    )
end

"""
    gpu_synthesis_safe(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)

Memory-safe GPU synthesis with automatic fallback to CPU.
"""
function gpu_synthesis_safe(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)
    if _is_cpu_device(device) || !CUDA.functional()
        return _cpu_synthesis_fallback(cfg, coeffs; real_output=real_output)
    end

    required_memory = estimate_memory_usage(cfg, :synthesis)
    if !check_gpu_memory(required_memory)
        @info "Falling back to CPU due to memory constraints"
        return _cpu_synthesis_fallback(cfg, coeffs; real_output=real_output)
    end

    return _with_cuda_oom_fallback(
        () -> gpu_synthesis(cfg, coeffs; device=device, real_output=real_output),
        () -> _cpu_synthesis_fallback(cfg, coeffs; real_output=real_output),
    )
end

# Types defined by this extension.
export CuFFTPlan, create_cufft_plan, gpu_fft!, gpu_ifft!

end # module SHTnsKitGPUExt
