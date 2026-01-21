module SHTnsKitGPUExt

using SHTnsKit
using KernelAbstractions, GPUArrays, GPUArraysCore
using LinearAlgebra, FFTW

# Import CUDA
using CUDA
using CUDA.CUFFT

# Import functions from SHTnsKit to extend them
import SHTnsKit: get_device, set_device!,
                 gpu_analysis, gpu_synthesis, gpu_analysis_safe, gpu_synthesis_safe,
                 gpu_spat_to_SHsphtor, gpu_SHsphtor_to_spat,
                 gpu_apply_laplacian!,
                 gpu_memory_info, check_gpu_memory, gpu_clear_cache!,
                 estimate_memory_usage, get_available_gpus, set_gpu_device,
                 create_multi_gpu_config, multi_gpu_analysis, multi_gpu_synthesis,
                 multi_gpu_analysis_streaming, multi_gpu_synthesis_streaming, estimate_streaming_chunks

# Import device utilities functions to override
import SHTnsKit: _to_gpu, on_device, _get_device_details, _ensure_cuda_initialized,
                 _notify_cuda_loaded!

# ============================================================================
# CUDA Backend Integration with device_utils.jl
# ============================================================================

# Notify the main module that CUDA is available
function __init__()
    if CUDA.functional()
        _notify_cuda_loaded!()
    end
end

"""
    _to_gpu(arr::AbstractArray)

Transfer array to CUDA GPU. Overrides the stub in device_utils.jl.
"""
function _to_gpu(arr::AbstractArray)
    if !CUDA.functional()
        error("CUDA is not functional")
    end
    return CuArray(arr)
end

# Avoid double conversion
_to_gpu(arr::CuArray) = arr

"""
    on_device(arr::CuArray) -> Symbol

Returns :gpu for CUDA arrays.
"""
on_device(::CuArray) = :gpu

"""
    _get_device_details(::Val{:gpu})

Get detailed GPU device information.
"""
function _get_device_details(::Val{:gpu})
    if !CUDA.functional()
        return (device_type = :gpu, available = false)
    end

    dev = CUDA.device()
    return (
        device_type = :gpu,
        gpu_backend = :cuda,
        available = true,
        device_id = Int(dev),
        device_name = CUDA.name(dev),
        compute_capability = CUDA.capability(dev),
        total_memory = CUDA.totalmem(dev),
        free_memory = CUDA.available_memory(),
        num_devices = CUDA.ndevices()
    )
end

"""
    _ensure_cuda_initialized()

Ensure CUDA is properly initialized.
"""
function _ensure_cuda_initialized()
    if !CUDA.functional()
        error("CUDA is not available")
    end
    # Trigger lazy initialization
    CUDA.device()
    return nothing
end

# ============================================================================
# Legacy Device Management (for backward compatibility)
# ============================================================================
"""
    SHTDevice

Enum representing supported compute devices for SHTnsKit operations.
"""
@enum SHTDevice begin
    CPU_DEVICE
    CUDA_DEVICE
end

"""
    get_device()

Returns CUDA_DEVICE if CUDA is functional, otherwise CPU_DEVICE.
"""
function get_device()
    if CUDA.functional()
        return CUDA_DEVICE
    else
        return CPU_DEVICE
    end
end

"""
    get_available_gpus()

Returns a list of available CUDA GPU devices with their IDs and names.
"""
function get_available_gpus()
    gpus = []
    if CUDA.functional()
        for i = 0:(CUDA.ndevices()-1)
            push!(gpus, (device=:cuda, id=i, name=CUDA.name(CUDA.CuDevice(i))))
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

# Legacy overload for compatibility
function set_gpu_device(device_type::Symbol, device_id::Int)
    if device_type == :cuda
        return set_gpu_device(device_id)
    end
    return false
end

"""
    MultiGPUConfig

Configuration for multi-GPU spherical harmonic transforms.
"""
struct MultiGPUConfig
    base_config::Any  # SHTConfig
    gpu_devices::Vector{NamedTuple}
    distribution_strategy::Symbol  # :latitude, :longitude, :spectral
    primary_gpu::Int
end

"""
    create_multi_gpu_config(lmax, nlat; nlon=nothing, strategy=:latitude, gpu_ids=nothing)

Create a multi-GPU configuration for spherical harmonic transforms.
"""
function create_multi_gpu_config(lmax::Int, nlat::Int;
                                 nlon::Union{Int,Nothing}=nothing,
                                 strategy::Symbol=:latitude,
                                 gpu_ids::Union{Vector{Int},Nothing}=nothing)

    available_gpus = get_available_gpus()
    if isempty(available_gpus)
        error("No CUDA GPUs available for multi-GPU configuration")
    end

    # Select GPUs to use
    selected_gpus = if gpu_ids === nothing
        available_gpus  # Use all available
    else
        [gpu for gpu in available_gpus if gpu.id in gpu_ids]
    end

    if isempty(selected_gpus)
        error("No valid GPUs found with specified IDs")
    end

    # Create base configuration
    base_cfg = SHTnsKit.create_gauss_config(lmax, nlat; nlon=nlon)

    # Validate distribution strategy
    if strategy ∉ [:latitude, :longitude, :spectral]
        error("Invalid distribution strategy: $strategy. Must be :latitude, :longitude, or :spectral")
    end

    # Enable P2P for CUDA GPUs if multiple devices
    if length(selected_gpus) >= 2
        try
            enable_gpu_p2p_access(selected_gpus)
        catch e
            @warn "Failed to enable P2P access: $e"
        end
    end

    return MultiGPUConfig(base_cfg, selected_gpus, strategy, selected_gpus[1].id)
end

"""
    enable_gpu_p2p_access(gpus::Vector)

Enable peer-to-peer access between all CUDA GPUs if possible.
"""
function enable_gpu_p2p_access(gpus::Vector)
    if !CUDA.functional() || length(gpus) < 2
        return false
    end

    enabled_pairs = 0
    total_pairs = 0

    for i in 1:length(gpus), j in 1:length(gpus)
        if i != j
            total_pairs += 1
            try
                src_device = CUDA.CuDevice(gpus[i].id)
                dest_device = CUDA.CuDevice(gpus[j].id)

                if CUDA.can_access_peer(src_device, dest_device)
                    CUDA.device!(gpus[i].id)
                    CUDA.enable_peer_access(dest_device)
                    enabled_pairs += 1
                end
            catch
                # P2P not available for this pair
            end
        end
    end

    if enabled_pairs > 0
        @info "Enabled P2P access for $enabled_pairs/$total_pairs GPU pairs"
        return true
    else
        @info "No P2P access available between GPUs"
        return false
    end
end

"""
    set_device!(device::SHTDevice)

Set the active compute device for SHTnsKit operations.
"""
function set_device!(device::SHTDevice)
    if device == CUDA_DEVICE
        if !CUDA.functional()
            error("CUDA not available. Install and load CUDA.jl first.")
        end
        CUDA.device!(CUDA.device())
    end
    # CPU_DEVICE doesn't require setup
    nothing
end

"""
    to_device(array, device::SHTDevice)

Transfer array to the specified device.
"""
function to_device(array::AbstractArray, device::SHTDevice)
    if device == CUDA_DEVICE
        if !CUDA.functional()
            error("CUDA not available")
        end
        return CuArray(array)
    else
        return Array(array)  # Transfer to CPU
    end
end

# ============================================================================
# cuFFT-based FFT operations with pre-planned transforms
# ============================================================================

"""
    CuFFTPlan

Pre-planned cuFFT operations for efficient repeated transforms.
"""
struct CuFFTPlan
    forward_plan::CUFFT.cCuFFTPlan
    inverse_plan::CUFFT.cCuFFTPlan
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
    plan.forward_plan * data
    return data
end

"""
    gpu_ifft!(plan::CuFFTPlan, data::CuArray)

Perform in-place inverse FFT using pre-planned cuFFT.
"""
function gpu_ifft!(plan::CuFFTPlan, data::CuArray)
    plan.inverse_plan * data
    return data
end

"""
    gpu_fft!(data::CuArray, dims)

Perform FFT on CUDA array along specified dimensions (without pre-planning).
"""
function gpu_fft!(data::CuArray, dims)
    plan = CUFFT.plan_fft!(data, dims)
    plan * data
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
    GPU kernel for computing associated Legendre polynomials P_l^m(x).
    Parallelized over (latitude, m) pairs for maximum GPU utilization.
    Uses stable three-term recurrence relations.
    Computes UNNORMALIZED Plm - normalization is applied during integration.
    """
    i, m_idx = @index(Global, NTuple)
    nlat = length(x)
    if i <= nlat && m_idx <= mmax + 1
        m = m_idx - 1
        xi = x[i]
        sint = sqrt(max(0.0, 1 - xi*xi))  # sin(θ) = sqrt(1 - cos²(θ))

        # Compute P_m^m (diagonal term) using recurrence from P_0^0
        # P_0^0 = 1, P_m^m = -(2m-1) * sin(θ) * P_{m-1}^{m-1}
        pmm = 1.0
        @inbounds for k = 1:m
            pmm *= -(2*k - 1) * sint
        end
        Plm[i, m+1, m_idx] = pmm

        # Compute P_{m+1}^m if m < lmax
        if m < lmax
            pm1m = (2*m + 1) * xi * pmm
            Plm[i, m+2, m_idx] = pm1m

            # Compute remaining P_l^m for l > m+1 using recurrence:
            # P_l^m = ((2l-1) * x * P_{l-1}^m - (l+m-1) * P_{l-2}^m) / (l-m)
            plm_prev2 = pmm      # P_{l-2}^m
            plm_prev1 = pm1m     # P_{l-1}^m
            @inbounds for l = m+2:lmax
                plm = ((2*l - 1) * xi * plm_prev1 - (l + m - 1) * plm_prev2) / (l - m)
                Plm[i, l+1, m_idx] = plm
                plm_prev2 = plm_prev1
                plm_prev1 = plm
            end
        end
    end
end

@kernel function legendre_and_derivative_kernel!(Plm, dPlm, x, lmax, mmax)
    """
    GPU kernel for computing associated Legendre polynomials P_l^m(x) AND their
    derivatives dP_l^m/dx. Required for vector spherical harmonic transforms.

    The derivative satisfies: dP_l^m/dx = [l*x*P_l^m - (l+m)*P_{l-1}^m] / (x²-1)
    """
    i, m_idx = @index(Global, NTuple)
    nlat = length(x)
    if i <= nlat && m_idx <= mmax + 1
        m = m_idx - 1
        xi = x[i]
        sint = sqrt(max(0.0, 1 - xi*xi))
        x2m1 = xi*xi - 1.0
        # Guard against x = ±1 (poles)
        inv_x2m1 = abs(x2m1) < 1e-14 ? 0.0 : 1.0 / x2m1

        # Compute P_m^m (diagonal term)
        pmm = 1.0
        @inbounds for k = 1:m
            pmm *= -(2*k - 1) * sint
        end
        Plm[i, m+1, m_idx] = pmm

        # dP_m^m/dx: use recurrence dP_m^m/dx = m*x*P_m^m / (x²-1) for m > 0
        # For m=0: dP_0^0/dx = 0
        if m == 0
            dPlm[i, 1, 1] = 0.0
        else
            dPlm[i, m+1, m_idx] = m * xi * pmm * inv_x2m1
        end

        # Compute P_{m+1}^m and its derivative if m < lmax
        if m < lmax
            pm1m = (2*m + 1) * xi * pmm
            Plm[i, m+2, m_idx] = pm1m
            # dP_{m+1}^m/dx = [(m+1)*x*P_{m+1}^m - (m+1+m)*P_m^m] / (x²-1)
            dPlm[i, m+2, m_idx] = ((m+1) * xi * pm1m - (2*m+1) * pmm) * inv_x2m1

            # Compute remaining P_l^m and dP_l^m/dx for l > m+1
            plm_prev2 = pmm
            plm_prev1 = pm1m
            @inbounds for l = m+2:lmax
                plm = ((2*l - 1) * xi * plm_prev1 - (l + m - 1) * plm_prev2) / (l - m)
                Plm[i, l+1, m_idx] = plm
                # dP_l^m/dx = [l*x*P_l^m - (l+m)*P_{l-1}^m] / (x²-1)
                dPlm[i, l+1, m_idx] = (l * xi * plm - (l + m) * plm_prev1) * inv_x2m1
                plm_prev2 = plm_prev1
                plm_prev1 = plm
            end
        end
    end
end

"""
    gpu_analysis(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true)

GPU-accelerated spherical harmonic analysis transform using cuFFT.

Implements: a_lm = ∫∫ f(θ,φ) Y_l^m*(θ,φ) sin(θ) dθ dφ
1. FFT along φ (dimension 2) to extract Fourier modes
2. Gauss-Legendre integration along θ (dimension 1) with P_l^m weights

Fully parallelized: all (l,m) coefficients computed in a single kernel launch.

Note: `real_output` parameter is kept for API compatibility but spherical harmonic
coefficients are always complex. The parameter has no effect on the output type.
"""
function gpu_analysis(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.analysis(cfg, spatial_data)
    end

    # Validate input dimensions
    nlat, nlon = cfg.nlat, cfg.nlon
    size(spatial_data, 1) == nlat || throw(DimensionMismatch("spatial_data must have $nlat rows (nlat), got $(size(spatial_data, 1))"))
    size(spatial_data, 2) == nlon || throw(DimensionMismatch("spatial_data must have $nlon columns (nlon), got $(size(spatial_data, 2))"))

    # Transfer input data to GPU
    gpu_data = CuArray(ComplexF64.(spatial_data))

    # Allocate GPU arrays
    coeffs = CUDA.zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Plm = CUDA.zeros(Float64, cfg.nlat, cfg.lmax+1, cfg.mmax+1)
    normalization = CuArray(cfg.Nlm)
    weights = CuArray(cfg.w)
    x_values = CuArray(cfg.x)  # cos(θ) values at Gauss points

    # Step 1: Precompute UNNORMALIZED Legendre polynomials on GPU
    # Parallelized over (latitude, m) pairs: nlat * (mmax+1) threads
    backend = CUDABackend()
    legendre_kernel! = legendre_associated_kernel!(backend)
    legendre_kernel!(Plm, x_values, cfg.lmax, cfg.mmax; ndrange=(cfg.nlat, cfg.mmax+1))
    CUDA.synchronize()

    # Step 2: FFT along φ direction (dimension 2) using cuFFT
    # After FFT: gpu_data[:, m+1] contains the m-th Fourier mode for m = 0, 1, ..., nlon-1
    gpu_fft!(gpu_data, 2)

    # Scaling factor for φ integration (matches CPU: cfg.cphi = 2π/nlon)
    scaleφ = cfg.cphi

    # Step 3: Fully parallel Legendre integration - ALL (l,m) pairs in one kernel
    # Each thread computes one a_lm coefficient
    @kernel function analysis_kernel!(coeffs, Fφ, Plm, weights, norm, nlat, nlon, lmax, mmax, scale)
        l_idx, m_idx = @index(Global, NTuple)
        if l_idx <= lmax + 1 && m_idx <= mmax + 1
            l = l_idx - 1
            m = m_idx - 1
            # Only compute for l >= m (triangular structure)
            if l >= m && m <= nlon ÷ 2
                result = ComplexF64(0, 0)
                @inbounds for i_lat = 1:nlat
                    # Gauss-Legendre quadrature: weight * Plm * Fourier_mode
                    # Fourier mode m is in column m+1
                    result += weights[i_lat] * Plm[i_lat, l_idx, m_idx] * Fφ[i_lat, m_idx]
                end
                # Apply normalization and φ scaling
                coeffs[l_idx, m_idx] = result * norm[l_idx, m_idx] * scale
            end
        end
    end

    analysis_k! = analysis_kernel!(backend)
    analysis_k!(coeffs, gpu_data, Plm, weights, normalization,
                cfg.nlat, cfg.nlon, cfg.lmax, cfg.mmax, scaleφ;
                ndrange=(cfg.lmax+1, cfg.mmax+1))
    CUDA.synchronize()

    # Transfer result back to CPU - coefficients are always complex
    return Array(coeffs)
end

"""
    gpu_synthesis(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)

GPU-accelerated spherical harmonic synthesis transform using cuFFT.

Implements: f(θ,φ) = Σ_l Σ_m a_lm Y_l^m(θ,φ)
1. Legendre summation along θ: F_m(θ) = Σ_l a_lm * P_l^m(cos θ) * N_lm
2. Inverse FFT along φ (dimension 2) to reconstruct spatial field

Fully parallelized: all (θ,m) Fourier modes computed in a single kernel launch.
"""
function gpu_synthesis(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.synthesis(cfg, coeffs; real_output=real_output)
    end

    # Validate input dimensions
    lmax, mmax = cfg.lmax, cfg.mmax
    size(coeffs, 1) == lmax + 1 || throw(DimensionMismatch("coeffs must have $(lmax+1) rows (lmax+1), got $(size(coeffs, 1))"))
    size(coeffs, 2) == mmax + 1 || throw(DimensionMismatch("coeffs must have $(mmax+1) columns (mmax+1), got $(size(coeffs, 2))"))

    # Transfer coefficients to GPU
    gpu_coeffs = CuArray(ComplexF64.(coeffs))

    # Allocate GPU arrays
    Plm = CUDA.zeros(Float64, cfg.nlat, cfg.lmax+1, cfg.mmax+1)
    normalization = CuArray(cfg.Nlm)
    x_values = CuArray(cfg.x)  # cos(θ) values at Gauss points

    backend = CUDABackend()

    # Step 1: Precompute UNNORMALIZED Legendre polynomials on GPU
    # Parallelized over (latitude, m) pairs: nlat * (mmax+1) threads
    legendre_kernel! = legendre_associated_kernel!(backend)
    legendre_kernel!(Plm, x_values, cfg.lmax, cfg.mmax; ndrange=(cfg.nlat, cfg.mmax+1))
    CUDA.synchronize()

    # Step 2: Fully parallel Legendre summation - ALL (θ, m) pairs in one kernel
    # Each thread computes F_m(θ_i) for one latitude and one m-mode
    fourier_modes = CUDA.zeros(ComplexF64, cfg.nlat, cfg.nlon)

    @kernel function synthesis_kernel!(Fφ, coeffs, Plm, norm, nlat, nlon, lmax, mmax, do_hermitian)
        i_lat, m_idx = @index(Global, NTuple)
        if i_lat <= nlat && m_idx <= mmax + 1
            m = m_idx - 1
            # Compute F_m(θ_i) = Σ_l a_lm * P_l^m(cos θ_i) * N_lm
            result = ComplexF64(0, 0)
            @inbounds for l = m:lmax
                l_idx = l + 1
                result += coeffs[l_idx, m_idx] * Plm[i_lat, l_idx, m_idx] * norm[l_idx, m_idx]
            end

            # Place in Fourier mode slots for IFFT
            # FFT convention: [0, 1, 2, ..., N/2, -N/2+1, ..., -1]
            if m == 0
                Fφ[i_lat, 1] = result
            elseif m <= nlon ÷ 2
                Fφ[i_lat, m + 1] = result
                # Hermitian symmetry for real output: F_{-m} = conj(F_m)
                if do_hermitian && m > 0
                    neg_m_idx = nlon - m + 1
                    if neg_m_idx >= 1 && neg_m_idx <= nlon
                        Fφ[i_lat, neg_m_idx] = conj(result)
                    end
                end
            end
        end
    end

    synthesis_k! = synthesis_kernel!(backend)
    synthesis_k!(fourier_modes, gpu_coeffs, Plm, normalization,
                 cfg.nlat, cfg.nlon, cfg.lmax, cfg.mmax, real_output;
                 ndrange=(cfg.nlat, cfg.mmax+1))
    CUDA.synchronize()

    # Step 3: Inverse FFT along φ direction (dimension 2) using cuFFT
    gpu_ifft!(fourier_modes, 2)

    # Apply inverse φ scaling (matches CPU: phi_inv_scale(cfg))
    # For Gauss grids: nlon; for regular grids: nlon/(2π)
    inv_scaleφ = SHTnsKit.phi_inv_scale(cfg)
    fourier_modes .*= inv_scaleφ

    # Transfer result back to CPU
    result = Array(fourier_modes)

    if real_output
        return real(result)
    else
        return result
    end
end

# ============================================================================
# Vector field GPU operations using proper spectral method
# ============================================================================

"""
    gpu_spat_to_SHsphtor(cfg::SHTConfig, vθ, vφ; device=get_device())

GPU-accelerated spheroidal-toroidal decomposition of vector fields using proper spectral method.

Uses the adjoint of the synthesis formula with Gauss-Legendre quadrature:
    S_lm = Σ_i w_i * scaleφ / (l(l+1)) * (F_θ * ∂Y_l^m/∂θ + conj(im·m/sinθ·Y_l^m) * F_φ)
    T_lm = Σ_i w_i * scaleφ / (l(l+1)) * (-conj(im·m/sinθ·Y_l^m) * F_θ + ∂Y_l^m/∂θ * F_φ)

Where F_θ, F_φ are Fourier modes of Vθ, Vφ and w_i are quadrature weights.
All computation stays on GPU for maximum performance.
"""
function gpu_spat_to_SHsphtor(cfg::SHTConfig, vθ, vφ; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.spat_to_SHsphtor(cfg, vθ, vφ)
    end

    backend = CUDABackend()
    nlat, nlon = cfg.nlat, cfg.nlon
    lmax, mmax = cfg.lmax, cfg.mmax

    # Transfer input to GPU and compute FFT along φ
    gpu_vθ = CuArray(ComplexF64.(vθ))
    gpu_vφ = CuArray(ComplexF64.(vφ))
    gpu_fft!(gpu_vθ, 2)
    gpu_fft!(gpu_vφ, 2)

    # Transfer config data to GPU
    x_values = CuArray(cfg.x)
    weights = CuArray(cfg.w)
    normalization = CuArray(cfg.Nlm)
    scaleφ = cfg.cphi
    robert_form = cfg.robert_form

    # Compute Legendre polynomials AND their derivatives on GPU
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
                                                      x_vals, w_vals, norm, scale,
                                                      nlat, lmax, mmax, do_robert)
        i_lat, l_idx, m_idx = @index(Global, NTuple)

        if i_lat <= nlat && l_idx <= lmax + 1 && m_idx <= mmax + 1
            l = l_idx - 1
            m = m_idx - 1

            # Only compute for valid (l, m) pairs where l >= max(1, m)
            if l >= max(1, m)
                x = x_vals[i_lat]
                sθ = sqrt(max(0.0, 1.0 - x * x))
                inv_sθ = sθ < 1e-14 ? 0.0 : 1.0 / sθ
                wi = w_vals[i_lat]

                # Get Fourier modes for this latitude and m
                Fθ_val = Fθ[i_lat, m_idx]
                Fφ_val = Fφ[i_lat, m_idx]

                # Robert-form handling: input is sin(θ)*V, divide by sin(θ)
                if do_robert && sθ > 1e-14
                    Fθ_val /= sθ
                    Fφ_val /= sθ
                end

                # Get Legendre values
                N = norm[l_idx, m_idx]
                P = Plm[i_lat, l_idx, m_idx]
                dP = dPlm[i_lat, l_idx, m_idx]

                # ∂Y_l^m/∂θ = -sinθ * N * dP/dx
                dθY = -sθ * N * dP
                Y = N * P

                # Compute coefficient and term
                coeff = wi * scale / (l * (l + 1))
                term_re = 0.0
                term_im = m * inv_sθ * Y

                # Adjoint of synthesis formulas:
                # S_lm += coeff * (Fθ * dθY + conj(term) * Fφ)
                # T_lm += coeff * (-conj(term) * Fθ + dθY * Fφ)

                # conj(term) = (term_re, -term_im) = (0, -m*inv_sθ*Y)
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
                    x_values, weights, normalization, scaleφ,
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

    # Convert from internal to external normalization if needed (matching CPU behavior)
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        Slm_out = similar(Slm)
        Tlm_out = similar(Tlm)
        SHTnsKit.convert_alm_norm!(Slm_out, Slm, cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(Tlm_out, Tlm, cfg; to_internal=false)
        return Slm_out, Tlm_out
    else
        return Slm, Tlm
    end
end

"""
    gpu_SHsphtor_to_spat(cfg::SHTConfig, sph_coeffs, tor_coeffs; device=get_device(), real_output=true)

GPU-accelerated synthesis of spheroidal-toroidal vector field components using proper spectral method.

Uses the spectral formula:
    V_θ = ∂S/∂θ - (1/sinθ) ∂T/∂φ = Σ_{l,m} [∂Y_l^m/∂θ * S_lm - im/sinθ * Y_l^m * T_lm]
    V_φ = (1/sinθ) ∂S/∂φ + ∂T/∂θ = Σ_{l,m} [im/sinθ * Y_l^m * S_lm + ∂Y_l^m/∂θ * T_lm]

Where ∂Y_l^m/∂θ = -sinθ * N_lm * dP_l^m/dx (x = cosθ)
"""
function gpu_SHsphtor_to_spat(cfg::SHTConfig, sph_coeffs, tor_coeffs; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.SHsphtor_to_spat(cfg, sph_coeffs, tor_coeffs; real_output=real_output)
    end

    backend = CUDABackend()
    nlat, nlon = cfg.nlat, cfg.nlon
    lmax, mmax = cfg.lmax, cfg.mmax

    # Convert to internal normalization if needed (matching CPU behavior)
    Slm_int, Tlm_int = sph_coeffs, tor_coeffs
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        Slm_int = similar(sph_coeffs)
        Tlm_int = similar(tor_coeffs)
        SHTnsKit.convert_alm_norm!(Slm_int, sph_coeffs, cfg; to_internal=true)
        SHTnsKit.convert_alm_norm!(Tlm_int, tor_coeffs, cfg; to_internal=true)
    end

    # Transfer coefficients and normalization to GPU
    gpu_Slm = CuArray(ComplexF64.(Slm_int))
    gpu_Tlm = CuArray(ComplexF64.(Tlm_int))
    gpu_Nlm = CuArray(cfg.Nlm)
    x_values = CuArray(cfg.x)

    # Compute Legendre polynomials AND their derivatives on GPU
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

    # Kernel for spectral vector synthesis - compute Fourier modes for each (latitude, m)
    @kernel function vector_spectral_synthesis_kernel!(Ftheta, Fphi, Slm, Tlm, Plm, dPlm, Nlm, sintheta, nlat, nlon, lmax, mmax, inv_scale)
        i, m_idx = @index(Global, NTuple)
        if i <= nlat && m_idx <= mmax + 1
            m = m_idx - 1
            sθ = sintheta[i]
            inv_sθ = abs(sθ) < 1e-14 ? 0.0 : 1.0 / sθ

            gθ = ComplexF64(0, 0)
            gφ = ComplexF64(0, 0)

            @inbounds for l = m:lmax
                l_idx = l + 1
                N = Nlm[l_idx, m_idx]
                P = Plm[i, l_idx, m_idx]
                dP = dPlm[i, l_idx, m_idx]

                # Y_l^m contribution (without exp(imφ) which comes from FFT)
                Y = N * P
                # ∂Y_l^m/∂θ = -sinθ * N * dP/dx
                dYdθ = -sθ * N * dP

                Sl = Slm[l_idx, m_idx]
                Tl = Tlm[l_idx, m_idx]

                # V_θ = ∂S/∂θ - (im/sinθ) * T
                gθ += dYdθ * Sl - ComplexF64(0, m) * inv_sθ * Y * Tl
                # V_φ = (im/sinθ) * S + ∂T/∂θ
                gφ += ComplexF64(0, m) * inv_sθ * Y * Sl + dYdθ * Tl
            end

            # Store in Fourier coefficient array
            Ftheta[i, m_idx] = inv_scale * gθ
            Fphi[i, m_idx] = inv_scale * gφ
        end
    end

    synth_kernel! = vector_spectral_synthesis_kernel!(backend)
    synth_kernel!(Fθ, Fφ, gpu_Slm, gpu_Tlm, Plm, dPlm, gpu_Nlm, sintheta, nlat, nlon, lmax, mmax, inv_scaleφ; ndrange=(nlat, mmax+1))
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

@kernel function laplacian_kernel!(output, input, lmax, mmax)
    l, m = @index(Global, NTuple)
    if l <= lmax + 1 && m <= mmax + 1
        l_val = l - 1
        m_val = m - 1
        if l_val >= m_val
            output[l, m] = -l_val * (l_val + 1) * input[l, m]
        end
    end
end

"""
    gpu_apply_laplacian!(cfg::SHTConfig, coeffs; device=get_device())

GPU-accelerated Laplacian operator in spectral space.
"""
function gpu_apply_laplacian!(cfg::SHTConfig, coeffs; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.apply_laplacian!(cfg, coeffs)
    end

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
    gpu_memory_info(; device=nothing)

Get CUDA memory information. The `device` argument is accepted for API compatibility
but currently only returns info for the active CUDA device.
Returns a named tuple with `free` and `total` fields (in bytes).
"""
function gpu_memory_info(; device=nothing)
    if CUDA.functional()
        mem = CUDA.MemoryInfo()
        return (free=mem.free_bytes, total=mem.total_bytes)
    else
        return (free=Sys.free_memory(), total=Sys.total_memory())
    end
end

# Overload to accept positional device argument for compatibility
gpu_memory_info(device) = gpu_memory_info(; device=device)

"""
    check_gpu_memory(required_bytes::Int; device=nothing)

Check if sufficient GPU memory is available.
"""
function check_gpu_memory(required_bytes::Int; device=nothing)
    try
        mem_info = gpu_memory_info(; device=device)
        if mem_info.free < required_bytes
            @warn "Insufficient memory: need $(required_bytes÷(1024^3)) GB, have $(mem_info.free÷(1024^3)) GB available"
            return false
        end
        return true
    catch e
        @warn "Could not check memory availability: $e"
        return true
    end
end

"""
    gpu_clear_cache!(; device=nothing)

Clear CUDA memory cache. The `device` argument is accepted for API compatibility.
"""
function gpu_clear_cache!(; device=nothing)
    if CUDA.functional()
        try
            CUDA.reclaim()
            @info "CUDA memory cache cleared"
        catch e
            @warn "Failed to clear CUDA cache: $e"
        end
    end
    return nothing
end

# Overload to accept positional device argument for compatibility
gpu_clear_cache!(device) = gpu_clear_cache!(; device=device)

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

"""
    gpu_analysis_safe(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true)

Memory-safe GPU analysis with automatic fallback to CPU.
"""
function gpu_analysis_safe(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.analysis(cfg, spatial_data)
    end

    required_memory = estimate_memory_usage(cfg, :analysis)
    if !check_gpu_memory(required_memory)
        @info "Falling back to CPU due to memory constraints"
        return SHTnsKit.analysis(cfg, spatial_data)
    end

    try
        return gpu_analysis(cfg, spatial_data; device=device, real_output=real_output)
    catch e
        if isa(e, CUDA.OutOfGPUMemoryError) || contains(string(e), "memory")
            @warn "GPU out of memory, falling back to CPU: $e"
            return SHTnsKit.analysis(cfg, spatial_data)
        else
            rethrow(e)
        end
    end
end

"""
    gpu_synthesis_safe(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)

Memory-safe GPU synthesis with automatic fallback to CPU.
"""
function gpu_synthesis_safe(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.synthesis(cfg, coeffs; real_output=real_output)
    end

    required_memory = estimate_memory_usage(cfg, :synthesis)
    if !check_gpu_memory(required_memory)
        @info "Falling back to CPU due to memory constraints"
        return SHTnsKit.synthesis(cfg, coeffs; real_output=real_output)
    end

    try
        return gpu_synthesis(cfg, coeffs; device=device, real_output=real_output)
    catch e
        if isa(e, CUDA.OutOfGPUMemoryError) || contains(string(e), "memory")
            @warn "GPU out of memory, falling back to CPU: $e"
            return SHTnsKit.synthesis(cfg, coeffs; real_output=real_output)
        else
            rethrow(e)
        end
    end
end

# ============================================================================
# Helper functions for multi-GPU configs
# ============================================================================

"""
    create_latitude_subset_config(base_cfg, lat_indices, gpu_device::Symbol)

Create a temporary SHTConfig for a subset of latitude points.
Properly subsets plm_tables and dplm_tables if they exist.
"""
function create_latitude_subset_config(base_cfg, lat_indices, gpu_device::Symbol)
    chunk_nlat = length(lat_indices)

    # Subset plm_tables if they exist: tables are [m+1][l+1, lat_idx]
    subset_plm_tables = if base_cfg.use_plm_tables && !isempty(base_cfg.plm_tables)
        [tbl[:, lat_indices] for tbl in base_cfg.plm_tables]
    else
        Matrix{Float64}[]
    end

    subset_dplm_tables = if base_cfg.use_plm_tables && !isempty(base_cfg.dplm_tables)
        [tbl[:, lat_indices] for tbl in base_cfg.dplm_tables]
    else
        Matrix{Float64}[]
    end

    return SHTnsKit.SHTConfig(
        lmax=base_cfg.lmax, mmax=base_cfg.mmax, mres=base_cfg.mres,
        nlat=chunk_nlat, nlon=base_cfg.nlon, grid_type=base_cfg.grid_type,
        θ=base_cfg.θ[lat_indices], φ=base_cfg.φ,
        x=base_cfg.x[lat_indices], w=base_cfg.w[lat_indices],
        wlat=base_cfg.w[lat_indices], Nlm=base_cfg.Nlm, cphi=base_cfg.cphi,
        nlm=base_cfg.nlm, li=base_cfg.li, mi=base_cfg.mi,
        nspat=chunk_nlat * base_cfg.nlon,
        ct=base_cfg.ct[lat_indices], st=base_cfg.st[lat_indices],
        sintheta=base_cfg.st[lat_indices],
        norm=base_cfg.norm, cs_phase=base_cfg.cs_phase,
        real_norm=base_cfg.real_norm, robert_form=base_cfg.robert_form,
        phi_scale=base_cfg.phi_scale,
        use_plm_tables=base_cfg.use_plm_tables,
        plm_tables=subset_plm_tables,
        dplm_tables=subset_dplm_tables,
        compute_device=gpu_device,
        device_preference=[gpu_device]
    )
end

# ============================================================================
# Multi-GPU functions
# ============================================================================

"""
    multi_gpu_analysis(mgpu_config::MultiGPUConfig, spatial_data; real_output=true)

Perform spherical harmonic analysis using multiple GPUs.
"""
function multi_gpu_analysis(mgpu_config::MultiGPUConfig, spatial_data; real_output=true)
    cfg = mgpu_config.base_config
    ngpus = length(mgpu_config.gpu_devices)

    if mgpu_config.distribution_strategy != :latitude
        error("Multi-GPU analysis currently only supports :latitude distribution strategy")
    end

    # Split by latitude bands
    lat_per_gpu = div(cfg.nlat, ngpus)
    lat_remainder = cfg.nlat % ngpus

    final_coeffs = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)

    lat_start = 1
    for (i, gpu) in enumerate(mgpu_config.gpu_devices)
        lat_count = lat_per_gpu + (i <= lat_remainder ? 1 : 0)
        lat_end = lat_start + lat_count - 1
        lat_indices = lat_start:lat_end

        set_gpu_device(gpu.id)

        temp_cfg = create_latitude_subset_config(cfg, lat_indices, :cuda)
        chunk_data = spatial_data[lat_indices, :]
        chunk_coeffs = gpu_analysis(temp_cfg, chunk_data; real_output=false)

        final_coeffs .+= chunk_coeffs
        lat_start = lat_end + 1
    end

    if real_output && eltype(spatial_data) <: Real
        return real(final_coeffs)
    else
        return final_coeffs
    end
end

"""
    multi_gpu_synthesis(mgpu_config::MultiGPUConfig, coeffs; real_output=true)

Perform spherical harmonic synthesis using multiple GPUs.
"""
function multi_gpu_synthesis(mgpu_config::MultiGPUConfig, coeffs; real_output=true)
    cfg = mgpu_config.base_config
    ngpus = length(mgpu_config.gpu_devices)

    if mgpu_config.distribution_strategy != :latitude
        error("Multi-GPU synthesis currently only supports :latitude distribution strategy")
    end

    lat_per_gpu = div(cfg.nlat, ngpus)
    lat_remainder = cfg.nlat % ngpus

    final_result = zeros(real_output ? Float64 : ComplexF64, cfg.nlat, cfg.nlon)

    lat_start = 1
    for (i, gpu) in enumerate(mgpu_config.gpu_devices)
        lat_count = lat_per_gpu + (i <= lat_remainder ? 1 : 0)
        lat_end = lat_start + lat_count - 1
        lat_indices = lat_start:lat_end

        set_gpu_device(gpu.id)

        temp_cfg = create_latitude_subset_config(cfg, lat_indices, :cuda)
        chunk_result = gpu_synthesis(temp_cfg, coeffs; real_output=real_output)

        final_result[lat_indices, :] = chunk_result
        lat_start = lat_end + 1
    end

    return final_result
end

"""
    estimate_streaming_chunks(mgpu_config::MultiGPUConfig, data_size, max_memory_per_gpu=4*1024^3)

Estimate optimal chunk sizes for memory streaming.
"""
function estimate_streaming_chunks(mgpu_config::MultiGPUConfig, data_size, max_memory_per_gpu=4*1024^3)
    element_size = 16  # ComplexF64
    total_memory_needed = prod(data_size) * element_size * 3.0  # 3x overhead

    ngpus = length(mgpu_config.gpu_devices)
    memory_per_gpu = total_memory_needed / ngpus

    if memory_per_gpu <= max_memory_per_gpu
        return 1
    else
        return ceil(Int, memory_per_gpu / max_memory_per_gpu)
    end
end

"""
    multi_gpu_analysis_streaming(mgpu_config::MultiGPUConfig, spatial_data; max_memory_per_gpu=4*1024^3, real_output=true)

Multi-GPU analysis with memory streaming for large problems.
"""
function multi_gpu_analysis_streaming(mgpu_config::MultiGPUConfig, spatial_data;
                                     max_memory_per_gpu=4*1024^3, real_output=true)
    chunks_needed = estimate_streaming_chunks(mgpu_config, size(spatial_data), max_memory_per_gpu)

    if chunks_needed == 1
        return multi_gpu_analysis(mgpu_config, spatial_data; real_output=real_output)
    end

    @info "Using memory streaming with $chunks_needed chunks per GPU"

    cfg = mgpu_config.base_config
    lat_chunk_size = div(cfg.nlat, chunks_needed)
    lat_remainder = cfg.nlat % chunks_needed

    final_coeffs = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)

    lat_start = 1
    for chunk_idx in 1:chunks_needed
        chunk_lat_size = lat_chunk_size + (chunk_idx <= lat_remainder ? 1 : 0)
        lat_end = lat_start + chunk_lat_size - 1
        lat_indices = lat_start:lat_end

        chunk_data = spatial_data[lat_indices, :]
        chunk_base_cfg = create_latitude_subset_config(cfg, lat_indices, :cuda)
        chunk_mgpu_config = MultiGPUConfig(chunk_base_cfg, mgpu_config.gpu_devices, :latitude, mgpu_config.primary_gpu)

        chunk_coeffs = multi_gpu_analysis(chunk_mgpu_config, chunk_data; real_output=false)
        final_coeffs .+= chunk_coeffs

        lat_start = lat_end + 1
        gpu_clear_cache!()
    end

    if real_output && eltype(spatial_data) <: Real
        return real(final_coeffs)
    else
        return final_coeffs
    end
end

"""
    multi_gpu_synthesis_streaming(mgpu_config::MultiGPUConfig, coeffs; max_memory_per_gpu=4*1024^3, real_output=true)

Multi-GPU synthesis with memory streaming for large problems.
"""
function multi_gpu_synthesis_streaming(mgpu_config::MultiGPUConfig, coeffs;
                                      max_memory_per_gpu=4*1024^3, real_output=true)
    cfg = mgpu_config.base_config
    chunks_needed = estimate_streaming_chunks(mgpu_config, (cfg.nlat, cfg.nlon), max_memory_per_gpu)

    if chunks_needed == 1
        return multi_gpu_synthesis(mgpu_config, coeffs; real_output=real_output)
    end

    @info "Using memory streaming with $chunks_needed chunks per GPU"

    lat_chunk_size = div(cfg.nlat, chunks_needed)
    lat_remainder = cfg.nlat % chunks_needed

    final_result = zeros(real_output ? Float64 : ComplexF64, cfg.nlat, cfg.nlon)

    lat_start = 1
    for chunk_idx in 1:chunks_needed
        chunk_lat_size = lat_chunk_size + (chunk_idx <= lat_remainder ? 1 : 0)
        lat_end = lat_start + chunk_lat_size - 1
        lat_indices = lat_start:lat_end

        chunk_base_cfg = create_latitude_subset_config(cfg, lat_indices, :cuda)
        chunk_mgpu_config = MultiGPUConfig(chunk_base_cfg, mgpu_config.gpu_devices, :latitude, mgpu_config.primary_gpu)

        chunk_result = multi_gpu_synthesis(chunk_mgpu_config, coeffs; real_output=real_output)
        final_result[lat_indices, :] = chunk_result

        lat_start = lat_end + 1
        gpu_clear_cache!()
    end

    return final_result
end

# ============================================================================
# Local exports (types defined in this extension)
# ============================================================================

# These types are defined in this extension and need to be exported
# The GPU functions are already exported by SHTnsKit.jl and we extend them via import
export SHTDevice, CPU_DEVICE, CUDA_DEVICE
export CuFFTPlan, create_cufft_plan, gpu_fft!, gpu_ifft!
export MultiGPUConfig

end # module SHTnsKitGPUExt
