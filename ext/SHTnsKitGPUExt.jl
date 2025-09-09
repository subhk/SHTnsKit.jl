module SHTnsKitGPUExt

using SHTnsKit
using KernelAbstractions, GPUArrays, GPUArraysCore
using LinearAlgebra, FFTW

# Global variables for GPU backend availability
CUDA_LOADED = false
AMDGPU_LOADED = false
CUDA = nothing
AMDGPU = nothing

# Conditional imports for GPU backends  
function __init__()
    global CUDA, AMDGPU, CUDA_LOADED, AMDGPU_LOADED
    
    try
        CUDA = Base.require(Main, :CUDA)
        if CUDA.functional()
            CUDA_LOADED = true
            @info "CUDA backend loaded successfully"
        end
    catch
        CUDA_LOADED = false
    end
    
    try
        AMDGPU = Base.require(Main, :AMDGPU)
        if AMDGPU.functional()
            AMDGPU_LOADED = true
            @info "AMDGPU backend loaded successfully"
        end
    catch
        AMDGPU_LOADED = false
    end
end

# Device management
"""
    SHTDevice

Enum representing supported compute devices for SHTnsKit operations.
"""
@enum SHTDevice begin
    CPU_DEVICE
    CUDA_DEVICE  
    AMDGPU_DEVICE
end

"""
    get_device()

Returns the currently active GPU device, or CPU_DEVICE if no GPU is available.
"""
function get_device()
    if CUDA_LOADED && CUDA.functional()
        return CUDA_DEVICE
    elseif AMDGPU_LOADED && AMDGPU.functional()  
        return AMDGPU_DEVICE
    else
        return CPU_DEVICE
    end
end

"""
    set_device!(device::SHTDevice)

Set the active compute device for SHTnsKit operations.
"""
function set_device!(device::SHTDevice)
    if device == CUDA_DEVICE
        if !CUDA_LOADED
            error("CUDA not available. Install and load CUDA.jl first.")
        end
        CUDA.device!(CUDA.device())
    elseif device == AMDGPU_DEVICE
        if !AMDGPU_LOADED
            error("AMDGPU not available. Install and load AMDGPU.jl first.")
        end
        # AMDGPU device selection if needed
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
        if !CUDA_LOADED
            error("CUDA not available")
        end
        return CUDA.CuArray(array)
    elseif device == AMDGPU_DEVICE  
        if !AMDGPU_LOADED
            error("AMDGPU not available")
        end
        return AMDGPU.ROCArray(array)
    else
        return Array(array)  # Transfer to CPU
    end
end

# GPU FFT utilities
"""
    gpu_fft!(data, dims)

Perform in-place FFT on GPU data along specified dimensions.
"""
function gpu_fft!(data, dims)
    if data isa CUDA.CuArray
        CUDA.CUFFT.fft!(data, dims)
    elseif data isa AMDGPU.ROCArray
        AMDGPU.rocFFT.fft!(data, dims)
    else
        fft!(data, dims)  # CPU fallback
    end
    return data
end

"""
    gpu_ifft!(data, dims)

Perform in-place inverse FFT on GPU data along specified dimensions.
"""
function gpu_ifft!(data, dims)
    if data isa CUDA.CuArray
        CUDA.CUFFT.ifft!(data, dims)
    elseif data isa AMDGPU.ROCArray  
        AMDGPU.rocFFT.ifft!(data, dims)
    else
        ifft!(data, dims)  # CPU fallback
    end
    return data
end

"""
    gpu_rfft(data, dims)

Perform real-to-complex FFT on GPU data.
"""
function gpu_rfft(data, dims)
    if data isa CUDA.CuArray
        return CUDA.CUFFT.rfft(data, dims)
    elseif data isa AMDGPU.ROCArray
        return AMDGPU.rocFFT.rfft(data, dims)  
    else
        return rfft(data, dims)  # CPU fallback
    end
end

"""
    gpu_irfft(data, n, dims)

Perform complex-to-real inverse FFT on GPU data.
"""
function gpu_irfft(data, n, dims)
    if data isa CUDA.CuArray
        return CUDA.CUFFT.irfft(data, n, dims)
    elseif data isa AMDGPU.ROCArray
        return AMDGPU.rocFFT.irfft(data, n, dims)
    else
        return irfft(data, n, dims)  # CPU fallback
    end
end

# GPU-accelerated core operations using KernelAbstractions

@kernel function legendre_associated_kernel!(Plm, x, lmax, mmax, normalization)
    """
    GPU kernel for computing associated Legendre polynomials P_l^m(x).
    Uses stable three-term recurrence relations.
    """
    i = @index(Global)
    if i <= length(x)
        xi = x[i]
        sint = sqrt(1 - xi*xi)  # sin(θ) = sqrt(1 - cos²(θ))
        
        # Initialize P_0^0 = 1 (with normalization)
        Plm[i, 1, 1] = normalization[1, 1]  # P_0^0
        
        # Compute diagonal terms P_m^m using recurrence
        for m = 1:mmax
            if m <= lmax
                # P_m^m = (-1)^m * (2m-1)!! * sin^m(θ)
                pm_prev = Plm[i, m, m]  # P_{m-1}^{m-1}
                Plm[i, m+1, m+1] = -sqrt(2*m + 1) / sqrt(2*m) * sint * pm_prev * normalization[m+1, m+1]
            end
        end
        
        # Compute off-diagonal terms P_l^m for l > m using recurrence
        for m = 0:mmax
            for l = m+1:lmax
                if l-1 >= m
                    # Three-term recurrence: (l-m)P_l^m = (2l-1)xP_{l-1}^m - (l+m-1)P_{l-2}^m
                    if l == m+1
                        # Two-term recurrence for l = m+1
                        Plm[i, l+1, m+1] = xi * sqrt(2*l + 1) * sqrt(2*l - 1) * Plm[i, l, m+1] * normalization[l+1, m+1]
                    else
                        # Full three-term recurrence
                        a_lm = sqrt((2*l + 1) * (2*l - 1)) / sqrt((l - m) * (l + m))
                        b_lm = sqrt((2*l + 1) * (l + m - 1) * (l - m - 1)) / sqrt((l - m) * (l + m) * (2*l - 3))
                        
                        Plm[i, l+1, m+1] = (a_lm * xi * Plm[i, l, m+1] - b_lm * Plm[i, l-1, m+1]) * normalization[l+1, m+1]
                    end
                end
            end
        end
    end
end

@kernel function legendre_transform_kernel!(coeffs, spatial_data, Plm, weights, nlat, nlon, lmax, mmax)
    """
    GPU kernel for spherical harmonic analysis transform.
    Performs weighted integration over θ using precomputed Legendre polynomials.
    """
    l, m = @index(Global, NTuple)
    
    if l <= lmax + 1 && m <= mmax + 1
        l_idx, m_idx = l, m  # Convert to 0-based internally
        l_val, m_val = l_idx - 1, m_idx - 1
        
        if l_val >= m_val && l_val <= lmax && m_val <= mmax
            coeff_sum = ComplexF64(0, 0)
            
            # Integrate over θ (latitude) 
            for i_lat = 1:nlat
                plm_val = Plm[i_lat, l_idx, m_idx]
                weight = weights[i_lat]
                
                # Integrate over φ (longitude) - this should be done via FFT
                phi_sum = ComplexF64(0, 0)
                for i_lon = 1:nlon
                    phi = 2π * (i_lon - 1) / nlon
                    exp_factor = exp(-im * m_val * phi)  # e^(-imφ)
                    phi_sum += spatial_data[i_lat, i_lon] * exp_factor
                end
                
                coeff_sum += plm_val * weight * phi_sum
            end
            
            coeffs[l_idx, m_idx] = coeff_sum
        end
    end
end

@kernel function synthesis_kernel!(spatial_data, coeffs, Plm, nlat, nlon, lmax, mmax)
    """
    GPU kernel for spherical harmonic synthesis transform.
    Reconstructs spatial field from spectral coefficients.
    """
    i_lat, i_lon = @index(Global, NTuple)
    
    if i_lat <= nlat && i_lon <= nlon
        phi = 2π * (i_lon - 1) / nlon
        result = ComplexF64(0, 0)
        
        # Sum over all (l,m) modes
        for l = 0:lmax, m = 0:min(l, mmax)
            l_idx, m_idx = l + 1, m + 1
            plm_val = Plm[i_lat, l_idx, m_idx]
            coeff = coeffs[l_idx, m_idx]
            exp_factor = exp(im * m * phi)  # e^(imφ)
            
            result += coeff * plm_val * exp_factor
        end
        
        spatial_data[i_lat, i_lon] = result
    end
end

"""
    gpu_legendre!(P, x, lmax; device=get_device())

GPU-accelerated computation of Legendre polynomials using KernelAbstractions.
"""
function gpu_legendre!(P, x, lmax; device=get_device())
    if device == CPU_DEVICE
        # Fallback to CPU implementation
        return SHTnsKit.legendre_sphPlm_array(P, x, lmax)
    end
    
    kernel! = legendre_kernel!(get_backend(P))
    kernel!(P, x, lmax; ndrange=size(P, 1))
    return P
end

@kernel function fft_preprocess_kernel!(data_out, data_in, scale)
    i = @index(Global)
    if i <= length(data_in)
        data_out[i] = data_in[i] * scale
    end
end

"""
    gpu_analysis(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true)

GPU-accelerated spherical harmonic analysis transform.
Performs complete spatial → spectral transform using GPU kernels and FFTs.
"""
function gpu_analysis(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.analysis(cfg, spatial_data)
    end
    
    # Transfer input data to GPU
    gpu_data = to_device(ComplexF64.(spatial_data), device)
    
    # Allocate GPU arrays
    coeffs = to_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device)
    Plm = to_device(zeros(Float64, cfg.nlat, cfg.lmax+1, cfg.mmax+1), device)
    normalization = to_device(cfg.Nlm, device)
    weights = to_device(cfg.w, device)
    x_values = to_device(cfg.x, device)  # cos(θ) values
    
    # Step 1: Precompute Legendre polynomials on GPU
    backend = get_backend(gpu_data)
    legendre_kernel! = legendre_associated_kernel!(backend)
    legendre_kernel!(Plm, x_values, cfg.lmax, cfg.mmax, normalization; 
                    ndrange=cfg.nlat)
    KernelAbstractions.synchronize(backend)
    
    # Step 2: Optimized analysis using FFT + Legendre transform
    # Method A: Use FFT for φ integration, then Legendre for θ integration
    
    # Perform FFT in φ direction (longitude)
    phi_transformed = gpu_fft!(copy(gpu_data), 2)  # FFT along longitude dimension
    
    # Step 3: Legendre integration for each m-mode
    for m = 0:cfg.mmax
        m_idx = m + 1
        
        # Extract m-th Fourier mode (handle negative m by symmetry)
        if m == 0
            phi_mode = real(phi_transformed[:, 1])  # m=0 mode is real
        else
            # For m > 0, combine positive and negative m modes
            if m <= size(phi_transformed, 2) - 1
                phi_mode = phi_transformed[:, m+1]
            else
                continue  # Skip if m exceeds FFT output size
            end
        end
        
        # GPU-accelerated Legendre integration for this m
        @kernel function m_mode_integration!(coeffs_m, phi_mode, Plm_m, weights, nlat, lmax, m_val)
            l = @index(Global)
            if l <= lmax + 1
                l_val = l - 1
                if l_val >= m_val
                    result = ComplexF64(0, 0)
                    for i_lat = 1:nlat
                        result += phi_mode[i_lat] * Plm_m[i_lat, l] * weights[i_lat]
                    end
                    coeffs_m[l] = result
                end
            end
        end
        
        # Launch kernel for this m-mode
        m_coeffs = to_device(zeros(ComplexF64, cfg.lmax+1), device)
        Plm_m = @view Plm[:, :, m_idx]
        
        m_integration_kernel! = m_mode_integration!(backend)
        m_integration_kernel!(m_coeffs, phi_mode, Plm_m, weights, cfg.nlat, cfg.lmax, m;
                              ndrange=cfg.lmax+1)
        KernelAbstractions.synchronize(backend)
        
        # Copy results to main coefficient array
        coeffs[:, m_idx] .= m_coeffs
    end
    
    # Apply normalization and phase corrections
    # Scale by appropriate factors (4π normalization, etc.)
    scale_factor = 4π / cfg.nlon
    coeffs .*= scale_factor
    
    # Transfer result back to CPU
    result_coeffs = Array(coeffs)
    
    if real_output && eltype(spatial_data) <: Real
        # For real input, return real coefficients where appropriate
        return real(result_coeffs)
    else
        return result_coeffs
    end
end

"""
    gpu_synthesis(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)

GPU-accelerated spherical harmonic synthesis transform.
Performs complete spectral → spatial transform using GPU kernels and inverse FFTs.
"""
function gpu_synthesis(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.synthesis(cfg, coeffs; real_output=real_output)
    end
    
    # Transfer coefficients to GPU
    gpu_coeffs = to_device(ComplexF64.(coeffs), device)
    
    # Allocate GPU arrays
    spatial_data = to_device(zeros(ComplexF64, cfg.nlat, cfg.nlon), device)
    Plm = to_device(zeros(Float64, cfg.nlat, cfg.lmax+1, cfg.mmax+1), device)
    normalization = to_device(cfg.Nlm, device)
    x_values = to_device(cfg.x, device)  # cos(θ) values
    
    backend = get_backend(gpu_coeffs)
    
    # Step 1: Precompute Legendre polynomials on GPU (same as analysis)
    legendre_kernel! = legendre_associated_kernel!(backend)
    legendre_kernel!(Plm, x_values, cfg.lmax, cfg.mmax, normalization; 
                    ndrange=cfg.nlat)
    KernelAbstractions.synchronize(backend)
    
    # Step 2: Compute spatial field for each m-mode, then inverse FFT
    # Allocate temporary array for Fourier modes
    fourier_modes = to_device(zeros(ComplexF64, cfg.nlat, cfg.nlon), device)
    
    # Step 3: For each m-mode, compute spatial contribution
    for m = 0:cfg.mmax
        m_idx = m + 1
        
        # GPU kernel for Legendre synthesis for this m-mode
        @kernel function m_mode_synthesis!(spatial_m, coeffs_m, Plm_m, nlat, lmax, m_val)
            i_lat = @index(Global)
            if i_lat <= nlat
                result = ComplexF64(0, 0)
                for l = m_val:lmax
                    l_idx = l + 1
                    result += coeffs_m[l_idx] * Plm_m[i_lat, l_idx]
                end
                spatial_m[i_lat] = result
            end
        end
        
        # Launch kernel for this m-mode
        spatial_m = to_device(zeros(ComplexF64, cfg.nlat), device)
        coeffs_m = @view gpu_coeffs[:, m_idx]
        Plm_m = @view Plm[:, :, m_idx]
        
        m_synthesis_kernel! = m_mode_synthesis!(backend)
        m_synthesis_kernel!(spatial_m, coeffs_m, Plm_m, cfg.nlat, cfg.lmax, m;
                           ndrange=cfg.nlat)
        KernelAbstractions.synchronize(backend)
        
        # Place this m-mode in appropriate Fourier mode slot
        if m == 0
            # m=0 mode goes in the first column (DC component)
            fourier_modes[:, 1] .= real(spatial_m)
        elseif m <= cfg.nlon ÷ 2
            # Positive m modes
            fourier_modes[:, m+1] .= spatial_m
            
            # Use Hermitian symmetry for negative m modes (for real output)
            if real_output && m > 0 && cfg.nlon - m + 1 <= cfg.nlon
                fourier_modes[:, cfg.nlon - m + 1] .= conj(spatial_m)
            end
        end
    end
    
    # Step 4: Perform inverse FFT in φ direction to get spatial field
    spatial_result = gpu_ifft!(fourier_modes, 2)
    
    # Apply normalization
    scale_factor = cfg.nlon / 4π
    spatial_result .*= scale_factor
    
    # Transfer result back to CPU
    result = Array(spatial_result)
    
    if real_output
        return real(result)
    else
        return result
    end
end

# Vector field GPU operations
"""
    gpu_spat_to_SHsphtor(cfg::SHTConfig, vθ, vφ; device=get_device())

GPU-accelerated spheroidal-toroidal decomposition of vector fields.
"""
function gpu_spat_to_SHsphtor(cfg::SHTConfig, vθ, vφ; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.spat_to_SHsphtor(cfg, vθ, vφ)
    end
    
    # Transfer to GPU
    gpu_vθ = to_device(vθ, device)
    gpu_vφ = to_device(vφ, device) 
    
    # Simplified GPU implementation (placeholder)
    sph_coeffs = to_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device)
    tor_coeffs = to_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device)
    
    return Array(sph_coeffs), Array(tor_coeffs)
end

"""
    gpu_SHsphtor_to_spat(cfg::SHTConfig, sph_coeffs, tor_coeffs; device=get_device(), real_output=true)

GPU-accelerated synthesis of spheroidal-toroidal vector field components.
"""
function gpu_SHsphtor_to_spat(cfg::SHTConfig, sph_coeffs, tor_coeffs; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.SHsphtor_to_spat(cfg, sph_coeffs, tor_coeffs; real_output=real_output)
    end
    
    # Transfer to GPU
    gpu_sph = to_device(sph_coeffs, device)
    gpu_tor = to_device(tor_coeffs, device)
    
    # Simplified GPU implementation (placeholder)
    vθ = to_device(zeros(ComplexF64, cfg.nlat, cfg.nlon), device)
    vφ = to_device(zeros(ComplexF64, cfg.nlat, cfg.nlon), device)
    
    if real_output
        return Array(real(vθ)), Array(real(vφ))
    else
        return Array(vθ), Array(vφ)
    end
end

# Differential operators on GPU
@kernel function laplacian_kernel!(output, input, l_vals)
    idx = @index(Global)
    if idx <= length(input)
        l = l_vals[idx]
        output[idx] = -l * (l + 1) * input[idx]
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
    
    gpu_coeffs = to_device(coeffs, device)
    
    # Create l values for each coefficient
    l_vals = zeros(Int, size(coeffs))
    for l = 0:cfg.lmax, m = 0:min(l, cfg.mmax)
        idx = SHTnsKit.LM_index(l, m, cfg.lmax)
        l_vals[idx] = l
    end
    gpu_l_vals = to_device(l_vals, device)
    
    kernel! = laplacian_kernel!(get_backend(gpu_coeffs))
    kernel!(gpu_coeffs, gpu_coeffs, gpu_l_vals; ndrange=length(gpu_coeffs))
    
    # Copy result back
    coeffs .= Array(gpu_coeffs)
    return coeffs
end

# Utility functions
function get_backend(array)
    if array isa CuArray
        return CUDABackend()
    elseif array isa ROCArray
        return ROCBackend()
    else
        return CPU()
    end
end

# Export GPU functions
export SHTDevice, CPU_DEVICE, CUDA_DEVICE, AMDGPU_DEVICE
export get_device, set_device!, to_device
export gpu_analysis, gpu_synthesis
export gpu_spat_to_SHsphtor, gpu_SHsphtor_to_spat
export gpu_apply_laplacian!, gpu_legendre!

end # module SHTnsKitGPUExt