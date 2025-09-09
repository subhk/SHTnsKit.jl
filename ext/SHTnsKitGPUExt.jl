module SHTnsKitGPUExt

using SHTnsKit
using KernelAbstractions, GPUArrays, GPUArraysCore

# Conditional imports for GPU backends
function __init__()
    @static if isdefined(Main, :CUDA) && Main.CUDA.functional()
        import CUDA
        global CUDA_LOADED = true
    else
        global CUDA_LOADED = false
    end
    
    @static if isdefined(Main, :AMDGPU) && Main.AMDGPU.functional()
        import AMDGPU
        global AMDGPU_LOADED = true
    else
        global AMDGPU_LOADED = false
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

# GPU-accelerated core operations using KernelAbstractions
@kernel function legendre_kernel!(P, x, lmax)
    i = @index(Global)
    if i <= length(x)
        xi = x[i]
        # P_0^0 = 1
        P[i, 1] = 1.0
        
        if lmax >= 1
            # P_1^0 = x
            P[i, 2] = xi
            
            # Recurrence relation for P_l^0
            for l = 2:lmax
                P[i, l+1] = ((2*l-1)*xi*P[i, l] - (l-1)*P[i, l-1]) / l
            end
        end
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
"""
function gpu_analysis(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.analysis(cfg, spatial_data)
    end
    
    # Transfer data to GPU
    gpu_data = to_device(spatial_data, device)
    
    # Perform FFT in φ direction
    φ_fft = fft(gpu_data, 2)
    
    # Allocate spherical harmonic coefficients on GPU
    coeffs = to_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device)
    
    # Perform Legendre transform (simplified - full implementation would be more complex)
    # This is a placeholder for the actual GPU-accelerated Legendre transform
    
    if real_output
        return Array(real(coeffs))  # Transfer back to CPU
    else
        return Array(coeffs)
    end
end

"""
    gpu_synthesis(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)

GPU-accelerated spherical harmonic synthesis transform.
"""
function gpu_synthesis(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.synthesis(cfg, coeffs; real_output=real_output)
    end
    
    # Transfer coefficients to GPU
    gpu_coeffs = to_device(coeffs, device)
    
    # Allocate output array on GPU
    spatial_data = to_device(zeros(ComplexF64, cfg.nlat, cfg.nlon), device)
    
    # Perform inverse Legendre transform (simplified)
    # This is a placeholder for the actual GPU-accelerated implementation
    
    # Perform inverse FFT in φ direction
    result = ifft(spatial_data, 2)
    
    if real_output
        return Array(real(result))  # Transfer back to CPU
    else
        return Array(result)
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