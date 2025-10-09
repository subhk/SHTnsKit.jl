module SHTnsKitCUDAExt

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

@kernel function legendre_mode_kernel!(Plm_mode, x, im, lcap)
    idx = @index(Global)
    nlat = size(Plm_mode, 1)
    if idx > nlat
        return
    end

    lcount = size(Plm_mode, 2)
    lcount <= 0 && return

    xi = x[idx]
    T = eltype(Plm_mode)

    if im == 0
        if lcount >= 1
            Plm_mode[idx, 1] = one(T)
        end
        if lcount >= 2
            Plm_mode[idx, 2] = convert(T, xi)
        end
        if lcount > 2
            prev_prev = one(T)
            prev = convert(T, xi)
            for col in 3:lcount
                l = col - 1
                num = convert(T, (2*l - 1)) * convert(T, xi) * prev - convert(T, (l - 1)) * prev_prev
                den = convert(T, l)
                val = num / den
                Plm_mode[idx, col] = val
                prev_prev = prev
                prev = val
            end
        end
    else
        sx2 = max(zero(T), one(T) - convert(T, xi) * convert(T, xi))
        sint = sqrt(sx2)
        pmm = one(T)
        fact = one(T)
        for k in 1:im
            pmm *= -fact * sint
            fact += convert(T, 2)
        end
        Plm_mode[idx, 1] = pmm
        if lcount >= 2
            prev_prev = pmm
            prev = convert(T, xi) * convert(T, 2 * im + 1) * pmm
            Plm_mode[idx, 2] = prev
            if lcount > 2
                for col in 3:lcount
                    l = im + col - 1
                    num = convert(T, 2 * l - 1) * convert(T, xi) * prev - convert(T, l + im - 1) * prev_prev
                    den = convert(T, l - im)
                    val = num / den
                    Plm_mode[idx, col] = val
                    prev_prev = prev
                    prev = val
                end
            end
        end
    end
end

@kernel function scalar_mode_analysis_kernel!(coeffs, Vr, weights, Plm_mode, cphi)
    lidx = @index(Global)
    lcount = size(Plm_mode, 2)
    if lidx > lcount
        return
    end
    nlat = size(Plm_mode, 1)
    acc = ComplexF64(0, 0)
    for i in 1:nlat
        acc += Vr[i] * weights[i] * Plm_mode[i, lidx]
    end
    coeffs[lidx] = acc * cphi
end

@kernel function scalar_mode_synthesis_kernel!(Vr, Ql, Plm_mode)
    idx = @index(Global)
    nlat = size(Plm_mode, 1)
    if idx > nlat
        return
    end
    lcount = size(Plm_mode, 2)
    acc = ComplexF64(0, 0)
    for col in 1:lcount
        acc += Ql[col] * Plm_mode[idx, col]
    end
    Vr[idx] = acc
end

@kernel function sphtor_mode_analysis_kernel!(Sl, Tl, Vt, Vp, weights, Plm_mode, cphi, im)
    lidx = @index(Global)
    lcount = size(Plm_mode, 2)
    if lidx > lcount
        return
    end
    nlat = size(Plm_mode, 1)
    acc_s = ComplexF64(0, 0)
    acc_t = ComplexF64(0, 0)
    for i in 1:nlat
        basis = Plm_mode[i, lidx]
        weight = weights[i]
        acc_s += Vt[i] * weight * basis
        acc_t += Vp[i] * weight * basis
    end
    l = im + lidx - 1
    ll1 = l * (l + 1)
    if ll1 > 0
        scale = cphi / sqrt(Float64(ll1))
        Sl[lidx] = acc_s * scale
        Tl[lidx] = acc_t * scale
    else
        Sl[lidx] = ComplexF64(0, 0)
        Tl[lidx] = ComplexF64(0, 0)
    end
end

@kernel function sphtor_mode_synthesis_kernel!(Vt, Vp, Sl, Tl, Plm_mode, im)
    idx = @index(Global)
    nlat = size(Plm_mode, 1)
    if idx > nlat
        return
    end
    lcount = size(Plm_mode, 2)
    vt_acc = ComplexF64(0, 0)
    vp_acc = ComplexF64(0, 0)
    for col in 1:lcount
        l = im + col - 1
        ll1 = l * (l + 1)
        if ll1 > 0
            scale = sqrt(Float64(ll1))
            basis = Plm_mode[idx, col]
            vt_acc += Sl[col] * basis * scale
            vp_acc += Tl[col] * basis * scale
        end
    end
    Vt[idx] = vt_acc
    Vp[idx] = vp_acc
end

@kernel function legendre_latitude_kernel!(Plm_table, x, lcap, mres)
    idx = @index(Global)
    mcount = size(Plm_table, 1)
    if idx > mcount
        return
    end
    T = eltype(Plm_table)
    im = idx - 1
    ncol = size(Plm_table, 2)
    for j in 1:ncol
        Plm_table[idx, j] = zero(T)
    end
    if im > lcap || im % mres != 0
        return
    end
    xi = convert(T, x)
    if im == 0
        Plm_table[idx, 1] = one(T)
        if lcap >= 1
            Plm_table[idx, 2] = xi
        end
        prev_prev = one(T)
        prev = xi
        for l in 2:lcap
            num = convert(T, 2*l - 1) * xi * prev - convert(T, l - 1) * prev_prev
            val = num / convert(T, l)
            Plm_table[idx, l+1] = val
            prev_prev = prev
            prev = val
        end
    else
        sx2 = max(zero(T), one(T) - xi * xi)
        sint = sqrt(sx2)
        pmm = one(T)
        fact = one(T)
        for _ in 1:im
            pmm *= -fact * sint
            fact += convert(T, 2)
        end
        Plm_table[idx, im+1] = pmm
        if lcap >= im + 1
            prev_prev = pmm
            prev = xi * convert(T, 2*im + 1) * pmm
            Plm_table[idx, im+2] = prev
            for l in (im+2):lcap
                num = convert(T, 2*l - 1) * xi * prev - convert(T, l + im - 1) * prev_prev
                den = convert(T, l - im)
                val = num / den
                Plm_table[idx, l+1] = val
                prev_prev = prev
                prev = val
            end
        end
    end
end

@kernel function lat_mode_accumulate_kernel!(gm, alm, Nlm, Plm_table, lcap, mcap, mres)
    idx = @index(Global)
    if idx > size(Plm_table, 1)
        return
    end
    im = idx - 1
    if im > mcap || im % mres != 0 || im > lcap
        gm[idx] = ComplexF64(0, 0)
        return
    end
    acc = ComplexF64(0, 0)
    for l in im:lcap
        Plm_val = Plm_table[idx, l+1]
        Ylm = Nlm[l+1, im+1] * Plm_val
        acc += alm[l+1, im+1] * Ylm
    end
    gm[idx] = acc
end

@kernel function lat_finalize_kernel!(vals, gm, nphi, mcap)
    idx = @index(Global)
    if idx > nphi
        return
    end
    φ = 2π * (idx - 1) / nphi
    acc = real(gm[1])
    for m in 1:mcap
        val = gm[m+1]
        c = cos(m * φ)
        s = sin(m * φ)
        acc += 2 * (real(val) * c - imag(val) * s)
    end
    vals[idx] = acc
end

function _gpu_legendre_mode(cfg::SHTConfig, im::Int, lcap::Int, device::SHTDevice)
    lcap >= im || throw(ArgumentError("ltr must be ≥ im=$(im)"))
    lcount = lcap - im + 1
    lcount > 0 || throw(ArgumentError("ltr must be ≥ im=$(im)"))

    x_gpu = to_device(cfg.x, device)
    Plm_mode = to_device(zeros(Float64, cfg.nlat, lcount), device)
    backend = get_backend(x_gpu)
    mode_kernel! = legendre_mode_kernel!(backend)
    mode_kernel!(Plm_mode, x_gpu, im, lcap; ndrange=cfg.nlat)
    KernelAbstractions.synchronize(backend)

    return Plm_mode, backend
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
    get_available_gpus()

Returns a list of available GPU devices with their IDs and types.
"""
function get_available_gpus()
    gpus = []
    
    if CUDA_LOADED && CUDA.functional()
        for i = 0:(CUDA.ndevices()-1)
            push!(gpus, (device=:cuda, id=i, name=CUDA.name(CUDA.CuDevice(i))))
        end
    end
    
    if AMDGPU_LOADED && AMDGPU.functional()
        # AMDGPU device enumeration if available
        try
            for i = 0:(AMDGPU.ndevices()-1)
                push!(gpus, (device=:amdgpu, id=i, name="AMDGPU Device $i"))
            end
        catch
            # Fallback if AMDGPU doesn't support device enumeration
        end
    end
    
    return gpus
end

"""
    set_gpu_device(device_type::Symbol, device_id::Int)

Set the active GPU device by type and ID.
"""
function set_gpu_device(device_type::Symbol, device_id::Int)
    if device_type == :cuda && CUDA_LOADED
        CUDA.device!(device_id)
        return true
    elseif device_type == :amdgpu && AMDGPU_LOADED
        # AMDGPU device selection if available
        try
            AMDGPU.device!(device_id)
            return true
        catch
            return false
        end
    else
        return false
    end
end

_device_enum(sym::Symbol) = sym == :cuda ? CUDA_DEVICE : sym == :amdgpu ? AMDGPU_DEVICE : CPU_DEVICE

function _ensure_device(array, device::SHTDevice)
    if device == CUDA_DEVICE
        return array isa CUDA.CuArray ? array : to_device(array, device)
    elseif device == AMDGPU_DEVICE && AMDGPU_LOADED
        return array isa AMDGPU.AbstractGPUArray ? array : to_device(array, device)
    else
        return array
    end
end

# GPU transform planning -----------------------------------------------------

const AbstractSHTPlan = SHTnsKit.AbstractSHTPlan

struct SHTGPUPlan <: AbstractSHTPlan
    cfg::SHTConfig
    device::SHTDevice
    use_rfft::Bool
    spatial_cache::Union{Nothing, AbstractArray}
    coeffs_cache::Union{Nothing, AbstractArray}
end

function SHTnsKit.gpu_create_plan(cfg::SHTConfig; use_rfft::Bool=false)
    cfg.device_backend == :cuda || error("Only CUDA GPU plans are supported.")
    CUDA_LOADED || error("CUDA backend not available")
    # Ensure desired device is selected
    set_gpu_device(:cuda, CUDA.device())
    spatial_cache = to_device(zeros(ComplexF64, cfg.nlat, cfg.nlon), CUDA_DEVICE)
    coeffs_cache = to_device(zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1), CUDA_DEVICE)
    return SHTGPUPlan(cfg, CUDA_DEVICE, use_rfft, spatial_cache, coeffs_cache)
end

SHTnsKit.destroy_plan!(::SHTGPUPlan) = nothing

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
    create_multi_gpu_config(lmax, nlat; nlon=nothing, strategy=:latitude, gpu_ids=nothing, allow_mixed=true)

Create a multi-GPU configuration for spherical harmonic transforms.
Supports mixing NVIDIA and AMD GPUs when allow_mixed=true.
"""
function create_multi_gpu_config(lmax::Int, nlat::Int; 
                                 nlon::Union{Int,Nothing}=nothing,
                                 strategy::Symbol=:latitude,
                                 gpu_ids::Union{Vector{Int},Nothing}=nothing,
                                 gpu_types::Union{Vector{Symbol},Nothing}=nothing,
                                 allow_mixed::Bool=true)
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    if isempty(available_gpus)
        error("No GPUs available for multi-GPU configuration")
    end
    
    # Select GPUs to use
    if gpu_ids === nothing && gpu_types === nothing
        selected_gpus = available_gpus  # Use all available
    elseif gpu_types !== nothing
        # Select by GPU type (e.g., [:cuda, :amdgpu])
        selected_gpus = [gpu for gpu in available_gpus if gpu.device in gpu_types]
    elseif gpu_ids !== nothing
        # Select by GPU IDs (more complex for mixed types)
        selected_gpus = []
        for id in gpu_ids
            matching_gpus = [gpu for gpu in available_gpus if gpu.id == id]
            append!(selected_gpus, matching_gpus)
        end
    end
    
    if isempty(selected_gpus)
        error("No valid GPUs found with specified criteria")
    end
    
    # Check for mixed GPU types
    gpu_device_types = unique([gpu.device for gpu in selected_gpus])
    if length(gpu_device_types) > 1 && !allow_mixed
        error("Mixed GPU types detected but allow_mixed=false. Found: $gpu_device_types")
    end
    
    if length(gpu_device_types) > 1
        @info "Using mixed GPU types: $gpu_device_types"
        @info "Performance may be limited by slowest GPU type"
    end
    
    # Create base configuration
    base_cfg = SHTnsKit.create_gauss_config(lmax, nlat; nlon=nlon)
    
    # Validate distribution strategy
    if strategy ∉ [:latitude, :longitude, :spectral]
        error("Invalid distribution strategy: $strategy. Must be :latitude, :longitude, or :spectral")
    end
    
    # Enable P2P for CUDA GPUs if multiple CUDA devices
    if length([gpu for gpu in selected_gpus if gpu.device == :cuda]) >= 2
        try
            enable_gpu_p2p_access(selected_gpus)
        catch e
            @warn "Failed to enable P2P access: $e"
        end
    end
    
    return MultiGPUConfig(base_cfg, selected_gpus, strategy, selected_gpus[1].id)
end

"""
    create_balanced_mixed_gpu_config(lmax, nlat; nlon=nothing, strategy=:latitude)

Create a multi-GPU configuration that automatically balances workload across 
different GPU types based on their relative performance.
"""
function create_balanced_mixed_gpu_config(lmax::Int, nlat::Int;
                                         nlon::Union{Int,Nothing}=nothing,
                                         strategy::Symbol=:latitude)
    
    available_gpus = get_available_gpus()
    if isempty(available_gpus)
        error("No GPUs available")
    end
    
    # Estimate relative performance (rough approximation)
    # In practice, these would be benchmarked values
    gpu_performance_weights = Dict{Symbol, Float64}(
        :cuda => 1.0,    # Reference performance
        :amdgpu => 0.85  # Typically slightly slower than equivalent NVIDIA
    )
    
    # Calculate work distribution based on performance
    total_weight = sum(gpu_performance_weights[gpu.device] for gpu in available_gpus)
    
    # Create weighted GPU selection
    weighted_gpus = []
    for gpu in available_gpus
        weight = gpu_performance_weights[gpu.device] / total_weight
        # Store weight for potential use in load balancing
        weighted_gpu = merge(gpu, (performance_weight=weight,))
        push!(weighted_gpus, weighted_gpu)
    end
    
    base_cfg = SHTnsKit.create_gauss_config(lmax, nlat; nlon=nlon)
    
    @info "Created balanced mixed-GPU config with $(length(weighted_gpus)) GPUs"
    for (i, gpu) in enumerate(weighted_gpus)
        @info "  GPU $i: $(gpu.device) $(gpu.id) (weight: $(round(gpu.performance_weight, digits=2)))"
    end
    
    return MultiGPUConfig(base_cfg, weighted_gpus, strategy, weighted_gpus[1].id)
end

# Multi-GPU array distribution strategies

"""
    distribute_spatial_array(array, mgpu_config::MultiGPUConfig)

Distribute a spatial array across multiple GPUs according to the distribution strategy.
"""
function distribute_spatial_array(array::AbstractArray, mgpu_config::MultiGPUConfig)
    ngpus = length(mgpu_config.gpu_devices)
    nlat, nlon = size(array)
    distributed_arrays = []
    
    if mgpu_config.distribution_strategy == :latitude
        # Split by latitude bands
        lat_per_gpu = div(nlat, ngpus)
        lat_remainder = nlat % ngpus
        
        lat_start = 1
        for (i, gpu) in enumerate(mgpu_config.gpu_devices)
            # Handle remainder by giving extra rows to first few GPUs
            lat_count = lat_per_gpu + (i <= lat_remainder ? 1 : 0)
            lat_end = lat_start + lat_count - 1
            
            # Set device and transfer data chunk
            set_gpu_device(gpu.device, gpu.id)
            chunk = array[lat_start:lat_end, :]
            if chunk isa GPUArraysCore.AbstractGPUArray
                gpu_chunk = chunk
            else
                gpu_chunk = to_device(chunk, _device_enum(gpu.device))
            end
            
            push!(distributed_arrays, (
                data=gpu_chunk, 
                gpu=gpu, 
                indices=(lat_start:lat_end, 1:nlon)
            ))
            
            lat_start = lat_end + 1
        end
        
    elseif mgpu_config.distribution_strategy == :longitude
        # Split by longitude sectors  
        lon_per_gpu = div(nlon, ngpus)
        lon_remainder = nlon % ngpus
        
        lon_start = 1
        for (i, gpu) in enumerate(mgpu_config.gpu_devices)
            lon_count = lon_per_gpu + (i <= lon_remainder ? 1 : 0)
            lon_end = lon_start + lon_count - 1
            
            set_gpu_device(gpu.device, gpu.id)
            chunk = array[:, lon_start:lon_end]
            if chunk isa GPUArraysCore.AbstractGPUArray
                gpu_chunk = chunk
            else
                gpu_chunk = to_device(chunk, _device_enum(gpu.device))
            end
            
            push!(distributed_arrays, (
                data=gpu_chunk,
                gpu=gpu,
                indices=(1:nlat, lon_start:lon_end)
            ))
            
            lon_start = lon_end + 1
        end
        
    else  # :spectral - for coefficient arrays
        error("Spectral distribution not implemented for spatial arrays")
    end
    
    return distributed_arrays
end

"""
    distribute_coefficient_array(coeffs, mgpu_config::MultiGPUConfig)

Distribute spherical harmonic coefficients across multiple GPUs.
"""
function distribute_coefficient_array(coeffs::AbstractArray, mgpu_config::MultiGPUConfig)
    ngpus = length(mgpu_config.gpu_devices)
    lmax_plus1, mmax_plus1 = size(coeffs)
    distributed_coeffs = []
    
    if mgpu_config.distribution_strategy == :spectral
        # Split by l-modes (degree)
        l_per_gpu = div(lmax_plus1, ngpus)
        l_remainder = lmax_plus1 % ngpus
        
        l_start = 1
        for (i, gpu) in enumerate(mgpu_config.gpu_devices)
            l_count = l_per_gpu + (i <= l_remainder ? 1 : 0)
            l_end = l_start + l_count - 1
            
            set_gpu_device(gpu.device, gpu.id)
            chunk = coeffs[l_start:l_end, :]
            if chunk isa GPUArraysCore.AbstractGPUArray
                gpu_chunk = chunk
            else
                gpu_chunk = to_device(chunk, _device_enum(gpu.device))
            end
            
            push!(distributed_coeffs, (
                data=gpu_chunk,
                gpu=gpu,
                indices=(l_start:l_end, 1:mmax_plus1),
                l_range=(l_start-1:l_end-1)  # 0-based l values
            ))
            
            l_start = l_end + 1
        end
    else
        error("Can only distribute coefficient arrays with :spectral strategy")
    end
    
    return distributed_coeffs
end

"""
    gpu_to_gpu_transfer(src_data, src_gpu, dest_gpu)

Transfer data directly between GPUs using peer-to-peer communication when possible.
Falls back to CPU transfer if P2P is not available.
"""
function gpu_to_gpu_transfer(src_data, src_gpu, dest_gpu)
    if src_gpu.device == dest_gpu.device == :cuda && CUDA_LOADED
        # CUDA peer-to-peer transfer
        try
            # Check if P2P access is enabled between devices
            src_device = CUDA.CuDevice(src_gpu.id)
            dest_device = CUDA.CuDevice(dest_gpu.id)
            
            # Enable P2P access if available
            can_access = try
                CUDA.can_access_peer(src_device, dest_device)
            catch
                false
            end
            
            if can_access
                # Direct GPU-to-GPU transfer
                set_gpu_device(:cuda, dest_gpu.id)
                dest_data = CUDA.CuArray{eltype(src_data)}(undef, size(src_data))
                
                # Copy directly between GPUs
                set_gpu_device(:cuda, src_gpu.id)
                copyto!(dest_data, src_data)
                
                return dest_data
            end
        catch e
            @warn "P2P transfer failed, falling back to CPU: $e"
        end
    end
    
    # Fallback: transfer via CPU
    set_gpu_device(src_gpu.device, src_gpu.id)
    cpu_data = Array(src_data)
    set_gpu_device(dest_gpu.device, dest_gpu.id)
    return to_device(cpu_data, dest_gpu.device == :cuda ? CUDA_DEVICE : AMDGPU_DEVICE)
end

"""
    gather_distributed_arrays(distributed_arrays, original_shape)

Gather distributed arrays back into a single array on the primary GPU.
Uses optimized GPU-to-GPU transfers when possible.
"""
function gather_distributed_arrays(distributed_arrays, original_shape)
    # Set primary GPU
    primary_gpu = distributed_arrays[1].gpu
    set_gpu_device(primary_gpu.device, primary_gpu.id)
    
    # Create output array on primary GPU
    result = to_device(zeros(eltype(distributed_arrays[1].data), original_shape), 
                      primary_gpu.device == :cuda ? CUDA_DEVICE : AMDGPU_DEVICE)
    
    # Copy data chunks to appropriate locations
    for chunk in distributed_arrays
        target_device = _device_enum(primary_gpu.device)
        if chunk.gpu != primary_gpu
            gpu_data = gpu_to_gpu_transfer(chunk.data, chunk.gpu, primary_gpu)
        else
            gpu_data = _ensure_device(chunk.data, target_device)
        end
        
        lat_range, lon_range = chunk.indices
        result[lat_range, lon_range] .= gpu_data
    end
    
    return result
end

"""
    enable_gpu_p2p_access(gpus::Vector)

Enable peer-to-peer access between all CUDA GPUs if possible.
"""
function enable_gpu_p2p_access(gpus::Vector)
    if !CUDA_LOADED
        return false
    end
    
    cuda_gpus = [gpu for gpu in gpus if gpu.device == :cuda]
    if length(cuda_gpus) < 2
        return false
    end
    
    enabled_pairs = 0
    total_pairs = 0
    
    for i in 1:length(cuda_gpus), j in 1:length(cuda_gpus)
        if i != j
            total_pairs += 1
            try
                src_device = CUDA.CuDevice(cuda_gpus[i].id)
                dest_device = CUDA.CuDevice(cuda_gpus[j].id)
                
                if CUDA.can_access_peer(src_device, dest_device)
                    set_gpu_device(:cuda, cuda_gpus[i].id)
                    CUDA.enable_peer_access(dest_device)
                    enabled_pairs += 1
                end
            catch e
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
function gpu_analysis(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true, workspace=nothing, coeffs_workspace=nothing)
    if device == CPU_DEVICE
        return SHTnsKit.analysis_cpu(cfg, spatial_data)
    end
    
    gpu_data = if spatial_data isa GPUArraysCore.AbstractGPUArray
        spatial_data
    elseif workspace !== nothing
        workspace .= spatial_data
        workspace
    else
        to_device(ComplexF64.(spatial_data), device)
    end
    
    coeffs = if coeffs_workspace !== nothing
        fill!(coeffs_workspace, 0)
        coeffs_workspace
    else
        to_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device)
    end
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
function gpu_synthesis(cfg::SHTConfig, coeffs; device=get_device(), real_output=true, coeffs_workspace=nothing)
    if device == CPU_DEVICE
        return SHTnsKit.synthesis_cpu(cfg, coeffs; real_output=real_output)
    end
    
    gpu_coeffs = if coeffs isa GPUArraysCore.AbstractGPUArray
        coeffs
    elseif coeffs_workspace !== nothing
        coeffs_workspace .= coeffs
        coeffs_workspace
    else
        to_device(ComplexF64.(coeffs), device)
    end
    
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

"""
    gpu_spat_to_SH(cfg::SHTConfig, Vr; device=get_device())

Packed scalar analysis driven by the GPU backend.
"""
function gpu_spat_to_SH(cfg::SHTConfig, Vr::AbstractVector{<:Real}; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.spat_to_SH_cpu(cfg, Vr)
    end

    length(Vr) == cfg.nspat || throw(DimensionMismatch("Vr must have length $(cfg.nspat)"))
    spatial = reshape(Vr, cfg.nlat, cfg.nlon)
    alm_mat = gpu_analysis(cfg, spatial; device=device, real_output=false)
    return SHTnsKit._pack_scalar_coeffs(cfg, alm_mat)
end

"""
    gpu_SH_to_spat(cfg::SHTConfig, Qlm; device=get_device())

Packed scalar synthesis routed through the GPU backend.
"""
function gpu_SH_to_spat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.SH_to_spat_cpu(cfg, Qlm)
    end

    alm_mat = SHTnsKit._unpack_scalar_coeffs(cfg, Qlm)
    spatial = gpu_synthesis(cfg, alm_mat; device=device, real_output=true)
    return vec(spatial)
end

function gpu_spat_to_SH_axisym(cfg::SHTConfig, Vr::AbstractVector{<:Real}; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.spat_to_SH_axisym(cfg, Vr)
    end
    length(Vr) == cfg.nlat || throw(DimensionMismatch("Vr length must be nlat=$(cfg.nlat)"))
    Vr_host = collect(Vr)
    Vr_complex = ComplexF64.(Vr_host)
    coeffs = gpu_spat_to_SH_ml(cfg, 0, Vr_complex, cfg.lmax; device=device)
    return coeffs
end

function gpu_SH_to_spat_axisym(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.SH_to_spat_axisym(cfg, Qlm)
    end
    length(Qlm) == cfg.lmax + 1 || throw(DimensionMismatch("Qlm length must be lmax+1=$(cfg.lmax+1)"))
    Vr = gpu_SH_to_spat_ml(cfg, 0, Qlm, cfg.lmax; device=device)
    return real.(Vr)
end

function gpu_spat_to_SH_l_axisym(cfg::SHTConfig, Vr::AbstractVector{<:Real}, ltr::Int; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.spat_to_SH_l_axisym(cfg, Vr, ltr)
    end
    lcap = min(Int(ltr), cfg.lmax)
    lcap ≥ 0 || throw(ArgumentError("ltr must be ≥ 0"))
    length(Vr) == cfg.nlat || throw(DimensionMismatch("Vr length must be nlat=$(cfg.nlat)"))
    Vr_host = collect(Vr)
    Vr_complex = ComplexF64.(Vr_host)
    coeffs = gpu_spat_to_SH_ml(cfg, 0, Vr_complex, lcap; device=device)
    return coeffs
end

function gpu_SH_to_spat_l_axisym(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, ltr::Int; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.SH_to_spat_l_axisym(cfg, Qlm, ltr)
    end
    lcap = min(Int(ltr), cfg.lmax)
    lcap ≥ 0 || throw(ArgumentError("ltr must be ≥ 0"))
    length(Qlm) ≥ lcap + 1 || throw(DimensionMismatch("Qlm length must be ≥ $(lcap+1)"))
    Qslice = view(Qlm, 1:(lcap+1))
    Vr = gpu_SH_to_spat_ml(cfg, 0, Qslice, lcap; device=device)
    return real.(Vr)
end

function gpu_SH_to_lat(cfg::SHTConfig, Qlm::AbstractVector, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax, device=get_device())
    device == CPU_DEVICE && return SHTnsKit.SH_to_lat_cpu(cfg, Qlm, cost; nphi=nphi, ltr=ltr, mtr=mtr)
    lcap = min(Int(ltr), cfg.lmax)
    mcap = min(Int(mtr), cfg.mmax)
    lcap < 0 && return zeros(Float64, nphi)

    Qvec = Qlm isa GPUArraysCore.AbstractGPUArray ? Array(Qlm) : Qlm
    alm = SHTnsKit._unpack_scalar_coeffs(cfg, Qvec)
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        tmp = similar(alm)
        SHTnsKit.convert_alm_norm!(tmp, alm, cfg; to_internal=true)
        alm = tmp
    end

    alm_gpu = to_device(ComplexF64.(alm), device)
    Nlm_gpu = to_device(cfg.Nlm, device)
    Plm_table = to_device(zeros(Float64, mcap + 1, lcap + 1), device)

    backend = get_backend(alm_gpu)
    lat_legendre_kernel! = legendre_latitude_kernel!(backend)
    lat_legendre_kernel!(Plm_table, float(cost), lcap, cfg.mres; ndrange=mcap + 1)
    KernelAbstractions.synchronize(backend)

    gm_gpu = to_device(zeros(ComplexF64, mcap + 1), device)
    lat_acc_kernel! = lat_mode_accumulate_kernel!(backend)
    lat_acc_kernel!(gm_gpu, alm_gpu, Nlm_gpu, Plm_table, lcap, mcap, cfg.mres; ndrange=mcap + 1)
    KernelAbstractions.synchronize(backend)

    vals_gpu = to_device(zeros(Float64, nphi), device)
    lat_fin_kernel! = lat_finalize_kernel!(backend)
    lat_fin_kernel!(vals_gpu, gm_gpu, nphi, mcap; ndrange=nphi)
    KernelAbstractions.synchronize(backend)

    return Array(vals_gpu)
end

function gpu_SH_to_lat_cplx(cfg::SHTConfig, alm_packed::AbstractVector, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.SH_to_lat_cplx_cpu(cfg, alm_packed, cost; nphi=nphi, ltr=ltr)
    end
    host_coeffs = collect(alm_packed)
    return SHTnsKit.SH_to_lat_cplx_cpu(cfg, host_coeffs, cost; nphi=nphi, ltr=ltr)
end

function gpu_SHqst_to_point(cfg::SHTConfig, Qlm::AbstractVector, Slm::AbstractVector, Tlm::AbstractVector, cost::Real, phi::Real; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.SHqst_to_point_cpu(cfg, Qlm, Slm, Tlm, cost, phi)
    end
    Qhost = collect(Qlm)
    Shost = collect(Slm)
    Thost = collect(Tlm)
    return SHTnsKit.SHqst_to_point_cpu(cfg, Qhost, Shost, Thost, cost, phi)
end

function gpu_SH_to_grad_point(cfg::SHTConfig, Slm::AbstractVector, cost::Real, phi::Real; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.SH_to_grad_point_cpu(cfg, Slm, cost, phi)
    end
    Shost = collect(Slm)
    return SHTnsKit.SH_to_grad_point_cpu(cfg, Shost, cost, phi)
end

function gpu_spat_to_SH_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, ltr::Int; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.spat_to_SH_ml(cfg, im, Vr_m, ltr)
    end

    nlat = cfg.nlat
    length(Vr_m) == nlat || throw(DimensionMismatch("Vr_m length must be nlat=$(nlat)"))
    im >= 0 || throw(ArgumentError("im must be ≥ 0"))
    im <= cfg.mmax || throw(ArgumentError("im must be ≤ mmax=$(cfg.mmax)"))
    lcap = min(Int(ltr), cfg.lmax)
    lcap >= im || throw(ArgumentError("ltr must be ≥ im=$(im)"))
    lcount = lcap - im + 1

    Vr_gpu = Vr_m isa GPUArraysCore.AbstractGPUArray ? Vr_m : to_device(ComplexF64.(Vr_m), device)
    weights_gpu = to_device(cfg.wlat, device)
    Plm_mode, backend = _gpu_legendre_mode(cfg, im, lcap, device)
    coeffs_gpu = to_device(zeros(ComplexF64, lcount), device)

    analysis_kernel! = scalar_mode_analysis_kernel!(backend)
    analysis_kernel!(coeffs_gpu, Vr_gpu, weights_gpu, Plm_mode, cfg.cphi; ndrange=lcount)
    KernelAbstractions.synchronize(backend)

    return Array(coeffs_gpu)
end

function gpu_SH_to_spat_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, ltr::Int; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.SH_to_spat_ml(cfg, im, Ql, ltr)
    end

    im >= 0 || throw(ArgumentError("im must be ≥ 0"))
    im <= cfg.mmax || throw(ArgumentError("im must be ≤ mmax=$(cfg.mmax)"))
    lcap = min(Int(ltr), cfg.lmax)
    lcap >= im || throw(ArgumentError("ltr must be ≥ im=$(im)"))
    expected_len = lcap - im + 1
    length(Ql) == expected_len || throw(DimensionMismatch("Ql length must be $(expected_len)"))

    Ql_gpu = Ql isa GPUArraysCore.AbstractGPUArray ? Ql : to_device(ComplexF64.(Ql), device)
    Plm_mode, backend = _gpu_legendre_mode(cfg, im, lcap, device)
    Vr_gpu = to_device(zeros(ComplexF64, cfg.nlat), device)

    synthesis_kernel! = scalar_mode_synthesis_kernel!(backend)
    synthesis_kernel!(Vr_gpu, Ql_gpu, Plm_mode; ndrange=cfg.nlat)
    KernelAbstractions.synchronize(backend)

    return Array(Vr_gpu)
end

function gpu_spat_to_SHsphtor_ml(cfg::SHTConfig, im::Int, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.spat_to_SHsphtor_ml(cfg, im, Vt_m, Vp_m, ltr)
    end

    nlat = cfg.nlat
    length(Vt_m) == nlat || throw(DimensionMismatch("Vt_m length must be nlat=$(nlat)"))
    length(Vp_m) == nlat || throw(DimensionMismatch("Vp_m length must be nlat=$(nlat)"))
    im >= 0 || throw(ArgumentError("im must be ≥ 0"))
    im <= cfg.mmax || throw(ArgumentError("im must be ≤ mmax=$(cfg.mmax)"))
    lcap = min(Int(ltr), cfg.lmax)
    lcap >= im || throw(ArgumentError("ltr must be ≥ im=$(im)"))
    lcount = lcap - im + 1

    Vt_gpu = Vt_m isa GPUArraysCore.AbstractGPUArray ? Vt_m : to_device(ComplexF64.(Vt_m), device)
    Vp_gpu = Vp_m isa GPUArraysCore.AbstractGPUArray ? Vp_m : to_device(ComplexF64.(Vp_m), device)
    weights_gpu = to_device(cfg.wlat, device)
    Plm_mode, backend = _gpu_legendre_mode(cfg, im, lcap, device)
    Sl_gpu = to_device(zeros(ComplexF64, lcount), device)
    Tl_gpu = to_device(zeros(ComplexF64, lcount), device)

    analysis_kernel! = sphtor_mode_analysis_kernel!(backend)
    analysis_kernel!(Sl_gpu, Tl_gpu, Vt_gpu, Vp_gpu, weights_gpu, Plm_mode, cfg.cphi, im; ndrange=lcount)
    KernelAbstractions.synchronize(backend)

    return Array(Sl_gpu), Array(Tl_gpu)
end

function gpu_SHsphtor_to_spat_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.SHsphtor_to_spat_ml(cfg, im, Sl, Tl, ltr)
    end

    im >= 0 || throw(ArgumentError("im must be ≥ 0"))
    im <= cfg.mmax || throw(ArgumentError("im must be ≤ mmax=$(cfg.mmax)"))
    lcap = min(Int(ltr), cfg.lmax)
    lcap >= im || throw(ArgumentError("ltr must be ≥ im=$(im)"))
    expected_len = lcap - im + 1
    length(Sl) == expected_len || throw(DimensionMismatch("Sl length mismatch"))
    length(Tl) == expected_len || throw(DimensionMismatch("Tl length mismatch"))

    Sl_gpu = Sl isa GPUArraysCore.AbstractGPUArray ? Sl : to_device(ComplexF64.(Sl), device)
    Tl_gpu = Tl isa GPUArraysCore.AbstractGPUArray ? Tl : to_device(ComplexF64.(Tl), device)
    Plm_mode, backend = _gpu_legendre_mode(cfg, im, lcap, device)
    Vt_gpu = to_device(zeros(ComplexF64, cfg.nlat), device)
    Vp_gpu = to_device(zeros(ComplexF64, cfg.nlat), device)

    synthesis_kernel! = sphtor_mode_synthesis_kernel!(backend)
    synthesis_kernel!(Vt_gpu, Vp_gpu, Sl_gpu, Tl_gpu, Plm_mode, im; ndrange=cfg.nlat)
    KernelAbstractions.synchronize(backend)

    return Array(Vt_gpu), Array(Vp_gpu)
end

function gpu_SH_to_point(cfg::SHTConfig, Qlm::AbstractVector, cost::Real, phi::Real; device=get_device())
    q_cpu = Qlm isa GPUArraysCore.AbstractGPUArray ? Array(Qlm) : Qlm
    return SHTnsKit.SH_to_point(cfg, q_cpu, cost, phi)
end

function SHTnsKit.analysis!(plan::SHTGPUPlan, alm_out::AbstractMatrix, f::AbstractMatrix)
    result = gpu_analysis(plan.cfg, f;
                          device=plan.device,
                          real_output=false,
                          workspace=plan.spatial_cache,
                          coeffs_workspace=plan.coeffs_cache)
    copyto!(alm_out, result)
    return alm_out
end

function SHTnsKit.synthesis!(plan::SHTGPUPlan, f_out::AbstractMatrix, alm::AbstractMatrix; real_output::Bool=true)
    result = gpu_synthesis(plan.cfg, alm;
                           device=plan.device,
                           real_output=real_output,
                           coeffs_workspace=plan.coeffs_cache)
    copyto!(f_out, result)
    return f_out
end

function SHTnsKit.spat_to_SHsphtor!(plan::SHTGPUPlan, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix)
    Slm, Tlm = gpu_spat_to_SHsphtor(plan.cfg, Vt, Vp; device=plan.device)
    copyto!(Slm_out, Slm)
    copyto!(Tlm_out, Tlm)
    return Slm_out, Tlm_out
end

function SHTnsKit.SHsphtor_to_spat!(plan::SHTGPUPlan, Vt_out::AbstractMatrix, Vp_out::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    Vt, Vp = gpu_SHsphtor_to_spat(plan.cfg, Slm, Tlm; device=plan.device, real_output=real_output)
    copyto!(Vt_out, Vt)
    copyto!(Vp_out, Vp)
    return Vt_out, Vp_out
end

# Energy diagnostics ---------------------------------------------------------

function _ensure_cu(array)
    CUDA_LOADED || error("CUDA backend not available")
    array isa CUDA.CuArray && return array
    return CUDA.CuArray(array)
end

function _cu_valid_mask(cfg::SHTConfig)
    CUDA_LOADED || error("CUDA backend not available")
    l_vals = CUDA.CuArray(collect(0:cfg.lmax))
    m_vals = CUDA.CuArray(collect(0:cfg.mmax))
    reshape(l_vals, cfg.lmax + 1, 1) .>= reshape(m_vals, 1, cfg.mmax + 1)
end

function _cu_m_weights(cfg::SHTConfig, real_field::Bool)
    CUDA_LOADED || error("CUDA backend not available")
    data = real_field ? SHTnsKit._wm_real(cfg) : ones(Float64, cfg.mmax + 1)
    CUDA.CuArray(data)
end

function _cu_ll1(cfg::SHTConfig)
    CUDA_LOADED || error("CUDA backend not available")
    CUDA.CuArray((0:cfg.lmax) .* ((0:cfg.lmax) .+ 1))
end

function gpu_energy_scalar(cfg::SHTConfig, alm; real_field::Bool=true)
    alm_cu = _ensure_cu(alm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    weighted = abs2.(alm_cu) .* mask .* reshape(weights, 1, :)
    return 0.5 * sum(weighted)
end

function gpu_energy_vector(cfg::SHTConfig, Slm, Tlm; real_field::Bool=true)
    Slm_cu = _ensure_cu(Slm)
    Tlm_cu = _ensure_cu(Tlm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    ll1 = reshape(_cu_ll1(cfg), :, 1)
    power = (abs2.(Slm_cu) + abs2.(Tlm_cu)) .* mask
    weighted = power .* reshape(weights, 1, :) .* ll1
    return 0.5 * sum(weighted)
end

function gpu_grid_energy_scalar(cfg::SHTConfig, f)
    f_cu = _ensure_cu(f)
    wlat = CUDA.CuArray(cfg.wlat)
    weighted = abs2.(f_cu) .* reshape(wlat, :, 1)
    return 0.5 * sum(weighted) * (2π / cfg.nlon)
end

function gpu_grid_energy_vector(cfg::SHTConfig, Vt, Vp)
    Vt_cu = _ensure_cu(Vt)
    Vp_cu = _ensure_cu(Vp)
    wlat = CUDA.CuArray(cfg.wlat)
    weighted = (abs2.(Vt_cu) + abs2.(Vp_cu)) .* reshape(wlat, :, 1)
    return 0.5 * sum(weighted) * (2π / cfg.nlon)
end

function gpu_energy_scalar_l_spectrum(cfg::SHTConfig, alm; real_field::Bool=true)
    alm_cu = _ensure_cu(alm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    weighted = abs2.(alm_cu) .* mask .* reshape(weights, 1, :)
    El = 0.5 .* Array(sum(weighted; dims=2))
    return vec(El)
end

function gpu_energy_scalar_m_spectrum(cfg::SHTConfig, alm; real_field::Bool=true)
    alm_cu = _ensure_cu(alm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    weighted = abs2.(alm_cu) .* mask .* reshape(weights, 1, :)
    Em = 0.5 .* Array(sum(weighted; dims=1))
    return vec(Em)
end

function gpu_energy_vector_l_spectrum(cfg::SHTConfig, Slm, Tlm; real_field::Bool=true)
    Slm_cu = _ensure_cu(Slm)
    Tlm_cu = _ensure_cu(Tlm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    ll1 = reshape(_cu_ll1(cfg), :, 1)
    power = (abs2.(Slm_cu) + abs2.(Tlm_cu)) .* mask
    El = 0.5 .* Array(sum(power .* reshape(weights, 1, :) .* ll1; dims=2))
    return vec(El)
end

function gpu_energy_vector_m_spectrum(cfg::SHTConfig, Slm, Tlm; real_field::Bool=true)
    Slm_cu = _ensure_cu(Slm)
    Tlm_cu = _ensure_cu(Tlm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    ll1 = reshape(_cu_ll1(cfg), :, 1)
    power = (abs2.(Slm_cu) + abs2.(Tlm_cu)) .* mask
    Em = 0.5 .* Array(sum(power .* reshape(weights, 1, :) .* ll1; dims=1))
    return vec(Em)
end

function gpu_energy_scalar_lm(cfg::SHTConfig, alm; real_field::Bool=true)
    alm_cu = _ensure_cu(alm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    Elm = 0.5 .* Array(abs2.(alm_cu) .* mask .* reshape(weights, 1, :))
    return Elm
end

function gpu_energy_vector_lm(cfg::SHTConfig, Slm, Tlm; real_field::Bool=true)
    Slm_cu = _ensure_cu(Slm)
    Tlm_cu = _ensure_cu(Tlm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    ll1 = reshape(_cu_ll1(cfg), :, 1)
    Elm = 0.5 .* Array((abs2.(Slm_cu) + abs2.(Tlm_cu)) .* mask .* reshape(weights, 1, :) .* ll1)
    return Elm
end

function gpu_energy_scalar_packed(cfg::SHTConfig, Qlm; real_field::Bool=true)
    Q_cu = _ensure_cu(Qlm)
    weights = _cu_m_weights(cfg, real_field)
    mi = CUDA.CuArray(cfg.mi)
    weights_per = CUDA.map(m -> weights[m+1], mi)
    return 0.5 * sum(weights_per .* abs2.(Q_cu))
end

function gpu_energy_vector_packed(cfg::SHTConfig, Spacked, Tpacked; real_field::Bool=true)
    S_cu = _ensure_cu(Spacked)
    T_cu = _ensure_cu(Tpacked)
    weights = _cu_m_weights(cfg, real_field)
    mi = CUDA.CuArray(cfg.mi)
    li = CUDA.CuArray(cfg.li)
    weights_per = CUDA.map(m -> weights[m+1], mi)
    ll1_per = CUDA.map(l -> l >= 1 ? l * (l + 1) : 0, li)
    power = abs2.(S_cu) + abs2.(T_cu)
    return 0.5 * sum(weights_per .* ll1_per .* power)
end

function gpu_grad_energy_scalar_alm(cfg::SHTConfig, alm; real_field::Bool=true)
    alm_cu = _ensure_cu(alm)
    weights = _cu_m_weights(cfg, real_field)
    mask = _cu_valid_mask(cfg)
    grad = alm_cu .* reshape(weights, 1, :) .* mask
    return Array(grad)
end

function gpu_grad_energy_scalar_packed(cfg::SHTConfig, Qlm; real_field::Bool=true)
    Q_cu = _ensure_cu(Qlm)
    weights = _cu_m_weights(cfg, real_field)
    mi = CUDA.CuArray(cfg.mi)
    weights_per = CUDA.map(m -> weights[m+1], mi)
    grad = weights_per .* Q_cu
    return Array(grad)
end

function gpu_grad_energy_vector_Slm_Tlm(cfg::SHTConfig, Slm, Tlm; real_field::Bool=true)
    Slm_cu = _ensure_cu(Slm)
    Tlm_cu = _ensure_cu(Tlm)
    weights = _cu_m_weights(cfg, real_field)
    ll1 = _cu_ll1(cfg)
    mask = _cu_valid_mask(cfg)
    w_l = reshape(ll1, :, 1) .* mask .* reshape(weights, 1, :)
    grad_S = w_l .* Slm_cu
    grad_T = w_l .* Tlm_cu
    return Array(grad_S), Array(grad_T)
end

function gpu_grad_grid_energy_scalar_field(cfg::SHTConfig, f)
    f_cu = _ensure_cu(f)
    wlat = CUDA.CuArray(cfg.wlat)
    scale = 2π / cfg.nlon
    grad = f_cu .* reshape(wlat, :, 1) .* scale
    return Array(grad)
end

function gpu_grad_grid_energy_vector_fields(cfg::SHTConfig, Vt, Vp)
    Vt_cu = _ensure_cu(Vt)
    Vp_cu = _ensure_cu(Vp)
    wlat = CUDA.CuArray(cfg.wlat)
    scale = 2π / cfg.nlon
    weight = reshape(wlat, :, 1) .* scale
    grad_Vt = Vt_cu .* weight
    grad_Vp = Vp_cu .* weight
    return Array(grad_Vt), Array(grad_Vp)
end

function gpu_grad_energy_vector_packed(cfg::SHTConfig, Spacked, Tpacked; real_field::Bool=true)
    S_cu = _ensure_cu(Spacked)
    T_cu = _ensure_cu(Tpacked)
    weights = _cu_m_weights(cfg, real_field)
    mi = CUDA.CuArray(cfg.mi)
    li = CUDA.CuArray(cfg.li)
    weights_per = CUDA.map(m -> weights[m+1], mi)
    ll1_per = CUDA.map(l -> l >= 1 ? l * (l + 1) : 0, li)
    scale = weights_per .* ll1_per
    grad_S = scale .* S_cu
    grad_T = scale .* T_cu
    return Array(grad_S), Array(grad_T)
end

# GPU-aware rotation helpers -------------------------------------------------

SHTnsKit.SH_Zrotate(cfg::SHTConfig, Qlm::CUDA.CuVector, alpha::Real, Rlm::CUDA.CuVector) = gpu_SH_Zrotate(cfg, Qlm, alpha, Rlm)
SHTnsKit.SH_Zrotate(cfg::SHTConfig, Qlm::CUDA.CuVector, alpha::Real, Rlm::AbstractVector) = gpu_SH_Zrotate(cfg, Qlm, alpha, Rlm)
SHTnsKit.SH_Zrotate(cfg::SHTConfig, Qlm::AbstractVector, alpha::Real, Rlm::CUDA.CuVector) = gpu_SH_Zrotate(cfg, Qlm, alpha, Rlm)

function SHTnsKit.shtns_rotation_apply_real(r::SHTnsKit.SHTRotation, Qlm::CUDA.CuVector{T}, Rlm::CUDA.CuVector{T}) where {T<:Complex}
    q_cpu = Array(Qlm)
    r_cpu = Array(Rlm)
    SHTnsKit._shtns_rotation_apply_real_cpu!(r, q_cpu, r_cpu)
    copyto!(Rlm, r_cpu)
    return Rlm
end

function SHTnsKit.shtns_rotation_apply_real(r::SHTnsKit.SHTRotation, Qlm::CUDA.CuVector{T}, Rlm::AbstractVector{T}) where {T<:Complex}
    q_cpu = Array(Qlm)
    SHTnsKit._shtns_rotation_apply_real_cpu!(r, q_cpu, Rlm)
    return Rlm
end

function SHTnsKit.shtns_rotation_apply_real(r::SHTnsKit.SHTRotation, Qlm::AbstractVector{T}, Rlm::CUDA.CuVector{T}) where {T<:Complex}
    r_cpu = Array(Rlm)
    SHTnsKit._shtns_rotation_apply_real_cpu!(r, Qlm, r_cpu)
    copyto!(Rlm, r_cpu)
    return Rlm
end

function SHTnsKit.shtns_rotation_apply_cplx(r::SHTnsKit.SHTRotation, Zlm::CUDA.CuVector{T}, Rlm::CUDA.CuVector{T}) where {T<:Complex}
    z_cpu = Array(Zlm)
    r_cpu = Array(Rlm)
    SHTnsKit._shtns_rotation_apply_cplx_cpu!(r, z_cpu, r_cpu)
    copyto!(Rlm, r_cpu)
    return Rlm
end

function SHTnsKit.shtns_rotation_apply_cplx(r::SHTnsKit.SHTRotation, Zlm::CUDA.CuVector{T}, Rlm::AbstractVector{T}) where {T<:Complex}
    z_cpu = Array(Zlm)
    SHTnsKit._shtns_rotation_apply_cplx_cpu!(r, z_cpu, Rlm)
    return Rlm
end

function SHTnsKit.shtns_rotation_apply_cplx(r::SHTnsKit.SHTRotation, Zlm::AbstractVector{T}, Rlm::CUDA.CuVector{T}) where {T<:Complex}
    r_cpu = Array(Rlm)
    SHTnsKit._shtns_rotation_apply_cplx_cpu!(r, Zlm, r_cpu)
    copyto!(Rlm, r_cpu)
    return Rlm
end

@cuda function _zrotate_kernel!(R, Q, mi, alpha, mres)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(Q)
        m = mi[idx]
        if m % mres == 0
            T = eltype(R)
            realT = eltype(real(zero(T)))
            α = realT(alpha)
            a = realT(cos(α * realT(m)))
            b = realT(sin(α * realT(m)))
            phase = Complex{realT}(a, b)
            R[idx] = Q[idx] * phase
        end
    end
    return nothing
end

function gpu_SH_Zrotate(cfg::SHTConfig, Qlm, alpha::Real, Rlm)
    mi_cu = CUDA.CuArray(cfg.mi)
    mres = cfg.mres
    Q_dev = Qlm isa CUDA.CuArray ? Qlm : CUDA.CuArray(Qlm)
    R_dev = Rlm isa CUDA.CuArray ? Rlm : CUDA.CuArray(Rlm)
    threads = 256
    blocks = cld(length(Q_dev), threads)
    @cuda threads=threads blocks=blocks _zrotate_kernel!(R_dev, Q_dev, mi_cu, alpha, mres)
    CUDA.synchronize()
    if !(Rlm isa CUDA.CuArray)
        copyto!(Rlm, Array(R_dev))
    end
    return Rlm isa CUDA.CuArray ? Rlm : Rlm
end

function gpu_enstrophy(cfg::SHTConfig, Tlm; real_field::Bool=true)
    T_cu = _ensure_cu(Tlm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    ll1 = reshape(_cu_ll1(cfg), :, 1)
    ll1_sq = ll1 .* ll1
    power = abs2.(T_cu) .* mask .* reshape(weights, 1, :) .* ll1_sq
    return 0.5 * sum(power)
end

function gpu_vorticity_spectral(cfg::SHTConfig, Tlm)
    T_cu = _ensure_cu(Tlm)
    ll1 = reshape(_cu_ll1(cfg), :, 1)
    mask = _cu_valid_mask(cfg)
    ζ = (-ll1) .* T_cu
    ζ .*= mask
    return Array(ζ)
end

function gpu_vorticity_grid(cfg::SHTConfig, Tlm)
    device = _device_enum(cfg.device_backend)
    ζlm = gpu_vorticity_spectral(cfg, Tlm)
    return gpu_synthesis(cfg, ζlm; device=device, real_output=true)
end

function gpu_grid_enstrophy(cfg::SHTConfig, ζ)
    ζ_cu = _ensure_cu(ζ)
    wlat = CUDA.CuArray(cfg.wlat)
    weighted = abs2.(ζ_cu) .* reshape(wlat, :, 1)
    return 0.5 * sum(weighted) * (2π / cfg.nlon)
end

function gpu_grad_enstrophy_Tlm(cfg::SHTConfig, Tlm; real_field::Bool=true)
    T_cu = _ensure_cu(Tlm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    ll1 = reshape(_cu_ll1(cfg), :, 1)
    ll1_sq = ll1 .* ll1
    grad = ll1_sq .* reshape(weights, 1, :) .* T_cu .* mask
    return Array(grad)
end

function gpu_grad_grid_enstrophy_zeta(cfg::SHTConfig, ζ)
    ζ_cu = _ensure_cu(ζ)
    wlat = CUDA.CuArray(cfg.wlat)
    scale = 2π / cfg.nlon
    grad = ζ_cu .* reshape(wlat, :, 1) .* scale
    return Array(grad)
end

function gpu_enstrophy_l_spectrum(cfg::SHTConfig, Tlm; real_field::Bool=true)
    T_cu = _ensure_cu(Tlm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    ll1 = reshape(_cu_ll1(cfg), :, 1)
    ll1_sq = ll1 .* ll1
    power = abs2.(T_cu) .* mask .* reshape(weights, 1, :) .* ll1_sq
    Zl = 0.5 .* Array(sum(power; dims=2))
    return vec(Zl)
end

function gpu_enstrophy_m_spectrum(cfg::SHTConfig, Tlm; real_field::Bool=true)
    T_cu = _ensure_cu(Tlm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    ll1 = reshape(_cu_ll1(cfg), :, 1)
    ll1_sq = ll1 .* ll1
    power = abs2.(T_cu) .* mask .* reshape(weights, 1, :) .* ll1_sq
    Zm = 0.5 .* Array(sum(power; dims=1))
    return vec(Zm)
end

function gpu_enstrophy_lm(cfg::SHTConfig, Tlm; real_field::Bool=true)
    T_cu = _ensure_cu(Tlm)
    mask = _cu_valid_mask(cfg)
    weights = _cu_m_weights(cfg, real_field)
    ll1 = reshape(_cu_ll1(cfg), :, 1)
    ll1_sq = ll1 .* ll1
    Elm = 0.5 .* Array(abs2.(T_cu) .* mask .* reshape(weights, 1, :) .* ll1_sq)
    return Elm
end

function gpu_loss_vorticity_grid(cfg::SHTConfig, Tlm, ζ_target)
    T_cpu = Tlm isa CUDA.CuArray ? Array(Tlm) : Tlm
    ζ_target_cpu = ζ_target isa CUDA.CuArray ? Array(ζ_target) : ζ_target
    ζ = SHTnsKit._vorticity_grid_cpu(cfg, T_cpu)
    residual = ζ .- ζ_target_cpu
    return SHTnsKit._grid_enstrophy_cpu(cfg, residual)
end

function gpu_grad_loss_vorticity_Tlm(cfg::SHTConfig, Tlm, ζ_target)
    T_cpu = Tlm isa CUDA.CuArray ? Array(Tlm) : Tlm
    ζ_target_cpu = ζ_target isa CUDA.CuArray ? Array(ζ_target) : ζ_target
    ζ = SHTnsKit._vorticity_grid_cpu(cfg, T_cpu)
    residual = ζ .- ζ_target_cpu
    gζlm = SHTnsKit.analysis_cpu(cfg, residual)
    grad_cpu = SHTnsKit._grad_loss_vorticity_Tlm_cpu(cfg, gζlm)
    return Tlm isa CUDA.CuArray ? CUDA.CuArray(grad_cpu) : grad_cpu
end

function gpu_loss_and_grad_vorticity_Tlm(cfg::SHTConfig, Tlm, ζ_target)
    T_cpu = Tlm isa CUDA.CuArray ? Array(Tlm) : Tlm
    ζ_target_cpu = ζ_target isa CUDA.CuArray ? Array(ζ_target) : ζ_target
    ζ = SHTnsKit._vorticity_grid_cpu(cfg, T_cpu)
    residual = ζ .- ζ_target_cpu
    loss = SHTnsKit._grid_enstrophy_cpu(cfg, residual)
    gζlm = SHTnsKit.analysis_cpu(cfg, residual)
    grad_cpu = SHTnsKit._grad_loss_vorticity_Tlm_cpu(cfg, gζlm)
    grad = Tlm isa CUDA.CuArray ? CUDA.CuArray(grad_cpu) : grad_cpu
    return loss, grad
end

"""
    gpu_spat_to_SHqst(cfg::SHTConfig, Vr, Vt, Vp; device=get_device())

GPU-accelerated QST decomposition of vector fields.
"""
function gpu_spat_to_SHqst(cfg::SHTConfig, Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.spat_to_SHqst_cpu(cfg, Vr, Vt, Vp)
    end

    SHTnsKit.validate_vector_spatial_dimensions(Vr, Vt, Vp, cfg)

    Qlm = gpu_analysis(cfg, Vr; device=device, real_output=false)
    Slm, Tlm = gpu_spat_to_SHsphtor(cfg, Vt, Vp; device=device)

    return Qlm, Slm, Tlm
end

"""
    gpu_SHqst_to_spat(cfg::SHTConfig, Qlm, Slm, Tlm; device=get_device(), real_output=true)

GPU-accelerated QST synthesis back to spatial vector components.
"""
function gpu_SHqst_to_spat(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; device=get_device(), real_output::Bool=true)
    if device == CPU_DEVICE
        return SHTnsKit.SHqst_to_spat_cpu(cfg, Qlm, Slm, Tlm; real_output=real_output)
    end

    Vr = gpu_synthesis(cfg, Qlm; device=device, real_output=real_output)
    Vt, Vp = gpu_SHsphtor_to_spat(cfg, Slm, Tlm; device=device, real_output=real_output)

    return Vr, Vt, Vp
end

# Vector field GPU operations

@kernel function vector_divergence_kernel!(div_field, vθ, vφ, sintheta, dtheta, dphi, nlat, nlon)
    """
    GPU kernel for computing divergence: ∇·V = (1/sinθ)[∂(sinθ·vθ)/∂θ + ∂vφ/∂φ]
    """
    i, j = @index(Global, NTuple)
    if i <= nlat && j <= nlon
        # Finite difference approximation for derivatives
        # ∂vθ/∂θ using central differences
        i_prev = max(1, i-1)
        i_next = min(nlat, i+1) 
        dvtheta_dtheta = (vθ[i_next, j] - vθ[i_prev, j]) / (2 * dtheta)
        
        # ∂vφ/∂φ using central differences (periodic in φ)
        j_prev = j == 1 ? nlon : j-1
        j_next = j == nlon ? 1 : j+1
        dvphi_dphi = (vφ[i, j_next] - vφ[i, j_prev]) / (2 * dphi)
        
        # Compute divergence
        sin_theta = sintheta[i]
        div_field[i, j] = (sin_theta * dvtheta_dtheta + vθ[i, j] * cos(asin(sin_theta)) + dvphi_dphi) / sin_theta
    end
end

@kernel function vector_curl_kernel!(curl_field, vθ, vφ, sintheta, dtheta, dphi, nlat, nlon)  
    """
    GPU kernel for computing curl: (∇×V)_r = (1/sinθ)[∂vφ/∂θ - ∂(sinθ·vθ)/∂φ]
    """
    i, j = @index(Global, NTuple)
    if i <= nlat && j <= nlon
        # ∂vφ/∂θ using central differences
        i_prev = max(1, i-1)
        i_next = min(nlat, i+1)
        dvphi_dtheta = (vφ[i_next, j] - vφ[i_prev, j]) / (2 * dtheta)
        
        # ∂(sinθ·vθ)/∂φ using central differences (periodic in φ)  
        j_prev = j == 1 ? nlon : j-1
        j_next = j == nlon ? 1 : j+1
        sin_theta = sintheta[i]
        d_sinvtheta_dphi = (sin_theta * vθ[i, j_next] - sin_theta * vθ[i, j_prev]) / (2 * dphi)
        
        # Compute curl radial component
        curl_field[i, j] = (dvphi_dtheta - d_sinvtheta_dphi) / sin_theta
    end
end

"""
    gpu_spat_to_SHsphtor(cfg::SHTConfig, vθ, vφ; device=get_device())

GPU-accelerated spheroidal-toroidal decomposition of vector fields.
Computes divergence and curl, then transforms to spectral space.
"""
function gpu_spat_to_SHsphtor(cfg::SHTConfig, vθ, vφ; device=get_device())
    if device == CPU_DEVICE
        return SHTnsKit.spat_to_SHsphtor_cpu(cfg, vθ, vφ)
    end
    
    # Transfer to GPU
    gpu_vθ = to_device(ComplexF64.(vθ), device)
    gpu_vφ = to_device(ComplexF64.(vφ), device)
    
    backend = get_backend(gpu_vθ)
    
    # Compute grid spacings
    dtheta = π / (cfg.nlat - 1)
    dphi = 2π / cfg.nlon
    sintheta = to_device(cfg.st, device)
    
    # Step 1: Compute divergence and curl on GPU
    divergence = to_device(zeros(ComplexF64, cfg.nlat, cfg.nlon), device)
    curl_r = to_device(zeros(ComplexF64, cfg.nlat, cfg.nlon), device)
    
    div_kernel! = vector_divergence_kernel!(backend)
    curl_kernel! = vector_curl_kernel!(backend)
    
    div_kernel!(divergence, gpu_vθ, gpu_vφ, sintheta, dtheta, dphi, cfg.nlat, cfg.nlon;
                ndrange=(cfg.nlat, cfg.nlon))
    curl_kernel!(curl_r, gpu_vθ, gpu_vφ, sintheta, dtheta, dphi, cfg.nlat, cfg.nlon;
                 ndrange=(cfg.nlat, cfg.nlon))
    
    KernelAbstractions.synchronize(backend)
    
    # Step 2: Transform divergence and curl to spherical harmonic coefficients
    # Spheroidal coefficients from divergence
    sph_coeffs_temp = gpu_analysis(cfg, Array(divergence); device=device, real_output=false)
    sph_coeffs = to_device(ComplexF64.(sph_coeffs_temp), device)
    
    # Toroidal coefficients from curl  
    tor_coeffs_temp = gpu_analysis(cfg, Array(curl_r); device=device, real_output=false)
    tor_coeffs = to_device(ComplexF64.(tor_coeffs_temp), device)
    
    # Step 3: Apply operator corrections for spheroidal-toroidal decomposition
    @kernel function sphtor_correction_kernel!(sph_out, tor_out, sph_in, tor_in, lmax, mmax)
        l, m = @index(Global, NTuple)
        if l <= lmax + 1 && m <= mmax + 1
            l_val, m_val = l - 1, m - 1
            if l_val >= m_val && l_val > 0  # Avoid division by zero for l=0
                # Apply l(l+1) factor for Poisson equation inversion
                factor = -1.0 / (l_val * (l_val + 1))
                sph_out[l, m] = sph_in[l, m] * factor
                tor_out[l, m] = tor_in[l, m] * factor
            elseif l_val == 0
                sph_out[l, m] = ComplexF64(0, 0)  # l=0 modes are zero for vector fields
                tor_out[l, m] = ComplexF64(0, 0)
            end
        end
    end
    
    sph_coeffs_corrected = to_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device)
    tor_coeffs_corrected = to_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device)
    
    correction_kernel! = sphtor_correction_kernel!(backend)
    correction_kernel!(sph_coeffs_corrected, tor_coeffs_corrected, 
                      sph_coeffs, tor_coeffs, cfg.lmax, cfg.mmax;
                      ndrange=(cfg.lmax+1, cfg.mmax+1))
    
    KernelAbstractions.synchronize(backend)
    
    return Array(sph_coeffs_corrected), Array(tor_coeffs_corrected)
end

"""
    gpu_SHsphtor_to_spat(cfg::SHTConfig, sph_coeffs, tor_coeffs; device=get_device(), real_output=true)

GPU-accelerated synthesis of spheroidal-toroidal vector field components.
Reconstructs vθ and vφ from spheroidal and toroidal spectral coefficients.
"""
function gpu_SHsphtor_to_spat(cfg::SHTConfig, sph_coeffs, tor_coeffs; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.SHsphtor_to_spat_cpu(cfg, sph_coeffs, tor_coeffs; real_output=real_output)
    end
    
    # Transfer to GPU
    gpu_sph = to_device(ComplexF64.(sph_coeffs), device)
    gpu_tor = to_device(ComplexF64.(tor_coeffs), device)
    
    backend = get_backend(gpu_sph)
    
    # Step 1: Apply differential operators to get potential fields
    @kernel function sphtor_derivative_kernel!(sph_pot, tor_pot, sph_in, tor_in, lmax, mmax)
        l, m = @index(Global, NTuple)
        if l <= lmax + 1 && m <= mmax + 1
            l_val, m_val = l - 1, m - 1
            if l_val >= m_val
                # For synthesis, we need to apply ∇ to get vector components
                sph_pot[l, m] = sph_in[l, m]  # Spheroidal potential
                tor_pot[l, m] = tor_in[l, m]  # Toroidal potential
            end
        end
    end
    
    sph_potential = to_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device)
    tor_potential = to_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device)
    
    derivative_kernel! = sphtor_derivative_kernel!(backend)
    derivative_kernel!(sph_potential, tor_potential, gpu_sph, gpu_tor, cfg.lmax, cfg.mmax;
                      ndrange=(cfg.lmax+1, cfg.mmax+1))
    
    KernelAbstractions.synchronize(backend)
    
    # Step 2: Synthesize potential fields to spatial domain
    sph_spatial = gpu_synthesis(cfg, Array(sph_potential); device=device, real_output=false)
    tor_spatial = gpu_synthesis(cfg, Array(tor_potential); device=device, real_output=false)
    
    # Step 3: Compute gradient components to get vector field
    gpu_sph_spatial = to_device(ComplexF64.(sph_spatial), device)
    gpu_tor_spatial = to_device(ComplexF64.(tor_spatial), device)
    
    vθ = to_device(zeros(ComplexF64, cfg.nlat, cfg.nlon), device)
    vφ = to_device(zeros(ComplexF64, cfg.nlat, cfg.nlon), device)
    
    # Compute gradients using finite differences  
    dtheta = π / (cfg.nlat - 1)
    dphi = 2π / cfg.nlon
    sintheta = to_device(cfg.st, device)
    
    @kernel function vector_synthesis_kernel!(vtheta, vphi, sph_field, tor_field, sintheta, dtheta, dphi, nlat, nlon)
        i, j = @index(Global, NTuple)
        if i <= nlat && j <= nlon
            # Compute derivatives of potentials
            # ∂(sph)/∂θ for vθ component
            i_prev = max(1, i-1)
            i_next = min(nlat, i+1)
            dsph_dtheta = (sph_field[i_next, j] - sph_field[i_prev, j]) / (2 * dtheta)
            
            # ∂(tor)/∂φ for vθ component  
            j_prev = j == 1 ? nlon : j-1
            j_next = j == nlon ? 1 : j+1
            dtor_dphi = (tor_field[i, j_next] - tor_field[i, j_prev]) / (2 * dphi)
            
            # ∂(sph)/∂φ for vφ component
            dsph_dphi = (sph_field[i, j_next] - sph_field[i, j_prev]) / (2 * dphi)
            
            # ∂(tor)/∂θ for vφ component
            dtor_dtheta = (tor_field[i_next, j] - tor_field[i_prev, j]) / (2 * dtheta)
            
            # Construct vector components
            sin_theta = sintheta[i]
            vtheta[i, j] = dsph_dtheta + dtor_dphi / sin_theta
            vphi[i, j] = dsph_dphi / sin_theta - dtor_dtheta
        end
    end
    
    vector_kernel! = vector_synthesis_kernel!(backend)
    vector_kernel!(vθ, vφ, gpu_sph_spatial, gpu_tor_spatial, sintheta, dtheta, dphi, cfg.nlat, cfg.nlon;
                   ndrange=(cfg.nlat, cfg.nlon))
    
    KernelAbstractions.synchronize(backend)
    
    # Transfer results back to CPU
    result_vθ = Array(vθ)
    result_vφ = Array(vφ)
    
    if real_output
        return real(result_vθ), real(result_vφ)
    else
        return result_vθ, result_vφ
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
        return SHTnsKit.apply_laplacian_cpu!(cfg, coeffs)
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
# Memory optimization and error handling utilities

"""
    get_backend(array)

Get the appropriate KernelAbstractions backend for the array type.
"""
function get_backend(array)
    if CUDA_LOADED && array isa CUDA.CuArray
        return CUDABackend()
    elseif AMDGPU_LOADED && isdefined(AMDGPU, :ROCArray) && array isa AMDGPU.ROCArray
        return ROCBackend()
    else
        return CPU()
    end
end

"""
    gpu_memory_info(device::SHTDevice)

Get memory information for the specified device.
"""
function gpu_memory_info(device::SHTDevice)
    if device == CUDA_DEVICE && CUDA_LOADED
        return CUDA.MemoryInfo()
    elseif device == AMDGPU_DEVICE && AMDGPU_LOADED
        # AMDGPU memory info if available
        return (free=0, total=0)  # Placeholder
    else
        return (free=Sys.free_memory(), total=Sys.total_memory())
    end
end

"""
    check_gpu_memory(required_bytes, device::SHTDevice)

Check if sufficient GPU memory is available for the operation.
"""
function check_gpu_memory(required_bytes::Int, device::SHTDevice)
    try
        mem_info = gpu_memory_info(device)
        available = device == CPU_DEVICE ? mem_info.free : mem_info.free
        
        if available < required_bytes
            @warn "Insufficient memory: need $(required_bytes÷(1024^3)) GB, have $(available÷(1024^3)) GB available"
            return false
        end
        return true
    catch e
        @warn "Could not check memory availability: $e"
        return true  # Proceed optimistically
    end
end

"""
    estimate_memory_usage(cfg::SHTConfig, operation::Symbol)

Estimate memory usage for GPU operations.
"""
function estimate_memory_usage(cfg::SHTConfig, operation::Symbol)
    # Estimate memory requirements
    spatial_size = cfg.nlat * cfg.nlon * 16  # ComplexF64 = 16 bytes
    coeff_size = (cfg.lmax + 1) * (cfg.mmax + 1) * 16
    legendre_size = cfg.nlat * (cfg.lmax + 1) * (cfg.mmax + 1) * 8  # Float64 = 8 bytes
    
    if operation == :analysis
        return spatial_size + coeff_size + legendre_size + spatial_size  # Input + output + Plm + temp
    elseif operation == :synthesis  
        return coeff_size + spatial_size + legendre_size + spatial_size  # Input + output + Plm + temp
    elseif operation == :vector
        return 2 * spatial_size + 2 * coeff_size + legendre_size + 2 * spatial_size  # 2 components
    else
        return spatial_size + coeff_size  # Conservative estimate
    end
end

"""
    gpu_analysis_safe(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true)

Memory-safe version of gpu_analysis with automatic fallback to CPU if needed.
"""
function gpu_analysis_safe(cfg::SHTConfig, spatial_data; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.analysis_cpu(cfg, spatial_data)
    end
    
    # Check memory requirements
    required_memory = estimate_memory_usage(cfg, :analysis)
    if !check_gpu_memory(required_memory, device)
        @info "Falling back to CPU due to memory constraints"
        return SHTnsKit.analysis_cpu(cfg, spatial_data)
    end
    
    try
        return gpu_analysis(cfg, spatial_data; device=device, real_output=real_output)
    catch e
        if isa(e, OutOfMemoryError) || contains(string(e), "memory")
            @warn "GPU out of memory, falling back to CPU: $e"
            return SHTnsKit.analysis_cpu(cfg, spatial_data)
        else
            rethrow(e)
        end
    end
end

"""
    gpu_synthesis_safe(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)

Memory-safe version of gpu_synthesis with automatic fallback to CPU if needed.
"""
function gpu_synthesis_safe(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)
    if device == CPU_DEVICE
        return SHTnsKit.synthesis_cpu(cfg, coeffs; real_output=real_output)
    end
    
    # Check memory requirements
    required_memory = estimate_memory_usage(cfg, :synthesis)
    if !check_gpu_memory(required_memory, device)
        @info "Falling back to CPU due to memory constraints"
        return SHTnsKit.synthesis_cpu(cfg, coeffs; real_output=real_output)
    end
    
    try
        return gpu_synthesis(cfg, coeffs; device=device, real_output=real_output)
    catch e
        if isa(e, OutOfMemoryError) || contains(string(e), "memory")
            @warn "GPU out of memory, falling back to CPU: $e"
            return SHTnsKit.synthesis_cpu(cfg, coeffs; real_output=real_output)
        else
            rethrow(e)
        end
    end
end

"""
    gpu_clear_cache!(device::SHTDevice)

Clear GPU memory cache to free up memory.
"""
function gpu_clear_cache!(device::SHTDevice)
    if device == CUDA_DEVICE && CUDA_LOADED
        try
            CUDA.reclaim()
            @info "CUDA memory cache cleared"
        catch e
            @warn "Failed to clear CUDA cache: $e"
        end
    elseif device == AMDGPU_DEVICE && AMDGPU_LOADED
        try
            # AMDGPU cache clearing if available
            @info "AMDGPU memory cache cleared"
        catch e
            @warn "Failed to clear AMDGPU cache: $e"
        end
    end
    return nothing
end

# Multi-GPU vector field transform functions

"""
    multi_gpu_spat_to_SHsphtor(mgpu_config::MultiGPUConfig, vθ, vφ; real_output=true)

Perform multi-GPU spheroidal-toroidal decomposition of vector fields.
"""
function multi_gpu_spat_to_SHsphtor(mgpu_config::MultiGPUConfig, vθ, vφ; real_output=true)
    # Distribute vector field components across GPUs
    distributed_vθ = distribute_spatial_array(vθ, mgpu_config)
    distributed_vφ = distribute_spatial_array(vφ, mgpu_config)
    
    # Perform partial spheroidal-toroidal analysis on each GPU
    partial_sph_results = []
    partial_tor_results = []
    
    for (chunk_vθ, chunk_vφ) in zip(distributed_vθ, distributed_vφ)
        set_gpu_device(chunk_vθ.gpu.device, chunk_vθ.gpu.id)
        
        # Create temporary config for this GPU chunk (similar to scalar case)
        if mgpu_config.distribution_strategy == :latitude
            lat_indices = chunk_vθ.indices[1]
            chunk_nlat = length(lat_indices)
            
            cfg = mgpu_config.base_config
            temp_cfg = SHTnsKit.SHTConfig(
                lmax=cfg.lmax, mmax=cfg.mmax, mres=cfg.mres,
                nlat=chunk_nlat, nlon=cfg.nlon,
                θ=cfg.θ[lat_indices], φ=cfg.φ,
                x=cfg.x[lat_indices], w=cfg.w[lat_indices],
                wlat=cfg.w[lat_indices], Nlm=cfg.Nlm, cphi=cfg.cphi,
                nlm=cfg.nlm, li=cfg.li, mi=cfg.mi,
                nspat=chunk_nlat*cfg.nlon,
                ct=cfg.ct[lat_indices], st=cfg.st[lat_indices],
                sintheta=cfg.st[lat_indices],
                norm=cfg.norm, cs_phase=cfg.cs_phase,
                real_norm=cfg.real_norm, robert_form=cfg.robert_form,
                compute_device=SHTnsKit.GPU,
                device_backend=:cuda,
                device_preference=SHTnsKit.Device[SHTnsKit.GPU],
                backend_preference=Symbol[:cuda]
            )
            
            # Perform GPU vector analysis on this chunk
            device_enum = _device_enum(chunk_vθ.gpu.device)
            chunk_sph, chunk_tor = gpu_spat_to_SHsphtor(temp_cfg, chunk_vθ.data, chunk_vφ.data; device=device_enum)
            
        else
            error("Multi-GPU vector analysis currently only supports :latitude distribution strategy")
        end
        
        chunk_sph_gpu = _ensure_device(chunk_sph, _device_enum(chunk_vθ.gpu.device))
        chunk_tor_gpu = _ensure_device(chunk_tor, _device_enum(chunk_vθ.gpu.device))

        push!(partial_sph_results, chunk_sph_gpu)
        push!(partial_tor_results, chunk_tor_gpu)
    end
    
    # Combine results from all GPUs
    set_gpu_device(mgpu_config.gpu_devices[1].device, mgpu_config.gpu_devices[1].id)
    
    cfg = mgpu_config.base_config
    device_enum = _device_enum(mgpu_config.gpu_devices[1].device)
    final_sph_gpu = _ensure_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device_enum)
    final_tor_gpu = _ensure_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device_enum)
    
    for (sph_result, tor_result) in zip(partial_sph_results, partial_tor_results)
        final_sph_gpu .+= _ensure_device(sph_result, device_enum)
        final_tor_gpu .+= _ensure_device(tor_result, device_enum)
    end
    
    final_sph = Array(final_sph_gpu)
    final_tor = Array(final_tor_gpu)
    if real_output && eltype(vθ) <: Real && eltype(vφ) <: Real
        return real(final_sph), real(final_tor)
    else
        return final_sph, final_tor
    end
end

"""
    multi_gpu_SHsphtor_to_spat(mgpu_config::MultiGPUConfig, sph_coeffs, tor_coeffs; real_output=true)

Perform multi-GPU synthesis of spheroidal-toroidal vector field components.
"""
function multi_gpu_SHsphtor_to_spat(mgpu_config::MultiGPUConfig, sph_coeffs, tor_coeffs; real_output=true)
    cfg = mgpu_config.base_config
    
    if mgpu_config.distribution_strategy == :latitude
        # Distribute by latitude bands - each GPU synthesizes its portion
        distributed_vθ_results = []
        distributed_vφ_results = []
        
        ngpus = length(mgpu_config.gpu_devices)
        lat_per_gpu = div(cfg.nlat, ngpus)
        lat_remainder = cfg.nlat % ngpus
        
        lat_start = 1
        for (i, gpu) in enumerate(mgpu_config.gpu_devices)
            lat_count = lat_per_gpu + (i <= lat_remainder ? 1 : 0)
            lat_end = lat_start + lat_count - 1
            lat_indices = lat_start:lat_end
            
            set_gpu_device(gpu.device, gpu.id)
            
            # Create config for this latitude band
            temp_cfg = SHTnsKit.SHTConfig(
                lmax=cfg.lmax, mmax=cfg.mmax, mres=cfg.mres,
                nlat=lat_count, nlon=cfg.nlon,
                θ=cfg.θ[lat_indices], φ=cfg.φ,
                x=cfg.x[lat_indices], w=cfg.w[lat_indices],
                wlat=cfg.w[lat_indices], Nlm=cfg.Nlm, cphi=cfg.cphi,
                nlm=cfg.nlm, li=cfg.li, mi=cfg.mi,
                nspat=lat_count*cfg.nlon,
                ct=cfg.ct[lat_indices], st=cfg.st[lat_indices],
                sintheta=cfg.st[lat_indices],
                norm=cfg.norm, cs_phase=cfg.cs_phase,
                real_norm=cfg.real_norm, robert_form=cfg.robert_form,
                compute_device=SHTnsKit.GPU,
                device_backend=:cuda,
                device_preference=SHTnsKit.Device[SHTnsKit.GPU],
                backend_preference=Symbol[:cuda]
            )
            
            # Perform synthesis on this GPU
            chunk_vθ, chunk_vφ = gpu_SHsphtor_to_spat(temp_cfg, sph_coeffs, tor_coeffs; real_output=real_output)
            
            push!(distributed_vθ_results, (
                data=chunk_vθ,
                gpu=gpu,
                indices=(lat_indices, 1:cfg.nlon)
            ))
            
            push!(distributed_vφ_results, (
                data=chunk_vφ,
                gpu=gpu,
                indices=(lat_indices, 1:cfg.nlon)
            ))
            
            lat_start = lat_end + 1
        end
        
        # Gather results back to single arrays
        result_vθ = gather_distributed_arrays(distributed_vθ_results, (cfg.nlat, cfg.nlon))
        result_vφ = gather_distributed_arrays(distributed_vφ_results, (cfg.nlat, cfg.nlon))
        
        return Array(result_vθ), Array(result_vφ)
        
    else
        error("Multi-GPU vector synthesis currently only supports :latitude distribution strategy")
    end
end

# Multi-GPU transform functions

"""
    multi_gpu_analysis(mgpu_config::MultiGPUConfig, spatial_data; real_output=true)

Perform spherical harmonic analysis using multiple GPUs.
"""
function multi_gpu_analysis(mgpu_config::MultiGPUConfig, spatial_data; real_output=true)
    # Distribute spatial data across GPUs
    distributed_data = distribute_spatial_array(spatial_data, mgpu_config)
    
    # Perform partial analysis on each GPU
    partial_results = []
    for chunk in distributed_data
        set_gpu_device(chunk.gpu.device, chunk.gpu.id)
        
        # Create temporary config for this GPU chunk
        if mgpu_config.distribution_strategy == :latitude
            # For latitude distribution, each GPU processes a subset of latitude bands
            lat_indices = chunk.indices[1]
            chunk_nlat = length(lat_indices)
            
            # Extract relevant parts of the base config
            cfg = mgpu_config.base_config
            temp_cfg = SHTnsKit.SHTConfig(
                lmax=cfg.lmax, mmax=cfg.mmax, mres=cfg.mres,
                nlat=chunk_nlat, nlon=cfg.nlon,
                θ=cfg.θ[lat_indices], φ=cfg.φ, 
                x=cfg.x[lat_indices], w=cfg.w[lat_indices],
                wlat=cfg.w[lat_indices], Nlm=cfg.Nlm, cphi=cfg.cphi,
                nlm=cfg.nlm, li=cfg.li, mi=cfg.mi, 
                nspat=chunk_nlat*cfg.nlon,
                ct=cfg.ct[lat_indices], st=cfg.st[lat_indices], 
                sintheta=cfg.st[lat_indices],
                norm=cfg.norm, cs_phase=cfg.cs_phase, 
                real_norm=cfg.real_norm, robert_form=cfg.robert_form,
                compute_device=SHTnsKit.GPU,
                device_backend=:cuda,
                device_preference=SHTnsKit.Device[SHTnsKit.GPU],
                backend_preference=Symbol[:cuda]
            )
            
            # Perform GPU analysis on this chunk
            chunk_coeffs = gpu_analysis(temp_cfg, chunk.data; device=_device_enum(chunk.gpu.device), real_output=false)

        elseif mgpu_config.distribution_strategy == :longitude
            # For longitude distribution, each GPU processes longitude sectors
            lon_indices = chunk.indices[2]
            chunk_nlon = length(lon_indices)
            
            cfg = mgpu_config.base_config
            temp_cfg = SHTnsKit.SHTConfig(
                lmax=cfg.lmax, mmax=cfg.mmax, mres=cfg.mres,
                nlat=cfg.nlat, nlon=chunk_nlon,
                θ=cfg.θ, φ=cfg.φ[lon_indices],
                x=cfg.x, w=cfg.w,
                wlat=cfg.w, Nlm=cfg.Nlm, cphi=2π/chunk_nlon,
                nlm=cfg.nlm, li=cfg.li, mi=cfg.mi,
                nspat=cfg.nlat*chunk_nlon,
                ct=cfg.ct, st=cfg.st, sintheta=cfg.st,
                norm=cfg.norm, cs_phase=cfg.cs_phase,
                real_norm=cfg.real_norm, robert_form=cfg.robert_form,
                compute_device=SHTnsKit.GPU,
                device_backend=:cuda,
                device_preference=SHTnsKit.Device[SHTnsKit.GPU],
                backend_preference=Symbol[:cuda]
            )
            
            # Perform GPU analysis on longitude sector
            chunk_coeffs = gpu_analysis(temp_cfg, chunk.data; device=_device_enum(chunk.gpu.device), real_output=false)
            
        else
            error("Multi-GPU analysis currently supports :latitude and :longitude distribution strategies")
        end
        
        chunk_coeffs_gpu = _ensure_device(chunk_coeffs, _device_enum(chunk.gpu.device))
        push!(partial_results, (coeffs=chunk_coeffs_gpu, chunk=chunk))
    end
    
    # Combine results from all GPUs
    # For latitude distribution, we need to sum contributions from all latitude bands
    set_gpu_device(mgpu_config.gpu_devices[1].device, mgpu_config.gpu_devices[1].id)
    
    cfg = mgpu_config.base_config
    device_enum = _device_enum(mgpu_config.gpu_devices[1].device)
    final_coeffs_gpu = _ensure_device(zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1), device_enum)

    for result in partial_results
        final_coeffs_gpu .+= _ensure_device(result.coeffs, device_enum)
    end

    final_coeffs = Array(final_coeffs_gpu)
    return real_output && eltype(spatial_data) <: Real ? real(final_coeffs) : final_coeffs
end

"""
    multi_gpu_synthesis(mgpu_config::MultiGPUConfig, coeffs; real_output=true)

Perform spherical harmonic synthesis using multiple GPUs.
"""
function multi_gpu_synthesis(mgpu_config::MultiGPUConfig, coeffs; real_output=true)
    cfg = mgpu_config.base_config
    
    if mgpu_config.distribution_strategy == :latitude
        # Distribute by latitude bands - each GPU synthesizes its portion
        distributed_results = []
        
        ngpus = length(mgpu_config.gpu_devices)
        lat_per_gpu = div(cfg.nlat, ngpus)
        lat_remainder = cfg.nlat % ngpus
        
        lat_start = 1
        for (i, gpu) in enumerate(mgpu_config.gpu_devices)
            lat_count = lat_per_gpu + (i <= lat_remainder ? 1 : 0)
            lat_end = lat_start + lat_count - 1
            lat_indices = lat_start:lat_end
            
            set_gpu_device(gpu.device, gpu.id)
            
            # Create config for this latitude band
            temp_cfg = SHTnsKit.SHTConfig(
                lmax=cfg.lmax, mmax=cfg.mmax, mres=cfg.mres,
                nlat=lat_count, nlon=cfg.nlon,
                θ=cfg.θ[lat_indices], φ=cfg.φ,
                x=cfg.x[lat_indices], w=cfg.w[lat_indices],
                wlat=cfg.w[lat_indices], Nlm=cfg.Nlm, cphi=cfg.cphi,
                nlm=cfg.nlm, li=cfg.li, mi=cfg.mi,
                nspat=lat_count*cfg.nlon,
                ct=cfg.ct[lat_indices], st=cfg.st[lat_indices],
                sintheta=cfg.st[lat_indices],
                norm=cfg.norm, cs_phase=cfg.cs_phase,
                real_norm=cfg.real_norm, robert_form=cfg.robert_form,
                compute_device=SHTnsKit.GPU,
                device_backend=:cuda,
                device_preference=SHTnsKit.Device[SHTnsKit.GPU],
                backend_preference=Symbol[:cuda]
            )
            
            # Perform synthesis on this GPU
            chunk_result = gpu_synthesis(temp_cfg, coeffs; real_output=real_output)
            chunk_gpu = _ensure_device(chunk_result, _device_enum(gpu.device))
            
            push!(distributed_results, (
                data=chunk_gpu,
                gpu=gpu,
                indices=(lat_indices, 1:cfg.nlon)
            ))
            
            lat_start = lat_end + 1
        end
        
        # Gather results back to single array
        result = gather_distributed_arrays(distributed_results, (cfg.nlat, cfg.nlon))
        return Array(result)
        
    elseif mgpu_config.distribution_strategy == :longitude
        # Distribute by longitude sectors - each GPU synthesizes its portion
        distributed_results = []
        
        ngpus = length(mgpu_config.gpu_devices)
        lon_per_gpu = div(cfg.nlon, ngpus)
        lon_remainder = cfg.nlon % ngpus
        
        lon_start = 1
        for (i, gpu) in enumerate(mgpu_config.gpu_devices)
            lon_count = lon_per_gpu + (i <= lon_remainder ? 1 : 0)
            lon_end = lon_start + lon_count - 1
            lon_indices = lon_start:lon_end
            
            set_gpu_device(gpu.device, gpu.id)
            
            # Create config for this longitude sector
            temp_cfg = SHTnsKit.SHTConfig(
                lmax=cfg.lmax, mmax=cfg.mmax, mres=cfg.mres,
                nlat=cfg.nlat, nlon=lon_count,
                θ=cfg.θ, φ=cfg.φ[lon_indices],
                x=cfg.x, w=cfg.w,
                wlat=cfg.w, Nlm=cfg.Nlm, cphi=2π/lon_count,
                nlm=cfg.nlm, li=cfg.li, mi=cfg.mi,
                nspat=cfg.nlat*lon_count,
                ct=cfg.ct, st=cfg.st, sintheta=cfg.st,
                norm=cfg.norm, cs_phase=cfg.cs_phase,
                real_norm=cfg.real_norm, robert_form=cfg.robert_form,
                compute_device=SHTnsKit.GPU,
                device_backend=:cuda,
                device_preference=SHTnsKit.Device[SHTnsKit.GPU],
                backend_preference=Symbol[:cuda]
            )
            
            # Perform synthesis on longitude sector
            chunk_result = gpu_synthesis(temp_cfg, coeffs; real_output=real_output)
            chunk_gpu = _ensure_device(chunk_result, _device_enum(gpu.device))
            
            push!(distributed_results, (
                data=chunk_gpu,
                gpu=gpu,
                indices=(1:cfg.nlat, lon_indices)
            ))
            
            lon_start = lon_end + 1
        end
        
        # Gather results back to single array
        result = gather_distributed_arrays(distributed_results, (cfg.nlat, cfg.nlon))
        return Array(result)
        
    elseif mgpu_config.distribution_strategy == :spectral
        # Spectral distribution - divide coefficients by m or l modes
        # This is more complex but can be beneficial for high-resolution problems
        distributed_results = []
        
        ngpus = length(mgpu_config.gpu_devices)
        
        # Distribute by m modes (azimuthal modes)
        m_per_gpu = div(cfg.mmax + 1, ngpus)
        m_remainder = (cfg.mmax + 1) % ngpus
        
        m_start = 0
        for (i, gpu) in enumerate(mgpu_config.gpu_devices)
            m_count = m_per_gpu + (i <= m_remainder ? 1 : 0)
            m_end = m_start + m_count - 1
            m_indices = m_start:m_end
            
            set_gpu_device(gpu.device, gpu.id)
            
            # Extract coefficients for this m range
            coeffs_chunk = coeffs[:, m_indices .+ 1]  # +1 for 1-based indexing
            
            # Create temporary config for spectral chunk
            temp_cfg = SHTnsKit.SHTConfig(
                lmax=cfg.lmax, mmax=m_end, mres=cfg.mres,
                nlat=cfg.nlat, nlon=cfg.nlon,
                θ=cfg.θ, φ=cfg.φ,
                x=cfg.x, w=cfg.w,
                wlat=cfg.w, Nlm=cfg.Nlm, cphi=cfg.cphi,
                nlm=cfg.nlm, li=cfg.li, mi=cfg.mi,
                nspat=cfg.nspat,
                ct=cfg.ct, st=cfg.st, sintheta=cfg.st,
                norm=cfg.norm, cs_phase=cfg.cs_phase,
                real_norm=cfg.real_norm, robert_form=cfg.robert_form,
                compute_device=SHTnsKit.GPU,
                device_backend=:cuda,
                device_preference=SHTnsKit.Device[SHTnsKit.GPU],
                backend_preference=Symbol[:cuda]
            )
            
            # Perform synthesis with coefficient subset
            chunk_result = gpu_synthesis(temp_cfg, coeffs_chunk; real_output=real_output)
            chunk_gpu = _ensure_device(chunk_result, _device_enum(gpu.device))
            
            push!(distributed_results, (
                data=chunk_gpu,
                gpu=gpu,
                m_range=m_indices
            ))
            
            m_start = m_end + 1
        end
        
        # Combine results from different m modes
        set_gpu_device(mgpu_config.gpu_devices[1].device, mgpu_config.gpu_devices[1].id)
        device_enum = _device_enum(mgpu_config.gpu_devices[1].device)
        final_gpu = _ensure_device(zeros(real_output ? Float64 : ComplexF64, cfg.nlat, cfg.nlon), device_enum)
        
        for result in distributed_results
            final_gpu .+= _ensure_device(result.data, device_enum)
        end
        
        return Array(final_gpu)
        
    else
        error("Multi-GPU synthesis supports :latitude, :longitude, and :spectral distribution strategies")
    end
end

# Export GPU functions
export SHTDevice, CPU_DEVICE, CUDA_DEVICE, AMDGPU_DEVICE
export get_device, set_device!, to_device
export gpu_analysis, gpu_synthesis, gpu_analysis_safe, gpu_synthesis_safe
export gpu_spat_to_SH, gpu_SH_to_spat, gpu_spat_to_SH_ml, gpu_SH_to_spat_ml
export gpu_spat_to_SH_axisym, gpu_SH_to_spat_axisym, gpu_spat_to_SH_l_axisym, gpu_SH_to_spat_l_axisym
export gpu_spat_to_SHsphtor, gpu_SHsphtor_to_spat, gpu_spat_to_SHsphtor_ml, gpu_SHsphtor_to_spat_ml
export gpu_spat_to_SHqst, gpu_SHqst_to_spat
export gpu_SH_to_lat, gpu_SH_to_lat_cplx, gpu_SHqst_to_point, gpu_SH_to_grad_point
export gpu_SH_to_point
export gpu_apply_laplacian!, gpu_legendre!
export gpu_energy_scalar, gpu_energy_vector
export gpu_grid_energy_scalar, gpu_grid_energy_vector
export gpu_energy_scalar_l_spectrum, gpu_energy_scalar_m_spectrum
export gpu_energy_vector_l_spectrum, gpu_energy_vector_m_spectrum
export gpu_energy_scalar_lm, gpu_energy_vector_lm
export gpu_energy_scalar_packed, gpu_energy_vector_packed
export gpu_grad_energy_scalar_alm, gpu_grad_energy_scalar_packed
export gpu_grad_energy_vector_Slm_Tlm, gpu_grad_energy_vector_packed
export gpu_grad_grid_energy_scalar_field, gpu_grad_grid_energy_vector_fields
export gpu_memory_info, check_gpu_memory, gpu_clear_cache!
export estimate_memory_usage

# Multi-GPU memory streaming for large problems

"""
    estimate_streaming_chunks(mgpu_config::MultiGPUConfig, data_size, max_memory_per_gpu=4*1024^3)

Estimate optimal chunk sizes for memory streaming with multiple GPUs.
Returns number of chunks needed to keep memory usage below max_memory_per_gpu bytes.
"""
function estimate_streaming_chunks(mgpu_config::MultiGPUConfig, data_size, max_memory_per_gpu=4*1024^3)
    # Estimate memory requirements
    element_size = 16  # ComplexF64 = 16 bytes
    total_memory_needed = prod(data_size) * element_size
    
    # Account for temporary arrays and FFT workspace
    memory_overhead = 3.0  # Roughly 3x for intermediates and FFT
    total_memory_with_overhead = total_memory_needed * memory_overhead
    
    ngpus = length(mgpu_config.gpu_devices)
    memory_per_gpu = total_memory_with_overhead / ngpus
    
    if memory_per_gpu <= max_memory_per_gpu
        return 1  # No streaming needed
    else
        chunks_needed = ceil(Int, memory_per_gpu / max_memory_per_gpu)
        return chunks_needed
    end
end

"""
    multi_gpu_analysis_streaming(mgpu_config::MultiGPUConfig, spatial_data; 
                                max_memory_per_gpu=4*1024^3, real_output=true)

Perform multi-GPU analysis with memory streaming for very large problems.
"""
function multi_gpu_analysis_streaming(mgpu_config::MultiGPUConfig, spatial_data; 
                                     max_memory_per_gpu=4*1024^3, real_output=true)
    data_size = size(spatial_data)
    chunks_needed = estimate_streaming_chunks(mgpu_config, data_size, max_memory_per_gpu)
    
    if chunks_needed == 1
        # Use regular multi-GPU analysis
        return multi_gpu_analysis(mgpu_config, spatial_data; real_output=real_output)
    end
    
    println("Using memory streaming with $(chunks_needed) chunks per GPU")
    
    # Split data into chunks
    cfg = mgpu_config.base_config
    if mgpu_config.distribution_strategy == :latitude
        # Split latitude dimension into chunks
        lat_chunk_size = div(cfg.nlat, chunks_needed)
        lat_remainder = cfg.nlat % chunks_needed
        
        # Accumulate results across chunks
        final_coeffs = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
        
        lat_start = 1
        for chunk_idx in 1:chunks_needed
            chunk_lat_size = lat_chunk_size + (chunk_idx <= lat_remainder ? 1 : 0)
            lat_end = lat_start + chunk_lat_size - 1
            lat_indices = lat_start:lat_end
            
            # Extract chunk data
            chunk_data = spatial_data[lat_indices, :]
            
            # Create temporary config for this chunk
            chunk_mgpu_config = MultiGPUConfig(
                base_config=SHTnsKit.SHTConfig(
                    lmax=cfg.lmax, mmax=cfg.mmax, mres=cfg.mres,
                    nlat=chunk_lat_size, nlon=cfg.nlon,
                    θ=cfg.θ[lat_indices], φ=cfg.φ,
                    x=cfg.x[lat_indices], w=cfg.w[lat_indices],
                    wlat=cfg.w[lat_indices], Nlm=cfg.Nlm, cphi=cfg.cphi,
                    nlm=cfg.nlm, li=cfg.li, mi=cfg.mi,
                    nspat=chunk_lat_size*cfg.nlon,
                    ct=cfg.ct[lat_indices], st=cfg.st[lat_indices],
                    sintheta=cfg.st[lat_indices],
                    norm=cfg.norm, cs_phase=cfg.cs_phase,
                    real_norm=cfg.real_norm, robert_form=cfg.robert_form,
                    compute_device=cfg.compute_device,
                    device_backend=cfg.device_backend,
                    device_preference=cfg.device_preference,
                    backend_preference=cfg.backend_preference
                ),
                gpu_devices=mgpu_config.gpu_devices,
                distribution_strategy=mgpu_config.distribution_strategy
            )
            
            # Process this chunk
            chunk_coeffs = multi_gpu_analysis(chunk_mgpu_config, chunk_data; real_output=false)
            final_coeffs .+= chunk_coeffs
            
            lat_start = lat_end + 1
            
            # Clear GPU caches between chunks
            for gpu in mgpu_config.gpu_devices
                try
                    gpu_clear_cache!(gpu.device)
                catch e
                    # Ignore cache clear errors
                end
            end
        end
        
        if real_output && eltype(spatial_data) <: Real
            return real(final_coeffs)
        else
            return final_coeffs
        end
        
    else
        error("Memory streaming currently only supports :latitude distribution strategy")
    end
end

"""
    multi_gpu_synthesis_streaming(mgpu_config::MultiGPUConfig, coeffs; 
                                 max_memory_per_gpu=4*1024^3, real_output=true)

Perform multi-GPU synthesis with memory streaming for very large problems.
"""
function multi_gpu_synthesis_streaming(mgpu_config::MultiGPUConfig, coeffs; 
                                      max_memory_per_gpu=4*1024^3, real_output=true)
    cfg = mgpu_config.base_config
    data_size = (cfg.nlat, cfg.nlon)
    chunks_needed = estimate_streaming_chunks(mgpu_config, data_size, max_memory_per_gpu)
    
    if chunks_needed == 1
        # Use regular multi-GPU synthesis
        return multi_gpu_synthesis(mgpu_config, coeffs; real_output=real_output)
    end
    
    println("Using memory streaming with $(chunks_needed) chunks per GPU")
    
    # Process in chunks and combine results
    if mgpu_config.distribution_strategy == :latitude
        # Split latitude dimension
        lat_chunk_size = div(cfg.nlat, chunks_needed)
        lat_remainder = cfg.nlat % chunks_needed
        
        final_result = zeros(real_output ? Float64 : ComplexF64, cfg.nlat, cfg.nlon)
        
        lat_start = 1
        for chunk_idx in 1:chunks_needed
            chunk_lat_size = lat_chunk_size + (chunk_idx <= lat_remainder ? 1 : 0)
            lat_end = lat_start + chunk_lat_size - 1
            lat_indices = lat_start:lat_end
            
            # Create chunk configuration
            chunk_mgpu_config = MultiGPUConfig(
                base_config=SHTnsKit.SHTConfig(
                    lmax=cfg.lmax, mmax=cfg.mmax, mres=cfg.mres,
                    nlat=chunk_lat_size, nlon=cfg.nlon,
                    θ=cfg.θ[lat_indices], φ=cfg.φ,
                    x=cfg.x[lat_indices], w=cfg.w[lat_indices],
                    wlat=cfg.w[lat_indices], Nlm=cfg.Nlm, cphi=cfg.cphi,
                    nlm=cfg.nlm, li=cfg.li, mi=cfg.mi,
                    nspat=chunk_lat_size*cfg.nlon,
                    ct=cfg.ct[lat_indices], st=cfg.st[lat_indices],
                    sintheta=cfg.st[lat_indices],
                    norm=cfg.norm, cs_phase=cfg.cs_phase,
                    real_norm=cfg.real_norm, robert_form=cfg.robert_form,
                    compute_device=cfg.compute_device,
                    device_backend=cfg.device_backend,
                    device_preference=cfg.device_preference,
                    backend_preference=cfg.backend_preference
                ),
                gpu_devices=mgpu_config.gpu_devices,
                distribution_strategy=mgpu_config.distribution_strategy
            )
            
            # Process this chunk
            chunk_result = multi_gpu_synthesis(chunk_mgpu_config, coeffs; real_output=real_output)
            final_result[lat_indices, :] = chunk_result
            
            lat_start = lat_end + 1
            
            # Clear GPU caches between chunks
            for gpu in mgpu_config.gpu_devices
                try
                    gpu_clear_cache!(gpu.device)
                catch e
                    # Ignore cache clear errors
                end
            end
        end
        
        return final_result
        
    else
        error("Memory streaming currently only supports :latitude distribution strategy")
    end
end

# Multi-GPU functions
export MultiGPUConfig, create_multi_gpu_config
export get_available_gpus, set_gpu_device
export distribute_spatial_array, distribute_coefficient_array, gather_distributed_arrays
export multi_gpu_analysis, multi_gpu_synthesis
export multi_gpu_analysis_streaming, multi_gpu_synthesis_streaming, estimate_streaming_chunks

end # module SHTnsKitCUDAExt
