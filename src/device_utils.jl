"""
Device utilities for GPU-accelerated spherical harmonic transforms.

This module provides device management and selection utilities for SHTnsKit.jl,
enabling transparent switching between CPU, CUDA, and AMDGPU backends.
"""

"""
    select_compute_device(preference_order=[:cpu, :cuda, :amdgpu])

Automatically select the best available compute device based on preference order.
Returns the selected device as a Symbol and whether GPU extensions are loaded.

# Arguments
- `preference_order`: Vector of preferred devices in order (:cpu, :cuda, :amdgpu)

# Returns
- `(device::Symbol, gpu_available::Bool)`: Selected device and GPU availability status

# Examples
```julia
device, gpu_ok = select_compute_device()
device, gpu_ok = select_compute_device([:cuda, :cpu])  # Prefer CUDA if available
```
"""
function select_compute_device(preference_order::Vector{Symbol}=[:cpu, :cuda, :amdgpu])
    for device in preference_order
        if device == :cpu
            return :cpu, false
        elseif device == :cuda
            # Check if CUDA is available (this will be properly implemented in the extension)
            if isdefined(Main, :CUDA) && try Main.CUDA.functional() catch; false end
                return :cuda, true
            end
        elseif device == :amdgpu  
            # Check if AMDGPU is available (this will be properly implemented in the extension)
            if isdefined(Main, :AMDGPU) && try Main.AMDGPU.functional() catch; false end
                return :amdgpu, true
            end
        end
    end
    
    # Fallback to CPU
    return :cpu, false
end

"""
    device_transfer_arrays(cfg::SHTConfig, arrays...)

Transfer arrays to the device specified in the configuration.
This is a placeholder that will be properly implemented in the GPU extension.
"""
function device_transfer_arrays(cfg, arrays...)
    if cfg.compute_device == :cpu
        return arrays  # No transfer needed for CPU
    else
        # This will be implemented in the GPU extension
        error("GPU extension required for device transfers. Load CUDA.jl or AMDGPU.jl with GPUArrays.")
    end
end