"""
Device utilities for GPU-accelerated spherical harmonic transforms.

This module provides device management and selection utilities for SHTnsKit.jl,
enabling transparent switching between CPU and GPU backends.
"""


"""
    select_compute_device(preference_order)

Automatically select the best available compute device based on preference order.
Returns both the logical device (`Device`) and the concrete backend symbol.

# Arguments
- `preference_order`: Vector describing preferred targets. Each entry can be a
  `Device` (`CPU` or `GPU`) or a backend symbol (`:cpu`, `:cuda`).

# Returns
- `(device::Device, backend::Symbol, gpu_available::Bool)`

# Examples
```julia
kind, backend, gpu_ok = select_compute_device()
kind, backend, gpu_ok = select_compute_device(Device[GPU, CPU])  # Prefer GPU when available
```
"""
function select_compute_device(preference_order=Device[GPU, CPU])
    normalized = [_normalize_device_entry(entry) for entry in preference_order]

    for backend in normalized
        if backend == :cpu
            return CPU, :cpu, false
        elseif backend == :cuda
            if cuda_available()
                return GPU, :cuda, true
            end
        elseif backend == :gpu
            if cuda_available()
                return GPU, :cuda, true
            end
        end
    end

    return CPU, :cpu, false
end

_normalize_device_entry(entry::Device) = device_symbol(entry)
_normalize_device_entry(entry::Symbol) = entry


"""
    device_transfer_arrays(cfg::SHTConfig, arrays...)

Transfer arrays to the device specified in the configuration.
This is a placeholder that will be properly implemented in the GPU extension.
"""
function device_transfer_arrays(cfg, arrays...)
    if cfg.compute_device == CPU
        return arrays  # No transfer needed for CPU
    else
        # This will be implemented in the GPU extension
        error("GPU extension required for device transfers. Load CUDA.jl so that SHTnsKitCUDAExt is activated.")
    end
end
