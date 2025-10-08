"""
Device utilities for GPU-accelerated spherical harmonic transforms.

Thin wrappers that keep the core package agnostic of specific GPU backends.
"""

"""
    device_transfer_arrays(cfg::SHTConfig, arrays...)

Transfer arrays to the device specified in the configuration. This is
implemented by GPU extensions; the base package only provides the CPU path.
"""
function device_transfer_arrays(cfg, arrays...)
    if cfg.compute_device == CPU
        return arrays  # No transfer needed for CPU
    else
        error("GPU extension required for device transfers. Load CUDA.jl so that SHTnsKitCUDAExt is activated.")
    end
end

