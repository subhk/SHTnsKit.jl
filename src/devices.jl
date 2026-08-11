# Typed compute-device markers shared by the core and optional GPU extension.

"""Abstract supertype for compute devices supported by SHTnsKit."""
abstract type ComputeDevice end

"""CPU compute device."""
struct CPU <: ComputeDevice end

"""GPU compute device (currently provided by the CUDA extension)."""
struct GPU <: ComputeDevice end
