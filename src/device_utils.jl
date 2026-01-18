#=
================================================================================
Device Utilities for SHTnsKit.jl
================================================================================

This module provides transparent backend switching between CPU and GPU for
spherical harmonic transforms. Users can:

1. Query available devices
2. Set a preferred backend (CPU or GPU)
3. Let the system automatically select the best available backend
4. Transfer arrays between devices transparently

USAGE
-----
```julia
using SHTnsKit

# Check what's available
println(available_backends())  # [:cpu] or [:cpu, :gpu]

# Set preferred backend
set_backend!(:gpu)   # Use GPU if available
set_backend!(:cpu)   # Force CPU

# Get current backend
println(current_backend())  # :cpu or :gpu

# Auto-select best backend
set_backend!(:auto)  # Picks GPU if available, else CPU

# Create config - uses current backend automatically
cfg = create_gauss_config(64, 66)

# Transforms use the configured backend transparently
alm = analysis(cfg, spatial_data)
```

DEVICE TYPES
------------
- :cpu  - Standard CPU computation (always available)
- :gpu  - GPU acceleration via CUDA.jl (requires CUDA.jl and compatible NVIDIA GPU)
- :auto - Automatically select best available backend

================================================================================
=#

# ============================================================================
# Device Types
# ============================================================================

"""
    SHTBackend

Enum representing supported compute backends for SHTnsKit operations.

# Values
- `CPU`: Standard CPU computation (always available)
- `GPU`: GPU acceleration (requires CUDA.jl for NVIDIA GPUs)
"""
@enum SHTBackend begin
    CPU
    GPU
end

# Convert Symbol to SHTBackend
function _symbol_to_backend(s::Symbol)::SHTBackend
    s == :cpu && return CPU
    s == :gpu && return GPU
    throw(ArgumentError("Unknown backend: $s. Valid options: :cpu, :gpu"))
end

# Convert SHTBackend to Symbol
function _backend_to_symbol(b::SHTBackend)::Symbol
    b == CPU && return :cpu
    b == GPU && return :gpu
end

# ============================================================================
# Global Backend State
# ============================================================================

# Global state for backend preference
const _BACKEND_STATE = Ref{Symbol}(:auto)
const _CUDA_CHECKED = Ref{Bool}(false)
const _CUDA_AVAILABLE = Ref{Bool}(false)

"""
    _check_cuda_available() -> Bool

Internal function to check if CUDA is available. Result is cached.
"""
function _check_cuda_available()
    if !_CUDA_CHECKED[]
        _CUDA_CHECKED[] = true
        # Check if CUDA extension is loaded by testing if gpu_analysis works
        _CUDA_AVAILABLE[] = try
            # The GPU extension sets this when loaded
            isdefined(@__MODULE__, :_cuda_extension_loaded) && _cuda_extension_loaded[]
        catch
            false
        end
    end
    return _CUDA_AVAILABLE[]
end

# Flag set by GPU extension when loaded
const _cuda_extension_loaded = Ref{Bool}(false)

"""
    _notify_cuda_loaded!()

Called by the GPU extension to notify that CUDA is available.
"""
function _notify_cuda_loaded!()
    _CUDA_CHECKED[] = true
    _CUDA_AVAILABLE[] = true
    _cuda_extension_loaded[] = true
end

# ============================================================================
# Public API: Backend Management
# ============================================================================

"""
    available_backends() -> Vector{Symbol}

Return a list of available compute backends.

# Returns
- `Vector{Symbol}`: Available backends (always includes `:cpu`, may include `:gpu`)

# Examples
```julia
backends = available_backends()
# Returns [:cpu] or [:cpu, :gpu]
```
"""
function available_backends()
    backends = [:cpu]
    if _check_cuda_available()
        push!(backends, :gpu)
    end
    return backends
end

"""
    current_backend() -> Symbol

Return the currently active compute backend.

# Returns
- `:cpu` if using CPU computation
- `:gpu` if using CUDA GPU computation

# Examples
```julia
backend = current_backend()
println("Using: \$backend")
```
"""
function current_backend()
    pref = _BACKEND_STATE[]
    if pref == :auto
        # Auto-select: prefer CUDA if available
        return _check_cuda_available() ? :gpu : :cpu
    elseif pref == :gpu
        # User requested CUDA - verify it's available
        if !_check_cuda_available()
            @warn "CUDA requested but not available, falling back to CPU"
            return :cpu
        end
        return :gpu
    else
        return :cpu
    end
end

"""
    set_backend!(backend::Symbol)

Set the preferred compute backend for SHTnsKit operations.

# Arguments
- `backend`: One of `:cpu`, `:gpu`, or `:auto`
  - `:cpu` - Force CPU computation
  - `:gpu` - Use CUDA GPU (falls back to CPU if unavailable)
  - `:auto` - Automatically select best available backend

# Examples
```julia
set_backend!(:gpu)  # Prefer GPU
set_backend!(:cpu)   # Force CPU
set_backend!(:auto)  # Auto-select (default)
```
"""
function set_backend!(backend::Symbol)
    if backend ∉ [:cpu, :gpu, :auto]
        throw(ArgumentError("Invalid backend: $backend. Valid options: :cpu, :gpu, :auto"))
    end
    if backend == :gpu && !_check_cuda_available()
        @warn "CUDA requested but not available. Will fall back to CPU when needed."
    end
    _BACKEND_STATE[] = backend
    return current_backend()
end

"""
    use_gpu() -> Bool

Check if GPU acceleration is currently enabled and available.

# Returns
- `true` if CUDA backend is active
- `false` if using CPU

# Examples
```julia
if use_gpu()
    println("GPU acceleration enabled")
else
    println("Using CPU")
end
```
"""
function use_gpu()
    return current_backend() == :gpu
end

"""
    with_backend(f, backend::Symbol)

Execute function `f` with a temporarily changed backend, then restore the original.

# Arguments
- `f`: Function to execute
- `backend`: Backend to use (`:cpu`, `:gpu`, or `:auto`)

# Examples
```julia
# Temporarily force CPU for debugging
result = with_backend(:cpu) do
    analysis(cfg, data)
end
```
"""
function with_backend(f, backend::Symbol)
    old_backend = _BACKEND_STATE[]
    try
        set_backend!(backend)
        return f()
    finally
        _BACKEND_STATE[] = old_backend
    end
end

# ============================================================================
# Device Selection Helpers
# ============================================================================

"""
    select_compute_device(preference_order=[:gpu, :cpu])

Select the best available compute device based on preference order.

# Arguments
- `preference_order`: Vector of preferred devices in order

# Returns
- `(device::Symbol, gpu_available::Bool)`: Selected device and GPU availability

# Examples
```julia
device, gpu_ok = select_compute_device()  # Prefers CUDA
device, gpu_ok = select_compute_device([:cpu])  # Force CPU
```
"""
function select_compute_device(preference_order::Vector{Symbol}=[:gpu, :cpu])
    available = available_backends()

    for device in preference_order
        if device ∈ available
            return device, (device == :gpu)
        end
    end

    return :cpu, false
end

# ============================================================================
# Array Transfer Utilities
# ============================================================================

"""
    to_device(arr, backend::Symbol=current_backend())

Transfer array to the specified compute device.

# Arguments
- `arr`: Array to transfer
- `backend`: Target backend (`:cpu` or `:gpu`)

# Returns
- Array on the target device

# Examples
```julia
gpu_arr = to_device(cpu_arr, :gpu)
cpu_arr = to_device(gpu_arr, :cpu)
```
"""
function to_device(arr::AbstractArray, backend::Symbol=current_backend())
    if backend == :cpu
        return _to_cpu(arr)
    elseif backend == :gpu
        return _to_gpu(arr)
    else
        throw(ArgumentError("Unknown backend: $backend"))
    end
end

# CPU transfer - always available
function _to_cpu(arr::AbstractArray)
    return Array(arr)
end

# GPU transfer - stub, overridden by GPU extension
function _to_gpu(arr::AbstractArray)
    if !_check_cuda_available()
        error("CUDA not available. Load CUDA.jl to enable GPU support.")
    end
    # This will be overridden by the GPU extension
    error("GPU extension not properly loaded")
end

"""
    on_device(arr) -> Symbol

Determine which device an array is currently on.

# Returns
- `:cpu` for standard Julia arrays
- `:gpu` for CUDA arrays

# Examples
```julia
println(on_device(rand(10)))  # :cpu
```
"""
function on_device(arr::AbstractArray)
    # Default: CPU. GPU extension overrides for CuArray
    return :cpu
end

"""
    device_transfer_arrays(target_backend::Symbol, arrays...)

Transfer multiple arrays to the target device.

# Arguments
- `target_backend`: Target backend (`:cpu` or `:gpu`)
- `arrays...`: Arrays to transfer

# Returns
- Tuple of transferred arrays
"""
function device_transfer_arrays(target_backend::Symbol, arrays...)
    return Tuple(to_device(arr, target_backend) for arr in arrays)
end

# Legacy compatibility
function device_transfer_arrays(cfg, arrays...)
    backend = hasfield(typeof(cfg), :compute_device) ? cfg.compute_device : current_backend()
    return device_transfer_arrays(backend, arrays...)
end

# ============================================================================
# Backend Dispatch Helpers
# ============================================================================

"""
    dispatch_to_backend(cpu_func, gpu_func, args...; kwargs...)

Dispatch to either CPU or GPU function based on current backend.

# Arguments
- `cpu_func`: Function to call for CPU backend
- `gpu_func`: Function to call for GPU backend
- `args...`: Arguments to pass to the function
- `kwargs...`: Keyword arguments to pass to the function

# Examples
```julia
result = dispatch_to_backend(cpu_analysis, gpu_analysis, cfg, data)
```
"""
function dispatch_to_backend(cpu_func, gpu_func, args...; kwargs...)
    if use_gpu()
        return gpu_func(args...; kwargs...)
    else
        return cpu_func(args...; kwargs...)
    end
end

"""
    @dispatch_backend cpu_expr gpu_expr

Macro for backend dispatch. Evaluates cpu_expr or gpu_expr based on current backend.

# Examples
```julia
result = @dispatch_backend analysis(cfg, data) gpu_analysis(cfg, data)
```
"""
macro dispatch_backend(cpu_expr, gpu_expr)
    quote
        if use_gpu()
            $(esc(gpu_expr))
        else
            $(esc(cpu_expr))
        end
    end
end

# ============================================================================
# Device Information
# ============================================================================

"""
    device_info() -> NamedTuple

Get information about the current compute device.

# Returns
NamedTuple with fields:
- `backend`: Current backend (`:cpu` or `:gpu`)
- `available_backends`: List of available backends
- `gpu_available`: Whether CUDA is available
- `details`: Backend-specific details (extended by GPU extension)
"""
function device_info()
    backend = current_backend()
    return (
        backend = backend,
        available_backends = available_backends(),
        gpu_available = _check_cuda_available(),
        details = _get_device_details(backend)
    )
end

# CPU details
function _get_device_details(::Val{:cpu})
    return (
        device_type = :cpu,
        threads = Threads.nthreads(),
        simd_available = true
    )
end

_get_device_details(backend::Symbol) = _get_device_details(Val(backend))

# GPU details - stub, overridden by GPU extension
function _get_device_details(::Val{:gpu})
    if !_check_cuda_available()
        return (device_type = :gpu, available = false)
    end
    # This will be extended by GPU extension
    return (device_type = :gpu, available = true, details = "Load CUDA.jl for full details")
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    ensure_backend_initialized()

Ensure the backend is properly initialized. Call this before intensive computations.
"""
function ensure_backend_initialized()
    backend = current_backend()
    if backend == :gpu
        _ensure_cuda_initialized()
    end
    return backend
end

# Stub - overridden by GPU extension
function _ensure_cuda_initialized()
    # GPU extension will implement proper initialization
end

"""
    reset_backend!()

Reset backend to auto-selection mode.
"""
function reset_backend!()
    _BACKEND_STATE[] = :auto
    return current_backend()
end

# ============================================================================
# Exports (handled in main module)
# ============================================================================

# Note: Exports are defined in SHTnsKit.jl main module
