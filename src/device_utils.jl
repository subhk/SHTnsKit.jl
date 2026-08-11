# Minimal typed device selection and array placement.

const _DEVICE_STATE = Ref{Union{Nothing,ComputeDevice}}(nothing)
const _CUDA_CHECKED = Ref(false)
const _CUDA_AVAILABLE = Ref(false)
const _cuda_extension_loaded = Ref(false)

function _check_cuda_available()
    if !_CUDA_CHECKED[]
        _CUDA_CHECKED[] = true
        _CUDA_AVAILABLE[] = _cuda_extension_loaded[]
    end
    return _CUDA_AVAILABLE[]
end

function _notify_cuda_loaded!()
    _CUDA_CHECKED[] = true
    _CUDA_AVAILABLE[] = true
    _cuda_extension_loaded[] = true
    return nothing
end

"""Return the active compute device, automatically preferring a functional GPU."""
function get_device()
    requested = _DEVICE_STATE[]
    requested === nothing && return _check_cuda_available() ? GPU() : CPU()
    requested isa GPU && !_check_cuda_available() && return CPU()
    return requested
end

"""Select `CPU()` or `GPU()` as the preferred compute device."""
function set_device!(device::ComputeDevice)
    if device isa GPU && !_check_cuda_available()
        @warn "CUDA requested but not available; using CPU until CUDA is functional"
    end
    _DEVICE_STATE[] = device
    return get_device()
end

to_device(arr::AbstractArray, ::CPU) = _to_cpu(arr)
to_device(arr::AbstractArray, ::GPU) = _to_gpu(arr)
to_device(arr::AbstractArray) = to_device(arr, get_device())

_to_cpu(arr::Array) = arr
_to_cpu(arr::AbstractArray) = Array(arr)

function _to_gpu(arr::AbstractArray)
    ext = Base.get_extension(@__MODULE__, :SHTnsKitGPUExt)
    ext === nothing && error("CUDA not available. Load CUDA.jl to enable GPU support.")
    return ext._to_gpu_impl(arr)
end

"""Return `CPU()` for ordinary arrays; the CUDA extension specializes this."""
on_device(::AbstractArray) = CPU()
