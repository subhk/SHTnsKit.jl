# Strict typed device selection and vendor-neutral GPU adapter routing.

const _DEVICE_STATE = Ref{Union{Nothing,ComputeDevice}}(nothing)

# Extension modules own adapter objects strongly. Core retains only weak
# references, keyed by stable vendor names and traversed in sorted order.
const _GPU_ADAPTERS = Dict{Symbol,WeakRef}()

function _register_gpu_adapter!(name::Symbol, adapter)
    _GPU_ADAPTERS[name] = WeakRef(adapter)
    return adapter
end

function _registered_gpu_adapters()
    adapters = Pair{Symbol,Any}[]
    for name in sort!(collect(keys(_GPU_ADAPTERS)))
        adapter = _GPU_ADAPTERS[name].value
        adapter === nothing || push!(adapters, name => adapter)
    end
    return adapters
end

# Adapter protocol. Vendor extensions add methods without introducing package
# types or vendor symbols into core.
_gpu_adapter_functional(adapter) = false
_gpu_adapter_matches(adapter, value) = false

function _gpu_adapter_adapt(adapter, value)
    throw(BackendUnavailableError(:to_device, "the selected GPU adapter cannot place this value"))
end

function _gpu_adapter_analysis(adapter, cfg, field; kwargs...)
    throw(BackendUnavailableError(:analysis, "the selected GPU adapter does not yet implement scalar analysis"))
end

function _gpu_adapter_synthesis(adapter, cfg, coefficients; kwargs...)
    throw(BackendUnavailableError(:synthesis, "the selected GPU adapter does not yet implement scalar synthesis"))
end

function _functional_gpu_adapters()
    return filter(_registered_gpu_adapters()) do entry
        try
            Bool(_gpu_adapter_functional(entry.second))
        catch
            false
        end
    end
end

function _no_gpu_error(operation::Symbol)
    return BackendUnavailableError(
        operation,
        "no loaded GPU adapter is functional; load CUDA.jl or AMDGPU.jl and verify its runtime",
    )
end

function _gpu_adapter(prototype=nothing; operation::Symbol=:gpu)
    if prototype !== nothing && on_device(prototype) isa GPU
        matches = filter(_registered_gpu_adapters()) do entry
            _gpu_adapter_matches(entry.second, prototype)
        end
        isempty(matches) && throw(BackendUnavailableError(
            operation,
            "no loaded GPU adapter recognizes prototype $(typeof(prototype))",
        ))
        length(matches) == 1 || throw(ArgumentError(
            "multiple GPU adapters recognize prototype $(typeof(prototype)) for `$operation`",
        ))
        name, adapter = only(matches)
        _gpu_adapter_functional(adapter) || throw(BackendUnavailableError(
            operation,
            "$name recognizes $(typeof(prototype)) but its runtime is not functional",
        ))
        return adapter
    elseif prototype !== nothing && !(on_device(prototype) isa CPU)
        throw(ArgumentError("prototype for `$operation` must be a CPU or GPU array"))
    end

    functional = _functional_gpu_adapters()
    isempty(functional) && throw(_no_gpu_error(operation))
    length(functional) == 1 && return only(functional).second
    names = join(first.(functional), ", ")
    throw(ArgumentError(
        "multiple functional GPU adapters are loaded ($names) for `$operation`; pass a device-array prototype",
    ))
end

"""Return the preferred compute device without silently relabelling requests."""
function get_device()
    requested = _DEVICE_STATE[]
    if requested isa GPU
        isempty(_functional_gpu_adapters()) && throw(_no_gpu_error(:get_device))
        return requested
    elseif requested isa CPU
        return requested
    end
    return isempty(_functional_gpu_adapters()) ? CPU() : GPU()
end

"""Select `CPU()` or `GPU()` as the preferred compute device."""
function set_device!(device::ComputeDevice)
    device isa GPU && isempty(_functional_gpu_adapters()) && throw(_no_gpu_error(:set_device!))
    _DEVICE_STATE[] = device
    return device
end

to_device(::CPU, value) = _to_cpu(value)

function to_device(::GPU, value)
    adapter = _gpu_adapter(on_device(value) isa GPU ? value : nothing; operation=:to_device)
    return _gpu_adapter_matches(adapter, value) ? value : _gpu_adapter_adapt(adapter, value)
end

function to_device(::GPU, value, prototype)
    on_device(prototype) isa GPU || throw(ArgumentError(
        "GPU placement prototype must already be a device array, got $(typeof(prototype))",
    ))
    adapter = _gpu_adapter(prototype; operation=:to_device)
    if on_device(value) isa GPU && !_gpu_adapter_matches(adapter, value)
        throw(ArgumentError("value and GPU prototype use different vendors"))
    end
    return _gpu_adapter_matches(adapter, value) ? value : _gpu_adapter_adapt(adapter, value)
end

# Preserve the original value-first order and add the prototype form.
to_device(value::AbstractArray, device::ComputeDevice) = to_device(device, value)
to_device(value::AbstractArray, device::ComputeDevice, prototype) = to_device(device, value, prototype)
to_device(value) = to_device(get_device(), value)

_to_cpu(value) = value
_to_cpu(value::Array) = value
_to_cpu(value::AbstractArray) = Array(value)

"""Return `CPU()` for ordinary arrays; vendor extensions specialize this."""
on_device(::AbstractArray) = CPU()

function _require_cpu_storage(operation::Symbol, value)
    on_device(value) isa CPU && return nothing
    throw(ArgumentError(
        "`$operation(CPU(), ...)` requires CPU storage; explicitly call to_device(CPU(), value) first",
    ))
end

function analysis(::CPU, cfg::SHTConfig, field::AbstractMatrix; kwargs...)
    _require_cpu_storage(:analysis, field)
    return analysis(cfg, field; kwargs...)
end

function synthesis(::CPU, cfg::SHTConfig, coefficients::AbstractMatrix; kwargs...)
    _require_cpu_storage(:synthesis, coefficients)
    return synthesis(cfg, coefficients; kwargs...)
end

function analysis(::GPU, cfg::SHTConfig, field::AbstractMatrix; prototype=nothing, kwargs...)
    selection = prototype === nothing && on_device(field) isa GPU ? field : prototype
    adapter = _gpu_adapter(selection; operation=:analysis)
    if on_device(field) isa GPU && !_gpu_adapter_matches(adapter, field)
        throw(ArgumentError("analysis input and GPU prototype use different vendors"))
    end
    device_field = _gpu_adapter_matches(adapter, field) ? field : _gpu_adapter_adapt(adapter, field)
    return _gpu_adapter_analysis(adapter, cfg, device_field; kwargs...)
end

function synthesis(::GPU, cfg::SHTConfig, coefficients::AbstractMatrix; prototype=nothing, kwargs...)
    selection = prototype === nothing && on_device(coefficients) isa GPU ? coefficients : prototype
    adapter = _gpu_adapter(selection; operation=:synthesis)
    if on_device(coefficients) isa GPU && !_gpu_adapter_matches(adapter, coefficients)
        throw(ArgumentError("synthesis input and GPU prototype use different vendors"))
    end
    device_coefficients = _gpu_adapter_matches(adapter, coefficients) ? coefficients : _gpu_adapter_adapt(adapter, coefficients)
    return _gpu_adapter_synthesis(adapter, cfg, device_coefficients; kwargs...)
end
