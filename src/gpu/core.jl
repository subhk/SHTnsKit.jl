"""
Common GPU helpers shared between CUDA extensions and the core package.

The actual kernels live in extension packages (e.g. `SHTnsKitCUDAExt`). These
utility functions live in the main package so both CPU and GPU code paths can
share device-selection logic without introducing heavy dependencies.
"""

function _normalize_device_preferences(entries::Vector{Union{Symbol,Device}})
    devices = Device[]
    backends = Symbol[]

    for entry in entries
        device = entry isa Device ? entry : device_from_symbol(entry)
        push!(devices, device)

        backend = entry isa Device ? device_symbol(entry) : _normalize_device_entry(entry)
        if device == GPU && backend in (:cpu, :gpu)
            backend = :cuda
        end
        push!(backends, backend)
    end

    return devices, backends
end

function _resolve_device_choice(choice::Union{Symbol,Device}, backend_pref::Vector{Symbol})
    if choice isa Symbol && choice === :auto
        return select_compute_device(backend_pref)
    end

    device = choice isa Device ? choice : device_from_symbol(choice)
    candidate_backend = choice isa Symbol ? _normalize_device_entry(choice) : (device == CPU ? :cpu : _first_gpu_backend(backend_pref))
    backend = _resolve_backend(candidate_backend, backend_pref, device)
    available = device == GPU ? (backend == :cuda && cuda_available()) : false

    return device, backend, available
end

function _resolve_backend(candidate::Symbol, backend_pref::Vector{Symbol}, device::Device)
    if device == CPU
        return :cpu
    end

    if candidate == :cuda
        return :cuda
    elseif candidate == :gpu || candidate == :cpu
        return _first_gpu_backend(backend_pref)
    else
        return candidate
    end
end

function _first_gpu_backend(backend_pref::Vector{Symbol})
    for backend in backend_pref
        backend != :cpu && return backend
    end
    return :cuda
end

function _promote_device_preference(device::Device, existing::Vector{Device})
    new_pref = Device[device]
    for entry in existing
        entry == device && continue
        push!(new_pref, entry)
    end
    return new_pref
end

function _promote_backend_preference(backend::Symbol, existing::Vector{Symbol})
    new_pref = Symbol[backend]
    for entry in existing
        entry == backend && continue
        push!(new_pref, entry)
    end
    return new_pref
end

