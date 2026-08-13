##########
# Strict MPI communication for GPU-backed PencilArrays
##########

# This file deliberately knows nothing about CUDA or AMDGPU.  The two compound
# package extensions register the small set of vendor operations needed here.
# Mathematical code calls `allreduce!`/`exchange!` and storage helpers only.

struct ParallelGPUAdapter{FM,FA,FD,FWD,FG,FS,FH,FHD,FDH}
    name::Symbol
    matches::FM
    array_type::FA
    device::FD
    with_device::FWD
    gpu_aware::FG
    synchronize::FS
    allocate_pinned::FH
    device_to_host!::FHD
    host_to_device!::FDH
end

ParallelGPUAdapter(name::Symbol, matches, array_type, device, gpu_aware,
                   synchronize, allocate_pinned, device_to_host!,
                   host_to_device!) = ParallelGPUAdapter(
    name, matches, array_type, device, (f, _device) -> f(), gpu_aware,
    synchronize, allocate_pinned, device_to_host!, host_to_device!,
)

const _PARALLEL_GPU_ADAPTERS = Dict{Symbol,WeakRef}()
const _PARALLEL_GPU_ADAPTER_LOCK = ReentrantLock()

function _register_parallel_gpu_adapter!(adapter::ParallelGPUAdapter)
    lock(_PARALLEL_GPU_ADAPTER_LOCK) do
        _PARALLEL_GPU_ADAPTERS[adapter.name] = WeakRef(adapter)
    end
    return adapter
end

function _parallel_gpu_adapters()
    snapshot = lock(_PARALLEL_GPU_ADAPTER_LOCK) do
        result = Pair{Symbol,Any}[]
        for (name, reference) in _PARALLEL_GPU_ADAPTERS
            value = reference.value
            value === nothing || push!(result, name => value)
        end
        result
    end
    return sort!(snapshot; by=first)
end

function _parallel_gpu_adapter(value)
    matches = Pair{Symbol,Any}[]
    for entry in _parallel_gpu_adapters()
        entry.second.matches(value) && push!(matches, entry)
    end
    isempty(matches) && return nothing
    length(matches) == 1 || throw(ArgumentError(
        "multiple MPI GPU adapters recognize $(typeof(value))",
    ))
    return only(matches).second
end

@inline _parallel_parent(value::PencilArray) = parent(value)
@inline _parallel_parent(value) = value

# Vendor device queries must inspect the allocation, not the task's current
# device. Peel the array wrappers that can sit between a PencilArray/view and
# the CUDA/ROCm allocation which owns its memory.
@inline _parallel_root_buffer(value::PencilArray) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::SubArray) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::Base.ReshapedArray) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::Base.ReinterpretArray) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::Base.LogicalIndex) =
    _parallel_root_buffer(getfield(value, :mask))
@inline _parallel_root_buffer(value::Base.PermutedDimsArray) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Adjoint) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Transpose) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Symmetric) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Hermitian) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Diagonal) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Bidiagonal) =
    _parallel_root_buffer(getfield(value, :dv))
@inline _parallel_root_buffer(value::LinearAlgebra.Tridiagonal) =
    _parallel_root_buffer(getfield(value, :d))
@inline _parallel_root_buffer(value::LinearAlgebra.UpperTriangular) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.LowerTriangular) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.UnitUpperTriangular) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.UnitLowerTriangular) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value) = value

function _parallel_array_type(prototype)
    adapter = _parallel_gpu_adapter(_parallel_parent(prototype))
    return adapter === nothing ? Array : adapter.array_type(prototype)
end

function _parallel_pencil(pen::Pencil, prototype)
    array_type = _parallel_array_type(prototype)
    return PencilArrays.typeof_array(pen) === array_type ? pen : similar(pen, array_type)
end

function _parallel_allocate(pen::Pencil, ::Type{T}, prototype,
                            extra_dims::Vararg{Integer}) where {T}
    return PencilArray{T}(undef, _parallel_pencil(pen, prototype), extra_dims...)
end

mutable struct _GPUAwarenessEntry
    comm::WeakRef
    lock::ReentrantLock
    ready::Bool
    supported::Bool
end

mutable struct _GPUStagingEntry
    root_owner::WeakRef
    logical_id::UInt
    signature::Tuple
    buffer::Any
    lock::ReentrantLock
    tick::UInt64
end

const _GPU_AWARENESS = Dict{Tuple,_GPUAwarenessEntry}()
const _GPU_STAGING = Dict{Tuple,_GPUStagingEntry}()
const _GPU_AWARENESS_LOCK = ReentrantLock()
const _GPU_STAGING_LOCK = ReentrantLock()
const _GPU_STAGING_LIMIT = Ref(8)
const _GPU_CACHE_TICK = Ref(UInt64(0))

const _GPU_DIRECT_CALLS = Threads.Atomic{Int}(0)
const _GPU_STAGED_CALLS = Threads.Atomic{Int}(0)
const _GPU_DIRECT_BYTES = Threads.Atomic{Int}(0)
const _GPU_STAGED_BYTES = Threads.Atomic{Int}(0)

@inline _communication_bytes(value) = sizeof(eltype(value)) * length(value)

function parallel_gpu_stats()
    return (
        direct_calls=_GPU_DIRECT_CALLS[],
        staged_calls=_GPU_STAGED_CALLS[],
        direct_bytes=_GPU_DIRECT_BYTES[],
        staged_bytes=_GPU_STAGED_BYTES[],
    )
end

function parallel_gpu_cache_sizes()
    awareness = lock(_GPU_AWARENESS_LOCK) do
        count(entry -> entry.comm.value !== nothing, values(_GPU_AWARENESS))
    end
    staging = lock(_GPU_STAGING_LOCK) do
        count(entry -> entry.root_owner.value !== nothing, values(_GPU_STAGING))
    end
    return (; awareness, staging)
end

function parallel_gpu_cache_limit!(limit::Integer)
    limit >= 1 || throw(ArgumentError("MPI GPU staging cache limit must be positive"))
    previous = _GPU_STAGING_LIMIT[]
    _GPU_STAGING_LIMIT[] = Int(limit)
    return previous
end

function parallel_gpu_clear_caches!()
    lock(_GPU_AWARENESS_LOCK) do
        empty!(_GPU_AWARENESS)
    end
    lock(_GPU_STAGING_LOCK) do
        empty!(_GPU_STAGING)
    end
    _GPU_DIRECT_CALLS[] = 0
    _GPU_STAGED_CALLS[] = 0
    _GPU_DIRECT_BYTES[] = 0
    _GPU_STAGED_BYTES[] = 0
    return nothing
end

function _gpu_awareness(adapter::ParallelGPUAdapter, comm, buffer)
    device = adapter.device(buffer)
    key = (objectid(comm), adapter.name, device)
    entry = lock(_GPU_AWARENESS_LOCK) do
        current = get(_GPU_AWARENESS, key, nothing)
        if current === nothing || current.comm.value !== comm
            current = _GPUAwarenessEntry(WeakRef(comm), ReentrantLock(), false, false)
            _GPU_AWARENESS[key] = current
        end
        current
    end
    # Detection may call into MPI/vendor code, so it runs outside the registry
    # lock.  The entry lock ensures one completed publication per key.
    return lock(entry.lock) do
        if !entry.ready
            entry.supported = Bool(adapter.gpu_aware(comm))
            entry.ready = true
        end
        entry.supported
    end
end

function _staging_entry(adapter::ParallelGPUAdapter, comm, owner, n::Int)
    root_owner = _parallel_root_buffer(owner)
    logical_id = objectid(owner)
    device = adapter.device(root_owner)
    signature = (objectid(comm), adapter.name, device, eltype(owner), n)
    key = (logical_id, objectid(root_owner), signature)

    found = lock(_GPU_STAGING_LOCK) do
        filter!(pair -> pair.second.root_owner.value !== nothing, _GPU_STAGING)
        entry = get(_GPU_STAGING, key, nothing)
        if entry !== nothing && entry.logical_id == logical_id &&
           entry.root_owner.value === root_owner
            _GPU_CACHE_TICK[] += 1
            entry.tick = _GPU_CACHE_TICK[]
            return entry
        end
        nothing
    end
    found === nothing || return found

    # Pinning/allocating can enter the vendor runtime and must not run under a
    # global cache lock.
    buffer = adapter.allocate_pinned(eltype(owner), n)
    candidate = _GPUStagingEntry(
        WeakRef(root_owner), logical_id, signature, buffer,
        ReentrantLock(), UInt64(0),
    )
    return lock(_GPU_STAGING_LOCK) do
        existing = get(_GPU_STAGING, key, nothing)
        if existing !== nothing && existing.logical_id == logical_id &&
           existing.root_owner.value === root_owner
            return existing
        end
        filter!(pair -> pair.second.root_owner.value !== nothing, _GPU_STAGING)
        limit = _GPU_STAGING_LIMIT[]
        if length(_GPU_STAGING) >= limit
            # Do not evict a live owner's entry: another task may already have
            # looked it up and be waiting on its per-entry lock. Replacing it
            # would create two independent staging buffers/locks for one device
            # allocation. Use this candidate for the current call without
            # caching it; the registry remains strictly bounded.
            return candidate
        end
        _GPU_CACHE_TICK[] += 1
        candidate.tick = _GPU_CACHE_TICK[]
        _GPU_STAGING[key] = candidate
        candidate
    end
end

function _with_staging(f, adapter::ParallelGPUAdapter, comm, owner)
    entry = _staging_entry(adapter, comm, owner, length(owner))
    # The per-entry lock serializes reuse. It is intentionally not a global
    # cache/registry lock, so synchronization and MPI callbacks cannot deadlock
    # registration or unrelated buffers/devices.
    return lock(entry.lock) do
        f(entry.buffer)
    end
end

function _host_staging_value(adapter::ParallelGPUAdapter, comm, value)
    raw = _parallel_parent(value)
    adapter.matches(raw) || return (value, nothing)
    entry = _staging_entry(adapter, comm, raw, length(raw))
    host = if value isa PencilArray
        host_pen = similar(pencil(value), Array)
        PencilArray(host_pen, reshape(entry.buffer, size(raw)))
    else
        reshape(entry.buffer, size(raw))
    end
    return host, entry
end

function _with_entry_locks(f, entries, index::Int=1)
    index > length(entries) && return f()
    return lock(entries[index].lock) do
        _with_entry_locks(f, entries, index + 1)
    end
end

function _device_result(adapter::ParallelGPUAdapter, prototype, result,
                        staged=(), originals=())
    for (pair, original) in zip(staged, originals)
        result === first(pair) && return original
    end
    if result isa PencilArray
        device = adapter.device(_parallel_root_buffer(prototype))
        return adapter.with_device(device) do
            array_type = adapter.array_type(prototype)
            device_pen = similar(pencil(result), array_type)
            output = PencilArray{eltype(result)}(
                undef, device_pen, PencilArrays.extra_dims(result)...,
            )
            adapter.host_to_device!(parent(output), parent(result))
            adapter.synchronize(parent(output))
            output
        end
    elseif result isa AbstractArray
        device = adapter.device(_parallel_root_buffer(prototype))
        return adapter.with_device(device) do
            array_type = adapter.array_type(prototype)
            output = array_type{eltype(result)}(undef, size(result))
            adapter.host_to_device!(output, result)
            adapter.synchronize(output)
            output
        end
    elseif result isa Tuple
        return map(value -> _device_result(
            adapter, prototype, value, staged, originals,
        ), result)
    elseif result isa NamedTuple
        return map(value -> _device_result(
            adapter, prototype, value, staged, originals,
        ), result)
    end
    return result
end

"""
    _staged_gpu_call(adapter, operation, comm, f, values...)

Execute an existing CPU Pencil algorithm through bounded pinned snapshots.  All
GPU inputs are copied before `f` runs and every array result is restored to the
same vendor before return.  This is the only permitted host-staging boundary
for ordinary MPI mathematical entry points.
"""
function _staged_gpu_call(adapter::ParallelGPUAdapter, operation::Symbol,
                          comm, f, values...;
                          mutated::Tuple=(), validate_storage::Bool=true)
    validate_storage && _validate_parallel_storage!(comm, operation, values...)
    all(index -> index in eachindex(values), mutated) || throw(ArgumentError(
        "$operation staged mutation index is out of bounds",
    ))
    staged = map(value -> _host_staging_value(adapter, comm, value), values)
    hosts = first.(staged)
    entries = unique(last(pair) for pair in staged if last(pair) !== nothing)
    sort!(entries; by=objectid)
    isempty(entries) && throw(ArgumentError(
        "$operation staged GPU call received no device-backed value",
    ))
    prototype = _parallel_parent(first(value for value in values
                                       if adapter.matches(_parallel_parent(value))))
    bytes = sum(_communication_bytes(_parallel_parent(value)) for value in values
                if adapter.matches(_parallel_parent(value)))
    result = _with_entry_locks(entries) do
        for (value, pair) in zip(values, staged)
            entry = last(pair)
            entry === nothing && continue
            raw = _parallel_parent(value)
            adapter.synchronize(raw)
            adapter.device_to_host!(entry.buffer, raw)
            adapter.synchronize(raw)
        end
        cpu_result = f(hosts...)
        for index in mutated
            entry = last(staged[index])
            entry === nothing && continue
            raw = _parallel_parent(values[index])
            adapter.host_to_device!(raw, entry.buffer)
            adapter.synchronize(raw)
        end
        _device_result(adapter, prototype, cpu_result, staged, values)
    end
    Threads.atomic_add!(_GPU_STAGED_CALLS, 1)
    Threads.atomic_add!(_GPU_STAGED_BYTES, bytes)
    return result
end

"""
    allreduce!(buffer, op, comm)

All-reduce `buffer` in place. GPU buffers use direct device communication only
when the registered vendor and MPI implementation report support; otherwise a
bounded reusable pinned-host stage is used.
"""
function allreduce!(buffer, op, comm;
                    adapter=_parallel_gpu_adapter(buffer),
                    collective=MPI.Allreduce!)
    adapter === nothing && return collective(buffer, op, comm)
    bytes = _communication_bytes(buffer)
    if _gpu_awareness(adapter, comm, buffer)
        adapter.synchronize(buffer)
        result = collective(buffer, op, comm)
        adapter.synchronize(buffer)
        Threads.atomic_add!(_GPU_DIRECT_CALLS, 1)
        Threads.atomic_add!(_GPU_DIRECT_BYTES, bytes)
        return result
    end
    result = _with_staging(adapter, comm, buffer) do host
        adapter.synchronize(buffer)
        adapter.device_to_host!(host, buffer)
        adapter.synchronize(buffer)
        collective(host, op, comm)
        adapter.host_to_device!(buffer, host)
        adapter.synchronize(buffer)
        buffer
    end
    Threads.atomic_add!(_GPU_STAGED_CALLS, 1)
    Threads.atomic_add!(_GPU_STAGED_BYTES, bytes)
    return result
end

"""
    exchange!(send, receive, comm)

Vendor-neutral two-buffer exchange used by transpose/redistribution paths.
The callback has signature `(send, receive, comm)`; callers may supply the
exact MPI collective required by their layout.
"""
function exchange!(send, receive, comm;
                   adapter=_parallel_gpu_adapter(send),
                   collective=MPI.Alltoall!)
    receive_adapter = _parallel_gpu_adapter(receive)
    if adapter === nothing && receive_adapter === nothing
        return collective(send, receive, comm)
    end
    adapter !== nothing && adapter.matches(send) && adapter.matches(receive) &&
        (receive_adapter === nothing || adapter.name === receive_adapter.name) ||
        throw(ArgumentError(
        "MPI exchange send/receive buffers must use the same GPU vendor",
    ))
    bytes = _communication_bytes(send) + _communication_bytes(receive)
    if _gpu_awareness(adapter, comm, send)
        adapter.synchronize(send)
        result = collective(send, receive, comm)
        adapter.synchronize(receive)
        Threads.atomic_add!(_GPU_DIRECT_CALLS, 1)
        Threads.atomic_add!(_GPU_DIRECT_BYTES, bytes)
        return result
    end
    send_entry = _staging_entry(adapter, comm, send, length(send))
    receive_entry = _staging_entry(adapter, comm, receive, length(receive))
    send_entry === receive_entry && throw(ArgumentError(
        "MPI exchange send/receive buffers may not alias",
    ))
    # Lock in object-id order to keep simultaneous exchanges deadlock-free.
    first_entry, second_entry = objectid(send_entry) < objectid(receive_entry) ?
        (send_entry, receive_entry) : (receive_entry, send_entry)
    result = lock(first_entry.lock) do
        lock(second_entry.lock) do
            send_host = send_entry.buffer
            receive_host = receive_entry.buffer
            adapter.synchronize(send)
            adapter.device_to_host!(send_host, send)
            adapter.synchronize(send)
            collective(send_host, receive_host, comm)
            adapter.host_to_device!(receive, receive_host)
            adapter.synchronize(receive)
            receive
        end
    end
    Threads.atomic_add!(_GPU_STAGED_CALLS, 1)
    Threads.atomic_add!(_GPU_STAGED_BYTES, bytes)
    return result
end

function _parallel_storage_code(value)
    adapter = _parallel_gpu_adapter(_parallel_parent(value))
    adapter === nothing && return 0
    adapter.name === :cuda && return 1
    adapter.name === :amdgpu && return 2
    return 3
end

"""Validate vendor/residency collectively before allocation or mutation."""
function _validate_parallel_storage!(comm, operation::Symbol, values...)
    flags = UInt32(0)
    codes = map(_parallel_storage_code, values)
    local_min = isempty(codes) ? 0 : minimum(codes)
    local_max = isempty(codes) ? 0 : maximum(codes)
    local_min == local_max || (flags |= 0x20000)
    global_min = MPI.Allreduce(local_min, min, comm)
    global_max = MPI.Allreduce(local_max, max, comm)
    global_min == global_max || (flags |= 0x20000)
    flags == 0 || throw(ArgumentError(
        "$operation collective validation failed: storage/vendor mismatch",
    ))
    return local_min
end

function _dist_transpose_gpu_analysis!(adapter, plan, output, input)
    return _dist_transpose_gpu_analysis!(Val(adapter.name), plan, output, input)
end

function _dist_transpose_gpu_synthesis!(adapter, plan, output, input)
    return _dist_transpose_gpu_synthesis!(Val(adapter.name), plan, output, input)
end


function _dist_transpose_gpu_vector_analysis!(adapter, plan, Sout, Tout, Vt, Vp)
    return _dist_transpose_gpu_vector_analysis!(
        Val(adapter.name), plan, Sout, Tout, Vt, Vp,
    )
end

function _dist_transpose_gpu_vector_synthesis!(adapter, plan, Vt, Vp, Sin, Tin)
    return _dist_transpose_gpu_vector_synthesis!(
        Val(adapter.name), plan, Vt, Vp, Sin, Tin,
    )
end

_missing_parallel_gpu_operation(name, operation) = throw(ArgumentError(
    "MPI GPU adapter $name does not implement $operation",
))
_dist_transpose_gpu_analysis!(::Val{name}, args...) where {name} =
    _missing_parallel_gpu_operation(name, :dist_analysis)
_dist_transpose_gpu_synthesis!(::Val{name}, args...) where {name} =
    _missing_parallel_gpu_operation(name, :dist_synthesis)
_dist_transpose_gpu_vector_analysis!(::Val{name}, args...) where {name} =
    _missing_parallel_gpu_operation(name, :dist_analysis_sphtor)
_dist_transpose_gpu_vector_synthesis!(::Val{name}, args...) where {name} =
    _missing_parallel_gpu_operation(name, :dist_synthesis_sphtor)
