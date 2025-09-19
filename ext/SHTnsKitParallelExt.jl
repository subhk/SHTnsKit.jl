module SHTnsKitParallelExt

"""
SHTnsKit Parallel Extension

This Julia package extension provides distributed/parallel spherical harmonic transform
capabilities using MPI for inter-process communication and PencilArrays for distributed
memory management. The extension is automatically loaded when both MPI.jl and 
PencilArrays.jl are available in the environment.

Key capabilities:
- Distributed spherical harmonic transforms across MPI processes
- Pencil decomposition for memory-distributed arrays (latitude/longitude/spectral)
- Parallel FFTs via PencilFFTs for longitude direction transforms
- Load balancing and data redistribution for optimal performance
- Caching of FFT plans for repeated operations (optional via environment variable)
"""

using Base.Threads                       # Threads.@threads and locks/macros
import MPI                               # Bring MPI module into scope for MPI.* calls
using MPI: Allreduce, Allreduce!, Allgather, Allgatherv, Comm_size, COMM_WORLD
import PencilArrays                      # Bring PencilArrays module for qualified calls
using PencilArrays: Pencil, PencilArray  # Distributed array framework
using PencilFFTs                         # Distributed FFTs
using SHTnsKit                           # Core spherical harmonic functionality

# ===== FFT PLAN CACHING =====
# Optional plan caching to avoid repeated planning overhead in performance-critical code
# Enable via: ENV["SHTNSKIT_CACHE_PENCILFFTS"] = "1"
const _CACHE_PENCILFFTS = Ref{Bool}(get(ENV, "SHTNSKIT_CACHE_PENCILFFTS", "0") == "1")

# Thread-safe cache storage for FFT plans indexed by array characteristics
const _pfft_cache = IdDict{Any,Any}()
const _cache_lock = Threads.ReentrantLock()
const _sparse_gather_cache = Dict{Tuple{DataType,Int}, NamedTuple{(:idx,:val),Tuple{Vector{Int},Any}}}()

# Compat helper: `ceildiv` was added in Julia 1.11
const _ceildiv = isdefined(Base, :ceildiv) ? Base.ceildiv : (a, b) -> cld(a, b)
ceildiv(a::Integer, b::Integer) = _ceildiv(a, b)

function SHTnsKit.fft_plan_cache_enabled()
    return _CACHE_PENCILFFTS[]
end
Base.@doc """
    SHTnsKit.fft_plan_cache_enabled() -> Bool

Return whether distributed FFT plan caching is currently enabled.
""" SHTnsKit.fft_plan_cache_enabled

function SHTnsKit.set_fft_plan_cache!(flag::Bool; clear::Bool=true)
    _CACHE_PENCILFFTS[] = flag
    if !flag && clear
        lock(_cache_lock) do
            empty!(_pfft_cache)
        end
    end
    return flag
end
Base.@doc """
    SHTnsKit.set_fft_plan_cache!(flag::Bool; clear::Bool=true)

Enable or disable caching of PencilFFT plans. When disabling and `clear=true`, any
cached plans are freed immediately to release memory.
""" SHTnsKit.set_fft_plan_cache!

function SHTnsKit.enable_fft_plan_cache!()
    return SHTnsKit.set_fft_plan_cache!(true)
end
Base.@doc """
    SHTnsKit.enable_fft_plan_cache!()

Convenience wrapper to enable distributed FFT plan caching.
""" SHTnsKit.enable_fft_plan_cache!

function SHTnsKit.disable_fft_plan_cache!(; clear::Bool=true)
    return SHTnsKit.set_fft_plan_cache!(false; clear)
end
Base.@doc """
    SHTnsKit.disable_fft_plan_cache!(; clear::Bool=true)

Disable distributed FFT plan caching. Pass `clear=false` to keep previously cached
plans in memory for later reuse.
""" SHTnsKit.disable_fft_plan_cache!

# Generate cache key based on array characteristics for FFT plan reuse
function _cache_key(kind::Symbol, A)
    # Basic array characteristics
    base_key = (kind, size(A,1), size(A,2), eltype(A))
    
    # Add communicator size with robust error handling
    comm_size = try
        MPI.Comm_size(communicator(A))
    catch
        1  # Default to single process
    end
    
    # Add decomposition hash with multiple fallback patterns
    decomp_hash = try
        hash(A.pencil.decomposition)
    catch
        try
            # Alternative field access patterns
            pencil = getfield(A, :pencil)
            if hasfield(typeof(pencil), :decomposition)
                hash(getfield(pencil, :decomposition))
            elseif hasfield(typeof(pencil), :plan)
                hash(getfield(pencil, :plan))
            else
                hash(size(A))  # Use array size as fallback identifier
            end
        catch
            hash(size(A))  # Ultimate fallback
        end
    end
    
    return (base_key..., comm_size, decomp_hash)
end

function _get_or_plan(kind::Symbol, A)
    # If caching disabled, create plan directly without storing
    if !_CACHE_PENCILFFTS[]
        return kind === :fft  ? plan_fft(A; dims=2) :     # Forward FFT along longitude (dim 2)
               kind === :ifft ? plan_fft(A; dims=2) :     # Inverse FFT along longitude  
               kind === :rfft ? (try plan_rfft(A; dims=2) catch; nothing end) :   # Real-to-complex FFT
               kind === :irfft ? (try plan_irfft(A; dims=2) catch; nothing end) : # Complex-to-real IFFT
               error("unknown plan kind")
    end
    
    # Thread-safe caching with optimized lookup
    key = _cache_key(kind, A)
    
    # Fast path: check without lock first (common case)
    if haskey(_pfft_cache, key)
        return _pfft_cache[key]  # Return cached plan
    end
    
    # Slow path: thread-safe plan creation and caching
    return lock(_cache_lock) do
        # Double-check pattern: another thread might have created the plan
        if haskey(_pfft_cache, key)
            return _pfft_cache[key]
        end
        
        # Create new plan and cache it for future use
        plan = kind === :fft  ? plan_fft(A; dims=2) :     # Forward FFT along longitude
               kind === :ifft ? plan_fft(A; dims=2) :     # Inverse FFT along longitude
               kind === :rfft ? (try plan_rfft(A; dims=2) catch; nothing end) :   # Real-to-complex FFT
               kind === :irfft ? (try plan_irfft(A; dims=2) catch; nothing end) : # Complex-to-real IFFT
               error("unknown plan kind")
        
        _pfft_cache[key] = plan
        return plan
    end
end


# ===== VERSION-AGNOSTIC COMPATIBILITY LAYER =====
# Robust API compatibility patterns for PencilArrays across versions
# Uses method-based dispatch and try-catch for maximum reliability

# Diagnostic function to detect PencilArrays version and capabilities
function _detect_pencilarray_version()
    version_info = Dict{Symbol, Any}()
    
    # Check for modern API (v0.17+)
    version_info[:has_communicator] = hasmethod(PencilArrays.communicator, Tuple{Any})
    version_info[:has_allocate] = hasmethod(PencilArrays.allocate, Tuple{Any, Any})
    version_info[:has_globalindices] = hasmethod(PencilArrays.globalindices, Tuple{Any, Any})
    
    # Check for legacy API patterns  
    version_info[:has_comm] = isdefined(PencilArrays, :comm)
    version_info[:has_global_indices] = isdefined(PencilArrays, :global_indices)
    version_info[:has_pencilarray_constructor] = isdefined(PencilArrays, :PencilArray)
    
    # Try to determine approximate version based on API availability
    if version_info[:has_communicator] && version_info[:has_allocate] && version_info[:has_globalindices]
        version_info[:estimated_version] = "v0.17+"
    elseif version_info[:has_comm] || version_info[:has_global_indices]
        version_info[:estimated_version] = "v0.15-v0.16"
    else
        version_info[:estimated_version] = "unknown/very_old"
    end
    
    return version_info
end

# Get version info (cached for performance)
const _PENCILARRAY_VERSION_INFO = Ref{Union{Nothing, Dict{Symbol, Any}}}(nothing)
function pencilarray_version_info()
    if _PENCILARRAY_VERSION_INFO[] === nothing
        _PENCILARRAY_VERSION_INFO[] = _detect_pencilarray_version()
    end
    return _PENCILARRAY_VERSION_INFO[]
end

# Get MPI communicator from PencilArray with robust fallback chain
function communicator(A)
    # Modern PencilArrays API (v0.17+)
    if hasmethod(PencilArrays.communicator, (typeof(A),))
        return PencilArrays.communicator(A)
    end
    
    # Legacy API patterns (v0.15-v0.16)
    if hasmethod(PencilArrays.comm, (typeof(A),))
        return PencilArrays.comm(A)
    end
    
    # Direct field access fallback (very old versions)
    try
        return A.pencil.comm
    catch
        # Last resort: try global communicator access patterns
        if hasfield(typeof(A), :pencil)
            pencil = getfield(A, :pencil)
            if hasfield(typeof(pencil), :comm)
                return getfield(pencil, :comm)
            elseif hasfield(typeof(pencil), :communicator)
                return getfield(pencil, :communicator)
            end
        end
    end
    
    error("Unable to extract MPI communicator from PencilArray. "
          * "This may indicate an incompatible PencilArrays version.")
end

# Allocate PencilArray with robust version detection
function allocate(args...; kwargs...)
    # Attempt direct call and fall back only on MethodError originating from allocate
    try
        return PencilArrays.allocate(args...; kwargs...)
    catch err
        if !(err isa MethodError && err.f === PencilArrays.allocate)
            rethrow(err)
        end
    end

    nargs = length(args)
    if nargs >= 2
        T, pencil = args[1], args[2]
        if isa(pencil, PencilArrays.Pencil)
            kw = Dict{Symbol,Any}(kwargs)
            kw[:eltype] = get(kw, :eltype, T)
            return PencilArrays.zeros(pencil; (;kw...)...)
        end
    end

    error("Unable to allocate PencilArray with provided arguments. " *
          "This may indicate an incompatible PencilArrays version or API change.")
end

"""
    SHTnsKit.suggest_pencil_grid(comm::MPI.Comm, nlat::Integer, nlon::Integer;
                                 prefer_square::Bool=true, allow_one_dim::Bool=true)

Suggest a `(pθ, pφ)` processor grid for a Pencil decomposition that splits the
θ and φ dimensions across the MPI communicator `comm`. When `prefer_square` is
true (default), the heuristic favours balanced local tiles; otherwise it
minimises the total local grid area. Set `allow_one_dim=false` to require both
factors to be greater than one (falls back to one-dimensional decomposition when
no such factorisation exists).
"""
function SHTnsKit.suggest_pencil_grid(comm::MPI.Comm, nlat::Integer, nlon::Integer;
                                      prefer_square::Bool=true, allow_one_dim::Bool=true)
    total = MPI.Comm_size(comm)
    best = (1, total)
    best_cost = typemax(Float64)
    for pθ in 1:total
        total % pθ == 0 || continue
        pφ = div(total, pθ)
        if !allow_one_dim && (pθ == 1 || pφ == 1)
            continue
        end
        lθ = ceildiv(Int(nlat), pθ)
        lφ = ceildiv(Int(nlon), pφ)
        cost = if prefer_square
            abs(lθ - lφ)
        else
            lθ * lφ
        end
        if cost < best_cost || (cost == best_cost && max(pθ, pφ) < max(best...))
            best_cost = cost
            best = (pθ, pφ)
        end
    end
    if !allow_one_dim && best == (1, total) && total > 1
        @warn "Unable to find a 2D pencil decomposition for $total processes; falling back to 1D split"
    end
    return best
end

# Get global indices with robust fallback patterns
function globalindices(A, dim)
    # Modern PencilArrays API (v0.17+)
    if hasmethod(PencilArrays.globalindices, (typeof(A), typeof(dim)))
        return PencilArrays.globalindices(A, dim)
    end
    
    # Legacy API patterns (v0.15-v0.16)
    if hasmethod(PencilArrays.global_indices, (typeof(A), typeof(dim)))
        return PencilArrays.global_indices(A, dim)
    end
    
    # Direct field access patterns
    try
        return A.pencil.axes[dim]
    catch
        # Alternative field access patterns
        try
            pencil = getfield(A, :pencil)
            if hasfield(typeof(pencil), :axes)
                axes = getfield(pencil, :axes)
                return axes[dim]
            elseif hasfield(typeof(pencil), :global_axes)
                global_axes = getfield(pencil, :global_axes)
                return global_axes[dim]
            end
        catch
            # Last resort: try to reconstruct from size information
            if hasmethod(size, (typeof(A),)) && hasmethod(PencilArrays.size_global, (typeof(A),))
                local_size = size(A)
                global_size = PencilArrays.size_global(A)
                if dim <= length(global_size)
                    # This is a rough approximation - may not be accurate for all decompositions
                    return 1:global_size[dim]
                end
            end
        end
    end
    
    error("Unable to get global indices for dimension $dim from PencilArray. "
          * "This may indicate an incompatible PencilArrays version.")
end

# ===== DISTRIBUTED FFT WRAPPERS =====
# Robust wrapper functions for PencilFFTs with version compatibility
# Uses method existence checks for maximum reliability

# Plan forward FFT with robust API detection
function plan_fft(A; dims=:)
    if hasmethod(PencilFFTs.plan_fft, (typeof(A),))
        return PencilFFTs.plan_fft(A; dims=dims)
    else
        error("PencilFFTs.plan_fft not available for array type $(typeof(A))")
    end
end

# Execute forward FFT
function fft(A, p)
    if hasmethod(PencilFFTs.fft, (typeof(A), typeof(p)))
        return PencilFFTs.fft(A, p)
    else
        error("PencilFFTs.fft not available for arguments $(typeof(A)), $(typeof(p))")
    end
end

# Execute inverse FFT
function ifft(A, p)
    if hasmethod(PencilFFTs.ifft, (typeof(A), typeof(p)))
        return PencilFFTs.ifft(A, p)
    else
        error("PencilFFTs.ifft not available for arguments $(typeof(A)), $(typeof(p))")
    end
end

# Plan real-to-complex FFT with error handling
function plan_rfft(A; dims=:)
    if hasmethod(PencilFFTs.plan_rfft, (typeof(A),))
        return PencilFFTs.plan_rfft(A; dims=dims)
    else
        return nothing  # Graceful fallback for unsupported operations
    end
end

# Plan complex-to-real inverse FFT
function plan_irfft(A; dims=:)
    if hasmethod(PencilFFTs.plan_irfft, (typeof(A),))
        return PencilFFTs.plan_irfft(A; dims=dims)
    else
        return nothing  # Graceful fallback for unsupported operations
    end
end

# Execute real-to-complex FFT
function rfft(A, p)
    if hasmethod(PencilFFTs.rfft, (typeof(A), typeof(p)))
        return PencilFFTs.rfft(A, p)
    else
        error("PencilFFTs.rfft not available for arguments $(typeof(A)), $(typeof(p))")
    end
end

# Execute complex-to-real inverse FFT
function irfft(A, p)
    if hasmethod(PencilFFTs.irfft, (typeof(A), typeof(p)))
        return PencilFFTs.irfft(A, p)
    else
        error("PencilFFTs.irfft not available for arguments $(typeof(A)), $(typeof(p))")
    end
end


# ===== PARALLEL EXTENSION MODULES =====
# Include specialized modules for different aspects of parallel spherical harmonic transforms
include("ParallelDiagnostics.jl")      # Diagnostic and profiling tools for parallel operations
include("ParallelDispatch.jl")         # Function dispatch and interface definitions  
include("ParallelPlans.jl")            # Distributed transform planning and setup
include("ParallelTransforms.jl")       # Core parallel transform implementations
include("ParallelOpsPencil.jl")       # Parallel differential operators using PencilArrays
include("ParallelRotationsPencil.jl") # Parallel spherical rotation operations
include("ParallelLocal.jl")            # Local (per-process) operations and utilities

# Optimized communication patterns for large spectral arrays
function efficient_spectral_reduce!(local_data::AbstractMatrix, comm)
    # Delegate to adaptive strategy which selects among sparse, segmented,
    # hierarchical, or dense reductions based on data/process characteristics.
    return adaptive_spectral_communication!(local_data, comm; operation=+)
end

function efficient_spectral_reduce!(local_data::AbstractVector, comm)
    # Vector specialization using the adaptive strategy as well.
    return adaptive_spectral_communication!(local_data, comm; operation=+)
end

function hierarchical_spectral_reduce!(local_data::AbstractMatrix, comm, ppn::Int)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Level 1: Intra-node reduction (shared memory optimization)
    node_id = rank ÷ ppn
    local_rank = rank % ppn
    
    # Create intra-node communicator for shared-memory optimization
    node_comm = MPI.Comm_split(comm, node_id, local_rank)
    node_nprocs = MPI.Comm_size(node_comm)
    
    if node_nprocs > 1
        # Reduce within each compute node using optimized shared-memory path
        MPI.Allreduce!(local_data, +, node_comm)
    end
    
    # Level 2: Inter-node reduction (network-aware)
    if local_rank == 0  # Node representatives
        inter_node_comm = MPI.Comm_split(comm, 0, node_id)
        inter_nprocs = MPI.Comm_size(inter_node_comm)
        
        if inter_nprocs > 1
            # Use tree-based reduction between nodes for network efficiency
            tree_reduce!(local_data, inter_node_comm)
        end
        
        MPI.Comm_free(inter_node_comm)
    else
        # Create dummy communicator for non-representatives
        MPI.Comm_split(comm, 1, 0)
    end
    
    # Level 3: Broadcast results back within nodes
    if node_nprocs > 1
        MPI.Bcast!(local_data, 0, node_comm)
    end
    
    MPI.Comm_free(node_comm)
end

function tree_reduce!(data::AbstractMatrix, comm)
    # Optimized binary tree reduction for inter-node communication
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    temp_buf = similar(data)
    
    # Up-sweep: reduce up the tree
    step = 1
    while step < nprocs
        if rank % (2 * step) == 0 && rank + step < nprocs
            # Receive and accumulate from partner
            MPI.Recv!(temp_buf, rank + step, step, comm)
            data .+= temp_buf
        elseif (rank - step) % (2 * step) == 0 && rank >= step
            # Send to parent and exit
            MPI.Send(data, rank - step, step, comm)
            return
        end
        step *= 2
    end
    
    # Down-sweep: broadcast final result
    step = prevpow(2, nprocs - 1)
    while step >= 1
        if rank % (2 * step) == 0 && rank + step < nprocs
            # Send result to child
            MPI.Send(data, rank + step, -step, comm)
        elseif (rank - step) % (2 * step) == 0 && rank >= step
            # Receive final result from parent
            MPI.Recv!(data, rank - step, -step, comm)
        end
        step ÷= 2
    end
end

function sparse_spectral_reduce!(local_data::AbstractVector{T}, comm) where {T}
    # True sparse reduction using Allgatherv of (indices, values)
    nz_idx = findall(!iszero, local_data)
    send_count = length(nz_idx)

    # Gather counts to all ranks and build displacements
    counts = Allgather(send_count, comm)
    displs = cumsum([0; counts[1:end-1]])
    total = sum(counts)

    # Prepare send buffers
    idx_send = collect(Int, nz_idx)
    val_send = local_data[nz_idx]

    # Receive buffers (reused across calls when possible)
    key = (T, hash(comm))
    idx_recv, val_recv = lock(_cache_lock) do
        if haskey(_sparse_gather_cache, key)
            buf = _sparse_gather_cache[key]
            idx_buf = buf.idx
            val_buf = buf.val
            if length(idx_buf) < total
                resize!(idx_buf, total)
            end
            if length(val_buf) < total
                resize!(val_buf, total)
            end
            (idx_buf, val_buf)
        else
            idx_buf = Vector{Int}(undef, total)
            val_buf = Vector{T}(undef, total)
            _sparse_gather_cache[key] = (idx=idx_buf, val=val_buf)
            (idx_buf, val_buf)
        end
    end

    # Exchange indices and values
    Allgatherv(idx_send, idx_recv, counts, displs, comm)
    Allgatherv(val_send, val_recv, counts, displs, comm)

    # Accumulate into the full dense vector
    fill!(local_data, zero(T))
    @inbounds for i in 1:total
        local_data[idx_recv[i]] += val_recv[i]
    end
    return local_data
end

function sparse_spectral_reduce!(local_data::AbstractMatrix{T}, comm) where {T}
    # Flattened sparse reduction on matrix storage
    A = vec(local_data)
    sparse_spectral_reduce!(A, comm)
    return local_data
end

# ===== ADVANCED COMMUNICATION PATTERNS =====

"""
    adaptive_spectral_communication!(data, comm; operation, sparse_threshold=0.1)

Adaptive communication pattern that automatically chooses the optimal strategy
based on data sparsity and process count for spherical harmonic coefficients.

Strategies:
- Dense data + few processes: Standard Allreduce
- Dense data + many processes: Hierarchical reduction  
- Sparse data: Sparse coefficient exchange
- Very large data: Segmented reduction with overlap
"""
function adaptive_spectral_communication!(data::AbstractArray, comm; operation=+, sparse_threshold=nothing)
    nprocs = MPI.Comm_size(comm)
    data_size = length(data)
    # Resolve sparsity threshold from ENV if not provided
    st = sparse_threshold === nothing ? try parse(Float64, get(ENV, "SHTNSKIT_SPARSE_THRESHOLD", "0.1")) catch; 0.1 end : sparse_threshold
    
    # Compute sparsity ratio
    nonzero_count = count(!iszero, data)
    sparsity = nonzero_count / data_size
    
    if sparsity < st && data_size > 1000
        # Use sparse communication for very sparse data (robust fallback)
        return sparse_spectral_reduce!(data, comm)
    elseif nprocs > 64 && data_size > 50000
        # Use segmented reduction for large-scale problems
        return segmented_spectral_reduce!(data, comm, operation)
    elseif nprocs > 16 && data_size > 5000
        # Use hierarchical reduction for medium-scale problems
        if isa(data, AbstractMatrix)
            return hierarchical_spectral_reduce!(data, comm, min(nprocs, 32))
        else
            return hierarchical_spectral_reduce_vector!(data, comm, min(nprocs, 32))
        end
    else
        # Use standard Allreduce for small problems
        MPI.Allreduce!(data, operation, comm)
        return data
    end
end

"""
    segmented_spectral_reduce!(data, comm, operation)

Segmented reduction for very large spectral arrays that don't fit in memory.
Processes data in chunks with communication/computation overlap.
"""
function segmented_spectral_reduce!(data::AbstractArray, comm, operation)
    nprocs = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    
    # Determine optimal segment size based on available memory
    max_segment_mb = parse(Int, get(ENV, "SHTNSKIT_MAX_SEGMENT_MB", "128"))
    bytes_per_element = sizeof(eltype(data))
    max_elements_per_segment = (max_segment_mb * 1024 * 1024) ÷ bytes_per_element
    
    segment_size = min(length(data), max_elements_per_segment)
    num_segments = (length(data) + segment_size - 1) ÷ segment_size
    
    # Process segments with overlapped communication (double-buffered)
    temp_a = similar(data, segment_size)
    temp_b = similar(data, segment_size)
    prev_req = nothing
    prev_buf = nothing  # :a or :b
    prev_len = 0
    prev_start = 0
    
    for seg in 1:num_segments
        start_idx = (seg - 1) * segment_size + 1
        end_idx = min(seg * segment_size, length(data))
        segment_data = view(data, start_idx:end_idx)
        cur_len = length(segment_data)
        
        # Choose buffer not in use by previous outstanding request
        buf_sym = (prev_buf == :a) ? :b : :a
        temp_view = buf_sym === :a ? view(temp_a, 1:cur_len) : view(temp_b, 1:cur_len)
        copyto!(temp_view, segment_data)
        
        # Launch nonblocking reduction for current segment
        req = MPI.Iallreduce!(temp_view, operation, comm)
        
        # Complete previous request and write back
        if prev_req !== nothing
            MPI.Wait(prev_req)
            # previous segment range
            pstart = prev_start
            pend = min(pstart + prev_len - 1, length(data))
            prev_data = view(data, pstart:pend)
            prev_view = (prev_buf === :a ? view(temp_a, 1:prev_len) : view(temp_b, 1:prev_len))
            copyto!(prev_data, prev_view)
        end
        
        # Promote current to previous
        prev_req = req
        prev_buf = buf_sym
        prev_len = cur_len
        prev_start = start_idx
    end
    
    # Final pending segment
    if prev_req !== nothing
        MPI.Wait(prev_req)
        pstart = prev_start
        pend = min(pstart + prev_len - 1, length(data))
        prev_data = view(data, pstart:pend)
        prev_view = (prev_buf === :a ? view(temp_a, 1:prev_len) : view(temp_b, 1:prev_len))
        copyto!(prev_data, prev_view)
    end
    
    return data
end

"""
    hierarchical_spectral_reduce_vector!(data, comm, ppn)

Vector-specialized hierarchical reduction with optimized memory access patterns.
"""
function hierarchical_spectral_reduce_vector!(data::AbstractVector, comm, ppn::Int)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Create node-local communicator
    node_id = rank ÷ ppn
    local_rank = rank % ppn
    node_comm = MPI.Comm_split(comm, node_id, local_rank)
    node_nprocs = MPI.Comm_size(node_comm)
    
    # Intra-node reduction with memory-friendly chunking
    if node_nprocs > 1
        chunk_size = min(length(data), 10000)  # Process in chunks for cache efficiency
        
        for start_idx in 1:chunk_size:length(data)
            end_idx = min(start_idx + chunk_size - 1, length(data))
            chunk_view = view(data, start_idx:end_idx)
            MPI.Allreduce!(chunk_view, +, node_comm)
        end
    end
    
    # Inter-node reduction (only node leaders participate)
    if local_rank == 0
        inter_node_comm = MPI.Comm_split(comm, 0, node_id)
        inter_nprocs = MPI.Comm_size(inter_node_comm)
        
        if inter_nprocs > 1
            # Use tree reduction for better network scaling
            tree_reduce_vector!(data, inter_node_comm)
        end
        
        MPI.Comm_free(inter_node_comm)
    else
        MPI.Comm_split(comm, 1, 0)  # Non-leaders create dummy communicator
    end
    
    # Broadcast results within nodes
    if node_nprocs > 1
        MPI.Bcast!(data, 0, node_comm)
    end
    
    MPI.Comm_free(node_comm)
    return data
end

"""
    tree_reduce_vector!(data, comm)

Binary tree reduction optimized for vector data with better memory locality.
"""
function tree_reduce_vector!(data::AbstractVector, comm)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Use chunked processing for better cache behavior
    chunk_size = min(length(data), 5000)
    temp_buffer = Vector{eltype(data)}(undef, chunk_size)
    
    # Up-sweep phase
    step = 1
    while step < nprocs
        if rank % (2 * step) == 0 && rank + step < nprocs
            # Receive and accumulate from partner in chunks
            for start_idx in 1:chunk_size:length(data)
                end_idx = min(start_idx + chunk_size - 1, length(data))
                chunk_view = view(data, start_idx:end_idx)
                buffer_view = view(temp_buffer, 1:(end_idx - start_idx + 1))
                
                MPI.Recv!(buffer_view, rank + step, step, comm)
                chunk_view .+= buffer_view
            end
        elseif (rank - step) % (2 * step) == 0 && rank >= step
            # Send to parent in chunks and exit
            for start_idx in 1:chunk_size:length(data)
                end_idx = min(start_idx + chunk_size - 1, length(data))
                chunk_view = view(data, start_idx:end_idx)
                MPI.Send(chunk_view, rank - step, step, comm)
            end
            return
        end
        step *= 2
    end
    
    # Down-sweep phase (broadcast final result)
    step = prevpow(2, nprocs - 1)
    while step >= 1
        if rank % (2 * step) == 0 && rank + step < nprocs
            # Send result to child in chunks
            for start_idx in 1:chunk_size:length(data)
                end_idx = min(start_idx + chunk_size - 1, length(data))
                chunk_view = view(data, start_idx:end_idx)
                MPI.Send(chunk_view, rank + step, -step, comm)
            end
        elseif (rank - step) % (2 * step) == 0 && rank >= step
            # Receive final result from parent in chunks
            for start_idx in 1:chunk_size:length(data)
                end_idx = min(start_idx + chunk_size - 1, length(data))
                chunk_view = view(data, start_idx:end_idx)
                MPI.Recv!(chunk_view, rank - step, -step, comm)
            end
        end
        step ÷= 2
    end
end

"""
    bandwidth_aware_broadcast!(data, root, comm)

Bandwidth-aware broadcasting that adapts to network topology and data size.
Uses pipeline broadcasting for large data and tree broadcasting for small data.
"""
function bandwidth_aware_broadcast!(data::AbstractArray, root::Int, comm)
    nprocs = MPI.Comm_size(comm)
    data_size_mb = (sizeof(data)) / (1024 * 1024)
    
    if nprocs > 32 && data_size_mb > 10.0
        # Use pipeline broadcast for large data on large clusters
        pipeline_broadcast!(data, root, comm)
    else
        # Use standard tree broadcast for smaller cases
        MPI.Bcast!(data, root, comm)
    end
    
    return data
end

"""
    pipeline_broadcast!(data, root, comm)

Pipeline broadcast that overlaps communication with local copying for better bandwidth utilization.
"""
function pipeline_broadcast!(data::AbstractArray, root::Int, comm)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Determine pipeline parameters
    pipeline_stages = min(nprocs, 8)  # Limit pipeline depth
    chunk_size = max(1, length(data) ÷ pipeline_stages)
    
    for stage in 1:pipeline_stages
        start_idx = (stage - 1) * chunk_size + 1
        end_idx = stage == pipeline_stages ? length(data) : stage * chunk_size
        chunk_view = view(data, start_idx:end_idx)
        
        # Pipeline broadcast of this chunk
        MPI.Bcast!(chunk_view, root, comm)
    end
    
    return data
end

# Note: Avoid forwarding Base.zeros(Pencil) to PencilArrays.zeros to prevent
# potential recursion when PencilArrays.zeros may call Base.zeros internally.

# ===== MIGRATION NOTES =====
# The previous version compatibility shims (prior to this update) used fragile patterns:
# - Simple isdefined() checks that could miss method signature changes
# - Direct field access without proper error handling  
# - Hardcoded fallback paths that assumed specific internal structures
# 
# The new robust patterns provide:
# - Method-based existence checking with proper type signatures
# - Multi-level fallback chains with comprehensive error handling
# - Version detection utilities for debugging compatibility issues
# - Graceful degradation rather than hard failures where possible
#
# This should maintain compatibility across PencilArrays v0.15+ while being
# more resilient to API changes in future versions.

end # module SHTnsKitParallelExt
