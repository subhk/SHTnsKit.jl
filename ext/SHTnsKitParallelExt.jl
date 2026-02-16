module SHTnsKitParallelExt

#=
================================================================================
SHTnsKit Parallel Extension - MPI-based Distributed Spherical Harmonic Transforms
================================================================================

This Julia package extension provides distributed/parallel spherical harmonic transform
capabilities using MPI for inter-process communication and PencilArrays for distributed
memory management. The extension is automatically loaded when both MPI.jl and
PencilArrays.jl are available in the environment.

ARCHITECTURE OVERVIEW
---------------------
The parallel transforms use a "pencil decomposition" strategy:
1. Input data (θ,φ grid) is distributed across MPI ranks along one or both dimensions
2. FFT along longitude (φ) transforms spatial data to Fourier coefficients
3. Legendre integration along latitude (θ) produces spherical harmonic coefficients
4. MPI reductions combine partial results from all ranks

Data flow for dist_analysis (spatial → spectral):
    PencilArray(θ,φ) → [gather φ if distributed] → FFT(φ) → Legendre(θ) → Alm

Data flow for dist_synthesis (spectral → spatial):
    Alm → Legendre(θ) → IFFT(φ) → [scatter φ if distributed] → PencilArray(θ,φ)

KEY FILES
---------
- SHTnsKitParallelExt.jl: Module setup, FFT wrappers, utility functions
- ParallelTransforms.jl: Core distributed transform implementations
- ParallelPlans.jl: Pre-allocated buffer management for repeated transforms

PERFORMANCE NOTES
-----------------
1. Memory allocations: The main loop uses "function barriers" to ensure type stability
   and eliminate boxing allocations. See _analysis_loop_no_tables!() in ParallelTransforms.jl

2. FFT plans: FFTW internally caches plans for arrays of the same size/type, so
   repeated transforms on same-sized arrays are efficient after warmup.

3. MPI communication: φ-distributed data requires MPI.Allgatherv! per latitude row.
   For large problems, consider distributing along θ only to minimize communication.

DEBUGGING TIPS
--------------
1. Check extension loaded: `Base.get_extension(SHTnsKit, :SHTnsKitParallelExt) !== nothing`

2. Verify PencilArray layout:
   - `PencilArrays.range_local(pencil(arr))` shows which global indices this rank owns
   - `size(parent(arr))` shows local array dimensions
   - `PencilArrays.size_global(arr)` shows full global dimensions

3. Common issues:
   - "Unable to get global indices": PencilArrays version incompatibility
   - Hanging on MPI calls: Ensure all ranks call collective operations
   - Wrong results: Check that θ/φ ranges are correctly identified

4. Enable verbose output: `ENV["SHTNSKIT_VERBOSE_STORAGE"] = "1"`

5. Allocation profiling: Use @allocated or @timed to measure memory usage:
   ```julia
   @timed Alm = SHTnsKit.dist_analysis(cfg, fθφ)
   ```

ENVIRONMENT VARIABLES
--------------------
- SHTNSKIT_CACHE_PENCILFFTS: "1" (default) to cache FFT plans, "0" to disable
- SHTNSKIT_VERBOSE_STORAGE: "1" to print storage optimization info
================================================================================
=#

"""
    SHTnsKitParallelExt

Parallel extension module providing MPI-distributed spherical harmonic transforms.
See module-level comments for architecture overview and debugging tips.
"""

using Base.Threads                       # Threads.@threads and locks/macros
import MPI                               # Bring MPI module into scope for MPI.* calls
using MPI: Allreduce, Allreduce!, Allgather, Allgatherv!, VBuffer, Comm_size, COMM_WORLD
import PencilArrays                      # Bring PencilArrays module for qualified calls
using PencilArrays: Pencil, PencilArray, ManyPencilArray  # Distributed array framework
import PencilArrays: pencil, range_local, size_local, size_global, topology, parent
using PencilFFTs                         # Distributed FFTs
using PencilFFTs: Transforms, PencilFFTPlan, allocate_input, allocate_output
using FFTW                               # For 1D FFTs on local arrays
using SHTnsKit                           # Core spherical harmonic functionality

# ===== FFT PLAN CACHING =====
# Optional plan caching to avoid repeated planning overhead in performance-critical code
# Enabled by default; disable with ENV["SHTNSKIT_CACHE_PENCILFFTS"] = "0"
const _CACHE_PENCILFFTS = Ref{Bool}(get(ENV, "SHTNSKIT_CACHE_PENCILFFTS", "1") == "1")

# Thread-safe cache storage for FFT plans indexed by array characteristics
const _pfft_cache = IdDict{Any,Any}()
const _cache_lock = Threads.ReentrantLock()
const _sparse_gather_cache = Dict{Tuple{DataType,Int}, NamedTuple{(:idx,:val),Tuple{Vector{Int},Any}}}()

# Compat helper: `ceildiv` was added in Julia 1.11
const _ceildiv = isdefined(Base, :ceildiv) ? Base.ceildiv : (a, b) -> cld(a, b)
ceildiv(a::Integer, b::Integer) = _ceildiv(a, b)

function _fft_plan_cache_enabled_impl()
    return _CACHE_PENCILFFTS[]
end

function _fft_plan_cache_set_impl(flag::Bool; clear::Bool=true)
    _CACHE_PENCILFFTS[] = flag
    if !flag && clear
        lock(_cache_lock) do
            empty!(_pfft_cache)
        end
    end
    return flag
end

function _fft_plan_cache_enable_impl()
    return _fft_plan_cache_set_impl(true)
end

function _fft_plan_cache_disable_impl(; clear::Bool=true)
    return _fft_plan_cache_set_impl(false; clear=clear)
end

SHTnsKit._fft_plan_cache_enabled_cb[] = _fft_plan_cache_enabled_impl
SHTnsKit._fft_plan_cache_set_cb[] = _fft_plan_cache_set_impl
SHTnsKit._fft_plan_cache_enable_cb[] = _fft_plan_cache_enable_impl
SHTnsKit._fft_plan_cache_disable_cb[] = _fft_plan_cache_disable_impl

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
    
    # Thread-safe plan lookup and creation
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

# ===== PENCIL GRID SUGGESTION =====
@inline function _infer_comm_size(comm_or_nprocs::Any)
    comm_or_nprocs === nothing && return 1
    if comm_or_nprocs isa Integer
        return max(1, Int(comm_or_nprocs))
    elseif comm_or_nprocs isa MPI.Comm
        return MPI.Comm_size(comm_or_nprocs)
    end

    try
        return MPI.Comm_size(comm_or_nprocs)
    catch
    end

    for accessor in (:nprocs, :size, :length)
        try
            val = getproperty(comm_or_nprocs, accessor)
            val isa Integer && val > 0 && return Int(val)
        catch
        end
        try
            val = getfield(comm_or_nprocs, accessor)
            val isa Integer && val > 0 && return Int(val)
        catch
        end
    end

    return 1
end

@inline _candidate_remainder_penalty(total::Int, splits::Int) = (total % splits == 0 ? 0.0 : 0.3)

@inline function _candidate_score(nlat::Int, nlon::Int, p_theta::Int, p_phi::Int, prefer_square::Bool)
    theta_chunk = cld(nlat, p_theta)
    phi_chunk = cld(nlon, p_phi)

    chunk_penalty = (max(theta_chunk, phi_chunk) / max(1, min(theta_chunk, phi_chunk))) - 1.0
    shape_ratio = (max(p_theta, p_phi) / max(1, min(p_theta, p_phi))) - 1.0
    shape_penalty = prefer_square ? shape_ratio : 0.25 * shape_ratio

    lat_penalty = _candidate_remainder_penalty(nlat, p_theta)
    lon_penalty = _candidate_remainder_penalty(nlon, p_phi)

    thin_penalty = (theta_chunk < 2 ? 1.0 : 0.0) + (phi_chunk < 2 ? 1.0 : 0.0)

    grid_ratio = nlon == 0 ? 1.0 : float(nlat) / max(1.0, float(nlon))
    proc_ratio = p_phi == 0 ? 1.0 : float(p_theta) / max(1.0, float(p_phi))
    anisotropy_penalty = abs(proc_ratio - grid_ratio) / max(grid_ratio, 1.0)

    return chunk_penalty + shape_penalty + lat_penalty + lon_penalty + thin_penalty + 0.3 * anisotropy_penalty
end

function _suggest_pencil_grid_impl(comm_or_nprocs::Any, nlat::Integer, nlon::Integer;
                                   prefer_square::Bool=true,
                                   allow_one_dim::Bool=true)
    nlat_val = Int(nlat)
    nlon_val = Int(nlon)
    nlat_val > 0 || throw(ArgumentError("nlat must be positive"))
    nlon_val > 0 || throw(ArgumentError("nlon must be positive"))

    nprocs = _infer_comm_size(comm_or_nprocs)
    nprocs <= 1 && return (1, 1)

    best = nothing
    best_score = Inf

    limit = isqrt(nprocs)
    for p_theta in 1:limit
        nprocs % p_theta == 0 || continue
        p_phi = nprocs ÷ p_theta
        for (a, b) in ((p_theta, p_phi), (p_phi, p_theta))
            (a > 0 && b > 0) || continue
            if !allow_one_dim && min(a, b) == 1
                continue
            end
            if a > nlat_val && b > nlon_val
                continue
            end
            score = _candidate_score(nlat_val, nlon_val, a, b, prefer_square)
            if score < best_score - 1e-8
                best = (a, b)
                best_score = score
            elseif best !== nothing && abs(score - best_score) <= 1e-8
                spread = max(a, b) - min(a, b)
                best_spread = max(best...) - min(best...)
                if spread < best_spread
                    best = (a, b)
                elseif spread == best_spread && a >= b && best[1] < best[2]
                    best = (a, b)
                end
            end
        end
    end

    if best === nothing
        if allow_one_dim
            return nlon_val >= nlat_val ? (1, nprocs) : (nprocs, 1)
        else
            return (1, nprocs)
        end
    end

    return best
end

function __init__()
    SHTnsKit._suggest_pencil_grid_cb[] = _suggest_pencil_grid_impl
end

# Diagnostic function to detect PencilArrays version and capabilities
function _detect_pencilarray_version()
    version_info = Dict{Symbol, Any}()

    # Check for v0.19+ API (uses get_comm, range_local)
    version_info[:has_get_comm] = isdefined(PencilArrays, :get_comm)
    version_info[:has_range_local] = isdefined(PencilArrays, :range_local)
    version_info[:has_size_local] = isdefined(PencilArrays, :size_local)

    # Check for modern API (v0.17+)
    version_info[:has_communicator] = isdefined(PencilArrays, :communicator)
    version_info[:has_allocate] = isdefined(PencilArrays, :allocate)
    version_info[:has_globalindices] = isdefined(PencilArrays, :globalindices)

    # Check for legacy API patterns
    version_info[:has_comm] = isdefined(PencilArrays, :comm)
    version_info[:has_global_indices] = isdefined(PencilArrays, :global_indices)
    version_info[:has_pencilarray_constructor] = isdefined(PencilArrays, :PencilArray)

    # Try to determine approximate version based on API availability
    if version_info[:has_get_comm] && version_info[:has_range_local]
        version_info[:estimated_version] = "v0.19+"
    elseif version_info[:has_communicator] && version_info[:has_allocate] && version_info[:has_globalindices]
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
    # PencilArrays v0.19+ uses get_comm
    if hasmethod(PencilArrays.get_comm, (typeof(A),))
        return PencilArrays.get_comm(A)
    end

    # Modern PencilArrays API (v0.17+)
    if isdefined(PencilArrays, :communicator) && hasmethod(PencilArrays.communicator, (typeof(A),))
        return PencilArrays.communicator(A)
    end

    # Legacy API patterns (v0.15-v0.16)
    if isdefined(PencilArrays, :comm) && hasmethod(PencilArrays.comm, (typeof(A),))
        return PencilArrays.comm(A)
    end

    # Try get_comm on the pencil object
    try
        pen = pencil(A)
        if hasmethod(PencilArrays.get_comm, (typeof(pen),))
            return PencilArrays.get_comm(pen)
        end
    catch
    end

    # Direct field access fallback (very old versions)
    try
        return A.pencil.comm
    catch
        # Last resort: try global communicator access patterns
        if hasfield(typeof(A), :pencil)
            p = getfield(A, :pencil)
            if hasfield(typeof(p), :comm)
                return getfield(p, :comm)
            elseif hasfield(typeof(p), :communicator)
                return getfield(p, :communicator)
            end
        end
    end

    error("Unable to extract MPI communicator from PencilArray. "
          * "This may indicate an incompatible PencilArrays version.")
end

# Allocate PencilArray - simplified API for SHTnsKit needs
"""
    allocate(prototype::PencilArray; eltype=eltype(prototype)) -> PencilArray

Allocate a new PencilArray with the same decomposition as the prototype.
The optional `eltype` parameter allows changing the element type.

Note: The `dims` keyword from legacy API is ignored - decomposition is inherited from prototype.
"""
function allocate(prototype::PencilArray; dims=nothing, eltype::Type{T}=eltype(prototype)) where T
    # Get the pencil configuration from the prototype
    pen = pencil(prototype)
    # Allocate a new PencilArray with the same configuration
    return PencilArray{T}(undef, pen)
end

"""
    allocate(T::Type, pen::Pencil) -> PencilArray

Allocate a new PencilArray with the specified type and pencil configuration.
"""
function allocate(::Type{T}, pen::Pencil) where T
    return PencilArray{T}(undef, pen)
end

"""
    allocate_like(prototype::PencilArray, ::Type{T}=eltype(prototype)) -> PencilArray

Create a new PencilArray with the same shape and decomposition as prototype but potentially different type.
"""
function allocate_like(prototype::PencilArray, ::Type{T}=eltype(prototype)) where T
    pen = pencil(prototype)
    return PencilArray{T}(undef, pen)
end

"""
    zeros_like(prototype::PencilArray, ::Type{T}=eltype(prototype)) -> PencilArray

Create a zero-initialized PencilArray with the same shape and decomposition.
"""
function zeros_like(prototype::PencilArray, ::Type{T}=eltype(prototype)) where T
    arr = allocate_like(prototype, T)
    fill!(parent(arr), zero(T))
    return arr
end

# Get global indices with robust fallback patterns
function globalindices(A, dim)
    # PencilArrays v0.19+ uses range_local on the pencil
    try
        pen = pencil(A)
        ranges = PencilArrays.range_local(pen)
        if dim <= length(ranges)
            return ranges[dim]
        end
    catch
    end

    # Modern PencilArrays API (v0.17+)
    if isdefined(PencilArrays, :globalindices) && hasmethod(PencilArrays.globalindices, (typeof(A), typeof(dim)))
        return PencilArrays.globalindices(A, dim)
    end

    # Legacy API patterns (v0.15-v0.16)
    if isdefined(PencilArrays, :global_indices) && hasmethod(PencilArrays.global_indices, (typeof(A), typeof(dim)))
        return PencilArrays.global_indices(A, dim)
    end

    # Direct field access patterns
    try
        return A.pencil.axes[dim]
    catch
        # Alternative field access patterns
        try
            p = getfield(A, :pencil)
            if hasfield(typeof(p), :axes)
                ax = getfield(p, :axes)
                return ax[dim]
            elseif hasfield(typeof(p), :global_axes)
                global_axes = getfield(p, :global_axes)
                return global_axes[dim]
            end
        catch
            # Last resort: try to reconstruct from size information
            # Only safe if local_size == global_size (no distribution along this dimension)
            if hasmethod(size, (typeof(A),)) && hasmethod(PencilArrays.size_global, (typeof(A),))
                local_size = size(A)
                global_size = PencilArrays.size_global(A)
                if dim <= length(global_size) && dim <= length(local_size)
                    if local_size[dim] == global_size[dim]
                        # No distribution along this dimension - safe to return full range
                        return 1:global_size[dim]
                    end
                    # Distributed along this dimension but can't determine local indices
                    # Fall through to error
                end
            end
        end
    end

    error("Unable to get global indices for dimension $dim from PencilArray. "
          * "This may indicate an incompatible PencilArrays version.")
end

# ===== DISTRIBUTED FFT WRAPPERS =====
# Use FFTW for 1D FFTs along the longitude dimension (not PencilFFTs which is for multi-D)
# PencilArrays provides the distributed array framework, FFTW provides the FFTs

# Cache for FFTW 1D plans (key includes inplace flag)
const _fftw_plan_cache = Dict{Tuple{Symbol, Int, DataType, Bool}, Any}()
const _fftw_cache_lock = Threads.ReentrantLock()

"""
    get_fftw_plan(kind, n, T) -> plan

Get or create a cached FFTW plan for 1D transforms.
"""
function get_fftw_plan(kind::Symbol, n::Int, ::Type{T}; inplace::Bool=false) where T
    key = (kind, n, T, inplace)
    lock(_fftw_cache_lock) do
        if haskey(_fftw_plan_cache, key)
            return _fftw_plan_cache[key]
        end

        # Create sample array for planning
        if kind == :fft
            sample = zeros(Complex{real(T)}, n)
            plan = inplace ? FFTW.plan_fft!(sample) : FFTW.plan_fft(sample)
        elseif kind == :ifft
            sample = zeros(Complex{real(T)}, n)
            plan = inplace ? FFTW.plan_ifft!(sample) : FFTW.plan_ifft(sample)
        elseif kind == :rfft
            sample = zeros(real(T), n)
            plan = FFTW.plan_rfft(sample)  # rfft is always out-of-place
        elseif kind == :irfft
            # For irfft, input size is n÷2+1
            sample = zeros(Complex{real(T)}, n ÷ 2 + 1)
            plan = FFTW.plan_irfft(sample, n)
        else
            error("Unknown FFT kind: $kind")
        end

        _fftw_plan_cache[key] = plan
        return plan
    end
end

"""
    fft_along_dim2!(output, input)

Perform forward FFT along dimension 2 (longitude) for each row.
Works on the local data of a PencilArray.
"""
function fft_along_dim2!(output::AbstractMatrix{Complex{T}}, input::AbstractMatrix{T2}) where {T<:AbstractFloat, T2}
    nlat, nlon = size(input)
    # Use a contiguous temp buffer for FFT (avoids stride mismatch with cached plans)
    # FFTW internally caches plans for the same size/type, so this is still efficient after warmup
    temp = Vector{Complex{T}}(undef, nlon)
    @inbounds for i in 1:nlat
        # Copy with conversion to complex into contiguous buffer
        for j in 1:nlon
            temp[j] = Complex{T}(input[i, j])
        end
        # In-place FFT on contiguous buffer (FFTW caches the plan internally)
        FFTW.fft!(temp)
        # Copy result back to output
        for j in 1:nlon
            output[i, j] = temp[j]
        end
    end
    return output
end

function fft_along_dim2!(output::AbstractMatrix{Complex{T}}, input::AbstractMatrix{Complex{T}}) where {T<:AbstractFloat}
    nlat, nlon = size(input)
    # Use a contiguous temp buffer for FFT
    temp = Vector{Complex{T}}(undef, nlon)
    @inbounds for i in 1:nlat
        # Copy to contiguous buffer
        for j in 1:nlon
            temp[j] = input[i, j]
        end
        # In-place FFT
        FFTW.fft!(temp)
        # Copy result back
        for j in 1:nlon
            output[i, j] = temp[j]
        end
    end
    return output
end

"""
    ifft_along_dim2!(output, input)

Perform inverse FFT along dimension 2 (longitude) for each row.
"""
function ifft_along_dim2!(output::AbstractMatrix{T}, input::AbstractMatrix{Complex{T2}}) where {T<:AbstractFloat, T2<:AbstractFloat}
    nlat, nlon = size(input)
    # Pre-allocate contiguous temp buffer OUTSIDE the loop
    temp = Vector{Complex{T2}}(undef, nlon)
    @inbounds for i in 1:nlat
        # Copy to contiguous buffer
        for j in 1:nlon
            temp[j] = input[i, j]
        end
        # In-place IFFT (FFTW caches plans internally)
        FFTW.ifft!(temp)
        # Copy real part back to output
        for j in 1:nlon
            output[i, j] = real(temp[j])
        end
    end
    return output
end

function ifft_along_dim2!(output::AbstractMatrix{Complex{T}}, input::AbstractMatrix{Complex{T}}) where {T<:AbstractFloat}
    nlat, nlon = size(input)
    # Pre-allocate contiguous temp buffer
    temp = Vector{Complex{T}}(undef, nlon)
    @inbounds for i in 1:nlat
        # Copy to contiguous buffer
        for j in 1:nlon
            temp[j] = input[i, j]
        end
        # In-place IFFT
        FFTW.ifft!(temp)
        # Copy back to output
        for j in 1:nlon
            output[i, j] = temp[j]
        end
    end
    return output
end

# Legacy API wrappers for backward compatibility (used by existing code)
function plan_fft(A::PencilArray; dims=:)
    # Return a placeholder that indicates we'll use FFTW on local data
    return (kind=:fft, local_size=size(parent(A)))
end

function plan_ifft(A::PencilArray; dims=:)
    return (kind=:ifft, local_size=size(parent(A)))
end

function fft(A::PencilArray, p)
    local_data = parent(A)
    nlat, nlon = size(local_data)
    output = similar(local_data, Complex{Float64})
    fft_along_dim2!(output, local_data)
    return output
end

function ifft(A::PencilArray, p)
    local_data = parent(A)
    nlat, nlon = size(local_data)
    output = similar(local_data)
    ifft_along_dim2!(output, local_data)
    return output
end

# RFFT/IRFFT variants
function plan_rfft(A::PencilArray; dims=:)
    return (kind=:rfft, local_size=size(parent(A)))
end

function plan_irfft(A::PencilArray; dims=:)
    return (kind=:irfft, local_size=size(parent(A)))
end

function rfft(A::PencilArray, p)
    local_data = parent(A)
    nlat, nlon = size(local_data)
    nk = nlon ÷ 2 + 1
    output = Matrix{ComplexF64}(undef, nlat, nk)
    @inbounds for i in 1:nlat
        row = Vector{Float64}(collect(view(local_data, i, :)))
        fft_result = FFTW.rfft(row)
        for j in 1:nk
            output[i, j] = fft_result[j]
        end
    end
    return output
end

function irfft(A::AbstractMatrix{<:Complex}, p)
    nlat, nk = size(A)
    # Assume original nlon was 2*(nk-1) for even-length arrays
    nlon = 2 * (nk - 1)
    output = Matrix{Float64}(undef, nlat, nlon)
    @inbounds for i in 1:nlat
        row = Vector{ComplexF64}(collect(view(A, i, :)))
        ifft_result = FFTW.irfft(row, nlon)
        for j in 1:nlon
            output[i, j] = ifft_result[j]
        end
    end
    return output
end

# ===== OPTIMIZED DISTRIBUTED FFT USING TRANSPOSE =====
# When φ is distributed, use a single all-to-all transpose instead of per-row Allgatherv.
# This reduces the number of MPI calls from O(nlat) to O(1).

"""
    distributed_fft_phi!(Fθm_out, local_data, θ_range, φ_range, nlon, comm)

Optimized distributed FFT along φ (longitude) dimension.
Uses a single MPI_Alltoallv instead of per-row Allgatherv for better performance.

After transform, each rank has complete Fourier modes (all m) for its local θ rows.

# Algorithm
1. Pack local data for all-to-all communication
2. Single MPI_Alltoallv to redistribute data (each rank gets complete φ rows)
3. FFT along φ for each local θ row
4. Result: Fθm_out[i, m+1] = Fourier mode m at local θ index i
"""
function distributed_fft_phi!(Fθm_out::AbstractMatrix{ComplexF64},
                               local_data::AbstractMatrix,
                               θ_range::AbstractRange, φ_range::AbstractRange,
                               nlon::Int, comm)
    nlat_local = length(θ_range)
    nlon_local = length(φ_range)
    nprocs = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    # Gather φ segment sizes from all processes
    local_nlon = Int32(nlon_local)
    all_nlons = MPI.Allgather(local_nlon, comm)
    φ_displs = cumsum([Int32(0); all_nlons[1:end-1]])

    # For efficiency, transpose all data in a single all-to-all operation
    # Each rank sends its local_data to appropriate ranks based on φ distribution

    # Allocate buffer for gathered rows (all θ_local rows with complete φ)
    gathered_data = Matrix{Float64}(undef, nlat_local, nlon)

    # Build send/receive counts and displacements for Alltoallv
    # Each rank r needs to receive φ data from all ranks for its local θ rows

    # Strategy: Use MPI.Allgatherv with column-major packing for efficiency
    # Pack all local columns together and do single Allgatherv
    send_buf = Vector{Float64}(undef, nlat_local * nlon_local)
    recv_buf = Vector{Float64}(undef, nlat_local * nlon)

    # Pack local data (column-major for contiguous access)
    idx = 1
    @inbounds for j in 1:nlon_local
        for i in 1:nlat_local
            send_buf[idx] = local_data[i, j]
            idx += 1
        end
    end

    # Compute counts and displacements for recv buffer
    recv_counts = [nlat_local * Int(all_nlons[r]) for r in 1:nprocs]
    recv_displs = cumsum([0; recv_counts[1:end-1]])

    # Single Allgatherv for all data
    MPI.Allgatherv!(send_buf, VBuffer(recv_buf, recv_counts, recv_displs), comm)

    # Unpack received data into gathered_data matrix
    @inbounds for r in 1:nprocs
        offset = recv_displs[r]
        r_nlon = all_nlons[r]
        φ_start = φ_displs[r] + 1

        idx = 1
        for j in 1:r_nlon
            φ_idx = φ_start + j - 1
            for i in 1:nlat_local
                gathered_data[i, φ_idx] = recv_buf[offset + idx]
                idx += 1
            end
        end
    end

    # Now perform FFT on complete rows
    fft_along_dim2!(Fθm_out, gathered_data)

    return Fθm_out
end

"""
    distributed_ifft_phi!(local_out, Fθm, θ_range, φ_range, nlon, comm)

Optimized distributed IFFT along φ (longitude) dimension.
Inverse of distributed_fft_phi! - performs IFFT then scatters back to distributed layout.
"""
function distributed_ifft_phi!(local_out::AbstractMatrix,
                                Fθm::AbstractMatrix{<:Complex},
                                θ_range::AbstractRange, φ_range::AbstractRange,
                                nlon::Int, comm)
    nlat_local = length(θ_range)
    nlon_local = length(φ_range)

    # Perform IFFT on complete rows
    spatial_full = Matrix{Float64}(undef, nlat_local, nlon)
    ifft_along_dim2!(spatial_full, Fθm)

    # Extract local portion (no communication needed - just take local φ slice)
    φ_start = first(φ_range)
    @inbounds for j in 1:nlon_local
        for i in 1:nlat_local
            local_out[i, j] = spatial_full[i, φ_start + j - 1]
        end
    end

    return local_out
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
        # Create dummy communicator for non-representatives and free it
        dummy_comm = MPI.Comm_split(comm, 1, 0)
        MPI.Comm_free(dummy_comm)
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
            # Send to parent, then wait for down-sweep
            MPI.Send(data, rank - step, step, comm)
            break
        end
        step *= 2
    end

    # Down-sweep: broadcast final result to all ranks
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
    # Note: Resize operations are performed outside the lock to reduce contention.
    # This is safe because each (T, comm) key maps to a unique buffer pair that
    # is only used by operations on that specific communicator.
    key = (T, MPI.Comm_rank(comm), MPI.Comm_size(comm), UInt64(comm.val))
    idx_recv, val_recv, needs_resize_idx, needs_resize_val = lock(_cache_lock) do
        if haskey(_sparse_gather_cache, key)
            buf = _sparse_gather_cache[key]
            idx_buf = buf.idx
            val_buf = buf.val
            # Return buffers and resize flags; actual resize happens outside lock
            (idx_buf, val_buf, length(idx_buf) < total, length(val_buf) < total)
        else
            idx_buf = Vector{Int}(undef, total)
            val_buf = Vector{T}(undef, total)
            _sparse_gather_cache[key] = (idx=idx_buf, val=val_buf)
            (idx_buf, val_buf, false, false)
        end
    end

    # Perform resize outside lock to reduce lock contention in multi-threaded scenarios
    if needs_resize_idx
        resize!(idx_recv, total)
    end
    if needs_resize_val
        resize!(val_recv, total)
    end

    # Exchange indices and values using MPI.jl v0.20+ API
    Allgatherv!(idx_send, VBuffer(idx_recv, counts), comm)
    Allgatherv!(val_send, VBuffer(val_recv, counts), comm)

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
        # Non-leaders create dummy communicator and free it
        dummy_comm = MPI.Comm_split(comm, 1, 0)
        MPI.Comm_free(dummy_comm)
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
            # Send to parent in chunks, then wait for down-sweep
            for start_idx in 1:chunk_size:length(data)
                end_idx = min(start_idx + chunk_size - 1, length(data))
                chunk_view = view(data, start_idx:end_idx)
                MPI.Send(chunk_view, rank - step, step, comm)
            end
            break
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

# ===== EXPORTS =====
# Export types and functions defined in this extension

# 1D Distributed spectral types and functions (from ParallelTransforms.jl)
export DistributedSpectralPlan, DistributedSpectralArray
export create_distributed_spectral_plan, create_distributed_spectral_array
export gather_to_dense, scatter_from_dense!
export dist_analysis_distributed, dist_synthesis_distributed
export distributed_spectral_reduce!
export estimate_distributed_memory_savings

# 2D Distributed spectral types and functions (from ParallelTransforms.jl)
export DistributedSpectralPlan2D, DistributedSpectralArray2D
export create_distributed_spectral_plan_2d, create_distributed_spectral_array_2d
export suggest_spectral_grid
export gather_to_dense_2d, gather_to_full_dense_2d, scatter_from_dense_2d!
export dist_analysis_distributed_2d, dist_synthesis_distributed_2d
export dist_synthesis_distributed_2d_optimized
export estimate_distributed_memory_savings_2d
export validate_2d_distribution_alignment

# Plan types (from ParallelPlans.jl)
export DistAnalysisPlan, DistPlan, DistSphtorPlan, DistQstPlan

# Utility functions
export local_size, global_size
export validate_plm_tables, estimate_plm_tables_memory
export estimate_memory_savings

end # module SHTnsKitParallelExt
