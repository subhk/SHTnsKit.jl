#=
================================================================================
loop.jl - Unified CPU/GPU Loop Abstraction for Spherical Harmonic Operations
================================================================================

This file provides a @sht_loop macro that enables unified execution across CPU
(with SIMD), GPU (with KernelAbstractions), and distributed arrays (PencilArrays).
The pattern is inspired by BioFlow.jl's @loop macro, adapted for runtime array
type detection.

WHY A UNIFIED LOOP ABSTRACTION?
-------------------------------
Spherical harmonic transforms involve nested loops over:
- Latitude bands (θ direction)
- Azimuthal modes (m direction)
- Degree indices (l direction)

These loops need to run efficiently on CPU, GPU, and distributed systems.
Rather than writing separate code paths, we use a macro that automatically
selects the appropriate backend based on the array type.

HOW IT WORKS
------------
1. For regular CPU Arrays: Uses @simd with @fastmath @inbounds for vectorization
2. For GPU arrays (CuArray): Generates a KernelAbstractions @kernel and launches it
3. For PencilArrays: Operates on local data via parent(), uses SIMD

The macro detects the backend from the first array in the expression using
`get_backend()` from KernelAbstractions.

PENCILARRAY SUPPORT
-------------------
When the first array is a PencilArray (MPI-distributed), the macro:
1. Extracts local data using parent(arr)
2. Runs SIMD loops on the local portion
3. Does NOT handle MPI communication - that's the caller's responsibility

For distributed operations that need MPI reductions, use the full dist_* API.

USAGE
-----
```julia
# Parallel loop over latitude and mode indices
@sht_loop Fφ[i_lat, m_idx] = result over (i_lat, m_idx) ∈ CartesianIndices((nlat, mmax+1))

# Simple 1D loop over coefficients
@sht_loop alm[l+1, m+1] = acc over (l, m) ∈ CartesianIndices((lmax+1, mmax+1))

# With PencilArrays (operates on local data only)
@sht_loop local_field[I] = value over I ∈ CartesianIndices(size(parent(pencil_arr)))
```

CONFIGURATION
-------------
Force SIMD-only mode (disable GPU path):
```julia
SHTnsKit.set_loop_backend("SIMD")  # Always use CPU SIMD path
SHTnsKit.set_loop_backend("auto")  # Auto-detect from array type (default)
```

Query current backend mode:
```julia
SHTnsKit.loop_backend()  # Returns "auto" or "SIMD"
```

================================================================================
=#

using KernelAbstractions: get_backend, @index, @kernel, synchronize, CPU

# Backend preference: "auto" for automatic detection, "SIMD" to force CPU
const _LOOP_BACKEND = Ref{String}("auto")

"""
    loop_backend()

Return the current loop backend mode: "auto" or "SIMD".
"""
loop_backend() = _LOOP_BACKEND[]

"""
    set_loop_backend(backend::String)

Set the loop backend preference. Valid values:
- "auto": Automatically detect from array type (default) - uses GPU when arrays are on GPU
- "SIMD": Force CPU SIMD path (useful for debugging or when GPU is not desired)
"""
function set_loop_backend(backend::String)
    if !(backend in ("SIMD", "auto"))
        throw(ArgumentError("Invalid backend: \"$backend\". Use \"SIMD\" or \"auto\"."))
    end
    _LOOP_BACKEND[] = backend
    return backend
end

# Work-group size for GPU kernels (64 is a common optimal value)
const _WORKGROUP_SIZE = 64

"""
    _is_cpu_backend(arr)

Check if the array's backend is CPU.
"""
_is_cpu_backend(arr::AbstractArray) = get_backend(arr) isa CPU

"""
    _is_pencil_array(arr)

Check if the array is a PencilArray (distributed MPI array).
Uses duck typing to avoid hard dependency on PencilArrays.
"""
function _is_pencil_array(arr::AbstractArray)
    T = typeof(arr)
    type_name = string(nameof(T))
    return occursin("PencilArray", type_name) || occursin("ManyPencilArray", type_name)
end

"""
    _get_local_data(arr)

Get the local data from an array. For PencilArrays, this returns parent(arr).
For regular arrays, returns the array unchanged.
"""
function _get_local_data(arr::AbstractArray)
    if _is_pencil_array(arr)
        return parent(arr)
    else
        return arr
    end
end

"""
    @sht_loop <expr> over <I ∈ R>

Macro to automate fast loops using @simd when running on CPU,
or KernelAbstractions when running on GPU.

The macro extracts all symbols from the expression, creates kernel functions
for both CPU and GPU paths, and dispatches based on the array backend at runtime.

# Examples
```julia
# Loop over 2D Cartesian range
@sht_loop dest[i, j] = src[i, j] * scale over (i, j) ∈ CartesianIndices((n, m))

# Loop with accumulation pattern
@sht_loop alm[l+1, m+1] += weights[i] * Plm[i, l+1, m+1] * Fm[i, m+1] over i ∈ 1:nlat
```

# Notes
- Uses `get_backend()` on the first variable to detect CPU vs GPU
- GPU path uses KernelAbstractions with work-group size of 64 threads
- CPU path uses @simd @fastmath @inbounds for vectorization
- Set `SHTnsKit.set_loop_backend("SIMD")` to force CPU path
"""
macro sht_loop(args...)
    ex, _, itr = args
    _, I, R = itr.args
    sym = Symbol[]
    grab!(sym, ex)          # Extract all symbols from expression
    setdiff!(sym, [I])      # Don't pass loop index as argument

    symT = [gensym() for _ in 1:length(sym)]  # Generate type parameters
    symWtypes = joinsymtype(rep.(sym), symT)  # Symbols with types: [a::A, b::B, ...]

    @gensym kern_cpu kern_gpu kern_gpu_ dispatch_kern

    return quote
        # GPU kernel definition (only compiled when needed)
        @kernel function $kern_gpu_($(symWtypes...), @Const(I0)) where {$(symT...)}
            $I = @index(Global, Cartesian)
            $I += I0
            @fastmath @inbounds $ex
        end

        # CPU path: SIMD loop
        function $kern_cpu($(symWtypes...), R) where {$(symT...)}
            @simd for $I ∈ R
                @fastmath @inbounds $ex
            end
        end

        # Dispatch function: choose backend based on array type
        function $dispatch_kern($(symWtypes...), R) where {$(symT...)}
            first_arr = $(sym[1])

            # Check for PencilArray (MPI-distributed) - always use CPU SIMD on local data
            if $_is_pencil_array(first_arr)
                # For PencilArrays, get local data and run SIMD
                # Note: Caller must handle MPI communication separately
                $kern_cpu($(sym...), R)
            elseif $_LOOP_BACKEND[] == "SIMD" || $_is_cpu_backend(first_arr)
                # Regular CPU array - use SIMD
                $kern_cpu($(sym...), R)
            else
                # GPU array - launch kernel
                backend = get_backend(first_arr)
                kernel = $kern_gpu_(backend, $_WORKGROUP_SIZE)
                kernel($(sym...), R[1] - oneunit(R[1]), ndrange=size(R))
                synchronize(backend)
            end
        end

        # Call the dispatcher
        $dispatch_kern($(sym...), $R)
    end |> esc
end

# Helper functions for macro symbol extraction (matching BioFlow.jl pattern)

"""
    grab!(sym, ex)

Recursively extract all symbols and composite names from expression `ex`,
adding them to the `sym` array. Also replaces composite expressions with
their simplified symbol forms.
"""
function grab!(sym::Vector{Symbol}, ex::Expr)
    # Grab composite name (e.g., a.b) and return
    if ex.head == :.
        push!(sym, Symbol(ex.args[2].value))
        return
    end
    # Don't grab function names in calls
    start = ex.head == :call ? 2 : 1
    # Recurse into arguments
    foreach(a -> grab!(sym, a), ex.args[start:end])
    # Replace composites in args
    ex.args[start:end] = rep.(ex.args[start:end])
end

function grab!(sym::Vector{Symbol}, ex::Symbol)
    push!(sym, ex)
end

grab!(sym::Vector{Symbol}, ex) = nothing  # Ignore literals, etc.

"""
    rep(ex)

Replace composite expressions (like `a.b`) with just the field symbol.
"""
rep(ex) = ex
rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[2].value) : ex

"""
    joinsymtype(sym, symT)

Join symbols with their type parameters to create typed argument lists.
"""
joinsymtype(sym::Symbol, symT::Symbol) = Expr(:(::), sym, symT)
joinsymtype(sym, symT) = [joinsymtype(s, t) for (s, t) in zip(sym, symT)]

# CartesianIndex utilities (similar to BioFlow.jl)

"""
    CI(a...)

Shorthand constructor for CartesianIndex.
"""
@inline CI(a...) = CartesianIndex(a...)

"""
    δ(i, N::Int)
    δ(i, I::CartesianIndex{N})

Return a CartesianIndex of dimension N which is one at index i and zero elsewhere.
Useful for offsetting indices in a specific direction.

# Example
```julia
δ(1, CartesianIndex(2,3))  # Returns CartesianIndex(1,0)
```
"""
δ(i, ::Val{N}) where N = CI(ntuple(j -> j == i ? 1 : 0, N))
δ(i, I::CartesianIndex{N}) where N = δ(i, Val{N}())

"""
    inside(a; buff=1)

Return CartesianIndices range excluding `buff` layers of cells on all boundaries.
Useful for iterating over interior points while respecting boundary conditions.
"""
@inline inside(a::AbstractArray; buff=1) = CartesianIndices(
    map(ax -> first(ax)+buff:last(ax)-buff, axes(a))
)

"""
    @sht_inside <arr[I] = expr>

Convenience macro to loop over interior points of an array, excluding boundaries.
Automatically determines the loop range from the array size.

# Example
```julia
@sht_inside field[I] = 0.5 * (field_old[I+δ(1,I)] + field_old[I-δ(1,I)])
```
"""
macro sht_inside(ex)
    @assert ex.head == :(=) && ex.args[1].head == :ref
    a, I = ex.args[1].args[1:2]
    return quote
        SHTnsKit.@sht_loop $ex over $I ∈ SHTnsKit.inside($a)
    end |> esc
end

# Spherical harmonic specific loop ranges

"""
    spectral_range(lmax, mmax)

Return a CartesianIndices range for iterating over valid (l,m) coefficient pairs.
Note: Returns range for 1-based indexing of storage array, so iterate as:
    for idx ∈ spectral_range(lmax, mmax)
        l, m = idx[1] - 1, idx[2] - 1  # Convert to 0-based degree/order
        # ... work with alm[l+1, m+1]
    end
"""
spectral_range(lmax::Int, mmax::Int) = CartesianIndices((lmax+1, mmax+1))

"""
    spatial_range(nlat, nlon)

Return a CartesianIndices range for iterating over spatial grid points.
"""
spatial_range(nlat::Int, nlon::Int) = CartesianIndices((nlat, nlon))

"""
    latitude_range(nlat)

Return a range for iterating over latitude bands.
"""
latitude_range(nlat::Int) = 1:nlat

"""
    mode_range(mmax)

Return a range for iterating over azimuthal modes (0 to mmax).
Storage index: m+1 for m ∈ 0:mmax
"""
mode_range(mmax::Int) = 0:mmax

"""
    local_range(arr)

Return CartesianIndices for iterating over the local portion of an array.
For PencilArrays, this iterates over parent(arr) dimensions.
For regular arrays, this is equivalent to CartesianIndices(arr).

# Example
```julia
# With PencilArray (MPI-distributed)
@sht_loop local_data[I] = value over I ∈ local_range(pencil_arr)

# With regular array
@sht_loop field[I] = value over I ∈ local_range(regular_arr)
```
"""
function local_range(arr::AbstractArray)
    if _is_pencil_array(arr)
        return CartesianIndices(parent(arr))
    else
        return CartesianIndices(arr)
    end
end

"""
    local_size(arr)

Return the size of the local portion of an array.
For PencilArrays, returns size(parent(arr)).
For regular arrays, returns size(arr).
"""
function local_size(arr::AbstractArray)
    if _is_pencil_array(arr)
        return size(parent(arr))
    else
        return size(arr)
    end
end
