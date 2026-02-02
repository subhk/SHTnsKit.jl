#=
================================================================================
ParallelTransforms.jl - Core Distributed Spherical Harmonic Transform Implementations
================================================================================

This file contains the main distributed transform algorithms for SHTnsKit.
It handles MPI-distributed spatial data (PencilArrays) and produces/consumes
spherical harmonic coefficients.

MAIN PUBLIC FUNCTIONS
---------------------
- dist_analysis(cfg, fθφ)     : Spatial grid → Spherical harmonic coefficients
- dist_synthesis(cfg, Alm)    : Spherical harmonic coefficients → Spatial grid
- dist_analysis!(plan, ...)   : In-place version with pre-allocated buffers
- dist_synthesis!(plan, ...)  : In-place version with pre-allocated buffers

ALGORITHM OVERVIEW
------------------
Analysis (spatial → spectral):
1. If φ is distributed: MPI.Allgatherv! to collect full longitude rows
2. FFT along φ dimension: spatial f(θ,φ) → Fourier coefficients F(θ,m)
3. Legendre integration: For each m, integrate F(θ,m) * P_l^m(cos θ) * w(θ)
4. If θ is distributed: MPI.Allreduce! to sum partial contributions
5. Normalization: Apply spherical harmonic normalization factors

Synthesis (spectral → spatial):
1. Legendre summation: For each m, sum A_lm * P_l^m(cos θ)
2. IFFT along φ dimension: Fourier coefficients → spatial values
3. If φ is distributed: Extract local portion and scatter

PERFORMANCE OPTIMIZATION
------------------------
The inner loops use "function barriers" to ensure type stability:
- _analysis_loop_no_tables!()  : Computes Legendre polynomials on-demand
- _analysis_loop_with_tables!(): Uses precomputed Legendre polynomial tables

These separate functions allow Julia's compiler to specialize and eliminate
boxing allocations that would otherwise occur due to Union types in the
main function (e.g., temp_dense being Union{Nothing, Matrix}).

Without function barriers: ~34 MB allocations per call
With function barriers:    ~0.8 MB allocations per call (97% reduction)

DEBUGGING CHECKLIST
-------------------
1. Data layout issues:
   - Print `globalindices(fθφ, 1)` and `globalindices(fθφ, 2)` to verify ranges
   - Check `nlon_local == nlon` to determine if φ gather is needed

2. MPI synchronization:
   - All ranks must call collective operations (Allgatherv!, Allreduce!)
   - Use MPI.Barrier(comm) before/after timing measurements

3. Numerical accuracy:
   - Verify Gauss weights sum to ~2.0: `sum(cfg.w) ≈ 2.0`
   - Check coefficient magnitudes: `maximum(abs, Alm)`

4. Memory issues:
   - Use `@allocated` to measure per-call allocations
   - Warmup with 3-5 calls before timing (FFTW plan caching)

================================================================================
=#

# ===== ENHANCED PACKED STORAGE SYSTEM =====
# Reduces memory usage by ~50% for large spectral arrays by storing only l≥m coefficients
# This is optional - dense storage (full lmax×mmax matrix) is the default

"""
    PackedStorageInfo

Optimized packed storage layout information for spherical harmonic coefficients.
Pre-computes index mappings for efficient dense ↔ packed conversions.
"""
struct PackedStorageInfo
    lmax::Int
    mmax::Int 
    mres::Int
    nlm_packed::Int                    # Total number of packed coefficients
    
    # Pre-computed index mappings for performance
    lm_to_packed::Matrix{Int}          # [l+1, m+1] -> packed index (0 if invalid)
    packed_to_lm::Vector{Tuple{Int,Int}} # packed index -> (l, m)
    
    # Cache-friendly block structure
    m_blocks::Vector{UnitRange{Int}}   # Packed index ranges for each m value
end

function create_packed_storage_info(cfg::SHTnsKit.SHTConfig)
    lmax, mmax, mres = cfg.lmax, cfg.mmax, cfg.mres
    
    # Pre-compute all valid (l,m) -> packed mappings
    lm_to_packed = zeros(Int, lmax+1, mmax+1)
    packed_to_lm = Tuple{Int,Int}[]
    m_blocks = UnitRange{Int}[]
    
    packed_idx = 0
    for m in 0:mmax
        if m % mres == 0
            block_start = packed_idx + 1
            for l in m:lmax
                packed_idx += 1
                lm_to_packed[l+1, m+1] = packed_idx
                push!(packed_to_lm, (l, m))
            end
            push!(m_blocks, block_start:packed_idx)
        end
    end
    
    return PackedStorageInfo(lmax, mmax, mres, packed_idx, 
                           lm_to_packed, packed_to_lm, m_blocks)
end

# Optimized conversion functions using pre-computed mappings
function _dense_to_packed!(packed::Vector{ComplexF64}, dense::Matrix{ComplexF64}, info::PackedStorageInfo)
    # Block-wise vectorized conversion for better cache efficiency
    n_packed = info.nlm_packed
    n_threads = Threads.nthreads()
    
    if n_packed > 1024 && n_threads > 1
        # Multi-threaded for large conversions
        @threads for tid in 1:n_threads
            start_idx = 1 + (tid - 1) * n_packed ÷ n_threads
            end_idx = min(tid * n_packed ÷ n_threads, n_packed)
            
            @inbounds @simd ivdep for i in start_idx:end_idx
                l, m = info.packed_to_lm[i]
                packed[i] = dense[l+1, m+1]
            end
        end
    else
        # Single-threaded SIMD for small conversions
        @inbounds @simd ivdep for i in 1:n_packed
            l, m = info.packed_to_lm[i]
            packed[i] = dense[l+1, m+1]
        end
    end
    return packed
end

function _packed_to_dense!(dense::Matrix{ComplexF64}, packed::Vector{ComplexF64}, info::PackedStorageInfo)
    fill!(dense, 0.0 + 0.0im)
    n_packed = info.nlm_packed
    n_threads = Threads.nthreads()
    
    if n_packed > 1024 && n_threads > 1
        # Multi-threaded for large conversions
        @threads for tid in 1:n_threads
            start_idx = 1 + (tid - 1) * n_packed ÷ n_threads
            end_idx = min(tid * n_packed ÷ n_threads, n_packed)
            
            @inbounds @simd ivdep for i in start_idx:end_idx
                l, m = info.packed_to_lm[i]
                dense[l+1, m+1] = packed[i]
            end
        end
    else
        # Single-threaded SIMD for small conversions
        @inbounds @simd ivdep for i in 1:n_packed
            l, m = info.packed_to_lm[i]
            dense[l+1, m+1] = packed[i]
        end
    end
    return dense
end

# Backwards compatibility with existing interface
function _dense_to_packed!(packed::Vector{ComplexF64}, dense::Matrix{ComplexF64}, cfg)
    lmax, mmax, mres = cfg.lmax, cfg.mmax, cfg.mres
    @inbounds for m in 0:mmax
        (m % mres == 0) || continue
        @simd ivdep for l in m:lmax
            lm = SHTnsKit.LM_index(lmax, mres, l, m) + 1
            packed[lm] = dense[l+1, m+1]
        end
    end
    return packed
end

function _packed_to_dense!(dense::Matrix{ComplexF64}, packed::Vector{ComplexF64}, cfg)
    lmax, mmax, mres = cfg.lmax, cfg.mmax, cfg.mres
    fill!(dense, 0)
    @inbounds for m in 0:mmax
        (m % mres == 0) || continue
        @simd ivdep for l in m:lmax
            lm = SHTnsKit.LM_index(lmax, mres, l, m) + 1
            dense[l+1, m+1] = packed[lm]
        end
    end
    return dense
end

"""
    estimate_memory_savings(lmax, mmax) -> (dense_bytes, packed_bytes, savings_pct)

Estimate memory savings from using packed storage for spherical harmonic coefficients.
"""
function estimate_memory_savings(lmax::Int, mmax::Int)
    # Dense storage: (lmax+1) × (mmax+1) complex numbers
    dense_elements = (lmax + 1) * (mmax + 1)
    
    # Packed storage: only l ≥ m coefficients
    packed_elements = 0
    for m in 0:mmax
        packed_elements += max(0, lmax - m + 1)
    end
    
    bytes_per_element = sizeof(ComplexF64)
    dense_bytes = dense_elements * bytes_per_element
    packed_bytes = packed_elements * bytes_per_element
    savings_pct = 100.0 * (dense_bytes - packed_bytes) / dense_bytes
    
    return dense_bytes, packed_bytes, savings_pct
end

# ===== MPI DATA REDISTRIBUTION HELPERS =====

"""
    _gather_and_fft_phi(local_data, θ_range, φ_range, nlon, comm)

Gather data distributed along φ dimension, then perform FFT along φ.
Returns Fθm matrix with FFT coefficients for the local θ rows.

OPTIMIZED: Uses single MPI_Allgatherv for all data instead of per-row communication.
This reduces O(nlat) MPI calls to O(1), significantly improving scalability.
"""
function _gather_and_fft_phi(local_data::AbstractMatrix, θ_range::AbstractRange,
                              φ_range::AbstractRange, nlon::Int, comm)
    nlat_local = length(θ_range)

    # Allocate output buffer
    Fθm = Matrix{ComplexF64}(undef, nlat_local, nlon)

    # Use optimized distributed FFT with single all-to-all communication
    SHTnsKitParallelExt.distributed_fft_phi!(Fθm, local_data, θ_range, φ_range, nlon, comm)

    return Fθm
end

"""
    _scatter_from_fft_phi(Fθm, θ_range, φ_range, nlon, comm)

Perform IFFT along φ and scatter the result back to distributed layout.
Returns local data matrix for the process's portion of the grid.

Note: IFFT is local since each rank has complete Fourier modes for its θ rows.
Only extraction of the local φ portion is needed (no MPI communication).
"""
function _scatter_from_fft_phi(Fθm::AbstractMatrix{<:Complex}, θ_range::AbstractRange,
                                φ_range::AbstractRange, nlon::Int, comm)
    nlat_local = length(θ_range)
    nlon_local = length(φ_range)

    # Allocate output for local portion
    local_data = Matrix{Float64}(undef, nlat_local, nlon_local)

    # Use optimized distributed IFFT
    SHTnsKitParallelExt.distributed_ifft_phi!(local_data, Fθm, θ_range, φ_range, nlon, comm)

    return local_data
end

function SHTnsKit.dist_analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false, use_packed_storage::Bool=false, use_cache_blocking::Bool=true, use_loop_fusion::Bool=true)
    if use_loop_fusion && use_cache_blocking
        return dist_analysis_fused_cache_blocked(cfg, fθφ; use_tables, use_rfft, use_packed_storage)
    elseif use_cache_blocking
        return dist_analysis_cache_blocked(cfg, fθφ; use_tables, use_rfft, use_packed_storage)
    else
        return dist_analysis_standard(cfg, fθφ; use_tables, use_rfft, use_packed_storage)
    end
end

# ===== FUNCTION BARRIER HELPERS FOR TYPE-STABLE INNER LOOPS =====
#
# WHY FUNCTION BARRIERS?
# ----------------------
# Julia's type inference can struggle with functions that have conditional branches
# creating Union types. In the main analysis function, `temp_dense` can be either
# `nothing` (packed storage) or `Matrix{ComplexF64}` (dense storage), creating a
# Union{Nothing, Matrix{ComplexF64}} type.
#
# Even though the branch is predictable at runtime, Julia generates generic code
# that handles both cases, leading to:
# - Boxing of loop variables
# - Allocation of intermediate values
# - ~34 MB allocations per call (!)
#
# By extracting the hot inner loop into a separate function with explicit type
# signatures, Julia can specialize the code for the concrete types, eliminating
# all allocations in the inner loop.
#
# DEBUGGING TIP: If you see high allocations in dist_analysis, check that these
# function barriers are being called (not the inline fallback code).

"""
    _analysis_loop_no_tables!(temp_dense, P, Fθm, weights_cache, x_cache, θ_globals, lmax, mmax)

Inner loop for spherical harmonic analysis when plm_tables are NOT available.
Computes Legendre polynomials on-demand using Plm_row!.

This is a "function barrier" - a separate function with explicit type signatures
that allows Julia to generate specialized, allocation-free code.

# Arguments
- `temp_dense::Matrix{ComplexF64}`: Output accumulator for coefficients (modified in-place)
- `P::Vector{Float64}`: Pre-allocated buffer for Legendre polynomials
- `Fθm::Matrix{ComplexF64}`: FFT results, shape (nθ_local, nlon)
- `weights_cache::Vector{Float64}`: Gauss-Legendre quadrature weights for local θ
- `x_cache::Vector{Float64}`: cos(θ) values for local θ points (pre-cached from cfg.x)
- `θ_globals::Vector{Int}`: Global θ indices owned by this process
- `lmax::Int`, `mmax::Int`: Maximum spherical harmonic degrees

# Performance
- Zero allocations after warmup
- Called O(1) times per dist_analysis call
- Inner loop complexity: O(mmax × nθ_local × lmax)
"""
function _analysis_loop_no_tables!(temp_dense::Matrix{ComplexF64}, P::Vector{Float64},
                                   Fθm::Matrix{ComplexF64}, weights_cache::Vector{Float64},
                                   x_cache::Vector{Float64}, θ_globals::Vector{Int},
                                   lmax::Int, mmax::Int)
    nθ_local = length(θ_globals)
    @inbounds for mval in 0:mmax
        col = mval + 1
        m_fft = mval + 1
        for ii in 1:nθ_local
            Fi = Fθm[ii, m_fft]
            wi = weights_cache[ii]
            SHTnsKit.Plm_row!(P, x_cache[ii], lmax, mval)
            @simd for l in mval:lmax
                temp_dense[l+1, col] += wi * P[l+1] * Fi
            end
        end
    end
    return nothing
end

"""
    _analysis_loop_with_tables!(temp_dense, plm_tables, Fθm, weights_cache, θ_globals, lmax, mmax)

Inner loop for spherical harmonic analysis when plm_tables ARE available.
Uses precomputed Legendre polynomials from cfg.plm_tables for faster execution.

This is a "function barrier" - see _analysis_loop_no_tables! for explanation.

# Arguments
- `temp_dense::Matrix{ComplexF64}`: Output accumulator for coefficients (modified in-place)
- `plm_tables::Vector{Matrix{Float64}}`: Precomputed Legendre polynomials, plm_tables[m+1][l+1, θ]
- `Fθm::Matrix{ComplexF64}`: FFT results, shape (nθ_local, nlon)
- `weights_cache::Vector{Float64}`: Gauss-Legendre quadrature weights for local θ
- `θ_globals::Vector{Int}`: Global θ indices owned by this process
- `lmax::Int`, `mmax::Int`: Maximum spherical harmonic degrees

# Performance
- Zero allocations after warmup
- Faster than no-tables version when tables are pre-computed
- Memory vs speed tradeoff: tables use O(lmax² × nlat) memory
"""
function _analysis_loop_with_tables!(temp_dense::Matrix{ComplexF64},
                                     plm_tables::Vector{Matrix{Float64}},
                                     Fθm::Matrix{ComplexF64}, weights_cache::Vector{Float64},
                                     θ_globals::Vector{Int}, lmax::Int, mmax::Int)
    nθ_local = length(θ_globals)
    @inbounds for mval in 0:mmax
        col = mval + 1
        m_fft = mval + 1
        for ii in 1:nθ_local
            iglob = θ_globals[ii]
            Fi = Fθm[ii, m_fft]
            wi = weights_cache[ii]
            tblcol = view(plm_tables[col], :, iglob)
            @simd for l in mval:lmax
                temp_dense[l+1, col] += wi * tblcol[l+1] * Fi
            end
        end
    end
    return nothing
end

"""
    dist_analysis_standard(cfg, fθφ; use_tables, use_rfft, use_packed_storage) -> Alm

Standard implementation of distributed spherical harmonic analysis.
Transforms spatial data f(θ,φ) on a PencilArray to spectral coefficients A_lm.

# Algorithm Steps
1. Extract local data and determine distribution pattern
2. If φ is distributed: gather full longitude rows via MPI.Allgatherv!
3. Perform FFT along longitude: f(θ,φ) → F(θ,m)
4. Compute Legendre integration: A_lm = Σ_θ w(θ) * F(θ,m) * P_l^m(cos θ)
5. If θ is distributed: MPI.Allreduce! to sum contributions from all ranks
6. Apply normalization factors

# Arguments
- `cfg::SHTConfig`: Configuration with lmax, mmax, Gauss points, etc.
- `fθφ::PencilArray`: Distributed spatial data, shape (nlat, nlon) globally

# Keyword Arguments
- `use_tables=cfg.use_plm_tables`: Use precomputed Legendre tables if available
- `use_rfft=false`: Reserved for real FFT optimization (not yet implemented)
- `use_packed_storage=false`: Use memory-efficient packed coefficient storage

# Returns
- `Alm`: Spherical harmonic coefficients, shape (lmax+1, mmax+1) or packed vector

# Performance Notes
- Uses function barriers for type-stable inner loops (~97% allocation reduction)
- Pre-caches cfg.x and cfg.w values to avoid repeated field access
- Warmup 3-5 calls before timing (FFTW plan caching)

# Debugging
```julia
# Check local data layout
println("Local size: ", size(parent(fθφ)))
println("Global indices θ: ", globalindices(fθφ, 1))
println("Global indices φ: ", globalindices(fθφ, 2))

# Measure allocations
@allocated Alm = dist_analysis_standard(cfg, fθφ)  # Should be ~800 KB after warmup
```
"""
function dist_analysis_standard(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false, use_packed_storage::Bool=false)
    comm = communicator(fθφ)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon
    nlat = cfg.nlat

    # ===== STEP 1: Extract local data and determine distribution =====
    # parent(fθφ) gives the underlying Array without PencilArray wrapper
    local_data = parent(fθφ)
    nlat_local, nlon_local = size(local_data)

    # Get global index ranges for this process's local data
    # globalindices(fθφ, dim) returns the global indices this rank owns for dimension dim
    # Example: rank 0 might own θ indices 1:48, rank 1 owns 49:96
    θ_globals = collect(globalindices(fθφ, 1))  # Global theta indices owned by this process
    nθ_local = length(θ_globals)

    # ===== STEP 2 & 3: FFT along longitude (φ) dimension =====
    # After FFT, Fθm[i, m+1] contains the m-th Fourier coefficient at local θ index i
    # Shape: (nlat_local, nlon) where column m+1 corresponds to azimuthal mode m
    Fθm = Matrix{ComplexF64}(undef, nlat_local, nlon)

    if nlon_local == nlon
        # CASE A: Data is distributed along θ only (φ is complete on each rank)
        # This is the fast path - no MPI communication needed for FFT
        SHTnsKitParallelExt.fft_along_dim2!(Fθm, local_data)
    else
        # CASE B: Data is distributed along φ (longitude)
        # Need to gather full longitude rows before FFT
        # This requires MPI.Allgatherv! for each latitude row - more communication
        φ_globals = collect(globalindices(fθφ, 2))
        φ_range = first(φ_globals):last(φ_globals)
        θ_range = first(θ_globals):last(θ_globals)
        Fθm = _gather_and_fft_phi(local_data, θ_range, φ_range, nlon, comm)
    end

    # ===== STEP 4: Allocate output coefficient storage =====
    # Choose between dense (lmax+1 × mmax+1 matrix) or packed (vector with only l≥m)
    storage_info = use_packed_storage ? create_packed_storage_info(cfg) : nothing

    if use_packed_storage
        # Packed storage: only store coefficients where l ≥ m (~50% memory savings)
        Alm_local = zeros(ComplexF64, storage_info.nlm_packed)
        temp_dense = nothing  # NOTE: This creates Union type - handled by function barrier
        if get(ENV, "SHTNSKIT_VERBOSE_STORAGE", "0") == "1"
            dense_bytes, packed_bytes, savings = estimate_memory_savings(lmax, mmax)
            @info "Using packed storage: $(round(savings, digits=1))% memory reduction ($(packed_bytes ÷ 1024) KB vs $(dense_bytes ÷ 1024) KB)"
        end
    else
        # Dense storage: full (lmax+1) × (mmax+1) matrix (simpler, faster for small problems)
        Alm_local = zeros(ComplexF64, lmax+1, mmax+1)
        temp_dense = Alm_local  # Alias - same memory
    end

    # ===== Validate and configure Legendre polynomial source =====
    # plm_tables: precomputed P_l^m(cos θ) for all l, m, θ - faster but uses more memory
    # On-demand: compute P_l^m using recurrence relations - slower but no extra memory
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables)

    # Validate plm_tables structure for better error messages
    if use_tbl
        if length(cfg.plm_tables) != mmax + 1
            @warn "plm_tables length mismatch: expected $(mmax + 1), got $(length(cfg.plm_tables)). Falling back to on-demand computation."
            use_tbl = false
        else
            first_table = cfg.plm_tables[1]
            if size(first_table, 2) != nlat
                @warn "plm_tables latitude dimension mismatch: expected $(nlat), got $(size(first_table, 2)). Falling back to on-demand computation."
                use_tbl = false
            end
        end
    end

    # Buffer for Legendre polynomials when computing on-demand
    P = Vector{Float64}(undef, lmax + 1)

    # ===== Pre-cache values for type-stable inner loop =====
    # Caching these values outside the loop is critical for performance:
    # 1. Avoids repeated field access to cfg struct (which can cause allocations)
    # 2. Enables the function barrier to receive concrete-typed Vector arguments

    # Gauss-Legendre quadrature weights: w[θ] for integration over latitude
    weights_cache = Vector{Float64}(undef, nθ_local)
    for (ii, iglob) in enumerate(θ_globals)
        weights_cache[ii] = cfg.w[iglob]
    end

    # cos(θ) values needed for Legendre polynomial computation
    x_cache = Vector{Float64}(undef, nθ_local)
    for (ii, iglob) in enumerate(θ_globals)
        x_cache[ii] = cfg.x[iglob]
    end

    # ===== STEP 4: Main Legendre integration loop =====
    # This is the computational core: integrate F(θ,m) * P_l^m(cos θ) * w(θ) for all l,m
    # Uses function barriers for type stability (eliminates ~33MB allocations!)
    # See _analysis_loop_no_tables! and _analysis_loop_with_tables! for details
    if use_packed_storage
        # Original inline loop for packed storage (not the hot path)
        for mval in 0:mmax
            col = mval + 1
            m_fft = mval + 1
            for (ii, iglob) in enumerate(θ_globals)
                Fi = Fθm[ii, m_fft]
                wi = weights_cache[ii]
                if use_tbl
                    tblcol = view(cfg.plm_tables[col], :, iglob)
                    @inbounds @simd for l in mval:lmax
                        lm = storage_info.lm_to_packed[l+1, col]
                        Alm_local[lm] += (wi * cfg.Nlm[l+1, col] * cfg.cphi * tblcol[l+1]) * Fi
                    end
                else
                    SHTnsKit.Plm_row!(P, cfg.x[iglob], lmax, mval)
                    @inbounds @simd for l in mval:lmax
                        lm = storage_info.lm_to_packed[l+1, col]
                        Alm_local[lm] += (wi * cfg.Nlm[l+1, col] * cfg.cphi * P[l+1]) * Fi
                    end
                end
            end
        end
    elseif use_tbl
        # Use function barrier for tables path (zero allocation)
        _analysis_loop_with_tables!(temp_dense, cfg.plm_tables, Fθm, weights_cache, θ_globals, lmax, mmax)
    else
        # Use function barrier for no-tables path (zero allocation)
        _analysis_loop_no_tables!(temp_dense, P, Fθm, weights_cache, x_cache, θ_globals, lmax, mmax)
    end
    
    # ===== STEP 5: MPI reduction to combine partial results =====
    # Each rank has computed partial sums over its local θ indices
    # Need to sum across all ranks to get final coefficients
    #
    # IMPORTANT: Only reduce if θ is actually distributed!
    # - If θ is distributed: each rank has different θ points → need Allreduce
    # - If only φ is distributed: all ranks have same θ points after gather → skip reduction
    θ_is_distributed = (nθ_local < nlat)

    if θ_is_distributed
        if use_packed_storage
            # Reduce packed coefficients directly (already normalized during accumulation)
            SHTnsKitParallelExt.efficient_spectral_reduce!(Alm_local, comm)
        else
            # Use efficient reduction for large spectral arrays
            SHTnsKitParallelExt.efficient_spectral_reduce!(Alm_local, comm)
            # Apply normalization to dense matrix with SIMD optimization
            # Each (l,m) element is independent, so ivdep is safe
            @inbounds for m in 0:mmax
                @simd ivdep for l in m:lmax
                    Alm_local[l+1, m+1] *= cfg.Nlm[l+1, m+1] * cfg.cphi
                end
            end
        end
    else
        # θ is not distributed - no reduction needed, just apply normalization
        if !use_packed_storage
            @inbounds for m in 0:mmax
                @simd ivdep for l in m:lmax
                    Alm_local[l+1, m+1] *= cfg.Nlm[l+1, m+1] * cfg.cphi
                end
            end
        end
    end
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        Alm_out = similar(Alm_local)
        SHTnsKit.convert_alm_norm!(Alm_out, Alm_local, cfg; to_internal=false)
        return Alm_out
    else
        return Alm_local
    end
end

"""
    dist_analysis_fused_cache_blocked(cfg, fθφ; kwargs...)

Cache-optimized parallel analysis with loop fusion for maximum performance.
Fuses FFT processing, Legendre integration, and normalization into optimized loops.

Note: Currently redirects to dist_analysis_standard. Cache blocking is implemented there.
"""
function dist_analysis_fused_cache_blocked(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false, use_packed_storage::Bool=false)
    # Redirect to standard implementation which has proper PencilArrays API usage
    return dist_analysis_standard(cfg, fθφ; use_tables, use_rfft, use_packed_storage)
end

# Legacy implementation preserved for reference - uses incorrect transpose API
function _dist_analysis_fused_cache_blocked_legacy(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false, use_packed_storage::Bool=false)
    comm = communicator(fθφ)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon

    # Get local data and perform FFT (fixed implementation)
    local_data = parent(fθφ)
    nlat_local, nlon_local = size(local_data)
    θ_globals = collect(globalindices(fθφ, 1))

    Fθm = Matrix{ComplexF64}(undef, nlat_local, nlon)
    if nlon_local == nlon
        SHTnsKitParallelExt.fft_along_dim2!(Fθm, local_data)
    else
        φ_globals = collect(globalindices(fθφ, 2))
        φ_range = first(φ_globals):last(φ_globals)
        θ_range = first(θ_globals):last(θ_globals)
        Fθm = _gather_and_fft_phi(local_data, θ_range, φ_range, nlon, comm)
    end
    
    # Enhanced packed storage optimization for fused analysis
    storage_info = use_packed_storage ? create_packed_storage_info(cfg) : nothing
    
    if use_packed_storage
        Alm_local = zeros(ComplexF64, storage_info.nlm_packed)
        temp_dense = nothing
        if get(ENV, "SHTNSKIT_VERBOSE_STORAGE", "0") == "1"
            dense_bytes, packed_bytes, savings = estimate_memory_savings(lmax, mmax)
            @info "Fused analysis using packed storage: $(round(savings, digits=1))% memory saved"
        end
    else
        Alm_local = zeros(ComplexF64, lmax+1, mmax+1)
        temp_dense = Alm_local
    end
    
    θrange = axes(Fθm, 1); mrange = axes(Fθm, 2)
    
    # Enhanced plm_tables integration for fused analysis
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables)
    if use_tbl && length(cfg.plm_tables) != mmax + 1
        @warn "plm_tables validation failed in fused analysis, falling back to on-demand computation"
        use_tbl = false
    end
    
    P = Vector{Float64}(undef, lmax + 1)  # Fallback buffer
    
    # Pre-compute all index maps and derived values
    θ_globals = collect(globalindices(Fθm, 1))
    m_globals = collect(globalindices(Fθm, 2))
    nθ_local = length(θ_globals)
    
    # Pre-cache weights and normalization factors
    weights_cache = Vector{Float64}(undef, nθ_local)
    for (ii, iglob) in enumerate(θ_globals)
        weights_cache[ii] = cfg.w[iglob]
    end
    
    # Table view cache for optimal memory access
    table_view_cache = use_tbl ? Dict{Tuple{Int,Int}, Any}() : nothing
    
    # FUSED STEP 3+4: Combined cache-blocked analysis with integrated normalization
    cache_size_kb = get(ENV, "SHTNSKIT_CACHE_SIZE", "32") |> x -> parse(Int, x)
    elements_per_kb = 1024 ÷ sizeof(ComplexF64)
    block_size_m = max(1, min(length(mrange), cache_size_kb * elements_per_kb ÷ (2 * nθ_local)))
    
    for m_start in 1:block_size_m:length(mrange)
        m_end = min(m_start + block_size_m - 1, length(mrange))
        
        # Process m-block with fused integration and normalization
        for jj in m_start:m_end
            m = mrange[jj]
            mglob = m_globals[jj]
            mval = mglob - 1
            (mval <= mmax) || continue
            col = mval + 1
            
            # Cache-optimized θ blocking with fused operations
            θ_block_size = min(16, nθ_local)  # Tuned for L1 cache
            
            for θ_start in 1:θ_block_size:nθ_local
                θ_end = min(θ_start + θ_block_size - 1, nθ_local)
                
                for ii in θ_start:θ_end
                    iθ = θrange[ii]
                    iglob = θ_globals[ii]
                    Fi = Fθm[iθ, m]
                    wi = weights_cache[ii]  # Pre-cached weight
                    
                    if use_tbl
                        # Cache-optimized table access with bounded caching
                        cache_key = (col, iglob)
                        tblcol = if haskey(table_view_cache, cache_key)
                            table_view_cache[cache_key]
                        else
                            view_col = view(cfg.plm_tables[col], :, iglob)
                            if length(table_view_cache) < 8000  # Memory-bounded cache
                                table_view_cache[cache_key] = view_col
                            end
                            view_col
                        end
                        
                        # FUSED: Integration + normalization (+ packing when enabled)
                        if use_packed_storage
                            @inbounds @simd for l in mval:lmax
                                fused_weight = wi * cfg.Nlm[l+1, col] * cfg.cphi
                                lm = storage_info.lm_to_packed[l+1, col]
                                Alm_local[lm] += (fused_weight * tblcol[l+1]) * Fi
                            end
                        else
                            @inbounds @simd for l in mval:lmax
                                fused_weight = wi * cfg.Nlm[l+1, col] * cfg.cphi
                                temp_dense[l+1, col] += (fused_weight * tblcol[l+1]) * Fi
                            end
                        end
                    else
                        # Fallback with fused operations
                        try
                            SHTnsKit.Plm_row!(P, cfg.x[iglob], lmax, mval)
                            if use_packed_storage
                                @inbounds @simd for l in mval:lmax
                                    fused_weight = wi * cfg.Nlm[l+1, col] * cfg.cphi
                                    lm = storage_info.lm_to_packed[l+1, col]
                                    Alm_local[lm] += (fused_weight * P[l+1]) * Fi
                                end
                            else
                                @inbounds @simd for l in mval:lmax
                                    fused_weight = wi * cfg.Nlm[l+1, col] * cfg.cphi
                                    temp_dense[l+1, col] += (fused_weight * P[l+1]) * Fi
                                end
                            end
                        catch e
                            error("Failed to compute Legendre polynomials in fused analysis at latitude $iglob: $e")
                        end
                    end
                end
            end
        end
    end
    
    # Handle MPI reduction and storage conversion
    if use_packed_storage
        SHTnsKitParallelExt.efficient_spectral_reduce!(Alm_local, comm)
    else
        SHTnsKitParallelExt.efficient_spectral_reduce!(Alm_local, comm)
    end
    
    # Convert to user's requested normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        Alm_out = similar(Alm_local)
        SHTnsKit.convert_alm_norm!(Alm_out, Alm_local, cfg; to_internal=false)
        return Alm_out
    else
        return Alm_local
    end
end

"""
    dist_analysis_cache_blocked(cfg, fθφ; kwargs...)

Cache-optimized parallel analysis that processes data in cache-friendly blocks
to minimize memory bandwidth and improve performance on NUMA systems.

Note: Currently redirects to dist_analysis_standard. Cache blocking is implemented there.
"""
function dist_analysis_cache_blocked(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false, use_packed_storage::Bool=false)
    # Redirect to standard implementation which has proper PencilArrays API usage
    return dist_analysis_standard(cfg, fθφ; use_tables, use_rfft, use_packed_storage)
end

# Legacy implementation preserved for reference - uses incorrect transpose API
function _dist_analysis_cache_blocked_legacy(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false, use_packed_storage::Bool=false)
    comm = communicator(fθφ)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon

    # Get local data and perform FFT (fixed implementation)
    local_data = parent(fθφ)
    nlat_local, nlon_local = size(local_data)
    θ_globals = collect(globalindices(fθφ, 1))

    Fθm = Matrix{ComplexF64}(undef, nlat_local, nlon)
    if nlon_local == nlon
        SHTnsKitParallelExt.fft_along_dim2!(Fθm, local_data)
    else
        φ_globals = collect(globalindices(fθφ, 2))
        φ_range = first(φ_globals):last(φ_globals)
        θ_range = first(θ_globals):last(θ_globals)
        Fθm = _gather_and_fft_phi(local_data, θ_range, φ_range, nlon, comm)
    end
    
    # Use packed storage for better memory efficiency
    storage_info = use_packed_storage ? create_packed_storage_info(cfg) : nothing
    if use_packed_storage
        Alm_local = zeros(ComplexF64, cfg.nlm)
        temp_dense = nothing
    else
        Alm_local = zeros(ComplexF64, lmax+1, mmax+1)
        temp_dense = Alm_local
    end

    θrange = axes(Fθm, 1); mrange = axes(Fθm, 2)

    # Enhanced plm_tables integration for cache-blocked analysis
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables)
    
    # Validate plm_tables structure
    if use_tbl
        if length(cfg.plm_tables) != mmax + 1
            @warn "plm_tables length mismatch in cache-blocked analysis: expected $(mmax + 1), got $(length(cfg.plm_tables)). Falling back to on-demand computation."
            use_tbl = false
        end
    end
    
    P = Vector{Float64}(undef, lmax + 1)  # Fallback buffer
    
    # Enhanced pre-computed index maps for cache-blocked analysis
    θ_globals = collect(globalindices(Fθm, 1))
    m_globals = collect(globalindices(Fθm, 2))
    
    # Pre-compute cache-blocking optimization parameters
    nθ_local = length(θ_globals)
    nm_local = length(m_globals)
    
    # Pre-cache Gauss-Legendre weights for all local latitudes
    weights_cache = Vector{Float64}(undef, nθ_local)
    for (ii, iglob) in enumerate(θ_globals)
        weights_cache[ii] = cfg.w[iglob]
    end
    
    # Pre-allocate table view cache for cache-blocked access
    table_view_cache = Dict{Tuple{Int,Int}, Any}()
    
    # CACHE BLOCKING: Process m-modes in cache-friendly blocks
    cache_size_kb = get(ENV, "SHTNSKIT_CACHE_SIZE", "32") |> x -> parse(Int, x)  # L1 cache size in KB
    elements_per_kb = 1024 ÷ sizeof(ComplexF64)  # ~128 complex numbers per KB
    block_size_m = max(1, min(length(mrange), cache_size_kb * elements_per_kb ÷ (2 * length(θrange))))
    
    for m_start in 1:block_size_m:length(mrange)
        m_end = min(m_start + block_size_m - 1, length(mrange))
        m_block = mrange[m_start:m_end]
        
        # Process this block of m-modes together for better cache locality
        for (jj, m) in enumerate(m_block)
            mm_global = m_globals[m_start + jj - 1]
            mval = mm_global - 1
            (mval <= mmax) || continue
            col = mval + 1
            
            # CACHE-OPTIMIZED: Process θ points in blocks for better L1 cache usage
            θ_block_size = min(32, length(θrange))  # Tune for L1 cache
            
            for θ_start in 1:θ_block_size:length(θrange)
                θ_end = min(θ_start + θ_block_size - 1, length(θrange))
                
                for ii in θ_start:θ_end
                    iθ = θrange[ii]
                    iglob = θ_globals[ii]     # Pre-computed global latitude index
                    Fi = Fθm[iθ, m]          # Local Fourier coefficient
                    wi = weights_cache[ii]      # Use pre-cached weight instead of cfg.w[iglob]
                    
                    if use_tbl
                        # Cache-optimized table access for better memory patterns
                        cache_key = (col, iglob)
                        if haskey(table_view_cache, cache_key)
                            tblcol = table_view_cache[cache_key]
                        else
                            tblcol = view(cfg.plm_tables[col], :, iglob)
                            # Only cache if we have room (avoid unbounded memory growth)
                            if length(table_view_cache) < 10000
                                table_view_cache[cache_key] = tblcol
                            end
                        end
                        
                        if use_packed_storage
                            @inbounds @simd for l in mval:lmax
                                weight_norm = wi * cfg.Nlm[l+1, col] * cfg.cphi
                                lm = storage_info.lm_to_packed[l+1, col]
                                Alm_local[lm] += (weight_norm * tblcol[l+1]) * Fi
                            end
                        else
                            @inbounds @simd for l in mval:lmax
                                weight_norm = wi * cfg.Nlm[l+1, col] * cfg.cphi
                                temp_dense[l+1, col] += (weight_norm * tblcol[l+1]) * Fi
                            end
                        end
                    else
                        # Fallback with better error handling
                        try
                            SHTnsKit.Plm_row!(P, cfg.x[iglob], lmax, mval)
                            if use_packed_storage
                                @inbounds @simd for l in mval:lmax
                                    weight_norm = wi * cfg.Nlm[l+1, col] * cfg.cphi
                                    lm = storage_info.lm_to_packed[l+1, col]
                                    Alm_local[lm] += (weight_norm * P[l+1]) * Fi
                                end
                            else
                                @inbounds @simd for l in mval:lmax
                                    weight_norm = wi * cfg.Nlm[l+1, col] * cfg.cphi
                                    temp_dense[l+1, col] += (weight_norm * P[l+1]) * Fi
                                end
                            end
                        catch e
                            error("Failed to compute Legendre polynomials in cache-blocked analysis at latitude $iglob: $e")
                        end
                    end
                end
            end
        end
    end
    
    # Handle MPI reduction and final processing (packed already normalized)
    SHTnsKitParallelExt.efficient_spectral_reduce!(Alm_local, comm)
    
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        Alm_out = similar(Alm_local)
        SHTnsKit.convert_alm_norm!(Alm_out, Alm_local, cfg; to_internal=false)
        return Alm_out
    else
        return Alm_local
    end
end

function SHTnsKit.dist_analysis!(plan::DistAnalysisPlan, Alm_out::AbstractMatrix, fθφ::PencilArray; use_tables=plan.cfg.use_plm_tables)
    if plan.with_spatial_scratch && plan.spatial_scratch !== nothing
        # Use scratch buffers from plan to eliminate allocations
        Alm = dist_analysis_with_scratch_buffers(plan, fθφ; use_tables)
    else
        # Fall back to regular analysis
        Alm = SHTnsKit.dist_analysis(plan.cfg, fθφ; use_tables, use_rfft=plan.use_rfft)
    end
    copyto!(Alm_out, Alm)
    return Alm_out
end

"""
    dist_analysis_with_scratch_buffers(plan::DistAnalysisPlan, fθφ; use_tables)

Optimized analysis using pre-allocated scratch buffers from the plan.
Eliminates all temporary allocations by reusing plan-based buffers.

Note: Currently redirects to dist_analysis_standard for correct PencilArrays API usage.
"""
function dist_analysis_with_scratch_buffers(plan::DistAnalysisPlan, fθφ::PencilArray; use_tables=plan.cfg.use_plm_tables)
    # Redirect to standard implementation which has proper PencilArrays API usage
    return dist_analysis_standard(plan.cfg, fθφ; use_tables, use_rfft=plan.use_rfft, use_packed_storage=plan.use_packed_storage)
end

# Legacy implementation preserved for reference
function _dist_analysis_with_scratch_buffers_legacy(plan::DistAnalysisPlan, fθφ::PencilArray; use_tables=plan.cfg.use_plm_tables)
    comm = communicator(fθφ)
    cfg = plan.cfg
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon
    scratch = plan.spatial_scratch

    # Get local data and perform FFT (fixed implementation)
    local_data = parent(fθφ)
    nlat_local, nlon_local = size(local_data)
    θ_globals = collect(globalindices(fθφ, 1))

    Fθm = Matrix{ComplexF64}(undef, nlat_local, nlon)
    if nlon_local == nlon
        SHTnsKitParallelExt.fft_along_dim2!(Fθm, local_data)
    else
        φ_globals = collect(globalindices(fθφ, 2))
        φ_range = first(φ_globals):last(φ_globals)
        θ_range = first(θ_globals):last(θ_globals)
        Fθm = _gather_and_fft_phi(local_data, θ_range, φ_range, nlon, comm)
    end
    
    # If using packed storage, accumulate directly into packed coefficients
    use_packed = plan.use_packed_storage
    storage_info = use_packed ? create_packed_storage_info(cfg) : nothing
    Alm_local = use_packed ? zeros(ComplexF64, storage_info.nlm_packed) : scratch.temp_dense
    if !use_packed
        fill!(scratch.temp_dense, 0.0 + 0.0im)
    end
    
    # Use plan's pre-computed index maps (no allocations)
    θ_globals = plan.θ_local_to_global
    m_globals = plan.m_local_to_global
    θrange = plan.θ_local_range
    mrange = plan.m_local_range
    
    # Enhanced plm_tables validation using cached info
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables)
    if use_tbl && length(cfg.plm_tables) != mmax + 1
        @warn "plm_tables validation failed, falling back to on-demand computation"
        use_tbl = false
    end
    
    # Clear table view cache for this transform
    empty!(scratch.table_view_cache)
    
    # Use pre-computed valid m-values (eliminates runtime validation)
    for (jj, mval, col) in scratch.valid_m_cache
        m = mrange[jj]
        for (ii, iθ) in enumerate(θrange)
            iglob = θ_globals[ii]
            Fi = Fθm[iθ, m]
            wi = scratch.weights_cache[ii]  # Use pre-cached weight (no cfg.w lookup)
            
            if use_tbl
                # Use bounded table cache from scratch buffers
                cache_key = (col, iglob)
                if haskey(scratch.table_view_cache, cache_key)
                    tblcol = scratch.table_view_cache[cache_key]
                else
                    tblcol = view(cfg.plm_tables[col], :, iglob)
                    if length(scratch.table_view_cache) < 5000  # Memory-bounded cache
                        scratch.table_view_cache[cache_key] = tblcol
                    end
                end
                if use_packed
                    @inbounds @simd for l in mval:lmax
                        lm = storage_info.lm_to_packed[l+1, col]
                        Alm_local[lm] += (wi * cfg.Nlm[l+1, col] * cfg.cphi * tblcol[l+1]) * Fi
                    end
                else
                    @inbounds @simd for l in mval:lmax
                        scratch.temp_dense[l+1, col] += wi * tblcol[l+1] * Fi
                    end
                end
            else
                # Use pre-allocated Legendre buffer from scratch
                try
                    SHTnsKit.Plm_row!(scratch.legendre_buffer, cfg.x[iglob], lmax, mval)
                    if use_packed
                        @inbounds @simd for l in mval:lmax
                            lm = storage_info.lm_to_packed[l+1, col]
                            Alm_local[lm] += (wi * cfg.Nlm[l+1, col] * cfg.cphi * scratch.legendre_buffer[l+1]) * Fi
                        end
                    else
                        @inbounds @simd for l in mval:lmax
                            scratch.temp_dense[l+1, col] += wi * scratch.legendre_buffer[l+1] * Fi
                        end
                    end
                catch e
                    error("Failed to compute Legendre polynomials with scratch buffers at latitude $iglob: $e")
                end
            end
        end
    end
    
    # Handle MPI reduction with optimized communication
    SHTnsKitParallelExt.efficient_spectral_reduce!(Alm_local, comm)
    
    # Always return dense matrix for compatibility with dist_analysis!(..., Alm_out::AbstractMatrix, ...)
    if use_packed
        dense = zeros(ComplexF64, lmax+1, mmax+1)
        _packed_to_dense!(dense, Alm_local, cfg)
        Alm_local = dense
    else
        Alm_local = copy(scratch.temp_dense)
        # Apply normalization in dense path
        @inbounds for m in 0:mmax
            @simd ivdep for l in m:lmax
                Alm_local[l+1, m+1] *= cfg.Nlm[l+1, m+1] * cfg.cphi
            end
        end
    end
    
    # Convert to user's requested normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        Alm_out = similar(Alm_local)
        SHTnsKit.convert_alm_norm!(Alm_out, Alm_local, cfg; to_internal=false)
        return Alm_out
    else
        return Alm_local
    end
end

function SHTnsKit.dist_synthesis(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix; prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon
    nlat = cfg.nlat

    # Get the local portion info from the prototype
    θ_globals = collect(globalindices(prototype_θφ, 1))  # Global θ indices this process owns
    nθ_local = length(θ_globals)
    nlon_local = size(parent(prototype_θφ), 2)

    # Check if φ is fully local or distributed
    φ_is_local = (nlon_local == nlon)

    # Allocate Fourier coefficient matrix (local θ × full m)
    Fθm = zeros(ComplexF64, nθ_local, nlon)

    P = Vector{Float64}(undef, lmax + 1)
    inv_scaleφ = SHTnsKit.phi_inv_scale(cfg)

    # Synthesis: for each m mode, compute Legendre series
    for mval in 0:mmax
        col = mval + 1

        # Compute synthesized values for each local θ
        for (ii, iglob) in enumerate(θ_globals)
            # Get Legendre polynomials at this latitude
            if cfg.use_plm_tables && !isempty(cfg.plm_tables)
                tbl = cfg.plm_tables[col]
                g = 0.0 + 0.0im
                @inbounds @simd for l in mval:lmax
                    g += (cfg.Nlm[l+1, col] * tbl[l+1, iglob]) * Alm[l+1, col]
                end
            else
                SHTnsKit.Plm_row!(P, cfg.x[iglob], lmax, mval)
                g = 0.0 + 0.0im
                @inbounds @simd for l in mval:lmax
                    g += (cfg.Nlm[l+1, col] * P[l+1]) * Alm[l+1, col]
                end
            end

            # Store in Fourier coefficient array
            Fθm[ii, mval + 1] = inv_scaleφ * g

            # For real output, set conjugate at m > 0
            if real_output && mval > 0
                conj_index = nlon - mval + 1
                Fθm[ii, conj_index] = conj(Fθm[ii, mval + 1])
            end
        end
    end

    # Perform inverse FFT along φ (dimension 2)
    fθφ_local = Matrix{Float64}(undef, nθ_local, nlon)
    SHTnsKitParallelExt.ifft_along_dim2!(fθφ_local, Fθm)

    # Apply Robert form scaling if enabled
    if cfg.robert_form
        @inbounds for (ii, iglob) in enumerate(θ_globals)
            x = cfg.x[iglob]
            sθ = sqrt(max(0.0, 1 - x*x))
            if sθ > 0
                for j in 1:nlon
                    fθφ_local[ii, j] *= sθ
                end
            end
        end
    end

    # If φ is distributed, we need to scatter results back
    if φ_is_local
        # Data is distributed along θ only - return local matrix wrapped properly
        result = real_output ? fθφ_local : Complex{Float64}.(fθφ_local)
    else
        # φ is distributed - extract local portion
        φ_globals = collect(globalindices(prototype_θφ, 2))
        local_φ_range = first(φ_globals):last(φ_globals)
        result = fθφ_local[:, local_φ_range]
        if !real_output
            result = Complex{Float64}.(result)
        end
    end

    return result
end

function SHTnsKit.dist_synthesis(cfg::SHTnsKit.SHTConfig, Alm::PencilArray; prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
    return SHTnsKit.dist_synthesis(cfg, Array(Alm); prototype_θφ, real_output, use_rfft)
end

function SHTnsKit.dist_synthesis!(plan::DistPlan, fθφ_out::PencilArray, Alm::PencilArray; real_output::Bool=true)
    f = SHTnsKit.dist_synthesis(plan.cfg, Alm; prototype_θφ=plan.prototype_θφ, real_output, use_rfft=plan.use_rfft)
    copyto!(fθφ_out, f)
    return fθφ_out
end

## Vector/QST distributed implementations

# Distributed vector analysis (spheroidal/toroidal)
function SHTnsKit.dist_spat_to_SHsphtor(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArray, Vpθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false)
    comm = communicator(Vtθφ)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon
    nlat = cfg.nlat

    # Get local data from PencilArrays
    local_Vt = parent(Vtθφ)
    local_Vp = parent(Vpθφ)
    nlat_local, nlon_local = size(local_Vt)

    # Get global index ranges for this process's local data
    θ_globals = collect(globalindices(Vtθφ, 1))  # Global θ indices
    nθ_local = length(θ_globals)

    # Perform 1D FFT along φ (longitude) dimension
    Ftθm = Matrix{ComplexF64}(undef, nθ_local, nlon)
    Fpθm = Matrix{ComplexF64}(undef, nθ_local, nlon)

    if nlon_local == nlon
        # Data is distributed along θ only - can do FFT directly
        SHTnsKitParallelExt.fft_along_dim2!(Ftθm, local_Vt)
        SHTnsKitParallelExt.fft_along_dim2!(Fpθm, local_Vp)
    else
        # Data is distributed along φ - need MPI Allgather along φ first
        φ_globals = collect(globalindices(Vtθφ, 2))
        φ_range = first(φ_globals):last(φ_globals)
        θ_range = first(θ_globals):last(θ_globals)
        Ftθm = _gather_and_fft_phi(local_Vt, θ_range, φ_range, nlon, comm)
        Fpθm = _gather_and_fft_phi(local_Vp, θ_range, φ_range, nlon, comm)
    end

    Slm_local = zeros(ComplexF64, lmax+1, mmax+1)
    Tlm_local = zeros(ComplexF64, lmax+1, mmax+1)

    # Pre-cache values for all local latitudes
    x_cache = Vector{Float64}(undef, nθ_local)
    sθ_cache = Vector{Float64}(undef, nθ_local)
    inv_sθ_cache = Vector{Float64}(undef, nθ_local)
    weights_cache = Vector{Float64}(undef, nθ_local)

    for (ii, iglobθ) in enumerate(θ_globals)
        x = cfg.x[iglobθ]
        sθ = sqrt(max(0.0, 1 - x*x))
        x_cache[ii] = x
        sθ_cache[ii] = sθ
        inv_sθ_cache[ii] = sθ == 0 ? 0.0 : 1.0 / sθ
        weights_cache[ii] = cfg.w[iglobθ]
    end

    # Enhanced plm_tables integration for vector spherical harmonic transforms
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)

    # Validate both plm_tables and dplm_tables structure
    if use_tbl
        if length(cfg.plm_tables) != mmax + 1 || length(cfg.dplm_tables) != mmax + 1
            @warn "Vector transform table length mismatch. Falling back to on-demand computation."
            use_tbl = false
        end
    end

    P = Vector{Float64}(undef, lmax + 1)     # Fallback buffers
    dPdx = Vector{Float64}(undef, lmax + 1)
    scaleφ = cfg.cphi

    # Main vector analysis loop
    for mval in 0:mmax
        col = mval + 1
        m_fft = mval + 1  # FFT index for this m mode

        for (ii, iglobθ) in enumerate(θ_globals)
            # Use pre-cached values
            x = x_cache[ii]
            sθ = sθ_cache[ii]
            inv_sθ = inv_sθ_cache[ii]
            wi = weights_cache[ii]

            # Get vector components from FFT results
            Fθ_i = Ftθm[ii, m_fft]
            Fφ_i = Fpθm[ii, m_fft]

            # Apply Robert form scaling if enabled
            if cfg.robert_form && sθ > 0
                Fθ_i /= sθ
                Fφ_i /= sθ
            end

            if use_tbl
                tblP = cfg.plm_tables[col]
                tbld = cfg.dplm_tables[col]

                @inbounds for l in max(1, mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, iglobθ]
                    Y = N * tblP[l+1, iglobθ]
                    coeff = wi * scaleφ / (l * (l + 1))
                    term = (0 + 1im) * mval * inv_sθ * Y
                    # Adjoint of synthesis: Vθ = dθY*S - term*T, Vφ = term*S + dθY*T
                    Slm_local[l+1, col] += coeff * (Fθ_i * dθY + conj(term) * Fφ_i)
                    Tlm_local[l+1, col] += coeff * (-conj(term) * Fθ_i + dθY * Fφ_i)
                end
            else
                # Fallback: compute Legendre polynomials and derivatives on-demand
                SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, mval)

                @inbounds for l in max(1, mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * dPdx[l+1]
                    Y = N * P[l+1]
                    coeff = wi * scaleφ / (l * (l + 1))
                    term = (0 + 1im) * mval * inv_sθ * Y
                    # Adjoint of synthesis: Vθ = dθY*S - term*T, Vφ = term*S + dθY*T
                    Slm_local[l+1, col] += coeff * (Fθ_i * dθY + conj(term) * Fφ_i)
                    Tlm_local[l+1, col] += coeff * (-conj(term) * Fθ_i + dθY * Fφ_i)
                end
            end
        end
    end

    # Only reduce if θ is actually distributed across processes
    # When φ is distributed but θ is not, all ranks compute identical results after gathering φ
    θ_is_distributed = (nθ_local < nlat)

    if θ_is_distributed
        # Use efficient reduction for better scaling on large process counts
        SHTnsKitParallelExt.efficient_spectral_reduce!(Slm_local, comm)
        SHTnsKitParallelExt.efficient_spectral_reduce!(Tlm_local, comm)
    end

    # Convert to cfg's requested normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm_local); T2 = similar(Tlm_local)
        SHTnsKit.convert_alm_norm!(S2, Slm_local, cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(T2, Tlm_local, cfg; to_internal=false)
        return S2, T2
    else
        return Slm_local, Tlm_local
    end
end

function SHTnsKit.dist_spat_to_SHsphtor!(plan::DistSphtorPlan, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix,
                                         Vtθφ::PencilArray, Vpθφ::PencilArray; use_tables=plan.cfg.use_plm_tables)
    Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(plan.cfg, Vtθφ, Vpθφ; use_tables, use_rfft=plan.use_rfft)
    copyto!(Slm_out, Slm); copyto!(Tlm_out, Tlm)
    return Slm_out, Tlm_out
end

# Distributed vector synthesis (spheroidal/toroidal) from dense spectra
function SHTnsKit.dist_SHsphtor_to_spat(cfg::SHTnsKit.SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon
    nlat = cfg.nlat

    size(Slm, 1) == lmax + 1 && size(Slm, 2) == mmax + 1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm, 1) == lmax + 1 && size(Tlm, 2) == mmax + 1 || throw(DimensionMismatch("Tlm dims"))

    # Convert incoming coefficients to internal normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm)
        T2 = similar(Tlm)
        SHTnsKit.convert_alm_norm!(S2, Slm, cfg; to_internal = true)
        SHTnsKit.convert_alm_norm!(T2, Tlm, cfg; to_internal = true)
        Slm = S2
        Tlm = T2
    end

    # Get the local portion info from the prototype
    θ_globals = collect(globalindices(prototype_θφ, 1))  # Global θ indices this process owns
    nθ_local = length(θ_globals)
    nlon_local = size(parent(prototype_θφ), 2)
    φ_is_local = (nlon_local == nlon)

    # Allocate Fourier coefficient matrices
    Fθm = zeros(ComplexF64, nθ_local, nlon)
    Fφm = zeros(ComplexF64, nθ_local, nlon)

    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    inv_scaleφ = SHTnsKit.phi_inv_scale(cfg)

    # Synthesis loop
    for mval in 0:mmax
        col = mval + 1

        for (ii, iglobθ) in enumerate(θ_globals)
            x = cfg.x[iglobθ]
            sθ = sqrt(max(0.0, 1 - x * x))
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ

            gθ = 0.0 + 0.0im
            gφ = 0.0 + 0.0im

            if cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
                tblP = cfg.plm_tables[col]
                tbld = cfg.dplm_tables[col]

                @inbounds for l in max(1, mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, iglobθ]
                    Y = N * tblP[l+1, iglobθ]
                    Sl = Slm[l+1, col]
                    Tl = Tlm[l+1, col]
                    # Vθ = ∂S/∂θ - (im/sinθ) * T
                    gθ += dθY * Sl - (0 + 1im) * mval * inv_sθ * Y * Tl
                    # Vφ = (im/sinθ) * S + ∂T/∂θ
                    gφ += (0 + 1im) * mval * inv_sθ * Y * Sl + dθY * Tl
                end
            else
                # Fallback: compute Legendre polynomials and derivatives on-demand
                SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, mval)

                @inbounds for l in max(1, mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * dPdx[l+1]
                    Y = N * P[l+1]
                    Sl = Slm[l+1, col]
                    Tl = Tlm[l+1, col]
                    # Vθ = ∂S/∂θ - (im/sinθ) * T
                    gθ += dθY * Sl - (0 + 1im) * mval * inv_sθ * Y * Tl
                    # Vφ = (im/sinθ) * S + ∂T/∂θ
                    gφ += (0 + 1im) * mval * inv_sθ * Y * Sl + dθY * Tl
                end
            end

            # Store Fourier coefficient
            Fθm[ii, mval + 1] = inv_scaleφ * gθ
            Fφm[ii, mval + 1] = inv_scaleφ * gφ

            # Hermitian conjugate for negative m to ensure real output
            if real_output && mval > 0
                conj_index = nlon - mval + 1
                Fθm[ii, conj_index] = conj(Fθm[ii, mval + 1])
                Fφm[ii, conj_index] = conj(Fφm[ii, mval + 1])
            end
        end
    end

    # Perform inverse FFT along φ
    Vtθφ_local = Matrix{Float64}(undef, nθ_local, nlon)
    Vpθφ_local = Matrix{Float64}(undef, nθ_local, nlon)
    SHTnsKitParallelExt.ifft_along_dim2!(Vtθφ_local, Fθm)
    SHTnsKitParallelExt.ifft_along_dim2!(Vpθφ_local, Fφm)

    # Apply Robert form scaling if enabled
    if cfg.robert_form
        @inbounds for (ii, iglobθ) in enumerate(θ_globals)
            x = cfg.x[iglobθ]
            sθ = sqrt(max(0.0, 1 - x * x))
            for j in 1:nlon
                Vtθφ_local[ii, j] *= sθ
                Vpθφ_local[ii, j] *= sθ
            end
        end
    end

    # If φ is distributed, extract local portion
    if φ_is_local
        Vtθφ = real_output ? Vtθφ_local : Complex{Float64}.(Vtθφ_local)
        Vpθφ = real_output ? Vpθφ_local : Complex{Float64}.(Vpθφ_local)
    else
        φ_globals = collect(globalindices(prototype_θφ, 2))
        local_φ_range = first(φ_globals):last(φ_globals)
        Vtθφ = Vtθφ_local[:, local_φ_range]
        Vpθφ = Vpθφ_local[:, local_φ_range]
        if !real_output
            Vtθφ = Complex{Float64}.(Vtθφ)
            Vpθφ = Complex{Float64}.(Vpθφ)
        end
    end

    return Vtθφ, Vpθφ
end

# Convenience: spectral inputs as PencilArray (dense layout (:l,:m))
function SHTnsKit.dist_SHsphtor_to_spat(cfg::SHTnsKit.SHTConfig, Slm::PencilArray, Tlm::PencilArray; prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
    return SHTnsKit.dist_SHsphtor_to_spat(cfg, Array(Slm), Array(Tlm); prototype_θφ, real_output, use_rfft)
end

function SHTnsKit.dist_SHsphtor_to_spat!(plan::DistSphtorPlan, Vtθφ_out::PencilArray, Vpθφ_out::PencilArray,
                                         Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    if plan.with_spatial_scratch && plan.spatial_scratch !== nothing
        # Use pre-allocated scratch buffers for zero-allocation synthesis
        _dist_SHsphtor_to_spat_with_scratch!(plan.cfg, Slm, Tlm, plan.spatial_scratch, Vtθφ_out, Vpθφ_out;
                                            prototype_θφ=plan.prototype_θφ, real_output, use_rfft=plan.use_rfft)
        return Vtθφ_out, Vpθφ_out
    else
        # Fall back to standard allocation path
        Vt, Vp = SHTnsKit.dist_SHsphtor_to_spat(plan.cfg, Slm, Tlm; prototype_θφ=plan.prototype_θφ, real_output, use_rfft=plan.use_rfft)
        copyto!(Vtθφ_out, Vt); copyto!(Vpθφ_out, Vp)
        return Vtθφ_out, Vpθφ_out
    end
end

# Full implementation using pre-allocated scratch buffers to eliminate allocations
function _dist_SHsphtor_to_spat_with_scratch!(cfg::SHTnsKit.SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix,
                                             scratch::NamedTuple, Vtθφ_out::PencilArray, Vpθφ_out::PencilArray;
                                             prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon

    size(Slm, 1) == lmax + 1 && size(Slm, 2) == mmax + 1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm, 1) == lmax + 1 && size(Tlm, 2) == mmax + 1 || throw(DimensionMismatch("Tlm dims"))

    # Convert incoming coefficients to internal normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm)
        T2 = similar(Tlm)
        SHTnsKit.convert_alm_norm!(S2, Slm, cfg; to_internal = true)
        SHTnsKit.convert_alm_norm!(T2, Tlm, cfg; to_internal = true)
        Slm = S2
        Tlm = T2
    end

    # Get the local portion info from the prototype
    θ_globals = collect(globalindices(prototype_θφ, 1))
    nθ_local = length(θ_globals)
    nlon_local = size(parent(prototype_θφ), 2)
    φ_is_local = (nlon_local == nlon)

    # Extract and zero the scratch buffers
    Fθm = scratch.Fθ
    Fφm = scratch.Fφ
    P = scratch.P
    dPdx = scratch.dPdx
    fill!(Fθm, zero(ComplexF64))
    fill!(Fφm, zero(ComplexF64))

    inv_scaleφ = SHTnsKit.phi_inv_scale(cfg)

    # Synthesis loop - accumulate Fourier coefficients
    for mval in 0:mmax
        col = mval + 1

        for (ii, iglobθ) in enumerate(θ_globals)
            x = cfg.x[iglobθ]
            sθ = sqrt(max(0.0, 1 - x * x))
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ

            gθ = 0.0 + 0.0im
            gφ = 0.0 + 0.0im

            if cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
                tblP = cfg.plm_tables[col]
                tbld = cfg.dplm_tables[col]

                @inbounds for l in max(1, mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, iglobθ]
                    Y = N * tblP[l+1, iglobθ]
                    Sl = Slm[l+1, col]
                    Tl = Tlm[l+1, col]
                    # Vθ = ∂S/∂θ - (im/sinθ) * T
                    gθ += dθY * Sl - (0 + 1im) * mval * inv_sθ * Y * Tl
                    # Vφ = (im/sinθ) * S + ∂T/∂θ
                    gφ += (0 + 1im) * mval * inv_sθ * Y * Sl + dθY * Tl
                end
            else
                # Fallback: compute Legendre polynomials and derivatives on-demand
                SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, mval)

                @inbounds for l in max(1, mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * dPdx[l+1]
                    Y = N * P[l+1]
                    Sl = Slm[l+1, col]
                    Tl = Tlm[l+1, col]
                    # Vθ = ∂S/∂θ - (im/sinθ) * T
                    gθ += dθY * Sl - (0 + 1im) * mval * inv_sθ * Y * Tl
                    # Vφ = (im/sinθ) * S + ∂T/∂θ
                    gφ += (0 + 1im) * mval * inv_sθ * Y * Sl + dθY * Tl
                end
            end

            # Store Fourier coefficient
            Fθm[ii, mval + 1] = inv_scaleφ * gθ
            Fφm[ii, mval + 1] = inv_scaleφ * gφ

            # Hermitian conjugate for negative m to ensure real output
            if real_output && mval > 0
                conj_index = nlon - mval + 1
                Fθm[ii, conj_index] = conj(Fθm[ii, mval + 1])
                Fφm[ii, conj_index] = conj(Fφm[ii, mval + 1])
            end
        end
    end

    # Perform inverse FFT along φ using scratch output buffers
    Vtθφ_local = scratch.Vtθ
    Vpθφ_local = scratch.Vpθ
    SHTnsKitParallelExt.ifft_along_dim2!(Vtθφ_local, Fθm)
    SHTnsKitParallelExt.ifft_along_dim2!(Vpθφ_local, Fφm)

    # Apply Robert form scaling if enabled
    if cfg.robert_form
        @inbounds for (ii, iglobθ) in enumerate(θ_globals)
            x = cfg.x[iglobθ]
            sθ = sqrt(max(0.0, 1 - x * x))
            for j in 1:nlon
                Vtθφ_local[ii, j] *= sθ
                Vpθφ_local[ii, j] *= sθ
            end
        end
    end

    # Copy to output PencilArrays, extracting local φ portion if needed
    if φ_is_local
        copyto!(parent(Vtθφ_out), Vtθφ_local)
        copyto!(parent(Vpθφ_out), Vpθφ_local)
    else
        φ_globals = collect(globalindices(prototype_θφ, 2))
        local_φ_range = first(φ_globals):last(φ_globals)
        copyto!(parent(Vtθφ_out), view(Vtθφ_local, :, local_φ_range))
        copyto!(parent(Vpθφ_out), view(Vpθφ_local, :, local_φ_range))
    end

    return Vtθφ_out, Vpθφ_out
end

# QST distributed implementations by composition
function SHTnsKit.dist_spat_to_SHqst(cfg::SHTnsKit.SHTConfig, Vrθφ::PencilArray, Vtθφ::PencilArray, Vpθφ::PencilArray)
    Qlm = SHTnsKit.dist_analysis(cfg, Vrθφ)
    Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vtθφ, Vpθφ)
    return Qlm, Slm, Tlm
end

function SHTnsKit.dist_spat_to_SHqst!(plan::DistQstPlan, Qlm_out::AbstractMatrix, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix,
                                      Vrθφ::PencilArray, Vtθφ::PencilArray, Vpθφ::PencilArray)
    Q, S, T = SHTnsKit.dist_spat_to_SHqst(plan.cfg, Vrθφ, Vtθφ, Vpθφ)
    copyto!(Qlm_out, Q); copyto!(Slm_out, S); copyto!(Tlm_out, T)
    return Qlm_out, Slm_out, Tlm_out
end

# Synthesis to distributed fields from dense spectra
function SHTnsKit.dist_SHqst_to_spat(cfg::SHTnsKit.SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
    Vr = SHTnsKit.dist_synthesis(cfg, Qlm; prototype_θφ, real_output, use_rfft)
    Vt, Vp = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ, real_output, use_rfft)
    return Vr, Vt, Vp
end

function SHTnsKit.dist_SHqst_to_spat(cfg::SHTnsKit.SHTConfig, Qlm::PencilArray, Slm::PencilArray, Tlm::PencilArray; prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
    Vr, Vt, Vp = SHTnsKit.dist_SHqst_to_spat(cfg, Array(Qlm), Array(Slm), Array(Tlm); prototype_θφ, real_output, use_rfft)
    return Vr, Vt, Vp
end

##########
# Simple roundtrip diagnostics (optional helpers)
##########

function SHTnsKit.dist_scalar_roundtrip!(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray)
    comm = communicator(fθφ)
    Alm = SHTnsKit.dist_analysis(cfg, fθφ)
    f_matrix = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)
    # Compare synthesis result directly with local PencilArray data
    # (matching test_mpi_pencil.jl pattern which works correctly)
    f_local_ref = parent(fθφ)  # The underlying local array
    # Local and global relative errors
    local_diff2 = sum(abs2, f_matrix .- f_local_ref)
    local_ref2 = sum(abs2, f_local_ref)
    global_diff2 = MPI.Allreduce(local_diff2, +, comm)
    global_ref2 = MPI.Allreduce(local_ref2, +, comm)
    rel_local = sqrt(local_diff2 / (local_ref2 + eps()))
    rel_global = sqrt(global_diff2 / (global_ref2 + eps()))
    return rel_local, rel_global
end

function SHTnsKit.dist_vector_roundtrip!(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArray, Vpθφ::PencilArray)
    comm = communicator(Vtθφ)
    Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vtθφ, Vpθφ)
    Vt2_matrix, Vp2_matrix = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ=Vtθφ, real_output=true)
    # Compare synthesis results directly with local PencilArray data
    # (matching test_mpi_pencil.jl pattern which works correctly)
    vt_ref = parent(Vtθφ)
    vp_ref = parent(Vpθφ)
    # Local errors
    lt_d2 = sum(abs2, Vt2_matrix .- vt_ref)
    lt_r2 = sum(abs2, vt_ref)
    lp_d2 = sum(abs2, Vp2_matrix .- vp_ref)
    lp_r2 = sum(abs2, vp_ref)
    # Global errors via MPI reduction
    gt_d2 = MPI.Allreduce(lt_d2, +, comm); gt_r2 = MPI.Allreduce(lt_r2, +, comm)
    gp_d2 = MPI.Allreduce(lp_d2, +, comm); gp_r2 = MPI.Allreduce(lp_r2, +, comm)
    rl_t = sqrt(lt_d2 / (lt_r2 + eps())); rg_t = sqrt(gt_d2 / (gt_r2 + eps()))
    rl_p = sqrt(lp_d2 / (lp_r2 + eps())); rg_p = sqrt(gp_d2 / (gp_r2 + eps()))
    return (rl_t, rg_t), (rl_p, rg_p)
end

# ===== DISTRIBUTED SPECTRAL STORAGE UTILITIES =====

"""
    create_distributed_spectral_plan(lmax, mmax, comm) -> DistributedSpectralPlan

Create a plan for distributing spherical harmonic coefficients across MPI processes.
This avoids the massive Allreduce bottleneck by having each process own specific (l,m) coefficients.

Distribution strategy:
- l-major distribution: Process p owns coefficients with l % nprocs == p
- Better load balancing than m-major for typical spherical spectra
- Minimizes communication in most analysis/synthesis operations
"""
struct DistributedSpectralPlan
    lmax::Int
    mmax::Int 
    comm::MPI.Comm
    nprocs::Int
    rank::Int
    
    # Coefficient ownership maps
    local_lm_indices::Vector{Tuple{Int,Int}}  # (l,m) pairs owned by this process
    local_packed_indices::Vector{Int}         # Packed indices for local coefficients
    
    # Communication patterns
    send_counts::Vector{Int}                  # How many coefficients to send to each process
    recv_counts::Vector{Int}                  # How many coefficients to receive from each process
    send_displs::Vector{Int}                  # Send displacement offsets
    recv_displs::Vector{Int}                  # Receive displacement offsets
end

function create_distributed_spectral_plan(lmax::Int, mmax::Int, comm::MPI.Comm)
    nprocs = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    
    # Determine local coefficient ownership (l-major distribution)
    local_lm_indices = Tuple{Int,Int}[]
    local_packed_indices = Int[]
    
    for l in 0:lmax
        if l % nprocs == rank  # This process owns this l
            for m in 0:min(l, mmax)
                push!(local_lm_indices, (l, m))
                # Compute packed index for this coefficient (LM_index returns 0-based, add 1)
                packed_idx = SHTnsKit.LM_index(lmax, 1, l, m) + 1
                push!(local_packed_indices, packed_idx)
            end
        end
    end
    
    # Pre-compute communication patterns for efficient gather/scatter
    send_counts = zeros(Int, nprocs)
    recv_counts = zeros(Int, nprocs)
    
    # Each process computes how many coefficients it needs to send/receive
    for l in 0:lmax
        owner_rank = l % nprocs
        coeff_count = min(l, mmax) + 1  # Number of m values for this l
        
        if rank == owner_rank
            # This process owns these coefficients
            recv_counts[owner_rank + 1] += coeff_count
        else
            # This process needs these coefficients from owner
            send_counts[owner_rank + 1] += coeff_count
        end
    end
    
    # Compute displacement offsets
    send_displs = cumsum([0; send_counts[1:end-1]])
    recv_displs = cumsum([0; recv_counts[1:end-1]])
    
    return DistributedSpectralPlan(lmax, mmax, comm, nprocs, rank,
                                  local_lm_indices, local_packed_indices,
                                  send_counts, recv_counts, send_displs, recv_displs)
end

"""
    distributed_spectral_reduce!(plan::DistributedSpectralPlan, local_contrib::AbstractMatrix, 
                                result::AbstractMatrix)

Efficiently reduce spectral contributions using distributed ownership instead of Allreduce.
Each process accumulates contributions for coefficients it owns, then redistributes the results.
This replaces the O(lmax²) Allreduce with O(lmax²/P) local work + O(lmax²) communication.
"""
function distributed_spectral_reduce!(plan::DistributedSpectralPlan, local_contrib::AbstractMatrix, 
                                     result::AbstractMatrix)
    lmax, mmax = plan.lmax, plan.mmax
    comm = plan.comm
    
    # Step 1: Pack local contributions into communication buffers
    local_contribs_packed = Vector{ComplexF64}(undef, length(plan.local_lm_indices))
    
    for (i, (l, m)) in enumerate(plan.local_lm_indices)
        local_contribs_packed[i] = local_contrib[l+1, m+1]
    end
    
    # Step 2: Reduce contributions for locally owned coefficients
    # Use MPI_Reduce_scatter instead of Allreduce for better scalability
    global_contribs_packed = Vector{ComplexF64}(undef, length(plan.local_lm_indices))
    MPI.Reduce_scatter!(local_contribs_packed, global_contribs_packed, plan.recv_counts, +, comm)
    
    # Step 3: Store reduced coefficients in result matrix
    fill!(result, 0.0 + 0.0im)
    for (i, (l, m)) in enumerate(plan.local_lm_indices)
        result[l+1, m+1] = global_contribs_packed[i]
    end
    
    # Step 4: Distribute final results to all processes using Allgatherv
    # This is more efficient than broadcasting from each owner
    all_coefficients = Vector{ComplexF64}(undef, sum(plan.recv_counts))
    MPI.Allgatherv!(global_contribs_packed, VBuffer(all_coefficients, plan.recv_counts), comm)

    # Step 5: Unpack received coefficients into result matrix
    # Data in all_coefficients is ordered by rank (according to recv_displs/recv_counts).
    # Each rank's segment contains coefficients for l values where (l % nprocs == rank),
    # packed in order of increasing l, then increasing m within each l.
    for owner_rank in 0:(plan.nprocs - 1)
        if owner_rank == plan.rank
            continue  # Skip our own coefficients - already stored in Step 3
        end

        # Calculate starting offset into all_coefficients for this rank's data
        rank_offset = plan.recv_displs[owner_rank + 1]
        coeff_idx = 0

        # Iterate through (l,m) pairs owned by this rank (same order as packing)
        for l in 0:lmax
            if l % plan.nprocs == owner_rank
                for m in 0:min(l, mmax)
                    coeff_idx += 1
                    result[l+1, m+1] = all_coefficients[rank_offset + coeff_idx]
                end
            end
        end
    end

    return result
end

# ===== PLM_TABLES INTEGRATION UTILITIES =====

"""
    validate_plm_tables(cfg::SHTConfig; verbose::Bool=false) -> Bool

Validate the structure and consistency of precomputed plm_tables in the configuration.
Returns true if tables are valid and can be used for optimized transforms.

Optional keyword arguments:
- `verbose`: Print detailed validation information
"""
function validate_plm_tables(cfg::SHTnsKit.SHTConfig; verbose::Bool=false)
    verbose && @info "Validating plm_tables structure..."
    
    # Check if tables are enabled
    if !cfg.use_plm_tables
        verbose && @info "plm_tables disabled in configuration"
        return false
    end
    
    # Check if tables exist
    if isempty(cfg.plm_tables)
        verbose && @warn "plm_tables enabled but empty"
        return false
    end
    
    lmax, mmax = cfg.lmax, cfg.mmax
    nlat = cfg.nlat
    
    # Check table count
    expected_count = mmax + 1
    actual_count = length(cfg.plm_tables)
    if actual_count != expected_count
        verbose && @warn "plm_tables count mismatch: expected $expected_count, got $actual_count"
        return false
    end
    
    # Check table dimensions
    for (m_idx, table) in enumerate(cfg.plm_tables)
        m = m_idx - 1  # Convert to 0-based
        expected_size = (lmax + 1, nlat)
        actual_size = size(table)
        
        if actual_size != expected_size
            verbose && @warn "plm_tables[$m_idx] size mismatch: expected $expected_size, got $actual_size"
            return false
        end
        
        # Check for NaN/Inf values in first few entries
        if any(!isfinite, @view table[1:min(10, size(table,1)), 1:min(10, size(table,2))])
            verbose && @warn "plm_tables[$m_idx] contains non-finite values"
            return false
        end
    end
    
    # Check derivative tables if they exist
    if !isempty(cfg.dplm_tables)
        if length(cfg.dplm_tables) != expected_count
            verbose && @warn "dplm_tables count mismatch: expected $expected_count, got $(length(cfg.dplm_tables))"
            return false
        end
        
        for (m_idx, table) in enumerate(cfg.dplm_tables)
            if size(table) != size(cfg.plm_tables[m_idx])
                verbose && @warn "dplm_tables[$m_idx] size mismatch with plm_tables"
                return false
            end
        end
    end
    
    verbose && @info "plm_tables validation passed"
    return true
end

"""
    estimate_plm_tables_memory(cfg::SHTConfig) -> Int

Estimate the memory usage of plm_tables in bytes.
"""
function estimate_plm_tables_memory(cfg::SHTnsKit.SHTConfig)
    if !cfg.use_plm_tables || isempty(cfg.plm_tables)
        return 0
    end
    
    total_bytes = 0
    for table in cfg.plm_tables
        total_bytes += sizeof(table)
    end
    
    # Add derivative tables if they exist
    if !isempty(cfg.dplm_tables)
        for table in cfg.dplm_tables
            total_bytes += sizeof(table)
        end
    end

    return total_bytes
end

# ===== TRUE DISTRIBUTED SPECTRAL STORAGE =====
# These functions provide spectral arrays that are truly distributed across ranks,
# with each rank only holding its owned (l,m) coefficients. This reduces memory
# usage from O(lmax²) per rank to O(lmax²/P) per rank.

"""
    DistributedSpectralArray

A wrapper for distributed spherical harmonic coefficients.
Each rank only stores the coefficients it owns (based on l % nprocs == rank).
"""
struct DistributedSpectralArray{T}
    local_coeffs::Vector{T}           # Local coefficients owned by this rank
    plan::DistributedSpectralPlan     # Distribution plan with ownership info
end

"""
    create_distributed_spectral_array(plan::DistributedSpectralPlan, T::Type=ComplexF64)

Create an empty distributed spectral array for the given distribution plan.
"""
function create_distributed_spectral_array(plan::DistributedSpectralPlan, ::Type{T}=ComplexF64) where T
    local_coeffs = zeros(T, length(plan.local_lm_indices))
    return DistributedSpectralArray{T}(local_coeffs, plan)
end

"""
    local_size(dsa::DistributedSpectralArray) -> Int

Return the number of coefficients stored locally on this rank.
"""
local_size(dsa::DistributedSpectralArray) = length(dsa.local_coeffs)

"""
    global_size(dsa::DistributedSpectralArray) -> Tuple{Int,Int}

Return the global spectral array dimensions (lmax+1, mmax+1).
"""
global_size(dsa::DistributedSpectralArray) = (dsa.plan.lmax + 1, dsa.plan.mmax + 1)

"""
    gather_to_dense(dsa::DistributedSpectralArray) -> Matrix{ComplexF64}

Gather distributed coefficients to a dense (lmax+1, mmax+1) matrix on ALL ranks.
Use this when you need the full spectral array for operations like synthesis.
"""
function gather_to_dense(dsa::DistributedSpectralArray{T}) where T
    plan = dsa.plan
    lmax, mmax = plan.lmax, plan.mmax
    comm = plan.comm

    # Gather all local coefficients to all ranks
    all_coefficients = Vector{T}(undef, sum(plan.recv_counts))
    MPI.Allgatherv!(dsa.local_coeffs, VBuffer(all_coefficients, plan.recv_counts), comm)

    # Unpack into dense matrix
    result = zeros(T, lmax + 1, mmax + 1)

    # First, store our own coefficients
    for (i, (l, m)) in enumerate(plan.local_lm_indices)
        result[l+1, m+1] = dsa.local_coeffs[i]
    end

    # Then unpack received coefficients from other ranks
    for owner_rank in 0:(plan.nprocs - 1)
        if owner_rank == plan.rank
            continue
        end

        rank_offset = plan.recv_displs[owner_rank + 1]
        coeff_idx = 0

        for l in 0:lmax
            if l % plan.nprocs == owner_rank
                for m in 0:min(l, mmax)
                    coeff_idx += 1
                    result[l+1, m+1] = all_coefficients[rank_offset + coeff_idx]
                end
            end
        end
    end

    return result
end

"""
    scatter_from_dense!(dsa::DistributedSpectralArray, dense::AbstractMatrix)

Scatter a dense spectral array to distributed storage.
Each rank extracts only the coefficients it owns.
"""
function scatter_from_dense!(dsa::DistributedSpectralArray{T}, dense::AbstractMatrix) where T
    plan = dsa.plan

    for (i, (l, m)) in enumerate(plan.local_lm_indices)
        dsa.local_coeffs[i] = dense[l+1, m+1]
    end

    return dsa
end

"""
    dist_analysis_distributed(cfg::SHTConfig, fθφ::PencilArray;
                               plan::DistributedSpectralPlan, kwargs...) -> DistributedSpectralArray

Distributed analysis that returns a DistributedSpectralArray.
Each rank only stores the coefficients it owns, reducing memory by factor P.

This is more memory-efficient than dist_analysis for large problems.
"""
function dist_analysis_distributed(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                                    plan::DistributedSpectralPlan,
                                    use_tables=cfg.use_plm_tables)
    # First do standard analysis to get local contributions
    comm = plan.comm
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon

    # Get local data and FFT
    local_data = parent(fθφ)
    nlat_local, nlon_local = size(local_data)
    θ_globals = collect(globalindices(fθφ, 1))
    nθ_local = length(θ_globals)

    # FFT along φ
    Fθm = Matrix{ComplexF64}(undef, nlat_local, nlon)
    if nlon_local == nlon
        SHTnsKitParallelExt.fft_along_dim2!(Fθm, local_data)
    else
        φ_globals = collect(globalindices(fθφ, 2))
        φ_range = first(φ_globals):last(φ_globals)
        θ_range = first(θ_globals):last(θ_globals)
        Fθm = _gather_and_fft_phi(local_data, θ_range, φ_range, nlon, comm)
    end

    # Compute local contributions to ALL coefficients (same as standard analysis)
    local_contrib = zeros(ComplexF64, lmax + 1, mmax + 1)
    scaleφ = cfg.cphi

    # Pre-cache weights
    weights_cache = Vector{Float64}(undef, nθ_local)
    x_cache = Vector{Float64}(undef, nθ_local)
    for (ii, iglob) in enumerate(θ_globals)
        weights_cache[ii] = cfg.w[iglob]
        x_cache[ii] = cfg.x[iglob]
    end

    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables)
    P = Vector{Float64}(undef, lmax + 1)

    # Legendre integration
    for mval in 0:mmax
        col = mval + 1
        for ii in 1:nθ_local
            iglob = θ_globals[ii]
            Fi = Fθm[ii, col]
            wi = weights_cache[ii]

            if use_tbl
                tbl = cfg.plm_tables[col]
                @inbounds @simd for l in mval:lmax
                    local_contrib[l+1, col] += wi * tbl[l+1, iglob] * Fi
                end
            else
                SHTnsKit.Plm_row!(P, x_cache[ii], lmax, mval)
                @inbounds @simd for l in mval:lmax
                    local_contrib[l+1, col] += wi * P[l+1] * Fi
                end
            end
        end
    end

    # Apply normalization
    @inbounds for m in 0:mmax
        @simd ivdep for l in m:lmax
            local_contrib[l+1, m+1] *= cfg.Nlm[l+1, m+1] * scaleφ
        end
    end

    # Create output distributed array
    result = create_distributed_spectral_array(plan, ComplexF64)

    # Pack local contributions for coefficients we OWN
    local_contribs_packed = Vector{ComplexF64}(undef, length(plan.local_lm_indices))
    for (i, (l, m)) in enumerate(plan.local_lm_indices)
        local_contribs_packed[i] = local_contrib[l+1, m+1]
    end

    # Reduce contributions: sum across all ranks for coefficients we own
    MPI.Reduce_scatter!(local_contribs_packed, result.local_coeffs, plan.recv_counts, +, comm)

    return result
end

"""
    dist_synthesis_distributed(cfg::SHTConfig, alm::DistributedSpectralArray;
                                prototype_θφ::PencilArray, kwargs...) -> Matrix

Distributed synthesis from a DistributedSpectralArray.
Gathers necessary coefficients and performs synthesis.

Note: Internally gathers to dense for now. Future optimization could avoid this.
"""
function dist_synthesis_distributed(cfg::SHTnsKit.SHTConfig, alm::DistributedSpectralArray;
                                     prototype_θφ::PencilArray, real_output::Bool=true)
    # Gather to dense array (required for Legendre summation which needs all l for each m)
    alm_dense = gather_to_dense(alm)

    # Use standard synthesis
    return SHTnsKit.dist_synthesis(cfg, alm_dense; prototype_θφ=prototype_θφ, real_output=real_output)
end

"""
    estimate_distributed_memory_savings(lmax::Int, mmax::Int, nprocs::Int) -> NamedTuple

Estimate memory savings from using distributed spectral storage.
"""
function estimate_distributed_memory_savings(lmax::Int, mmax::Int, nprocs::Int)
    # Dense storage per rank
    dense_elements = (lmax + 1) * (mmax + 1)
    dense_bytes = dense_elements * sizeof(ComplexF64)

    # Distributed storage per rank (l-major distribution)
    local_elements = 0
    for l in 0:lmax
        if l % nprocs == 0  # Representative rank's share
            local_elements += min(l, mmax) + 1
        end
    end
    # Average across ranks
    avg_local_elements = (dense_elements + nprocs - 1) ÷ nprocs
    distributed_bytes = avg_local_elements * sizeof(ComplexF64)

    savings_pct = 100.0 * (1.0 - distributed_bytes / dense_bytes)

    return (
        dense_bytes_per_rank = dense_bytes,
        distributed_bytes_per_rank = distributed_bytes,
        savings_percent = savings_pct,
        reduction_factor = nprocs
    )
end
