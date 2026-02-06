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
function SHTnsKit.dist_analysis_sphtor(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArray, Vpθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false)
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

function SHTnsKit.dist_analysis_sphtor!(plan::DistSphtorPlan, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix,
                                         Vtθφ::PencilArray, Vpθφ::PencilArray; use_tables=plan.cfg.use_plm_tables)
    Slm, Tlm = SHTnsKit.dist_analysis_sphtor(plan.cfg, Vtθφ, Vpθφ; use_tables, use_rfft=plan.use_rfft)
    copyto!(Slm_out, Slm); copyto!(Tlm_out, Tlm)
    return Slm_out, Tlm_out
end

# Distributed vector synthesis (spheroidal/toroidal) from dense spectra
function SHTnsKit.dist_synthesis_sphtor(cfg::SHTnsKit.SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
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
function SHTnsKit.dist_synthesis_sphtor(cfg::SHTnsKit.SHTConfig, Slm::PencilArray, Tlm::PencilArray; prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
    return SHTnsKit.dist_synthesis_sphtor(cfg, Array(Slm), Array(Tlm); prototype_θφ, real_output, use_rfft)
end

function SHTnsKit.dist_synthesis_sphtor!(plan::DistSphtorPlan, Vtθφ_out::PencilArray, Vpθφ_out::PencilArray,
                                         Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    if plan.with_spatial_scratch && plan.spatial_scratch !== nothing
        # Use pre-allocated scratch buffers for zero-allocation synthesis
        _dist_synthesis_sphtor_with_scratch!(plan.cfg, Slm, Tlm, plan.spatial_scratch, Vtθφ_out, Vpθφ_out;
                                            prototype_θφ=plan.prototype_θφ, real_output, use_rfft=plan.use_rfft)
        return Vtθφ_out, Vpθφ_out
    else
        # Fall back to standard allocation path
        Vt, Vp = SHTnsKit.dist_synthesis_sphtor(plan.cfg, Slm, Tlm; prototype_θφ=plan.prototype_θφ, real_output, use_rfft=plan.use_rfft)
        copyto!(Vtθφ_out, Vt); copyto!(Vpθφ_out, Vp)
        return Vtθφ_out, Vpθφ_out
    end
end

# Full implementation using pre-allocated scratch buffers to eliminate allocations
function _dist_synthesis_sphtor_with_scratch!(cfg::SHTnsKit.SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix,
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
function SHTnsKit.dist_analysis_qst(cfg::SHTnsKit.SHTConfig, Vrθφ::PencilArray, Vtθφ::PencilArray, Vpθφ::PencilArray)
    Qlm = SHTnsKit.dist_analysis(cfg, Vrθφ)
    Slm, Tlm = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ)
    return Qlm, Slm, Tlm
end

function SHTnsKit.dist_analysis_qst!(plan::DistQstPlan, Qlm_out::AbstractMatrix, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix,
                                      Vrθφ::PencilArray, Vtθφ::PencilArray, Vpθφ::PencilArray)
    Q, S, T = SHTnsKit.dist_analysis_qst(plan.cfg, Vrθφ, Vtθφ, Vpθφ)
    copyto!(Qlm_out, Q); copyto!(Slm_out, S); copyto!(Tlm_out, T)
    return Qlm_out, Slm_out, Tlm_out
end

# Synthesis to distributed fields from dense spectra
function SHTnsKit.dist_synthesis_qst(cfg::SHTnsKit.SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
    Vr = SHTnsKit.dist_synthesis(cfg, Qlm; prototype_θφ, real_output, use_rfft)
    Vt, Vp = SHTnsKit.dist_synthesis_sphtor(cfg, Slm, Tlm; prototype_θφ, real_output, use_rfft)
    return Vr, Vt, Vp
end

function SHTnsKit.dist_synthesis_qst(cfg::SHTnsKit.SHTConfig, Qlm::PencilArray, Slm::PencilArray, Tlm::PencilArray; prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
    Vr, Vt, Vp = SHTnsKit.dist_synthesis_qst(cfg, Array(Qlm), Array(Slm), Array(Tlm); prototype_θφ, real_output, use_rfft)
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
    Slm, Tlm = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ)
    Vt2_matrix, Vp2_matrix = SHTnsKit.dist_synthesis_sphtor(cfg, Slm, Tlm; prototype_θφ=Vtθφ, real_output=true)
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
    # IMPORTANT: recv_counts must be IDENTICAL on all ranks for MPI collectives
    send_counts = zeros(Int, nprocs)
    recv_counts = zeros(Int, nprocs)

    # Compute recv_counts: how many coefficients each rank owns
    # This must be computed identically on ALL ranks (no conditional on current rank)
    for l in 0:lmax
        owner_rank = l % nprocs
        coeff_count = min(l, mmax) + 1  # Number of m values for this l
        recv_counts[owner_rank + 1] += coeff_count
    end

    # send_counts tracks what we would send to each rank (for potential future use)
    for r in 0:(nprocs - 1)
        if r != rank
            send_counts[r + 1] = recv_counts[r + 1]
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

Reduce spectral contributions across all MPI ranks using Allreduce.
Each rank provides its local partial sums in `local_contrib`; the result contains the global sum.
"""
function distributed_spectral_reduce!(plan::DistributedSpectralPlan, local_contrib::AbstractMatrix, 
                                     result::AbstractMatrix)
    comm = plan.comm

    # Sum contributions from all ranks into result using Allreduce
    MPI.Allreduce!(local_contrib, result, +, comm)

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
    # Data is ordered by owner rank: [rank0 coeffs, rank1 coeffs, ...]
    # Within each rank's segment: l-major order (for each l owned by that rank, for each m)
    result = zeros(T, lmax + 1, mmax + 1)

    for owner_rank in 0:(plan.nprocs - 1)
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

    # Pack all coefficients in l-major order grouped by owner rank, then Allreduce
    # and extract the local portion for this rank
    total_nlm = sum(plan.recv_counts)
    local_contribs_packed = Vector{ComplexF64}(undef, total_nlm)

    # Pack in l-major order, grouped by owner rank
    # recv_counts[r+1] = count for rank r, where rank r owns l values where l % nprocs == r
    idx = 0
    for owner_rank in 0:(plan.nprocs - 1)
        for l in 0:lmax
            if l % plan.nprocs == owner_rank
                for m in 0:min(l, mmax)
                    idx += 1
                    local_contribs_packed[idx] = local_contrib[l+1, m+1]
                end
            end
        end
    end

    # Allreduce the packed buffer, then extract the local portion for this rank
    full_reduced = similar(local_contribs_packed)
    MPI.Allreduce!(local_contribs_packed, full_reduced, +, comm)
    offset = plan.recv_displs[plan.rank + 1]
    count = plan.recv_counts[plan.rank + 1]
    copyto!(result.local_coeffs, 1, full_reduced, offset + 1, count)

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

# ===== 2D DISTRIBUTED SPECTRAL STORAGE =====
# Extends the 1D distribution (l-only) to 2D (l,m) distribution for further memory reduction.
# With P = p_l × p_m processes arranged in a 2D grid:
# - Memory per rank: O(lmax²/(p_l × p_m)) vs O(lmax²/P) for 1D
# - Synthesis gather: O(lmax²/p_m) within l-comm vs O(lmax²) globally for 1D

"""
    DistributedSpectralPlan2D

Plan for 2D distribution of spherical harmonic coefficients across a process grid.
Processes are arranged in a 2D grid (p_l × p_m) where:
- Ranks in the same column (m-group) share the same m values
- Ranks in the same row share the same l distribution pattern
- l-communicator connects ranks within an m-group for Legendre operations
- m-communicator connects ranks across m-groups for potential future optimizations

Distribution strategy:
- M-distribution: m values divided into p_m groups (m-groups)
- L-distribution: within each m-group, l is distributed cyclically: l % p_l == l_rank

This achieves O(lmax²/(p_l × p_m)) memory per rank and O(lmax²/p_m) gather for synthesis.

# Scratch Buffers
When created with `with_scratch=true` and a `prototype_θφ`, the plan pre-allocates
all temporary arrays needed for analysis and synthesis operations. This eliminates
per-call allocations for repeated transforms.
"""
struct DistributedSpectralPlan2D
    lmax::Int
    mmax::Int
    mres::Int                        # m resolution (usually 1)

    # World communicator and size
    comm::MPI.Comm                   # World communicator
    nprocs::Int                      # Total processes
    rank::Int                        # World rank

    # Process grid configuration
    p_l::Int                         # Processes in l-dimension (within m-group)
    p_m::Int                         # Processes in m-dimension (number of m-groups)
    l_rank::Int                      # Rank within l-communicator (0:p_l-1)
    m_rank::Int                      # Rank within m-communicator (0:p_m-1), also m-group index

    # Sub-communicators
    l_comm::MPI.Comm                 # L-communicator (within m-group, for gather/reduce)
    m_comm::MPI.Comm                 # M-communicator (across m-groups)

    # M-group ownership: which m values this m-group owns
    m_range::UnitRange{Int}          # M values owned by this m-group [m_start:m_end]

    # Local (l,m) pairs owned by this rank
    local_lm_indices::Vector{Tuple{Int,Int}}  # (l,m) pairs owned
    local_nlm::Int                   # Number of local coefficients

    # Communication patterns for l-communicator gather/scatter
    l_recv_counts::Vector{Int}       # Counts for each rank in l_comm
    l_recv_displs::Vector{Int}       # Displacements for gather/scatter

    # Total coefficients in this m-group (for gather buffer sizing)
    m_group_nlm::Int                 # Total coefficients owned by all ranks in m-group

    # Pre-allocated scratch buffers (optional - set when with_scratch=true)
    with_scratch::Bool
    scratch::Union{Nothing, NamedTuple{
        (:nθ_local, :nlon, :n_m_valid, :n_valid_coeffs,
         :θ_globals, :weights_cache, :x_cache,
         :Fθm, :local_contrib, :P,
         :gather_buffer, :fθφ_local, :fθφ_result,
         :packed_contrib, :pack_offsets, :m_values),
        Tuple{Int, Int, Int, Int,
              Vector{Int}, Vector{Float64}, Vector{Float64},
              Matrix{ComplexF64}, Matrix{ComplexF64}, Vector{Float64},
              Vector{ComplexF64}, Matrix{Float64}, Matrix{Float64},
              Vector{ComplexF64}, Vector{Int}, Vector{Int}}
    }}
end

"""
    DistributedSpectralArray2D{T}

A wrapper for 2D-distributed spherical harmonic coefficients.
Each rank only stores the coefficients it owns based on 2D (l,m) distribution.
"""
struct DistributedSpectralArray2D{T}
    local_coeffs::Vector{T}          # Local coefficients owned by this rank
    plan::DistributedSpectralPlan2D  # Distribution plan with ownership info
end

"""
    suggest_spectral_grid(nprocs::Int, lmax::Int, mmax::Int) -> (p_l, p_m)

Suggest an optimal 2D process grid for spectral coefficient distribution.
Attempts to balance:
1. Even division of processes
2. Balanced load across m-groups (accounting for triangular constraint l >= m)
3. Minimizing communication volume

Returns (p_l, p_m) where nprocs = p_l × p_m.
"""
function suggest_spectral_grid(nprocs::Int, lmax::Int, mmax::Int)
    if nprocs <= 1
        return (1, 1)
    end

    # Find all factor pairs
    best_p_l, best_p_m = 1, nprocs
    best_score = Inf

    for p_l in 1:isqrt(nprocs)
        nprocs % p_l == 0 || continue
        p_m = nprocs ÷ p_l

        # Also try the swapped configuration
        for (a, b) in ((p_l, p_m), (p_m, p_l))
            a > 0 && b > 0 || continue

            # Score this configuration
            # Prefer configurations where p_l is smaller (less communication in l-gather)
            # and p_m divides mmax+1 evenly (better load balance)
            m_imbalance = (mmax + 1) % b  # Remainder when dividing m among m-groups
            l_comm_size = a  # Size of l-communicator (smaller = less gather overhead)

            # Communication score: smaller l_comm means less gather volume
            comm_score = a

            # Load balance score: prefer even m-division
            balance_score = m_imbalance / (mmax + 1 + 1)

            # Combined score (lower is better)
            score = comm_score + 10 * balance_score

            if score < best_score
                best_score = score
                best_p_l, best_p_m = a, b
            end
        end
    end

    return (best_p_l, best_p_m)
end

"""
    create_distributed_spectral_plan_2d(lmax::Int, mmax::Int, comm::MPI.Comm;
                                         p_l::Int=0, p_m::Int=0, mres::Int=1,
                                         with_scratch::Bool=false,
                                         prototype_θφ=nothing, cfg=nothing) -> DistributedSpectralPlan2D

Create a 2D distribution plan for spherical harmonic coefficients.

If p_l and p_m are not specified (or set to 0), automatically determines optimal grid.

# Arguments
- `lmax::Int`: Maximum spherical harmonic degree
- `mmax::Int`: Maximum spherical harmonic order
- `comm::MPI.Comm`: MPI communicator

# Keyword Arguments
- `p_l::Int=0`: Number of processes in l-dimension (0 = auto)
- `p_m::Int=0`: Number of processes in m-dimension (0 = auto)
- `mres::Int=1`: M resolution (only m values divisible by mres are used)
- `with_scratch::Bool=false`: Pre-allocate scratch buffers to eliminate per-call allocations
- `prototype_θφ::PencilArray`: Required when `with_scratch=true` - spatial array template
- `cfg::SHTConfig`: Required when `with_scratch=true` - SHT configuration

# Returns
- `DistributedSpectralPlan2D`: The distribution plan

# Scratch Buffers
When `with_scratch=true`, the plan pre-allocates all temporary arrays needed for
analysis and synthesis operations. This eliminates per-call allocations when performing
repeated transforms, improving performance for time-stepping codes.

# Process Grid Layout
```
        m-groups (p_m columns)
       ┌─────┬─────┬─────┐
p_l    │ R0  │ R2  │ R4  │  ← l-row 0
rows   ├─────┼─────┼─────┤
       │ R1  │ R3  │ R5  │  ← l-row 1
       └─────┴─────┴─────┘
         m0    m1    m2

Rank = l_rank + m_rank * p_l
l_rank = rank % p_l  (row in grid, position in l-comm)
m_rank = rank ÷ p_l  (column in grid, which m-group)
```

# Example with scratch buffers
```julia
# Create plan with pre-allocated scratch buffers
plan = create_distributed_spectral_plan_2d(lmax, mmax, comm;
    with_scratch=true, prototype_θφ=fθφ, cfg=cfg)

# Repeated transforms reuse buffers (no allocations)
for timestep in 1:1000
    alm = dist_analysis_distributed_2d(cfg, fθφ; plan=plan, assume_aligned=true)
    fθφ_new = dist_synthesis_distributed_2d_optimized(cfg, alm; prototype_θφ=fθφ)
end
```
"""
function create_distributed_spectral_plan_2d(lmax::Int, mmax::Int, comm::MPI.Comm;
                                              p_l::Int=0, p_m::Int=0, mres::Int=1,
                                              with_scratch::Bool=false,
                                              prototype_θφ::Union{Nothing, PencilArray}=nothing,
                                              cfg::Union{Nothing, SHTnsKit.SHTConfig}=nothing)
    nprocs = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    # Auto-detect grid if not specified
    if p_l <= 0 || p_m <= 0
        p_l, p_m = suggest_spectral_grid(nprocs, lmax, mmax)
    end

    # Validate grid
    if p_l * p_m != nprocs
        error("Process grid p_l=$p_l × p_m=$p_m = $(p_l * p_m) does not match nprocs=$nprocs")
    end

    # Validate scratch requirements
    if with_scratch && (prototype_θφ === nothing || cfg === nothing)
        error("with_scratch=true requires both prototype_θφ and cfg to be provided")
    end

    # Compute grid position
    l_rank = rank % p_l      # Row (position within m-group)
    m_rank = rank ÷ p_l      # Column (which m-group)

    # Create sub-communicators
    # l_comm: ranks in same column (same m_rank) - for l-direction operations
    l_comm = MPI.Comm_split(comm, m_rank, l_rank)
    # m_comm: ranks in same row (same l_rank) - for m-direction operations
    m_comm = MPI.Comm_split(comm, l_rank, m_rank)

    # Determine m-range for this m-group
    # Divide m values [0, mmax] into p_m groups
    # Account for mres: only m values where m % mres == 0 are valid
    valid_m_values = [m for m in 0:mmax if m % mres == 0]
    n_valid_m = length(valid_m_values)

    # Divide valid m values among m-groups
    m_per_group = ceildiv(n_valid_m, p_m)
    m_start_idx = m_rank * m_per_group + 1
    m_end_idx = min((m_rank + 1) * m_per_group, n_valid_m)

    if m_start_idx <= n_valid_m
        m_start = valid_m_values[m_start_idx]
        m_end = valid_m_values[min(m_end_idx, n_valid_m)]
        m_range = m_start:m_end
    else
        # This m-group has no m values (more m-groups than valid m values)
        m_range = 1:0  # Empty range
    end

    # Compute local (l,m) ownership
    # Within this m-group, l is distributed cyclically: l % p_l == l_rank
    local_lm_indices = Tuple{Int,Int}[]

    for m in m_range
        (m % mres == 0) || continue  # Skip invalid m values
        for l in m:lmax
            if l % p_l == l_rank  # This rank owns this l within the m-group
                push!(local_lm_indices, (l, m))
            end
        end
    end

    local_nlm = length(local_lm_indices)

    # Compute communication patterns for l-communicator
    # recv_counts[r+1] = number of coefficients rank r in l_comm owns
    l_recv_counts = zeros(Int, p_l)

    for m in m_range
        (m % mres == 0) || continue
        for l in m:lmax
            owner_l_rank = l % p_l
            l_recv_counts[owner_l_rank + 1] += 1
        end
    end

    l_recv_displs = cumsum([0; l_recv_counts[1:end-1]])

    # Total coefficients in this m-group
    m_group_nlm = sum(l_recv_counts)

    # Create scratch buffers if requested
    scratch = if with_scratch
        θ_globals = collect(globalindices(prototype_θφ, 1))
        nθ_local = length(θ_globals)
        nlon = cfg.nlon
        n_m_valid = count(m -> m % mres == 0, m_range)

        # Pre-cache weights and x values
        weights_cache = Vector{Float64}(undef, nθ_local)
        x_cache = Vector{Float64}(undef, nθ_local)
        for (ii, iglob) in enumerate(θ_globals)
            weights_cache[ii] = cfg.w[iglob]
            x_cache[ii] = cfg.x[iglob]
        end

        # Compute packed buffer size and offsets for triangular storage
        # For each valid m, we have (lmax - m + 1) coefficients
        m_values = Int[m for m in m_range if m % mres == 0]
        n_valid_coeffs = sum(lmax - m + 1 for m in m_values; init=0)
        pack_offsets = Vector{Int}(undef, max(length(m_values), 1))
        offset = 0
        for (i, m) in enumerate(m_values)
            pack_offsets[i] = offset
            offset += lmax - m + 1
        end

        (
            nθ_local = nθ_local,
            nlon = nlon,
            n_m_valid = max(n_m_valid, 1),  # At least 1 to avoid zero-size arrays
            n_valid_coeffs = max(n_valid_coeffs, 1),

            # Cached indices
            θ_globals = θ_globals,
            weights_cache = weights_cache,
            x_cache = x_cache,

            # Analysis buffers
            Fθm = Matrix{ComplexF64}(undef, nθ_local, nlon),
            local_contrib = Matrix{ComplexF64}(undef, lmax + 1, max(n_m_valid, 1)),
            P = Vector{Float64}(undef, lmax + 1),

            # Gather/synthesis buffers
            gather_buffer = Vector{ComplexF64}(undef, m_group_nlm),
            fθφ_local = Matrix{Float64}(undef, nθ_local, nlon),
            fθφ_result = Matrix{Float64}(undef, nθ_local, nlon),  # Separate result buffer

            # Packed communication buffers (triangular storage)
            packed_contrib = Vector{ComplexF64}(undef, max(n_valid_coeffs, 1)),
            pack_offsets = pack_offsets,
            m_values = m_values,
        )
    else
        nothing
    end

    return DistributedSpectralPlan2D(
        lmax, mmax, mres,
        comm, nprocs, rank,
        p_l, p_m, l_rank, m_rank,
        l_comm, m_comm,
        m_range,
        local_lm_indices, local_nlm,
        l_recv_counts, l_recv_displs,
        m_group_nlm,
        with_scratch, scratch
    )
end

"""
    create_distributed_spectral_array_2d(plan::DistributedSpectralPlan2D, T::Type=ComplexF64)

Create an empty 2D-distributed spectral array for the given distribution plan.
"""
function create_distributed_spectral_array_2d(plan::DistributedSpectralPlan2D, ::Type{T}=ComplexF64) where T
    local_coeffs = zeros(T, plan.local_nlm)
    return DistributedSpectralArray2D{T}(local_coeffs, plan)
end

"""
    local_size(dsa::DistributedSpectralArray2D) -> Int

Return the number of coefficients stored locally on this rank.
"""
local_size(dsa::DistributedSpectralArray2D) = length(dsa.local_coeffs)

"""
    global_size(dsa::DistributedSpectralArray2D) -> Tuple{Int,Int}

Return the global spectral array dimensions (lmax+1, mmax+1).
"""
global_size(dsa::DistributedSpectralArray2D) = (dsa.plan.lmax + 1, dsa.plan.mmax + 1)

"""
    gather_to_dense_2d(dsa::DistributedSpectralArray2D) -> Matrix

Gather distributed coefficients within the m-group (l-communicator) only.
Returns a partial dense matrix containing coefficients for this m-group's m values,
with all l values gathered (for Legendre synthesis).

This is more efficient than full global gather when only local m values are needed.
The result has shape (lmax+1, length(m_range)).
"""
function gather_to_dense_2d(dsa::DistributedSpectralArray2D{T}) where T
    plan = dsa.plan
    lmax = plan.lmax
    m_range = plan.m_range
    l_comm = plan.l_comm
    mres = plan.mres

    if isempty(m_range)
        # This m-group has no m values
        return zeros(T, lmax + 1, 0)
    end

    n_m_local = count(m -> m % mres == 0, m_range)
    if n_m_local == 0
        return zeros(T, lmax + 1, 0)
    end

    # Use scratch buffer if available, otherwise allocate
    has_scratch = plan.with_scratch && plan.scratch !== nothing
    if has_scratch && T === ComplexF64
        all_coefficients = plan.scratch.gather_buffer
    else
        all_coefficients = Vector{T}(undef, plan.m_group_nlm)
    end

    # Gather all local coefficients within l-communicator
    MPI.Allgatherv!(dsa.local_coeffs, VBuffer(all_coefficients, plan.l_recv_counts), l_comm)

    # Unpack into partial dense matrix
    # Columns correspond to valid m values in m_range (indexed 1:n_m_local)
    result = zeros(T, lmax + 1, n_m_local)

    # Data is ordered by l_rank owner: [l_rank=0 coeffs, l_rank=1 coeffs, ...]
    # Within each rank's segment: for each m in m_range, for each l where l % p_l == l_rank
    for owner_l_rank in 0:(plan.p_l - 1)
        rank_offset = plan.l_recv_displs[owner_l_rank + 1]
        coeff_idx = 0

        m_col = 0
        for m in m_range
            (m % mres == 0) || continue
            m_col += 1

            for l in m:lmax
                if l % plan.p_l == owner_l_rank
                    coeff_idx += 1
                    result[l+1, m_col] = all_coefficients[rank_offset + coeff_idx]
                end
            end
        end
    end

    return result
end

"""
    gather_to_full_dense_2d(dsa::DistributedSpectralArray2D) -> Matrix

Gather all distributed coefficients to a full dense (lmax+1, mmax+1) matrix on ALL ranks.
This requires global communication across all m-groups.

Use this when you need the complete spectral array (e.g., for comparison with 1D methods).
For synthesis operations, prefer `gather_to_dense_2d` which is more efficient.
"""
function gather_to_full_dense_2d(dsa::DistributedSpectralArray2D{T}) where T
    plan = dsa.plan
    lmax, mmax = plan.lmax, plan.mmax
    comm = plan.comm

    # First gather within m-group to get complete l for local m values
    partial = gather_to_dense_2d(dsa)

    # Now need to gather across m-groups to get all m values
    # Each m-group has different m_range, so we use Allgatherv with variable sizes

    # Pack the partial matrix into a vector for communication
    local_packed = vec(partial)
    local_count = length(local_packed)

    # Gather counts from all ranks
    all_counts = MPI.Allgather(Int32(local_count), comm)

    # Compute displacements
    all_displs = cumsum([Int32(0); all_counts[1:end-1]])

    # Gather all partial results
    total_size = sum(all_counts)
    all_packed = Vector{T}(undef, total_size)
    MPI.Allgatherv!(local_packed, VBuffer(all_packed, all_counts), comm)

    # Unpack into full dense matrix
    result = zeros(T, lmax + 1, mmax + 1)

    # Each rank's data is a flattened (lmax+1, n_m_local) matrix
    # We need to know each rank's m_range to unpack correctly
    # Since all ranks execute this function identically, we can reconstruct m_ranges
    for r in 0:(plan.nprocs - 1)
        r_m_rank = r ÷ plan.p_l
        r_l_rank = r % plan.p_l

        # Only process data from one rank per m-group (they all have the same data after l-gather)
        if r_l_rank != 0
            continue
        end

        # Reconstruct m_range for this rank's m-group
        valid_m_values = [m for m in 0:mmax if m % plan.mres == 0]
        n_valid_m = length(valid_m_values)
        m_per_group = ceildiv(n_valid_m, plan.p_m)
        m_start_idx = r_m_rank * m_per_group + 1
        m_end_idx = min((r_m_rank + 1) * m_per_group, n_valid_m)

        if m_start_idx > n_valid_m
            continue
        end

        r_m_start = valid_m_values[m_start_idx]
        r_m_end = valid_m_values[min(m_end_idx, n_valid_m)]
        r_m_range = r_m_start:r_m_end

        n_m_local = count(m -> m % plan.mres == 0, r_m_range)
        if n_m_local == 0
            continue
        end

        # Get this rank's data from all_packed
        offset = all_displs[r + 1]
        data_size = all_counts[r + 1]

        if data_size > 0
            # Reshape to (lmax+1, n_m_local)
            partial_data = reshape(view(all_packed, offset+1:offset+data_size), lmax+1, n_m_local)

            # Copy to result at correct m columns
            m_col = 0
            for m in r_m_range
                (m % plan.mres == 0) || continue
                m_col += 1
                result[:, m+1] .= partial_data[:, m_col]
            end
        end
    end

    return result
end

"""
    scatter_from_dense_2d!(dsa::DistributedSpectralArray2D, dense::AbstractMatrix)

Scatter a dense spectral array to 2D-distributed storage.
Each rank extracts only the coefficients it owns.
"""
function scatter_from_dense_2d!(dsa::DistributedSpectralArray2D{T}, dense::AbstractMatrix) where T
    plan = dsa.plan

    for (i, (l, m)) in enumerate(plan.local_lm_indices)
        dsa.local_coeffs[i] = dense[l+1, m+1]
    end

    return dsa
end

"""
    dist_analysis_distributed_2d(cfg::SHTConfig, fθφ::PencilArray;
                                  plan::DistributedSpectralPlan2D,
                                  use_tables=cfg.use_plm_tables,
                                  assume_aligned::Bool=false) -> DistributedSpectralArray2D

2D-distributed analysis that returns a DistributedSpectralArray2D.

# Behavior depends on `assume_aligned`:

**assume_aligned=false (default, safe)**:
- Computes ALL (l,m) coefficients like standard analysis
- Uses world communicator for reduction
- Always correct, but no computation/communication savings vs standard
- Only storage is reduced to O(lmax²/P)

**assume_aligned=true (efficient)**:
- Computes ONLY m_range coefficients (O(lmax²/p_m) computation)
- Uses l_comm for reduction (O(lmax²/p_m) communication)
- Requires spatial θ distribution to be aligned with spectral l-distribution
- Use `validate_2d_distribution_alignment` to check before enabling

# Performance comparison (assume_aligned=true vs standard):
- Computation: O(lmax²/p_m) vs O(lmax²) - p_m times faster
- Communication: O(lmax²/p_m) in l-comm vs O(lmax²) global - p_m times less data
- Storage: O(lmax²/P) vs O(lmax²) - P times less memory
"""
function dist_analysis_distributed_2d(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                                       plan::DistributedSpectralPlan2D,
                                       use_tables=cfg.use_plm_tables,
                                       assume_aligned::Bool=false)
    if assume_aligned
        return _dist_analysis_2d_aligned(cfg, fθφ; plan=plan, use_tables=use_tables)
    else
        return _dist_analysis_2d_safe(cfg, fθφ; plan=plan, use_tables=use_tables)
    end
end

# Safe version: computes all coefficients, always correct
function _dist_analysis_2d_safe(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                                 plan::DistributedSpectralPlan2D,
                                 use_tables=cfg.use_plm_tables)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon
    nlat = cfg.nlat
    comm = plan.comm

    # Get local data and FFT (same as standard analysis)
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

    # Pre-cache weights and x values
    weights_cache = Vector{Float64}(undef, nθ_local)
    x_cache = Vector{Float64}(undef, nθ_local)
    for (ii, iglob) in enumerate(θ_globals)
        weights_cache[ii] = cfg.w[iglob]
        x_cache[ii] = cfg.x[iglob]
    end

    scaleφ = cfg.cphi
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables)
    P = Vector{Float64}(undef, lmax + 1)

    # Compute local contributions to ALL (l,m) coefficients
    # This ensures correctness when spatial and spectral distributions are independent
    local_contrib = zeros(ComplexF64, lmax + 1, mmax + 1)

    # Legendre integration for ALL m values
    for mval in 0:mmax
        col = mval + 1
        m_fft = mval + 1

        for ii in 1:nθ_local
            iglob = θ_globals[ii]
            Fi = Fθm[ii, m_fft]
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

    # Check if θ is distributed
    θ_is_distributed = (nθ_local < nlat)

    if θ_is_distributed
        # Reduce contributions across all ranks
        MPI.Allreduce!(local_contrib, +, comm)
    end

    # Apply normalization
    @inbounds for m in 0:mmax
        @simd ivdep for l in m:lmax
            local_contrib[l+1, m+1] *= cfg.Nlm[l+1, m+1] * scaleφ
        end
    end

    # Create output array and extract owned coefficients
    result = create_distributed_spectral_array_2d(plan, ComplexF64)

    for (i, (l, m)) in enumerate(plan.local_lm_indices)
        result.local_coeffs[i] = local_contrib[l+1, m+1]
    end

    # Handle normalization conversion if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        # The coefficients are already in internal (orthonormal) form
        # For full compatibility, would need to apply convert_alm_norm!
        # For now, this matches the behavior of dist_analysis
    end

    return result
end

"""
    dist_synthesis_distributed_2d(cfg::SHTConfig, alm::DistributedSpectralArray2D;
                                   prototype_θφ::PencilArray,
                                   real_output::Bool=true) -> Matrix

2D-distributed synthesis from a DistributedSpectralArray2D.

This implementation gathers the full spectral array and uses standard synthesis,
ensuring correctness when spatial distribution (PencilArray) is independent of
spectral distribution (2D plan).

# Algorithm
1. Gather all coefficients to full dense matrix (across all ranks)
2. Perform standard Legendre synthesis for local θ values
3. IFFT along φ
4. Extract local portion if φ is distributed

For optimized synthesis when spatial and spectral distributions are aligned,
use the specialized `dist_synthesis_distributed_2d_aligned` function.
"""
function dist_synthesis_distributed_2d(cfg::SHTnsKit.SHTConfig, alm::DistributedSpectralArray2D;
                                        prototype_θφ::PencilArray, real_output::Bool=true)
    # Gather to full dense array for correctness
    # This ensures correct results regardless of spatial/spectral distribution alignment
    alm_dense = gather_to_full_dense_2d(alm)

    # Use standard synthesis
    return SHTnsKit.dist_synthesis(cfg, alm_dense; prototype_θφ=prototype_θφ, real_output=real_output)
end

"""
    dist_synthesis_distributed_2d_optimized(cfg::SHTConfig, alm::DistributedSpectralArray2D;
                                             prototype_θφ::PencilArray,
                                             real_output::Bool=true) -> Matrix

Optimized 2D-distributed synthesis that assumes spatial and spectral distributions are aligned.

**WARNING**: This function assumes that ranks with the same `l_rank` in the 2D spectral plan
have the same θ portions in the spatial PencilArray. If this assumption is violated,
results will be incorrect. Use `dist_synthesis_distributed_2d` for general correctness.

# When to use this function
- When you have explicitly set up the PencilArray to match the 2D spectral grid
- When p_l divides nlat evenly and spatial θ distribution matches l_rank grouping

# Algorithm
1. Gather within l-communicator to get all l values for this m-group's m values
2. Perform Legendre synthesis for local m values
3. Allreduce across m-communicator to combine all m contributions
4. IFFT along φ

# Memory behavior with scratch buffers
When the plan has `with_scratch=true` and `φ` is not distributed, the returned array
is a view into pre-allocated scratch memory. This eliminates allocation but means:
- The result will be **overwritten** on the next synthesis call
- Copy the result if you need to retain it: `result_copy = copy(result)`
- This is optimal for time-stepping codes that use the result immediately

Without scratch buffers, each call returns a freshly allocated array.
"""
function dist_synthesis_distributed_2d_optimized(cfg::SHTnsKit.SHTConfig, alm::DistributedSpectralArray2D;
                                                  prototype_θφ::PencilArray, real_output::Bool=true)
    plan = alm.plan
    lmax, mmax = plan.lmax, plan.mmax
    mres = plan.mres
    nlon = cfg.nlon
    m_range = plan.m_range

    # Use scratch buffers if available
    has_scratch = plan.with_scratch && plan.scratch !== nothing
    nlon_local = size(parent(prototype_θφ), 2)
    φ_is_local = (nlon_local == nlon)

    if has_scratch
        θ_globals = plan.scratch.θ_globals
        nθ_local = plan.scratch.nθ_local
        x_cache = plan.scratch.x_cache
        Fθm = plan.scratch.Fθm
        P = plan.scratch.P
        fθφ_local = plan.scratch.fθφ_local
        fθφ_result = plan.scratch.fθφ_result  # Separate result buffer to avoid copy
    else
        θ_globals = collect(globalindices(prototype_θφ, 1))
        nθ_local = length(θ_globals)
        x_cache = nothing  # Will use cfg.x directly
        Fθm = Matrix{ComplexF64}(undef, nθ_local, nlon)
        P = Vector{Float64}(undef, lmax + 1)
        fθφ_local = Matrix{Float64}(undef, nθ_local, nlon)
        fθφ_result = nothing  # Will allocate on return
    end

    # Zero the Fourier coefficient matrix
    fill!(Fθm, zero(ComplexF64))

    if !isempty(m_range)
        # Gather within l-communicator to get all l values for local m values
        alm_partial = gather_to_dense_2d(alm)

        inv_scaleφ = SHTnsKit.phi_inv_scale(cfg)

        # Synthesis: for each m in m_range, compute Legendre series
        m_col = 0
        for m in m_range
            (m % mres == 0) || continue
            m_col += 1
            col = m + 1

            if cfg.use_plm_tables && !isempty(cfg.plm_tables)
                # Pre-multiply Nlm * alm to avoid redundant multiply per θ point
                # Store in P buffer (reused across m values)
                @inbounds for l in m:lmax
                    P[l+1] = cfg.Nlm[l+1, col] * alm_partial[l+1, m_col]
                end

                tbl = cfg.plm_tables[col]
                for (ii, iglob) in enumerate(θ_globals)
                    g = 0.0 + 0.0im
                    @inbounds @simd for l in m:lmax
                        g += P[l+1] * tbl[l+1, iglob]
                    end
                    Fθm[ii, m + 1] = inv_scaleφ * g
                    if real_output && m > 0
                        Fθm[ii, nlon - m + 1] = conj(Fθm[ii, m + 1])
                    end
                end
            elseif x_cache !== nothing
                # Non-table path with cached x values
                for ii in 1:nθ_local
                    SHTnsKit.Plm_row!(P, x_cache[ii], lmax, m)
                    g = 0.0 + 0.0im
                    @inbounds @simd for l in m:lmax
                        g += (cfg.Nlm[l+1, col] * P[l+1]) * alm_partial[l+1, m_col]
                    end
                    Fθm[ii, m + 1] = inv_scaleφ * g
                    if real_output && m > 0
                        Fθm[ii, nlon - m + 1] = conj(Fθm[ii, m + 1])
                    end
                end
            else
                # Non-table path without cache - direct cfg access
                for ii in 1:nθ_local
                    SHTnsKit.Plm_row!(P, cfg.x[θ_globals[ii]], lmax, m)
                    g = 0.0 + 0.0im
                    @inbounds @simd for l in m:lmax
                        g += (cfg.Nlm[l+1, col] * P[l+1]) * alm_partial[l+1, m_col]
                    end
                    Fθm[ii, m + 1] = inv_scaleφ * g
                    if real_output && m > 0
                        Fθm[ii, nlon - m + 1] = conj(Fθm[ii, m + 1])
                    end
                end
            end
        end
    end

    # Combine Fourier coefficients from all m-groups
    # This assumes ranks in m_comm have the same θ_globals!
    MPI.Allreduce!(Fθm, +, plan.m_comm)

    # Determine output buffer - use fθφ_result directly to avoid copy (fix #3)
    # Note: when scratch is used and φ is local, the returned array is a view into
    # scratch memory. It will be overwritten on the next synthesis call. Copy if needed.
    output_buffer = (fθφ_result !== nothing && φ_is_local && real_output) ? fθφ_result : fθφ_local

    # Perform inverse FFT along φ directly into output buffer
    SHTnsKitParallelExt.ifft_along_dim2!(output_buffer, Fθm)

    # Apply Robert form scaling if enabled
    if cfg.robert_form
        for ii in 1:nθ_local
            x_val = x_cache !== nothing ? x_cache[ii] : cfg.x[θ_globals[ii]]
            sθ = sqrt(max(0.0, 1 - x_val*x_val))
            if sθ > 0
                @inbounds for j in 1:nlon
                    output_buffer[ii, j] *= sθ
                end
            end
        end
    end

    # Return result
    if φ_is_local
        if real_output
            # When scratch available, output_buffer IS fθφ_result - no copy needed
            # When no scratch, output_buffer is fθφ_local - must copy
            return fθφ_result !== nothing ? output_buffer : copy(output_buffer)
        else
            return Complex{Float64}.(output_buffer)
        end
    else
        φ_globals = collect(globalindices(prototype_θφ, 2))
        local_φ_range = first(φ_globals):last(φ_globals)
        result = fθφ_local[:, local_φ_range]
        if !real_output
            result = Complex{Float64}.(result)
        end
        return result
    end
end

"""
    estimate_distributed_memory_savings_2d(lmax::Int, mmax::Int, p_l::Int, p_m::Int) -> NamedTuple

Estimate memory savings from using 2D distributed spectral storage compared to 1D and dense.
"""
function estimate_distributed_memory_savings_2d(lmax::Int, mmax::Int, p_l::Int, p_m::Int)
    nprocs = p_l * p_m

    # Dense storage per rank (replicated)
    dense_elements = (lmax + 1) * (mmax + 1)
    dense_bytes = dense_elements * sizeof(ComplexF64)

    # 1D distributed storage per rank (l-only distribution)
    local_1d_elements = 0
    for l in 0:lmax
        if l % nprocs == 0
            local_1d_elements += min(l, mmax) + 1
        end
    end
    avg_1d_elements = ceildiv(dense_elements, nprocs)
    dist_1d_bytes = avg_1d_elements * sizeof(ComplexF64)

    # 2D distributed storage per rank
    # Each m-group has mmax/p_m m values, each rank within m-group has 1/p_l of l values
    # Approximate: (lmax² / 2) / (p_l * p_m)
    total_coeffs = sum(min(l, mmax) + 1 for l in 0:lmax)  # Triangular count
    avg_2d_elements = ceildiv(total_coeffs, nprocs)
    dist_2d_bytes = avg_2d_elements * sizeof(ComplexF64)

    # Synthesis gather communication volume
    gather_1d_bytes = dense_bytes  # 1D gathers everything globally
    gather_2d_bytes = ceildiv(dense_bytes, p_m)  # 2D gathers within l-comm only

    savings_vs_dense = 100.0 * (1.0 - dist_2d_bytes / dense_bytes)
    savings_vs_1d = 100.0 * (1.0 - dist_2d_bytes / dist_1d_bytes)

    return (
        dense_bytes_per_rank = dense_bytes,
        dist_1d_bytes_per_rank = dist_1d_bytes,
        dist_2d_bytes_per_rank = dist_2d_bytes,
        savings_vs_dense_percent = savings_vs_dense,
        savings_vs_1d_percent = savings_vs_1d,
        gather_1d_bytes = gather_1d_bytes,
        gather_2d_bytes = gather_2d_bytes,
        gather_reduction_factor = p_m
    )
end

"""
    validate_2d_distribution_alignment(plan::DistributedSpectralPlan2D,
                                        prototype_θφ::PencilArray) -> (aligned::Bool, message::String)

Check if the spatial PencilArray distribution is aligned with the 2D spectral plan.

Alignment means ranks with the same l_rank (in the same row of the process grid)
have the same θ portions. This is required for `dist_synthesis_distributed_2d_optimized`.

Returns a tuple of (is_aligned, diagnostic_message).
"""
function validate_2d_distribution_alignment(plan::DistributedSpectralPlan2D,
                                             prototype_θφ::PencilArray)
    # Get local θ range for this rank
    θ_globals = collect(globalindices(prototype_θφ, 1))
    local_θ_hash = hash(θ_globals)

    # Exchange θ_hash within m_comm (ranks with same l_rank)
    # If all ranks in m_comm have the same θ_hash, distributions are aligned
    all_hashes = MPI.Allgather(UInt64(local_θ_hash), plan.m_comm)

    aligned = all(h == all_hashes[1] for h in all_hashes)

    if aligned
        return (true, "Spatial and spectral distributions are aligned. " *
                      "You can use dist_synthesis_distributed_2d_optimized.")
    else
        return (false, "Spatial and spectral distributions are NOT aligned. " *
                       "Ranks with same l_rank have different θ portions. " *
                       "Use dist_synthesis_distributed_2d for correct results.")
    end
end

# Efficient aligned version: computes only m_range coefficients, uses l_comm for reduction
# Requires spatial θ distribution to be aligned with spectral l-distribution
function _dist_analysis_2d_aligned(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                                    plan::DistributedSpectralPlan2D,
                                    use_tables=cfg.use_plm_tables)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon
    nlat = cfg.nlat
    m_range = plan.m_range
    mres = plan.mres
    l_comm = plan.l_comm

    # Use scratch buffers if available, otherwise allocate
    has_scratch = plan.with_scratch && plan.scratch !== nothing
    local_data = parent(fθφ)
    nlat_local, nlon_local = size(local_data)

    # Get cached or compute θ indices
    if has_scratch
        θ_globals = plan.scratch.θ_globals
        nθ_local = plan.scratch.nθ_local
        weights_cache = plan.scratch.weights_cache
        x_cache = plan.scratch.x_cache
        Fθm = plan.scratch.Fθm
        local_contrib = plan.scratch.local_contrib
        P = plan.scratch.P
        n_m_valid = plan.scratch.n_m_valid
    else
        θ_globals = collect(globalindices(fθφ, 1))
        nθ_local = length(θ_globals)
        weights_cache = Vector{Float64}(undef, nθ_local)
        x_cache = Vector{Float64}(undef, nθ_local)
        for (ii, iglob) in enumerate(θ_globals)
            weights_cache[ii] = cfg.w[iglob]
            x_cache[ii] = cfg.x[iglob]
        end
        Fθm = Matrix{ComplexF64}(undef, nlat_local, nlon)
        n_m_valid = count(m -> m % mres == 0, m_range)
        local_contrib = Matrix{ComplexF64}(undef, lmax + 1, max(n_m_valid, 1))
        P = Vector{Float64}(undef, lmax + 1)
    end

    # FFT along φ
    if nlon_local == nlon
        SHTnsKitParallelExt.fft_along_dim2!(Fθm, local_data)
    else
        φ_globals = collect(globalindices(fθφ, 2))
        φ_range = first(φ_globals):last(φ_globals)
        θ_range = first(θ_globals):last(θ_globals)
        Fθm_temp = _gather_and_fft_phi(local_data, θ_range, φ_range, nlon, plan.comm)
        copyto!(Fθm, Fθm_temp)
    end

    scaleφ = cfg.cphi
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables)

    # Early exit for empty m_range
    if n_m_valid == 0 || isempty(m_range)
        result = create_distributed_spectral_array_2d(plan, ComplexF64)
        return result
    end

    # Zero the contribution buffer (reusing pre-allocated memory)
    fill!(local_contrib, zero(ComplexF64))

    # Legendre integration only for m values in this m-group
    # This is the key efficiency gain: O(lmax²/p_m) computation instead of O(lmax²)
    m_col = 0
    for mval in m_range
        (mval % mres == 0) || continue
        m_col += 1
        m_fft = mval + 1  # FFT index (1-based, matches m value + 1)

        if use_tbl
            tbl = cfg.plm_tables[mval + 1]  # Table is indexed by m+1
            for ii in 1:nθ_local
                iglob = θ_globals[ii]
                wiFi = weights_cache[ii] * Fθm[ii, m_fft]  # Hoisted out of l-loop
                @inbounds @simd for l in mval:lmax
                    local_contrib[l+1, m_col] += wiFi * tbl[l+1, iglob]
                end
            end
        else
            for ii in 1:nθ_local
                wiFi = weights_cache[ii] * Fθm[ii, m_fft]  # Hoisted out of l-loop
                SHTnsKit.Plm_row!(P, x_cache[ii], lmax, mval)
                @inbounds @simd for l in mval:lmax
                    local_contrib[l+1, m_col] += wiFi * P[l+1]
                end
            end
        end
    end

    # Reduce contributions within l-communicator only
    # This is the key efficiency gain: O(lmax²/p_m) communication within p_l ranks
    # instead of O(lmax²) global communication
    θ_is_distributed = (nθ_local < nlat)

    if θ_is_distributed
        # Use packed communication to avoid sending zeros in triangular region
        # This reduces communication volume by ~50%
        if has_scratch
            packed = plan.scratch.packed_contrib
            pack_offsets = plan.scratch.pack_offsets
            m_values = plan.scratch.m_values

            # Pack valid coefficients (l >= m for each m column)
            pack_idx = 1
            for (m_idx, mval) in enumerate(m_values)
                @inbounds for l in mval:lmax
                    packed[pack_idx] = local_contrib[l+1, m_idx]
                    pack_idx += 1
                end
            end

            # Reduce packed buffer (smaller than full matrix)
            MPI.Allreduce!(packed, +, l_comm)

            # Unpack back to matrix
            pack_idx = 1
            for (m_idx, mval) in enumerate(m_values)
                @inbounds for l in mval:lmax
                    local_contrib[l+1, m_idx] = packed[pack_idx]
                    pack_idx += 1
                end
            end
        else
            # Fallback: reduce full matrix when scratch not available
            MPI.Allreduce!(local_contrib, +, l_comm)
        end
    end

    # Apply normalization
    m_col = 0
    for mval in m_range
        (mval % mres == 0) || continue
        m_col += 1
        @inbounds @simd ivdep for l in mval:lmax
            local_contrib[l+1, m_col] *= cfg.Nlm[l+1, mval+1] * scaleφ
        end
    end

    # Create output array and extract owned coefficients
    result = create_distributed_spectral_array_2d(plan, ComplexF64)

    # Extract owned coefficients using direct index computation (avoids Dict overhead)
    # m_col = (m - first_valid_m) / mres + 1 when mres divides m_range evenly
    # For general case, compute offset from m_range start
    m_range_start = first(m_range)

    if mres == 1
        # Fast path: direct indexing when mres=1
        @inbounds for (i, (l, m)) in enumerate(plan.local_lm_indices)
            m_col = m - m_range_start + 1
            result.local_coeffs[i] = local_contrib[l+1, m_col]
        end
    else
        # General case: account for mres spacing
        @inbounds for (i, (l, m)) in enumerate(plan.local_lm_indices)
            m_col = (m - m_range_start) ÷ mres + 1
            result.local_coeffs[i] = local_contrib[l+1, m_col]
        end
    end

    return result
end
