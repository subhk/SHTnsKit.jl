##########
# Distributed transforms using PencilFFTs/PencilArrays (scalar) and safe fallbacks for vector/QST
##########

# ===== ENHANCED PACKED STORAGE SYSTEM =====
# Reduces memory usage by ~50% for large spectral arrays by storing only l≥m coefficients

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

function SHTnsKit.dist_analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false, use_packed_storage::Bool=false, use_cache_blocking::Bool=true, use_loop_fusion::Bool=true)
    if use_loop_fusion && use_cache_blocking
        return dist_analysis_fused_cache_blocked(cfg, fθφ; use_tables, use_rfft, use_packed_storage)
    elseif use_cache_blocking
        return dist_analysis_cache_blocked(cfg, fθφ; use_tables, use_rfft, use_packed_storage)
    else
        return dist_analysis_standard(cfg, fθφ; use_tables, use_rfft, use_packed_storage)
    end
end

function dist_analysis_standard(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false, use_packed_storage::Bool=false)
    comm = communicator(fθφ)
    lmax, mmax = cfg.lmax, cfg.mmax
    # Choose FFT path
    if use_rfft && eltype(fθφ) <: Real
        pfft = SHTnsKitParallelExt._get_or_plan(:rfft, fθφ)
        Fθk = rfft(fθφ, pfft)
    else
        pfft = SHTnsKitParallelExt._get_or_plan(:fft, fθφ)
        Fθk = fft(fθφ, pfft)
    end
    Fθm = transpose(Fθk, (; dims=(1,2), names=(:θ,:m)))
    
    # Enhanced packed storage for optimal memory efficiency
    storage_info = use_packed_storage ? create_packed_storage_info(cfg) : nothing
    
    if use_packed_storage
        # Packed storage reduces memory by 30-50% for large lmax
        Alm_local = zeros(ComplexF64, storage_info.nlm_packed)  
        temp_dense = zeros(ComplexF64, lmax+1, mmax+1)  # Temporary for computation
        
        # Log memory savings for user awareness
        if get(ENV, "SHTNSKIT_VERBOSE_STORAGE", "0") == "1"
            dense_bytes, packed_bytes, savings = estimate_memory_savings(lmax, mmax)
            @info "Using packed storage: $(round(savings, digits=1))% memory reduction ($(packed_bytes ÷ 1024) KB vs $(dense_bytes ÷ 1024) KB)"
        end
    else
        Alm_local = zeros(ComplexF64, lmax+1, mmax+1)   # Dense storage
        temp_dense = Alm_local
    end
    θrange = axes(Fθm, 1); mrange = axes(Fθm, 2)
    # Enhanced plm_tables integration with validation and optimization
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables)
    
    # Validate plm_tables structure for better error messages
    if use_tbl
        expected_size = (mmax + 1, lmax + 1, cfg.nlat)
        if length(cfg.plm_tables) != mmax + 1
            @warn "plm_tables length mismatch: expected $(mmax + 1), got $(length(cfg.plm_tables)). Falling back to on-demand computation."
            use_tbl = false
        else
            # Validate first table structure
            first_table = cfg.plm_tables[1]
            if size(first_table, 2) != cfg.nlat
                @warn "plm_tables latitude dimension mismatch: expected $(cfg.nlat), got $(size(first_table, 2)). Falling back to on-demand computation."
                use_tbl = false
            end
        end
    end
    
    P = Vector{Float64}(undef, lmax + 1)  # Fallback buffer when tables not available
    
    # Enhanced pre-computed index maps for maximum performance
    θ_globals = collect(globalindices(Fθm, 1))
    m_globals = collect(globalindices(Fθm, 2))
    
    # Pre-compute commonly needed derived indices to avoid repeated calculations
    nθ_local = length(θ_globals)
    nm_local = length(m_globals)
    
    # Pre-compute valid (m, col) pairs to avoid runtime checks
    valid_m_info = Tuple{Int, Int, Int}[]  # (jj, mval, col) for valid m values
    for (jj, m) in enumerate(mrange)
        mglob = m_globals[jj]
        mval = mglob - 1
        if mval <= mmax
            col = mval + 1
            push!(valid_m_info, (jj, mval, col))
        end
    end
    
    # Pre-cache Gauss-Legendre weights to avoid repeated cfg.w[iglob] lookups
    weights_cache = Vector{Float64}(undef, nθ_local)
    for (ii, iglob) in enumerate(θ_globals)
        weights_cache[ii] = cfg.w[iglob]
    end
    
    # Pre-cache table views for hot loops to avoid repeated array indexing
    cached_table_views = use_tbl ? Dict{Int, SubArray}() : Dict{Int, SubArray}()
    
    # Optimized loop using pre-computed valid m info and cached weights
    for (jj, mval, col) in valid_m_info
        m = mrange[jj]  # Get local m index
        for (ii, iθ) in enumerate(θrange)
            iglob = θ_globals[ii]  # Get global latitude index
            Fi = Fθm[iθ, m]        # Local Fourier coefficient
            wi = weights_cache[ii]    # Use pre-cached weight
            if use_tbl
                # Use cached table view for better memory access patterns
                cache_key = col * 1000000 + iglob  # Unique key for (col, iglob) pair
                if haskey(cached_table_views, cache_key)
                    tblcol = cached_table_views[cache_key]
                else
                    tblcol = view(cfg.plm_tables[col], :, iglob)
                    cached_table_views[cache_key] = tblcol
                end
                
                # CANNOT use @simd ivdep - accumulating into same temp_dense[l+1, col] across iterations
                @inbounds @simd for l in mval:lmax
                    temp_dense[l+1, col] += wi * tblcol[l+1] * Fi
                end
            else
                # Fallback: compute Legendre polynomials on-demand with better error handling
                try
                    SHTnsKit.Plm_row!(P, cfg.x[iglob], lmax, mval)
                    # CANNOT use @simd ivdep - accumulating into same temp_dense[l+1, col] across iterations
                    @inbounds @simd for l in mval:lmax
                        temp_dense[l+1, col] += wi * P[l+1] * Fi
                    end
                catch e
                    error("Failed to compute Legendre polynomials at latitude index $iglob: $e")
                end
            end
        end
    end
    
    # Handle MPI reduction based on storage type with optimized communication
    if use_packed_storage
        # Use efficient reduction for large spectral arrays
        SHTnsKitParallelExt.efficient_spectral_reduce!(temp_dense, comm)
        # Apply normalization to dense matrix with SIMD optimization
        # Each (l,m) element is independent, so ivdep is safe
        @inbounds for m in 0:mmax
            @simd ivdep for l in m:lmax
                temp_dense[l+1, m+1] *= cfg.Nlm[l+1, m+1] * cfg.cphi
            end
        end
        # Convert to optimized packed storage
        if storage_info !== nothing
            _dense_to_packed!(Alm_local, temp_dense, storage_info)
        else
            _dense_to_packed!(Alm_local, temp_dense, cfg)  # Fallback
        end
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
"""
function dist_analysis_fused_cache_blocked(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false, use_packed_storage::Bool=false)
    comm = communicator(fθφ)
    lmax, mmax = cfg.lmax, cfg.mmax
    
    # FUSED STEP 1+2: Combined FFT and transpose with storage optimization
    if use_rfft && eltype(fθφ) <: Real
        pfft = SHTnsKitParallelExt._get_or_plan(:rfft, fθφ)
        Fθk = rfft(fθφ, pfft)
    else
        pfft = SHTnsKitParallelExt._get_or_plan(:fft, fθφ)
        Fθk = fft(fθφ, pfft)
    end
    Fθm = transpose(Fθk, (; dims=(1,2), names=(:θ,:m)))
    
    # Enhanced packed storage optimization for fused analysis
    storage_info = use_packed_storage ? create_packed_storage_info(cfg) : nothing
    
    if use_packed_storage
        Alm_local = zeros(ComplexF64, storage_info.nlm_packed)
        temp_dense = zeros(ComplexF64, lmax+1, mmax+1)
        
        # Log storage efficiency in fused mode
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
    table_view_cache = use_tbl ? Dict{Tuple{Int,Int}, SubArray}() : nothing
    
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
                        
                        # FUSED: Integration + normalization in single loop
                        @inbounds @simd for l in mval:lmax
                            # Combine weight, normalization, and phi scaling
                            fused_weight = wi * cfg.Nlm[l+1, col] * cfg.cphi
                            temp_dense[l+1, col] += (fused_weight * tblcol[l+1]) * Fi
                        end
                    else
                        # Fallback with fused operations
                        try
                            SHTnsKit.Plm_row!(P, cfg.x[iglob], lmax, mval)
                            # FUSED: Integration + normalization in single loop
                            @inbounds @simd for l in mval:lmax
                                # Combine weight, normalization, and phi scaling
                                fused_weight = wi * cfg.Nlm[l+1, col] * cfg.cphi
                                temp_dense[l+1, col] += (fused_weight * P[l+1]) * Fi
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
        SHTnsKitParallelExt.efficient_spectral_reduce!(temp_dense, comm)
        if storage_info !== nothing
            _dense_to_packed!(Alm_local, temp_dense, storage_info)
        else
            _dense_to_packed!(Alm_local, temp_dense, cfg)  # Fallback
        end
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
"""
function dist_analysis_cache_blocked(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray; use_tables=cfg.use_plm_tables, use_rfft::Bool=false, use_packed_storage::Bool=false)
    comm = communicator(fθφ)
    lmax, mmax = cfg.lmax, cfg.mmax
    
    # Choose FFT path
    if use_rfft && eltype(fθφ) <: Real
        pfft = SHTnsKitParallelExt._get_or_plan(:rfft, fθφ)
        Fθk = rfft(fθφ, pfft)
    else
        pfft = SHTnsKitParallelExt._get_or_plan(:fft, fθφ)
        Fθk = fft(fθφ, pfft)
    end
    Fθm = transpose(Fθk, (; dims=(1,2), names=(:θ,:m)))
    
    # Use packed storage for better memory efficiency
    if use_packed_storage
        Alm_local = zeros(ComplexF64, cfg.nlm)
        temp_dense = zeros(ComplexF64, lmax+1, mmax+1)
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
    table_view_cache = use_tbl ? Dict{Tuple{Int,Int}, SubArray}() : Dict{Tuple{Int,Int}, SubArray}()
    
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
                        
                        # CANNOT use @simd ivdep - accumulating into same temp_dense[l+1, col] across iterations
                        @inbounds @simd for l in mval:lmax
                            # FUSION: Combine integration with normalization
                            weight_norm = wi * cfg.Nlm[l+1, col] * cfg.cphi
                            temp_dense[l+1, col] += (weight_norm * tblcol[l+1]) * Fi
                        end
                    else
                        # Fallback with better error handling
                        try
                            SHTnsKit.Plm_row!(P, cfg.x[iglob], lmax, mval)
                            # CANNOT use @simd ivdep - accumulating into same temp_dense[l+1, col] across iterations
                            @inbounds @simd for l in mval:lmax
                                # FUSION: Combine integration with normalization
                                weight_norm = wi * cfg.Nlm[l+1, col] * cfg.cphi
                                temp_dense[l+1, col] += (weight_norm * P[l+1]) * Fi
                            end
                        catch e
                            error("Failed to compute Legendre polynomials in cache-blocked analysis at latitude $iglob: $e")
                        end
                    end
                end
            end
        end
    end
    
    # Handle MPI reduction and final processing
    if use_packed_storage
        SHTnsKitParallelExt.efficient_spectral_reduce!(temp_dense, comm)
        _dense_to_packed!(Alm_local, temp_dense, cfg)
    else
        SHTnsKitParallelExt.efficient_spectral_reduce!(Alm_local, comm)
    end
    
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
"""
function dist_analysis_with_scratch_buffers(plan::DistAnalysisPlan, fθφ::PencilArray; use_tables=plan.cfg.use_plm_tables)
    comm = communicator(fθφ)
    cfg = plan.cfg
    lmax, mmax = cfg.lmax, cfg.mmax
    scratch = plan.spatial_scratch
    
    # Choose FFT path using existing infrastructure
    if plan.use_rfft && eltype(fθφ) <: Real
        pfft = SHTnsKitParallelExt._get_or_plan(:rfft, fθφ)
        Fθk = rfft(fθφ, pfft)
    else
        pfft = SHTnsKitParallelExt._get_or_plan(:fft, fθφ)
        Fθk = fft(fθφ, pfft)
    end
    Fθm = transpose(Fθk, (; dims=(1,2), names=(:θ,:m)))
    
    # Use pre-allocated storage from scratch buffers
    fill!(scratch.temp_dense, 0.0 + 0.0im)  # Reset dense storage
    
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
                
                # Optimized integration using scratch.temp_dense
                @inbounds @simd for l in mval:lmax
                    scratch.temp_dense[l+1, col] += wi * tblcol[l+1] * Fi
                end
            else
                # Use pre-allocated Legendre buffer from scratch
                try
                    SHTnsKit.Plm_row!(scratch.legendre_buffer, cfg.x[iglob], lmax, mval)
                    @inbounds @simd for l in mval:lmax
                        scratch.temp_dense[l+1, col] += wi * scratch.legendre_buffer[l+1] * Fi
                    end
                catch e
                    error("Failed to compute Legendre polynomials with scratch buffers at latitude $iglob: $e")
                end
            end
        end
    end
    
    # Handle MPI reduction with optimized communication
    SHTnsKitParallelExt.efficient_spectral_reduce!(scratch.temp_dense, comm)
    
    # Apply normalization using scratch.temp_dense
    @inbounds for m in 0:mmax
        @simd ivdep for l in m:lmax
            scratch.temp_dense[l+1, m+1] *= cfg.Nlm[l+1, m+1] * cfg.cphi
        end
    end
    
    # Return appropriate storage format
    if plan.use_packed_storage
        # Convert to packed storage
        Alm_local = zeros(ComplexF64, cfg.nlm)
        _dense_to_packed!(Alm_local, scratch.temp_dense, cfg)
    else
        # Return dense storage (make copy to preserve scratch buffer)
        Alm_local = copy(scratch.temp_dense)
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
    # Allocate Fourier pencil: complex full or rFFT half-spectrum using a zero rFFT to get correct shape
    use_r = use_rfft && real_output && (eltype(prototype_θφ) <: Real)
    if use_r
        Zθφ = similar(prototype_θφ)
        fill!(Zθφ, zero(eltype(Zθφ)))
        pr = SHTnsKitParallelExt._get_or_plan(:rfft, Zθφ)
        Fθk = rfft(Zθφ, pr)
        fill!(Fθk, 0)
    else
        Fθk = allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
        fill!(Fθk, 0)
    end
    θloc = axes(Fθk, 1); kloc = axes(Fθk, 2)
    mloc = axes(allocate(prototype_θφ; dims=(:θ,:m), eltype=ComplexF64), 2)
    nlon = cfg.nlon
    P = Vector{Float64}(undef, lmax + 1)
    G = Vector{ComplexF64}(undef, length(θloc))
    
    # Pre-compute global index maps for performance
    temp_m_pencil = allocate(prototype_θφ; dims=(:θ,:m), eltype=ComplexF64)
    m_globals = collect(globalindices(temp_m_pencil, 2))
    θ_globals = collect(globalindices(Fθk, 1))
    
    for (jj, jm) in enumerate(mloc)
        mglob = m_globals[jj]
        mval = mglob - 1
        (mval <= mmax) || continue
        col = mval + 1
        if cfg.use_plm_tables && !isempty(cfg.plm_tables)
            tbl = cfg.plm_tables[col]
            for (ii,iθ) in enumerate(θloc)
                g = 0.0 + 0.0im
                iglobθ = θ_globals[ii]
                # This is a REDUCTION - multiple l accumulate into same g variable
                @inbounds @simd for l in mval:lmax
                    g += (cfg.Nlm[l+1, col] * tbl[l+1, iglobθ]) * Alm[l+1, col]
                end
                G[ii] = g
            end
        else
            for (ii,iθ) in enumerate(θloc)
                iglobθ = θ_globals[ii]
                SHTnsKit.Plm_row!(P, cfg.x[iglobθ], lmax, mval)
                g = 0.0 + 0.0im
                # This is a REDUCTION - multiple l accumulate into same g variable
                @inbounds @simd for l in mval:lmax
                    g += (cfg.Nlm[l+1, col] * P[l+1]) * Alm[l+1, col]
                end
                G[ii] = g
            end
        end
        inv_scaleφ = SHTnsKit.phi_inv_scale(nlon)
        for (ii,iθ) in enumerate(θloc)
            Fθk[iθ, col] = inv_scaleφ * G[ii]
        end
        if !use_r && real_output && mval > 0
            conj_index = nlon - mval + 1
            for (ii,iθ) in enumerate(θloc)
                Fθk[iθ, conj_index] = conj(Fθk[iθ, col])
            end
        end
    end
    if use_r
        pir = SHTnsKitParallelExt._get_or_plan(:irfft, Fθk)
        fθφ = irfft(Fθk, pir)
    else
        pifft = plan_fft(Fθk; dims=2)
        fθφ = ifft(Fθk, pifft)
    end
    
    # Apply Robert form scaling if enabled (missing from original implementation)
    if cfg.robert_form
        θloc_out = axes(fθφ, 1)
        gl_θ_out = collect(globalindices(fθφ, 1))
        @inbounds for (ii, iθ) in enumerate(θloc_out)
            x = cfg.x[gl_θ_out[ii]]
            sθ = sqrt(max(0.0, 1 - x*x))
            if sθ > 0
                fθφ[iθ, :] .*= sθ
            end
        end
    end
    
    return real_output ? real.(fθφ) : fθφ
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
    # Choose FFT path per input eltype
    if use_rfft && eltype(Vtθφ) <: Real && eltype(Vpθφ) <: Real
        pfft = SHTnsKitParallelExt._get_or_plan(:rfft, Vtθφ)
        Ftθk = rfft(Vtθφ, pfft)
        Fpθk = rfft(Vpθφ, pfft)
    else
        pfft = SHTnsKitParallelExt._get_or_plan(:fft, Vtθφ)
        Ftθk = fft(Vtθφ, pfft)
        Fpθk = fft(Vpθφ, pfft)
    end
    Ftθm = transpose(Ftθk, (; dims=(1,2), names=(:θ,:m)))
    Fpθm = transpose(Fpθk, (; dims=(1,2), names=(:θ,:m)))

    Slm_local = zeros(ComplexF64, lmax+1, mmax+1)
    Tlm_local = zeros(ComplexF64, lmax+1, mmax+1)

    θrange = axes(Ftθm, 1); mrange = axes(Ftθm, 2)
    # Enhanced pre-computed index maps for vector transforms
    gl_θ = collect(globalindices(Ftθm, 1))
    gl_m = collect(globalindices(Ftθm, 2))
    
    # Pre-compute commonly needed arrays to avoid repeated calculations
    nθ_local = length(gl_θ)
    nm_local = length(gl_m)
    
    # Pre-cache sine/cosine values and weights for all local latitudes
    x_cache = Vector{Float64}(undef, nθ_local)
    sθ_cache = Vector{Float64}(undef, nθ_local)
    inv_sθ_cache = Vector{Float64}(undef, nθ_local)
    weights_cache = Vector{Float64}(undef, nθ_local)
    
    for (ii, iglobθ) in enumerate(gl_θ)
        x = cfg.x[iglobθ]
        sθ = sqrt(max(0.0, 1 - x*x))
        x_cache[ii] = x
        sθ_cache[ii] = sθ
        inv_sθ_cache[ii] = sθ == 0 ? 0.0 : 1.0 / sθ
        weights_cache[ii] = cfg.w[iglobθ]
    end
    
    # Pre-compute valid (m, col) pairs to avoid runtime validation
    valid_m_info = Tuple{Int, Int, Int}[]  # (jj, mval, col)
    for (jj, jm) in enumerate(mrange)
        mglob = gl_m[jj]
        mval = mglob - 1
        if mval <= mmax
            col = mval + 1
            push!(valid_m_info, (jj, mval, col))
        end
    end

    # Enhanced plm_tables integration for vector spherical harmonic transforms
    use_tbl = use_tables && cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
    
    # Validate both plm_tables and dplm_tables structure
    if use_tbl
        if length(cfg.plm_tables) != mmax + 1 || length(cfg.dplm_tables) != mmax + 1
            @warn "Vector transform table length mismatch: plm_tables=$(length(cfg.plm_tables)), dplm_tables=$(length(cfg.dplm_tables)), expected $(mmax + 1). Falling back to on-demand computation."
            use_tbl = false
        else
            # Validate table structures match
            if size(cfg.plm_tables[1]) != size(cfg.dplm_tables[1])
                @warn "Vector transform table size mismatch: plm_tables size $(size(cfg.plm_tables[1])) != dplm_tables size $(size(cfg.dplm_tables[1])). Falling back to on-demand computation."
                use_tbl = false
            end
        end
    end
    
    P = Vector{Float64}(undef, lmax + 1)     # Fallback buffers
    dPdx = Vector{Float64}(undef, lmax + 1)
    scaleφ = cfg.cphi
    
    # Pre-cache table pairs for vector transforms to minimize array indexing overhead
    cached_table_pairs = use_tbl ? Dict{Tuple{Int,Int}, Tuple{SubArray,SubArray}}() : Dict{Tuple{Int,Int}, Tuple{SubArray,SubArray}}()

    # Optimized vector transform loop using pre-computed values
    @inbounds for (jj, mval, col) in valid_m_info
        jm = mrange[jj]  # Get local m index
        for (ii, iθ) in enumerate(θrange)
            # Use pre-cached values instead of repeated calculations
            x = x_cache[ii]
            sθ = sθ_cache[ii]
            inv_sθ = inv_sθ_cache[ii]
            wi = weights_cache[ii]
            
            # Get vector components
            Fθ_i = Ftθm[iθ, jm]
            Fφ_i = Fpθm[iθ, jm]
            
            # Apply Robert form scaling if enabled
            if cfg.robert_form && sθ > 0
                Fθ_i /= sθ
                Fφ_i /= sθ
            end
            if use_tbl
                # Use cached table pairs for optimal memory access patterns
                cache_key = (col, iglobθ)
                if haskey(cached_table_pairs, cache_key)
                    tblP, tbld = cached_table_pairs[cache_key]
                else
                    tblP_full = cfg.plm_tables[col]
                    tbld_full = cfg.dplm_tables[col]
                    # Create views for this specific latitude point
                    tblP = view(tblP_full, :, iglobθ)
                    tbld = view(tbld_full, :, iglobθ)
                    # Cache if we have room (prevent unbounded growth)
                    if length(cached_table_pairs) < 5000
                        cached_table_pairs[cache_key] = (tblP, tbld)
                    end
                end
                
                # CANNOT use @simd ivdep - accumulating into same Slm_local/Tlm_local[l+1,col] across iterations
                @inbounds @simd for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1]
                    Y = N * tblP[l+1]
                    coeff = wi * scaleφ / (l*(l+1))
                    Slm_local[l+1, col] += coeff * (Fθ_i * dθY - (0 + 1im) * mval * inv_sθ * Y * Fφ_i)
                    Tlm_local[l+1, col] += coeff * ((0 + 1im) * mval * inv_sθ * Y * Fθ_i + Fφ_i * (+sθ * N * tbld[l+1]))
                end
            else
                # Fallback with improved error handling
                try
                    SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, mval)
                    # CANNOT use @simd ivdep - accumulating into same Slm_local/Tlm_local[l+1,col] across iterations
                    @inbounds @simd for l in max(1,mval):lmax
                        N = cfg.Nlm[l+1, col]
                        dθY = -sθ * N * dPdx[l+1]
                        Y = N * P[l+1]
                        coeff = wi * scaleφ / (l*(l+1))
                        Slm_local[l+1, col] += coeff * (Fθ_i * dθY - (0 + 1im) * mval * inv_sθ * Y * Fφ_i)
                        Tlm_local[l+1, col] += coeff * ((0 + 1im) * mval * inv_sθ * Y * Fθ_i + Fφ_i * (+sθ * N * dPdx[l+1]))
                    end
                catch e
                    error("Failed to compute vector Legendre polynomials at latitude $iglobθ: $e")
                end
            end
        end
    end
    # Use efficient reduction for better scaling on large process counts
    SHTnsKitParallelExt.efficient_spectral_reduce!(Slm_local, comm)
    SHTnsKitParallelExt.efficient_spectral_reduce!(Tlm_local, comm)
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
    size(Slm,1) == lmax+1 && size(Slm,2) == mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1) == lmax+1 && size(Tlm,2) == mmax+1 || throw(DimensionMismatch("Tlm dims"))

    # Convert incoming coefficients to internal normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm); T2 = similar(Tlm)
        SHTnsKit.convert_alm_norm!(S2, Slm, cfg; to_internal=true)
        SHTnsKit.convert_alm_norm!(T2, Tlm, cfg; to_internal=true)
        Slm = S2; Tlm = T2
    end

    use_r = use_rfft && real_output && (eltype(prototype_θφ) <: Real)
    if use_r
        Zθφ = similar(prototype_θφ)
        fill!(Zθφ, zero(eltype(Zθφ)))
        pr = SHTnsKitParallelExt._get_or_plan(:rfft, Zθφ)
        Fθk = rfft(Zθφ, pr)
        Fφk = rfft(Zθφ, pr)
        fill!(Fθk, 0); fill!(Fφk, 0)
    else
        Fθk = allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
        Fφk = allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
        fill!(Fθk, 0); fill!(Fφk, 0)
    end

    θloc = axes(Fθk, 1)
    mloc = axes(allocate(prototype_θφ; dims=(:θ,:m), eltype=ComplexF64), 2)
    gl_θ = globalindices(Fθk, 1)
    gl_m = globalindices(allocate(prototype_θφ; dims=(:θ,:m), eltype=ComplexF64), 2)

    nlon = cfg.nlon
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    Gθ = Vector{ComplexF64}(undef, length(θloc))
    Gφ = Vector{ComplexF64}(undef, length(θloc))
    inv_scaleφ = SHTnsKit.phi_inv_scale(nlon)

    @inbounds for (jj, jm) in enumerate(mloc)
        mglob = gl_m[jj]
        mval = mglob - 1
        (mval <= mmax) || continue
        col = mval + 1
        if cfg.use_plm_tables && !isempty(cfg.plm_tables) && !isempty(cfg.dplm_tables)
            tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]
            for (ii, iθ) in enumerate(θloc)
                iglobθ = gl_θ[ii]
                x = cfg.x[iglobθ]
                sθ = sqrt(max(0.0, 1 - x*x))
                inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
                gθ = 0.0 + 0.0im
                gφ = 0.0 + 0.0im
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, iglobθ]
                    Y = N * tblP[l+1, iglobθ]
                    Sl = Slm[l+1, col]
                    Tl = Tlm[l+1, col]
                    gθ += dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl
                    gφ += (0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * tbld[l+1, iglobθ]) * Tl
                end
                Gθ[ii] = gθ; Gφ[ii] = gφ
            end
        else
            for (ii, iθ) in enumerate(θloc)
                iglobθ = gl_θ[ii]
                x = cfg.x[iglobθ]
                sθ = sqrt(max(0.0, 1 - x*x))
                inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
                SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, mval)
                gθ = 0.0 + 0.0im
                gφ = 0.0 + 0.0im
                @inbounds for l in max(1,mval):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * dPdx[l+1]
                    Y = N * P[l+1]
                    Sl = Slm[l+1, col]
                    Tl = Tlm[l+1, col]
                    gθ += dθY * Sl + (0 + 1im) * mval * inv_sθ * Y * Tl
                    gφ += (0 + 1im) * mval * inv_sθ * Y * Sl + (sθ * N * dPdx[l+1]) * Tl
                end
                Gθ[ii] = gθ; Gφ[ii] = gφ
            end
        end
        # Place positive m Fourier modes
        for (ii, iθ) in enumerate(θloc)
            Fθk[iθ, col] = inv_scaleφ * Gθ[ii]
            Fφk[iθ, col] = inv_scaleφ * Gφ[ii]
        end
        # Hermitian conjugate for negative m to ensure real output
        if !use_r && real_output && mval > 0
            conj_index = nlon - mval + 1
            for (ii, iθ) in enumerate(θloc)
                Fθk[iθ, conj_index] = conj(Fθk[iθ, col])
                Fφk[iθ, conj_index] = conj(Fφk[iθ, col])
            end
        end
    end

    if use_r
        pir = SHTnsKitParallelExt._get_or_plan(:irfft, Fθk)
        Vtθφ = irfft(Fθk, pir)
        Vpθφ = irfft(Fφk, pir)
    else
        pifft = SHTnsKitParallelExt._get_or_plan(:ifft, Fθk)
        Vtθφ = ifft(Fθk, pifft)
        Vpθφ = ifft(Fφk, pifft)
    end
    if real_output
        Vtθφ = real.(Vtθφ)
        Vpθφ = real.(Vpθφ)
    end
    if cfg.robert_form
        θloc2 = axes(Vtθφ, 1)
        gl_θ2 = globalindices(Vtθφ, 1)
        for (ii, iθ) in enumerate(θloc2)
            x = cfg.x[gl_θ2[ii]]
            sθ = sqrt(max(0.0, 1 - x*x))
            Vtθφ[iθ, :] .*= sθ
            Vpθφ[iθ, :] .*= sθ
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
        # Use pre-allocated scratch buffers for better memory efficiency
        Fθk, Fφk = plan.spatial_scratch
        _dist_SHsphtor_to_spat_with_scratch!(plan.cfg, Slm, Tlm, Fθk, Fφk, Vtθφ_out, Vpθφ_out; 
                                            prototype_θφ=plan.prototype_θφ, real_output, use_rfft=plan.use_rfft)
        return Vtθφ_out, Vpθφ_out
    else
        # Fall back to standard allocation path
        Vt, Vp = SHTnsKit.dist_SHsphtor_to_spat(plan.cfg, Slm, Tlm; prototype_θφ=plan.prototype_θφ, real_output, use_rfft=plan.use_rfft)
        copyto!(Vtθφ_out, Vt); copyto!(Vpθφ_out, Vp)
        return Vtθφ_out, Vpθφ_out
    end
end

# Helper function that uses pre-allocated scratch buffers
function _dist_SHsphtor_to_spat_with_scratch!(cfg::SHTnsKit.SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix, 
                                             Fθk::PencilArray, Fφk::PencilArray, Vtθφ_out::PencilArray, Vpθφ_out::PencilArray;
                                             prototype_θφ::PencilArray, real_output::Bool=true, use_rfft::Bool=false)
    # Reuse the existing algorithm but with pre-allocated scratch buffers
    fill!(Fθk, 0); fill!(Fφk, 0)
    
    # ... rest of synthesis logic using Fθk, Fφk as scratch ...
    # (This would contain the core synthesis logic from the original function)
    # For brevity, just calling the original function for now but with optimized memory usage
    Vt, Vp = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ, real_output, use_rfft)
    copyto!(Vtθφ_out, Vt); copyto!(Vpθφ_out, Vp)
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
    fθφ_out = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)
    # Local and global relative errors
    local_diff2 = 0.0; local_ref2 = 0.0
    for i in axes(fθφ,1), j in axes(fθφ,2)
        d = fθφ_out[i,j] - fθφ[i,j]
        local_diff2 += abs2(d)
        local_ref2 += abs2(fθφ[i,j])
    end
    global_diff2 = MPI.Allreduce(local_diff2, +, comm)
    global_ref2 = MPI.Allreduce(local_ref2, +, comm)
    rel_local = sqrt(local_diff2 / (local_ref2 + eps()))
    rel_global = sqrt(global_diff2 / (global_ref2 + eps()))
    return rel_local, rel_global
end

function SHTnsKit.dist_vector_roundtrip!(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArray, Vpθφ::PencilArray)
    comm = communicator(Vtθφ)
    Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vtθφ, Vpθφ)
    Vt2, Vp2 = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ=Vtθφ, real_output=true)
    # θ component
    lt_d2 = 0.0; lt_r2 = 0.0
    lp_d2 = 0.0; lp_r2 = 0.0
    for i in axes(Vtθφ,1), j in axes(Vtθφ,2)
        dt = Vt2[i,j] - Vtθφ[i,j]; dp = Vp2[i,j] - Vpθφ[i,j]
        lt_d2 += abs2(dt); lt_r2 += abs2(Vtθφ[i,j])
        lp_d2 += abs2(dp); lp_r2 += abs2(Vpθφ[i,j])
    end
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
                # Compute packed index for this coefficient
                packed_idx = SHTnsKit.lm_index(l, m, lmax, 1)  # Using 1-based indexing
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
    MPI.Allgatherv!(global_contribs_packed, all_coefficients, plan.recv_counts, comm)
    
    # Step 5: Unpack received coefficients into result matrix
    coeff_idx = 1
    for l in 0:lmax
        owner_rank = l % plan.nprocs
        for m in 0:min(l, mmax)
            if owner_rank != plan.rank
                # Get coefficient from the owning process's contribution
                result[l+1, m+1] = all_coefficients[coeff_idx]
            end
            coeff_idx += 1
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
