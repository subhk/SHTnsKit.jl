##########
# Minimal plan structs to keep API stable
##########

struct DistAnalysisPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
    # Pre-computed index maps for performance
    θ_local_to_global::Vector{Int}
    m_local_to_global::Vector{Int}
    m_local_range::UnitRange{Int}
    θ_local_range::UnitRange{Int}
    # Memory layout optimization
    use_packed_storage::Bool
    # Scratch buffers to eliminate per-call allocations
    with_spatial_scratch::Bool
    spatial_scratch::Union{Nothing,NamedTuple}  # Contains all temporary arrays needed
end

function DistAnalysisPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; use_rfft::Bool=false, use_packed_storage::Bool=true, with_spatial_scratch::Bool=false)
    # Pre-compute index mappings to avoid expensive lookups in tight loops
    temp_pencil = allocate(prototype_θφ; dims=(:θ,:m), eltype=ComplexF64)
    
    θ_range = axes(temp_pencil, 1)
    m_range = axes(temp_pencil, 2)
    
    θ_local_to_global = collect(globalindices(temp_pencil, 1))
    m_local_to_global = collect(globalindices(temp_pencil, 2))
    
    # Create comprehensive scratch buffers if requested
    scratch = if with_spatial_scratch
        lmax, mmax = cfg.lmax, cfg.mmax
        nθ_local = length(θ_local_to_global)
        
        # Pre-allocate all temporary arrays used in analysis transforms
        scratch_buffers = (
            # Legendre polynomial buffer (fallback when tables not available)
            legendre_buffer = Vector{Float64}(undef, lmax + 1),
            
            # Pre-cached weights and derived values
            weights_cache = Vector{Float64}(undef, nθ_local),
            
            # Storage for spectral coefficients
            temp_dense = zeros(ComplexF64, lmax+1, mmax+1),
            
            # Table view cache for plm_tables optimization
            table_view_cache = Dict{Tuple{Int,Int}, SubArray}(),
            
            # Valid m-value information cache
            valid_m_cache = Tuple{Int, Int, Int}[],
        )
        
        # Pre-populate weights cache
        for (ii, iglob) in enumerate(θ_local_to_global)
            scratch_buffers.weights_cache[ii] = cfg.w[iglob]
        end
        
        # Pre-populate valid m-values cache  
        for (jj, m) in enumerate(m_range)
            mglob = m_local_to_global[jj]
            mval = mglob - 1
            if mval <= mmax
                col = mval + 1
                push!(scratch_buffers.valid_m_cache, (jj, mval, col))
            end
        end
        
        scratch_buffers
    else
        nothing
    end
    
    return DistAnalysisPlan(cfg, prototype_θφ, use_rfft, θ_local_to_global, m_local_to_global, 
                           m_range, θ_range, use_packed_storage, with_spatial_scratch, scratch)
end

struct DistPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
end

DistPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; use_rfft::Bool=false) = DistPlan(cfg, prototype_θφ, use_rfft)

struct DistSphtorPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
    with_spatial_scratch::Bool
    spatial_scratch::Union{Nothing,Tuple{PencilArray,PencilArray}}
end

function DistSphtorPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    scratch = if with_spatial_scratch
        # Pre-allocate complex spatial scratch buffers for IFFT operations
        scratch_θ = allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
        scratch_φ = allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
        (scratch_θ, scratch_φ)
    else
        nothing
    end
    return DistSphtorPlan(cfg, prototype_θφ, use_rfft, with_spatial_scratch, scratch)
end

struct DistQstPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
    with_spatial_scratch::Bool
    spatial_scratch::Union{Nothing,PencilArray}
end

function DistQstPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    scratch = if with_spatial_scratch
        # Pre-allocate complex spatial scratch buffer for scalar IFFT operations
        allocate(prototype_θφ; dims=(:θ,:k), eltype=ComplexF64)
    else
        nothing
    end
    return DistQstPlan(cfg, prototype_θφ, use_rfft, with_spatial_scratch, scratch)
end
