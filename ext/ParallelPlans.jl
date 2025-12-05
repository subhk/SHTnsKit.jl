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
    # Get local portion information from the prototype PencilArray
    θ_local_to_global = collect(globalindices(prototype_θφ, 1))

    # For m indices, we use the full range 0:mmax since we work with full FFT result
    m_local_to_global = collect(1:(cfg.mmax + 1))

    θ_range = 1:length(θ_local_to_global)
    m_range = 1:length(m_local_to_global)

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
        for (jj, mglob) in enumerate(m_local_to_global)
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
    spatial_scratch::Union{Nothing,Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}
end

function DistSphtorPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    scratch = if with_spatial_scratch
        # Pre-allocate complex spatial scratch buffers for IFFT operations
        θ_globals = collect(globalindices(prototype_θφ, 1))
        nθ_local = length(θ_globals)
        nlon = cfg.nlon
        scratch_θ = Matrix{ComplexF64}(undef, nθ_local, nlon)
        scratch_φ = Matrix{ComplexF64}(undef, nθ_local, nlon)
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
    spatial_scratch::Union{Nothing,Matrix{ComplexF64}}
end

function DistQstPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    scratch = if with_spatial_scratch
        # Pre-allocate complex spatial scratch buffer for scalar IFFT operations
        θ_globals = collect(globalindices(prototype_θφ, 1))
        nθ_local = length(θ_globals)
        nlon = cfg.nlon
        Matrix{ComplexF64}(undef, nθ_local, nlon)
    else
        nothing
    end
    return DistQstPlan(cfg, prototype_θφ, use_rfft, with_spatial_scratch, scratch)
end
