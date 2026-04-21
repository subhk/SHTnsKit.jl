##########
# Minimal plan structs to keep API stable
##########

"""
    _validate_cfg_replicated(cfg, comm)

Rank-0 broadcasts a hash of the key cfg fields (lmax, mmax, mres, nlat, nlon,
norm, cs_phase, robert_form); each rank compares and errors on mismatch. Cheap
guard against users constructing divergent configs per rank (silent wrong
results otherwise).
"""
function _validate_cfg_replicated(cfg::SHTnsKit.SHTConfig, comm)
    MPI.Comm_size(comm) > 1 || return
    sig = hash((cfg.lmax, cfg.mmax, cfg.mres, cfg.nlat, cfg.nlon,
                cfg.norm, cfg.cs_phase, cfg.robert_form))
    root_sig = MPI.bcast(sig, 0, comm)
    if sig != root_sig
        throw(ArgumentError("SHTConfig diverges across ranks (rank $(MPI.Comm_rank(comm))). All ranks must construct cfg with identical parameters."))
    end
    return
end

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
    # use_rfft=true is wired through dist_analysis_standard and dist_synthesis
    # for real inputs/outputs. Case A (φ replicated) uses FFTW.rfft directly;
    # Case B (φ split) uses a row-subcomm gather + FFTW.rfft via
    # distributed_rfft_phi!. Complex-valued callers still use the complex FFT.
    _validate_cfg_replicated(cfg, communicator(prototype_θφ))
    # Get local portion information from the prototype PencilArray
    θ_local_to_global = collect(globalindices(prototype_θφ, 1))

    # For m indices, we use the full range 0:mmax since we work with full FFT result
    m_local_to_global = collect(1:(cfg.mmax + 1))

    θ_range = 1:length(θ_local_to_global)
    m_range = 1:length(m_local_to_global)

    # Scratch-backed analysis still routes through dist_analysis_standard, so
    # avoid pinning dead buffers until a dedicated scratch implementation lands.
    active_spatial_scratch = false
    scratch = nothing

    return DistAnalysisPlan(cfg, prototype_θφ, use_rfft, θ_local_to_global, m_local_to_global,
                           m_range, θ_range, use_packed_storage, active_spatial_scratch, scratch)
end

struct DistPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
end

function DistPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; use_rfft::Bool=false)
    # use_rfft=true is wired through dist_analysis_standard and dist_synthesis
    # for real inputs/outputs. Case A (φ replicated) uses FFTW.rfft directly;
    # Case B (φ split) uses a row-subcomm gather + FFTW.rfft via
    # distributed_rfft_phi!. Complex-valued callers still use the complex FFT.
    _validate_cfg_replicated(cfg, communicator(prototype_θφ))
    return DistPlan(cfg, prototype_θφ, use_rfft)
end

struct DistSphtorPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
    with_spatial_scratch::Bool
    spatial_scratch::Union{Nothing,NamedTuple{(:Fθ, :Fφ, :Vtθ, :Vpθ, :P, :dPdx),
                                              Tuple{Matrix{ComplexF64}, Matrix{ComplexF64},
                                                    Matrix{Float64}, Matrix{Float64},
                                                    Vector{Float64}, Vector{Float64}}}}
end

function DistSphtorPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    # use_rfft=true is wired through dist_analysis_standard and dist_synthesis
    # for real inputs/outputs. Case A (φ replicated) uses FFTW.rfft directly;
    # Case B (φ split) uses a row-subcomm gather + FFTW.rfft via
    # distributed_rfft_phi!. Complex-valued callers still use the complex FFT.
    _validate_cfg_replicated(cfg, communicator(prototype_θφ))
    scratch = if with_spatial_scratch
        # Pre-allocate all scratch buffers needed for synthesis
        θ_globals = collect(globalindices(prototype_θφ, 1))
        nθ_local = length(θ_globals)
        nlon = cfg.nlon
        lmax = cfg.lmax
        (
            Fθ = Matrix{ComplexF64}(undef, nθ_local, nlon),   # Fourier coeffs for Vθ
            Fφ = Matrix{ComplexF64}(undef, nθ_local, nlon),   # Fourier coeffs for Vφ
            Vtθ = Matrix{Float64}(undef, nθ_local, nlon),     # Real output for Vθ
            Vpθ = Matrix{Float64}(undef, nθ_local, nlon),     # Real output for Vφ
            P = Vector{Float64}(undef, lmax + 1),             # Legendre polynomial buffer
            dPdx = Vector{Float64}(undef, lmax + 1),          # Legendre derivative buffer
        )
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
