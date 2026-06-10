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
    # φ-distributed prototypes need the longitude gather; dist_analysis! falls
    # back to the allocating standard path for them (that layout anti-scales
    # and already warns).
    fallback_standard::Bool
    # Per-call scratch, sized once from cfg + the prototype's local θ slab so
    # dist_analysis! runs allocation-free after warmup.
    θ_globals::Vector{Int}
    weights_cache::Vector{Float64}
    x_cache::Vector{Float64}
    P::Vector{Float64}
    Fθm::Matrix{ComplexF64}
    Alm_work::Matrix{ComplexF64}
    θ_is_distributed::Bool
    # θ-column subcomm for the partial-sum reduction (Comm_split once here
    # instead of every call). Equals the full communicator when θ is not
    # distributed or for the fallback path; freed by MPI_Finalize with the
    # plan's lifetime (plans are long-lived by design).
    reduce_comm::MPI.Comm
end

function DistAnalysisPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; use_rfft::Bool=false, use_packed_storage::Bool=true, with_spatial_scratch::Bool=false)
    # use_rfft=true is wired through dist_analysis_standard and dist_synthesis
    # for real inputs/outputs. Case A (φ replicated) uses FFTW.rfft directly;
    # Case B (φ split) uses a row-subcomm gather + FFTW.rfft via
    # distributed_rfft_phi!. Complex-valued callers still use the complex FFT.
    comm = communicator(prototype_θφ)
    _validate_cfg_replicated(cfg, comm)
    # Keep the keywords for API compatibility; the planned path always uses
    # dense coefficient storage.
    _ = use_packed_storage
    _ = with_spatial_scratch
    θ_globals = collect(Int, globalindices(prototype_θφ, 1))
    nθ_local = length(θ_globals)
    nlon_local = size(parent(prototype_θφ), 2)
    fallback_standard = nlon_local != cfg.nlon
    weights_cache = Float64[cfg.w[i] for i in θ_globals]
    x_cache = Float64[cfg.x[i] for i in θ_globals]
    P = Vector{Float64}(undef, cfg.lmax + 1)
    nbins = use_rfft ? (cfg.nlon ÷ 2 + 1) : cfg.nlon
    Fθm = Matrix{ComplexF64}(undef, nθ_local, nbins)
    Alm_work = Matrix{ComplexF64}(undef, cfg.lmax + 1, cfg.mmax + 1)
    θ_is_distributed = nθ_local < cfg.nlat
    reduce_comm = if θ_is_distributed && !fallback_standard
        # Reduce only across ranks sharing this φ-segment (see
        # dist_analysis_standard STEP 5 for the 2D-pencil rationale).
        φ_globals = globalindices(prototype_θφ, 2)
        MPI.Comm_split(comm, Int(first(φ_globals)), MPI.Comm_rank(comm))
    else
        comm
    end
    return DistAnalysisPlan(cfg, prototype_θφ, use_rfft, fallback_standard,
                            θ_globals, weights_cache, x_cache, P, Fθm, Alm_work,
                            θ_is_distributed, reduce_comm)
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

const _SphtorScratch = NamedTuple{(:Fθ, :Fφ, :Vtθ, :Vpθ, :P, :dPdx),
                                   Tuple{Matrix{ComplexF64}, Matrix{ComplexF64},
                                         Matrix{Float64}, Matrix{Float64},
                                         Vector{Float64}, Vector{Float64}}}

struct DistSphtorPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
    with_spatial_scratch::Bool
    spatial_scratch::Union{Nothing, _SphtorScratch}
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
end

function DistQstPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    # Keep the keyword for API compatibility; QST plans currently do not own
    # separate scratch storage beyond the scalar/vector sub-transforms.
    _ = with_spatial_scratch
    return DistQstPlan(cfg, prototype_θφ, use_rfft)
end
