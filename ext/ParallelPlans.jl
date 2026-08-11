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
    sig = hash((
        cfg.lmax, cfg.mmax, cfg.mres, cfg.nlat, cfg.nlon, cfg.nlm,
        cfg.norm, cfg.cs_phase, cfg.real_norm, cfg.robert_form,
        cfg.grid_type, cfg.phi_scale, cfg.south_pole_first,
        cfg.cphi, cfg.use_plm_tables,
        hash(cfg.θ), hash(cfg.φ), hash(cfg.x), hash(cfg.w), hash(cfg.st),
        hash(cfg.Nlm), hash(cfg.norm_scale_matrix),
        hash(cfg.plm_tables), hash(cfg.NP_tables),
    ))
    root_sig = MPI.bcast(sig, 0, comm)
    # Decide the throw COLLECTIVELY: a lone throw on the mismatched rank(s) would
    # leave the matching ranks (incl. rank 0, which always matches) proceeding into
    # the plan's later collectives → hang. Allreduce the mismatch so all ranks
    # raise together and no rank is left waiting.
    n_mismatch = MPI.Allreduce(sig != root_sig ? 1 : 0, +, comm)
    if n_mismatch != 0
        throw(ArgumentError("SHTConfig diverges across ranks ($(n_mismatch) mismatched). " *
                            "All ranks must construct cfg with identical parameters."))
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

function DistAnalysisPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; use_rfft::Bool=false)
    # use_rfft=true is wired through dist_analysis_standard and dist_synthesis
    # for real inputs/outputs. Case A (φ replicated) uses FFTW.rfft directly;
    # Case B (φ split) uses a row-subcomm gather + FFTW.rfft via
    # distributed_rfft_phi!. Complex-valued callers still use the complex FFT.
    comm = communicator(prototype_θφ)
    _validate_cfg_replicated(cfg, comm)
    θ_globals = collect(Int, globalindices(prototype_θφ, 1))
    nθ_local = length(θ_globals)
    nlon_local = size(parent(prototype_θφ), 2)
    # Reduced, like θ_is_distributed and φ_is_local_all: this selects which BRANCH
    # `dist_analysis!` takes, and the two branches enter different full-comm
    # collectives. Per-rank, a pencil with more φ-partitions than columns sends
    # the owner into the planned Allreduce and the empty ranks into
    # `dist_analysis_standard`'s own Allreduce — they never pair, and the job hangs.
    fallback_standard = MPI.Allreduce(nlon_local != cfg.nlon, |, comm)
    weights_cache = Float64[cfg.w[i] for i in θ_globals]
    x_cache = Float64[cfg.x[i] for i in θ_globals]
    P = Vector{Float64}(undef, cfg.lmax + 1)
    nbins = use_rfft ? (cfg.nlon ÷ 2 + 1) : cfg.nlon
    Fθm = Matrix{ComplexF64}(undef, nθ_local, nbins)
    Alm_work = Matrix{ComplexF64}(undef, cfg.lmax + 1, cfg.mmax + 1)
    # Reduced, not per-rank. The consumers (`dist_analysis!`,
    # `dist_analysis_sphtor!`) guard a full-comm `MPI.Allreduce!` with this flag,
    # so a topology where one rank owns every latitude and the rest own none
    # (nlat=1 over ≥2 θ-partitions) would have the owner skip while the empty
    # ranks block forever. Computed once at plan construction, not per call.
    θ_is_distributed = MPI.Allreduce(nθ_local < cfg.nlat, |, comm)
    # No Comm_split: this branch requires `!fallback_standard`, i.e. the rank owns
    # the COMPLETE φ range, so every rank's φ-colour would be 1 and the
    # split just duplicates `comm` — at the cost of a synchronizing collective
    # and a live communicator per plan, reclaimed only when GC runs the
    # finalizer. Code that rebuilds a plan per shell or timestep can exhaust the
    # MPI communicator pool that way.
    reduce_comm = comm
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
    # --- analysis scratch (always allocated; see DistAnalysisPlan) ---
    fallback_standard::Bool
    θ_globals::Vector{Int}
    x_cache::Vector{Float64}
    sθ_cache::Vector{Float64}
    inv_sθ_cache::Vector{Float64}
    weights_cache::Vector{Float64}
    P::Vector{Float64}
    dPdtheta::Vector{Float64}
    P_over_sth::Vector{Float64}
    Pbuf::Vector{Float64}
    Ftθm::Matrix{ComplexF64}
    Fpθm::Matrix{ComplexF64}
    Slm_work::Matrix{ComplexF64}
    Tlm_work::Matrix{ComplexF64}
    θ_is_distributed::Bool
    reduce_comm::MPI.Comm
end

function DistSphtorPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    # use_rfft=true is wired through dist_analysis_standard and dist_synthesis
    # for real inputs/outputs. Case A (φ replicated) uses FFTW.rfft directly;
    # Case B (φ split) uses a row-subcomm gather + FFTW.rfft via
    # distributed_rfft_phi!. Complex-valued callers still use the complex FFT.
    comm = communicator(prototype_θφ)
    _validate_cfg_replicated(cfg, comm)
    θ_globals = collect(Int, globalindices(prototype_θφ, 1))
    nθ_local = length(θ_globals)
    nlon = cfg.nlon
    lmax = cfg.lmax
    scratch = if with_spatial_scratch
        # Pre-allocate all scratch buffers needed for synthesis
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
    nlon_local = size(parent(prototype_θφ), 2)
    # Reduced — see DistAnalysisPlan: a per-rank value sends different ranks
    # into different branches, each entering its own full-comm collective.
    fallback_standard = MPI.Allreduce(nlon_local != nlon, |, comm)
    x_cache = Vector{Float64}(undef, nθ_local)
    sθ_cache = Vector{Float64}(undef, nθ_local)
    inv_sθ_cache = Vector{Float64}(undef, nθ_local)
    weights_cache = Vector{Float64}(undef, nθ_local)
    for (ii, iglob) in enumerate(θ_globals)
        x = cfg.x[iglob]
        sθ = sqrt(max(0.0, 1 - x * x))
        x_cache[ii] = x
        sθ_cache[ii] = sθ
        inv_sθ_cache[ii] = sθ == 0 ? 0.0 : 1.0 / sθ
        weights_cache[ii] = cfg.w[iglob]
    end
    nbins = use_rfft ? (nlon ÷ 2 + 1) : nlon
    Ftθm = Matrix{ComplexF64}(undef, nθ_local, nbins)
    Fpθm = Matrix{ComplexF64}(undef, nθ_local, nbins)
    Slm_work = Matrix{ComplexF64}(undef, lmax + 1, cfg.mmax + 1)
    Tlm_work = Matrix{ComplexF64}(undef, lmax + 1, cfg.mmax + 1)
    # Reduced, not per-rank. The consumers (`dist_analysis!`,
    # `dist_analysis_sphtor!`) guard a full-comm `MPI.Allreduce!` with this flag,
    # so a topology where one rank owns every latitude and the rest own none
    # (nlat=1 over ≥2 θ-partitions) would have the owner skip while the empty
    # ranks block forever. Computed once at plan construction, not per call.
    θ_is_distributed = MPI.Allreduce(nθ_local < cfg.nlat, |, comm)
    reduce_comm = comm   # see DistAnalysisPlan: the split was provably a no-op here
    return DistSphtorPlan(cfg, prototype_θφ, use_rfft, with_spatial_scratch, scratch,
                          fallback_standard, θ_globals, x_cache, sθ_cache, inv_sθ_cache,
                          weights_cache,
                          Vector{Float64}(undef, lmax + 1), Vector{Float64}(undef, lmax + 1),
                          Vector{Float64}(undef, lmax + 1), Vector{Float64}(undef, lmax + 2),
                          Ftθm, Fpθm, Slm_work, Tlm_work, θ_is_distributed, reduce_comm)
end

struct DistQstPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
    # QST analysis = scalar (radial) + sphtor (tangential); delegate to the
    # planned sub-transforms so all scratch lives in the sub-plans.
    scalar_plan::DistAnalysisPlan
    sphtor_plan::DistSphtorPlan
end

function DistQstPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    scalar_plan = DistAnalysisPlan(cfg, prototype_θφ; use_rfft)
    sphtor_plan = DistSphtorPlan(cfg, prototype_θφ; with_spatial_scratch, use_rfft)
    return DistQstPlan(cfg, prototype_θφ, use_rfft, scalar_plan, sphtor_plan)
end
