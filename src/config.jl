"""
Configuration for Spherical Harmonic Transforms.

This struct contains all parameters and precomputed data needed for efficient
spherical harmonic transforms. It encapsulates both the mathematical parameters
(degrees, grid sizes) and computational optimizations (precomputed tables).

Fields
- `lmax, mmax`: maximum degree and order for spherical harmonics
- `mres`: resolution parameter for m-modes (typically 1)
- `nlat, nlon`: grid size in latitude (Gauss–Legendre) and longitude (equiangular)
- `θ, φ`: polar and azimuth angle arrays for the computational grid
- `x, w`: Gauss–Legendre nodes and weights for numerical integration (x = cos(θ))
- `Nlm`: normalization factors matrix indexed as (l+1, m+1)
- `cphi`: longitude step size (2π / nlon) for FFT operations
"""
Base.@kwdef mutable struct SHTConfig
    # Core spherical harmonic parameters
    lmax::Int                    # Maximum spherical harmonic degree
    mmax::Int                    # Maximum spherical harmonic order  
    mres::Int                    # M-resolution parameter
    nlat::Int                    # Number of latitude points (Gauss-Legendre)
    nlon::Int                    # Number of longitude points (equiangular)
    
    # Grid coordinates and quadrature
    θ::Vector{Float64}          # Polar angles (colatitude) [0, π]
    φ::Vector{Float64}          # Azimuthal angles [0, 2π)
    x::Vector{Float64}          # Gauss-Legendre nodes: x = cos(θ) ∈ [-1,1]
    w::Vector{Float64}          # Gauss-Legendre integration weights
    wlat::Vector{Float64}       # Alias: latitude weights (Gauss-Legendre) = w
    Nlm::Matrix{Float64}        # Normalization factors for Y_l^m
    cphi::Float64               # Longitude spacing: 2π / nlon
    
    # SHTns-compatible helper fields for efficient indexing
    nlm::Int                    # Total number of (l,m) modes
    li::Vector{Int}             # Degree indices for flattened (l,m) arrays
    mi::Vector{Int}             # Order indices for flattened (l,m) arrays  
    nspat::Int                  # Total spatial grid points: nlat × nlon
    ct::Vector{Float64}         # Precomputed cos(θ) values
    st::Vector{Float64}         # Precomputed sin(θ) values
    sintheta::Vector{Float64}   # Alias: sin(θ) values = st
    
    # Transform normalization and phase conventions
    norm::Symbol                # Normalization type (:orthonormal, :schmidt, etc.)
    cs_phase::Bool              # Condon-Shortley phase convention
    real_norm::Bool             # Real-valued normalization
    robert_form::Bool           # Robert form for spectral derivatives
    
    # Performance optimization: precomputed Legendre polynomials
    use_plm_tables::Bool = false                              # Enable/disable table lookup
    
    # GPU Computing support
    compute_device::Device = CPU                             # Logical compute target
    device_backend::Symbol = :cpu                            # Concrete backend identifier
    device_preference::Vector{Device} = Device[CPU]          # Preferred device order
    backend_preference::Vector{Symbol} = [:cpu]              # Backend preference (for GPU backends)
    plm_tables::Vector{Matrix{Float64}} = Matrix{Float64}[]   # P_l^m values: [m+1][l+1, lat_idx]
    dplm_tables::Vector{Matrix{Float64}} = Matrix{Float64}[]  # dP_l^m/dx values: [m+1][l+1, lat_idx]
end

"""
    create_gauss_config(lmax::Int, nlat::Int; mmax::Int=lmax, nlon::Int=max(2*lmax+1, 4)) -> SHTConfig

Create a Gauss–Legendre based SHT configuration. Constraints:
- `nlat ≥ lmax+1` for exactness up to `lmax` in θ integration.
- `nlon ≥ 2*mmax+1` to resolve azimuthal orders up to `mmax`.
"""
function create_gauss_config(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1, nlon::Int=max(2*lmax+1, 4), norm::Symbol=:orthonormal, cs_phase::Bool=true, real_norm::Bool=false, robert_form::Bool=false)
    # Validate input parameters to ensure mathematical accuracy requirements
    lmax ≥ 0 || throw(ArgumentError("lmax must be ≥ 0"))
    mmax ≥ 0 || throw(ArgumentError("mmax must be ≥ 0"))
    mmax ≤ lmax || throw(ArgumentError("mmax must be ≤ lmax"))
    mres ≥ 1 || throw(ArgumentError("mres must be ≥ 1"))
    nlat ≥ lmax + 1 || throw(ArgumentError("nlat must be ≥ lmax+1 for Gauss–Legendre accuracy"))
    nlon ≥ (2*mmax + 1) || throw(ArgumentError("nlon must be ≥ 2*mmax+1"))

    # Build the computational grid using Gauss-Legendre quadrature
    θ, φ, x, w = thetaphi_from_nodes(nlat, nlon)
    
    # Compute normalization factors for spherical harmonics
    Nlm = Nlm_table(lmax, mmax)  # currently orthonormal; future: adjust per norm/cs_phase
    
    # Calculate indexing helpers for efficient (l,m) mode access
    nlm = nlm_calc(lmax, mmax, mres)              # Total number of spectral modes
    li, mi = build_li_mi(lmax, mmax, mres)        # Degree and order index arrays
    
    # Precompute trigonometric values for performance
    ct = cos.(θ)  # cosine of colatitude
    st = sin.(θ)  # sine of colatitude
    
    # Construct and return the complete configuration
    return SHTConfig(; lmax, mmax, mres, nlat, nlon, θ, φ, x, w, wlat = w, Nlm,
                     cphi = 2π / nlon, nlm, li, mi, nspat = nlat*nlon,
                     ct, st, sintheta = st, norm, cs_phase, real_norm, robert_form,
                     compute_device = CPU, device_backend = :cpu,
                     device_preference = Device[CPU], backend_preference = [:cpu])
end

"""
    create_config(lmax::Int; mmax=lmax, mres=1, nlat=lmax+2, nlon=max(2*lmax+1,4),
                   norm::Symbol=:orthonormal, cs_phase::Bool=true,
                   real_norm::Bool=false, robert_form::Bool=false,
                   grid_type::Symbol=:gauss) -> SHTConfig

Compatibility wrapper for configuration creation used in some docs/snippets.
Currently supports only `grid_type = :gauss`, and forwards to `create_gauss_config`.
`nlat`/`nlon` default to typical exactness choices for Gauss grids.
"""
function create_config(lmax::Int; mmax::Int=lmax, mres::Int=1, nlat::Int=lmax+2,
                       nlon::Int=max(2*lmax+1, 4), norm::Symbol=:orthonormal,
                       cs_phase::Bool=true, real_norm::Bool=false,
                       robert_form::Bool=false, grid_type::Symbol=:gauss)
    if grid_type != :gauss
        throw(ArgumentError("only grid_type=:gauss is supported; use create_gauss_config for Gauss grids"))
    end
    # Make args robust to underspecified values from older docs/snippets
    nlat_eff = max(nlat, lmax + 1)              # Gauss exactness requires ≥ lmax+1
    nlon_eff = max(nlon, 2*mmax + 1)            # Azimuthal resolution requires ≥ 2*mmax+1
    return create_gauss_config(lmax, nlat_eff; mmax=mmax, mres=mres, nlon=nlon_eff,
                               norm=norm, cs_phase=cs_phase,
                               real_norm=real_norm, robert_form=robert_form)
end

"""
    create_config(::Type{T}, lmax::Int, nlat::Int, mres::Int=1; kwargs...) -> SHTConfig

Compatibility method that ignores the element type `T` and calls `create_config`.
Provided to match older example signatures like `create_config(Float64, 20, 22, 1)`.
"""
function create_config(::Type{T}, lmax::Int, nlat::Int, mres::Int=1; kwargs...) where {T}
    return create_config(lmax; nlat=nlat, mres=mres, kwargs...)
end

"""
    prepare_plm_tables!(cfg::SHTConfig)

Precompute associated Legendre tables P_l^m(x_i) for all i and m, stored as
`cfg.plm_tables[m+1][l+1, i]`. Enables faster scalar transforms on regular grids.
"""
function prepare_plm_tables!(cfg::SHTConfig)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlat = cfg.nlat
    
    # Allocate storage for Legendre polynomial tables
    # Each m-order gets its own matrix: (degree+1) × (latitude points)
    tables = [zeros(Float64, lmax + 1, nlat) for _ in 0:mmax]    # P_l^m values
    dtables = [zeros(Float64, lmax + 1, nlat) for _ in 0:mmax]  # dP_l^m/dx derivatives
    
    # Working arrays for computing one row at a time
    P = Vector{Float64}(undef, lmax + 1)      # P_l^m(x) for fixed (x,m), varying l
    dPdx = Vector{Float64}(undef, lmax + 1)   # dP_l^m/dx for fixed (x,m), varying l
    
    # Compute tables for each azimuthal order m
    for m in 0:mmax
        tbl = tables[m+1]    # Access using 1-based indexing
        dtbl = dtables[m+1]
        
        # Compute Legendre polynomials at each latitude point
        for i in 1:nlat
            # Compute all degrees l for this (x_i, m) pair
            Plm_and_dPdx_row!(P, dPdx, cfg.x[i], lmax, m)
            
            # Store results in precomputed tables
            @inbounds @views tbl[:, i] .= P      # P_l^m(x_i) for l=0:lmax
            @inbounds @views dtbl[:, i] .= dPdx  # dP_l^m/dx(x_i) for l=0:lmax
        end
    end
    
    # Enable table usage and store in configuration
    cfg.plm_tables = tables
    cfg.dplm_tables = dtables
    cfg.use_plm_tables = true
    return cfg
end

"""
    enable_plm_tables!(cfg::SHTConfig)

Alias for `prepare_plm_tables!`.
"""
enable_plm_tables!(cfg::SHTConfig) = prepare_plm_tables!(cfg)

"""
    disable_plm_tables!(cfg::SHTConfig)

Disable use of precomputed Legendre tables.
"""
function disable_plm_tables!(cfg::SHTConfig)
    cfg.use_plm_tables = false
    cfg.plm_tables = Matrix{Float64}[]
    cfg.dplm_tables = Matrix{Float64}[]
    return cfg
end

"""
    destroy_config(cfg::SHTConfig)

No-op placeholder for API symmetry with libraries that require explicit teardown.
"""
destroy_config(::SHTConfig) = nothing

# ==== GPU DEVICE MANAGEMENT FUNCTIONS ====

"""
    create_gauss_config_gpu(lmax, nlat; nlon=nothing, mres=1, device=:auto, kwargs...)

Create a spherical harmonic configuration with GPU device selection.
Enhanced version of `create_gauss_config` with automatic device detection.

# Arguments
- `device::Symbol`: Target device (:auto, :cpu, :cuda, :amdgpu)
- `device_preference::Vector{Symbol}`: Preference order when device=:auto
"""
function create_gauss_config_gpu(lmax::Int, nlat::Int; 
                                nlon::Union{Int,Nothing}=nothing, 
                                mres::Int=1,
                                device::Union{Symbol,Device}=:auto,
                                device_preference::Vector{Union{Symbol,Device}}=Device[GPU, CPU],
                                kwargs...)
    
    # Create the base configuration
    cfg = create_gauss_config(lmax, nlat; nlon=nlon, mres=mres, kwargs...)
    
    # Determine the compute device
    devices_pref, backend_pref = _normalize_device_preferences(device_preference)

    selected_device, backend, gpu_available = _resolve_device_choice(device, backend_pref)

    if selected_device == GPU && !gpu_available
        @warn "GPU device selected but CUDA backend unavailable. Falling back to CPU." 
        selected_device, backend = CPU, :cpu
    end

    cfg.compute_device = selected_device
    cfg.device_backend = backend
    cfg.device_preference = devices_pref
    cfg.backend_preference = backend_pref

    return cfg
end

function _normalize_device_preferences(entries::Vector{Union{Symbol,Device}})
    devices = Device[]
    backends = Symbol[]

    for entry in entries
        device = entry isa Device ? entry : device_from_symbol(entry)
        push!(devices, device)

        backend = entry isa Device ? device_symbol(entry) : _normalize_device_entry(entry)
        if device == GPU && backend in (:cpu, :gpu)
            backend = :cuda
        end
        push!(backends, backend)
    end

    return devices, backends
end

function _resolve_device_choice(choice::Union{Symbol,Device}, backend_pref::Vector{Symbol})
    if choice isa Symbol && choice === :auto
        return select_compute_device(backend_pref)
    end

    device = choice isa Device ? choice : device_from_symbol(choice)
    candidate_backend = choice isa Symbol ? _normalize_device_entry(choice) : (device == CPU ? :cpu : _first_gpu_backend(backend_pref))
    backend = _resolve_backend(candidate_backend, backend_pref, device)
    available = device == GPU ? (backend == :cuda && cuda_available()) : false

    return device, backend, available
end

function _resolve_backend(candidate::Symbol, backend_pref::Vector{Symbol}, device::Device)
    if device == CPU
        return :cpu
    end

    if candidate == :cuda
        return :cuda
    elseif candidate == :gpu || candidate == :cpu
        return _first_gpu_backend(backend_pref)
    else
        return candidate
    end
end

function _first_gpu_backend(backend_pref::Vector{Symbol})
    for backend in backend_pref
        backend != :cpu && return backend
    end
    return :cuda
end

function _promote_device_preference(device::Device, existing::Vector{Device})
    new_pref = Device[device]
    for entry in existing
        entry == device && continue
        push!(new_pref, entry)
    end
    return new_pref
end

function _promote_backend_preference(backend::Symbol, existing::Vector{Symbol})
    new_pref = Symbol[backend]
    for entry in existing
        entry == backend && continue
        push!(new_pref, entry)
    end
    return new_pref
end

"""
    set_config_device!(cfg::SHTConfig, device)

Change the compute device for an existing configuration. `device` can be either
the `Device` enum (`CPU`, `GPU`) or a backend symbol such as `:cpu`, `:cuda`, or
`:auto`.
"""
function set_config_device!(cfg::SHTConfig, device::Union{Device,Symbol})
    selected_device, backend, gpu_available = _resolve_device_choice(device, cfg.backend_preference)

    if selected_device == GPU && !gpu_available
        @warn "GPU device selected but CUDA backend unavailable. Falling back to CPU."
        selected_device, backend = CPU, :cpu
    end

    cfg.compute_device = selected_device
    cfg.device_backend = backend
    cfg.device_preference = _promote_device_preference(selected_device, cfg.device_preference)
    cfg.backend_preference = _promote_backend_preference(backend, cfg.backend_preference)

    return cfg
end

"""Get the logical compute device for a configuration."""
get_config_device(cfg::SHTConfig) = cfg.compute_device

"""Return the concrete backend symbol associated with the configuration."""
get_config_backend(cfg::SHTConfig) = cfg.device_backend

"""Check if a configuration is set up for GPU computing."""
is_gpu_config(cfg::SHTConfig) = cfg.compute_device == GPU
