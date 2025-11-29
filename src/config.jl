"""
Configuration for Spherical Harmonic Transforms.

This struct contains all parameters and precomputed data needed for efficient
spherical harmonic transforms. It encapsulates both the mathematical parameters
(degrees, grid sizes) and computational optimizations (precomputed tables).

Fields
- `lmax, mmax`: maximum degree and order for spherical harmonics
- `mres`: resolution parameter for m-modes (typically 1)
- `nlat, nlon`: grid size in latitude (Gauss–Legendre) and longitude (equiangular)
- `grid_type`: :gauss, :regular, or :regular_poles
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
    grid_type::Symbol = :gauss   # Grid type (:gauss, :regular, :regular_poles)
    
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
    phi_scale::Symbol = :auto    # :dft or :quad; :auto chooses by grid_type

    # Performance optimization: precomputed Legendre polynomials
    use_plm_tables::Bool = false                              # Enable/disable table lookup
    
    # GPU Computing support
    compute_device::Symbol = :cpu                             # Computing device: :cpu, :cuda, :amdgpu
    device_preference::Vector{Symbol} = [:cpu]               # Preferred device order
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
    return SHTConfig(; lmax, mmax, mres, nlat, nlon, grid_type=:gauss, θ, φ, x, w, wlat = w, Nlm,
                     cphi = 2π / nlon, nlm, li, mi, nspat = nlat*nlon,
                     ct, st, sintheta = st, norm, cs_phase, real_norm, robert_form,
                     compute_device = :cpu, device_preference = [:cpu])
end

"""
    create_regular_config(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1,
                          nlon::Int=max(2*lmax+1, 4), norm::Symbol=:orthonormal,
                          cs_phase::Bool=true, real_norm::Bool=false,
                          robert_form::Bool=false, include_poles::Bool=false,
                          precompute_plm::Bool=true, use_dh_weights::Bool=false) -> SHTConfig

Create an equiangular (regular) grid configuration. Regular grids use simple
`θ = (i+0.5)π/nlat` nodes by default; set `include_poles=true` to place nodes
directly on the poles. By default associated Legendre tables are precomputed,
which mirrors SHTns' regular-grid behaviour and improves performance.

# Driscoll-Healy Quadrature

Set `use_dh_weights=true` to use Driscoll-Healy quadrature for exact spherical
harmonic transforms. This requires:
- `include_poles=true`
- `nlat = 2*(lmax+1)` for exactness up to degree lmax
- `nlat` must be even

The DH grid uses θ = πj/n for j=0,...,n-1 (includes north pole, excludes south pole)
and provides exact quadrature via specially computed weights.

Reference: Driscoll & Healy (1994), "Computing Fourier transforms and convolutions
on the 2-sphere", Adv. Appl. Math., 15, 202-250.
"""
function create_regular_config(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1,
                               nlon::Int=max(2*lmax+1, 4), norm::Symbol=:orthonormal,
                               cs_phase::Bool=true, real_norm::Bool=false,
                               robert_form::Bool=false, include_poles::Bool=false,
                               precompute_plm::Bool=true, use_dh_weights::Bool=false)
    lmax ≥ 0 || throw(ArgumentError("lmax must be ≥ 0"))
    mmax ≥ 0 || throw(ArgumentError("mmax must be ≥ 0"))
    mmax ≤ lmax || throw(ArgumentError("mmax must be ≤ lmax"))
    mres ≥ 1 || throw(ArgumentError("mres must be ≥ 1"))
    nlon ≥ (2*mmax + 1) || throw(ArgumentError("nlon must be ≥ 2*mmax+1"))
    # Regular grids benefit from a slight oversampling in latitude for accuracy
    min_nlat = include_poles ? (lmax + 1) : (lmax + 2)
    nlat ≥ min_nlat || throw(ArgumentError("nlat must be ≥ $(min_nlat) for regular grids"))

    θ = zeros(Float64, nlat)
    w = zeros(Float64, nlat)
    x = zeros(Float64, nlat)
    if include_poles
        # Check if we should use DH weights
        if use_dh_weights
            # Validate DH requirements
            iseven(nlat) || throw(ArgumentError("DH weights require even nlat"))
            nlat == 2*(lmax + 1) || @warn "DH weights are exact when nlat=2*(lmax+1)=$(2*(lmax+1)), got nlat=$nlat"

            # Use Driscoll-Healy grid: θ = π*j/n for j=0,...,n-1
            # This includes north pole (j=0, θ=0) but not south pole (j=n would give θ=π)
            # The last point is at θ = π*(n-1)/n, just before the south pole
            w = driscoll_healy_weights(nlat)
            for i in 0:(nlat-1)
                θi = π * i / nlat
                θ[i+1] = θi
                x[i+1] = cos(θi)
            end
        else
            # Use simple trapezoidal rule with both poles
            for i in 0:(nlat-1)
                θi = i * (π / (nlat - 1))
                θ[i+1] = θi
                w[i+1] = (π / (nlat - 1)) * sin(θi)
                x[i+1] = cos(θi)
            end
        end
    else
        for i in 0:(nlat-1)
            θi = (i + 0.5) * (π / nlat)
            θ[i+1] = θi
            w[i+1] = (π / nlat) * sin(θi)
            x[i+1] = cos(θi)
        end
    end
    φ = (2π / nlon) .* collect(0:(nlon-1))

    # Compute normalization factors and indexing helpers
    Nlm = Nlm_table(lmax, mmax)
    nlm = nlm_calc(lmax, mmax, mres)
    li, mi = build_li_mi(lmax, mmax, mres)
    ct = cos.(θ); st = sin.(θ)

    # Determine grid type based on configuration
    grid_type = if use_dh_weights
        :driscoll_healy
    elseif include_poles
        :regular_poles
    else
        :regular
    end

    cfg = SHTConfig(; lmax, mmax, mres, nlat, nlon, grid_type,
                    θ, φ, x, w, wlat = w, Nlm,
                    cphi = 2π / nlon, nlm, li, mi, nspat = nlat*nlon,
                    ct, st, sintheta = st, norm, cs_phase, real_norm, robert_form,
                    compute_device = :cpu, device_preference = [:cpu])

    if precompute_plm
        prepare_plm_tables!(cfg)
    end
    return cfg
end

"""
    create_config(lmax::Int; mmax=lmax, mres=1, nlat=lmax+2, nlon=max(2*lmax+1,4),
                   norm::Symbol=:orthonormal, cs_phase::Bool=true,
                   real_norm::Bool=false, robert_form::Bool=false,
                   grid_type::Symbol=:gauss) -> SHTConfig

Compatibility wrapper for configuration creation used in some docs/snippets.
Supports Gauss–Legendre (`grid_type = :gauss`) and regular equiangular
(`grid_type = :regular` or `:regular_poles`) grids, forwarding to the
appropriate creator. `nlat`/`nlon` defaults are adjusted to satisfy accuracy
constraints for the chosen grid.
"""
function create_config(lmax::Int; mmax::Int=lmax, mres::Int=1, nlat::Int=lmax+2,
                       nlon::Int=max(2*lmax+1, 4), norm::Symbol=:orthonormal,
                       cs_phase::Bool=true, real_norm::Bool=false,
                       robert_form::Bool=false, grid_type::Symbol=:gauss,
                       include_poles::Bool=false, precompute_plm::Bool=grid_type != :gauss)
    # Make args robust to underspecified values from older docs/snippets
    min_lat = grid_type == :gauss ? (lmax + 1) : (grid_type == :regular_poles ? (lmax + 1) : (lmax + 2))
    include_poles_eff = include_poles || grid_type == :regular_poles
    nlat_eff = max(nlat, min_lat)               # Regular grids benefit from a small oversampling
    nlon_eff = max(nlon, 2*mmax + 1)            # Azimuthal resolution requires ≥ 2*mmax+1
    if grid_type == :gauss
        return create_gauss_config(lmax, nlat_eff; mmax=mmax, mres=mres, nlon=nlon_eff,
                                   norm=norm, cs_phase=cs_phase,
                                   real_norm=real_norm, robert_form=robert_form)
    elseif grid_type == :regular || grid_type == :regular_poles
        return create_regular_config(lmax, nlat_eff; mmax=mmax, mres=mres, nlon=nlon_eff,
                                     norm=norm, cs_phase=cs_phase, real_norm=real_norm,
                                     robert_form=robert_form, include_poles=include_poles_eff,
                                     precompute_plm=precompute_plm)
    else
        throw(ArgumentError("unsupported grid_type=$(grid_type); choose :gauss or :regular"))
    end
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
                                device::Symbol=:auto,
                                device_preference::Vector{Symbol}=[:cuda, :amdgpu, :cpu],
                                kwargs...)
    
    # Create the base configuration
    cfg = create_gauss_config(lmax, nlat; nlon=nlon, mres=mres, kwargs...)
    
    # Determine the compute device
    if device == :auto
        selected_device, gpu_available = select_compute_device(device_preference)
    else
        selected_device = device
        gpu_available = (device != :cpu)
    end
    
    # Set device configuration
    cfg.compute_device = selected_device
    cfg.device_preference = copy(device_preference)
    
    return cfg
end

"""
    set_config_device!(cfg::SHTConfig, device::Symbol)

Change the compute device for an existing configuration.
"""
function set_config_device!(cfg::SHTConfig, device::Symbol)
    if device ∉ [:cpu, :cuda, :amdgpu]
        throw(ArgumentError("Unsupported device: $device. Must be :cpu, :cuda, or :amdgpu"))
    end
    
    cfg.compute_device = device
    
    # Update device preference to put the selected device first
    new_preference = [device]
    for dev in cfg.device_preference
        if dev != device
            push!(new_preference, dev)
        end
    end
    cfg.device_preference = new_preference
    
    return cfg
end

"""Get the current compute device for a configuration."""
get_config_device(cfg::SHTConfig) = cfg.compute_device

"""Check if a configuration is set up for GPU computing."""
is_gpu_config(cfg::SHTConfig) = cfg.compute_device != :cpu
