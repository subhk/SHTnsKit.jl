#=
================================================================================
config.jl - SHTConfig Structure and Configuration Functions
================================================================================

This file defines the core SHTConfig struct that holds all parameters and
precomputed data for spherical harmonic transforms. Understanding this struct
is essential for debugging and optimizing SHT computations.

KEY CONCEPTS
------------
1. Grid Setup:
   - nlat: Number of Gauss-Legendre points in latitude (θ direction)
   - nlon: Number of equispaced points in longitude (φ direction)
   - x[i] = cos(θ[i]): Gauss-Legendre nodes, ordered from +1 (north) to -1 (south)
   - w[i]: Gauss-Legendre weights, sum(w) ≈ 2.0

2. Spectral Setup:
   - lmax: Maximum degree l (total wavenumber), determines angular resolution
   - mmax: Maximum order m (azimuthal wavenumber), usually equals lmax
   - mres: M-resolution (usually 1), allows skipping m values

3. Accuracy Requirements:
   - nlat ≥ lmax + 1: Required for exact Gauss-Legendre integration
   - nlon ≥ 2*mmax + 1: Required for Nyquist sampling of m modes

4. Normalization:
   - Nlm[l+1, m+1]: Normalization factor for Y_l^m
   - cphi = 2π/nlon: Longitude spacing factor for FFT

COMMON DEBUGGING CHECKS
-----------------------
```julia
# Verify Gauss weights sum correctly
@assert abs(sum(cfg.w) - 2.0) < 1e-10 "Gauss weights should sum to 2"

# Check grid ordering (north to south by default)
@assert cfg.x[1] > cfg.x[end] "x should decrease (north to south)"
@assert cfg.x[1] ≈ cos(cfg.θ[1]) "x = cos(θ)"

# Verify dimensions match
@assert size(cfg.Nlm) == (cfg.lmax+1, cfg.mmax+1)
@assert length(cfg.x) == cfg.nlat
@assert length(cfg.φ) == cfg.nlon
```

PERFORMANCE MODES
-----------------
1. On-the-fly (cfg.on_the_fly = true):
   - Computes Legendre polynomials during each transform
   - Lower memory usage, slightly slower
   - Good for large lmax or memory-constrained systems

2. Table lookup (cfg.use_plm_tables = true):
   - Precomputes all P_l^m values in cfg.plm_tables
   - Higher memory usage, faster transforms
   - Good for repeated transforms at same configuration

CREATING CONFIGURATIONS
----------------------
- create_gauss_config(lmax, nlat): Standard Gauss-Legendre grid
- create_regular_config(lmax, nlat): Equispaced latitude grid
- create_gauss_config_spf(lmax, nlat): South-pole-first ordering

================================================================================
=#

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
    phi_scale::Symbol = :auto    # :dft, :quad, or :auto (grid-driven)

    # Performance optimization: Legendre polynomial computation mode
    use_plm_tables::Bool = false                              # Enable/disable table lookup
    on_the_fly::Bool = false                                  # Force on-the-fly computation (never use tables)

    # GPU Computing support
    compute_device::Symbol = :cpu                             # Computing device: :cpu, :cuda, :amdgpu
    device_preference::Vector{Symbol} = [:cpu]               # Preferred device order
    plm_tables::Vector{Matrix{Float64}} = Matrix{Float64}[]   # P_l^m values: [m+1][l+1, lat_idx]
    dplm_tables::Vector{Matrix{Float64}} = Matrix{Float64}[]  # dP_l^m/dx values: [m+1][l+1, lat_idx]

    # Batch transform configuration (for processing multiple fields simultaneously)
    howmany::Int = 1                                          # Number of fields to process in batch
    spec_dist::Int = 0                                        # Distance between spectral arrays (0 = contiguous)

    # Data ordering configuration
    south_pole_first::Bool = false                            # If true, latitude data starts at south pole (θ=π) instead of north pole (θ=0)

    # Memory padding configuration (for cache optimization)
    allow_padding::Bool = false                               # If true, arrays may have padding for cache optimization
    nlat_padded::Int = 0                                      # Padded nlat value (stride between phi values), 0 means no padding (use nlat)
    spat_dist::Int = 0                                        # Distance between spatial arrays in batch mode (0 = contiguous)
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
                     ct, st, sintheta = st, norm, cs_phase, real_norm, robert_form, phi_scale=:dft,
                     compute_device = :cpu, device_preference = [:cpu])
end

"""
    create_gauss_fly_config(lmax::Int, nlat::Int; kwargs...) -> SHTConfig

Create a Gauss–Legendre configuration with on-the-fly Legendre polynomial computation.

This mode computes Legendre polynomials during each transform instead of using
precomputed tables. Benefits:
- **Lower memory usage**: No storage for P_l^m tables (saves O(lmax² × nlat) memory)
- **Better for large lmax**: When lmax is large, tables become memory-intensive
- **Good cache behavior**: On modern CPUs, on-the-fly computation can be faster
  due to better cache utilization

Trade-offs:
- May be slower for repeated transforms with small lmax
- No speedup from table precomputation

This mirrors the `sht_gauss_fly` mode in SHTns C library.

# Arguments
Same as `create_gauss_config`.

# Example
```julia
# For large lmax, on-the-fly mode saves significant memory
cfg = create_gauss_fly_config(1024, 1026)

# Performs transform computing Legendre polynomials on-the-fly
alm = analysis(cfg, field)
```

See also: [`create_gauss_config`](@ref), [`set_on_the_fly!`](@ref), [`set_use_tables!`](@ref)
"""
function create_gauss_fly_config(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1,
                                  nlon::Int=max(2*lmax+1, 4), norm::Symbol=:orthonormal,
                                  cs_phase::Bool=true, real_norm::Bool=false,
                                  robert_form::Bool=false)
    # Create base Gauss config
    cfg = create_gauss_config(lmax, nlat; mmax=mmax, mres=mres, nlon=nlon,
                              norm=norm, cs_phase=cs_phase, real_norm=real_norm,
                              robert_form=robert_form)

    # Force on-the-fly mode
    cfg.on_the_fly = true
    cfg.use_plm_tables = false
    cfg.plm_tables = Matrix{Float64}[]
    cfg.dplm_tables = Matrix{Float64}[]

    return cfg
end

"""
    set_on_the_fly!(cfg::SHTConfig)

Enable on-the-fly Legendre polynomial computation mode. This disables precomputed
tables and computes P_l^m(x) during each transform.

Benefits:
- Lower memory usage (no table storage)
- Better for large lmax values
- Can be faster on modern CPUs due to cache effects

See also: [`set_use_tables!`](@ref), [`create_gauss_fly_config`](@ref)
"""
function set_on_the_fly!(cfg::SHTConfig)
    cfg.on_the_fly = true
    cfg.use_plm_tables = false
    cfg.plm_tables = Matrix{Float64}[]
    cfg.dplm_tables = Matrix{Float64}[]
    return cfg
end

"""
    set_use_tables!(cfg::SHTConfig)

Enable precomputed Legendre polynomial tables. This precomputes P_l^m(x_i) for
all grid points and stores them for reuse.

Benefits:
- Faster for repeated transforms with same grid
- No redundant computation of Legendre polynomials

Trade-offs:
- Higher memory usage: O(lmax × mmax × nlat)
- Initial table computation overhead

See also: [`set_on_the_fly!`](@ref), [`prepare_plm_tables!`](@ref)
"""
function set_use_tables!(cfg::SHTConfig)
    cfg.on_the_fly = false
    if !cfg.use_plm_tables || isempty(cfg.plm_tables)
        prepare_plm_tables!(cfg)
    end
    return cfg
end

"""
    is_on_the_fly(cfg::SHTConfig) -> Bool

Check if configuration is set to on-the-fly Legendre computation mode.
"""
is_on_the_fly(cfg::SHTConfig) = cfg.on_the_fly || !cfg.use_plm_tables

"""
    estimate_table_memory(lmax::Int, mmax::Int, nlat::Int) -> Int

Estimate memory usage (in bytes) for precomputed Legendre polynomial tables.

Returns the approximate memory needed for both P_l^m and dP_l^m/dx tables.
"""
function estimate_table_memory(lmax::Int, mmax::Int, nlat::Int)
    # Each table has (lmax+1) × nlat Float64 values per m
    # We have (mmax+1) such tables, for both P and dP
    bytes_per_table = (lmax + 1) * nlat * sizeof(Float64)
    num_tables = (mmax + 1) * 2  # P and dP tables
    return bytes_per_table * num_tables
end

"""
    estimate_table_memory(cfg::SHTConfig) -> Int

Estimate memory usage for Legendre tables based on configuration.
"""
estimate_table_memory(cfg::SHTConfig) = estimate_table_memory(cfg.lmax, cfg.mmax, cfg.nlat)

# ============================================================================
# SOUTH POLE FIRST MODE FUNCTIONS
# ============================================================================

"""
    set_south_pole_first!(cfg::SHTConfig)

Enable south-pole-first latitude ordering. In this mode, latitude data starts
at the south pole (θ=π) instead of the default north pole (θ=0).

This reverses the internal grid arrays (θ, x, w, ct, st) and recalculates
any precomputed Legendre tables if necessary. This matches the `SHT_SOUTH_POLE_FIRST`
flag behavior in the SHTns C library.

# Example
```julia
cfg = create_gauss_config(32, 34)
set_south_pole_first!(cfg)  # Data now starts at south pole
```

See also: [`set_north_pole_first!`](@ref), [`is_south_pole_first`](@ref)
"""
function set_south_pole_first!(cfg::SHTConfig)
    if cfg.south_pole_first
        return cfg  # Already in south-pole-first mode
    end

    # Reverse latitude-dependent arrays
    cfg.θ = reverse(cfg.θ)
    cfg.w = reverse(cfg.w)
    cfg.x = reverse(cfg.x)
    cfg.wlat = cfg.w  # wlat is an alias for w
    cfg.ct = cos.(cfg.θ)
    cfg.st = sin.(cfg.θ)
    cfg.sintheta = cfg.st

    # Mark as south-pole-first
    cfg.south_pole_first = true

    # Recompute Legendre tables if they were precomputed
    if cfg.use_plm_tables && !isempty(cfg.plm_tables)
        prepare_plm_tables!(cfg)
    end

    return cfg
end

"""
    set_north_pole_first!(cfg::SHTConfig)

Enable north-pole-first latitude ordering (the default). In this mode, latitude
data starts at the north pole (θ=0).

This reverses the internal grid arrays if the configuration was previously in
south-pole-first mode, and recalculates any precomputed Legendre tables.

# Example
```julia
cfg = create_gauss_config(32, 34)
set_south_pole_first!(cfg)  # Data starts at south pole
set_north_pole_first!(cfg)  # Back to default (north pole first)
```

See also: [`set_south_pole_first!`](@ref), [`is_south_pole_first`](@ref)
"""
function set_north_pole_first!(cfg::SHTConfig)
    if !cfg.south_pole_first
        return cfg  # Already in north-pole-first mode
    end

    # Reverse latitude-dependent arrays back to north-pole-first
    cfg.θ = reverse(cfg.θ)
    cfg.w = reverse(cfg.w)
    cfg.x = reverse(cfg.x)
    cfg.wlat = cfg.w
    cfg.ct = cos.(cfg.θ)
    cfg.st = sin.(cfg.θ)
    cfg.sintheta = cfg.st

    # Mark as north-pole-first
    cfg.south_pole_first = false

    # Recompute Legendre tables if they were precomputed
    if cfg.use_plm_tables && !isempty(cfg.plm_tables)
        prepare_plm_tables!(cfg)
    end

    return cfg
end

"""
    is_south_pole_first(cfg::SHTConfig) -> Bool

Check if the configuration uses south-pole-first latitude ordering.

Returns `true` if latitude data starts at the south pole (θ=π),
`false` if it starts at the north pole (θ=0, the default).
"""
is_south_pole_first(cfg::SHTConfig) = cfg.south_pole_first

"""
    create_gauss_config_spf(lmax::Int, nlat::Int; kwargs...) -> SHTConfig

Create a Gauss-Legendre configuration with south-pole-first latitude ordering.

This is equivalent to calling `create_gauss_config` followed by `set_south_pole_first!`.
All keyword arguments are passed to `create_gauss_config`.

# Example
```julia
# Create a south-pole-first configuration
cfg = create_gauss_config_spf(32, 34)
is_south_pole_first(cfg)  # true
```

See also: [`create_gauss_config`](@ref), [`set_south_pole_first!`](@ref)
"""
function create_gauss_config_spf(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1,
                                  nlon::Int=max(2*lmax+1, 4), norm::Symbol=:orthonormal,
                                  cs_phase::Bool=true, real_norm::Bool=false,
                                  robert_form::Bool=false)
    cfg = create_gauss_config(lmax, nlat; mmax=mmax, mres=mres, nlon=nlon,
                              norm=norm, cs_phase=cs_phase, real_norm=real_norm,
                              robert_form=robert_form)
    set_south_pole_first!(cfg)
    return cfg
end

# ============================================================================
# MEMORY PADDING FUNCTIONS
# ============================================================================

"""
    compute_optimal_padding(nlat::Int, nlon::Int; sizeof_real::Int=8) -> Int

Compute the optimal padding to add between latitude lines to avoid cache bank conflicts.
This mimics the SHTns C library's padding strategy.

# Arguments
- `nlat`: Number of latitude points
- `nlon`: Number of longitude points (phi)
- `sizeof_real`: Size of real number in bytes (default 8 for Float64)

# Returns
The number of extra elements to add per latitude line (padding amount).

# Details
The padding strategy aims to:
1. Align memory on 32-byte or 64-byte boundaries
2. Avoid 8KB cache bank conflicts
3. Avoid L2 cache bank/channel conflicts (especially important for AMD GPUs)

This is most beneficial for:
- Large transforms where cache effects matter
- Multi-threaded transforms
- GPU computing
"""
function compute_optimal_padding(nlat::Int, nlon::Int; sizeof_real::Int=8)
    if nlon <= 1
        return 0  # No padding needed for 1D or scalar
    end

    stride_bytes = nlat * sizeof_real
    min_pad_bytes = 8 * sizeof_real  # Minimum 8 elements padding

    # Compute alignment-based padding
    # Check if full data fits in L2 cache (~8MB typical)
    # Use 64-byte alignment for L2-resident data, 32-byte otherwise
    estimated_total = nlon * (stride_bytes + min_pad_bytes)
    alignment = estimated_total < 8192 * 1024 ? 64 : 32

    # Compute padding needed for alignment
    alignment_remainder = stride_bytes % alignment
    alignment_pad = alignment_remainder == 0 ? 0 : (alignment - alignment_remainder)

    # Use the larger of minimum padding and alignment padding
    pad_bytes = max(min_pad_bytes, alignment_pad)

    # Avoid multiples of 8KB (cache bank conflicts)
    if (stride_bytes + pad_bytes) % 8192 == 0
        pad_bytes += 64
    end

    # Avoid 128-byte alignment issues (L2 cache bank conflicts)
    if (stride_bytes + pad_bytes) % 128 == 0
        pad_bytes += alignment == 64 ? 64 : 32
    end

    # Convert to number of elements
    return div(pad_bytes, sizeof_real)
end

"""
    set_allow_padding!(cfg::SHTConfig; auto_compute::Bool=true)

Enable memory padding for cache optimization. When enabled, spatial arrays
can have extra padding between latitude lines to avoid cache bank conflicts.

This can improve performance by 1-50% depending on the problem size and hardware,
especially for multi-threaded and GPU transforms.

# Arguments
- `cfg`: SHTConfig to modify
- `auto_compute`: If true, automatically compute optimal padding. If false,
  you must set `cfg.nlat_padded` manually.

# Example
```julia
cfg = create_gauss_config(64, 66)
set_allow_padding!(cfg)
println("Padded nlat: \$(cfg.nlat_padded)")  # May be > 66
```

See also: [`disable_padding!`](@ref), [`is_padding_enabled`](@ref), [`get_nlat_padded`](@ref)
"""
function set_allow_padding!(cfg::SHTConfig; auto_compute::Bool=true)
    cfg.allow_padding = true

    if auto_compute
        pad = compute_optimal_padding(cfg.nlat, cfg.nlon)
        cfg.nlat_padded = cfg.nlat + pad
    elseif cfg.nlat_padded == 0
        cfg.nlat_padded = cfg.nlat
    end

    # Compute spatial distance for batch transforms
    cfg.spat_dist = cfg.nlat_padded * cfg.nlon

    return cfg
end

"""
    disable_padding!(cfg::SHTConfig)

Disable memory padding. Spatial arrays will use contiguous memory without
extra padding between latitude lines.

See also: [`set_allow_padding!`](@ref), [`is_padding_enabled`](@ref)
"""
function disable_padding!(cfg::SHTConfig)
    cfg.allow_padding = false
    cfg.nlat_padded = 0
    cfg.spat_dist = 0
    return cfg
end

"""
    is_padding_enabled(cfg::SHTConfig) -> Bool

Check if memory padding is enabled for the configuration.
"""
is_padding_enabled(cfg::SHTConfig) = cfg.allow_padding

"""
    get_nlat_padded(cfg::SHTConfig) -> Int

Get the padded latitude dimension. Returns `cfg.nlat` if padding is disabled,
or `cfg.nlat_padded` if padding is enabled.

This is the stride between successive phi (longitude) values in padded spatial arrays.
"""
function get_nlat_padded(cfg::SHTConfig)
    if cfg.allow_padding && cfg.nlat_padded > 0
        return cfg.nlat_padded
    else
        return cfg.nlat
    end
end

"""
    get_spat_dist(cfg::SHTConfig) -> Int

Get the distance between spatial arrays in batch mode. Returns `cfg.nspat` if
padding is disabled, or the padded spatial distance if enabled.
"""
function get_spat_dist(cfg::SHTConfig)
    if cfg.allow_padding && cfg.spat_dist > 0
        return cfg.spat_dist
    else
        return cfg.nspat
    end
end

"""
    allocate_padded_spatial(cfg::SHTConfig, T::Type=Float64) -> Array{T}

Allocate a spatial array with proper padding if enabled.

Returns a 2D array of size (nlat_padded, nlon) where nlat_padded >= nlat.
The first nlat rows contain the actual data; remaining rows are padding.

# Example
```julia
cfg = create_gauss_config(64, 66)
set_allow_padding!(cfg)
field = allocate_padded_spatial(cfg)
# field has size (nlat_padded, nlon), use field[1:cfg.nlat, :] for data
```
"""
function allocate_padded_spatial(cfg::SHTConfig, T::Type=Float64)
    nlat_p = get_nlat_padded(cfg)
    return zeros(T, nlat_p, cfg.nlon)
end

"""
    allocate_padded_spatial_batch(cfg::SHTConfig, nfields::Int, T::Type=Float64) -> Array{T}

Allocate spatial arrays for batch transforms with proper padding if enabled.

Returns a 3D array of size (nlat_padded, nlon, nfields).
"""
function allocate_padded_spatial_batch(cfg::SHTConfig, nfields::Int, T::Type=Float64)
    nlat_p = get_nlat_padded(cfg)
    return zeros(T, nlat_p, cfg.nlon, nfields)
end

"""
    copy_to_padded!(dest::AbstractMatrix, src::AbstractMatrix, cfg::SHTConfig)

Copy data from a contiguous spatial array to a padded spatial array.
"""
function copy_to_padded!(dest::AbstractMatrix, src::AbstractMatrix, cfg::SHTConfig)
    @assert size(src, 1) == cfg.nlat
    @assert size(src, 2) == cfg.nlon
    @assert size(dest, 1) >= cfg.nlat
    @assert size(dest, 2) == cfg.nlon

    dest[1:cfg.nlat, :] .= src
    return dest
end

"""
    copy_from_padded!(dest::AbstractMatrix, src::AbstractMatrix, cfg::SHTConfig)

Copy data from a padded spatial array to a contiguous spatial array.
"""
function copy_from_padded!(dest::AbstractMatrix, src::AbstractMatrix, cfg::SHTConfig)
    @assert size(dest, 1) == cfg.nlat
    @assert size(dest, 2) == cfg.nlon
    @assert size(src, 1) >= cfg.nlat
    @assert size(src, 2) == cfg.nlon

    dest .= src[1:cfg.nlat, :]
    return dest
end

"""
    estimate_padding_overhead(cfg::SHTConfig) -> Float64

Estimate the memory overhead from padding as a percentage.

Returns 0.0 if padding is disabled.
"""
function estimate_padding_overhead(cfg::SHTConfig)
    if !cfg.allow_padding || cfg.nlat_padded <= cfg.nlat
        return 0.0
    end
    return 100.0 * (cfg.nlat_padded - cfg.nlat) / cfg.nlat
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

    # Determine grid type and phi_scale based on configuration
    grid_type = if use_dh_weights
        :driscoll_healy
    elseif include_poles
        :regular_poles
    else
        :regular
    end

    # Regular/equiangular grids use "quad" scaling (nlon/(2π)) for proper roundtrip
    # This matches the spherical quadrature convention where the φ integral is
    # normalized by 1/(2π) rather than using DFT normalization
    phi_scale = :quad

    cfg = SHTConfig(; lmax, mmax, mres, nlat, nlon, grid_type,
                    θ, φ, x, w, wlat = w, Nlm,
                    cphi = 2π / nlon, nlm, li, mi, nspat = nlat*nlon,
                    ct, st, sintheta = st, norm, cs_phase, real_norm, robert_form, phi_scale,
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
- `device::Symbol`: Target device (:auto, :cpu, :cuda)
- `device_preference::Vector{Symbol}`: Preference order when device=:auto
"""
function create_gauss_config_gpu(lmax::Int, nlat::Int;
                                nlon::Union{Int,Nothing}=nothing,
                                mres::Int=1,
                                device::Symbol=:auto,
                                device_preference::Vector{Symbol}=[:cuda, :cpu],
                                kwargs...)

    # Compute effective nlon: use provided value or default
    nlon_eff = isnothing(nlon) ? max(2*lmax + 1, 4) : nlon

    # Create the base configuration
    cfg = create_gauss_config(lmax, nlat; nlon=nlon_eff, mres=mres, kwargs...)
    
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
