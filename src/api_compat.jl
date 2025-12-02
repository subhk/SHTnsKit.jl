"""
API-compatibility layer that mirrors the C `shtns.h` surface in pure Julia.
"""

# SHTns flag constants (mirror shtns.h)
const SHT_GAUSS           = 0
const SHT_AUTO            = 1
const SHT_REGULAR         = 2
const SHT_REG_FAST        = 3
const SHT_QUICK_INIT      = 4
const SHT_REGULAR_POLES   = 5
const SHT_GAUSS_FLY       = 6
const SHT_SOUTH_POLE_FIRST = 256*32
const SHT_NO_CS_PHASE     = 256*4
const SHT_REAL_NORM       = 256*8

_grid_symbol(code::Int) = code == SHT_GAUSS ? :gauss :
                          code == SHT_AUTO ? :gauss :
                          code == SHT_REGULAR ? :regular :
                          code == SHT_REG_FAST ? :regular :
                          code == SHT_QUICK_INIT ? :regular :
                          code == SHT_REGULAR_POLES ? :regular_poles :
                          code == SHT_GAUSS_FLY ? :gauss : :gauss

_min_nlat_for_grid(code::Int, lmax::Int) = code == SHT_REGULAR_POLES ? (lmax + 1) :
                                           (code == SHT_REGULAR || code == SHT_REG_FAST || code == SHT_QUICK_INIT ? (lmax + 2) : (lmax + 1))

function _parse_norm_bits(nval::Int)
    base_norm = nval % 256
    cs_phase = (nval & SHT_NO_CS_PHASE) == 0
    real_norm = (nval & SHT_REAL_NORM) != 0
    nsym = base_norm == 0 ? :orthonormal : base_norm == 1 ? :fourpi : base_norm == 2 ? :schmidt : :orthonormal
    return nsym, cs_phase, real_norm
end

const _SHTNS_VERBOSE = Ref(0)

"""shtns_verbose(level::Integer)"""
function shtns_verbose(level::Integer)
    _SHTNS_VERBOSE[] = Int(level)
    return nothing
end

"""shtns_print_version()"""
function shtns_print_version()
    println("SHTnsKit julia-compatible API; pure Julia core")
end

"""shtns_get_build_info() -> String"""
function shtns_get_build_info()
    return "SHTnsKit.jl: pure Julia SHT; orthonormal normalization; Gauss grid"
end

"""shtns_init(flags, lmax, mmax, mres, nlat, nphi) -> SHTConfig"""
function shtns_init(flags::Integer, lmax::Integer, mmax::Integer, mres::Integer, nlat::Integer, nphi::Integer)
    f = Int(flags)
    lmax = Int(lmax); mmax = Int(mmax); mres = Int(mres)
    nlat = Int(nlat); nphi = Int(nphi)
  
    grid_code = f % 256
    grid_sym = _grid_symbol(grid_code)
    min_lat = _min_nlat_for_grid(grid_code, lmax)
    nlat_eff = max(nlat, min_lat)
    nphi_eff = max(nphi, 2*mmax + 1, 4)
    include_poles = grid_sym == :regular_poles
    precompute_plm = !(grid_code == SHT_QUICK_INIT || grid_code == SHT_GAUSS_FLY)

    cfg = if grid_sym == :gauss
        create_gauss_config(lmax, nlat_eff; mmax=mmax, mres=mres, nlon=nphi_eff)
    else
        create_regular_config(lmax, nlat_eff; mmax=mmax, mres=mres, nlon=nphi_eff,
                              include_poles=include_poles, precompute_plm=precompute_plm)
    end

    if (f & SHT_SOUTH_POLE_FIRST) != 0
        cfg.θ = reverse(cfg.θ); cfg.w = reverse(cfg.w); cfg.x = reverse(cfg.x); cfg.wlat = cfg.w
        cfg.ct = cos.(cfg.θ); cfg.st = sin.(cfg.θ); cfg.sintheta = cfg.st
        if cfg.use_plm_tables
            prepare_plm_tables!(cfg)
        end
    end
    return cfg
end

"""shtns_create(lmax, mmax, mres, norm) -> SHTConfig"""
function shtns_create(lmax::Integer, mmax::Integer, mres::Integer, norm::Integer)
    # Parse norm and flags per SHTns
    nsym, cs_phase, real_norm = _parse_norm_bits(Int(norm))
    # defer grid selection to set_grid; but provide minimal Gauss grid consistent with sizes
    nlat = Int(lmax) + 1
    nphi = max(2*Int(mmax)+1, 4)
    return create_gauss_config(Int(lmax), nlat; mmax=Int(mmax), mres=Int(mres), nlon=nphi, norm=nsym, cs_phase=cs_phase, real_norm=real_norm)
end

"""shtns_set_grid(cfg, flags, eps, nlat, nphi) -> Int"""
function shtns_set_grid(cfg::SHTConfig, flags::Integer, eps::Real, nlat::Integer, nphi::Integer)
    # Parse flags: low byte = grid type, higher bytes carry options
    f = Int(flags)
    grid_type = f % 256
    south_pole_first = (f & (256*32)) != 0
    grid_sym = _grid_symbol(grid_type)
    min_lat = _min_nlat_for_grid(grid_type, cfg.lmax)
    nlat = max(Int(nlat), min_lat)
    nphi = max(Int(nphi), 2*cfg.mmax + 1, 4)
  
    # Build grid
    θ = zeros(Float64, nlat); w = zeros(Float64, nlat)
    x = zeros(Float64, nlat)
    φ = (2π / nphi) .* collect(0:(nphi-1))
  
    if grid_type == 0 || grid_type == 1 || grid_type == 6  # gauss, auto, gauss_fly
        x, w = gausslegendre(nlat)
        θ = acos.(x)
    elseif grid_type == 2 || grid_type == 3 || grid_type == 4  # reg_fast, reg_dct, quick_init
        # equiangular midpoints (no poles)
        for i in 0:(nlat-1)
            θ[i+1] = (i + 0.5) * (π / nlat)
            w[i+1] = (π / nlat) * sin(θ[i+1])
            x[i+1] = cos(θ[i+1])
        end
    elseif grid_type == 5  # reg_poles, include poles
        for i in 0:(nlat-1)
            θ[i+1] = i * (π / (nlat-1))
            w[i+1] = (π / (nlat-1)) * sin(θ[i+1])
            x[i+1] = cos(θ[i+1])
        end
    else
        # default to gauss
        x, w = gausslegendre(nlat)
        θ = acos.(x)
    end
    if south_pole_first
        θ = reverse(θ); w = reverse(w); x = reverse(x)
    end
  
    # Update cfg in-place
    cfg.nlat = nlat; cfg.nlon = nphi; cfg.grid_type = grid_sym
    cfg.θ = θ; cfg.φ = φ; 
    cfg.x = x; cfg.w = w; 
    cfg.wlat = w
    cfg.ct = cos.(θ); cfg.st = sin.(θ); 
    cfg.sintheta = cfg.st
    cfg.nspat = nlat * nphi
    cfg.cphi = 2π / nphi
  
    # Precompute Plm tables for regular grids (skip quick_init)
    if grid_type == SHT_REGULAR || grid_type == SHT_REG_FAST || grid_type == SHT_REGULAR_POLES
        prepare_plm_tables!(cfg)
    else
        cfg.use_plm_tables = false
        cfg.plm_tables = Matrix{Float64}[]
        cfg.dplm_tables = Matrix{Float64}[]
    end
    return 0
end

"""shtns_set_grid_auto(cfg, flags, eps, nl_order, nlat_ref, nphi_ref) -> Int"""
function shtns_set_grid_auto(cfg::SHTConfig, flags::Integer, eps::Real, nl_order::Integer, nlat_ref::Ref{Int}, nphi_ref::Ref{Int})
    f = Int(flags)
    grid_type = f % 256
    nlat_ref[] = _min_nlat_for_grid(grid_type, cfg.lmax)
    nphi_ref[] = max(2*cfg.mmax+1, 4)
    return 0
end

"""shtns_create_with_grid(cfg, mmax_new, nofft) -> SHTConfig"""
function shtns_create_with_grid(cfg::SHTConfig, mmax_new::Integer, nofft::Integer)
    mmax2 = Int(mmax_new)
    mmax2 ≤ cfg.mmax || throw(ArgumentError("mmax_new must be ≤ cfg.mmax"))
    
  include_poles = cfg.grid_type === :regular_poles
  
    return create_config(cfg.lmax; mmax=mmax2, mres=cfg.mres, nlat=cfg.nlat, nlon=cfg.nlon,
                         norm=cfg.norm, cs_phase=cfg.cs_phase, real_norm=cfg.real_norm,
                         robert_form=cfg.robert_form, grid_type=cfg.grid_type,
                         include_poles=include_poles, precompute_plm=cfg.use_plm_tables)
end

"""shtns_use_threads(num_threads) -> Int"""
function shtns_use_threads(num_threads::Integer)
    # FFTW threading can be set externally; just return clamped value
    return max(1, Int(num_threads))
end

"""shtns_reset()"""
shtns_reset() = nothing

"""shtns_destroy(cfg)"""
shtns_destroy(::SHTConfig) = nothing

"""shtns_unset_grid(cfg)"""
shtns_unset_grid(::SHTConfig) = nothing

"""shtns_robert_form(cfg, robert)"""
function shtns_robert_form(cfg::SHTConfig, robert::Integer)
    cfg.robert_form = robert != 0
    return nothing
end

"""sh00_1(cfg) -> Float64"""
sh00_1(::SHTConfig) = sqrt(4π)

"""sh10_ct(cfg) -> Float64"""
sh10_ct(::SHTConfig) = sqrt(4π/3)

"""sh11_st(cfg) -> Float64"""
sh11_st(::SHTConfig) = -sqrt(2π/3)  # coefficient for sinθ cosφ in complex basis (m=+1)

"""shlm_e1(cfg, l::Integer, m::Integer) -> Float64"""
function shlm_e1(cfg::SHTConfig, l::Integer, m::Integer)
    (0 ≤ m ≤ cfg.mmax && m ≤ l ≤ cfg.lmax) || return 0.0
    return 1.0
end

"""shtns_gauss_wts(cfg, wts::AbstractVector{<:Real}) -> Int"""
function shtns_gauss_wts(cfg::SHTConfig, wts)
    n = min(length(wts), cfg.nlat)
    @inbounds for i in 1:n
        wts[i] = cfg.w[i]
    end
    return n
end

# Print configuration
"""shtns_print_cfg(cfg::SHTConfig)"""
function shtns_print_cfg(cfg::SHTConfig)
    println("SHTConfig:")
    println("  lmax=$(cfg.lmax), mmax=$(cfg.mmax), mres=$(cfg.mres)")
    println("  nlat=$(cfg.nlat), nlon=$(cfg.nlon)")
    println("  nlm=$(cfg.nlm), nspat=$(cfg.nspat)")
    return nothing
end

# Legendre arrays (spherical-harmonic normalized)
"""
    legendre_sphPlm_array(cfg::SHTConfig, lmax::Integer, im::Integer, x::Real, yl::AbstractVector{<:Real}) -> Int

Fill `yl[k]` with normalized P_l^m(x) for l = m..lmax where m = im*mres.
Returns the number of values written.
"""
function legendre_sphPlm_array(cfg::SHTConfig, lmax::Integer, im::Integer, x::Real, yl)
    lmax = Int(lmax); im = Int(im); x = float(x)
    m = im * cfg.mres
    (0 ≤ m ≤ cfg.mmax) || return 0
    (m ≤ lmax ≤ cfg.lmax) || return 0
    P = Vector{Float64}(undef, cfg.lmax + 1)
    Plm_row!(P, x, cfg.lmax, m)
    n = min(length(yl), lmax - m + 1)
  
    @inbounds for (k, l) in enumerate(m:(m + n - 1))
        yl[k] = cfg.Nlm[l+1, m+1] * P[l+1]
    end
    return n
end

"""
    legendre_sphPlm_deriv_array(cfg::SHTConfig, lmax::Integer, im::Integer, x::Real, sint::Real,
                                yl::AbstractVector{<:Real}, dyl::AbstractVector{<:Real}) -> Int

Fill `yl` with normalized P_l^m(x) and `dyl` with ∂θ of the normalized functions: `dyl = -sinθ * d/dx (normalized P)`.
`x = cosθ`, `sint = sinθ`.
Returns the number of values written.
"""
function legendre_sphPlm_deriv_array(cfg::SHTConfig, lmax::Integer, im::Integer, x::Real, sint::Real, yl, dyl)
    lmax = Int(lmax); im = Int(im); x = float(x); sint = float(sint)
    m = im * cfg.mres
    (0 ≤ m ≤ cfg.mmax) || return 0
    (m ≤ lmax ≤ cfg.lmax) || return 0
    P = Vector{Float64}(undef, cfg.lmax + 1)
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)
    Plm_and_dPdx_row!(P, dPdx, x, cfg.lmax, m)
    n = min(length(yl), length(dyl), lmax - m + 1)
  
    @inbounds for (k, l) in enumerate(m:(m + n - 1))
        N = cfg.Nlm[l+1, m+1]
        yl[k]  = N * P[l+1]
        dyl[k] = -sint * N * dPdx[l+1]
    end
    return n
end

# Memory helpers (no-ops in Julia)
"""shtns_malloc(bytes::Integer) -> Vector{UInt8}"""
function shtns_malloc(bytes::Integer)
    return Vector{UInt8}(undef, Int(bytes))
end
"""shtns_free(::Any)""" shtns_free(::Any) = nothing

"""shtns_set_many(cfg::SHTConfig, howmany::Integer, spec_dist::Integer) -> Int"""
function shtns_set_many(::SHTConfig, howmany::Integer, ::Integer)
    return max(1, Int(howmany))
end

# Remaining unimplemented APIs
for fname in (
)
    @eval function ($fname)(args...)
        throw(ErrorException(string($(QuoteNode(Symbol(fname))), " not implemented in pure Julia core yet")))
    end
end

"""
    SH_to_spat_time(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Vr::AbstractVector{<:Real}) -> Float64

Timed scalar synthesis; returns elapsed seconds and writes result into `Vr`.
"""
function SH_to_spat_time(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Vr::AbstractVector{<:Real})
    alm_mat = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            alm_mat[l+1, m+1] = Qlm[lm]
        end
    end
    t = @elapsed f = synthesis(cfg, alm_mat; real_output=true)
    Vr .= vec(f)
    return t
end

"""
    spat_to_SH_time(cfg::SHTConfig, Vr::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex}) -> Float64

Timed scalar analysis; returns elapsed seconds and writes result into `Qlm`.
"""
function spat_to_SH_time(cfg::SHTConfig, Vr::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex})
    f = reshape(Vr, cfg.nlat, cfg.nlon)
    t = @elapsed alm = analysis(cfg, f)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            Qlm[lm] = alm[l+1, m+1]
        end
    end
    return t
end

"""shtns_profiling(cfg, on)"""
shtns_profiling(::SHTConfig, on::Integer) = nothing

"""shtns_profiling_read_time(cfg, t1::Ref, t2::Ref) -> Float64"""
function shtns_profiling_read_time(::SHTConfig, t1::Ref{Float64}, t2::Ref{Float64})
    t1[] = 0.0; t2[] = 0.0; return 0.0
end
