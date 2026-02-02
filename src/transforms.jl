#=
================================================================================
transforms.jl - Specialized Spherical Harmonic Transform Functions
================================================================================

This file provides specialized transform variants for cases where full 2D
transforms are unnecessary or inefficient.

WHEN TO USE SPECIALIZED TRANSFORMS
----------------------------------
- Axisymmetric fields (m=0 only): Use *_axisym functions
- Single azimuthal mode: Use *_ml functions
- Point evaluation: Use SH_to_point
- Degree truncation: Use *_l variants
- Packed coefficient layout: Use SH_to_spat / spat_to_SH

FUNCTION CATEGORIES
-------------------
1. Packed Layout Transforms:
   spat_to_SH(cfg, Vr)      : Grid → packed coefficients (1D vector)
   SH_to_spat(cfg, Qlm)     : Packed coefficients → flattened grid

2. Axisymmetric (m=0) Transforms:
   spat_to_SH_axisym(cfg, Vr)     : Latitude values → l-coefficients
   SH_to_spat_axisym(cfg, Qlm)    : l-coefficients → latitude values
   *_l_axisym variants            : Degree-limited versions

3. Mode-Limited (single m) Transforms:
   spat_to_SH_ml(cfg, m, Vr_m, ltr)  : Single-mode analysis
   SH_to_spat_ml(cfg, m, Ql, ltr)    : Single-mode synthesis

4. Degree-Limited Transforms:
   spat_to_SH_l(cfg, Vr, ltr)  : Analysis with l ≤ ltr
   SH_to_spat_l(cfg, Qlm, ltr) : Synthesis with l ≤ ltr

5. Point Evaluation:
   SH_to_point(cfg, Qlm, cosθ, φ)  : Evaluate at single point

PERFORMANCE BENEFITS
--------------------
- Axisymmetric: O(lmax × nlat) vs O(lmax × nlat × nlon) for full transform
- Mode-limited: Process only needed azimuthal modes
- Point evaluation: No grid storage needed
- Degree truncation: Skip high-degree computations

USAGE EXAMPLES
--------------
```julia
cfg = create_gauss_config(32, 64)

# Axisymmetric field (zonal average)
f_zonal = mean(f, dims=2)[:, 1]  # Average over longitude
Ql = spat_to_SH_axisym(cfg, f_zonal)

# Point evaluation (avoid full synthesis for single point)
val = SH_to_point(cfg, Qlm, cos(θ), φ)

# Degree-limited synthesis (e.g., for low-pass filtering)
f_smooth = SH_to_spat_l(cfg, Qlm, 10)  # Only l ≤ 10
```

================================================================================
=#

"""
Specialized Spherical Harmonic Transforms

This module provides specialized transform functions for specific use cases:
- Vector transforms for individual l,m modes
- Point evaluation without full grid computation
- Degree-limited and mode-limited transforms

These functions are optimized for cases where only partial spectral information
is needed, avoiding the computational overhead of full 2D transforms.
"""

"""
    spat_to_SH_axisym(cfg, Vr) -> Vector{ComplexF64}

Axisymmetric (m=0) transform from Gauss latitudes to degree-only coefficients.
Input `Vr` should contain values at Gauss latitudes for a specific longitude mode.
Returns coefficients Q_l for l = 0..lmax.
"""
function spat_to_SH_axisym(cfg::SHTConfig, Vr::AbstractVector{<:Real})
    nlat, lmax = cfg.nlat, cfg.lmax
    length(Vr) == nlat || throw(DimensionMismatch("Vr length must be nlat=$(nlat)"))
    
    Ql = Vector{ComplexF64}(undef, lmax + 1)
    fill!(Ql, zero(ComplexF64))
    
    P = Vector{Float64}(undef, lmax + 1)
    
    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, lmax, 0)  # m=0 case (axisymmetric)

        weighted_Vr = Vr[i] * cfg.wlat[i]
        @inbounds for l in 0:lmax
            Ql[l+1] += weighted_Vr * cfg.Nlm[l+1, 1] * P[l+1]
        end
    end

    return Ql  # No phi scaling needed for single-mode transform (proper inverse of SH_to_spat_axisym)
end

"""
    spat_to_SH(cfg, Vr_flat::AbstractVector{<:Real}) -> Vector{ComplexF64}

Packed scalar analysis from flattened grid values (length nlat*nlon) to Qlm (LM order).
"""
function spat_to_SH(cfg::SHTConfig, Vr::AbstractVector{<:Real})
    length(Vr) == cfg.nspat || throw(DimensionMismatch("Vr must have length $(cfg.nspat)"))
    f = reshape(Vr, cfg.nlat, cfg.nlon)
    alm_mat = analysis(cfg, f)
    Qlm = Vector{eltype(alm_mat)}(undef, cfg.nlm)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            Qlm[lm] = alm_mat[l+1, m+1]
        end
    end
    return Qlm
end

"""
    SH_to_spat(cfg, Qlm::AbstractVector{<:Complex}) -> Vector{Float64}

Packed scalar synthesis from Qlm (LM order) to flattened real grid (length nlat*nlon).
"""
function SH_to_spat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex})
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    alm_mat = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            alm_mat[l+1, m+1] = Qlm[lm]
        end
    end
    f = synthesis(cfg, alm_mat; real_output=true)
    return vec(f)
end

"""
    spat_to_SH_l(cfg::SHTConfig, Vr::AbstractVector{<:Real}, ltr::Integer) -> Vector{ComplexF64}

Scalar analysis truncated to degrees `l ≤ ltr`. The returned packed coefficient
vector has length `cfg.nlm`; coefficients with `l > ltr` are set to zero.
"""
function spat_to_SH_l(cfg::SHTConfig, Vr::AbstractVector{<:Real}, ltr::Integer)
    length(Vr) == cfg.nspat || throw(DimensionMismatch("Vr must have length $(cfg.nspat)"))
    lcap = min(Int(ltr), cfg.lmax)
    lcap ≥ 0 || throw(ArgumentError("ltr must be ≥ 0"))
    f = reshape(Vr, cfg.nlat, cfg.nlon)
    alm_mat = analysis(cfg, f)
    Qlm = zeros(eltype(alm_mat), cfg.nlm)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        m > lcap && continue
        for l in m:lcap
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            Qlm[lm] = alm_mat[l+1, m+1]
        end
    end
    return Qlm
end

"""
    SH_to_spat_l(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, ltr::Integer) -> Vector{Float64}

Scalar synthesis truncated to degrees `l ≤ ltr`. Contributions from higher
degrees are ignored.
"""
function SH_to_spat_l(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, ltr::Integer)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    lcap = min(Int(ltr), cfg.lmax)
    lcap ≥ 0 || throw(ArgumentError("ltr must be ≥ 0"))
    alm_mat = zeros(eltype(Qlm), cfg.lmax+1, cfg.mmax+1)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        m > lcap && continue
        for l in m:lcap
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            alm_mat[l+1, m+1] = Qlm[lm]
        end
    end
    f = synthesis(cfg, alm_mat; real_output=true)
    return vec(f)
end

"""
    SH_to_spat_axisym(cfg, Qlm) -> Vector{Float64}

Axisymmetric synthesis from degree-only coefficients to Gauss latitudes.
Input `Qlm` should contain coefficients Q_l for l = 0..lmax.
Returns spatial values at Gauss latitudes.
"""
function SH_to_spat_axisym(cfg::SHTConfig, Qlm::AbstractVector{<:Complex})
    nlat, lmax = cfg.nlat, cfg.lmax
    length(Qlm) == lmax + 1 || throw(DimensionMismatch("Qlm length must be lmax+1=$(lmax+1)"))
    
    Vr = Vector{Float64}(undef, nlat)
    P = Vector{Float64}(undef, lmax + 1)
    
    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, lmax, 0)  # m=0 case

        val = 0.0
        @inbounds for l in 0:lmax
            val += real(Qlm[l+1] * cfg.Nlm[l+1, 1] * P[l+1])  # Take real part for spatial field
        end
        Vr[i] = val
    end

    return Vr
end

"""
    spat_to_SH_l_axisym(cfg, Vr, ltr) -> Vector{ComplexF64}

Axisymmetric degree-limited transform up to degree ltr.
"""
function spat_to_SH_l_axisym(cfg::SHTConfig, Vr::AbstractVector{<:Real}, ltr::Int)
    nlat = cfg.nlat
    length(Vr) == nlat || throw(DimensionMismatch("Vr length must be nlat=$(nlat)"))
    ltr <= cfg.lmax || throw(ArgumentError("ltr must be <= lmax=$(cfg.lmax)"))
    
    Ql = Vector{ComplexF64}(undef, ltr + 1)
    fill!(Ql, zero(ComplexF64))
    
    P = Vector{Float64}(undef, ltr + 1)
    
    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, ltr, 0)

        weighted_Vr = Vr[i] * cfg.wlat[i]
        @inbounds for l in 0:ltr
            Ql[l+1] += weighted_Vr * cfg.Nlm[l+1, 1] * P[l+1]
        end
    end

    return Ql  # No phi scaling needed for single-mode transform (proper inverse of SH_to_spat_l_axisym)
end

"""
    SH_to_spat_l_axisym(cfg, Qlm, ltr) -> Vector{Float64}

Axisymmetric degree-limited synthesis using degrees up to ltr.
"""
function SH_to_spat_l_axisym(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    ltr_qlm = length(Qlm) - 1  # Convert length to max degree
    ltr <= cfg.lmax || throw(ArgumentError("ltr must be <= lmax=$(cfg.lmax)"))
    ltr <= ltr_qlm || throw(ArgumentError("ltr must be <= length(Qlm)-1=$(ltr_qlm)"))
    
    Vr = Vector{Float64}(undef, nlat)
    P = Vector{Float64}(undef, ltr + 1)
    
    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, ltr, 0)

        val = 0.0
        @inbounds for l in 0:ltr
            val += real(Qlm[l+1] * cfg.Nlm[l+1, 1] * P[l+1])
        end
        Vr[i] = val
    end

    return Vr
end

"""
    spat_to_SH_ml(cfg, im, Vr_m, ltr) -> Vector{ComplexF64}

Transform spatial field for specific azimuthal mode m to spherical harmonic coefficients.
`im` is the m-index (0-based), `Vr_m` contains complex spatial values for that mode.
Returns coefficients Q_l for degrees l = m..ltr.
"""
function spat_to_SH_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    length(Vr_m) == nlat || throw(DimensionMismatch("Vr_m length must be nlat=$(nlat)"))
    im >= 0 || throw(ArgumentError("im must be >= 0"))
    im <= cfg.mmax || throw(ArgumentError("im must be <= mmax=$(cfg.mmax)"))
    ltr <= cfg.lmax || throw(ArgumentError("ltr must be <= lmax=$(cfg.lmax)"))
    ltr >= im || throw(ArgumentError("ltr must be >= im=$(im)"))

    num_l = ltr - im + 1
    Ql = Vector{ComplexF64}(undef, num_l)
    fill!(Ql, zero(ComplexF64))

    P = Vector{Float64}(undef, ltr + 1)
    scaleφ = cfg.cphi  # Match full transform normalization

    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, ltr, im)

        weighted_Vr = Vr_m[i] * cfg.wlat[i]
        @inbounds for l in im:ltr
            Ql[l-im+1] += weighted_Vr * cfg.Nlm[l+1, im+1] * P[l+1]
        end
    end

    # Apply phi scaling to match full transform normalization
    Ql .*= scaleφ
    return Ql
end

"""
    SH_to_spat_ml(cfg, im, Ql, ltr) -> Vector{ComplexF64}

Transform spherical harmonic coefficients for specific mode m to spatial field.
`im` is the m-index, `Ql` contains coefficients for degrees l = im..ltr.
Returns complex spatial values for that azimuthal mode.
"""
function SH_to_spat_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    im >= 0 || throw(ArgumentError("im must be >= 0"))
    im <= cfg.mmax || throw(ArgumentError("im must be <= mmax=$(cfg.mmax)"))
    ltr <= cfg.lmax || throw(ArgumentError("ltr must be <= lmax=$(cfg.lmax)"))
    ltr >= im || throw(ArgumentError("ltr must be >= im=$(im)"))

    expected_len = ltr - im + 1
    length(Ql) == expected_len || throw(DimensionMismatch("Ql length must be $(expected_len)"))

    Vr_m = Vector{ComplexF64}(undef, nlat)
    P = Vector{Float64}(undef, ltr + 1)
    inv_scaleφ = phi_inv_scale(cfg)  # Match full transform normalization

    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, ltr, im)

        val = zero(ComplexF64)
        @inbounds for l in im:ltr
            val += Ql[l-im+1] * cfg.Nlm[l+1, im+1] * P[l+1]
        end
        Vr_m[i] = val * inv_scaleφ
    end

    return Vr_m
end

"""
    SH_to_point(cfg, Qlm, cost, phi) -> ComplexF64

Evaluate spherical harmonic expansion at a single point (θ,φ).
`cost` = cos(θ), `phi` is the azimuthal angle.
`Qlm` should be a matrix of size (lmax+1, mmax+1) with standard indexing.
Returns the field value at the specified point.
"""
function SH_to_point(cfg::SHTConfig, Qlm::AbstractMatrix{<:Complex}, cost::Real, phi::Real)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Qlm, 1) == lmax + 1 || throw(DimensionMismatch("Qlm first dim must be lmax+1"))
    size(Qlm, 2) == mmax + 1 || throw(DimensionMismatch("Qlm second dim must be mmax+1"))

    result = zero(ComplexF64)
    P = Vector{Float64}(undef, lmax + 1)

    # Process each azimuthal mode
    for m in 0:mmax
        Plm_row!(P, cost, lmax, m)
        phase = m == 0 ? one(ComplexF64) : cos(m * phi) + 1.0im * sin(m * phi)  # e^(imφ)

        @inbounds for l in m:lmax
            result += Qlm[l+1, m+1] * cfg.Nlm[l+1, m+1] * P[l+1] * phase
        end
    end

    return result
end
