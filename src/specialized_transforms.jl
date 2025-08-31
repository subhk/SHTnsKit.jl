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
    spat_to_SH(cfg, Vr) -> Vector{ComplexF64}

Transform spatial radial field values at Gauss points to spherical harmonic coefficients.
Input `Vr` should contain values at Gauss latitudes for a specific longitude mode.
Returns packed coefficients Q_l for l = 0..lmax.
"""
function spat_to_SH(cfg::SHTConfig, Vr::AbstractVector{<:Real})
    nlat, lmax = cfg.nlat, cfg.lmax
    length(Vr) == nlat || throw(DimensionMismatch("Vr length must be nlat=$(nlat)"))
    
    Ql = Vector{ComplexF64}(undef, lmax + 1)
    fill!(Ql, 0.0 + 0.0im)
    
    P = Vector{Float64}(undef, lmax + 1)
    
    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, lmax, 0)  # m=0 case (axisymmetric)
        
        weighted_Vr = Vr[i] * cfg.wlat[i]
        for l in 0:lmax
            Ql[l+1] += weighted_Vr * P[l+1]
        end
    end
    
    return Ql .* cfg.cphi  # Apply longitude scaling
end

"""
    SH_to_spat(cfg, Qlm) -> Vector{Float64}

Transform spherical harmonic coefficients to spatial values at Gauss points.
Input `Qlm` should contain coefficients Q_l for l = 0..lmax.
Returns spatial values at Gauss latitudes.
"""
function SH_to_spat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex})
    nlat, lmax = cfg.nlat, cfg.lmax
    length(Qlm) == lmax + 1 || throw(DimensionMismatch("Qlm length must be lmax+1=$(lmax+1)"))
    
    Vr = Vector{Float64}(undef, nlat)
    P = Vector{Float64}(undef, lmax + 1)
    
    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, lmax, 0)  # m=0 case
        
        val = 0.0
        for l in 0:lmax
            val += real(Qlm[l+1] * P[l+1])  # Take real part for spatial field
        end
        Vr[i] = val
    end
    
    return Vr
end

"""
    spat_to_SH_l(cfg, Vr, ltr) -> Vector{ComplexF64}

Transform spatial field to spherical harmonic coefficients up to degree ltr.
This is a degree-limited version of spat_to_SH for efficiency.
"""
function spat_to_SH_l(cfg::SHTConfig, Vr::AbstractVector{<:Real}, ltr::Int)
    nlat = cfg.nlat
    length(Vr) == nlat || throw(DimensionMismatch("Vr length must be nlat=$(nlat)"))
    ltr <= cfg.lmax || throw(ArgumentError("ltr must be <= lmax=$(cfg.lmax)"))
    
    Ql = Vector{ComplexF64}(undef, ltr + 1)
    fill!(Ql, 0.0 + 0.0im)
    
    P = Vector{Float64}(undef, ltr + 1)
    
    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, ltr, 0)
        
        weighted_Vr = Vr[i] * cfg.wlat[i]
        for l in 0:ltr
            Ql[l+1] += weighted_Vr * P[l+1]
        end
    end
    
    return Ql .* cfg.cphi
end

"""
    SH_to_spat_l(cfg, Qlm, ltr) -> Vector{Float64}

Transform spherical harmonic coefficients to spatial field using degrees up to ltr.
This is a degree-limited version of SH_to_spat.
"""
function SH_to_spat_l(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, ltr::Int)
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
        for l in 0:ltr
            val += real(Qlm[l+1] * P[l+1])
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
    fill!(Ql, 0.0 + 0.0im)
    
    P = Vector{Float64}(undef, ltr + 1)
    
    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, ltr, im)
        
        weighted_Vr = Vr_m[i] * cfg.wlat[i]
        for l in im:ltr
            Ql[l-im+1] += weighted_Vr * P[l+1]
        end
    end
    
    return Ql .* cfg.cphi
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
    
    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, ltr, im)
        
        val = 0.0 + 0.0im
        for l in im:ltr
            val += Ql[l-im+1] * P[l+1]
        end
        Vr_m[i] = val
    end
    
    return Vr_m
end

"""
    SH_to_point(cfg, Qlm, cost, phi) -> ComplexF64

Evaluate spherical harmonic expansion at a single point (θ,φ).
`cost` = cos(θ), `phi` is the azimuthal angle.
`Qlm` should contain all coefficients in (l+1,m+1) indexing.
Returns the field value at the specified point.
"""
function SH_to_point(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    lmax = cfg.lmax
    length(Qlm) == (lmax+1)^2 || throw(DimensionMismatch("Qlm must have length (lmax+1)^2"))
    
    result = 0.0 + 0.0im
    P = Vector{Float64}(undef, lmax + 1)
    
    # Process each azimuthal mode
    idx = 1
    for l in 0:lmax, m in 0:l
        if m == 0
            # m=0 case: only real part, no φ dependence
            Plm_row!(P, cost, l, 0)
            result += Qlm[idx] * P[l+1]
        else
            # m>0 case: include e^(imφ) factor
            Plm_row!(P, cost, l, m)
            phase = cos(m * phi) + im * sin(m * phi)  # e^(imφ)
            result += Qlm[idx] * P[l+1] * phase
        end
        idx += 1
    end
    
    return result
end