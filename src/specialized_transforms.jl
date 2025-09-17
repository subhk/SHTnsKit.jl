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
        for l in 0:lmax
            val += real(Qlm[l+1] * P[l+1])  # Take real part for spatial field
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
