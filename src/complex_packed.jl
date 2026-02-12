#=
================================================================================
complex_packed.jl - Complex Packed Layout (LM_cplx) Support
================================================================================

This file implements the packed coefficient storage for complex-valued fields
where both positive and negative m values are stored (no Hermitian symmetry).

WHEN IS THIS NEEDED?
--------------------
Real-valued fields have Hermitian symmetry: a_{l,-m} = (-1)^m conj(a_{l,m})
So we only need to store m ≥ 0 coefficients (the "real" packed layout).

Complex-valued fields have NO symmetry - we must store all m from -l to +l.
This doubles the storage but is necessary for:
- Complex scalar fields
- Intermediate calculations in rotations
- Some mathematical analyses

LAYOUT (LM_cplx)
----------------
Coefficients are packed in order:
    l=0: m=0                    (1 coefficient)
    l=1: m=-1, 0, +1            (3 coefficients)
    l=2: m=-2, -1, 0, +1, +2    (5 coefficients)
    ...

For l ≤ mmax: index = l(l+1) + m
For l > mmax: index = mmax(2l - mmax) + l + m

This matches the SHTns C library LM_cplx macro (for mres=1).

FUNCTIONS
---------
    LM_cplx_index(lmax, mmax, l, m)    : Get packed index for (l,m)
    LM_cplx(cfg, l, m)                 : Compatibility wrapper
    nlm_cplx_calc(lmax, mmax, mres)    : Count total complex coefficients

    synthesis_packed_cplx(cfg, alm_packed)   : Synthesis for complex field
    analysis_packed_cplx(cfg, z)            : Analysis for complex field
    synthesis_point_cplx(cfg, alm, cosθ, φ): Point evaluation for complex field

COMPARISON WITH REAL LAYOUT
---------------------------
For lmax=mmax=2:
    Real layout:    (0,0), (1,0), (2,0), (1,1), (2,1), (2,2)     = 6 coeffs
    Complex layout: (0,0), (1,-1),(1,0),(1,1), (2,-2),...,(2,2)  = 9 coeffs

================================================================================
=#

"""
Complex packed layout (LM_cplx) support and transforms in pure Julia.
Compatible with SHTns `LM_cplx` macro for `mres == 1`.
"""

"""
    LM_cplx_index(lmax::Int, mmax::Int, l::Int, m::Int) -> Int

Packed complex index for coefficient `(l,m)` where `-min(l,mmax) ≤ m ≤ min(l,mmax)`.
Matches SHTns `LM_cplx` macro (assumes `mres == 1`).
"""
function LM_cplx_index(lmax::Int, mmax::Int, l::Int, m::Int)
    (0 ≤ l ≤ lmax) || throw(ArgumentError("l out of range"))
    mm = min(l, mmax)
    (-mm ≤ m ≤ mm) || throw(ArgumentError("m out of range for given l and mmax"))
    if l ≤ mmax
        return l*(l + 1) + m
    else
        return mmax*(2*l - mmax) + l + m
    end
end

"""
    LM_cplx(cfg::SHTConfig, l::Integer, m::Integer) -> Int

Compatibility helper that mirrors the C macro `LM_cplx` for `mres == 1`,
returning the zero-based packed complex index for coefficient `(l,m)`.
"""
function LM_cplx(cfg::SHTConfig, l::Integer, m::Integer)
    cfg.mres == 1 || throw(ArgumentError("LM_cplx is only defined for mres == 1"))
    return LM_cplx_index(cfg.lmax, cfg.mmax, Int(l), Int(m))
end

"""
    synthesis_packed_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex}) -> Matrix{ComplexF64}

Synthesize complex spatial field from packed complex coefficients (LM_cplx order).
Returns an `nlat × nlon` complex array.
"""
function synthesis_packed_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex})
    mres = cfg.mres
    mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    expected = nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    length(alm_packed) == expected || throw(DimensionMismatch("alm length $(length(alm_packed)) != expected $(expected)"))

    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm_packed)
    Fφ = Matrix{CT}(undef, nlat, nlon)
    fill!(Fφ, zero(CT))

    lmax, mmax = cfg.lmax, cfg.mmax
    P = Vector{Float64}(undef, lmax + 1)
    G = Vector{CT}(undef, nlat)
    # Scale continuous Fourier coefficients to DFT bins for ifft
    inv_scaleφ = phi_inv_scale(cfg)

    for m in -mmax:mmax
        # build G_m(θ) = sum_l Nlm P_l^{|m|} alm(l,m)
        am = abs(m)
        # skip if no degrees for given am
        if am > lmax; continue; end
        for i in 1:nlat
            Plm_row!(P, cfg.x[i], lmax, am)
            g = zero(CT)
            @inbounds for l in am:lmax
                idx = LM_cplx_index(lmax, mmax, l, m) + 1
                a = alm_packed[idx]
                if cfg.norm !== :orthonormal || cfg.cs_phase == false
                    k = norm_scale_from_orthonormal(l, am, cfg.norm)
                    α = cs_phase_factor(m, true, cfg.cs_phase)
                    a *= (k * α)
                end
                g += (cfg.Nlm[l+1, am+1] * P[l+1]) * a
            end
            G[i] = g
        end
        # place Fourier bin for mode m
        j = m ≥ 0 ? (m + 1) : (nlon + m + 1)  # because (j-1) ≡ m mod nlon
        @inbounds for i in 1:nlat
            Fφ[i, j] = inv_scaleφ * G[i]
        end
    end

    z = ifft_phi(Fφ)
    return z
end

"""
    analysis_packed_cplx(cfg::SHTConfig, z::AbstractMatrix{<:Complex}) -> Vector{ComplexF64}

Analyze complex spatial field into packed complex coefficients (LM_cplx order).
Input `z` must be `nlat × nlon` complex.
"""
function analysis_packed_cplx(cfg::SHTConfig, z::AbstractMatrix{<:Complex})
    size(z,1) == cfg.nlat || throw(DimensionMismatch("z first dim must be nlat"))
    size(z,2) == cfg.nlon || throw(DimensionMismatch("z second dim must be nlon"))
    mres = cfg.mres
    mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(z)
    alm = Vector{CT}(undef, nlm_cplx_calc(lmax, mmax, 1))
    fill!(alm, zero(CT))

    # FFT along φ
    Fφ = fft_phi(complex.(z))
    P = Vector{Float64}(undef, lmax + 1)
    scaleφ = cfg.cphi

    for m in -mmax:mmax
        am = abs(m)
        col = m ≥ 0 ? (m + 1) : (cfg.nlon + m + 1)
        for i in 1:cfg.nlat
            Plm_row!(P, cfg.x[i], lmax, am)
            Fi = Fφ[i, col]
            wi = cfg.w[i]
            @inbounds for l in am:lmax
                idx = LM_cplx_index(lmax, mmax, l, m) + 1
                a = (wi * P[l+1]) * Fi * cfg.Nlm[l+1, am+1] * scaleφ
                # Convert from internal to cfg normalization if needed when storing
                if cfg.norm !== :orthonormal || cfg.cs_phase == false
                    k = norm_scale_from_orthonormal(l, am, cfg.norm)
                    α = cs_phase_factor(m, true, cfg.cs_phase)
                    a /= (k * α)
                end
                alm[idx] += a
            end
        end
    end
    return alm
end

"""
    synthesis_point_cplx(cfg::SHTConfig, alm::AbstractVector{<:Complex}, cost::Real, phi::Real) -> ComplexF64

Evaluate a complex field represented by packed `alm` at a single point.
"""
function synthesis_point_cplx(cfg::SHTConfig, alm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    expected = nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    length(alm) == expected || throw(DimensionMismatch("alm length mismatch"))
    x = float(cost)
    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(alm)
    P = Vector{Float64}(undef, lmax + 1)
    acc = zero(CT)
    # m from -mmax..mmax
    for m in -mmax:mmax
        am = abs(m)
        Plm_row!(P, x, lmax, am)
        gm = zero(CT)
        @inbounds for l in am:lmax
            idx = LM_cplx_index(lmax, mmax, l, m) + 1
            a = alm[idx]
            if cfg.norm !== :orthonormal || cfg.cs_phase == false
                k = norm_scale_from_orthonormal(l, am, cfg.norm)
                α = cs_phase_factor(m, true, cfg.cs_phase)
                a *= (k * α)
            end
            gm += cfg.Nlm[l+1, am+1] * P[l+1] * a
        end
        acc += gm * cis(m * phi)
    end
    return acc
end
