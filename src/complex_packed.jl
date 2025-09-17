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
    SH_to_spat_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex}) -> Matrix{ComplexF64}

Synthesize complex spatial field from packed complex coefficients (LM_cplx order).
Returns an `nlat × nlon` complex array.
"""
function SH_to_spat_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex})
    mres = cfg.mres
    mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    expected = nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    length(alm_packed) == expected || throw(DimensionMismatch("alm length $(length(alm_packed)) != expected $(expected)"))

    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm_packed)
    Fφ = Matrix{CT}(undef, nlat, nlon)
    fill!(Fφ, 0.0 + 0.0im)

    lmax, mmax = cfg.lmax, cfg.mmax
    P = Vector{Float64}(undef, lmax + 1)
    G = Vector{CT}(undef, nlat)
    # Scale continuous Fourier coefficients to DFT bins for ifft (factor nlon)
    inv_scaleφ = phi_inv_scale(nlon)

    for m in -mmax:mmax
        # build G_m(θ) = sum_l Nlm P_l^{|m|} alm(l,m)
        am = abs(m)
        # skip if no degrees for given am
        if am > lmax; continue; end
        for i in 1:nlat
            Plm_row!(P, cfg.x[i], lmax, am)
            g = 0.0 + 0.0im
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
    spat_cplx_to_SH(cfg::SHTConfig, z::AbstractMatrix{<:Complex}) -> Vector{ComplexF64}

Analyze complex spatial field into packed complex coefficients (LM_cplx order).
Input `z` must be `nlat × nlon` complex.
"""
function spat_cplx_to_SH(cfg::SHTConfig, z::AbstractMatrix{<:Complex})
    size(z,1) == cfg.nlat || throw(DimensionMismatch("z first dim must be nlat"))
    size(z,2) == cfg.nlon || throw(DimensionMismatch("z second dim must be nlon"))
    mres = cfg.mres
    mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(z)
    alm = Vector{CT}(undef, nlm_cplx_calc(lmax, mmax, 1))
    fill!(alm, 0.0 + 0.0im)

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
    SH_to_point_cplx(cfg::SHTConfig, alm::AbstractVector{<:Complex}, cost::Real, phi::Real) -> ComplexF64

Evaluate a complex field represented by packed `alm` at a single point.
"""
function SH_to_point_cplx(cfg::SHTConfig, alm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    expected = nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    length(alm) == expected || throw(DimensionMismatch("alm length mismatch"))
    x = float(cost)
    lmax, mmax = cfg.lmax, cfg.mmax
    P = Vector{Float64}(undef, lmax + 1)
    acc = 0.0 + 0.0im
    # m from -mmax..mmax
    for m in -mmax:mmax
        am = abs(m)
        Plm_row!(P, x, lmax, am)
        gm = 0.0 + 0.0im
        @inbounds for l in am:lmax
            idx = LM_cplx_index(lmax, mmax, l, m) + 1
            gm += cfg.Nlm[l+1, am+1] * P[l+1] * alm[idx]
        end
        acc += gm * cis(m * phi)
    end
    return acc
end
