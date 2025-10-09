"""
Local/partial evaluations along latitude circles and at points.
"""

"""
    SH_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real;
              nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) -> Vector{Float64}

Evaluate a real field along a latitude (fixed cosθ = cost) at `nphi` equispaced longitudes.
Uses orthonormal harmonics and packed real coefficients `Qlm` (LM order).
"""
function SH_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    if is_gpu_config(cfg)
        return gpu_SH_to_lat(cfg, Qlm, cost; nphi=nphi, ltr=ltr, mtr=mtr)
    end
    return SH_to_lat_cpu(cfg, Qlm, cost; nphi=nphi, ltr=ltr, mtr=mtr)
end

function SH_to_lat_cpu(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    (0 ≤ mtr ≤ cfg.mmax) || throw(ArgumentError("mtr must be within [0, mmax]"))
    x = float(cost)
    lmax = cfg.lmax
    P = Vector{Float64}(undef, lmax + 1)
    vals = Vector{Float64}(undef, nphi)
    fill!(vals, 0.0)

    # m=0 contribution
    Plm_row!(P, x, lmax, 0)
    g0 = 0.0 + 0.0im
    @inbounds for l in 0:ltr
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        a = Qlm[lm]
        if cfg.norm !== :orthonormal || cfg.cs_phase == false
            k = norm_scale_from_orthonormal(l, 0, cfg.norm)
            α = cs_phase_factor(0, true, cfg.cs_phase)
            a *= (k * α)
        end
        g0 += cfg.Nlm[l+1, 1] * P[l+1] * a
    end
    @inbounds for j in 0:(nphi-1)
        vals[j+1] = real(g0)
    end

    # m>0
    for m in 1:mtr
        (m % cfg.mres == 0) || continue
        Plm_row!(P, x, lmax, m)
        gm = 0.0 + 0.0im
        col = m + 1
        @inbounds for l in m:min(ltr, lmax)
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            a = Qlm[lm]
            if cfg.norm !== :orthonormal || cfg.cs_phase == false
                k = norm_scale_from_orthonormal(l, m, cfg.norm)
                α = cs_phase_factor(m, true, cfg.cs_phase)
                a *= (k * α)
            end
            gm += cfg.Nlm[l+1, col] * P[l+1] * a
        end
        for j in 0:(nphi-1)
            vals[j+1] += 2 * real(gm * cis(2π * m * j / nphi))
        end
    end
    return vals
end

"""
    SH_to_lat_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax) -> Vector{ComplexF64}

Evaluate a complex field along a latitude using packed LM_cplx coefficients.
"""
function SH_to_lat_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax)
    if is_gpu_config(cfg)
        return gpu_SH_to_lat_cplx(cfg, alm_packed, cost; nphi=nphi, ltr=ltr)
    end
    return SH_to_lat_cplx_cpu(cfg, alm_packed, cost; nphi=nphi, ltr=ltr)
end

function SH_to_lat_cplx_cpu(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax)
    lmax, mmax = cfg.lmax, cfg.mmax
    length(alm_packed) == nlm_cplx_calc(lmax, mmax, 1) || throw(DimensionMismatch("alm_packed length"))
    x = float(cost)
    P = Vector{Float64}(undef, lmax + 1)
    vals = Vector{ComplexF64}(undef, nphi)
    fill!(vals, 0.0 + 0.0im)
    # m=0
    Plm_row!(P, x, lmax, 0)
    g0 = 0.0 + 0.0im
    @inbounds for l in 0:min(ltr, lmax)
        idx = LM_cplx_index(lmax, mmax, l, 0) + 1
        a = alm_packed[idx]
        if cfg.norm !== :orthonormal || cfg.cs_phase == false
            k = norm_scale_from_orthonormal(l, 0, cfg.norm)
            α = cs_phase_factor(0, true, cfg.cs_phase)
            a *= (k * α)
        end
        g0 += cfg.Nlm[l+1, 1] * P[l+1] * a
    end
    @inbounds for j in 1:nphi
        vals[j] += g0
    end
    # m ≠ 0
    for m in 1:mmax
        Plm_row!(P, x, lmax, m)
        col = m + 1
        gm = 0.0 + 0.0im; gn = 0.0 + 0.0im
        @inbounds for l in m:min(ltr, lmax)
            Ylm = cfg.Nlm[l+1, col] * P[l+1]
            # positive m
            ap = alm_packed[LM_cplx_index(lmax, mmax, l, m) + 1]
            # negative m
            an = alm_packed[LM_cplx_index(lmax, mmax, l, -m) + 1]
            if cfg.norm !== :orthonormal || cfg.cs_phase == false
                k = norm_scale_from_orthonormal(l, m, cfg.norm)
                αp = cs_phase_factor(m, true, cfg.cs_phase)
                αn = cs_phase_factor(-m, true, cfg.cs_phase)
                ap *= (k * αp); an *= (k * αn)
            end
            gm += Ylm * ap
            gn += Ylm * an
        end
        for j in 0:(nphi-1)
            phase = cis(2π * m * j / nphi)
            vals[j+1] += gm * phase + gn * conj(phase)
        end
    end
    return vals
end

"""
    SHqst_to_point(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real, phi::Real)
        -> vr::Float64, vt::Float64, vp::Float64

Evaluate 3D field at a single point using packed real spectra.
"""
function SHqst_to_point(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    if is_gpu_config(cfg)
        return gpu_SHqst_to_point(cfg, Qlm, Slm, Tlm, cost, phi)
    end
    return SHqst_to_point_cpu(cfg, Qlm, Slm, Tlm, cost, phi)
end

function SHqst_to_point_cpu(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length"))
    length(Slm) == cfg.nlm || throw(DimensionMismatch("Slm length"))
    length(Tlm) == cfg.nlm || throw(DimensionMismatch("Tlm length"))
    x = float(cost)
    lmax = cfg.lmax; mmax = cfg.mmax
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    sθ = sqrt(max(0.0, 1 - x*x))
    inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
    CT = promote_type(eltype(Qlm), eltype(Slm), eltype(Tlm))
    vr = zero(CT)
    vt = zero(CT)
    vp = zero(CT)
    # m=0
    Plm_and_dPdx_row!(P, dPdx, x, lmax, 0)
    for l in 0:lmax
        N = cfg.Nlm[l+1, 1]
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        aQ = Qlm[lm]; aS = Slm[lm]; aT = Tlm[lm]
        if cfg.norm !== :orthonormal || cfg.cs_phase == false
            k = norm_scale_from_orthonormal(l, 0, cfg.norm)
            α = cs_phase_factor(0, true, cfg.cs_phase)
            s = k * α
            aQ *= s; aS *= s; aT *= s
        end
        Y = N * P[l+1]
        dθY = -sθ * N * dPdx[l+1]
        vr += Y   * aQ
        vt += dθY * aS
        vp += (sθ * N * dPdx[l+1]) * aT
    end
    # m>0
    for m in 1:mmax
        (m % cfg.mres == 0) || continue
        Plm_and_dPdx_row!(P, dPdx, x, lmax, m)
        gvr = zero(CT)
        gvt = zero(CT)
        gvp = zero(CT)
        col = m + 1
        for l in m:lmax
            N = cfg.Nlm[l+1, col]
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            aQ = Qlm[lm]; aS = Slm[lm]; aT = Tlm[lm]
            if cfg.norm !== :orthonormal || cfg.cs_phase == false
                k = norm_scale_from_orthonormal(l, m, cfg.norm)
                α = cs_phase_factor(m, true, cfg.cs_phase)
                s = k * α
                aQ *= s; aS *= s; aT *= s
            end
            Y = N * P[l+1]
            dθY = -sθ * N * dPdx[l+1]
            gvr += Y   * aQ
            gvt += dθY * aS + (0 + 1im) * m * inv_sθ * Y * aT
            gvp += (0 + 1im) * m * inv_sθ * Y * aS + (sθ * N * dPdx[l+1]) * aT
        end
        ph = cis(m * phi)
        vr += 2 * real(gvr * ph)
        vt += 2 * real(gvt * ph)
        vp += 2 * real(gvp * ph)
    end
    return real(vr), real(vt), real(vp)
end

"""
    SH_to_grad_point(cfg::SHTConfig, DrSlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, cost::Real, phi::Real)
        -> vr::Float64, vt::Float64, vp::Float64

Evaluate gradient of a scalar field at a point. Vr is returned as 0.0.
`DrSlm` is ignored for this pure-Julia core.
"""
function SH_to_grad_point(cfg::SHTConfig, ::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    if is_gpu_config(cfg)
        return gpu_SH_to_grad_point(cfg, Slm, cost, phi)
    end
    return SH_to_grad_point_cpu(cfg, Slm, cost, phi)
end

function SH_to_grad_point_cpu(cfg::SHTConfig, Slm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    zeroQ = zeros(ComplexF64, cfg.nlm)
    zeroT = zeros(ComplexF64, cfg.nlm)
    return SHqst_to_point_cpu(cfg, zeroQ, Slm, zeroT, cost, phi)
end

"""
    SHqst_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real;
                 nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) -> Vr, Vt, Vp

Evaluate 3D field along latitude (cosθ = cost) at `nphi` longitudes from packed real spectra.
Inputs `Qlm, Slm, Tlm` are all packed (LM order) vectors for each component.
"""
function SHqst_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real;
                      nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    if is_gpu_config(cfg)
        return gpu_SHqst_to_lat(cfg, Qlm, Slm, Tlm, cost; nphi=nphi, ltr=ltr, mtr=mtr)
    end
    return SHqst_to_lat_cpu(cfg, Qlm, Slm, Tlm, cost; nphi=nphi, ltr=ltr, mtr=mtr)
end

function SHqst_to_lat_cpu(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real;
                          nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length"))
    length(Slm) == cfg.nlm || throw(DimensionMismatch("Slm length"))
    length(Tlm) == cfg.nlm || throw(DimensionMismatch("Tlm length"))
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    (0 ≤ mtr ≤ cfg.mmax) || throw(ArgumentError("mtr must be within [0, mmax]"))
    x = float(cost)
    lmax = cfg.lmax
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    Vr = Vector{Float64}(undef, nphi)
    Vt = Vector{Float64}(undef, nphi)
    Vp = Vector{Float64}(undef, nphi)
    fill!(Vr, 0.0); fill!(Vt, 0.0); fill!(Vp, 0.0)

    # m=0
    Plm_and_dPdx_row!(P, dPdx, x, lmax, 0)
    g0 = 0.0 + 0.0im
    gθ0 = 0.0 + 0.0im
    gφ0 = 0.0 + 0.0im
    sθ = sqrt(max(0.0, 1 - x*x))
    @inbounds for l in 0:ltr
        N = cfg.Nlm[l+1, 1]
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        aQ = Qlm[lm]; aS = Slm[lm]; aT = Tlm[lm]
        if cfg.norm !== :orthonormal || cfg.cs_phase == false
            k = norm_scale_from_orthonormal(l, 0, cfg.norm)
            α = cs_phase_factor(0, true, cfg.cs_phase)
            s = k * α
            aQ *= s; aS *= s; aT *= s
        end
        Y = N * P[l+1]
        dθY = -sθ * N * dPdx[l+1]
        g0  += Y * aQ
        gθ0 += dθY * aS
        gφ0 += (sθ * N * dPdx[l+1]) * aT
    end
    @inbounds for j in 1:nphi
        Vr[j] += real(g0); Vt[j] += real(gθ0); Vp[j] += real(gφ0)
    end

    # m>0
    for m in 1:mtr
        (m % cfg.mres == 0) || continue
        Plm_and_dPdx_row!(P, dPdx, x, lmax, m)
        g  = 0.0 + 0.0im
        gθ = 0.0 + 0.0im
        gφ = 0.0 + 0.0im
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
        col = m + 1
        @inbounds for l in m:min(ltr, lmax)
            N = cfg.Nlm[l+1, col]
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            aQ = Qlm[lm]; aS = Slm[lm]; aT = Tlm[lm]
            if cfg.norm !== :orthonormal || cfg.cs_phase == false
                k = norm_scale_from_orthonormal(l, m, cfg.norm)
                α = cs_phase_factor(m, true, cfg.cs_phase)
                s = k * α
                aQ *= s; aS *= s; aT *= s
            end
            Y = N * P[l+1]
            dθY = -sθ * N * dPdx[l+1]
            g  += Y   * aQ
            gθ += dθY * aS + (0 + 1im) * m * inv_sθ * Y * aT
            gφ += (0 + 1im) * m * inv_sθ * Y * aS + (sθ * N * dPdx[l+1]) * aT
        end
        for j in 0:(nphi-1)
            phase = cis(2π * m * j / nphi)
            Vr[j+1] += 2 * real(g * phase)
            Vt[j+1] += 2 * real(gθ * phase)
            Vp[j+1] += 2 * real(gφ * phase)
        end
    end
    return Vr, Vt, Vp
end
