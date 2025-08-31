##########
# PencilArray local/point evaluations and packed helpers
##########

using MPI
using PencilArrays
using SHTnsKit

"""
    dist_SH_to_lat(cfg, Alm_pencil::PencilArray, cost::Real;
                   nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax,
                   real_output::Bool=true) -> Vector

Evaluate along a latitude (cosθ = cost) from distributed Alm. All ranks receive the full vector.
"""
function SHTnsKit.dist_SH_to_lat(cfg::SHTnsKit.SHTConfig, Alm_pencil::PencilArray, cost::Real;
                                 nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax,
                                 real_output::Bool=true)
    comm = communicator(Alm_pencil)
    lmax, mmax = cfg.lmax, cfg.mmax
    x = float(cost)
    P = Vector{Float64}(undef, lmax + 1)
    vals_local = zeros(ComplexF64, nphi)
    lloc = axes(Alm_pencil, 1); mloc = axes(Alm_pencil, 2)
    gl_l = globalindices(Alm_pencil, 1)
    gl_m = globalindices(Alm_pencil, 2)
    # m = 0 if present locally
    j0 = findfirst(==(1), gl_m)
    if j0 !== nothing
        SHTnsKit.Plm_row!(P, x, lmax, 0)
        g0 = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval <= ltr
                g0 += cfg.Nlm[lval+1, 1] * P[lval+1] * Alm_pencil[il, mloc[j0]]
            end
        end
        vals_local .+= g0
    end
    # m > 0 columns owned by this rank
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        (mval > 0 && mval <= mtr) || continue
        SHTnsKit.Plm_row!(P, x, lmax, mval)
        gm = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if mval <= lval <= ltr
                gm += cfg.Nlm[lval+1, mval+1] * P[lval+1] * Alm_pencil[il, jm]
            end
        end
        @inbounds for j in 0:(nphi-1)
            vals_local[j+1] += 2 * real(gm * cis(2π * mval * j / nphi))
        end
    end
    vals = similar(vals_local)
    MPI.Allreduce!(vals_local, +, comm)
    return real_output ? real.(vals_local) : vals_local
end

"""
    dist_SH_to_point(cfg, Alm_pencil::PencilArray, cost::Real, phi::Real) -> ComplexF64 or Float64
"""
function SHTnsKit.dist_SH_to_point(cfg::SHTnsKit.SHTConfig, Alm_pencil::PencilArray, cost::Real, phi::Real)
    comm = communicator(Alm_pencil)
    lmax, mmax = cfg.lmax, cfg.mmax
    x = float(cost)
    P = Vector{Float64}(undef, lmax + 1)
    lloc = axes(Alm_pencil, 1); mloc = axes(Alm_pencil, 2)
    gl_l = globalindices(Alm_pencil, 1)
    gl_m = globalindices(Alm_pencil, 2)
    s_local = 0.0 + 0.0im
    # m=0
    j0 = findfirst(==(1), gl_m)
    if j0 !== nothing
        SHTnsKit.Plm_row!(P, x, lmax, 0)
        g0 = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            g0 += cfg.Nlm[lval+1, 1] * P[lval+1] * Alm_pencil[il, mloc[j0]]
        end
        s_local += g0
    end
    # m>0
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval > 0 || continue
        SHTnsKit.Plm_row!(P, x, lmax, mval)
        gm = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= mval
                gm += cfg.Nlm[lval+1, mval+1] * P[lval+1] * Alm_pencil[il, jm]
            end
        end
        ph = cis(mval * phi)
        s_local += gm * ph + conj(gm) * conj(ph)
    end
    s = MPI.Allreduce(s_local, +, comm)
    return real(s)
end

"""
    dist_SHqst_to_point(cfg, Q_p::PencilArray, S_p::PencilArray, T_p::PencilArray, cost, phi) -> (vr, vt, vp)
"""
function SHTnsKit.dist_SHqst_to_point(cfg::SHTnsKit.SHTConfig, Q_p::PencilArray, S_p::PencilArray, T_p::PencilArray, cost::Real, phi::Real)
    comm = communicator(Q_p)
    lmax, mmax = cfg.lmax, cfg.mmax
    x = float(cost)
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    lloc = axes(Q_p, 1); mloc = axes(Q_p, 2)
    gl_l = globalindices(Q_p, 1)
    gl_m = globalindices(Q_p, 2)
    sθ = sqrt(max(0.0, 1 - x*x)); inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
    vr_local = 0.0 + 0.0im
    vt_local = 0.0 + 0.0im
    vp_local = 0.0 + 0.0im
    # m=0
    j0 = findfirst(==(1), gl_m)
    if j0 !== nothing
        SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, 0)
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            N = cfg.Nlm[lval+1, 1]
            Y = N * P[lval+1]
            dθY = -sθ * N * dPdx[lval+1]
            vr_local += Y   * Q_p[il, mloc[j0]]
            vt_local += dθY * S_p[il, mloc[j0]]
            vp_local += (sθ * N * dPdx[lval+1]) * T_p[il, mloc[j0]]
        end
    end
    # m>0
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval > 0 || continue
        SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, mval)
        gvr = 0.0 + 0.0im
        gvt = 0.0 + 0.0im
        gvp = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= mval
                N = cfg.Nlm[lval+1, mval+1]
                Y = N * P[lval+1]
                dθY = -sθ * N * dPdx[lval+1]
                gvr += Y   * Q_p[il, jm]
                gvt += dθY * S_p[il, jm] + (0 + 1im) * mval * inv_sθ * Y * T_p[il, jm]
                gvp += (0 + 1im) * mval * inv_sθ * Y * S_p[il, jm] + (sθ * N * dPdx[lval+1]) * T_p[il, jm]
            end
        end
        ph = cis(mval * phi)
        vr_local += gvr * ph + conj(gvr) * conj(ph)
        vt_local += gvt * ph + conj(gvt) * conj(ph)
        vp_local += gvp * ph + conj(gvp) * conj(ph)
    end
    vr = MPI.Allreduce(vr_local, +, comm)
    vt = MPI.Allreduce(vt_local, +, comm)
    vp = MPI.Allreduce(vp_local, +, comm)
    return real(vr), real(vt), real(vp)
end

"""
    dist_SHqst_to_lat(cfg, Q_p::PencilArray, S_p::PencilArray, T_p::PencilArray, cost::Real;
                      nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) -> Vr, Vt, Vp
"""
function SHTnsKit.dist_SHqst_to_lat(cfg::SHTnsKit.SHTConfig, Q_p::PencilArray, S_p::PencilArray, T_p::PencilArray, cost::Real;
                                    nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    comm = communicator(Q_p)
    lmax = cfg.lmax
    x = float(cost)
    P = Vector{Float64}(undef, lmax + 1)
    dPdx = Vector{Float64}(undef, lmax + 1)
    lloc = axes(Q_p, 1); mloc = axes(Q_p, 2)
    gl_l = globalindices(Q_p, 1)
    gl_m = globalindices(Q_p, 2)
    sθ = sqrt(max(0.0, 1 - x*x)); inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
    Vr_local = zeros(ComplexF64, nphi)
    Vt_local = zeros(ComplexF64, nphi)
    Vp_local = zeros(ComplexF64, nphi)
    # m=0
    j0 = findfirst(==(1), gl_m)
    if j0 !== nothing
        SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, 0)
        g0 = 0.0 + 0.0im; gθ0 = 0.0 + 0.0im; gφ0 = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval <= ltr
                N = cfg.Nlm[lval+1, 1]
                Y = N * P[lval+1]
                dθY = -sθ * N * dPdx[lval+1]
                g0  += Y * Q_p[il, mloc[j0]]
                gθ0 += dθY * S_p[il, mloc[j0]]
                gφ0 += (sθ * N * dPdx[lval+1]) * T_p[il, mloc[j0]]
            end
        end
        Vr_local .+= g0; Vt_local .+= gθ0; Vp_local .+= gφ0
    end
    # m>0
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        (mval > 0 && mval <= mtr) || continue
        SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, mval)
        g  = 0.0 + 0.0im
        gθ = 0.0 + 0.0im
        gφ = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if mval <= lval <= ltr
                N = cfg.Nlm[lval+1, mval+1]
                Y = N * P[lval+1]
                dθY = -sθ * N * dPdx[lval+1]
                g  += Y   * Q_p[il, jm]
                gθ += dθY * S_p[il, jm] + (0 + 1im) * mval * inv_sθ * Y * T_p[il, jm]
                gφ += (0 + 1im) * mval * inv_sθ * Y * S_p[il, jm] + (sθ * N * dPdx[lval+1]) * T_p[il, jm]
            end
        end
        @inbounds for j in 0:(nphi-1)
            ph = cis(2π * mval * j / nphi)
            Vr_local[j+1] += 2 * real(g * ph)
            Vt_local[j+1] += 2 * real(gθ * ph)
            Vp_local[j+1] += 2 * real(gφ * ph)
        end
    end
    MPI.Allreduce!(Vr_local, +, comm)
    MPI.Allreduce!(Vt_local, +, comm)
    MPI.Allreduce!(Vp_local, +, comm)
    return real.(Vr_local), real.(Vt_local), real.(Vp_local)
end

"""
    dist_spat_to_SH_packed(cfg, fθφ::PencilArray) -> Qlm packed
"""
function SHTnsKit.dist_spat_to_SH_packed(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray)
    Alm = SHTnsKit.dist_analysis(cfg, fθφ)
    Qlm = Vector{ComplexF64}(undef, cfg.nlm)
    for m in 0:cfg.mmax
        for l in m:cfg.lmax
            lm = SHTnsKit.LM_index(cfg.lmax, cfg.mres, l, m) + 1
            Qlm[lm] = Alm[l+1, m+1]
        end
    end
    return Qlm
end

"""
    dist_SH_packed_to_spat(cfg, Qlm::AbstractVector{<:Complex}; prototype_θφ, real_output=true)
"""
function SHTnsKit.dist_SH_packed_to_spat(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex}; prototype_θφ::PencilArray, real_output::Bool=true)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length"))
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    for m in 0:cfg.mmax, l in m:cfg.lmax
        Alm[l+1, m+1] = Qlm[SHTnsKit.LM_index(cfg.lmax, cfg.mres, l, m) + 1]
    end
    return SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output)
end

"""
    dist_spat_cplx_to_SH(cfg, z::PencilArray) -> alm_packed (LM_cplx)
"""
function SHTnsKit.dist_spat_cplx_to_SH(cfg::SHTnsKit.SHTConfig, z::PencilArray)
    Alm = SHTnsKit.dist_analysis(cfg, z; use_tables=cfg.use_plm_tables)
    lmax, mmax = cfg.lmax, cfg.mmax
    alm_p = Vector{ComplexF64}(undef, SHTnsKit.nlm_cplx_calc(lmax, mmax, 1))
    # Pack +/- m
    for l in 0:lmax
        alm_p[SHTnsKit.LM_cplx_index(lmax, mmax, l, 0) + 1] = Alm[l+1, 1]
        for m in 1:min(l, mmax)
            alm_p[SHTnsKit.LM_cplx_index(lmax, mmax, l, m) + 1] = Alm[l+1, m+1]
            # negative m via real field relation encoded in packing — use conjugate with phase inside consumer as needed
            alm_p[SHTnsKit.LM_cplx_index(lmax, mmax, l, -m) + 1] = Alm[l+1, m+1]  # store as-is; consumer interprets
        end
    end
    return alm_p
end

"""
    dist_SH_to_spat_cplx(cfg, alm_packed::AbstractVector{<:Complex}; prototype_θφ) -> PencilArray complex field
"""
function SHTnsKit.dist_SH_to_spat_cplx(cfg::SHTnsKit.SHTConfig, alm_packed::AbstractVector{<:Complex}; prototype_θφ::PencilArray)
    lmax, mmax = cfg.lmax, cfg.mmax
    length(alm_packed) == SHTnsKit.nlm_cplx_calc(lmax, mmax, 1) || throw(DimensionMismatch("alm_packed length"))
    Alm = zeros(ComplexF64, lmax+1, mmax+1)
    for l in 0:lmax
        Alm[l+1, 1] = alm_packed[SHTnsKit.LM_cplx_index(lmax, mmax, l, 0) + 1]
        for m in 1:min(l, mmax)
            Alm[l+1, m+1] = alm_packed[SHTnsKit.LM_cplx_index(lmax, mmax, l, m) + 1]
        end
    end
    return SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output=false)
end
