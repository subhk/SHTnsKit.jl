#=
================================================================================
kernels.jl - Legendre Accumulation Kernels for Spherical Harmonic Transforms
================================================================================

Low-level kernel functions that perform per-latitude Legendre polynomial
accumulation. These are the innermost loops of all SHT transforms, extracted
here as a single source of truth.

Each kernel handles one latitude point for one azimuthal mode m.
Table variants receive precomputed Legendre values; on-the-fly variants
compute them via Plm_row! or Plm_dPdtheta_over_sinth_row!.

All kernels are @inline to allow the compiler to fuse them into the
orchestrator's loop without function-call overhead.
================================================================================
=#

# ============================================================================
# SCALAR ANALYSIS KERNELS
# ============================================================================

"""Scalar analysis kernel using precomputed Legendre tables."""
@inline function _scalar_analysis_kernel!(alm, cfg, Fph, tbl, i, col, m, lmax, scale_phi)
    Fi = Fph[i, col]
    wi = cfg.w[i]
    @inbounds for l in m:lmax
        alm[l+1, col] += (wi * cfg.Nlm[l+1, col] * tbl[l+1, i] * scale_phi) * Fi
    end
end

"""Scalar analysis kernel computing Legendre polynomials on the fly."""
@inline function _scalar_analysis_kernel_otf!(alm, cfg, Fph, P, i, col, m, lmax, scale_phi)
    Plm_row!(P, cfg.x[i], lmax, m)
    Fi = Fph[i, col]
    wi = cfg.w[i]
    @inbounds for l in m:lmax
        alm[l+1, col] += (wi * cfg.Nlm[l+1, col] * P[l+1] * scale_phi) * Fi
    end
end

# ============================================================================
# SCALAR SYNTHESIS KERNELS
# ============================================================================

"""Scalar synthesis kernel using precomputed Legendre tables. Returns accumulated value."""
@inline function _scalar_synthesis_kernel(cfg, alm, tbl, i, col, m, lmax)
    acc = zero(eltype(alm))
    @inbounds for l in m:lmax
        acc += (cfg.Nlm[l+1, col] * tbl[l+1, i]) * alm[l+1, col]
    end
    return acc
end

"""Scalar synthesis kernel computing Legendre polynomials on the fly. Returns accumulated value."""
@inline function _scalar_synthesis_kernel_otf(cfg, alm, P, i, col, m, lmax)
    Plm_row!(P, cfg.x[i], lmax, m)
    acc = zero(eltype(alm))
    @inbounds for l in m:lmax
        acc += (cfg.Nlm[l+1, col] * P[l+1]) * alm[l+1, col]
    end
    return acc
end

# ============================================================================
# POLE HELPERS (moved from sphtor_transforms.jl)
# ============================================================================

"""Compute dP_l^m/dtheta * N at a pole (x = +/-1) using analytical limits."""
@inline function _dPdtheta_at_pole(l::Int, m::Int, x::Float64, N::Float64)
    if m == 0
        return 0.0
    elseif m == 1
        if x > 0
            return N * (-Float64(l * (l + 1)) / 2)
        else
            return N * Float64((-1)^(l+1)) * l * (l + 1) / 2
        end
    else
        return 0.0
    end
end

"""Compute P_l^m/sin(theta) * N at a pole (x = +/-1) using analytical limits."""
@inline function _P_over_sinth_at_pole(l::Int, m::Int, x::Float64, N::Float64)
    if m == 0
        return 0.0
    elseif m == 1
        if x > 0
            return N * (-Float64(l * (l + 1)) / 2)
        else
            return N * Float64((-1)^l) * l * (l + 1) / 2
        end
    else
        return 0.0
    end
end

# ============================================================================
# SPHTOR SYNTHESIS KERNELS
# ============================================================================

"""Sphtor synthesis kernel using precomputed tables. Returns (g_theta, g_phi)."""
@inline function _sphtor_synthesis_kernel(cfg, Slm, Tlm, tblP, tbld, i, col, m, ltr)
    x = cfg.x[i]
    s_theta = sqrt(max(0.0, 1 - x*x))
    is_pole = s_theta < POLE_TOLERANCE_FACTOR * eps(Float64)
    inv_s_theta = is_pole ? 0.0 : 1.0 / s_theta
    g_theta = zero(ComplexF64)
    g_phi = zero(ComplexF64)
    @inbounds for l in max(1, m):ltr
        N = cfg.Nlm[l+1, col]
        if is_pole
            dtheta_Y = _dPdtheta_at_pole(l, m, x, N)
            Y_over_s = _P_over_sinth_at_pole(l, m, x, N)
        else
            dtheta_Y = -s_theta * N * tbld[l+1, i]
            Y_over_s = N * tblP[l+1, i] * inv_s_theta
        end
        Sl = Slm[l+1, col]; Tl = Tlm[l+1, col]
        g_theta += dtheta_Y * Sl - 1.0im * m * Y_over_s * Tl
        g_phi += 1.0im * m * Y_over_s * Sl + dtheta_Y * Tl
    end
    return g_theta, g_phi
end

"""Sphtor synthesis kernel computing Legendre on the fly. Returns (g_theta, g_phi)."""
@inline function _sphtor_synthesis_kernel_otf(cfg, Slm, Tlm, P, dPdtheta, P_over_sinth, i, col, m, ltr)
    Plm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, cfg.x[i], cfg.lmax, m)
    g_theta = zero(ComplexF64)
    g_phi = zero(ComplexF64)
    @inbounds for l in max(1, m):ltr
        N = cfg.Nlm[l+1, col]
        dtheta_Y = N * dPdtheta[l+1]
        Y_over_s = N * P_over_sinth[l+1]
        Sl = Slm[l+1, col]; Tl = Tlm[l+1, col]
        g_theta += dtheta_Y * Sl - 1.0im * m * Y_over_s * Tl
        g_phi += 1.0im * m * Y_over_s * Sl + dtheta_Y * Tl
    end
    return g_theta, g_phi
end

# ============================================================================
# SPHTOR ANALYSIS KERNELS
# ============================================================================

"""Sphtor analysis kernel using precomputed tables. Accumulates into Sacc, Tacc."""
@inline function _sphtor_analysis_kernel!(Sacc, Tacc, cfg, Ftheta_i, Fphi_i, wi, tblP, tbld, i, col, m, ltr, scale_phi)
    x = cfg.x[i]
    s_theta = sqrt(max(0.0, 1 - x*x))
    is_pole = s_theta < POLE_TOLERANCE_FACTOR * eps(Float64)
    inv_s_theta = is_pole ? 0.0 : 1.0 / s_theta
    @inbounds for l in max(1, m):ltr
        N = cfg.Nlm[l+1, col]
        if is_pole
            dtheta_Y = _dPdtheta_at_pole(l, m, x, N)
            Y_over_s = _P_over_sinth_at_pole(l, m, x, N)
        else
            dtheta_Y = -s_theta * N * tbld[l+1, i]
            Y_over_s = N * tblP[l+1, i] * inv_s_theta
        end
        coeff = wi * scale_phi / (l * (l + 1))
        term = 1.0im * m * Y_over_s
        Sacc[l+1] += coeff * (Ftheta_i * dtheta_Y + conj(term) * Fphi_i)
        Tacc[l+1] += coeff * (-conj(term) * Ftheta_i + dtheta_Y * Fphi_i)
    end
end

"""Sphtor analysis kernel computing Legendre on the fly. Accumulates into Sacc, Tacc."""
@inline function _sphtor_analysis_kernel_otf!(Sacc, Tacc, cfg, Ftheta_i, Fphi_i, wi, P, dPdtheta, P_over_sinth, i, col, m, ltr, scale_phi)
    Plm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, cfg.x[i], cfg.lmax, m)
    @inbounds for l in max(1, m):ltr
        N = cfg.Nlm[l+1, col]
        dtheta_Y = N * dPdtheta[l+1]
        Y_over_s = N * P_over_sinth[l+1]
        coeff = wi * scale_phi / (l * (l + 1))
        term = 1.0im * m * Y_over_s
        Sacc[l+1] += coeff * (Ftheta_i * dtheta_Y + conj(term) * Fphi_i)
        Tacc[l+1] += coeff * (-conj(term) * Ftheta_i + dtheta_Y * Fphi_i)
    end
end
