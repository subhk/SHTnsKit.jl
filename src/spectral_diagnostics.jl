"""
Spectral Analysis and Spectrum Functions

This module provides functions for analyzing the spectral properties of
spherical harmonic fields, including energy spectra by degree (l) and order (m),
and per-mode energy distributions.

These diagnostics are essential for understanding the scale-dependent properties
of geophysical flows and validating numerical methods.
"""

"""
    energy_scalar_l_spectrum(cfg, alm; real_field=true) -> Vector{Float64}

Compute energy spectrum as a function of spherical harmonic degree l.
Returns E(l) = Σₘ |a_lm|² for each l = 0..lmax.
"""
function energy_scalar_l_spectrum(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    if is_gpu_config(cfg)
        return gpu_energy_scalar_l_spectrum(cfg, alm; real_field=real_field)
    end
    lmax, mmax = cfg.lmax, cfg.mmax
    wm = real_field ? _wm_real(cfg) : ones(mmax+1)
    
    El = zeros(lmax + 1)
    for l in 0:lmax, m in 0:min(l, mmax)
        El[l+1] += wm[m+1] * abs2(alm[l+1, m+1])
    end
    return 0.5 * El
end

"""
    energy_scalar_m_spectrum(cfg, alm; real_field=true) -> Vector{Float64}

Compute energy spectrum as a function of spherical harmonic order m.
Returns E(m) = Σₗ |a_lm|² for each m = 0..mmax.
"""
function energy_scalar_m_spectrum(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    if is_gpu_config(cfg)
        return gpu_energy_scalar_m_spectrum(cfg, alm; real_field=real_field)
    end
    lmax, mmax = cfg.lmax, cfg.mmax
    wm = real_field ? _wm_real(cfg) : ones(mmax+1)
    
    Em = zeros(mmax + 1)
    for m in 0:mmax, l in m:lmax
        Em[m+1] += wm[m+1] * abs2(alm[l+1, m+1])
    end
    return 0.5 * Em
end

"""
    energy_vector_l_spectrum(cfg, Slm, Tlm; real_field=true) -> Vector{Float64}

Compute kinetic energy spectrum by degree l for vector fields.
Returns KE(l) = Σₘ l(l+1)[|S_lm|² + |T_lm|²] for each l = 1..lmax.
"""
function energy_vector_l_spectrum(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    if is_gpu_config(cfg)
        return gpu_energy_vector_l_spectrum(cfg, Slm, Tlm; real_field=real_field)
    end
    lmax, mmax = cfg.lmax, cfg.mmax
    wm = real_field ? _wm_real(cfg) : ones(mmax+1)
    
    El = zeros(lmax + 1)
    for l in 1:lmax, m in 0:min(l, mmax)  # Vector fields start at l=1
        ll1 = l * (l + 1)
        El[l+1] += wm[m+1] * ll1 * (abs2(Slm[l+1, m+1]) + abs2(Tlm[l+1, m+1]))
    end
    return 0.5 * El
end

"""
    energy_vector_m_spectrum(cfg, Slm, Tlm; real_field=true) -> Vector{Float64}

Compute kinetic energy spectrum by order m for vector fields.
Returns KE(m) = Σₗ l(l+1)[|S_lm|² + |T_lm|²] for each m = 0..mmax.
"""
function energy_vector_m_spectrum(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    if is_gpu_config(cfg)
        return gpu_energy_vector_m_spectrum(cfg, Slm, Tlm; real_field=real_field)
    end
    lmax, mmax = cfg.lmax, cfg.mmax
    wm = real_field ? _wm_real(cfg) : ones(mmax+1)
    
    Em = zeros(mmax + 1)
    for m in 0:mmax, l in max(1,m):lmax
        ll1 = l * (l + 1)
        Em[m+1] += wm[m+1] * ll1 * (abs2(Slm[l+1, m+1]) + abs2(Tlm[l+1, m+1]))
    end
    return 0.5 * Em
end

"""
    energy_scalar_lm(cfg, alm; real_field=true) -> Matrix{Float64}

Compute per-mode energy distribution E_lm = |a_lm|² for each (l,m) mode.
Returns matrix of size (lmax+1, mmax+1) with energy contributions.
"""
function energy_scalar_lm(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    if is_gpu_config(cfg)
        return gpu_energy_scalar_lm(cfg, alm; real_field=real_field)
    end
    lmax, mmax = cfg.lmax, cfg.mmax
    wm = real_field ? _wm_real(cfg) : ones(mmax+1)
    
    Elm = Matrix{Float64}(undef, lmax+1, mmax+1)
    fill!(Elm, 0.0)
    
    for m in 0:mmax, l in m:lmax
        Elm[l+1, m+1] = 0.5 * wm[m+1] * abs2(alm[l+1, m+1])
    end
    return Elm
end

"""
    energy_vector_lm(cfg, Slm, Tlm; real_field=true) -> Matrix{Float64}

Compute per-mode kinetic energy KE_lm = l(l+1)[|S_lm|² + |T_lm|²] for vector fields.
Returns matrix of size (lmax+1, mmax+1) with energy contributions.
"""
function energy_vector_lm(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    if is_gpu_config(cfg)
        return gpu_energy_vector_lm(cfg, Slm, Tlm; real_field=real_field)
    end
    lmax, mmax = cfg.lmax, cfg.mmax
    wm = real_field ? _wm_real(cfg) : ones(mmax+1)
    
    Elm = Matrix{Float64}(undef, lmax+1, mmax+1)
    fill!(Elm, 0.0)
    
    for m in 0:mmax, l in max(1,m):lmax
        ll1 = l * (l + 1)
        Elm[l+1, m+1] = 0.5 * wm[m+1] * ll1 * (abs2(Slm[l+1, m+1]) + abs2(Tlm[l+1, m+1]))
    end
    return Elm
end
