"""
Vorticity and Enstrophy Diagnostics

This module provides functions for computing vorticity-related diagnostics
from spherical harmonic fields, including enstrophy calculations, vorticity
transforms, and optimization functions for vorticity-based inverse problems.

Key physical concepts:
- Enstrophy: ½∫ζ²dΩ where ζ is relative vorticity
- Vorticity: ζ = ∇×V for velocity field V
- These are fundamental diagnostics in fluid dynamics
"""

"""
    enstrophy(cfg, Tlm; real_field=true) -> Float64

Compute the enstrophy (kinetic energy of vorticity) from toroidal coefficients.

For a toroidal field with coefficients T_lm, the enstrophy is:
Z = (1/2) ∫ |∇×V|² dΩ = (1/2) Σ l²(l+1)²|T_lm|²

This is a measure of the small-scale intensity of rotational motion.
"""
function _enstrophy_cpu(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    wm = real_field ? _wm_real(cfg) : ones(mmax+1)
    
    Z = 0.0
    for m in 0:mmax, l in max(1,m):lmax  # Vorticity starts at l=1
        ll1_sq = (l * (l + 1))^2
        Z += wm[m+1] * ll1_sq * abs2(Tlm[l+1, m+1])
    end
    return 0.5 * Z
end

function enstrophy(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    if is_gpu_config(cfg)
        return gpu_enstrophy(cfg, Tlm; real_field=real_field)
    end
    return _enstrophy_cpu(cfg, Tlm; real_field=real_field)
end

"""
    vorticity_spectral(cfg, Tlm) -> Matrix{ComplexF64}

Compute vorticity coefficients ζ_lm from toroidal stream function coefficients T_lm.
Relationship: ζ_lm = -l(l+1) T_lm
"""
function _vorticity_spectral_cpu(cfg::SHTConfig, Tlm::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    ζlm = similar(Tlm)
    fill!(ζlm, 0.0)
    
    for m in 0:mmax, l in max(1,m):lmax
        ll1 = l * (l + 1)
        ζlm[l+1, m+1] = -ll1 * Tlm[l+1, m+1]
    end
    return ζlm
end

function vorticity_spectral(cfg::SHTConfig, Tlm::AbstractMatrix)
    if is_gpu_config(cfg)
        return gpu_vorticity_spectral(cfg, Tlm)
    end
    return _vorticity_spectral_cpu(cfg, Tlm)
end

"""
    vorticity_grid(cfg, Tlm) -> Matrix

Transform toroidal coefficients to vorticity on the spatial grid.
This combines spectral vorticity calculation with spherical harmonic synthesis.
"""
function vorticity_grid(cfg::SHTConfig, Tlm::AbstractMatrix)
    if is_gpu_config(cfg)
        return gpu_vorticity_grid(cfg, Tlm)
    end
    ζlm = _vorticity_spectral_cpu(cfg, Tlm)
    return synthesis(cfg, ζlm; real_output=true)
end

"""
    grid_enstrophy(cfg, ζ) -> Float64

Compute enstrophy directly from gridded vorticity field using quadrature.
Z = (1/2) ∫ ζ² dΩ using Gauss-Legendre integration.
"""
function _grid_enstrophy_cpu(cfg::SHTConfig, ζ::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    wlat = cfg.wlat  # Gauss-Legendre weights
    
    Z = 0.0
    for j in 1:nlon, i in 1:nlat
        Z += wlat[i] * abs2(ζ[i, j])
    end
    return 0.5 * Z * (2π / nlon)
end

function grid_enstrophy(cfg::SHTConfig, ζ::AbstractMatrix)
    if is_gpu_config(cfg)
        return gpu_grid_enstrophy(cfg, ζ)
    end
    return _grid_enstrophy_cpu(cfg, ζ)
end

"""
    grad_enstrophy_Tlm(cfg, Tlm; real_field=true) -> Matrix

Compute gradient of enstrophy with respect to toroidal coefficients.
Returns ∂Z/∂T_lm = l²(l+1)² T_lm for optimization applications.
"""
function _grad_enstrophy_Tlm_cpu(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    wm = real_field ? _wm_real(cfg) : ones(mmax+1)
    
    grad = similar(Tlm)
    fill!(grad, 0.0)
    
    for m in 0:mmax, l in max(1,m):lmax
        ll1_sq = (l * (l + 1))^2
        grad[l+1, m+1] = wm[m+1] * ll1_sq * Tlm[l+1, m+1]
    end
    return grad
end

function grad_enstrophy_Tlm(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    if is_gpu_config(cfg)
        return gpu_grad_enstrophy_Tlm(cfg, Tlm; real_field=real_field)
    end
    return _grad_enstrophy_Tlm_cpu(cfg, Tlm; real_field=real_field)
end

"""
    grad_grid_enstrophy_zeta(cfg, ζ) -> Matrix

Compute gradient of grid-based enstrophy with respect to vorticity field values.
Returns ∂Z/∂ζ for each grid point.
"""
function _grad_grid_enstrophy_zeta_cpu(cfg::SHTConfig, ζ::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    wlat = cfg.wlat
    scale = (2π / nlon)
    
    grad = similar(ζ)
    for j in 1:nlon, i in 1:nlat
        grad[i, j] = scale * wlat[i] * ζ[i, j]
    end
    return grad
end

function grad_grid_enstrophy_zeta(cfg::SHTConfig, ζ::AbstractMatrix)
    if is_gpu_config(cfg)
        return gpu_grad_grid_enstrophy_zeta(cfg, ζ)
    end
    return _grad_grid_enstrophy_zeta_cpu(cfg, ζ)
end

"""
    enstrophy_l_spectrum(cfg, Tlm; real_field=true) -> Vector{Float64}

Compute enstrophy spectrum as a function of degree l.
Returns Z(l) = Σₘ l²(l+1)²|T_lm|² for each l = 1..lmax.
"""
function _enstrophy_l_spectrum_cpu(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    wm = real_field ? _wm_real(cfg) : ones(mmax+1)
    
    Zl = zeros(lmax + 1)
    for l in 1:lmax, m in 0:min(l, mmax)
        ll1_sq = (l * (l + 1))^2
        Zl[l+1] += wm[m+1] * ll1_sq * abs2(Tlm[l+1, m+1])
    end
    return 0.5 * Zl
end

function enstrophy_l_spectrum(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    if is_gpu_config(cfg)
        return gpu_enstrophy_l_spectrum(cfg, Tlm; real_field=real_field)
    end
    return _enstrophy_l_spectrum_cpu(cfg, Tlm; real_field=real_field)
end

"""
    enstrophy_m_spectrum(cfg, Tlm; real_field=true) -> Vector{Float64}

Compute enstrophy spectrum as a function of order m.
Returns Z(m) = Σₗ l²(l+1)²|T_lm|² for each m = 0..mmax.
"""
function _enstrophy_m_spectrum_cpu(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    wm = real_field ? _wm_real(cfg) : ones(mmax+1)
    
    Zm = zeros(mmax + 1)
    for m in 0:mmax, l in max(1,m):lmax
        ll1_sq = (l * (l + 1))^2
        Zm[m+1] += wm[m+1] * ll1_sq * abs2(Tlm[l+1, m+1])
    end
    return 0.5 * Zm
end

function enstrophy_m_spectrum(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    if is_gpu_config(cfg)
        return gpu_enstrophy_m_spectrum(cfg, Tlm; real_field=real_field)
    end
    return _enstrophy_m_spectrum_cpu(cfg, Tlm; real_field=real_field)
end

"""
    enstrophy_lm(cfg, Tlm; real_field=true) -> Matrix{Float64}

Compute per-mode enstrophy Z_lm = l²(l+1)²|T_lm|² for each (l,m) mode.
Returns matrix of size (lmax+1, mmax+1) with enstrophy contributions.
"""
function _enstrophy_lm_cpu(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    wm = real_field ? _wm_real(cfg) : ones(mmax+1)
    
    Zlm = Matrix{Float64}(undef, lmax+1, mmax+1)
    fill!(Zlm, 0.0)
    
    for m in 0:mmax, l in max(1,m):lmax
        ll1_sq = (l * (l + 1))^2
        Zlm[l+1, m+1] = 0.5 * wm[m+1] * ll1_sq * abs2(Tlm[l+1, m+1])
    end
    return Zlm
end

function enstrophy_lm(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    if is_gpu_config(cfg)
        return gpu_enstrophy_lm(cfg, Tlm; real_field=real_field)
    end
    return _enstrophy_lm_cpu(cfg, Tlm; real_field=real_field)
end

# ===== INVERSE PROBLEM FUNCTIONS =====
# These functions support optimization problems where we want to find
# spectral coefficients that produce a desired vorticity pattern

"""
    loss_vorticity_grid(cfg, Tlm, ζ_target) -> Float64

Compute loss function for vorticity matching problem.
L = (1/2) ∫ [ζ(Tlm) - ζ_target]² dΩ where ζ(Tlm) is computed vorticity.
"""
function loss_vorticity_grid(cfg::SHTConfig, Tlm::AbstractMatrix, ζ_target::AbstractMatrix)
    if is_gpu_config(cfg)
        return gpu_loss_vorticity_grid(cfg, Tlm, ζ_target)
    end
    ζ = _vorticity_grid_cpu(cfg, Tlm)
    diff = ζ .- ζ_target
    return _grid_enstrophy_cpu(cfg, diff)
end

"""
    grad_loss_vorticity_Tlm(cfg, Tlm, ζ_target) -> Matrix

Compute gradient of vorticity loss with respect to toroidal coefficients.
Uses adjoint method for efficient computation.
"""
function _vorticity_grid_cpu(cfg::SHTConfig, Tlm::AbstractMatrix)
    ζlm = _vorticity_spectral_cpu(cfg, Tlm)
    return synthesis(cfg, ζlm; real_output=true)
end

function grad_loss_vorticity_Tlm(cfg::SHTConfig, Tlm::AbstractMatrix, ζ_target::AbstractMatrix)
    if is_gpu_config(cfg)
        return gpu_grad_loss_vorticity_Tlm(cfg, Tlm, ζ_target)
    end
    ζ = _vorticity_grid_cpu(cfg, Tlm)
    residual = ζ .- ζ_target
    gζlm = analysis(cfg, residual)
    lmax, mmax = cfg.lmax, cfg.mmax
    gT = similar(Tlm)
    fill!(gT, 0.0)
    for m in 0:mmax, l in max(1,m):lmax
        L2 = l * (l + 1)
        gT[l+1, m+1] = -L2 * gζlm[l+1, m+1]
    end
    return gT
end

"""
    loss_and_grad_vorticity_Tlm(cfg, Tlm, ζ_target) -> (loss, grad)

Combined computation of loss and gradient for vorticity optimization.
More efficient than separate computations due to shared forward pass.
"""
function loss_and_grad_vorticity_Tlm(cfg::SHTConfig, Tlm::AbstractMatrix, ζ_target::AbstractMatrix)
    if is_gpu_config(cfg)
        return gpu_loss_and_grad_vorticity_Tlm(cfg, Tlm, ζ_target)
    end
    ζ = _vorticity_grid_cpu(cfg, Tlm)
    residual = ζ .- ζ_target
    loss = _grid_enstrophy_cpu(cfg, residual)
    gζlm = analysis(cfg, residual)
    lmax, mmax = cfg.lmax, cfg.mmax
    gT = similar(Tlm)
    fill!(gT, 0.0)
    for m in 0:mmax, l in max(1,m):lmax
        L2 = l * (l + 1)
        gT[l+1, m+1] = -L2 * gζlm[l+1, m+1]
    end
    return loss, gT
end
