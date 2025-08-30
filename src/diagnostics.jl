"""
Diagnostic Functions for Spherical Harmonic Fields

This module provides functions to compute physical diagnostics from spectral
coefficients, including energy, enstrophy, and vorticity. These diagnostics
are fundamental for analyzing geophysical flows and validating numerical methods.

Key assumptions:
- Uses orthonormal spherical harmonics with Condon-Shortley phase convention
- All spectral coefficient matrices have shape (lmax+1, mmax+1) with m ≥ 0
- For real fields, m>0 modes contribute twice due to Hermitian symmetry
"""

# ===== HERMITIAN SYMMETRY WEIGHTS =====
# For real fields, we store only m≥0 coefficients but need to account for
# the symmetric m<0 modes in energy calculations
_wm_real(cfg::SHTConfig) = [m == 0 ? 1.0 : 2.0 for m in 0:cfg.mmax]

"""
    energy_scalar(cfg, alm; real_field=true) -> Float64

Compute the total energy of a scalar field from its spherical harmonic coefficients.

For a scalar field f(θ,φ) = Σ a_lm Y_l^m(θ,φ), the energy is defined as:
E = (1/2) ∫ |f|² dΩ = (1/2) Σ |a_lm|²

This represents the L² norm of the field, which is conserved under orthonormal
spherical harmonic transforms (Parseval's identity).
"""
function energy_scalar(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    # Validate coefficient array dimensions
    size(alm,1) == cfg.lmax+1 && size(alm,2) == cfg.mmax+1 || throw(DimensionMismatch("alm dims"))
    
    # Set up weights for real field symmetry
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    
    # Accumulate energy from all modes
    e = 0.0
    @inbounds for m in 0:cfg.mmax           # Loop over azimuthal orders
        for l in m:cfg.lmax                 # Loop over degrees (l ≥ m constraint)
            e += w[m+1] * abs2(alm[l+1, m+1])  # |a_lm|² with symmetry weight
        end
    end
    
    # Factor of 1/2 from energy definition
    return 0.5 * e
end

"""
    energy_vector(cfg, Slm, Tlm; real_field=true) -> Float64

Compute the kinetic energy of a vector field from its spheroidal/toroidal decomposition.

For a vector field V(θ,φ) = ∇×(T r̂) + ∇(S), where S and T are the spheroidal
and toroidal scalars, the kinetic energy is:

E = (1/2) ∫ |V|² dΩ = (1/2) Σ l(l+1) [|S_lm|² + |T_lm|²]

The l(l+1) factor arises from the gradient operators in the spheroidal/toroidal
decomposition. This energy is fundamental in fluid dynamics applications.
"""
function energy_vector(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    # Validate coefficient array dimensions
    size(Slm) == (cfg.lmax+1, cfg.mmax+1) && size(Tlm) == size(Slm) || throw(DimensionMismatch("Slm/Tlm dims"))
    
    # Set up weights for real field symmetry
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    
    # Accumulate kinetic energy from both spheroidal and toroidal components
    e = 0.0
    @inbounds for m in 0:cfg.mmax
        for l in max(1,m):cfg.lmax           # Start from l=1 since l=0 has no gradient
            L2 = l*(l+1)                     # Laplacian eigenvalue factor
            e += w[m+1] * L2 * (abs2(Slm[l+1, m+1]) + abs2(Tlm[l+1, m+1]))
        end
    end
    
    # Factor of 1/2 from kinetic energy definition
    return 0.5 * e
end

"""
    enstrophy(cfg, Tlm; real_field=true) -> Float64

Compute the enstrophy from the toroidal component of a vector field.

Enstrophy measures the mean-square vorticity and is defined as:
Z = (1/2) ∫ |ζ|² dΩ = (1/2) Σ [l(l+1)]² |T_lm|²

where ζ is the vorticity (curl of the velocity field). In the spheroidal/toroidal
decomposition, vorticity comes solely from the toroidal component T, with the
[l(l+1)]² factor arising from the curl operation.

Enstrophy is a key diagnostic in 2D turbulence and geostrophic flows.
"""
function enstrophy(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    # Validate coefficient array dimensions
    size(Tlm) == (cfg.lmax+1, cfg.mmax+1) || throw(DimensionMismatch("Tlm dims"))
    
    # Set up weights for real field symmetry
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    
    # Accumulate enstrophy from toroidal modes
    z = 0.0
    @inbounds for m in 0:cfg.mmax
        for l in max(1,m):cfg.lmax           # Start from l=1 since l=0 has no curl
            L2 = l*(l+1)                     # Laplacian eigenvalue
            z += w[m+1] * (L2^2) * abs2(Tlm[l+1, m+1])  # [l(l+1)]² factor from curl
        end
    end
    
    # Factor of 1/2 from enstrophy definition
    return 0.5 * z
end

"""vorticity_spectral(cfg, Tlm::AbstractMatrix) -> Matrix{ComplexF64}
Compute vorticity coefficients ζ_lm = l(l+1) T_lm (m ≥ 0)."""
function vorticity_spectral(cfg::SHTConfig, Tlm::AbstractMatrix)
    size(Tlm) == (cfg.lmax+1, cfg.mmax+1) || throw(DimensionMismatch("Tlm dims"))
    ζ = zeros(ComplexF64, size(Tlm))
    @inbounds for m in 0:cfg.mmax
        for l in max(1,m):cfg.lmax
            ζ[l+1, m+1] = (l*(l+1)) * Tlm[l+1, m+1]
        end
    end
    return ζ
end

"""vorticity_grid(cfg, Tlm::AbstractMatrix) -> Matrix{Float64}
Synthesize vorticity field on the grid from toroidal spectrum Tlm."""
function vorticity_grid(cfg::SHTConfig, Tlm::AbstractMatrix)
    ζlm = vorticity_spectral(cfg, Tlm)
    ζ = synthesis(cfg, ζlm; real_output=true)
    return ζ
end

"""vorticity_radial_grid(cfg, Tlm::AbstractMatrix, r::Real) -> Matrix{Float64}
Radial vorticity (3D) on the grid from toroidal spectrum T_lm at radius r.

This scales the unit-sphere vorticity by 1/r to obtain the 3D radial component.
"""
function vorticity_radial_grid(cfg::SHTConfig, Tlm::AbstractMatrix, r::Real)
    r == 0 && throw(ArgumentError("radius r must be > 0"))
    ζ = vorticity_grid(cfg, Tlm)
    return ζ ./ r
end

"""grid_energy_scalar(cfg, f::AbstractMatrix) -> Float64"""
function grid_energy_scalar(cfg::SHTConfig, f::AbstractMatrix)
    size(f,1) == cfg.nlat && size(f,2) == cfg.nlon || throw(DimensionMismatch("f dims"))
    e = 0.0
    @inbounds for i in 1:cfg.nlat
        wi = cfg.w[i]
        for j in 1:cfg.nlon
            e += wi * abs2(f[i,j])
        end
    end
    e *= (2π / cfg.nlon)
    return 0.5 * e
end

"""grid_energy_vector(cfg, Vt::AbstractMatrix, Vp::AbstractMatrix) -> Float64"""
function grid_energy_vector(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    size(Vt) == (cfg.nlat, cfg.nlon) && size(Vp) == size(Vt) || throw(DimensionMismatch("V dims"))
    e = 0.0
    @inbounds for i in 1:cfg.nlat
        wi = cfg.w[i]
        for j in 1:cfg.nlon
            e += wi * (abs2(Vt[i,j]) + abs2(Vp[i,j]))
        end
    end
    e *= (2π / cfg.nlon)
    return 0.5 * e
end

"""grid_enstrophy(cfg, ζ::AbstractMatrix) -> Float64"""
function grid_enstrophy(cfg::SHTConfig, ζ::AbstractMatrix)
    size(ζ) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch("ζ dims"))
    z = 0.0
    @inbounds for i in 1:cfg.nlat
        wi = cfg.w[i]
        for j in 1:cfg.nlon
            z += wi * abs2(ζ[i,j])
        end
    end
    z *= (2π / cfg.nlon)
    return 0.5 * z
end

# --- Convenience gradients ---

"""
    grad_energy_scalar_alm(cfg, alm; real_field=true) -> Matrix

Gradient of scalar energy 0.5 ∑ w_m |a_{l,m}|^2 with respect to `alm`.
Returns `G[l+1, m+1] = w_m * alm[l+1, m+1]` for l≥m (and zero otherwise).
"""
function grad_energy_scalar_alm(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    size(alm,1) == cfg.lmax+1 && size(alm,2) == cfg.mmax+1 || throw(DimensionMismatch("alm dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    G = zeros(eltype(alm), size(alm))
    @inbounds for m in 0:cfg.mmax
        wm = w[m+1]
        for l in m:cfg.lmax
            G[l+1, m+1] = wm * alm[l+1, m+1]
        end
    end
    return G
end

# --- Packed (real) scalar energy helpers ---

"""
    energy_scalar_packed(cfg, Qlm; real_field=true) -> Float64

Energy computed directly from packed real coefficients Qlm (m ≥ 0).
"""
function energy_scalar_packed(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}; real_field::Bool=true)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    e = 0.0
    @inbounds for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]
        e += (w[m+1]/2) * abs2(Qlm[lm0+1])
    end
    return e
end

"""
    grad_energy_scalar_packed(cfg, Qlm; real_field=true) -> Vector

Gradient of energy_scalar_packed with respect to packed coefficients Qlm.
"""
function grad_energy_scalar_packed(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}; real_field::Bool=true)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    G = similar(Qlm)
    @inbounds for lm0 in 0:(cfg.nlm-1)
        m = cfg.mi[lm0+1]
        G[lm0+1] = w[m+1] * Qlm[lm0+1]
    end
    return G
end

"""
    grad_energy_vector_Slm_Tlm(cfg, Slm, Tlm; real_field=true) -> (∂E/∂Slm, ∂E/∂Tlm)

Gradient of vector energy 0.5 ∑ w_m l(l+1) (|S|^2 + |T|^2).
"""
function grad_energy_vector_Slm_Tlm(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    size(Slm) == (cfg.lmax+1, cfg.mmax+1) && size(Tlm) == size(Slm) || throw(DimensionMismatch("dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    GS = zeros(eltype(Slm), size(Slm))
    GT = zeros(eltype(Tlm), size(Tlm))
    @inbounds for m in 0:cfg.mmax
        wm = w[m+1]
        for l in max(1,m):cfg.lmax
            L2 = l*(l+1)
            factor = wm * L2
            GS[l+1, m+1] = factor * Slm[l+1, m+1]
            GT[l+1, m+1] = factor * Tlm[l+1, m+1]
        end
    end
    return GS, GT
end

"""
    grad_enstrophy_Tlm(cfg, Tlm; real_field=true) -> Matrix

Gradient of enstrophy 0.5 ∑ w_m [l(l+1)]^2 |T|^2 with respect to Tlm.
"""
function grad_enstrophy_Tlm(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    size(Tlm) == (cfg.lmax+1, cfg.mmax+1) || throw(DimensionMismatch("dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    GT = zeros(eltype(Tlm), size(Tlm))
    @inbounds for m in 0:cfg.mmax
        wm = w[m+1]
        for l in max(1,m):cfg.lmax
            L2 = l*(l+1)
            GT[l+1, m+1] = wm * (L2^2) * Tlm[l+1, m+1]
        end
    end
    return GT
end

"""
    grad_grid_energy_scalar_field(cfg, f) -> Matrix

Gradient of discrete grid scalar energy 0.5 (2π/nlon) ∑ w_i ∑ |f|^2 with respect to `f`.
Returns g[i,j] = (2π/nlon) * w[i] * f[i,j].
"""
function grad_grid_energy_scalar_field(cfg::SHTConfig, f::AbstractMatrix)
    size(f) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch("f dims"))
    g = similar(f)
    φscale = 2π / cfg.nlon
    @inbounds for i in 1:cfg.nlat
        wi = cfg.w[i] * φscale
        for j in 1:cfg.nlon
            g[i,j] = wi * f[i,j]
        end
    end
    return g
end

"""
    grad_grid_energy_vector_fields(cfg, Vt, Vp) -> (Gt, Gp)

Gradient of discrete grid vector energy with respect to Vt and Vp.
"""
function grad_grid_energy_vector_fields(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    size(Vt) == (cfg.nlat, cfg.nlon) && size(Vp) == size(Vt) || throw(DimensionMismatch("V dims"))
    Gt = similar(Vt)
    Gp = similar(Vp)
    φscale = 2π / cfg.nlon
    @inbounds for i in 1:cfg.nlat
        wi = cfg.w[i] * φscale
        for j in 1:cfg.nlon
            Gt[i,j] = wi * Vt[i,j]
            Gp[i,j] = wi * Vp[i,j]
        end
    end
    return Gt, Gp
end

"""
    grad_grid_enstrophy_zeta(cfg, ζ) -> Matrix

Gradient of discrete grid enstrophy 0.5 (2π/nlon) ∑ w_i ∑ ζ^2 with respect to ζ.
"""
function grad_grid_enstrophy_zeta(cfg::SHTConfig, ζ::AbstractMatrix)
    size(ζ) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch("ζ dims"))
    g = similar(ζ)
    φscale = 2π / cfg.nlon
    @inbounds for i in 1:cfg.nlat
        wi = cfg.w[i] * φscale
        for j in 1:cfg.nlon
            g[i,j] = wi * ζ[i,j]
        end
    end
    return g
end

# --- Spectra by l and m ---

"""
    energy_scalar_l_spectrum(cfg, alm; real_field=true) -> Vector

Return per-degree energy spectrum E_l with length lmax+1 for a scalar field.
E_l[l+1] = 1/2 ∑_{m=0}^{min(l,mmax)} w_m |a_{l,m}|^2 where w_0=1, w_{m>0}=2 for real fields.
"""
function energy_scalar_l_spectrum(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    size(alm,1) == cfg.lmax+1 && size(alm,2) == cfg.mmax+1 || throw(DimensionMismatch("alm dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    T = typeof(abs2(alm[1,1]))
    E = zeros(T, cfg.lmax + 1)
    @inbounds for m in 0:cfg.mmax
        wm = w[m+1]
        for l in m:cfg.lmax
            E[l+1] += (wm/2) * abs2(alm[l+1, m+1])
        end
    end
    return E
end

"""
    energy_scalar_m_spectrum(cfg, alm; real_field=true) -> Vector

Return per-order energy spectrum E_m with length mmax+1 for a scalar field.
E_m[m+1] = 1/2 w_m ∑_{l=m}^{lmax} |a_{l,m}|^2.
"""
function energy_scalar_m_spectrum(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    size(alm,1) == cfg.lmax+1 && size(alm,2) == cfg.mmax+1 || throw(DimensionMismatch("alm dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    T = typeof(abs2(alm[1,1]))
    E = zeros(T, cfg.mmax + 1)
    @inbounds for m in 0:cfg.mmax
        s = zero(T)
        for l in m:cfg.lmax
            s += abs2(alm[l+1, m+1])
        end
        E[m+1] = (w[m+1]/2) * s
    end
    return E
end

"""
    energy_vector_l_spectrum(cfg, Slm, Tlm; real_field=true) -> Vector
"""
function energy_vector_l_spectrum(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    size(Slm) == (cfg.lmax+1, cfg.mmax+1) && size(Tlm) == size(Slm) || throw(DimensionMismatch("dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    T = typeof(abs2(Slm[1,1]) + abs2(Tlm[1,1]))
    E = zeros(T, cfg.lmax + 1)
    @inbounds for m in 0:cfg.mmax
        wm = w[m+1]
        for l in max(1,m):cfg.lmax
            L2 = l*(l+1)
            E[l+1] += (wm/2) * L2 * (abs2(Slm[l+1, m+1]) + abs2(Tlm[l+1, m+1]))
        end
    end
    return E
end

"""
    energy_vector_m_spectrum(cfg, Slm, Tlm; real_field=true) -> Vector
"""
function energy_vector_m_spectrum(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    size(Slm) == (cfg.lmax+1, cfg.mmax+1) && size(Tlm) == size(Slm) || throw(DimensionMismatch("dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    T = typeof(abs2(Slm[1,1]) + abs2(Tlm[1,1]))
    E = zeros(T, cfg.mmax + 1)
    @inbounds for m in 0:cfg.mmax
        s = zero(T)
        for l in max(1,m):cfg.lmax
            L2 = l*(l+1)
            s += L2 * (abs2(Slm[l+1, m+1]) + abs2(Tlm[l+1, m+1]))
        end
        E[m+1] = (w[m+1]/2) * s
    end
    return E
end

"""
    enstrophy_l_spectrum(cfg, Tlm; real_field=true) -> Vector
"""
function enstrophy_l_spectrum(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    size(Tlm) == (cfg.lmax+1, cfg.mmax+1) || throw(DimensionMismatch("dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    T = typeof(abs2(Tlm[1,1]))
    Z = zeros(T, cfg.lmax + 1)
    @inbounds for m in 0:cfg.mmax
        wm = w[m+1]
        for l in max(1,m):cfg.lmax
            L2 = l*(l+1)
            Z[l+1] += (wm/2) * (L2^2) * abs2(Tlm[l+1, m+1])
        end
    end
    return Z
end

"""
    enstrophy_m_spectrum(cfg, Tlm; real_field=true) -> Vector
"""
function enstrophy_m_spectrum(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    size(Tlm) == (cfg.lmax+1, cfg.mmax+1) || throw(DimensionMismatch("dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    T = typeof(abs2(Tlm[1,1]))
    Z = zeros(T, cfg.mmax + 1)
    @inbounds for m in 0:cfg.mmax
        s = zero(T)
        for l in max(1,m):cfg.lmax
            L2 = l*(l+1)
            s += (L2^2) * abs2(Tlm[l+1, m+1])
        end
        Z[m+1] = (w[m+1]/2) * s
    end
    return Z
end

"""
    energy_scalar_lm(cfg, alm; real_field=true) -> Matrix

Return 2D energy density matrix of size (lmax+1, mmax+1) with entries 1/2 w_m |a_{l,m}|^2 (zeros for l<m).
"""
function energy_scalar_lm(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    size(alm,1) == cfg.lmax+1 && size(alm,2) == cfg.mmax+1 || throw(DimensionMismatch("alm dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    T = typeof(abs2(alm[1,1]))
    D = zeros(T, cfg.lmax + 1, cfg.mmax + 1)
    @inbounds for m in 0:cfg.mmax
        wm = w[m+1]
        for l in m:cfg.lmax
            D[l+1, m+1] = (wm/2) * abs2(alm[l+1, m+1])
        end
    end
    return D
end

"""
    energy_vector_lm(cfg, Slm, Tlm; real_field=true) -> Matrix
"""
function energy_vector_lm(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    size(Slm) == (cfg.lmax+1, cfg.mmax+1) && size(Tlm) == size(Slm) || throw(DimensionMismatch("dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    T = typeof(abs2(Slm[1,1]) + abs2(Tlm[1,1]))
    D = zeros(T, cfg.lmax + 1, cfg.mmax + 1)
    @inbounds for m in 0:cfg.mmax
        wm = w[m+1]
        for l in max(1,m):cfg.lmax
            L2 = l*(l+1)
            D[l+1, m+1] = (wm/2) * L2 * (abs2(Slm[l+1, m+1]) + abs2(Tlm[l+1, m+1]))
        end
    end
    return D
end

"""
    enstrophy_lm(cfg, Tlm; real_field=true) -> Matrix
"""
function enstrophy_lm(cfg::SHTConfig, Tlm::AbstractMatrix; real_field::Bool=true)
    size(Tlm) == (cfg.lmax+1, cfg.mmax+1) || throw(DimensionMismatch("dims"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    T = typeof(abs2(Tlm[1,1]))
    D = zeros(T, cfg.lmax + 1, cfg.mmax + 1)
    @inbounds for m in 0:cfg.mmax
        wm = w[m+1]
        for l in max(1,m):cfg.lmax
            L2 = l*(l+1)
            D[l+1, m+1] = (wm/2) * (L2^2) * abs2(Tlm[l+1, m+1])
        end
    end
    return D
end

# --- Packed vector energy helpers and vorticity control losses ---

"""
    energy_vector_packed(cfg, Spacked, Tpacked; real_field=true) -> Float64

Vector energy computed from packed (m≥0) coefficients for S and T.
"""
function energy_vector_packed(cfg::SHTConfig, Spacked::AbstractVector{<:Complex}, Tpacked::AbstractVector{<:Complex}; real_field::Bool=true)
    length(Spacked) == cfg.nlm || throw(DimensionMismatch("packed length must be nlm"))
    length(Tpacked) == cfg.nlm || throw(DimensionMismatch("packed length must be nlm"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    e = 0.0
    @inbounds for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]
        if l == 0
            continue
        end
        L2 = l*(l+1)
        e += (w[m+1]/2) * L2 * (abs2(Spacked[lm0+1]) + abs2(Tpacked[lm0+1]))
    end
    return e
end

"""
    grad_energy_vector_packed(cfg, Spacked, Tpacked; real_field=true) -> (∂E/∂Spacked, ∂E/∂Tpacked)
"""
function grad_energy_vector_packed(cfg::SHTConfig, Spacked::AbstractVector{<:Complex}, Tpacked::AbstractVector{<:Complex}; real_field::Bool=true)
    length(Spacked) == cfg.nlm || throw(DimensionMismatch("packed length must be nlm"))
    length(Tpacked) == cfg.nlm || throw(DimensionMismatch("packed length must be nlm"))
    w = real_field ? _wm_real(cfg) : ones(cfg.mmax + 1)
    GS = similar(Spacked)
    GT = similar(Tpacked)
    fill!(GS, zero(eltype(GS)))
    fill!(GT, zero(eltype(GT)))
    @inbounds for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]
        if l == 0
            continue
        end
        L2 = l*(l+1)
        factor = w[m+1] * L2
        GS[lm0+1] = factor * Spacked[lm0+1]
        GT[lm0+1] = factor * Tpacked[lm0+1]
    end
    return GS, GT
end

"""
    loss_vorticity_grid(cfg, Tlm, ζ_target) -> Float64

Loss = 0.5 ∫ (ζ(Tlm) - ζ_target)^2 dΩ.
"""
function loss_vorticity_grid(cfg::SHTConfig, Tlm::AbstractMatrix, ζ_target::AbstractMatrix)
    ζ = vorticity_grid(cfg, Tlm)
    size(ζ_target) == size(ζ) || throw(DimensionMismatch("ζ_target dims"))
    return grid_enstrophy(cfg, ζ .- ζ_target)
end

"""
    grad_loss_vorticity_Tlm(cfg, Tlm, ζ_target) -> Matrix

Gradient of 0.5 ∫ (ζ(Tlm) - ζ_target)^2 dΩ with respect to Tlm.
"""
function grad_loss_vorticity_Tlm(cfg::SHTConfig, Tlm::AbstractMatrix, ζ_target::AbstractMatrix)
    size(Tlm) == (cfg.lmax+1, cfg.mmax+1) || throw(DimensionMismatch("Tlm dims"))
    size(ζ_target) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch("ζ_target dims"))
    lmax, mmax = cfg.lmax, cfg.mmax
    L2 = [l*(l+1) for l in 0:lmax]
    ζlm = zeros(eltype(Tlm), lmax+1, mmax+1)
    @inbounds for m in 0:mmax
        for l in max(1,m):lmax
            ζlm[l+1, m+1] = L2[l+1] * Tlm[l+1, m+1]
        end
    end
    ζ = synthesis(cfg, ζlm; real_output=true)
    gζ = grad_grid_enstrophy_zeta(cfg, ζ .- ζ_target)
    gζlm = analysis(cfg, gζ)
    gT = zeros(eltype(Tlm), size(Tlm))
    @inbounds for m in 0:mmax
        for l in max(1,m):lmax
            gT[l+1, m+1] = L2[l+1] * gζlm[l+1, m+1]
        end
    end
    return gT
end

"""
    loss_and_grad_vorticity_Tlm(cfg, Tlm, ζ_target) -> (loss, grad)
"""
function loss_and_grad_vorticity_Tlm(cfg::SHTConfig, Tlm::AbstractMatrix, ζ_target::AbstractMatrix)
    return loss_vorticity_grid(cfg, Tlm, ζ_target), grad_loss_vorticity_Tlm(cfg, Tlm, ζ_target)
end
