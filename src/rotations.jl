#=
================================================================================
rotations.jl - Rotations of Spherical Harmonic Expansions
================================================================================

This file implements rotation operations on spherical harmonic coefficients.
Rotations can be applied directly in spectral space without going through
physical space, which is efficient for rotating fields on the sphere.

WHY SPECTRAL ROTATIONS?
-----------------------
Rotating a function on the sphere in physical space requires:
1. Synthesize to grid: O((lmax)² × nlon)
2. Interpolate to new grid positions: O(nlat × nlon)
3. Analyze back: O((lmax)² × nlon)

Spectral rotation is more direct:
- Rotation of Y_l^m produces a linear combination of Y_l^{m'} for |m'| ≤ l
- The mixing is given by Wigner-d matrices d^l_{mm'}(β)
- Complexity: O((lmax)³) but exact with no interpolation errors

EULER ANGLE CONVENTIONS
-----------------------
Rotations are specified using Euler angles (α, β, γ) in either:
- ZYZ convention (default): R = Rz(α) Ry(β) Rz(γ)
- ZXZ convention: R = Rz(α) Rx(β) Rz(γ)

Special cases implemented efficiently:
- Z-rotation: Just phase multiplication (m-dependent), O(nlm)
- Y-rotation: Requires full Wigner-d matrix application
- 90° rotations: Common in coordinate transformations

WIGNER-D MATRICES
-----------------
The little Wigner-d matrix d^l_{mm'}(β) gives the transformation of
spherical harmonics under rotation by angle β about the y-axis:

    R_y(β) Y_l^m = Σ_{m'} d^l_{m m'}(β) Y_l^{m'}

Full rotation R(α,β,γ) in ZYZ convention:
    a'_{lm} = Σ_{m'} e^{-imα} d^l_{mm'}(β) e^{-im'γ} a_{lm'}

FUNCTIONS
---------
Fast axis rotations:
    SH_Zrotate(cfg, Qlm, α, Rlm)     : Z-axis rotation (very fast)
    SH_Yrotate(cfg, Qlm, α, Rlm)     : Y-axis rotation
    SH_Xrotate90(cfg, Qlm, Rlm)      : X-axis 90° rotation
    SH_Yrotate90(cfg, Qlm, Rlm)      : Y-axis 90° rotation

General rotations:
    SHTRotation                       : Rotation specification struct
    shtns_rotation_set_angles_ZYZ     : Set Euler angles (ZYZ)
    shtns_rotation_apply_real         : Apply to real-field coefficients
    shtns_rotation_apply_cplx         : Apply to complex-field coefficients

Wigner-d computation:
    wigner_d_matrix(l, β)            : Compute d^l_{mm'}(β) matrix
    wigner_d_matrix_deriv(l, β)      : Derivative ∂d^l/∂β

USAGE EXAMPLE
-------------
```julia
cfg = create_gauss_config(32, 64)
Qlm = pack_alm(cfg, alm)  # Original coefficients
Rlm = similar(Qlm)        # Output

# Z-rotation by 45°
SH_Zrotate(cfg, Qlm, π/4, Rlm)

# General rotation using Euler angles
rot = SHTRotation(cfg.lmax, cfg.mmax)
shtns_rotation_set_angles_ZYZ(rot, α=π/6, β=π/3, γ=π/4)
shtns_rotation_apply_real(rot, Qlm, Rlm)
```

DEBUGGING
---------
```julia
# Z-rotation should just multiply by exp(imα)
# For m=2 mode, rotation by α should multiply by exp(2iα)
rot_coeff = Rlm[idx] / Qlm[idx]  # where idx is a mode with m=2
@assert rot_coeff ≈ cis(2 * α)
```

================================================================================
=#

"""
Rotations of spherical harmonic expansions.

Currently supports fast rotation around the Z-axis by angle `alpha` in radians.
"""

"""
    SH_Zrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})

Rotate a real-field SH expansion around the Z-axis by angle `alpha`.
Input and output are packed `Qlm` vectors (LM order, m ≥ 0). In-place supported if `Rlm === Qlm`.
"""
function SH_Zrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    length(Rlm) == cfg.nlm || throw(DimensionMismatch("Rlm length must be nlm=$(cfg.nlm)"))
    lmax = cfg.lmax; mres = cfg.mres
    @inbounds for m in 0:cfg.mmax
        (m % mres == 0) || continue
        phase = cis(m * alpha)
        for l in m:lmax
            lm = LM_index(lmax, mres, l, m) + 1
            Rlm[lm] = Qlm[lm] * phase
        end
    end
    return Rlm
end

"""
    struct SHTRotation
Holds Euler angles and target sizes for rotation.
- `lmax, mmax`: degrees/orders supported by the rotation.
- `α, β, γ`: Euler angles (ZYZ by default) in radians.
- `conv`: `:ZYZ` or `:ZXZ` convention.
"""
Base.@kwdef mutable struct SHTRotation
    lmax::Int
    mmax::Int
    α::Float64 = 0.0
    β::Float64 = 0.0
    γ::Float64 = 0.0
    conv::Symbol = :ZYZ
end

# Convenient outer constructor with keyword defaults (to mirror SHTns usage)
function SHTRotation(lmax::Integer, mmax::Integer; α::Real=0.0, β::Real=0.0, γ::Real=0.0, conv::Symbol=:ZYZ)
    return SHTRotation(Int(lmax), Int(mmax), float(α), float(β), float(γ), conv)
end

"""
    SH_Yrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})

Rotate a real-field SH expansion around the Y-axis by angle `alpha`.
Uses Wigner-d mixing per l; dispatches to the general rotation engine.
"""
function SH_Yrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    r = SHTRotation(cfg.lmax, cfg.mmax)
    shtns_rotation_set_angles_ZYZ(r, 0.0, float(alpha), 0.0)
    return shtns_rotation_apply_real(r, Qlm, Rlm)
end

"""
    SH_Yrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
"""
function SH_Yrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    return SH_Yrotate(cfg, Qlm, π/2, Rlm)
end

"""
    SH_Xrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Rotate around X-axis by 90 degrees using ZYZ equivalence: Rz(π/2)·Ry(π/2)·Rz(-π/2).
"""
function SH_Xrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    r = SHTRotation(cfg.lmax, cfg.mmax)
    shtns_rotation_set_angles_ZYZ(r, π/2, π/2, -π/2)
    return shtns_rotation_apply_real(r, Qlm, Rlm)
end

"""
    shtns_rotation_set_angle_axis(r::SHTRotation, theta::Real, Vx::Real, Vy::Real, Vz::Real)

Define rotation from angle-axis (theta around vector V).
Angles are set in ZYZ convention.
"""
function shtns_rotation_set_angle_axis(r::SHTRotation, theta::Real, Vx::Real, Vy::Real, Vz::Real)
    θ = float(theta)
    v = collect(float.((Vx, Vy, Vz)))
    n = hypot(v[1], hypot(v[2], v[3]))
    if n == 0
        r.α = 0.0; r.β = 0.0; r.γ = 0.0; r.conv = :ZYZ
        return nothing
    end
    kx, ky, kz = v ./ n
    c = cos(θ); s = sin(θ); t = 1 - c
    
    # Rotation matrix R = c I + s [k]_x + t k k^T
    R11 = c + t*kx*kx
    R12 = t*kx*ky - s*kz
    R13 = t*kx*kz + s*ky
    R21 = t*ky*kx + s*kz
    R22 = c + t*ky*ky
    R23 = t*ky*kz - s*kx
    R31 = t*kz*kx - s*ky
    R32 = t*kz*ky + s*kx
    R33 = c + t*kz*kz
    
    # Extract ZYZ Euler angles
    # For R = Rz(α)Ry(β)Rz(γ): R13 = cα*sβ, R23 = sα*sβ, R31 = -sβ*cγ, R32 = sβ*sγ
    β = acos(clamp(R33, -1.0, 1.0))
    if abs(sin(β)) > 1e-12
        α = atan(R23, R13)    # atan2(sα*sβ, cα*sβ) = α
        γ = atan(R32, -R31)   # atan2(sβ*sγ, sβ*cγ) = γ
    else
        # β ~ 0 or π: set γ = 0 and α from R11,R21
        α = atan(R21, R11)
        γ = 0.0
    end
    r.α = α; r.β = β; r.γ = γ; r.conv = :ZYZ
    return nothing
end

"""
    wigner_d_matrix(l::Int, beta::Float64) -> Matrix{Float64}

Compute little Wigner-d matrix d^l_{m m'}(β) with m,m' in [-l..l], returned as a
(2l+1)×(2l+1) real matrix where index is `m+l+1, m'+l+1`.
"""
function wigner_d_matrix(l::Int, beta::Float64)
    l ≥ 0 || throw(ArgumentError("l must be ≥ 0"))
    n = 2l + 1
    d = Matrix{Float64}(undef, n, n)
    cb = cos(beta/2)
    sb = sin(beta/2)
    for m in -l:l
        for mp in -l:l
            kmin = max(0, m - mp)
            kmax = min(l + m, l - mp)
            # prefactor sqrt((l+m)! (l-m)! (l+mp)! (l-mp)!)
            logpref = 0.5*(loggamma(l+m+1) + loggamma(l-m+1) + loggamma(l+mp+1) + loggamma(l-mp+1))
            s = 0.0
            for k in kmin:kmax
                # denominator (l+m-k)! k! (mp-m+k)! (l-mp-k)!
                logden = loggamma(l+m-k+1) + loggamma(k+1) + loggamma(mp-m+k+1) + loggamma(l-mp-k+1)
                p = 2l + m - mp - 2k
                q = mp - m + 2k
                term = (-1.0)^k * exp(logpref - logden) * (cb^p) * (sb^q)
                s += term
            end
            d[m + l + 1, mp + l + 1] = s
        end
    end
    return d
end

"""
    wigner_d_matrix_deriv(l::Int, beta::Float64) -> Matrix{Float64}

Derivative d/dβ of little Wigner-d matrix d^l_{m m'}(β).
"""
function wigner_d_matrix_deriv(l::Int, beta::Float64)
    l ≥ 0 || throw(ArgumentError("l must be ≥ 0"))
    n = 2l + 1
    dβ = Matrix{Float64}(undef, n, n)
    cb = cos(beta/2)
    sb = sin(beta/2)
    dcb = -0.5 * sb
    dsb =  0.5 * cb
    for m in -l:l
        for mp in -l:l
            kmin = max(0, m - mp)
            kmax = min(l + m, l - mp)
            logpref = 0.5*(loggamma(l+m+1) + loggamma(l-m+1) + loggamma(l+mp+1) + loggamma(l-mp+1))
            s = 0.0
            for k in kmin:kmax
                logden = loggamma(l+m-k+1) + loggamma(k+1) + loggamma(mp-m+k+1) + loggamma(l-mp-k+1)
                p = 2l + m - mp - 2k
                q = mp - m + 2k
                amp = (-1.0)^k * exp(logpref - logden)
                # derivative of cb^p sb^q
                term = amp * (cb^p) * (sb^q)
                dterm = 0.0
                if p != 0
                    dterm += term * p * (dcb / cb)
                end
                if q != 0
                    dterm += term * q * (dsb / sb)
                end
                s += dterm
            end
            dβ[m + l + 1, mp + l + 1] = s
        end
    end
    return dβ
end

"""
    shtns_rotation_create(lmax::Integer, mmax::Integer, norm::Integer) -> SHTRotation
"""
function shtns_rotation_create(lmax::Integer, mmax::Integer, norm::Integer)
    norm == 0 || throw(ArgumentError("only orthonormal normalization supported"))
    return SHTRotation(Int(lmax), Int(mmax))
end

"""shtns_rotation_destroy(r::SHTRotation)"""
shtns_rotation_destroy(::SHTRotation) = nothing

"""shtns_rotation_set_angles_ZYZ(r, alpha, beta, gamma)"""
function shtns_rotation_set_angles_ZYZ(r::SHTRotation, alpha::Real, beta::Real, gamma::Real)
    r.α = float(alpha); r.β = float(beta); r.γ = float(gamma); r.conv = :ZYZ; return nothing
end

"""shtns_rotation_set_angles_ZXZ(r, alpha, beta, gamma)"""
function shtns_rotation_set_angles_ZXZ(r::SHTRotation, alpha::Real, beta::Real, gamma::Real)
    r.α = float(alpha); r.β = float(beta); r.γ = float(gamma); r.conv = :ZXZ; return nothing
end

"""
    shtns_rotation_wigner_d_matrix(r::SHTRotation, l::Integer, mx::AbstractVector{<:Real}) -> Int

Fill `mx` (length ≥ (2l+1)^2) with d^l in row-major order. Returns size 2l+1.
"""
function shtns_rotation_wigner_d_matrix(r::SHTRotation, l::Integer, mx::AbstractVector{<:Real})
    l = Int(l)
    n = 2l + 1
    length(mx) ≥ n*n || throw(DimensionMismatch("mx must have length ≥ (2l+1)^2"))
    d = wigner_d_matrix(l, r.β)
    @inbounds for i in 1:n, j in 1:n
        mx[(i-1)*n + j] = d[i, j]
    end
    return n
end

"""
    shtns_rotation_apply_cplx(r::SHTRotation, Zlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Apply rotation with Euler angles (ZYZ/ZXZ) to complex SH coefficients in LM_cplx packing (mres==1).
"""
function shtns_rotation_apply_cplx(r::SHTRotation, Zlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    r.lmax ≥ 0 || return Rlm
    length(Zlm) == length(Rlm) || throw(DimensionMismatch("Zlm and Rlm length mismatch"))
    mres = 1
    expected = nlm_cplx_calc(r.lmax, r.mmax, mres)
    length(Zlm) == expected || throw(DimensionMismatch("LM_cplx size mismatch"))
    α, β, γ = r.α, r.β, r.γ
    
    # Apply R = diag(e^{-i m α}) * d^l(β) * diag(e^{-i m γ}) for each l
    for l in 0:r.lmax
        mm = min(l, r.mmax)
        # Build input vector b_m' = e^{-i m' γ} A_{m'} for m' in [-mm..mm]
        ncols = 2*mm + 1
        b = Vector{ComplexF64}(undef, 2l + 1)
        fill!(b, 0.0 + 0.0im)
        for mp in -mm:mm
            idx = LM_cplx_index(r.lmax, r.mmax, l, mp) + 1
            b[mp + l + 1] = Zlm[idx] * cis(-mp * γ)
        end
        # Multiply with d^l(β)
        dl = wigner_d_matrix(l, β)
        c = Vector{ComplexF64}(undef, 2l + 1)
        fill!(c, 0.0 + 0.0im)
        # c_m = sum_{m'} d_{m m'} b_{m'}
        for mi in -l:l
            acc = 0.0 + 0.0im
            for mp in -l:l
                acc += dl[mi + l + 1, mp + l + 1] * b[mp + l + 1]
            end
            c[mi + l + 1] = acc
        end
        # Apply phase e^{-i m α} and write back only for allowed |m| ≤ mm
        for m in -mm:mm
            idx = LM_cplx_index(r.lmax, r.mmax, l, m) + 1
            Rlm[idx] = c[m + l + 1] * cis(-m * α)
        end
    end
    return Rlm
end

"""
    shtns_rotation_apply_real(r::SHTRotation, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Apply rotation to real-field SH coefficients in packed LM layout (m ≥ 0). Requires `mres==1`.
"""
function shtns_rotation_apply_real(r::SHTRotation, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    expected = nlm_calc(r.lmax, r.mmax, 1)
    length(Qlm) == expected || throw(DimensionMismatch("LM packed size mismatch"))
    length(Rlm) == expected || throw(DimensionMismatch("LM packed size mismatch"))
    # Build LM_cplx array Zlm from real-packed Qlm using Hermitian symmetry a_{-m} = (-1)^m conj(a_m)
    Z = Vector{ComplexF64}(undef, nlm_cplx_calc(r.lmax, r.mmax, 1))
    
    # initialize zeros
    fill!(Z, 0.0 + 0.0im)
    for l in 0:r.lmax
        mm = min(l, r.mmax)
        # m = 0
        idxp = LM_index(r.lmax, 1, l, 0) + 1
        idxc = LM_cplx_index(r.lmax, r.mmax, l, 0) + 1
        Z[idxc] = Qlm[idxp]
        for m in 1:mm
            idxp = LM_index(r.lmax, 1, l, m) + 1
            idxc_p = LM_cplx_index(r.lmax, r.mmax, l, m) + 1
            idxc_n = LM_cplx_index(r.lmax, r.mmax, l, -m) + 1
            Am = Qlm[idxp]
            Z[idxc_p] = Am
            Z[idxc_n] = (-1)^m * conj(Am)
        end
    end
    R = similar(Z)
    shtns_rotation_apply_cplx(r, Z, R)
    # Pack back to positive-m layout
    for l in 0:r.lmax
        mm = min(l, r.mmax)
        idxp0 = LM_index(r.lmax, 1, l, 0) + 1
        idxc0 = LM_cplx_index(r.lmax, r.mmax, l, 0) + 1
        Rlm[idxp0] = R[idxc0]
        for m in 1:mm
            idxp = LM_index(r.lmax, 1, l, m) + 1
            idxc = LM_cplx_index(r.lmax, r.mmax, l, m) + 1
            Rlm[idxp] = R[idxc]
        end
    end
    return Rlm
end
