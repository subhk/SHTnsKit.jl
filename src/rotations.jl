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
function SH_Zrotate(::CPU, cfg::SHTConfig, Qlm::AbstractVector{<:Complex},
                    alpha::Real, Rlm::AbstractVector{<:Complex})
    _require_cpu_storage(:SH_Zrotate, Qlm)
    _require_cpu_storage(:SH_Zrotate, Rlm)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    length(Rlm) == cfg.nlm || throw(DimensionMismatch("Rlm length must be nlm=$(cfg.nlm)"))
    isfinite(alpha) || throw(ArgumentError("rotation angle must be finite"))
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

SH_Zrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real,
           Rlm::AbstractVector{<:Complex}) =
    SH_Zrotate(CPU(), cfg, Qlm, alpha, Rlm)

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
    norm::Symbol = :orthonormal
    cs_phase::Bool = true
    real_norm::Bool = false
    # SHTns' setter arguments name intrinsic rotations in call order, while
    # the coefficient kernel consumes the outer Z factors in reverse order.
    # Direct constructors retain the package's historical matrix convention.
    reverse_outer::Bool = false
end

# Convenient outer constructor with keyword defaults (to mirror SHTns usage)
function SHTRotation(lmax::Integer, mmax::Integer; α::Real=0.0, β::Real=0.0,
                     γ::Real=0.0, conv::Symbol=:ZYZ,
                     norm::Symbol=:orthonormal, cs_phase::Bool=true,
                     real_norm::Bool=false)
    lmax ≥ 0 || throw(ArgumentError("lmax must be nonnegative"))
    0 ≤ mmax ≤ lmax || throw(ArgumentError("mmax must satisfy 0 ≤ mmax ≤ lmax"))
    conv in (:ZYZ, :ZXZ) || throw(ArgumentError("conv must be :ZYZ or :ZXZ"))
    norm in (:orthonormal, :fourpi, :schmidt) ||
        throw(ArgumentError("unsupported rotation normalization: $norm"))
    all(isfinite, (α, β, γ)) || throw(ArgumentError("rotation angles must be finite"))
    return SHTRotation(Int(lmax), Int(mmax), float(α), float(β), float(γ),
                       conv, norm, cs_phase, real_norm, false)
end

# Preserve the pre-convention positional constructor generated by the original
# six-field public struct.
SHTRotation(lmax::Integer, mmax::Integer, α::Real, β::Real, γ::Real,
            conv::Symbol) =
    SHTRotation(lmax, mmax; α, β, γ, conv)

"""
    SH_Yrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})

Rotate a real-field SH expansion around the Y-axis by angle `alpha`.
Uses Wigner-d mixing per l; dispatches to the general rotation engine.
"""
function SH_Yrotate(::CPU, cfg::SHTConfig, Qlm::AbstractVector{<:Complex},
                    alpha::Real, Rlm::AbstractVector{<:Complex})
    _require_cpu_storage(:SH_Yrotate, Qlm)
    _require_cpu_storage(:SH_Yrotate, Rlm)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    length(Rlm) == cfg.nlm || throw(DimensionMismatch("Rlm length must be nlm=$(cfg.nlm)"))
    cfg.mres == 1 || throw(ArgumentError(
        "SH_Yrotate requires mres==1 (got mres=$(cfg.mres)); a Y-rotation mixes orders and cannot be represented in an mres-strided layout",
    ))
    r = SHTRotation(cfg.lmax, cfg.mmax)
    shtns_rotation_set_angles_ZYZ(r, 0.0, float(alpha), 0.0)
    canonical = _uses_canonical_convention(cfg) ? Qlm :
        convert_alm_norm!(similar(Qlm), Qlm, cfg; to_internal=true)
    shtns_rotation_apply_real(CPU(), r, canonical, Rlm)
    return _externalize_coefficients!(Rlm, cfg)
end


SH_Yrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real,
           Rlm::AbstractVector{<:Complex}) =
    SH_Yrotate(CPU(), cfg, Qlm, alpha, Rlm)

"""
    SH_Yrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
"""
SH_Yrotate90(::CPU, cfg::SHTConfig, Qlm::AbstractVector{<:Complex},
             Rlm::AbstractVector{<:Complex}) =
    SH_Yrotate(CPU(), cfg, Qlm, π/2, Rlm)
SH_Yrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex},
             Rlm::AbstractVector{<:Complex}) =
    SH_Yrotate90(CPU(), cfg, Qlm, Rlm)

"""
    SH_Xrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Rotate around X-axis by 90 degrees using ZYZ equivalence: Rz(π/2)·Ry(π/2)·Rz(-π/2).
"""
function SH_Xrotate90(::CPU, cfg::SHTConfig, Qlm::AbstractVector{<:Complex},
                      Rlm::AbstractVector{<:Complex})
    _require_cpu_storage(:SH_Xrotate90, Qlm)
    _require_cpu_storage(:SH_Xrotate90, Rlm)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    length(Rlm) == cfg.nlm || throw(DimensionMismatch("Rlm length must be nlm=$(cfg.nlm)"))
    cfg.mres == 1 || throw(ArgumentError(
        "SH_Xrotate90 requires mres==1 (got mres=$(cfg.mres)); an X-rotation mixes orders and cannot be represented in an mres-strided layout",
    ))
    r = SHTRotation(cfg.lmax, cfg.mmax)
    shtns_rotation_set_angles_ZYZ(r, π/2, π/2, -π/2)
    canonical = _uses_canonical_convention(cfg) ? Qlm :
        convert_alm_norm!(similar(Qlm), Qlm, cfg; to_internal=true)
    shtns_rotation_apply_real(CPU(), r, canonical, Rlm)
    return _externalize_coefficients!(Rlm, cfg)
end

SH_Xrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex},
             Rlm::AbstractVector{<:Complex}) =
    SH_Xrotate90(CPU(), cfg, Qlm, Rlm)

"""
    shtns_rotation_set_angle_axis(r::SHTRotation, theta::Real, Vx::Real, Vy::Real, Vz::Real)

Define rotation from angle-axis (theta around vector V).
Angles are set in ZYZ convention.
"""
function shtns_rotation_set_angle_axis(r::SHTRotation, theta::Real, Vx::Real, Vy::Real, Vz::Real)
    all(isfinite, (theta, Vx, Vy, Vz)) ||
        throw(ArgumentError("rotation angle and axis must be finite"))
    θ = float(theta)
    v = collect(float.((Vx, Vy, Vz)))
    n = hypot(v[1], hypot(v[2], v[3]))
    if n == 0
        r.α = 0.0; r.β = 0.0; r.γ = 0.0; r.conv = :ZYZ; r.reverse_outer = false
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
    β = acos(clamp(R33, -one(R33), one(R33)))
    singular_tolerance = sqrt(eps(typeof(θ)))
    if abs(sin(β)) > singular_tolerance
        α = atan(R23, R13)    # atan2(sα*sβ, cα*sβ) = α
        γ = atan(R32, -R31)   # atan2(sβ*sγ, sβ*cγ) = γ
    elseif R33 > 0
        # β ≈ 0: only α+γ is identifiable. Choose γ=0.
        α = atan(R21, R11)
        γ = zero(β)
    else
        # β ≈ π: only α-γ is identifiable. With γ=0 the ZYZ matrix has
        # R11=-cos(α), R12=-sin(α), so extract that difference directly.
        α = atan(-R12, -R11)
        γ = zero(β)
    end
    r.α = α; r.β = β; r.γ = γ; r.conv = :ZYZ; r.reverse_outer = false
    return nothing
end

"""
    wigner_d_matrix!(d::AbstractMatrix{Float64}, l::Int, beta::Float64)

In-place computation of little Wigner-d matrix d^l_{m m'}(β). Writes into the
top-left (2l+1)×(2l+1) block of `d`. Caller must ensure `size(d,1) ≥ 2l+1`.
"""
function wigner_d_matrix!(d::AbstractMatrix{T}, l::Int, beta::Real) where {T<:AbstractFloat}
    l ≥ 0 || throw(ArgumentError("l must be ≥ 0"))
    size(d, 1) ≥ 2l + 1 && size(d, 2) ≥ 2l + 1 ||
        throw(DimensionMismatch("d must contain a (2l+1)×(2l+1) block"))
    # Precompute log-factorials: lg[i+1] = loggamma(i+1) for i in 0:2l.
    lg = T[_loggamma(i + 1) for i in 0:(2l)]
    return wigner_d_matrix!(d, l, T(beta), lg)
end

"""
    wigner_d_matrix!(d, l, beta, lg)

Scratch overload: `lg` is a caller-supplied buffer with `lg[i+1] = loggamma(i+1)`
and `length(lg) ≥ 2l+1`. Lets a per-`l` rotation loop hoist the O(l) log-factorial
table once (sized to `2*lmax+1`) instead of reallocating it every degree.
"""
function wigner_d_matrix!(d::AbstractMatrix{T}, l::Int, beta::Real,
                          lg::AbstractVector{T}) where {T<:AbstractFloat}
    l ≥ 0 || throw(ArgumentError("l must be ≥ 0"))
    length(lg) ≥ 2l + 1 || throw(ArgumentError("lg must have length ≥ 2l+1"))
    size(d, 1) ≥ 2l + 1 && size(d, 2) ≥ 2l + 1 ||
        throw(DimensionMismatch("d must contain a (2l+1)×(2l+1) block"))
    cb = cos(T(beta)/T(2))
    sb = sin(T(beta)/T(2))
    for m in -l:l
        for mp in -l:l
            kmin = max(0, m - mp)
            kmax = min(l + m, l - mp)
            logpref = 0.5*(lg[l+m+1] + lg[l-m+1] + lg[l+mp+1] + lg[l-mp+1])
            s = zero(T)
            for k in kmin:kmax
                logden = lg[l+m-k+1] + lg[k+1] + lg[mp-m+k+1] + lg[l-mp-k+1]
                p = 2l + m - mp - 2k
                q = mp - m + 2k
                term = (isodd(k) ? -one(T) : one(T)) *
                    exp(logpref - logden) * (cb^p) * (sb^q)
                s += term
            end
            d[m + l + 1, mp + l + 1] = s
        end
    end
    return d
end

"""
    wigner_d_matrix(l::Int, beta::Float64) -> Matrix{Float64}

Compute little Wigner-d matrix d^l_{m m'}(β) with m,m' in [-l..l], returned as a
(2l+1)×(2l+1) real matrix where index is `m+l+1, m'+l+1`.
"""
function wigner_d_matrix(l::Int, beta::T) where {T<:AbstractFloat}
    n = 2l + 1
    d = Matrix{T}(undef, n, n)
    return wigner_d_matrix!(d, l, beta)
end

"""
    WignerCache(lmax::Int, β::Real) -> WignerCache

Precompute Wigner-d matrices `d^l(β)` for `l = 0:lmax` and hand them out via
[`wigner_d(cache, l)`]. Reuse across many rotations at fixed β (e.g.
time-stepping), amortizing the per-call construction cost.
"""
struct WignerCache
    β::Float64
    matrices::Vector{Matrix{Float64}}
end

function WignerCache(lmax::Int, β::Real)
    lmax ≥ 0 || throw(ArgumentError("lmax must be ≥ 0"))
    βf = float(β)
    mats = Vector{Matrix{Float64}}(undef, lmax + 1)
    for l in 0:lmax
        mats[l + 1] = wigner_d_matrix(l, βf)
    end
    return WignerCache(βf, mats)
end

"""
    wigner_d(cache::WignerCache, l::Int) -> Matrix{Float64}

Retrieve cached `d^l(β)`. Errors if `l` exceeds the cache's `lmax`.
"""
@inline function wigner_d(cache::WignerCache, l::Int)
    (0 ≤ l < length(cache.matrices)) || throw(ArgumentError("l=$l outside cache (lmax=$(length(cache.matrices) - 1))"))
    return cache.matrices[l + 1]
end

"""
    wigner_d_matrix_deriv(l::Int, beta::Float64) -> Matrix{Float64}

Derivative d/dβ of little Wigner-d matrix d^l_{m m'}(β).
"""
function wigner_d_matrix_deriv(l::Int, beta::T) where {T<:AbstractFloat}
    l ≥ 0 || throw(ArgumentError("l must be ≥ 0"))
    n = 2l + 1
    dβ = Matrix{T}(undef, n, n)
    cb = cos(beta/T(2))
    sb = sin(beta/T(2))
    dcb = -T(0.5) * sb
    dsb =  T(0.5) * cb
    lg = T[_loggamma(i + 1) for i in 0:(2l)]
    for m in -l:l
        for mp in -l:l
            kmin = max(0, m - mp)
            kmax = min(l + m, l - mp)
            logpref = 0.5*(lg[l+m+1] + lg[l-m+1] + lg[l+mp+1] + lg[l-mp+1])
            s = zero(T)
            for k in kmin:kmax
                logden = lg[l+m-k+1] + lg[k+1] + lg[mp-m+k+1] + lg[l-mp-k+1]
                p = 2l + m - mp - 2k
                q = mp - m + 2k
                amp = (isodd(k) ? -one(T) : one(T)) * exp(logpref - logden)
                # derivative of cb^p * sb^q using direct powers (avoids 0/0 at beta=0,pi)
                dterm = zero(T)
                if p != 0
                    dterm += amp * p * dcb * (cb^(p-1)) * (sb^q)
                end
                if q != 0
                    dterm += amp * q * dsb * (cb^p) * (sb^(q-1))
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
    value = Int(norm)
    base = value & 0xff
    normalization = base == 0 ? :orthonormal :
                    base == 1 ? :fourpi :
                    base == 2 ? :schmidt :
                    throw(ArgumentError("unsupported SHTns normalization code: $base"))
    supported = 0xff | (256 * 4) | (256 * 8)
    value & ~supported == 0 || throw(ArgumentError(
        "rotation normalization contains unsupported flags",
    ))
    return SHTRotation(
        Int(lmax), Int(mmax); norm=normalization,
        cs_phase=(value & (256 * 4)) == 0,
        real_norm=(value & (256 * 8)) != 0,
    )
end

"""shtns_rotation_destroy(r::SHTRotation)"""
shtns_rotation_destroy(::SHTRotation) = nothing

"""shtns_rotation_set_angles_ZYZ(r, alpha, beta, gamma)"""
function shtns_rotation_set_angles_ZYZ(r::SHTRotation, alpha::Real, beta::Real, gamma::Real)
    all(isfinite, (alpha, beta, gamma)) || throw(ArgumentError("rotation angles must be finite"))
    r.α = float(alpha); r.β = float(beta); r.γ = float(gamma); r.conv = :ZYZ
    r.reverse_outer = true
    return nothing
end

"""shtns_rotation_set_angles_ZXZ(r, alpha, beta, gamma)"""
function shtns_rotation_set_angles_ZXZ(r::SHTRotation, alpha::Real, beta::Real, gamma::Real)
    all(isfinite, (alpha, beta, gamma)) || throw(ArgumentError("rotation angles must be finite"))
    r.α = float(alpha); r.β = float(beta); r.γ = float(gamma); r.conv = :ZXZ
    r.reverse_outer = true
    return nothing
end

"""
    shtns_rotation_wigner_d_matrix(r::SHTRotation, l::Integer, mx::AbstractVector{<:Real}) -> Int

Fill `mx` (length ≥ (2l+1)^2) with d^l in row-major order. Returns size 2l+1.
"""
function shtns_rotation_wigner_d_matrix(r::SHTRotation, l::Integer, mx::AbstractVector{<:Real})
    l = Int(l)
    0 ≤ l ≤ r.lmax || throw(ArgumentError("l must satisfy 0 ≤ l ≤ lmax"))
    n = 2l + 1
    length(mx) ≥ n*n || throw(DimensionMismatch("mx must have length ≥ (2l+1)^2"))
    T = eltype(mx)
    d = wigner_d_matrix(l, T(r.β))
    @inbounds for i in 1:n, j in 1:n
        mx[(i-1)*n + j] = d[i, j]
    end
    return n
end

"""
    _lmcplx_ybasis_signs(lmax, mmax) -> Vector{Float64}

The diagonal ε relating this package's LM_cplx layout to the Y_l^m basis that
the Wigner-d engine works in: `ε_m = (-1)^m` for `m < 0`, `1` otherwise.

Both signs of m share the SAME P̄_l^{|m|} row in this layout, so a real field
satisfies `a_{-m} = conj(a_m)` with no CS factor — unlike the Y_l^m convention,
whose rule is `a_{-m} = (-1)^m conj(a_m)`. ε is real and self-inverse, so
`ε ∘ P ∘ ε` is the rotation expressed in the packed layout, and because
`|ε x| = |x|` any norm-based loss is unchanged by it (which is why the angle
gradients only need ε applied to their inputs).
"""
function _lmcplx_ybasis_signs(lmax::Integer, mmax::Integer)
    lmax = Int(lmax); mmax = Int(mmax)
    v = ones(Float64, nlm_cplx_calc(lmax, mmax, 1))
    for l in 0:lmax, m in -min(l, mmax):-1
        isodd(m) && (v[LM_cplx_index(lmax, mmax, l, m) + 1] = -1.0)
    end
    return v
end

"""
    shtns_rotation_apply_cplx(r::SHTRotation, Zlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Apply rotation with Euler angles (ZYZ/ZXZ) to complex SH coefficients in LM_cplx packing (mres==1).
"""
@inline function _rotation_zyz_angles(r::SHTRotation, ::Type{T}) where {T<:AbstractFloat}
    all(isfinite, (r.α, r.β, r.γ)) ||
        throw(ArgumentError("rotation angles must be finite"))
    if r.conv === :ZYZ
        return r.reverse_outer ? (T(r.γ), T(r.β), T(r.α)) :
                                 (T(r.α), T(r.β), T(r.γ))
    elseif r.conv === :ZXZ
        return r.reverse_outer ?
            (T(r.γ) - T(pi / 2), T(r.β), T(r.α) + T(pi / 2)) :
            (T(r.α) + T(pi / 2), T(r.β), T(r.γ) - T(pi / 2))
    end
    throw(ArgumentError("rotation convention must be :ZYZ or :ZXZ"))
end

@inline function _rotation_coefficient_scale(r::SHTRotation, l::Int, m::Int)
    scale = norm_scale_from_orthonormal(l, abs(m), r.norm)
    r.real_norm && m != 0 && (scale *= inv(sqrt(2.0)))
    scale *= cs_phase_factor(abs(m), true, r.cs_phase)
    return scale
end

"""Typed flattened Wigner blocks and phases for one immutable rotation."""
function _rotation_host_blocks(r::SHTRotation, ::Type{T}) where {T<:AbstractFloat}
    α, β, γ = _rotation_zyz_angles(r, T)
    offsets = Vector{Int32}(undef, r.lmax + 2)
    total = sum((2l + 1)^2 for l in 0:r.lmax)
    values = Vector{T}(undef, total)
    cursor = 1
    @inbounds for l in 0:r.lmax
        offsets[l + 1] = Int32(cursor)
        d = wigner_d_matrix(l, β)
        for m in -l:l, mp in -l:l
            values[cursor] = d[m + l + 1, mp + l + 1]
            cursor += 1
        end
    end
    offsets[end] = Int32(cursor)
    input_scales = Vector{T}(undef, nlm_cplx_calc(r.lmax, r.mmax, 1))
    output_scales = similar(input_scales)
    @inbounds for l in 0:r.lmax, m in -min(l, r.mmax):min(l, r.mmax)
        k = LM_cplx_index(r.lmax, r.mmax, l, m) + 1
        scale = T(_rotation_coefficient_scale(r, l, m))
        input_scales[k] = scale
        output_scales[k] = inv(scale)
    end
    return (; offsets, values, input_scales, output_scales, alpha=α, gamma=γ)
end

function _rotation_apply_cplx_canonical!(r::SHTRotation,
                                         Zlm::AbstractVector{<:Complex},
                                         Rlm::AbstractVector{<:Complex})
    r.lmax ≥ 0 || return Rlm
    RT = typeof(real(zero(eltype(Rlm))))
    α, β, γ = _rotation_zyz_angles(r, RT)

    # Pre-allocate working arrays at maximum size to avoid per-l allocations
    nmax = 2 * r.lmax + 1
    b = Vector{Complex{RT}}(undef, nmax)
    c = Vector{Complex{RT}}(undef, nmax)
    dl = Matrix{RT}(undef, nmax, nmax)  # Reusable Wigner d-matrix buffer
    lg = RT[_loggamma(i + 1) for i in 0:(2 * r.lmax)]  # hoisted log-factorial table (reused every l)

    # Apply R = diag(e^{-i m α}) * d^l(β) * diag(e^{-i m γ}) for each l
    for l in 0:r.lmax
        mm = min(l, r.mmax)
        n = 2l + 1
        # Build input vector b_m' = e^{-i m' γ} A_{m'} for m' in [-mm..mm].
        #
        # The Wigner-d machinery below is written for the Y_l^m basis, whose
        # Hermitian rule is a_{-m} = (-1)^m conj(a_m). This package's LM_cplx
        # layout is NOT that basis: both signs of m share the SAME P̄_l^{|m|} row,
        # so a real field there satisfies a_{-m} = conj(a_m) with no (-1)^m (see
        # `synthesis_packed_cplx` / `SH_to_lat_cplx`). The two differ by the
        # diagonal ε_m = (-1)^m on m < 0 only, applied on the way in and undone
        # on the way out.
        #
        # Without it this function was a valid rotation in the wrong basis: it
        # stayed unitary (per-l norm preserved), so nothing caught it, but
        # rotating a REAL field produced a complex one — a_{l,0} came back with a
        # non-zero imaginary part. Verified against a spatial-rotation reference.
        # ε is diagonal and commutes with the α/γ phase diagonals, so pure
        # Z-rotations are unaffected.
        fill!(view(b, 1:n), zero(Complex{RT}))
        for mp in -mm:mm
            idx = LM_cplx_index(r.lmax, r.mmax, l, mp) + 1
            εp = (mp < 0 && isodd(mp)) ? -one(RT) : one(RT)
            b[mp + l + 1] = (εp * Zlm[idx]) * cis(-mp * γ)
        end
        # Multiply with d^l(β) — computed in-place into pre-allocated buffer
        wigner_d_matrix!(dl, l, β, lg)
        fill!(view(c, 1:n), zero(Complex{RT}))
        # c_m = sum_{m'} d_{m m'} b_{m'}
        for mi in -l:l
            acc = zero(Complex{RT})
            for mp in -l:l
                acc += dl[mi + l + 1, mp + l + 1] * b[mp + l + 1]
            end
            c[mi + l + 1] = acc
        end
        # Apply phase e^{-i m α} and write back only for allowed |m| ≤ mm,
        # undoing the ε basis change applied when b was built.
        for m in -mm:mm
            idx = LM_cplx_index(r.lmax, r.mmax, l, m) + 1
            εm = (m < 0 && isodd(m)) ? -one(RT) : one(RT)
            Rlm[idx] = εm * (c[m + l + 1] * cis(-m * α))
        end
    end
    return Rlm
end

function shtns_rotation_apply_cplx(::CPU, r::SHTRotation,
                                   Zlm::AbstractVector{<:Complex},
                                   Rlm::AbstractVector{<:Complex})
    _require_cpu_storage(:shtns_rotation_apply_cplx, Zlm)
    _require_cpu_storage(:shtns_rotation_apply_cplx, Rlm)
    length(Zlm) == length(Rlm) || throw(DimensionMismatch("Zlm and Rlm length mismatch"))
    expected = nlm_cplx_calc(r.lmax, r.mmax, 1)
    length(Zlm) == expected || throw(DimensionMismatch("LM_cplx size mismatch"))
    eltype(Zlm) === eltype(Rlm) || throw(ArgumentError(
        "rotation input and output element types must match",
    ))
    source = Base.mightalias(Zlm, Rlm) ? copy(Zlm) : Zlm
    canonical = if r.norm === :orthonormal && !r.real_norm && r.cs_phase
        source
    else
        result = similar(source)
        @inbounds for l in 0:r.lmax, m in -min(l, r.mmax):min(l, r.mmax)
            k = LM_cplx_index(r.lmax, r.mmax, l, m) + 1
            result[k] = _rotation_coefficient_scale(r, l, m) * source[k]
        end
        result
    end
    _rotation_apply_cplx_canonical!(r, canonical, Rlm)
    if !(r.norm === :orthonormal && !r.real_norm && r.cs_phase)
        @inbounds for l in 0:r.lmax, m in -min(l, r.mmax):min(l, r.mmax)
            k = LM_cplx_index(r.lmax, r.mmax, l, m) + 1
            Rlm[k] /= _rotation_coefficient_scale(r, l, m)
        end
    end
    return Rlm
end

shtns_rotation_apply_cplx(r::SHTRotation,
                          Zlm::AbstractVector{<:Complex},
                          Rlm::AbstractVector{<:Complex}) =
    shtns_rotation_apply_cplx(CPU(), r, Zlm, Rlm)

"""
    shtns_rotation_apply_real(r::SHTRotation, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Apply rotation to real-field SH coefficients in packed LM layout (m ≥ 0). Requires `mres==1`.
"""
function shtns_rotation_apply_real(::CPU, r::SHTRotation,
                                   Qlm::AbstractVector{<:Complex},
                                   Rlm::AbstractVector{<:Complex})
    _require_cpu_storage(:shtns_rotation_apply_real, Qlm)
    _require_cpu_storage(:shtns_rotation_apply_real, Rlm)
    expected = nlm_calc(r.lmax, r.mmax, 1)
    length(Qlm) == expected || throw(DimensionMismatch("LM packed size mismatch"))
    length(Rlm) == expected || throw(DimensionMismatch("LM packed size mismatch"))
    eltype(Qlm) === eltype(Rlm) || throw(ArgumentError(
        "rotation input and output element types must match",
    ))
    source = Base.mightalias(Qlm, Rlm) ? copy(Qlm) : Qlm
    # Build LM_cplx array Zlm from real-packed Qlm using this layout's Hermitian
    # rule a_{-m} = conj(a_m) — NO (-1)^m; see the note at the write below and
    # `_lmcplx_ybasis_signs` for why the CS factor lives inside apply_cplx now.
    RT = typeof(real(zero(eltype(Rlm))))
    Z = Vector{Complex{RT}}(undef, nlm_cplx_calc(r.lmax, r.mmax, 1))
    
    # initialize zeros
    fill!(Z, zero(Complex{RT}))
    for l in 0:r.lmax
        mm = min(l, r.mmax)
        # m = 0
        idxp = LM_index(r.lmax, 1, l, 0) + 1
        idxc = LM_cplx_index(r.lmax, r.mmax, l, 0) + 1
        Z[idxc] = source[idxp]
        for m in 1:mm
            idxp = LM_index(r.lmax, 1, l, m) + 1
            idxc_p = LM_cplx_index(r.lmax, r.mmax, l, m) + 1
            idxc_n = LM_cplx_index(r.lmax, r.mmax, l, -m) + 1
            Am = source[idxp]
            Z[idxc_p] = Am
            # LM_cplx here is the P̄_l^{|m|} layout, whose real-field rule carries
            # NO (-1)^m; `shtns_rotation_apply_cplx` now converts to/from the
            # Y_l^m basis itself, so this must not pre-apply that factor too.
            Z[idxc_n] = conj(Am)
        end
    end
    R = similar(Z)
    shtns_rotation_apply_cplx(CPU(), r, Z, R)
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


shtns_rotation_apply_real(r::SHTRotation,
                          Qlm::AbstractVector{<:Complex},
                          Rlm::AbstractVector{<:Complex}) =
    shtns_rotation_apply_real(CPU(), r, Qlm, Rlm)
