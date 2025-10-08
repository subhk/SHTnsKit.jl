"""
Zygote Extension for Reverse-Mode Automatic Differentiation

This extension provides Zygote.jl integration for reverse-mode automatic differentiation 
through SHTnsKit operations. Zygote excels at gradients of scalar-valued functions and 
is particularly well-suited for optimization and machine learning workflows.

Key Features:
- Reverse-mode AD through spherical harmonic transforms
- Custom adjoints (pullback rules) for SHTnsKit operations  
- Efficient gradient computation for energy functionals
- Support for rotation and operator gradients
- Compatible with both distributed and regular arrays

Comparison with ForwardDiff:
- Zygote: Better for many inputs → one output (optimization problems)
- ForwardDiff: Better for few inputs → many outputs (Jacobians)

The extension includes custom @adjoint definitions to ensure proper gradient flow
through complex mathematical operations like rotations and matrix operations.
"""
module SHTnsKitZygoteExt

using Zygote
using SHTnsKit
using GPUArraysCore

const _zygote_gpu_warned = Ref(false)

_identity(x) = x

function _stage_cfg_for_zygote(cfg::SHTnsKit.SHTConfig)
    if SHTnsKit.is_gpu_config(cfg)
        cfg_cpu = deepcopy(cfg)
        cfg_cpu.compute_device = SHTnsKit.CPU
        cfg_cpu.device_backend = :cpu
        if !_zygote_gpu_warned[]
            @warn "Zygote gradients for GPU configs stage through CPU fallbacks"
            _zygote_gpu_warned[] = true
        end
        return cfg_cpu, true
    end
    return cfg, false
end

function _stage_array_for_zygote(x)
    if x isa GPUArraysCore.AbstractGPUArray
        cpu = Array(x)
        restore = y -> begin
            out = similar(x)
            copyto!(out, y)
            out
        end
        return cpu, restore
    else
        return x, _identity
    end
end

# ===== SCALAR FIELD GRADIENTS =====

"""
    zgrad_scalar_energy(cfg, f) -> ∂E/∂f

Zygote gradient of scalar energy E = 0.5 ∫ |f|^2 under spectral transform.

This function uses Zygote's reverse-mode automatic differentiation to compute
the gradient of the scalar energy functional. The computation flows through:
spatial field → spherical harmonic analysis → energy computation → gradient.

Parameters:
- cfg: SHTnsKit configuration
- f: Input scalar field matrix [nlat × nlon]

Returns:
- Gradient matrix of same size as f, computed via reverse-mode AD
"""
function SHTnsKit.zgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, f::AbstractMatrix)
    cfg_cpu, staged = _stage_cfg_for_zygote(cfg)
    f_cpu, restore = _stage_array_for_zygote(f)
    loss(x) = SHTnsKit.energy_scalar(cfg_cpu, SHTnsKit.analysis(cfg_cpu, x))
    grad_cpu = Zygote.gradient(loss, f_cpu)[1]
    grad = restore(grad_cpu)
    return grad
end

# ===== DISTRIBUTED ARRAY SUPPORT =====
# Generic distributed/array wrappers (avoid hard dependency on PencilArrays)
# These provide automatic differentiation support for distributed computing

"""
    zgrad_scalar_energy(cfg, fθφ::AbstractArray) -> ∂E/∂fθφ

Zygote gradient computation for distributed arrays using reverse-mode AD.

This overload works with any AbstractArray type, including distributed arrays
like PencilArrays. Zygote automatically handles the gradient computation through
the distributed array operations without requiring explicit conversions.

The advantage over ForwardDiff here is that Zygote can work directly with
the distributed array types, preserving their structure throughout the
automatic differentiation process.
"""
function SHTnsKit.zgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, fθφ::AbstractArray)
    cfg_cpu, _ = _stage_cfg_for_zygote(cfg)
    x_cpu, restore = _stage_array_for_zygote(fθφ)
    loss(x) = SHTnsKit.energy_scalar(cfg_cpu, SHTnsKit.analysis(cfg_cpu, x))
    grad_cpu = Zygote.gradient(loss, x_cpu)[1]
    return restore(grad_cpu)
end

# ===== VECTOR FIELD GRADIENTS =====

"""
    zgrad_vector_energy(cfg, Vtθφ, Vpθφ) -> (∂E/∂Vt, ∂E/∂Vp)

Zygote gradient of vector field energy for distributed arrays.

This function computes gradients of vector energy functionals using Zygote's
reverse-mode automatic differentiation. The vector energy typically involves
kinetic energy, enstrophy, or other quadratic functionals in fluid dynamics.

Parameters:
- cfg: SHTnsKit configuration
- Vtθφ: Theta component of vector field (distributed array)
- Vpθφ: Phi component of vector field (distributed array)

Returns:
- Tuple of gradient arrays (∂E/∂Vt, ∂E/∂Vp) with same structure as inputs
"""
function SHTnsKit.zgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vtθφ::AbstractArray, Vpθφ::AbstractArray)
    cfg_cpu, _ = _stage_cfg_for_zygote(cfg)
    Xt_cpu, restore_t = _stage_array_for_zygote(Vtθφ)
    Xp_cpu, restore_p = _stage_array_for_zygote(Vpθφ)
    loss(Xt, Xp) = begin
        Slm, Tlm = SHTnsKit.spat_to_SHsphtor(cfg_cpu, Xt, Xp)
        SHTnsKit.energy_vector(cfg_cpu, Slm, Tlm)
    end
    gT_cpu, gP_cpu = Zygote.gradient(loss, Xt_cpu, Xp_cpu)
    return restore_t(gT_cpu), restore_p(gP_cpu)
end

"""
    zgrad_vector_energy(cfg, Vt, Vp) -> (∂E/∂Vt, ∂E/∂Vp)
"""
function SHTnsKit.zgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    cfg_cpu, _ = _stage_cfg_for_zygote(cfg)
    Xt_cpu, restore_t = _stage_array_for_zygote(Vt)
    Xp_cpu, restore_p = _stage_array_for_zygote(Vp)
    loss(Xt, Xp) = begin
        Slm, Tlm = SHTnsKit.spat_to_SHsphtor(cfg_cpu, Xt, Xp)
        SHTnsKit.energy_vector(cfg_cpu, Slm, Tlm)
    end
    gT_cpu, gP_cpu = Zygote.gradient(loss, Xt_cpu, Xp_cpu)
    return restore_t(gT_cpu), restore_p(gP_cpu)
end

"""
    zgrad_enstrophy_Tlm(cfg, Tlm) -> ∂Z/∂Tlm

Zygote gradient of enstrophy with respect to toroidal spectrum Tlm.
"""
function SHTnsKit.zgrad_enstrophy_Tlm(cfg::SHTnsKit.SHTConfig, Tlm::AbstractMatrix)
    cfg_cpu, _ = _stage_cfg_for_zygote(cfg)
    X_cpu, restore = _stage_array_for_zygote(Tlm)
    loss(X) = SHTnsKit.enstrophy(cfg_cpu, X)
    grad_cpu = Zygote.gradient(loss, X_cpu)[1]
    return restore(grad_cpu)
end

"""
    zgrad_rotation_angles_real(cfg, Qlm, α, β, γ) -> (∂L/∂α, ∂L/∂β, ∂L/∂γ)

Gradient of L = 0.5 || R ||^2 where R = rotation_apply_real(cfg, Qlm; α,β,γ) using Zygote.
Assumes mres == 1.
"""
function SHTnsKit.zgrad_rotation_angles_real(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector, α::Real, β::Real, γ::Real)
    lmax, mmax = cfg.lmax, cfg.mmax
    # Forward rotate to get R (used as cotangent in loss 0.5||R||^2)
    R = similar(Qlm)
    r = SHTnsKit.SHTRotation(lmax, mmax; α=float(α), β=float(β), γ=float(γ))
    SHTnsKit.shtns_rotation_apply_real(r, Qlm, R)
    gα = 0.0; gβ = 0.0; gγ = 0.0
    for l in 0:lmax
        mm = min(l, mmax)
        dl = SHTnsKit.wigner_d_matrix(l, float(β))
        ddl = SHTnsKit.wigner_d_matrix_deriv(l, float(β))
        n = 2l + 1
        b = Vector{ComplexF64}(undef, n)
        # Build complex A_m' then b = e^{-i m' γ} A
        for mp in -mm:mm
            idxp = SHTnsKit.LM_index(lmax, 1, l, abs(mp)) + 1
            if mp == 0
                A = Qlm[idxp]
            elseif mp > 0
                A = Qlm[idxp]
            else
                A = (-1)^(-mp) * conj(Qlm[SHTnsKit.LM_index(lmax, 1, l, -mp) + 1])
            end
            b[mp + l + 1] = A * cis(-mp * float(γ))
        end
        c = dl * b
        for m in 0:mm
            idxp = SHTnsKit.LM_index(lmax, 1, l, m) + 1
            Rm = c[m + l + 1] * cis(-m * float(α))
            ȳ = R[idxp]
            gα += real(conj(ȳ) * ((0 - 1im) * m * Rm))
            # β gradient via d'(β)
            sβ = zero(ComplexF64)
            sγ = zero(ComplexF64)
            for mp in -l:l
                sβ += ddl[m + l + 1, mp + l + 1] * b[mp + l + 1]
                sγ += dl[m + l + 1, mp + l + 1] * ((0 - 1im) * mp * b[mp + l + 1])
            end
            gβ += real(conj(ȳ) * (sβ * cis(-m * float(α))))
            gγ += real(conj(ȳ) * (sγ * cis(-m * float(α))))
        end
    end
    return gα, gβ, gγ
end

"""
    zgrad_rotation_angles_cplx(lmax, mmax, Zlm, α, β, γ) -> (∂L/∂α, ∂L/∂β, ∂L/∂γ)

Gradient of L = 0.5 || R ||^2 where R = rotation_apply_cplx(lmax,mmax,Zlm; α,β,γ) using Zygote.
"""
function SHTnsKit.zgrad_rotation_angles_cplx(lmax::Integer, mmax::Integer, Zlm::AbstractVector, α::Real, β::Real, γ::Real)
    lmax = Int(lmax); mmax = Int(mmax)
    R = similar(Zlm)
    r = SHTnsKit.SHTRotation(lmax, mmax; α=float(α), β=float(β), γ=float(γ))
    SHTnsKit.shtns_rotation_apply_cplx(r, Zlm, R)
    gα = 0.0; gβ = 0.0; gγ = 0.0
    for l in 0:lmax
        mm = min(l, mmax)
        dl = SHTnsKit.wigner_d_matrix(l, float(β))
        ddl = SHTnsKit.wigner_d_matrix_deriv(l, float(β))
        n = 2l + 1
        # Build b_m' = e^{-i m' γ} Z_{l,m'} for m' in [-l..l]
        b = Vector{ComplexF64}(undef, n)
        for mp in -l:l
            idx = SHTnsKit.LM_cplx_index(lmax, mmax, l, mp) + 1
            b[mp + l + 1] = Zlm[idx] * cis(-mp * float(γ))
        end
        # c_m = sum_{m'} d_{m m'}(β) b_{m'}
        c = dl * b
        for m in -mm:mm
            idxm = SHTnsKit.LM_cplx_index(lmax, mmax, l, m) + 1
            ȳ = R[idxm]
            Rm = c[m + l + 1] * cis(-m * float(α))
            # α-gradient: -i m R_m
            gα += real(conj(ȳ) * ((0 - 1im) * m * Rm))
            # γ-gradient: through input phase of b
            sγ = zero(ComplexF64)
            for mp in -l:l
                sγ += dl[m + l + 1, mp + l + 1] * ((0 - 1im) * mp * b[mp + l + 1])
            end
            gγ += real(conj(ȳ) * (sγ * cis(-m * float(α))))
            # β-gradient: via d'(β)
            sβ = zero(ComplexF64)
            for mp in -l:l
                sβ += ddl[m + l + 1, mp + l + 1] * b[mp + l + 1]
            end
            gβ += real(conj(ȳ) * (sβ * cis(-m * float(α))))
        end
    end
    return gα, gβ, gγ
end

# -----------------------------
## Zygote-specific adjoints for rotations/operators to ensure gradients are not `nothing`
## These mirror the ChainRules rrules but live here to guarantee Zygote picks them up.

Zygote.@adjoint function SHTnsKit.SH_Zrotate(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    y = SHTnsKit.SH_Zrotate(cfg, Qlm, alpha, Rlm)
    function back(ȳ)
        # Adjoint of SH_Zrotate w.r.t Q under real inner product:
        # If upstream cotangent is conj(R), map Q̄ = conj(A ȳ) to recover Q (A = diag(e^{i m α}))
        tmp = similar(Qlm)
        SHTnsKit.SH_Zrotate(cfg, ȳ, alpha, tmp)
        Q̄ = conj.(tmp)
        dα = 0.0
        for m in 0:cfg.mmax
            (m % cfg.mres == 0) || continue
            for l in m:cfg.lmax
                lm = SHTnsKit.LM_index(cfg.lmax, cfg.mres, l, m) + 1
                Rval = Qlm[lm] * cis(m * alpha)
                dα += real(conj(ȳ[lm]) * ((0 + 1im) * m * Rval))
            end
        end
        return (nothing, Q̄, dα, nothing)
    end
    return y, back
end

Zygote.@adjoint function SHTnsKit.SH_Yrotate(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    y = SHTnsKit.SH_Yrotate(cfg, Qlm, alpha, Rlm)
    function back(ȳ)
        Q̄ = similar(Qlm)
        SHTnsKit.SH_Yrotate(cfg, ȳ, -alpha, Q̄)
        # angle gradient via derivative of Wigner-d at beta=alpha
        dα = 0.0
        lmax, mmax = cfg.lmax, cfg.mmax
        for l in 0:lmax
            mm = min(l, mmax)
            b = zeros(eltype(ȳ), 2l+1)
            for mp in -mm:mm
                idxp = SHTnsKit.LM_index(lmax, 1, l, abs(mp)) + 1
                if mp == 0
                    b[mp + l + 1] = Qlm[idxp]
                elseif mp > 0
                    b[mp + l + 1] = Qlm[idxp]
                    b[-mp + l + 1] = (-1)^mp * conj(Qlm[idxp])
                end
            end
            dd = SHTnsKit.wigner_d_matrix_deriv(l, float(alpha))
            for m in 0:mm
                lm = SHTnsKit.LM_index(lmax, 1, l, m) + 1
                s = zero(eltype(ȳ))
                for mp in -l:l
                    s += dd[m + l + 1, mp + l + 1] * b[mp + l + 1]
                end
                dα += real(conj(ȳ[lm]) * s)
            end
        end
        return (nothing, Q̄, dα, nothing)
    end
    return y, back
end

Zygote.@adjoint function SHTnsKit.SH_mul_mx(cfg::SHTnsKit.SHTConfig, mx::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    y = SHTnsKit.SH_mul_mx(cfg, mx, Qlm, Rlm)
    function back(ȳ)
        lmax = cfg.lmax; mres = cfg.mres
        Q̄ = zeros(eltype(Qlm), length(Qlm))
        mx̄ = zeros(eltype(mx), length(mx))
        @inbounds for lm0 in 0:(cfg.nlm-1)
            l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]
            c_minus = mx[2*lm0 + 1]
            c_plus  = mx[2*lm0 + 2]
            rbar = ȳ[lm0 + 1]
            if l > m && l > 0
                lm_prev = SHTnsKit.LM_index(lmax, mres, l-1, m)
                Q̄[lm_prev + 1] += conj(c_minus) * rbar
                mx̄[2*lm0 + 1] += real(conj(rbar) * Qlm[lm_prev + 1])
            end
            if l < lmax
                lm_next = SHTnsKit.LM_index(lmax, mres, l+1, m)
                Q̄[lm_next + 1] += conj(c_plus) * rbar
                mx̄[2*lm0 + 2] += real(conj(rbar) * Qlm[lm_next + 1])
            end
        end
        return (nothing, mx̄, Q̄, nothing)
    end
    return y, back
end

end # module
