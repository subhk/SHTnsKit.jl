module SHTnsKitAdvancedADExt

using ChainRulesCore
using SHTnsKit
using SHTnsKit: LM_index, LM_cplx_index, wigner_d_matrix
import SHTnsKit: wigner_d_matrix_deriv

    # Helper to ensure array eltype is complex for adjoints when needed
    _to_complex(A) = eltype(A) <: Complex ? A : complex.(A)

    # analysis(cfg, f) :: (nlat×nlon) -> (lmax+1)×(mmax+1)
    # Helper: exact adjoint of analysis (no Hermitian duplication)
    function _adjoint_analysis(cfg::SHTnsKit.SHTConfig, Alm̄)
        nlat, nlon = cfg.nlat, cfg.nlon
        Fφ = Matrix{ComplexF64}(undef, nlat, nlon)
        fill!(Fφ, 0.0 + 0.0im)
        lmax, mmax = cfg.lmax, cfg.mmax
        P = Vector{Float64}(undef, lmax + 1)
        # scaling for adjoint: nlon (adjoint of fft) × scaleφ (2π/nlon) = 2π
        φadj = 2π
        for m in 0:mmax
            col = m + 1
            if cfg.use_plm_tables && length(cfg.plm_tables) == mmax+1
                tbl = cfg.plm_tables[m+1]
                for i in 1:nlat
                    s = 0.0 + 0.0im
                    @inbounds for l in m:lmax
                        s += (cfg.Nlm[l+1, col] * tbl[l+1, i]) * Alm̄[l+1, col]
                    end
                    Fφ[i, col] = φadj * cfg.w[i] * s
                end
            else
                for i in 1:nlat
                    SHTnsKit.Plm_row!(P, cfg.x[i], lmax, m)
                    s = 0.0 + 0.0im
                    @inbounds for l in m:lmax
                        s += (cfg.Nlm[l+1, col] * P[l+1]) * Alm̄[l+1, col]
                    end
                    Fφ[i, col] = φadj * cfg.w[i] * s
                end
            end
            # Do NOT fill negative-m columns: adjoint places mass only in measured bins
        end
        f̄c = SHTnsKit.ifft_phi(Fφ)  # includes 1/nlon scaling already accounted via φadj
        return real.(f̄c)
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis), cfg::SHTnsKit.SHTConfig, f)
        y = SHTnsKit.analysis(cfg, f)
        function pullback(ȳ)
            ȳA = _to_complex(ȳ)
            f̄ = _adjoint_analysis(cfg, ȳA)
            return NoTangent(), NoTangent(), f̄
        end
        return y, pullback
    end

# synthesis(cfg, alm; real_output=true) :: (lmax+1)×(mmax+1) -> (nlat×nlon)
    function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis), cfg::SHTnsKit.SHTConfig, 
                                alm; real_output::Bool=true)
        y = SHTnsKit.synthesis(cfg, alm; real_output)
        function pullback(ȳ)
            ȳA = _to_complex(ȳ)
            alm̄ = SHTnsKit.analysis(cfg, ȳA)
            return NoTangent(), NoTangent(), alm̄, (; real_output=NoTangent())
end
        return y, pullback
end

    # Packed scalar transforms: spat_to_SH, SH_to_spat
    function ChainRulesCore.rrule(::typeof(SHTnsKit.spat_to_SH), cfg::SHTnsKit.SHTConfig, Vr)
        y = SHTnsKit.spat_to_SH(cfg, Vr)
        function pullback(ȳ)
            Vr̄ = SHTnsKit.SH_to_spat(cfg, _to_complex(ȳ))
            return NoTangent(), NoTangent(), Vr̄
end
        return y, pullback
end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_to_spat), cfg::SHTnsKit.SHTConfig, Qlm)
        y = SHTnsKit.SH_to_spat(cfg, Qlm)
        function pullback(ȳ)
            Qlm̄ = SHTnsKit.spat_to_SH(cfg, ȳ)
            return NoTangent(), NoTangent(), Qlm̄
        end
        return y, pullback
    end

    # Vector sphtor transforms
    function ChainRulesCore.rrule(::typeof(SHTnsKit.spat_to_SHsphtor), cfg::SHTnsKit.SHTConfig, Vt, Vp)
        Slm, Tlm = SHTnsKit.spat_to_SHsphtor(cfg, Vt, Vp)
        function pullback(ṠTl)
            Slm̄, Tlm̄ = ṠTl
            Vt̄, Vp̄ = SHTnsKit.SHsphtor_to_spat(cfg, _to_complex(Slm̄), _to_complex(Tlm̄); real_output=true)
            return NoTangent(), NoTangent(), Vt̄, Vp̄
        end
        return (Slm, Tlm), pullback
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.SHsphtor_to_spat), cfg::SHTnsKit.SHTConfig, 
                                Slm, Tlm; real_output::Bool=true)
        Vt, Vp = SHTnsKit.SHsphtor_to_spat(cfg, Slm, Tlm; real_output)
        function pullback(Ṽ)
            Vt̄, Vp̄ = Ṽ
            Slm̄, Tlm̄ = SHTnsKit.spat_to_SHsphtor(cfg, Vt̄, Vp̄)
            return NoTangent(), NoTangent(), Slm̄, Tlm̄, (; real_output=NoTangent())
        end
        return (Vt, Vp), pullback
    end

    # Complex packed
    function ChainRulesCore.rrule(::typeof(SHTnsKit.spat_cplx_to_SH), cfg::SHTnsKit.SHTConfig, z)
        y = SHTnsKit.spat_cplx_to_SH(cfg, z)
        function pullback(ȳ)
            z̄ = SHTnsKit.SH_to_spat_cplx(cfg, _to_complex(ȳ))
            return NoTangent(), NoTangent(), z̄
        end
        return y, pullback
    end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_to_spat_cplx), cfg::SHTnsKit.SHTConfig, alm)
        y = SHTnsKit.SH_to_spat_cplx(cfg, alm)
        function pullback(ȳ)
            alm̄ = SHTnsKit.spat_cplx_to_SH(cfg, _to_complex(ȳ))
            return NoTangent(), NoTangent(), alm̄
        end
        return y, pullback
end

# Rotations: adjoints via inverse/adjoint rotation

function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Zrotate), cfg::SHTnsKit.SHTConfig, Qlm, alpha::Real, Rlm)
    y = SHTnsKit.SH_Zrotate(cfg, Qlm, alpha, Rlm)
    function pullback(ȳ)
        # Adjoint w.r.t Q under real inner product: Q̄ = conj(A ȳ)
        tmp = similar(Qlm)
        SHTnsKit.SH_Zrotate(cfg, ȳ, alpha, tmp)
        Q̄ = conj.(tmp)
        # angle gradient: dR/dα = i m R
        dα = 0.0
        for m in 0:cfg.mmax
            (m % cfg.mres == 0) || continue
            for l in m:cfg.lmax
                lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
                # R = Q * e^{i m α}
                Rval = Qlm[lm] * cis(m * alpha)
                dα += real(conj(ȳ[lm]) * ((0 + 1im) * m * Rval))
            end
        end
        return NoTangent(), NoTangent(), Q̄, dα, ZeroTangent()
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Yrotate), cfg::SHTnsKit.SHTConfig, Qlm, alpha::Real, Rlm)
    y = SHTnsKit.SH_Yrotate(cfg, Qlm, alpha, Rlm)
    function pullback(ȳ)
        Q̄ = similar(Qlm)
        SHTnsKit.SH_Yrotate(cfg, ȳ, -alpha, Q̄)
        # angle gradient via d/dβ of Wigner-d at β=alpha
        dα = 0.0
        lmax, mmax = cfg.lmax, cfg.mmax
        for l in 0:lmax
            mm = min(l, mmax)
            b = zeros(eltype(ȳ), 2l+1)
            # b = A because γ=0, A from packed Qlm
            for mp in -mm:mm
                idxp = LM_index(lmax, 1, l, abs(mp)) + 1
                # reconstruct complex A using hermitian symmetry for real field
                if mp == 0
                    b[mp + l + 1] = Qlm[idxp]
                elseif mp > 0
                    b[mp + l + 1] = Qlm[idxp]
                    b[-mp + l + 1] = (-1)^mp * conj(Qlm[idxp])
                end
            end
            dd = wigner_d_matrix_deriv(l, float(alpha))
            # ∂R_m = (dd * b)_m for m>=0 (no left/right phases)
            for m in 0:mm
                lm = LM_index(lmax, 1, l, m) + 1
                s = zero(eltype(ȳ))
                for mp in -l:l
                    s += dd[m + l + 1, mp + l + 1] * b[mp + l + 1]
                end
                dα += real(conj(ȳ[lm]) * s)
            end
        end
        return NoTangent(), NoTangent(), Q̄, dα, ZeroTangent()
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Yrotate90), cfg::SHTnsKit.SHTConfig, Qlm, Rlm)
    y = SHTnsKit.SH_Yrotate90(cfg, Qlm, Rlm)
    function pullback(ȳ)
        Q̄ = similar(Qlm)
        SHTnsKit.SH_Yrotate(cfg, ȳ, -π/2, Q̄)
        return NoTangent(), NoTangent(), Q̄, ZeroTangent()
    end
    return y, pullback
end

    function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_Xrotate90), cfg::SHTnsKit.SHTConfig, Qlm, Rlm)
        y = SHTnsKit.SH_Xrotate90(cfg, Qlm, Rlm)
        function pullback(ȳ)
            Q̄ = similar(Qlm)
            # Inverse of Xrotate90 is rotation by -90 around X: Rz(-π/2)·Ry(-π/2)·Rz(π/2)
            r = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax)
            SHTnsKit.shtns_rotation_set_angles_ZYZ(r, -π/2, -π/2, π/2)
            tmp = similar(Qlm)
            SHTnsKit.shtns_rotation_apply_real(r, ȳ, tmp)
            copyto!(Q̄, tmp)
            return NoTangent(), NoTangent(), Q̄, ZeroTangent()
        end
        return y, pullback
    end

# Adjoint for complex rotation using conjugate-transpose of Wigner-D
function ChainRulesCore.rrule(::typeof(SHTnsKit.shtns_rotation_apply_cplx), r::SHTnsKit.SHTRotation, Zlm, Rlm)
    y = SHTnsKit.shtns_rotation_apply_cplx(r, Zlm, Rlm)
    function pullback(ȳ)
        lmax, mmax = r.lmax, r.mmax
        Z̄ = similar(Zlm)
        fill!(Z̄, zero(eltype(Z̄)))
        α, β, γ = r.α, r.β, r.γ
        gα = 0.0; gβ = 0.0; gγ = 0.0
        for l in 0:lmax
            mm = min(l, mmax)
            n = 2l + 1
            # c̄_m = e^{+i m α} ȳ_m
            cbar = zeros(eltype(Zlm), n)
            for m in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, m) + 1
                cbar[m + l + 1] = ȳ[idx] * cis(m * α)
                # α gradient uses -i m R_m -> inner product conj(ȳ_m) * (-i m R_m)
                # R_m = e^{-i m α} c_m
                # We need c_m; recompute below after d multiplication
            end
            # b̄ = d^T(β) c̄
            dl = wigner_d_matrix(l, β)
            bbar = zeros(eltype(Zlm), n)
            for mp in -l:l
                s = zero(eltype(Zlm))
                for m in -l:l
                    s += dl[m + l + 1, mp + l + 1] * cbar[m + l + 1]
                end
                bbar[mp + l + 1] = s
            end
            # compute forward intermediates for angle grads
            b = zeros(eltype(Zlm), n)
            for mp in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, mp) + 1
                b[mp + l + 1] = Zlm[idx] * cis(-mp * γ)
            end
            c = dl * b
            # α-grad: sum_m conj(ȳ_m) * (-i m) R_m = real(sum conj(ȳ_m) * (-i m) e^{-i m α} c_m )
            for m in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, m) + 1
                Rm = c[m + l + 1] * cis(-m * α)
                gα += real(conj(ȳ[idx]) * ((0 - 1im) * m * Rm))
            end
            # γ-grad: sum_m conj(ȳ_m) * phaseL * d * (-i m') b_{m'}
            gγ_l = 0.0
            for m in -mm:mm
                idxm = LM_cplx_index(lmax, mmax, l, m) + 1
                s = zero(eltype(Zlm))
                for mp in -l:l
                    s += dl[m + l + 1, mp + l + 1] * ((0 - 1im) * mp * b[mp + l + 1])
                end
                gγ_l += real(conj(ȳ[idxm]) * (s * cis(-m * α)))
            end
            gγ += gγ_l
            # β-grad: use derivative d'(β)
            ddl = wigner_d_matrix_deriv(l, β)
            gβ_l = 0.0
            for m in -mm:mm
                idxm = LM_cplx_index(lmax, mmax, l, m) + 1
                s = zero(eltype(Zlm))
                for mp in -l:l
                    s += ddl[m + l + 1, mp + l + 1] * b[mp + l + 1]
                end
                gβ_l += real(conj(ȳ[idxm]) * (s * cis(-m * α)))
            end
            gβ += gβ_l
            # Ā_m' = e^{+i m' γ} b̄_m'
            for mp in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, mp) + 1
                Z̄[idx] += bbar[mp + l + 1] * cis(mp * γ)
            end
        end
        rt = Tangent{SHTnsKit.SHTRotation}(; α=gα, β=gβ, γ=gγ)
        return NoTangent(), rt, Z̄, ZeroTangent()
    end
    return y, pullback
end

# Adjoint for real packed rotation: extend to full, apply cplx adjoint, fold back
function ChainRulesCore.rrule(::typeof(SHTnsKit.shtns_rotation_apply_real), r::SHTnsKit.SHTRotation, Qlm, Rlm)
    y = SHTnsKit.shtns_rotation_apply_real(r, Qlm, Rlm)
    function pullback(ȳ)
        lmax, mmax = r.lmax, r.mmax
        # Extend cotangent on packed to full complex
        Zbar_full = zeros(eltype(Qlm), SHTnsKit.nlm_cplx_calc(lmax, mmax, 1))
        for l in 0:lmax
            mm = min(l, mmax)
            # m = 0
            idxp0 = LM_index(lmax, 1, l, 0) + 1
            idxc0 = LM_cplx_index(lmax, mmax, l, 0) + 1
            Zbar_full[idxc0] = ȳ[idxp0]
            for m in 1:mm
                idxp = LM_index(lmax, 1, l, m) + 1
                idxc = LM_cplx_index(lmax, mmax, l, m) + 1
                Zbar_full[idxc] = ȳ[idxp]
                # negative m gets zero from packing adjoint
            end
        end
        # Compute adjoint of complex rotation (transpose of Wigner-d matrix)
        Z̄ = zeros(eltype(Zbar_full), length(Zbar_full))
        α, β, γ = r.α, r.β, r.γ
        for l in 0:lmax
            mm = min(l, mmax)
            n = 2l + 1
            cbar = zeros(eltype(Z̄), n)
            for m in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, m) + 1
                cbar[m + l + 1] = Zbar_full[idx] * cis(m * α)
            end
            dl = wigner_d_matrix(l, β)
            bbar = zeros(eltype(Z̄), n)
            for mp in -l:l
                s = zero(eltype(Z̄))
                for m in -l:l
                    s += dl[m + l + 1, mp + l + 1] * cbar[m + l + 1]
                end
                bbar[mp + l + 1] = s
            end
            for mp in -mm:mm
                idx = LM_cplx_index(lmax, mmax, l, mp) + 1
                Z̄[idx] += bbar[mp + l + 1] * cis(mp * γ)
            end
        end
        # Angle gradients (pack-domain contribution using m≥0 only)
        gα = 0.0; gβ = 0.0; gγ = 0.0
        for l in 0:lmax
            mm = min(l, mmax)
            dl = wigner_d_matrix(l, r.β)
            b = zeros(eltype(Z̄), 2l + 1)
            for mp in -mm:mm
                idx = SHTnsKit.LM_cplx_index(lmax, mmax, l, mp) + 1
                b[mp + l + 1] = (SHTnsKit.LM_cplx_index(lmax, mmax, l, mp) >= 0) ? (begin
                    # reconstruct from packed Qlm
                    if mp == 0
                        Qlm[SHTnsKit.LM_index(lmax, 1, l, 0) + 1]
                    elseif mp > 0
                        Qlm[SHTnsKit.LM_index(lmax, 1, l, mp) + 1]
                    else
                        (-1)^(-mp) * conj(Qlm[SHTnsKit.LM_index(lmax, 1, l, -mp) + 1])
                    end
                end) : 0
                b[mp + l + 1] *= cis(-mp * r.γ)
            end
            c = dl * b
            for m in 0:mm
                idxp = SHTnsKit.LM_index(lmax, 1, l, m) + 1
                Rm = c[m + l + 1] * cis(-m * r.α)
                gα += real(conj(ȳ[idxp]) * ((0 - 1im) * m * Rm))
                # β
                ddl = wigner_d_matrix_deriv(l, r.β)
                sβ = zero(eltype(Z̄))
                sγ = zero(eltype(Z̄))
                for mp in -l:l
                    sβ += ddl[m + l + 1, mp + l + 1] * b[mp + l + 1]
                    sγ += dl[m + l + 1, mp + l + 1] * ((0 - 1im) * mp * b[mp + l + 1])
                end
                gβ += real(conj(ȳ[idxp]) * (sβ * cis(-m * r.α)))
                gγ += real(conj(ȳ[idxp]) * (sγ * cis(-m * r.α)))
            end
        end
        # Fold back to packed positive-m: q̄(m) = Z̄(m) + (-1)^m conj(Z̄(-m))
        Q̄ = zeros(eltype(Qlm), length(Qlm))
        for l in 0:lmax
            mm = min(l, mmax)
            # m=0
            idxp0 = LM_index(lmax, 1, l, 0) + 1
            idxc0 = LM_cplx_index(lmax, mmax, l, 0) + 1
            Q̄[idxp0] = Z̄[idxc0]
            for m in 1:mm
                idxp = LM_index(lmax, 1, l, m) + 1
                idxc_p = LM_cplx_index(lmax, mmax, l, m) + 1
                idxc_n = LM_cplx_index(lmax, mmax, l, -m) + 1
                Q̄[idxp] = Z̄[idxc_p] + (-1)^m * conj(Z̄[idxc_n])
            end
        end
        rt = Tangent{SHTnsKit.SHTRotation}(; α=gα, β=gβ, γ=gγ)
        return NoTangent(), rt, Q̄, ZeroTangent()
    end
    return y, pullback
end

# Operator application: SH_mul_mx(cfg, mx, Qlm, Rlm)
# Forward: R[lm0] = mx[2*lm_prev+2]*Q[lm_prev] + mx[2*lm_next+1]*Q[lm_next]
# where lm_prev = LM_index(l-1,m) and lm_next = LM_index(l+1,m)
function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_mul_mx), cfg::SHTnsKit.SHTConfig, mx, Qlm, Rlm)
    y = SHTnsKit.SH_mul_mx(cfg, mx, Qlm, Rlm)
    function pullback(ȳ)
        lmax = cfg.lmax; mres = cfg.mres
        Q̄ = zeros(eltype(Qlm), length(Qlm))
        mx̄ = zeros(eltype(mx), length(mx))
        @inbounds for lm0 in 0:(cfg.nlm-1)
            l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]
            rbar = ȳ[lm0 + 1]
            # Contribution from lower neighbor Y_{l-1}^m (uses mx[2*lm_prev + 2])
            if l > m && l > 0
                lm_prev = LM_index(lmax, mres, l-1, m)
                c_from_below = mx[2*lm_prev + 2]  # b_{l-1}^m coefficient
                Q̄[lm_prev + 1] += c_from_below * rbar  # mx is real, no conj needed
                mx̄[2*lm_prev + 2] += real(conj(rbar) * Qlm[lm_prev + 1])
            end
            # Contribution from upper neighbor Y_{l+1}^m (uses mx[2*lm_next + 1])
            if l < lmax
                lm_next = LM_index(lmax, mres, l+1, m)
                c_from_above = mx[2*lm_next + 1]  # a_{l+1}^m coefficient
                Q̄[lm_next + 1] += c_from_above * rbar  # mx is real, no conj needed
                mx̄[2*lm_next + 1] += real(conj(rbar) * Qlm[lm_next + 1])
            end
        end
        return NoTangent(), NoTangent(), mx̄, Q̄, ZeroTangent()
    end
    return y, pullback
end

end # module
