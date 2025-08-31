##########
# PencilArray rotations
##########

function SHTnsKit.dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig, 
                            Alm_pencil::PencilArray, alpha::Real)
    mloc = axes(Alm_pencil, 2)
    gl_m = globalindices(Alm_pencil, 2)
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        phase = cis(mval * alpha)
        Alm_pencil[:, jm] .*= phase
    end

    return Alm_pencil
end

function SHTnsKit.dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig,
                            Alm_pencil::PencilArray, alpha::Real,
                            R_pencil::PencilArray)
    mloc = axes(Alm_pencil, 2)
    gl_m = globalindices(Alm_pencil, 2)
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        phase = cis(mval * alpha)
        @inbounds R_pencil[:, jm] .= phase .* Alm_pencil[:, jm]
    end
    return R_pencil
end

function SHTnsKit.dist_SH_Yrotate_allgatherm!(cfg::SHTnsKit.SHTConfig, 
                                            Alm_pencil::PencilArray, 
                                            beta::Real, 
                                            R_pencil::PencilArray)

    lmax, mmax = cfg.lmax, cfg.mmax
    
    comm = communicator(Alm_pencil)

    lloc = axes(Alm_pencil, 1)
    mloc = axes(Alm_pencil, 2)
    
    gl_l = globalindices(Alm_pencil, 1)
    gl_m = globalindices(Alm_pencil, 2)

    nm_local = length(mloc)
    counts_m = Allgather(nm_local, comm)
    displs_m = cumsum([0; counts_m[1:end-1]])
    a_full = Vector{ComplexF64}(undef, mmax + 1)

    for (ii, il) in enumerate(lloc)
        lval = gl_l[ii] - 1
        a_local = Array(view(Alm_pencil, il, :))
        Allgatherv(a_local, a_full, counts_m, displs_m, comm)
        mm = min(lval, mmax)
        n2 = 2*lval + 1
        b = Vector{ComplexF64}(undef, n2); fill!(b, 0.0 + 0.0im)
        if lval >= 0
            k0 = SHTnsKit.norm_scale_from_orthonormal(lval, 0, cfg.norm)
            α0 = SHTnsKit.cs_phase_factor(0, true, cfg.cs_phase)
            b[0 + lval + 1] = (k0 * α0) * a_full[1]
        end

        for m in 1:mm
            km = SHTnsKit.norm_scale_from_orthonormal(lval, m, cfg.norm)
            αm = SHTnsKit.cs_phase_factor(m, true, cfg.cs_phase)
            a_int = (km * αm) * a_full[m+1]
            b[m + lval + 1] = a_int
            b[-m + lval + 1] = (-1.0)^m * conj(a_int)
        end

        dl = SHTnsKit.wigner_d_matrix(lval, float(beta))
        c = Vector{ComplexF64}(undef, n2)
        @inbounds for mi in -lval:lval
            acc = 0.0 + 0.0im
            for mp in -lval:lval
                acc += dl[mi + lval + 1, mp + lval + 1] * b[mp + lval + 1]
            end
            c[mi + lval + 1] = acc
        end

        for (jj, jm) in enumerate(mloc)
            mval = gl_m[jj] - 1
            if mval <= lval
                cm = c[mval + lval + 1]
                km = SHTnsKit.norm_scale_from_orthonormal(lval, mval, cfg.norm)
                αm = SHTnsKit.cs_phase_factor(mval, true, cfg.cs_phase)
                R_pencil[il, jm] = cm / (km * αm)
            else
                R_pencil[il, jm] = 0.0 + 0.0im
            end
        end
    end
    
    return R_pencil
end

"""
    dist_SH_Yrotate_truncgatherm!(cfg, Alm_pencil, beta, R_pencil)

Allgather only m-columns with m ≤ l for each l-row, reducing communication for small l.
"""
function SHTnsKit.dist_SH_Yrotate_truncgatherm!(cfg::SHTnsKit.SHTConfig,
                                               Alm_pencil::PencilArray,
                                               beta::Real,
                                               R_pencil::PencilArray)
    lmax, mmax = cfg.lmax, cfg.mmax
    comm = communicator(Alm_pencil)
    lloc = axes(Alm_pencil, 1)
    mloc = axes(Alm_pencil, 2)
    gl_l = globalindices(Alm_pencil, 1)
    gl_m = globalindices(Alm_pencil, 2)

    for (ii, il) in enumerate(lloc)
        lval = gl_l[ii] - 1
        mm = min(lval, mmax)
        # Build local subset for m ≤ lval
        msel_val = Int[]
        a_loc = ComplexF64[]
        for (jj, jm) in enumerate(mloc)
            mval = gl_m[jj] - 1
            (0 <= mval <= mm) || continue
            push!(msel_val, mval)
            push!(a_loc, Alm_pencil[il, jm])
        end
        # Gather sizes
        count_local = length(msel_val)
        counts = Allgather(count_local, comm)
        displs = cumsum([0; counts[1:end-1]])
        total = sum(counts)
        m_all = Vector{Int}(undef, total)
        a_all = Vector{ComplexF64}(undef, total)
        Allgatherv(msel_val, m_all, counts, displs, comm)
        Allgatherv(a_loc, a_all, counts, displs, comm)
        # Reconstruct a_full[0:mm]
        a_full = zeros(ComplexF64, mm + 1)
        for k in 1:total
            mval = m_all[k]
            (0 <= mval <= mm) || continue
            a_full[mval + 1] = a_all[k]
        end
        # Build symmetric b of size 2l+1 from positive m part
        n2 = 2*lval + 1
        b = Vector{ComplexF64}(undef, n2); fill!(b, 0.0 + 0.0im)
        if lval >= 0
            k0 = SHTnsKit.norm_scale_from_orthonormal(lval, 0, cfg.norm)
            α0 = SHTnsKit.cs_phase_factor(0, true, cfg.cs_phase)
            b[0 + lval + 1] = (k0 * α0) * (mm >= 0 ? a_full[1] : 0.0 + 0.0im)
        end
        for m in 1:mm
            km = SHTnsKit.norm_scale_from_orthonormal(lval, m, cfg.norm)
            αm = SHTnsKit.cs_phase_factor(m, true, cfg.cs_phase)
            a_int = (km * αm) * a_full[m+1]
            b[m + lval + 1] = a_int
            b[-m + lval + 1] = (-1.0)^m * conj(a_int)
        end
        # d-matrix multiply
        dl = SHTnsKit.wigner_d_matrix(lval, float(beta))
        c = Vector{ComplexF64}(undef, n2)
        @inbounds for mi in -lval:lval
            acc = 0.0 + 0.0im
            for mp in -lval:lval
                acc += dl[mi + lval + 1, mp + lval + 1] * b[mp + lval + 1]
            end
            c[mi + lval + 1] = acc
        end
        # Write back local columns
        for (jj, jm) in enumerate(mloc)
            mval = gl_m[jj] - 1
            if mval <= lval
                cm = c[mval + lval + 1]
                km = SHTnsKit.norm_scale_from_orthonormal(lval, mval, cfg.norm)
                αm = SHTnsKit.cs_phase_factor(mval, true, cfg.cs_phase)
                R_pencil[il, jm] = cm / (km * αm)
            else
                R_pencil[il, jm] = 0.0 + 0.0im
            end
        end
    end
    return R_pencil
end
function SHTnsKit.dist_SH_Yrotate(cfg::SHTnsKit.SHTConfig,
                                  Alm_pencil::PencilArray,
                                  beta::Real,
                                  R_pencil::PencilArray)
    # Truncated gather reduces bandwidth for small l
    return SHTnsKit.dist_SH_Yrotate_truncgatherm!(cfg, Alm_pencil, beta, R_pencil)
end

"""
    dist_SH_Yrotate90(cfg, Alm_pencil::PencilArray, R_pencil::PencilArray)

Rotate distributed Alm by +90° around Y in Pencil layout.
"""
function SHTnsKit.dist_SH_Yrotate90(cfg::SHTnsKit.SHTConfig,
                                    Alm_pencil::PencilArray,
                                    R_pencil::PencilArray)
    return SHTnsKit.dist_SH_Yrotate(cfg, Alm_pencil, π/2, R_pencil)
end

"""
    dist_SH_Xrotate90(cfg, Alm_pencil::PencilArray, R_pencil::PencilArray)

Rotate distributed Alm by +90° around X using Z(π/2) → Y(π/2) → Z(-π/2).
"""
function SHTnsKit.dist_SH_Xrotate90(cfg::SHTnsKit.SHTConfig,
                                    Alm_pencil::PencilArray,
                                    R_pencil::PencilArray)
    return SHTnsKit.dist_SH_rotate_euler(cfg, Alm_pencil, π/2, π/2, -π/2, R_pencil)
end

##########
# Composite Euler rotation on PencilArrays: Z(α) then Y(β) then Z(γ)
##########

function SHTnsKit.dist_SH_rotate_euler(cfg::SHTnsKit.SHTConfig,
                                       Alm_pencil::PencilArray,
                                       α::Real, β::Real, γ::Real,
                                       R_pencil::PencilArray)
    # Temp buffer with same layout
    tmp1 = similar(Alm_pencil)
    tmp2 = similar(Alm_pencil)
    # Z(α)
    SHTnsKit.dist_SH_Zrotate(cfg, Alm_pencil, α, tmp1)
    # Y(β): requires allgather over m
    SHTnsKit.dist_SH_Yrotate_allgatherm!(cfg, tmp1, β, tmp2)
    # Z(γ)
    SHTnsKit.dist_SH_Zrotate(cfg, tmp2, γ, R_pencil)
    return R_pencil
end

##########
# Convenience wrappers: packed Qlm vectors rotated via distributed Pencil operations
##########

"""
    dist_SH_Zrotate_packed(cfg, Qlm::AbstractVector{<:Complex}, α; prototype_lm::PencilArray) -> Rlm::Vector

Rotate packed real-field Qlm around Z by α using distributed Pencil operations.
"""
function SHTnsKit.dist_SH_Zrotate_packed(cfg::SHTnsKit.SHTConfig,
                                         Qlm::AbstractVector{<:Complex}, α::Real;
                                         prototype_lm::PencilArray)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be $(cfg.nlm)"))
    lmax, mmax = cfg.lmax, cfg.mmax
    # Unpack to dense Alm
    Alm = zeros(ComplexF64, lmax+1, mmax+1)
    @inbounds for m in 0:mmax, l in m:lmax
        Alm[l+1, m+1] = Qlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1]
    end
    # Distribute and rotate
    Alm_p = PencilArray(Alm)
    R_p = allocate(Alm_p; dims=(:l,:m), eltype=ComplexF64)
    SHTnsKit.dist_SH_Zrotate(cfg, Alm_p, α, R_p)
    # Gather to dense
    Rlm_mat = zeros(ComplexF64, lmax+1, mmax+1)
    lloc = axes(R_p, 1); mloc = axes(R_p, 2)
    gl_l = globalindices(R_p, 1)
    gl_m = globalindices(R_p, 2)
    for (ii, il) in enumerate(lloc), (jj, jm) in enumerate(mloc)
        Rlm_mat[gl_l[ii], gl_m[jj]] = R_p[il, jm]
    end
    MPI.Allreduce!(Rlm_mat, +, communicator(R_p))
    # Pack back
    Rlm = similar(Qlm)
    @inbounds for m in 0:mmax, l in m:lmax
        Rlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1] = Rlm_mat[l+1, m+1]
    end
    return Rlm
end

"""
    dist_SH_Yrotate_packed(cfg, Qlm, β; prototype_lm) -> Rlm
"""
function SHTnsKit.dist_SH_Yrotate_packed(cfg::SHTnsKit.SHTConfig,
                                         Qlm::AbstractVector{<:Complex}, β::Real;
                                         prototype_lm::PencilArray)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be $(cfg.nlm)"))
    lmax, mmax = cfg.lmax, cfg.mmax
    Alm = zeros(ComplexF64, lmax+1, mmax+1)
    @inbounds for m in 0:mmax, l in m:lmax
        Alm[l+1, m+1] = Qlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1]
    end
    Alm_p = PencilArray(Alm)
    R_p = allocate(Alm_p; dims=(:l,:m), eltype=ComplexF64)
    SHTnsKit.dist_SH_Yrotate(cfg, Alm_p, β, R_p)
    Rlm_mat = zeros(ComplexF64, lmax+1, mmax+1)
    lloc = axes(R_p, 1); mloc = axes(R_p, 2)
    gl_l = globalindices(R_p, 1)
    gl_m = globalindices(R_p, 2)
    for (ii, il) in enumerate(lloc), (jj, jm) in enumerate(mloc)
        Rlm_mat[gl_l[ii], gl_m[jj]] = R_p[il, jm]
    end
    MPI.Allreduce!(Rlm_mat, +, communicator(R_p))
    Rlm = similar(Qlm)
    @inbounds for m in 0:mmax, l in m:lmax
        Rlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1] = Rlm_mat[l+1, m+1]
    end
    return Rlm
end

"""
    dist_SH_Yrotate90_packed(cfg, Qlm; prototype_lm) -> Rlm
"""
function SHTnsKit.dist_SH_Yrotate90_packed(cfg::SHTnsKit.SHTConfig,
                                           Qlm::AbstractVector{<:Complex};
                                           prototype_lm::PencilArray)
    return SHTnsKit.dist_SH_Yrotate_packed(cfg, Qlm, π/2; prototype_lm)
end

"""
    dist_SH_Xrotate90_packed(cfg, Qlm; prototype_lm) -> Rlm
"""
function SHTnsKit.dist_SH_Xrotate90_packed(cfg::SHTnsKit.SHTConfig,
                                           Qlm::AbstractVector{<:Complex};
                                           prototype_lm::PencilArray)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be $(cfg.nlm)"))
    lmax, mmax = cfg.lmax, cfg.mmax
    Alm = zeros(ComplexF64, lmax+1, mmax+1)
    @inbounds for m in 0:mmax, l in m:lmax
        Alm[l+1, m+1] = Qlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1]
    end
    Alm_p = PencilArray(Alm)
    R_p = allocate(Alm_p; dims=(:l,:m), eltype=ComplexF64)
    SHTnsKit.dist_SH_rotate_euler(cfg, Alm_p, π/2, π/2, -π/2, R_p)
    Rlm_mat = zeros(ComplexF64, lmax+1, mmax+1)
    lloc = axes(R_p, 1); mloc = axes(R_p, 2)
    gl_l = globalindices(R_p, 1)
    gl_m = globalindices(R_p, 2)
    for (ii, il) in enumerate(lloc), (jj, jm) in enumerate(mloc)
        Rlm_mat[gl_l[ii], gl_m[jj]] = R_p[il, jm]
    end
    MPI.Allreduce!(Rlm_mat, +, communicator(R_p))
    Rlm = similar(Qlm)
    @inbounds for m in 0:mmax, l in m:lmax
        Rlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1] = Rlm_mat[l+1, m+1]
    end
    return Rlm
end
