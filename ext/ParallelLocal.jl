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

`Alm_pencil` holds ORTHONORMAL coefficients — the form `dist_analysis` returns,
matching serial `analysis`/`synthesis_point`. It is NOT the packed `SH_to_lat`
convention, which applies the cfg norm/CS scale.
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
    gl_l = collect(Int, globalindices(Alm_pencil, 1))
    gl_m = collect(Int, globalindices(Alm_pencil, 2))
    # m = 0 if present locally
    j0 = findfirst(==(1), gl_m)
    if j0 !== nothing
        SHTnsKit.Plm_norm_row!(P, x, lmax, 0)
        g0 = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval <= ltr
                g0 += P[lval+1] * Alm_pencil[il, mloc[j0]]
            end
        end
        vals_local .+= g0
    end
    # m > 0 columns owned by this rank
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        (mval > 0 && mval <= mtr) || continue
        SHTnsKit.Plm_norm_row!(P, x, lmax, mval)
        gm = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if mval <= lval <= ltr
                gm += P[lval+1] * Alm_pencil[il, jm]
            end
        end
        @inbounds for j in 0:(nphi-1)
            vals_local[j+1] += 2 * real(gm * cis(2π * mval * j / nphi))
        end
    end
    MPI.Allreduce!(vals_local, +, comm)
    return real_output ? real.(vals_local) : vals_local
end

"""
    dist_SH_to_point(cfg, Alm_pencil::PencilArray, cost::Real, phi::Real) -> Float64

Evaluate spherical harmonic expansion at a single point for a real-valued field.
Uses Hermitian symmetry: negative-m contribution added via 2*real(...) for m > 0.

`Alm_pencil` holds orthonormal coefficients (see [`dist_SH_to_lat`](@ref)).
"""
function SHTnsKit.dist_SH_to_point(cfg::SHTnsKit.SHTConfig, Alm_pencil::PencilArray, cost::Real, phi::Real)
    comm = communicator(Alm_pencil)
    lmax, mmax = cfg.lmax, cfg.mmax
    x = float(cost)
    P = Vector{Float64}(undef, lmax + 1)
    lloc = axes(Alm_pencil, 1); mloc = axes(Alm_pencil, 2)
    gl_l = collect(Int, globalindices(Alm_pencil, 1))
    gl_m = collect(Int, globalindices(Alm_pencil, 2))
    s_local = 0.0
    # m=0
    j0 = findfirst(==(1), gl_m)
    if j0 !== nothing
        SHTnsKit.Plm_norm_row!(P, x, lmax, 0)
        g0 = 0.0
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            g0 += P[lval+1] * real(Alm_pencil[il, mloc[j0]])
        end
        s_local += g0
    end
    # m>0: add both +m and -m via 2*real(...)
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval > 0 || continue
        SHTnsKit.Plm_norm_row!(P, x, lmax, mval)
        gm = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= mval
                gm += P[lval+1] * Alm_pencil[il, jm]
            end
        end
        ph = cis(mval * phi)
        s_local += 2 * real(gm * ph)
    end
    s = MPI.Allreduce(s_local, +, comm)
    return s
end

"""
    dist_SHqst_to_point(cfg, Q_p::PencilArray, S_p::PencilArray, T_p::PencilArray, cost, phi) -> (vr, vt, vp)

`Q_p`/`S_p`/`T_p` hold orthonormal coefficients (see [`dist_SH_to_lat`](@ref)).
"""
function SHTnsKit.dist_SHqst_to_point(cfg::SHTnsKit.SHTConfig, Q_p::PencilArray, S_p::PencilArray, T_p::PencilArray, cost::Real, phi::Real)
    comm = communicator(Q_p)
    lmax, mmax = cfg.lmax, cfg.mmax
    x = float(cost)
    P = Vector{Float64}(undef, lmax + 1)
    dPdtheta = Vector{Float64}(undef, lmax + 1)
    P_over_sinth = Vector{Float64}(undef, lmax + 1)
    lloc = axes(Q_p, 1); mloc = axes(Q_p, 2)
    gl_l = collect(Int, globalindices(Q_p, 1))
    gl_m = collect(Int, globalindices(Q_p, 2))
    vr_local = 0.0 + 0.0im
    vt_local = 0.0 + 0.0im
    vp_local = 0.0 + 0.0im
    # m=0
    j0 = findfirst(==(1), gl_m)
    if j0 !== nothing
        SHTnsKit.Plm_norm_and_dPdtheta_row!(P, dPdtheta, x, lmax, 0)
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            Y = P[lval+1]
            dθY = dPdtheta[lval+1]
                aQ = Q_p[il, mloc[j0]]; aS = S_p[il, mloc[j0]]; aT = T_p[il, mloc[j0]]
            vr_local += Y   * aQ
            vt_local += dθY * aS
            vp_local += dθY * aT  # Vφ = dθY * T for m=0
        end
    end
    # m>0 (use pole-safe Legendre functions)
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval > 0 || continue
        SHTnsKit.Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, mval)
        gvr = 0.0 + 0.0im
        gvt = 0.0 + 0.0im
        gvp = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= mval
                Y = P[lval+1]
                dθY = dPdtheta[lval+1]
                Y_over_sθ = P_over_sinth[lval+1]
                aQ = Q_p[il, jm]; aS = S_p[il, jm]; aT = T_p[il, jm]
                gvr += Y   * aQ
                # Vθ = ∂S/∂θ - (im/sinθ) * T
                gvt += dθY * aS - (0 + 1im) * mval * Y_over_sθ * aT
                # Vφ = (im/sinθ) * S + ∂T/∂θ
                gvp += (0 + 1im) * mval * Y_over_sθ * aS + dθY * aT
            end
        end
        ph = cis(mval * phi)
        vr_local += gvr * ph + conj(gvr) * conj(ph)
        vt_local += gvt * ph + conj(gvt) * conj(ph)
        vp_local += gvp * ph + conj(gvp) * conj(ph)
    end
    # One batched collective instead of three separate round-trips (vr,vt,vp).
    red = MPI.Allreduce!(ComplexF64[vr_local, vt_local, vp_local], +, comm)
    return real(red[1]), real(red[2]), real(red[3])
end

"""
    dist_SHqst_to_lat(cfg, Q_p::PencilArray, S_p::PencilArray, T_p::PencilArray, cost::Real;
                      nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) -> Vr, Vt, Vp

`Q_p`/`S_p`/`T_p` hold orthonormal coefficients (see [`dist_SH_to_lat`](@ref)).
"""
function SHTnsKit.dist_SHqst_to_lat(cfg::SHTnsKit.SHTConfig, Q_p::PencilArray, S_p::PencilArray, T_p::PencilArray, cost::Real;
                                    nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    comm = communicator(Q_p)
    lmax = cfg.lmax
    x = float(cost)
    P = Vector{Float64}(undef, lmax + 1)
    dPdtheta = Vector{Float64}(undef, lmax + 1)
    P_over_sinth = Vector{Float64}(undef, lmax + 1)
    lloc = axes(Q_p, 1); mloc = axes(Q_p, 2)
    gl_l = collect(Int, globalindices(Q_p, 1))
    gl_m = collect(Int, globalindices(Q_p, 2))
    Vr_local = zeros(ComplexF64, nphi)
    Vt_local = zeros(ComplexF64, nphi)
    Vp_local = zeros(ComplexF64, nphi)
    # m=0
    j0 = findfirst(==(1), gl_m)
    if j0 !== nothing
        SHTnsKit.Plm_norm_and_dPdtheta_row!(P, dPdtheta, x, lmax, 0)
        g0 = 0.0 + 0.0im; gθ0 = 0.0 + 0.0im; gφ0 = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval <= ltr
                Y = P[lval+1]
                dθY = dPdtheta[lval+1]
                aQ = Q_p[il, mloc[j0]]; aS = S_p[il, mloc[j0]]; aT = T_p[il, mloc[j0]]
                g0  += Y * aQ
                gθ0 += dθY * aS
                gφ0 += dθY * aT  # Vφ = dθY * T for m=0
            end
        end
        Vr_local .+= g0; Vt_local .+= gθ0; Vp_local .+= gφ0
    end
    # m>0 (use pole-safe Legendre functions)
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        (mval > 0 && mval <= mtr) || continue
        SHTnsKit.Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, mval)
        g  = 0.0 + 0.0im
        gθ = 0.0 + 0.0im
        gφ = 0.0 + 0.0im
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if mval <= lval <= ltr
                Y = P[lval+1]
                dθY = dPdtheta[lval+1]
                Y_over_sθ = P_over_sinth[lval+1]
                aQ = Q_p[il, jm]; aS = S_p[il, jm]; aT = T_p[il, jm]
                g  += Y   * aQ
                # Vθ = ∂S/∂θ - (im/sinθ) * T
                gθ += dθY * aS - (0 + 1im) * mval * Y_over_sθ * aT
                # Vφ = (im/sinθ) * S + ∂T/∂θ
                gφ += (0 + 1im) * mval * Y_over_sθ * aS + dθY * aT
            end
        end
        @inbounds for j in 0:(nphi-1)
            ph = cis(2π * mval * j / nphi)
            Vr_local[j+1] += 2 * real(g * ph)
            Vt_local[j+1] += 2 * real(gθ * ph)
            Vp_local[j+1] += 2 * real(gφ * ph)
        end
    end
    # One batched collective over the stacked (Vr,Vt,Vp) buffer instead of three.
    combined = vcat(Vr_local, Vt_local, Vp_local)
    MPI.Allreduce!(combined, +, comm)
    return real.(@view combined[1:nphi]),
           real.(@view combined[nphi+1:2nphi]),
           real.(@view combined[2nphi+1:3nphi])
end

"""
    dist_analysis_packed(cfg, fθφ::PencilArray) -> Qlm packed
"""
function SHTnsKit.dist_analysis_packed(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray)
    Alm = SHTnsKit.dist_analysis(cfg, fθφ)
    # Shared with the serial twin `analysis_packed`; `pack_lm` carries the
    # `m % mres == 0` stride that `LM_index` requires.
    return SHTnsKit.pack_lm(cfg, Alm)
end

"""
    dist_synthesis_packed(cfg, Qlm::AbstractVector{<:Complex}; prototype_θφ, real_output=true)
"""
function SHTnsKit.dist_synthesis_packed(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex}; prototype_θφ::PencilArray, real_output::Bool=true)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length"))
    # Shared with the serial twin `synthesis_packed`; `unpack_lm` carries the
    # `m % mres == 0` stride that `LM_index` requires.
    Alm = SHTnsKit.unpack_lm(cfg, Qlm)
    return SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output)
end

"""
    dist_analysis_packed_cplx(cfg, z::PencilArray) -> alm_packed (LM_cplx)
"""
function SHTnsKit.dist_analysis_packed_cplx(cfg::SHTnsKit.SHTConfig, z::PencilArray)
    cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    lmax, mmax = cfg.lmax, cfg.mmax
    # `dist_analysis` returns the m ≥ 0 columns of exactly this expansion, so the
    # +m half is already correct for a genuinely complex field. The −m half lives
    # in φ-FFT bins `dist_analysis` never returns; recover it from
    #     a_{l,-m}[z] = conj(a_{l,+m}[conj(z)])
    # which holds because the quadrature weights, P̄_l^{|m|} and the norm scale are
    # all real and the φ-FFT of conj(z) maps bin −m onto bin +m.
    #
    # For a REAL field conj(z) == z, so this collapses to a_{l,-m} = conj(a_{l,m}).
    # There is NO (-1)^m: this LM_cplx layout uses the SAME P̄_l^{|m|} row for both
    # signs of m (see `synthesis_packed_cplx` / `SH_to_lat_cplx`), unlike the
    # Y_l^m convention where the Hermitian relation carries that factor.
    #
    # A complex field is split into its real and imaginary parts rather than fed
    # to `dist_analysis` directly: analysis is ℂ-linear in the field, so the two
    # real transforms give BOTH halves at once —
    #     A(z) = A(Re z) + i·A(Im z),   A(conj z) = A(Re z) − i·A(Im z)
    # — and, unlike a complex PencilArray, real data survives the φ-gather path
    # (`_gather_phi_rows` packs into a `Vector{Float64}`, so a ComplexF64 array on
    # a φ-decomposed pencil threw `InexactError` on every rank). A single-pass
    # variant that kept the ±m φ-FFT bins `dist_analysis` discards would halve
    # this again; that needs a distributed analysis returning both halves.
    Aplus, Aminus = if eltype(z) <: Real
        A = SHTnsKit.dist_analysis(cfg, z; use_tables=cfg.use_plm_tables)
        A, A                        # conj(z) == z — one transform covers both
    else
        RT = real(eltype(z))
        pen = pencil(z)
        zr = PencilArray{RT}(undef, pen)
        zi = PencilArray{RT}(undef, pen)
        parent(zr) .= real.(parent(z))
        parent(zi) .= imag.(parent(z))
        Ar = SHTnsKit.dist_analysis(cfg, zr; use_tables=cfg.use_plm_tables)
        Ai = SHTnsKit.dist_analysis(cfg, zi; use_tables=cfg.use_plm_tables)
        (Ar .+ im .* Ai), (Ar .- im .* Ai)
    end
    alm_p = Vector{ComplexF64}(undef, SHTnsKit.nlm_cplx_calc(lmax, mmax, 1))
    for l in 0:lmax
        alm_p[SHTnsKit.LM_cplx_index(lmax, mmax, l, 0) + 1] = Aplus[l+1, 1]
        for m in 1:min(l, mmax)
            alm_p[SHTnsKit.LM_cplx_index(lmax, mmax, l, m) + 1] = Aplus[l+1, m+1]
            alm_p[SHTnsKit.LM_cplx_index(lmax, mmax, l, -m) + 1] = conj(Aminus[l+1, m+1])
        end
    end
    return alm_p
end

"""
    dist_synthesis_packed_cplx(cfg, alm_packed::AbstractVector{<:Complex}; prototype_θφ) -> PencilArray complex field
"""
function SHTnsKit.dist_synthesis_packed_cplx(cfg::SHTnsKit.SHTConfig, alm_packed::AbstractVector{<:Complex}; prototype_θφ::PencilArray)
    cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    lmax, mmax = cfg.lmax, cfg.mmax
    length(alm_packed) == SHTnsKit.nlm_cplx_calc(lmax, mmax, 1) || throw(DimensionMismatch("alm_packed length"))
    # `dist_synthesis` only knows the m ≥ 0 columns of `Alm` and never writes the
    # negative-m DFT bins, so feeding it the +m half alone silently dropped every
    # m < 0 coefficient (serial `synthesis_packed_cplx` writes both bin `am+1`
    # and bin `nlon-am+1`). Recover the −m half from the same ℂ-linearity the
    # analysis twin uses: with
    #     S(A)[θ,φ] = Σ_{m≥0} A[l,m] P̄_l^m(cosθ) e^{imφ}
    # the m < 0 sum is conj(S(conj(A₋))) because P̄ and the norm scale are real —
    # the conjugation flips e^{imφ} to e^{-imφ}. The m = 0 column of A₋ is zeroed
    # so it is not counted twice.
    #
    # There is NO (-1)^m: this layout uses the SAME P̄_l^{|m|} row for both signs
    # of m (see `synthesis_packed_cplx`), unlike the Y_l^m convention.
    Aplus  = zeros(ComplexF64, lmax+1, mmax+1)
    Aminus = zeros(ComplexF64, lmax+1, mmax+1)
    for l in 0:lmax
        Aplus[l+1, 1] = alm_packed[SHTnsKit.LM_cplx_index(lmax, mmax, l, 0) + 1]
        for m in 1:min(l, mmax)
            Aplus[l+1, m+1]  = alm_packed[SHTnsKit.LM_cplx_index(lmax, mmax, l,  m) + 1]
            Aminus[l+1, m+1] = conj(alm_packed[SHTnsKit.LM_cplx_index(lmax, mmax, l, -m) + 1])
        end
    end
    # Single pass: `dist_synthesis` fills the −m φ-FFT bins from `Aminus` in the
    # same θ/m traversal that fills the +m bins, reusing one Legendre row per
    # (m, θ) — P̄_l^{|m|} depends only on |m|. This used to be two full distributed
    # syntheses combined as `zp + conj(zn)`, which doubled the Legendre work, the
    # inverse FFT and, on a φ-distributed pencil, the communication. Same shape as
    # the serial twin `synthesis_packed_cplx` (src/complex_packed.jl:110-130).
    return SHTnsKit.dist_synthesis(cfg, Aplus; prototype_θφ, real_output=false, Aminus)
end
