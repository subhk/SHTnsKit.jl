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
function SHTnsKit.dist_analysis_packed(cfg::SHTnsKit.SHTConfig,
                                       fθφ::PencilArray;
                                       ltr::Integer=cfg.lmax,
                                       use_rfft::Bool=false)
    comm = communicator(fθφ)
    _validate_cfg_replicated(cfg, comm)
    _collective_validation_error(
        comm, eltype(fθφ) <: Real ? UInt32(0) : UInt32(0x0400),
        :dist_analysis_packed,
    )
    spectral = dist_analysis_pencil(cfg, fθφ; use_rfft, ltr)
    lcap = _collective_truncation(comm, ltr, cfg.lmax, :dist_analysis_packed)
    l_globals = collect(Int, globalindices(spectral, 1))
    m_globals = collect(Int, globalindices(spectral, 2))
    coefficients = parent(spectral)
    active_count = sum(lcap - m + 1 for m in 0:cfg.mres:min(cfg.mmax, lcap))
    local_active = zeros(eltype(spectral), active_count)
    active_index = 0
    @inbounds for m in 0:cfg.mres:min(cfg.mmax, lcap)
        j = findfirst(==(m + 1), m_globals)
        for l in m:lcap
            active_index += 1
            if j !== nothing
                i = findfirst(==(l + 1), l_globals)
                i === nothing || (local_active[active_index] = coefficients[i, j])
            end
        end
    end
    _record_pencil_scalar_stat!(
        :analysis_packed_max_message_elements, length(local_active); maximum=true,
    )
    MPI.Allreduce!(local_active, +, comm)

    packed = zeros(eltype(spectral), cfg.nlm)
    active_index = 0
    @inbounds for m in 0:cfg.mres:min(cfg.mmax, lcap), l in m:lcap
        active_index += 1
        packed[SHTnsKit.LM_index(cfg.lmax, cfg.mres, l, m) + 1] =
            local_active[active_index]
    end
    return packed
end

"""
    dist_synthesis_packed(cfg, Qlm::AbstractVector{<:Complex}; prototype_θφ, real_output=true)
"""
function SHTnsKit.dist_synthesis_packed(cfg::SHTnsKit.SHTConfig,
                                        Qlm::AbstractVector{<:Complex};
                                        prototype_θφ::PencilArray,
                                        real_output::Bool=true,
                                        ltr::Integer=cfg.lmax)
    comm = communicator(prototype_θφ)
    _validate_cfg_replicated(cfg, comm)
    lcap = _collective_truncation(comm, ltr, cfg.lmax, :dist_synthesis_packed)
    flags = UInt32(0)
    length(Qlm) == cfg.nlm || (flags |= 0x0001)
    code = _scalar_precision_code(eltype(Qlm))
    code in (2, 4) || (flags |= 0x0004)
    MPI.Allreduce(code, min, comm) == MPI.Allreduce(code, max, comm) ||
        (flags |= 0x0004)
    _collective_validation_error(comm, flags, :dist_synthesis_packed)

    reference = copy(Qlm)
    MPI.Bcast!(reference, 0, comm)
    mismatch = MPI.Allreduce(!isequal(Qlm, reference), |, comm)
    mismatch && throw(ArgumentError(
        "dist_synthesis_packed requires coefficients replicated identically on every rank",
    ))
    # Shared with the serial twin `synthesis_packed`; `unpack_lm` carries the
    # `m % mres == 0` stride that `LM_index` requires.
    Alm = SHTnsKit.unpack_lm(cfg, Qlm)
    @inbounds for m in 0:cfg.mmax, l in max(m, lcap + 1):cfg.lmax
        Alm[l + 1, m + 1] = zero(eltype(Alm))
    end
    return SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output)
end

"""
    dist_analysis_packed_cplx(cfg, z::PencilArray) -> alm_packed (LM_cplx)
"""
function SHTnsKit.dist_analysis_packed_cplx(cfg::SHTnsKit.SHTConfig, z::PencilArray)
    comm = communicator(z)
    _validate_cfg_replicated(cfg, comm)
    _validate_scalar_pencil!(
        cfg, z, (cfg.nlat, cfg.nlon), :dist_analysis_packed_cplx; comm,
    )
    _collective_validation_error(
        comm, cfg.mres == 1 ? UInt32(0) : UInt32(0x0200),
        :dist_analysis_packed_cplx,
    )
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
    alm_p = Vector{eltype(Aplus)}(undef, SHTnsKit.nlm_cplx_calc(lmax, mmax, 1))
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
    comm = communicator(prototype_θφ)
    _validate_cfg_replicated(cfg, comm)
    _validate_scalar_pencil!(
        cfg, prototype_θφ, (cfg.nlat, cfg.nlon), :dist_synthesis_packed_cplx;
        comm,
    )
    lmax, mmax = cfg.lmax, cfg.mmax
    expected = SHTnsKit.nlm_cplx_calc(lmax, mmax, 1)
    flags = UInt32(0)
    cfg.mres == 1 || (flags |= 0x0200)
    length(alm_packed) == expected || (flags |= 0x0001)
    code = _scalar_precision_code(eltype(alm_packed))
    code in (2, 4) || (flags |= 0x0004)
    MPI.Allreduce(code, min, comm) == MPI.Allreduce(code, max, comm) ||
        (flags |= 0x0004)
    _collective_validation_error(comm, flags, :dist_synthesis_packed_cplx)

    reference = copy(alm_packed)
    MPI.Bcast!(reference, 0, comm)
    MPI.Allreduce(!isequal(alm_packed, reference), |, comm) && throw(ArgumentError(
        "dist_synthesis_packed_cplx requires coefficients replicated identically on every rank",
    ))
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
    CT = complex(float(real(eltype(alm_packed))))
    Aplus  = zeros(CT, lmax+1, mmax+1)
    Aminus = zeros(CT, lmax+1, mmax+1)
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

# ===== ORDINARY SAME-NAME SCALAR VARIANTS ON DISTRIBUTED STORAGE =====

function _validate_variant_vector!(cfg::SHTnsKit.SHTConfig, values::PencilArray,
                                   expected_length::Int, operation::Symbol;
                                   require_real::Bool=false,
                                   require_complex::Bool=false,
                                   peer=nothing)
    comm = communicator(values)
    _validate_cfg_replicated(cfg, comm)
    flags = UInt32(0)
    size_global(values) == (expected_length, 1) || (flags |= 0x0001)
    ranges = PencilArrays.range_local(pencil(values))
    size(parent(values)) == (length(ranges[1]), length(ranges[2])) ||
        (flags |= 0x0002)
    PencilArrays.decomposition(pencil(values)) == (1,) || (flags |= 0x0002)
    code = _scalar_precision_code(eltype(values))
    code == 0 && (flags |= 0x0004)
    require_real && !(eltype(values) <: Real) && (flags |= 0x0400)
    require_complex && !(eltype(values) <: Complex) && (flags |= 0x0004)
    MPI.Allreduce(code, min, comm) == MPI.Allreduce(code, max, comm) ||
        (flags |= 0x0004)
    if peer !== nothing
        peer_comm = communicator(peer)
        compatible = MPI.Comm_size(peer_comm) == MPI.Comm_size(comm) &&
                     MPI.Comm_compare(peer_comm, comm) in (MPI.IDENT, MPI.CONGRUENT)
        compatible || (flags |= 0x0008)
    end
    _collective_validation_error(comm, flags, operation)
    return comm
end

function _distributed_vector(::Type{T}, n::Int, comm) where {T}
    pen = Pencil((n, 1), (1,), comm)
    result = PencilArray{T}(undef, pen)
    fill!(parent(result), zero(T))
    return result
end

function _validate_packed_synthesis_prototype!(
        cfg::SHTnsKit.SHTConfig, prototype::PencilArray,
        coefficients::PencilArray, operation::Symbol;
        complex_output::Bool)
    comm = communicator(coefficients)
    _validate_scalar_pencil!(
        cfg, prototype, (cfg.nlat, cfg.nlon), operation;
        comm, peer=coefficients, require_complex_input=complex_output,
    )
    flags = UInt32(0)
    decomposition = PencilArrays.decomposition(pencil(prototype))
    decomposition in ((1,), (2,), (1, 2)) || (flags |= 0x0002)
    prototype_code = _scalar_precision_code(eltype(prototype))
    coefficient_code = _scalar_precision_code(eltype(coefficients))
    expected_prototype_code = complex_output ? coefficient_code : coefficient_code - 1
    prototype_code == expected_prototype_code || (flags |= 0x0004)
    complex_output || eltype(prototype) <: Real || (flags |= 0x0400)
    _collective_validation_error(comm, flags, operation)
    return nothing
end

function _rank_ranges(values::PencilArray, comm, dimension::Int=1)
    globals = collect(Int, globalindices(values, dimension))
    first_index = isempty(globals) ? 1 : first(globals)
    return MPI.Allgather(first_index, comm), MPI.Allgather(length(globals), comm)
end

@inline _owns_index(first_index::Int, count::Int, index::Int) =
    count > 0 && first_index <= index < first_index + count

function _exchange_owner_values(send_chunks::Vector{Vector{T}}, comm) where {T}
    nranks = MPI.Comm_size(comm)
    length(send_chunks) == nranks || throw(ArgumentError("one send chunk per rank required"))
    send_counts = Cint[length(chunk) for chunk in send_chunks]
    recv_counts = Vector{Cint}(undef, nranks)
    MPI.Alltoall!(
        MPI.UBuffer(send_counts, 1), MPI.UBuffer(recv_counts, 1), comm,
    )
    send = reduce(vcat, send_chunks; init=T[])
    receive = Vector{T}(undef, sum(recv_counts))
    MPI.Alltoallv!(
        MPI.VBuffer(send, send_counts), MPI.VBuffer(receive, recv_counts), comm,
    )
    return receive, Int.(recv_counts), Int.(send_counts)
end

@inline function _packed_lm_pairs(cfg::SHTnsKit.SHTConfig, lcap::Int,
                                  first_index::Int, count::Int)
    last_index = first_index + count - 1
    pairs = Tuple{Int,Int,Int}[]
    @inbounds for m in 0:cfg.mres:min(cfg.mmax, lcap), l in m:lcap
        packed_index = SHTnsKit.LM_index(cfg.lmax, cfg.mres, l, m) + 1
        first_index ≤ packed_index ≤ last_index &&
            push!(pairs, (packed_index, l, m))
    end
    return pairs
end

function _pack_spectral_pencil(cfg::SHTnsKit.SHTConfig, spectral::PencilArray,
                               lcap::Int)
    comm = communicator(spectral)
    output = _distributed_vector(eltype(spectral), cfg.nlm, comm)
    starts, counts = _rank_ranges(output, comm)
    lstarts, lcounts = _rank_ranges(spectral, comm, 1)
    mstarts, mcounts = _rank_ranges(spectral, comm, 2)
    rank = MPI.Comm_rank(comm) + 1
    chunks = [eltype(spectral)[] for _ in 1:MPI.Comm_size(comm)]
    @inbounds for destination in eachindex(chunks)
        for (_, l, m) in _packed_lm_pairs(
                cfg, lcap, starts[destination], counts[destination])
            _owns_index(lstarts[rank], lcounts[rank], l + 1) || continue
            _owns_index(mstarts[rank], mcounts[rank], m + 1) || continue
            push!(chunks[destination], parent(spectral)[
                l + 2 - lstarts[rank], m + 2 - mstarts[rank],
            ])
        end
    end
    receive, recv_counts, send_counts = _exchange_owner_values(chunks, comm)
    _record_pencil_scalar_stat!(
        :analysis_packed_max_message_elements, maximum(send_counts; init=0);
        maximum=true,
    )
    _record_pencil_scalar_stat!(:analysis_packed_sent_elements, sum(send_counts))
    offset = 0
    @inbounds for source in eachindex(recv_counts)
        for (packed_index, l, m) in _packed_lm_pairs(
                cfg, lcap, starts[rank], counts[rank])
            _owns_index(lstarts[source], lcounts[source], l + 1) || continue
            _owns_index(mstarts[source], mcounts[source], m + 1) || continue
            offset += 1
            parent(output)[packed_index - starts[rank] + 1, 1] = receive[offset]
        end
    end
    offset == length(receive) || error("packed analysis owner-map mismatch")
    return output
end

function _unpack_spectral_pencil(cfg::SHTnsKit.SHTConfig, packed::PencilArray,
                                 lcap::Int)
    comm = communicator(packed)
    spectral = PencilArray{eltype(packed)}(
        undef, SHTnsKit.create_spectral_pencil(cfg; comm),
    )
    fill!(parent(spectral), zero(eltype(spectral)))
    lstarts, lcounts = _rank_ranges(spectral, comm, 1)
    mstarts, mcounts = _rank_ranges(spectral, comm, 2)
    pstarts, pcounts = _rank_ranges(packed, comm, 1)
    rank = MPI.Comm_rank(comm) + 1
    chunks = [eltype(packed)[] for _ in 1:MPI.Comm_size(comm)]
    @inbounds for destination in eachindex(chunks)
        for (packed_index, l, m) in _packed_lm_pairs(
                cfg, lcap, pstarts[rank], pcounts[rank])
            _owns_index(lstarts[destination], lcounts[destination], l + 1) || continue
            _owns_index(mstarts[destination], mcounts[destination], m + 1) || continue
            push!(chunks[destination], parent(packed)[
                packed_index - pstarts[rank] + 1, 1,
            ])
        end
    end
    receive, recv_counts, send_counts = _exchange_owner_values(chunks, comm)
    _record_pencil_scalar_stat!(
        :synthesis_packed_max_message_elements, maximum(send_counts; init=0);
        maximum=true,
    )
    _record_pencil_scalar_stat!(:synthesis_packed_sent_elements, sum(send_counts))
    offset = 0
    @inbounds for source in eachindex(recv_counts)
        for (_, l, m) in _packed_lm_pairs(
                cfg, lcap, pstarts[source], pcounts[source])
            _owns_index(lstarts[rank], lcounts[rank], l + 1) || continue
            _owns_index(mstarts[rank], mcounts[rank], m + 1) || continue
            offset += 1
            parent(spectral)[
                l + 2 - lstarts[rank], m + 2 - mstarts[rank],
            ] = receive[offset]
        end
    end
    offset == length(receive) || error("packed synthesis owner-map mismatch")
    return spectral
end

function SHTnsKit.analysis_packed(cfg::SHTnsKit.SHTConfig, field::PencilArray;
                                  use_rfft::Bool=false)
    return SHTnsKit.analysis_packed_l(cfg, field, cfg.lmax; use_rfft)
end

function SHTnsKit.analysis_packed_l(cfg::SHTnsKit.SHTConfig, field::PencilArray,
                                    ltr::Integer; use_rfft::Bool=false)
    comm = communicator(field)
    _validate_cfg_replicated(cfg, comm)
    _collective_validation_error(
        comm, eltype(field) <: Real ? UInt32(0) : UInt32(0x0400),
        :analysis_packed_l,
    )
    lcap = _collective_truncation(comm, ltr, cfg.lmax, :analysis_packed_l)
    spectral = dist_analysis_pencil(cfg, field; use_rfft, ltr=lcap)
    return _pack_spectral_pencil(cfg, spectral, lcap)
end

function SHTnsKit.synthesis_packed(cfg::SHTnsKit.SHTConfig,
                                   coefficients::PencilArray;
                                   prototype_θφ::PencilArray,
                                   use_rfft::Bool=false)
    return SHTnsKit.synthesis_packed_l(
        cfg, coefficients, cfg.lmax; prototype_θφ, use_rfft,
    )
end

function SHTnsKit.synthesis_packed_l(cfg::SHTnsKit.SHTConfig,
                                     coefficients::PencilArray, ltr::Integer;
                                     prototype_θφ::PencilArray,
                                     use_rfft::Bool=false)
    comm = _validate_variant_vector!(
        cfg, coefficients, cfg.nlm, :synthesis_packed_l;
        require_complex=true, peer=prototype_θφ,
    )
    _validate_packed_synthesis_prototype!(
        cfg, prototype_θφ, coefficients, :synthesis_packed_l;
        complex_output=false,
    )
    lcap = _collective_truncation(comm, ltr, cfg.lmax, :synthesis_packed_l)
    spectral = _unpack_spectral_pencil(cfg, coefficients, lcap)
    local_result = SHTnsKit.dist_synthesis(
        cfg, spectral; prototype_θφ, real_output=true, use_rfft,
    )
    output = PencilArray{eltype(local_result)}(undef, pencil(prototype_θφ))
    copyto!(parent(output), local_result)
    return output
end

function _complex_packed_entries(cfg::SHTnsKit.SHTConfig,
                                 first_index::Int, count::Int,
                                 lcap::Int=cfg.lmax)
    last_index = first_index + count - 1
    entries = Tuple{Int,Int,Int}[]
    @inbounds for l in 0:lcap
        for m in -min(l, cfg.mmax):min(l, cfg.mmax)
            packed_index = SHTnsKit.LM_cplx_index(
                cfg.lmax, cfg.mmax, l, m,
            ) + 1
            first_index ≤ packed_index ≤ last_index &&
                push!(entries, (packed_index, l, m))
        end
    end
    return entries
end

function _pack_complex_spectral_pencils(cfg::SHTnsKit.SHTConfig,
                                        Aplus::PencilArray,
                                        Aminus::PencilArray,
                                        lcap::Int=cfg.lmax)
    comm = communicator(Aplus)
    expected = SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    output = _distributed_vector(eltype(Aplus), expected, comm)
    starts, counts = _rank_ranges(output, comm)
    lstarts, lcounts = _rank_ranges(Aplus, comm, 1)
    mstarts, mcounts = _rank_ranges(Aplus, comm, 2)
    rank = MPI.Comm_rank(comm) + 1
    chunks = [eltype(Aplus)[] for _ in 1:MPI.Comm_size(comm)]
    @inbounds for destination in eachindex(chunks)
        for (_, l, m) in _complex_packed_entries(
                cfg, starts[destination], counts[destination], lcap)
            _owns_index(lstarts[rank], lcounts[rank], l + 1) || continue
            _owns_index(mstarts[rank], mcounts[rank], abs(m) + 1) || continue
            i = l + 2 - lstarts[rank]
            j = abs(m) + 2 - mstarts[rank]
            push!(chunks[destination], m < 0 ? conj(parent(Aminus)[i, j]) :
                                               parent(Aplus)[i, j])
        end
    end
    receive, recv_counts, send_counts = _exchange_owner_values(chunks, comm)
    _record_pencil_scalar_stat!(
        :analysis_packed_max_message_elements, maximum(send_counts; init=0);
        maximum=true,
    )
    _record_pencil_scalar_stat!(:analysis_packed_sent_elements, sum(send_counts))
    offset = 0
    @inbounds for source in eachindex(recv_counts)
        for (packed_index, l, m) in _complex_packed_entries(
                cfg, starts[rank], counts[rank], lcap)
            _owns_index(lstarts[source], lcounts[source], l + 1) || continue
            _owns_index(mstarts[source], mcounts[source], abs(m) + 1) || continue
            offset += 1
            parent(output)[packed_index - starts[rank] + 1, 1] = receive[offset]
        end
    end
    offset == length(receive) || error("complex packed analysis owner-map mismatch")
    return output
end

function SHTnsKit.analysis_packed_cplx(cfg::SHTnsKit.SHTConfig,
                                       field::PencilArray)
    return SHTnsKit.analysis_packed_cplx_l(cfg, field, cfg.lmax)
end

function SHTnsKit.analysis_packed_cplx_l(cfg::SHTnsKit.SHTConfig,
                                         field::PencilArray,
                                         ltr::Integer)
    comm = communicator(field)
    _validate_cfg_replicated(cfg, comm)
    _collective_validation_error(
        comm, cfg.mres == 1 ? UInt32(0) : UInt32(0x0200),
        :analysis_packed_cplx_l,
    )
    lcap = _collective_truncation(
        comm, ltr, cfg.lmax, :analysis_packed_cplx_l,
    )
    _validate_scalar_pencil!(
        cfg, field, (cfg.nlat, cfg.nlon), :analysis_packed_cplx_l;
        comm, require_complex_input=true,
    )
    RT = real(eltype(field))
    real_field = PencilArray{RT}(undef, pencil(field))
    imag_field = PencilArray{RT}(undef, pencil(field))
    parent(real_field) .= real.(parent(field))
    parent(imag_field) .= imag.(parent(field))
    analyzed_real = dist_analysis_pencil(cfg, real_field; ltr=lcap)
    analyzed_imag = dist_analysis_pencil(cfg, imag_field; ltr=lcap)
    Aplus = similar(analyzed_real)
    Aminus = similar(analyzed_real)
    parent(Aplus) .= parent(analyzed_real) .+ im .* parent(analyzed_imag)
    parent(Aminus) .= parent(analyzed_real) .- im .* parent(analyzed_imag)
    return _pack_complex_spectral_pencils(cfg, Aplus, Aminus, lcap)
end

function _unpack_complex_spectral_pencils(cfg::SHTnsKit.SHTConfig,
                                          packed::PencilArray,
                                          lcap::Int=cfg.lmax)
    comm = communicator(packed)
    spectral_pen = SHTnsKit.create_spectral_pencil(cfg; comm)
    Aplus = PencilArray{eltype(packed)}(undef, spectral_pen)
    Aminus = PencilArray{eltype(packed)}(undef, spectral_pen)
    fill!(parent(Aplus), zero(eltype(Aplus)))
    fill!(parent(Aminus), zero(eltype(Aminus)))
    lstarts, lcounts = _rank_ranges(Aplus, comm, 1)
    mstarts, mcounts = _rank_ranges(Aplus, comm, 2)
    pstarts, pcounts = _rank_ranges(packed, comm, 1)
    rank = MPI.Comm_rank(comm) + 1
    chunks = [eltype(packed)[] for _ in 1:MPI.Comm_size(comm)]
    @inbounds for destination in eachindex(chunks)
        for (packed_index, l, signed_m) in _complex_packed_entries(
                cfg, pstarts[rank], pcounts[rank], lcap)
            m = abs(signed_m)
            _owns_index(lstarts[destination], lcounts[destination], l + 1) || continue
            _owns_index(mstarts[destination], mcounts[destination], m + 1) || continue
            value = parent(packed)[packed_index - pstarts[rank] + 1, 1]
            push!(chunks[destination], signed_m < 0 ? conj(value) : value)
        end
    end
    receive, recv_counts, send_counts = _exchange_owner_values(chunks, comm)
    _record_pencil_scalar_stat!(
        :synthesis_packed_max_message_elements, maximum(send_counts; init=0);
        maximum=true,
    )
    _record_pencil_scalar_stat!(:synthesis_packed_sent_elements, sum(send_counts))
    offset = 0
    @inbounds for source in eachindex(recv_counts)
        for (_, l, signed_m) in _complex_packed_entries(
                cfg, pstarts[source], pcounts[source], lcap)
            m = abs(signed_m)
            _owns_index(lstarts[rank], lcounts[rank], l + 1) || continue
            _owns_index(mstarts[rank], mcounts[rank], m + 1) || continue
            offset += 1
            i = l + 2 - lstarts[rank]
            j = m + 2 - mstarts[rank]
            if signed_m < 0
                parent(Aminus)[i, j] = receive[offset]
            else
                parent(Aplus)[i, j] = receive[offset]
            end
        end
    end
    offset == length(receive) || error("complex packed synthesis owner-map mismatch")
    return Aplus, Aminus
end

function SHTnsKit.synthesis_packed_cplx(cfg::SHTnsKit.SHTConfig,
                                        coefficients::PencilArray;
                                        prototype_θφ::PencilArray)
    return SHTnsKit.synthesis_packed_cplx_l(
        cfg, coefficients, cfg.lmax; prototype_θφ,
    )
end

function SHTnsKit.synthesis_packed_cplx_l(cfg::SHTnsKit.SHTConfig,
                                          coefficients::PencilArray,
                                          ltr::Integer;
                                          prototype_θφ::PencilArray)
    comm = communicator(coefficients)
    _validate_cfg_replicated(cfg, comm)
    _collective_validation_error(
        comm, cfg.mres == 1 ? UInt32(0) : UInt32(0x0200),
        :synthesis_packed_cplx_l,
    )
    expected = SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    _validate_variant_vector!(
        cfg, coefficients, expected, :synthesis_packed_cplx_l;
        require_complex=true, peer=prototype_θφ,
    )
    _validate_packed_synthesis_prototype!(
        cfg, prototype_θφ, coefficients, :synthesis_packed_cplx_l;
        complex_output=true,
    )
    lcap = _collective_truncation(
        comm, ltr, cfg.lmax, :synthesis_packed_cplx_l,
    )
    Aplus, Aminus = _unpack_complex_spectral_pencils(
        cfg, coefficients, lcap,
    )
    local_result = SHTnsKit.dist_synthesis(
        cfg, Aplus; prototype_θφ, real_output=false, Aminus,
    )
    output = PencilArray{eltype(local_result)}(undef, pencil(prototype_θφ))
    copyto!(parent(output), local_result)
    return output
end

function _analysis_mode_pencil(cfg::SHTnsKit.SHTConfig, im::Int,
                               field::PencilArray, ltr::Int;
                               axisymmetric::Bool=false)
    comm = communicator(field)
    _validate_cfg_replicated(cfg, comm)
    im_min = MPI.Allreduce(im, min, comm)
    im_max = MPI.Allreduce(im, max, comm)
    flags = UInt32(0)
    im_min == im_max || (flags |= 0x0100)
    0 ≤ im ≤ cfg.mmax ÷ cfg.mres || (flags |= 0x0100)
    m = im * cfg.mres
    m ≤ ltr ≤ cfg.lmax || (flags |= 0x0100)
    axisymmetric && !(eltype(field) <: Real) && (flags |= 0x0400)
    _collective_validation_error(comm, flags, :analysis_packed_ml)
    _validate_variant_vector!(
        cfg, field, cfg.nlat, axisymmetric ? :analysis_axisym_l : :analysis_packed_ml;
        require_real=axisymmetric, require_complex=!axisymmetric,
    )

    CT = complex(float(real(eltype(field))))
    output_length = axisymmetric ? ltr + 1 : ltr - m + 1
    output = _distributed_vector(CT, output_length, comm)
    starts, counts = _rank_ranges(output, comm)
    θglobals = collect(Int, globalindices(field, 1))
    RT = typeof(real(zero(CT)))
    P = Vector{Float64}(undef, ltr + 1)
    rank = MPI.Comm_rank(comm)
    phi_scale = axisymmetric ? cfg.cphi * cfg.nlon : cfg.cphi
    for root in 0:(MPI.Comm_size(comm) - 1)
        send = zeros(CT, counts[root + 1])
        @inbounds for (i, θindex) in pairs(θglobals)
            SHTnsKit.Plm_norm_row!(P, cfg.x[θindex], ltr, m)
            weighted = CT(parent(field)[i, 1]) * RT(cfg.w[θindex])
            for k in eachindex(send)
                qindex = starts[root + 1] + k - 1
                l = axisymmetric ? qindex - 1 : m + qindex - 1
                send[k] += weighted * RT(P[l + 1])
            end
        end
        send .*= RT(phi_scale)
        receive = similar(send)
        MPI.Reduce!(send, receive, +, root, comm)
        if rank == root
            @inbounds for k in eachindex(receive)
                qindex = starts[root + 1] + k - 1
                l = axisymmetric ? qindex - 1 : m + qindex - 1
                scale = RT(SHTnsKit.coefficient_scale_to_canonical(cfg, l, m))
                parent(output)[k, 1] = receive[k] / scale
            end
        end
    end
    return output
end

function _synthesis_mode_pencil(cfg::SHTnsKit.SHTConfig, im::Int,
                                coefficients::PencilArray, ltr::Int;
                                axisymmetric::Bool=false)
    comm = communicator(coefficients)
    _validate_cfg_replicated(cfg, comm)
    im_min = MPI.Allreduce(im, min, comm)
    im_max = MPI.Allreduce(im, max, comm)
    flags = UInt32(0)
    im_min == im_max || (flags |= 0x0100)
    0 ≤ im ≤ cfg.mmax ÷ cfg.mres || (flags |= 0x0100)
    m = im * cfg.mres
    m ≤ ltr ≤ cfg.lmax || (flags |= 0x0100)
    _collective_validation_error(comm, flags, :synthesis_packed_ml)
    expected = axisymmetric ? ltr + 1 : ltr - m + 1
    _validate_variant_vector!(
        cfg, coefficients, expected,
        axisymmetric ? :synthesis_axisym_l : :synthesis_packed_ml;
        require_complex=true,
    )

    CT = eltype(coefficients)
    RT = typeof(real(zero(CT)))
    output_type = axisymmetric ? RT : CT
    output = _distributed_vector(output_type, cfg.nlat, comm)
    θstarts, θcounts = _rank_ranges(output, comm)
    qglobals = collect(Int, globalindices(coefficients, 1))
    P = Vector{Float64}(undef, ltr + 1)
    rank = MPI.Comm_rank(comm)
    inverse_scale = axisymmetric ? one(RT) : RT(SHTnsKit.phi_inv_scale(cfg))
    for root in 0:(MPI.Comm_size(comm) - 1)
        send = zeros(CT, θcounts[root + 1])
        @inbounds for k in eachindex(send)
            θindex = θstarts[root + 1] + k - 1
            SHTnsKit.Plm_norm_row!(P, cfg.x[θindex], ltr, m)
            for (i, qindex) in pairs(qglobals)
                l = axisymmetric ? qindex - 1 : m + qindex - 1
                scale = RT(SHTnsKit.coefficient_scale_to_canonical(cfg, l, m))
                send[k] += RT(P[l + 1]) * scale * parent(coefficients)[i, 1]
            end
            send[k] *= inverse_scale
        end
        receive = similar(send)
        MPI.Reduce!(send, receive, +, root, comm)
        if rank == root
            @inbounds for k in eachindex(receive)
                parent(output)[k, 1] = axisymmetric ? real(receive[k]) : receive[k]
            end
        end
    end
    return output
end

SHTnsKit.analysis_axisym(cfg::SHTnsKit.SHTConfig, field::PencilArray) =
    _analysis_mode_pencil(cfg, 0, field, cfg.lmax; axisymmetric=true)

function SHTnsKit.analysis_axisym_l(cfg::SHTnsKit.SHTConfig,
                                    field::PencilArray, ltr::Integer)
    comm = communicator(field)
    lcap = _collective_truncation(comm, ltr, cfg.lmax, :analysis_axisym_l)
    return _analysis_mode_pencil(cfg, 0, field, lcap; axisymmetric=true)
end

function SHTnsKit.synthesis_axisym(cfg::SHTnsKit.SHTConfig,
                                   coefficients::PencilArray)
    return _synthesis_mode_pencil(
        cfg, 0, coefficients, cfg.lmax; axisymmetric=true,
    )
end

function SHTnsKit.synthesis_axisym_l(cfg::SHTnsKit.SHTConfig,
                                     coefficients::PencilArray, ltr::Integer)
    comm = communicator(coefficients)
    lcap = _collective_truncation(comm, ltr, cfg.lmax, :synthesis_axisym_l)
    return _synthesis_mode_pencil(
        cfg, 0, coefficients, lcap; axisymmetric=true,
    )
end

function SHTnsKit.analysis_packed_ml(cfg::SHTnsKit.SHTConfig, im::Int,
                                     field::PencilArray, ltr::Integer)
    comm = communicator(field)
    lcap = _collective_truncation(comm, ltr, cfg.lmax, :analysis_packed_ml)
    return _analysis_mode_pencil(cfg, im, field, lcap)
end

function SHTnsKit.synthesis_packed_ml(cfg::SHTnsKit.SHTConfig, im::Int,
                                      coefficients::PencilArray, ltr::Integer)
    comm = communicator(coefficients)
    lcap = _collective_truncation(comm, ltr, cfg.lmax, :synthesis_packed_ml)
    return _synthesis_mode_pencil(cfg, im, coefficients, lcap)
end

function _validate_batch_pencil!(cfg::SHTnsKit.SHTConfig, values::PencilArray,
                                 expected_spatial::Tuple{Int,Int},
                                 operation::Symbol; require_real::Bool=false,
                                 require_complex::Bool=false, peer=nothing)
    comm = communicator(values)
    _validate_cfg_replicated(cfg, comm)
    globals = size_global(values)
    flags = UInt32(0)
    length(globals) == 3 || (flags |= 0x0001)
    length(globals) == 3 && globals[1:2] != expected_spatial && (flags |= 0x0001)
    length(globals) == 3 && globals[3] < 1 && (flags |= 0x0001)
    code = _scalar_precision_code(eltype(values))
    code == 0 && (flags |= 0x0004)
    require_real && !(eltype(values) <: Real) && (flags |= 0x0400)
    require_complex && !(eltype(values) <: Complex) && (flags |= 0x0004)
    MPI.Allreduce(code, min, comm) == MPI.Allreduce(code, max, comm) ||
        (flags |= 0x0004)
    nfields = length(globals) == 3 ? globals[3] : 0
    MPI.Allreduce(nfields, min, comm) == MPI.Allreduce(nfields, max, comm) ||
        (flags |= 0x0001)
    if peer !== nothing
        peer_comm = communicator(peer)
        compatible = MPI.Comm_size(peer_comm) == MPI.Comm_size(comm) &&
                     MPI.Comm_compare(peer_comm, comm) in (MPI.IDENT, MPI.CONGRUENT)
        compatible || (flags |= 0x0008)
        peer_globals = size_global(peer)
        (length(peer_globals) == 3 && length(globals) == 3 &&
         peer_globals[3] == globals[3]) || (flags |= 0x0001)
    end
    _collective_validation_error(comm, flags, operation)
    return nfields
end

function SHTnsKit.analysis_batch(cfg::SHTnsKit.SHTConfig,
                                 fields::PencilArray{T,3};
                                 use_rfft::Bool=false) where {T<:Real}
    nfields = _validate_batch_pencil!(
        cfg, fields, (cfg.nlat, cfg.nlon), :analysis_batch; require_real=true,
    )
    comm = communicator(fields)
    output = PencilArray{complex(float(T))}(
        undef, SHTnsKit.create_spectral_pencil(cfg; comm), nfields,
    )
    for k in 1:nfields
        field = PencilArray{T}(undef, pencil(fields))
        @views copyto!(parent(field), parent(fields)[:, :, k])
        transformed = dist_analysis_pencil(cfg, field; use_rfft)
        @views copyto!(parent(output)[:, :, k], parent(transformed))
    end
    return output
end

function SHTnsKit.analysis_batch!(cfg::SHTnsKit.SHTConfig,
                                  output::PencilArray{TO,3},
                                  fields::PencilArray{TI,3};
                                  use_rfft::Bool=false, fft_batch=nothing) where
                                  {TO<:Complex,TI<:Real}
    comm = communicator(fields)
    _validate_batch_pencil!(
        cfg, output, (cfg.lmax + 1, cfg.mmax + 1), :analysis_batch_output;
        require_complex=true, peer=fields,
    )
    expected = PencilArray{TO}(
        undef, SHTnsKit.create_spectral_pencil(cfg; comm), size_global(fields)[3],
    )
    _validate_identical_pencil_layout!(
        expected, output, :analysis_batch_output_layout,
    )
    MPI.Allreduce(fft_batch === nothing ? 0 : 1, +, comm) == 0 ||
        throw(ArgumentError(
        "distributed analysis_batch! owns its per-call Fourier scratch",
    ))
    transformed = SHTnsKit.analysis_batch(cfg, fields; use_rfft)
    copyto!(parent(output), parent(transformed))
    return output
end

function SHTnsKit.synthesis_batch(cfg::SHTnsKit.SHTConfig,
                                  coefficients::PencilArray{T,3};
                                  prototype_θφ::PencilArray,
                                  real_output::Bool=true,
                                  use_rfft::Bool=false) where {T<:Complex}
    nfields = _validate_batch_pencil!(
        cfg, coefficients, (cfg.lmax + 1, cfg.mmax + 1), :synthesis_batch;
        require_complex=true, peer=prototype_θφ,
    )
    prototype_fields = _validate_batch_pencil!(
        cfg, prototype_θφ, (cfg.nlat, cfg.nlon), :synthesis_batch_prototype;
        peer=coefficients,
    )
    prototype_fields == nfields || throw(DimensionMismatch(
        "prototype batch size must match the coefficient batch",
    ))
    output_type = real_output ? real(T) : T
    output = PencilArray{output_type}(undef, pencil(prototype_θφ), nfields)
    for k in 1:nfields
        spectral = PencilArray{T}(undef, pencil(coefficients))
        @views copyto!(parent(spectral), parent(coefficients)[:, :, k])
        prototype = PencilArray{eltype(prototype_θφ)}(
            undef, pencil(prototype_θφ),
        )
        local_result = SHTnsKit.dist_synthesis(
            cfg, spectral; prototype_θφ=prototype, real_output, use_rfft,
        )
        @views copyto!(parent(output)[:, :, k], local_result)
    end
    return output
end

function SHTnsKit.synthesis_batch_cplx(cfg::SHTnsKit.SHTConfig,
                                       coefficients::PencilArray{T,3};
                                       prototype_θφ::PencilArray,
                                       use_rfft::Bool=false) where {T<:Complex}
    return SHTnsKit.synthesis_batch(
        cfg, coefficients; prototype_θφ, real_output=false, use_rfft,
    )
end

function SHTnsKit.synthesis_batch!(cfg::SHTnsKit.SHTConfig,
                                   output::PencilArray,
                                   coefficients::PencilArray{T,3};
                                   prototype_θφ::PencilArray=output,
                                   real_output::Bool=true,
                                   use_rfft::Bool=false, fft_batch=nothing) where
                                   {T<:Complex}
    comm = communicator(coefficients)
    _validate_batch_pencil!(
        cfg, output, (cfg.nlat, cfg.nlon), :synthesis_batch_output;
        peer=coefficients,
    )
    _validate_identical_pencil_layout!(
        prototype_θφ, output, :synthesis_batch_output_layout,
    )
    MPI.Allreduce(fft_batch === nothing ? 0 : 1, +, comm) == 0 ||
        throw(ArgumentError(
        "distributed synthesis_batch! owns its per-call Fourier scratch",
    ))
    transformed = SHTnsKit.synthesis_batch(
        cfg, coefficients; prototype_θφ, real_output, use_rfft,
    )
    copyto!(parent(output), parent(transformed))
    return output
end
