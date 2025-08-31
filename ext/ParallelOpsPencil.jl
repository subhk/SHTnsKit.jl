##########
# PencilArray operators
##########

"""
    dist_apply_laplacian!(cfg, Alm_pencil::PencilArray)

In-place multiply by -l(l+1) for distributed Alm with dims (:l,:m). No communication.
"""
function SHTnsKit.dist_apply_laplacian!(cfg::SHTnsKit.SHTConfig, Alm_pencil::PencilArray)
    lloc = axes(Alm_pencil, 1); gl_l = globalindices(Alm_pencil, 1)
    for (ii, il) in enumerate(lloc)
        lval = gl_l[ii] - 1
        Alm_pencil[il, :] .*= -(lval * (lval + 1))
    end
    return Alm_pencil
end

"""
    dist_SH_mul_mx!(cfg, mx, Alm_pencil::PencilArray, R_pencil::PencilArray)

Apply 3-diagonal operator to distributed Alm pencils using per-m Allgatherv of l-columns.
"""
function SHTnsKit.dist_SH_mul_mx!(cfg::SHTnsKit.SHTConfig, mx::AbstractVector{<:Real}, Alm_pencil::PencilArray, R_pencil::PencilArray)
    lmax, mmax = cfg.lmax, cfg.mmax
    comm = communicator(Alm_pencil)
    lloc = axes(Alm_pencil, 1); mloc = axes(Alm_pencil, 2)
    gl_l = globalindices(Alm_pencil, 1)
    gl_m = globalindices(Alm_pencil, 2)
    nl_local = length(lloc)
    counts = Allgather(nl_local, comm)
    displs = cumsum([0; counts[1:end-1]])
    col_full = Vector{ComplexF64}(undef, lmax + 1)
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval > mmax && continue
        col_local = Array(view(Alm_pencil, :, jm))
        Allgatherv(col_local, col_full, counts, displs, comm)
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            idx = SHTnsKit.LM_index(lmax, cfg.mres, lval, mval)
            c_minus = mx[2*idx + 1]; c_plus = mx[2*idx + 2]
            acc = 0.0 + 0.0im
            if lval > mval && lval > 0
                acc += c_minus * col_full[lval]
            end
            if lval < lmax
                acc += c_plus * col_full[lval + 2]
            end
            R_pencil[il, jm] = acc
        end
    end
    return R_pencil
end
