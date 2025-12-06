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
        Allgatherv!(col_local, VBuffer(col_full, counts), comm)
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

"""
    dist_spatial_divergence(cfg, Vtθφ, Vpθφ; prototype_θφ=Vtθφ, use_rfft=false, real_output=true)

Compute ∇·V for a distributed horizontal vector field using spectral decomposition.
"""
function SHTnsKit.dist_spatial_divergence(cfg::SHTnsKit.SHTConfig,
                                          Vtθφ::PencilArray, Vpθφ::PencilArray;
                                          prototype_θφ::PencilArray=Vtθφ,
                                          use_rfft::Bool=false,
                                          real_output::Bool=true)
    Slm, _ = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vtθφ, Vpθφ; use_rfft)
    δlm = SHTnsKit.divergence_from_spheroidal(cfg, Slm)
    return SHTnsKit.dist_synthesis(cfg, δlm; prototype_θφ=prototype_θφ,
                                   real_output=real_output, use_rfft=use_rfft)
end

"""
    dist_spatial_vorticity(cfg, Vtθφ, Vpθφ; prototype_θφ=Vtθφ, use_rfft=false, real_output=true)

Compute vertical vorticity (∇×V)·r̂ for a distributed horizontal vector field.
"""
function SHTnsKit.dist_spatial_vorticity(cfg::SHTnsKit.SHTConfig,
                                         Vtθφ::PencilArray, Vpθφ::PencilArray;
                                         prototype_θφ::PencilArray=Vtθφ,
                                         use_rfft::Bool=false,
                                         real_output::Bool=true)
    _, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vtθφ, Vpθφ; use_rfft)
    ζlm = SHTnsKit.vorticity_from_toroidal(cfg, Tlm)
    return SHTnsKit.dist_synthesis(cfg, ζlm; prototype_θφ=prototype_θφ,
                                   real_output=real_output, use_rfft=use_rfft)
end

"""
    dist_scalar_laplacian(cfg, fθφ; prototype_θφ=fθφ, use_rfft=false, real_output=true)

Apply spherical Laplacian to a distributed scalar field by transforming to spectral
space, scaling by −l(l+1), and synthesizing back.
"""
function SHTnsKit.dist_scalar_laplacian(cfg::SHTnsKit.SHTConfig,
                                        fθφ::PencilArray;
                                        prototype_θφ::PencilArray=fθφ,
                                        use_rfft::Bool=false,
                                        real_output::Bool=true)
    Alm = SHTnsKit.dist_analysis(cfg, fθφ; use_rfft)
    SHTnsKit.apply_laplacian!(cfg, Alm)
    return SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=prototype_θφ,
                                   real_output=real_output, use_rfft=use_rfft)
end

"""
    dist_scalar_laplacian!(cfg, outθφ, inθφ; use_rfft=false, real_output=true)

In-place version that writes the Laplacian of `inθφ` into `outθφ`.
"""
function SHTnsKit.dist_scalar_laplacian!(cfg::SHTnsKit.SHTConfig,
                                         outθφ::PencilArray,
                                         inθφ::PencilArray;
                                         use_rfft::Bool=false,
                                         real_output::Bool=true)
    result = SHTnsKit.dist_scalar_laplacian(cfg, inθφ; prototype_θφ=outθφ,
                                            use_rfft=use_rfft, real_output=real_output)
    copyto!(outθφ, result)
    return outθφ
end
