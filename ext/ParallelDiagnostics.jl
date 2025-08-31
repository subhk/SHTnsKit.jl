##########
# Pencil-aware diagnostics (MPI reductions)
##########

function SHTnsKit.energy_scalar(cfg::SHTnsKit.SHTConfig, Alm::PencilArray; real_field::Bool=true)
    mloc = axes(Alm, 2)
    gl_m = globalindices(Alm, 2)
    e_local = 0.0
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        w = (real_field && mval > 0) ? 2.0 : 1.0
        for (ii, il) in enumerate(axes(Alm, 1))
            e_local += w * abs2(Alm[il, jm])
        end
    end
    e = Allreduce(e_local, +, communicator(Alm))
    return 0.5 * e
end

function SHTnsKit.energy_scalar_l_spectrum(cfg::SHTnsKit.SHTConfig, Alm::PencilArray; real_field::Bool=true)
    lmax = cfg.lmax
    E = zeros(Float64, lmax + 1)
    lloc = axes(Alm, 1); mloc = axes(Alm, 2)
    gl_l = globalindices(Alm, 1)
    gl_m = globalindices(Alm, 2)
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        wm = (real_field && mval > 0) ? 1.0 : 0.5
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= mval
                E[lval + 1] += wm * abs2(Alm[il, jm])
            end
        end
    end
    MPI.Allreduce!(E, +, communicator(Alm))
    return E
end

function SHTnsKit.energy_scalar_m_spectrum(cfg::SHTnsKit.SHTConfig, Alm::PencilArray; real_field::Bool=true)
    mmax = cfg.mmax
    E = zeros(Float64, mmax + 1)
    mloc = axes(Alm, 2)
    gl_m = globalindices(Alm, 2)
    lloc = axes(Alm, 1)
    gl_l = globalindices(Alm, 1)
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        s = 0.0
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= mval
                s += abs2(Alm[il, jm])
            end
        end
        wm2 = (real_field && mval > 0) ? 1.0 : 0.5
        E[mval + 1] += wm2 * s
    end
    MPI.Allreduce!(E, +, communicator(Alm))
    return E
end

function SHTnsKit.energy_vector_l_spectrum(cfg::SHTnsKit.SHTConfig, Slm::PencilArray, Tlm::PencilArray; real_field::Bool=true)
    lmax = cfg.lmax
    E = zeros(Float64, lmax + 1)
    lloc = axes(Slm, 1); mloc = axes(Slm, 2)
    gl_l = globalindices(Slm, 1)
    gl_m = globalindices(Slm, 2)
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        wm2 = (real_field && mval > 0) ? 1.0 : 0.5
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= max(1, mval)
                L2 = lval*(lval+1)
                E[lval + 1] += wm2 * L2 * (abs2(Slm[il, jm]) + abs2(Tlm[il, jm]))
            end
        end
    end
    MPI.Allreduce!(E, +, communicator(Slm))
    return E
end

function SHTnsKit.energy_vector_m_spectrum(cfg::SHTnsKit.SHTConfig, Slm::PencilArray, Tlm::PencilArray; real_field::Bool=true)
    mmax = cfg.mmax
    E = zeros(Float64, mmax + 1)
    lloc = axes(Slm, 1); mloc = axes(Slm, 2)
    gl_l = globalindices(Slm, 1)
    gl_m = globalindices(Slm, 2)
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        s = 0.0
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= max(1, mval)
                L2 = lval*(lval+1)
                s += L2 * (abs2(Slm[il, jm]) + abs2(Tlm[il, jm]))
            end
        end
        wm2 = (real_field && mval > 0) ? 1.0 : 0.5
        E[mval + 1] += wm2 * s
    end
    MPI.Allreduce!(E, +, communicator(Slm))
    return E
end

function SHTnsKit.enstrophy_l_spectrum(cfg::SHTnsKit.SHTConfig, Tlm::PencilArray; real_field::Bool=true)
    lmax = cfg.lmax
    Z = zeros(Float64, lmax + 1)
    lloc = axes(Tlm, 1); mloc = axes(Tlm, 2)
    gl_l = globalindices(Tlm, 1)
    gl_m = globalindices(Tlm, 2)
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        wm2 = (real_field && mval > 0) ? 1.0 : 0.5
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= max(1, mval)
                L2 = lval*(lval+1)
                Z[lval + 1] += wm2 * (L2^2) * abs2(Tlm[il, jm])
            end
        end
    end
    MPI.Allreduce!(Z, +, communicator(Tlm))
    return Z
end

function SHTnsKit.enstrophy_m_spectrum(cfg::SHTnsKit.SHTConfig, Tlm::PencilArray; real_field::Bool=true)
    mmax = cfg.mmax
    Z = zeros(Float64, mmax + 1)
    lloc = axes(Tlm, 1); mloc = axes(Tlm, 2)
    gl_l = globalindices(Tlm, 1)
    gl_m = globalindices(Tlm, 2)
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        s = 0.0
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= max(1, mval)
                L2 = lval*(lval+1)
                s += (L2^2) * abs2(Tlm[il, jm])
            end
        end
        wm2 = (real_field && mval > 0) ? 1.0 : 0.5
        Z[mval + 1] += wm2 * s
    end
    MPI.Allreduce!(Z, +, communicator(Tlm))
    return Z
end

function SHTnsKit.grid_energy_scalar(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray)
    θloc = axes(fθφ, 1)
    φscale = 2π / cfg.nlon
    e_local = 0.0
    @inbounds for (ii, iθ) in enumerate(θloc)
        iglobθ = globalindices(fθφ, 1)[ii]
        wi = cfg.w[iglobθ]
        for j in axes(fθφ, 2)
            e_local += wi * abs2(fθφ[iθ, j])
        end
    end
    e = Allreduce(e_local, +, communicator(fθφ))
    return 0.5 * (φscale * e)
end

function SHTnsKit.grid_energy_vector(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArray, Vpθφ::PencilArray)
    θloc = axes(Vtθφ, 1)
    φscale = 2π / cfg.nlon
    e_local = 0.0
    @inbounds for (ii, iθ) in enumerate(θloc)
        iglobθ = globalindices(Vtθφ, 1)[ii]
        wi = cfg.w[iglobθ]
        for j in axes(Vtθφ, 2)
            e_local += wi * (abs2(Vtθφ[iθ, j]) + abs2(Vpθφ[iθ, j]))
        end
    end
    e = Allreduce(e_local, +, communicator(Vtθφ))
    return 0.5 * (φscale * e)
end

function SHTnsKit.grid_enstrophy(cfg::SHTnsKit.SHTConfig, ζθφ::PencilArray)
    θloc = axes(ζθφ, 1)
    φscale = 2π / cfg.nlon
    z_local = 0.0
    @inbounds for (ii, iθ) in enumerate(θloc)
        iglobθ = globalindices(ζθφ, 1)[ii]
        wi = cfg.w[iglobθ]
        for j in axes(ζθφ, 2)
            z_local += wi * abs2(ζθφ[iθ, j])
        end
    end
    z = Allreduce(z_local, +, communicator(ζθφ))
    return 0.5 * (φscale * z)
end
