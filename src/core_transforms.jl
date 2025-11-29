"""
Core Spherical Harmonic Transforms

Fundamental forward (analysis) and backward (synthesis) transforms between
2D spatial grids and spherical harmonic spectra. Uses Gauss–Legendre quadrature
in latitude and FFT along longitude. Orthonormal spherical harmonics with
Condon–Shortley phase are used internally; normalization conversion is handled
by higher-level helpers when needed.
"""

"""
    analysis(cfg::SHTConfig, f::AbstractMatrix) -> Matrix{ComplexF64}

Forward transform on Gauss–Legendre × equiangular grid.
Returns coefficients `alm[l+1, m+1]` with orthonormal normalization.
"""
function analysis(cfg::SHTConfig, f::AbstractMatrix; use_fused_loops::Bool=true, fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing)
    if use_fused_loops
        return analysis_fused(cfg, f; fft_scratch=fft_scratch)
    else
        return analysis_unfused(cfg, f; fft_scratch=fft_scratch)
    end
end

"""
    analysis!(cfg::SHTConfig, alm_out::AbstractMatrix, f::AbstractMatrix; use_fused_loops=true)

In-place forward transform. Writes coefficients into `alm_out` to avoid allocating
the output matrix each call. `alm_out` must be size `(lmax+1, mmax+1)`.
"""
function analysis!(cfg::SHTConfig, alm_out::AbstractMatrix, f::AbstractMatrix; use_fused_loops::Bool=true, fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing)
    size(alm_out, 1) == cfg.lmax + 1 || throw(DimensionMismatch("alm_out first dim must be lmax+1=$(cfg.lmax+1)"))
    size(alm_out, 2) == cfg.mmax + 1 || throw(DimensionMismatch("alm_out second dim must be mmax+1=$(cfg.mmax+1)"))
    fill!(alm_out, zero(eltype(alm_out)))
    if use_fused_loops
        return analysis_fused!(alm_out, cfg, f; fft_scratch=fft_scratch)
    else
        return analysis_unfused!(alm_out, cfg, f; fft_scratch=fft_scratch)
    end
end

function analysis_unfused(cfg::SHTConfig, f::AbstractMatrix; fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))

    Fφ = fft_scratch === nothing ? fft_phi(complex.(f)) : fft_phi!(fft_scratch, f)
    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(Fφ)
    alm = Matrix{CT}(undef, lmax + 1, mmax + 1)
    fill!(alm, 0)
    scaleφ = cfg.cphi

    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]

    @threads for m in 0:mmax
        col = m + 1
        P = thread_local_P[Threads.threadid()]
        if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
            tbl = cfg.plm_tables[m+1]
            @inbounds for i in 1:nlat
                Fi = Fφ[i, col]
                wi = cfg.w[i]
                for l in m:lmax
                    alm[l+1, col] += (wi * tbl[l+1, i]) * Fi
                end
            end
        else
            @inbounds for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)
                Fi = Fφ[i, col]
                wi = cfg.w[i]
                for l in m:lmax
                    alm[l+1, col] += (wi * P[l+1]) * Fi
                end
            end
        end
        @inbounds for l in m:lmax
            alm[l+1, col] *= cfg.Nlm[l+1, col] * scaleφ
        end
    end
    return alm
end

function analysis_unfused!(alm::AbstractMatrix, cfg::SHTConfig, f::AbstractMatrix; fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))

    Fφ = fft_scratch === nothing ? fft_phi(complex.(f)) : fft_phi!(fft_scratch, f)
    lmax, mmax = cfg.lmax, cfg.mmax
    scaleφ = cfg.cphi

    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]

    @threads for m in 0:mmax
        col = m + 1
        P = thread_local_P[Threads.threadid()]
        if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
            tbl = cfg.plm_tables[m+1]
            @inbounds for i in 1:nlat
                Fi = Fφ[i, col]
                wi = cfg.w[i]
                for l in m:lmax
                    alm[l+1, col] += (wi * tbl[l+1, i]) * Fi
                end
            end
        else
            @inbounds for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)
                Fi = Fφ[i, col]
                wi = cfg.w[i]
                for l in m:lmax
                    alm[l+1, col] += (wi * P[l+1]) * Fi
                end
            end
        end
        @inbounds for l in m:lmax
            alm[l+1, col] *= cfg.Nlm[l+1, col] * scaleφ
        end
    end
    return alm
end

function analysis_fused(cfg::SHTConfig, f::AbstractMatrix; fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))

    Fφ = fft_scratch === nothing ? fft_phi(complex.(f)) : fft_phi!(fft_scratch, f)
    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(Fφ)
    alm = Matrix{CT}(undef, lmax + 1, mmax + 1)
    fill!(alm, 0)
    scaleφ = cfg.cphi

    if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
        @threads for m in 0:mmax
            col = m + 1
            tbl = cfg.plm_tables[m+1]
            @inbounds for i in 1:nlat
                Fi = Fφ[i, col]
                wi = cfg.w[i]
                for l in m:lmax
                    alm[l+1, col] += (wi * cfg.Nlm[l+1, col] * tbl[l+1, i] * scaleφ) * Fi
                end
            end
        end
    else
        thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]
        @threads for m in 0:mmax
            col = m + 1
            P = thread_local_P[Threads.threadid()]
            for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)
                Fi = Fφ[i, col]
                wi = cfg.w[i]
                @inbounds for l in m:lmax
                    alm[l+1, col] += (wi * cfg.Nlm[l+1, col] * P[l+1] * scaleφ) * Fi
                end
            end
        end
    end
    return alm
end

function analysis_fused!(alm::AbstractMatrix, cfg::SHTConfig, f::AbstractMatrix; fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))

    Fφ = fft_scratch === nothing ? fft_phi(complex.(f)) : fft_phi!(fft_scratch, f)
    lmax, mmax = cfg.lmax, cfg.mmax
    scaleφ = cfg.cphi

    if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
        @threads for m in 0:mmax
            col = m + 1
            tbl = cfg.plm_tables[m+1]
            @inbounds for i in 1:nlat
                Fi = Fφ[i, col]
                wi = cfg.w[i]
                for l in m:lmax
                    alm[l+1, col] += (wi * cfg.Nlm[l+1, col] * tbl[l+1, i] * scaleφ) * Fi
                end
            end
        end
    else
        thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]
        @threads for m in 0:mmax
            col = m + 1
            P = thread_local_P[Threads.threadid()]
            for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)
                Fi = Fφ[i, col]
                wi = cfg.w[i]
                @inbounds for l in m:lmax
                    alm[l+1, col] += (wi * cfg.Nlm[l+1, col] * P[l+1] * scaleφ) * Fi
                end
            end
        end
    end
    return alm
end

"""
    synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true) -> Matrix

Inverse transform back to a grid `(nlat, nlon)`. If `real_output=true`,
Hermitian symmetry is enforced before IFFT. Optional `fft_scratch` lets you
reuse a preallocated `(nlat,nlon)` complex buffer for lower allocations.
"""
function synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true, use_fused_loops::Bool=true, fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing)
    if use_fused_loops
        return synthesis_fused(cfg, alm; real_output=real_output, fft_scratch=fft_scratch)
    else
        return synthesis_unfused(cfg, alm; real_output=real_output, fft_scratch=fft_scratch)
    end
end

"""
    synthesis!(cfg::SHTConfig, f_out::AbstractMatrix, alm::AbstractMatrix;
               real_output::Bool=true, use_fused_loops::Bool=true)

In-place inverse transform. Writes the spatial field into `f_out` to reduce
allocations. `f_out` must be `(nlat, nlon)` and real if `real_output=true`.
You may pass a complex `fft_scratch` buffer of size `(nlat,nlon)` to reuse FFT
workspace.
"""
function synthesis!(cfg::SHTConfig, f_out::AbstractMatrix, alm::AbstractMatrix; real_output::Bool=true, use_fused_loops::Bool=true, fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing)
    size(f_out, 1) == cfg.nlat || throw(DimensionMismatch("f_out first dim must be nlat=$(cfg.nlat)"))
    size(f_out, 2) == cfg.nlon || throw(DimensionMismatch("f_out second dim must be nlon=$(cfg.nlon)"))
    Fφ = fft_scratch === nothing ? Matrix{ComplexF64}(undef, cfg.nlat, cfg.nlon) : fft_scratch
    size(Fφ,1) == cfg.nlat && size(Fφ,2) == cfg.nlon || throw(DimensionMismatch("fft_scratch wrong size"))
    fill!(Fφ, 0)
    if use_fused_loops
        synthesis_fused(cfg, alm; real_output=false, fft_scratch=Fφ)
    else
        synthesis_unfused(cfg, alm; real_output=false, fft_scratch=Fφ)
    end
    if real_output
        @inbounds for j in 1:cfg.nlon, i in 1:cfg.nlat
            f_out[i, j] = real(Fφ[i, j])
        end
    else
        f_out .= Fφ
    end
    return f_out
end

function synthesis_unfused(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true, fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))

    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    Fφ = fft_scratch === nothing ? Matrix{CT}(undef, nlat, nlon) : fft_scratch
    size(Fφ,1) == nlat && size(Fφ,2) == nlon || throw(DimensionMismatch("fft_scratch wrong size"))
    fill!(Fφ, zero(CT))
    inv_scaleφ = phi_inv_scale(cfg)

    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]
    @threads for m in 0:mmax
        col = m + 1
        P = thread_local_P[Threads.threadid()]
        if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
            tbl = cfg.plm_tables[m+1]
            for i in 1:nlat
                acc = zero(CT)
                for l in m:lmax
                    acc += (cfg.Nlm[l+1, col] * tbl[l+1, i]) * alm[l+1, col]
                end
                Fφ[i, col] = inv_scaleφ * acc
            end
        else
            for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)
                acc = zero(CT)
                for l in m:lmax
                    acc += (cfg.Nlm[l+1, col] * P[l+1]) * alm[l+1, col]
                end
                Fφ[i, col] = inv_scaleφ * acc
            end
        end
    end
    if real_output
        for m in 1:mmax
            col = m + 1
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                Fφ[i, conj_index] = conj(Fφ[i, col])
            end
        end
    end
    f = ifft_phi!(Fφ, Fφ)
    return real_output ? real.(f) : f
end

function synthesis_fused(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true, fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))

    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    Fφ = fft_scratch === nothing ? Matrix{CT}(undef, nlat, nlon) : fft_scratch
    size(Fφ,1) == nlat && size(Fφ,2) == nlon || throw(DimensionMismatch("fft_scratch wrong size"))
    fill!(Fφ, zero(CT))
    inv_scaleφ = phi_inv_scale(cfg)

    if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
        @threads for m in 0:mmax
            col = m + 1
            tbl = cfg.plm_tables[m+1]
            @inbounds for i in 1:nlat, l in m:lmax
                Fφ[i, col] += (cfg.Nlm[l+1, col] * tbl[l+1, i]) * alm[l+1, col]
            end
        end
    else
        thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]
        @threads for m in 0:mmax
            col = m + 1
            P = thread_local_P[Threads.threadid()]
            for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)
                sum_val = zero(CT)
                @inbounds for l in m:lmax
                    sum_val += (cfg.Nlm[l+1, col] * P[l+1]) * alm[l+1, col]
                end
                Fφ[i, col] = sum_val
            end
        end
    end
    @inbounds for i in 1:nlat, j in 1:nlon
        Fφ[i, j] *= inv_scaleφ
    end
    if real_output
        for m in 1:mmax
            col = m + 1
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                Fφ[i, conj_index] = conj(Fφ[i, col])
            end
        end
    end
    f = ifft_phi!(Fφ, Fφ)
    return real_output ? real.(f) : f
end
