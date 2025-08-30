module SHTnsKitLoopVecExt

using LoopVectorization
using SHTnsKit
using Base.Threads: @threads

# Turbo-optimized variants live under the SHTnsKit namespace so users can call
# SHTnsKit.analysis_turbo, synthesis_turbo, etc., when LoopVectorization is loaded.

"""
    SHTnsKit.analysis_turbo(cfg::SHTnsKit.SHTConfig, f::AbstractMatrix)

Forward transform with LoopVectorization-optimized inner loops. Same API and
output as `SHTnsKit.analysis`.
"""
function SHTnsKit.analysis_turbo(cfg::SHTnsKit.SHTConfig, f::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))

    fC = complex.(f)
    Fφ = SHTnsKit.fft_phi(fC)

    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(Fφ)
    alm = Matrix{CT}(undef, lmax + 1, mmax + 1)
    fill!(alm, 0.0 + 0.0im)

    P = Vector{Float64}(undef, lmax + 1)
    scaleφ = cfg.cphi
    # Adaptive threading: use nested parallelism for better load balancing
    n_threads = Threads.nthreads()
    if mmax + 1 < n_threads ÷ 2 && nlat > 32
        # Few m modes: parallelize over latitude points instead
        for m in 0:mmax
            col = m + 1
            if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
                tbl = cfg.plm_tables[m + 1]
                @threads for i in 1:nlat
                    Fi = Fφ[i, col]
                    wi = cfg.w[i]
                    @tturbo warn_check_args=false for l in m:lmax
                        alm[l + 1, col] += (wi * tbl[l + 1, i]) * Fi
                    end
                end
            else
                @threads for i in 1:nlat
                    thread_P = Vector{Float64}(undef, lmax + 1)
                    SHTnsKit.Plm_row!(thread_P, cfg.x[i], lmax, m)
                    Fi = Fφ[i, col]
                    wi = cfg.w[i]
                    @tturbo warn_check_args=false for l in m:lmax
                        alm[l + 1, col] += (wi * thread_P[l + 1]) * Fi
                    end
                end
            end
            @tturbo warn_check_args=false for l in m:lmax
                alm[l + 1, col] *= cfg.Nlm[l + 1, col] * scaleφ
            end
        end
    else
        # Standard m-parallel approach with dynamic scheduling
        @threads :dynamic for m in 0:mmax
            col = m + 1
            if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
                tbl = cfg.plm_tables[m + 1]
                for i in 1:nlat
                    Fi = Fφ[i, col]
                    wi = cfg.w[i]
                    @tturbo warn_check_args=false for l in m:lmax
                        alm[l + 1, col] += (wi * tbl[l + 1, i]) * Fi
                    end
                end
            else
                for i in 1:nlat
                    SHTnsKit.Plm_row!(P, cfg.x[i], lmax, m)
                    Fi = Fφ[i, col]
                    wi = cfg.w[i]
                    @tturbo warn_check_args=false for l in m:lmax
                        alm[l + 1, col] += (wi * P[l + 1]) * Fi
                    end
                end
            end
            @tturbo warn_check_args=false for l in m:lmax
                alm[l + 1, col] *= cfg.Nlm[l + 1, col] * scaleφ
            end
        end
    end
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        alm2 = similar(alm)
        SHTnsKit.convert_alm_norm!(alm2, alm, cfg; to_internal=false)
        return alm2
    else
        return alm
    end
end

"""
    SHTnsKit.synthesis_turbo(cfg::SHTnsKit.SHTConfig, alm::AbstractMatrix; real_output::Bool=true)

Inverse transform with LoopVectorization-optimized inner loops. Same API and
output as `SHTnsKit.synthesis`.
"""
function SHTnsKit.synthesis_turbo(cfg::SHTnsKit.SHTConfig, alm::AbstractMatrix; real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))

    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    Fφ = Matrix{CT}(undef, nlat, nlon)
    fill!(Fφ, 0.0 + 0.0im)

    P = Vector{Float64}(undef, lmax + 1)
    G = Vector{CT}(undef, nlat)
    inv_scaleφ = nlon / (2π)

    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        alm_int = similar(alm)
        SHTnsKit.convert_alm_norm!(alm_int, alm, cfg; to_internal=true)
        alm = alm_int
    end

    # Adaptive threading: use nested parallelism for better load balancing
    n_threads = Threads.nthreads()
    if mmax + 1 < n_threads ÷ 2 && nlat > 32
        # Few m modes: parallelize over latitude points instead
        for m in 0:mmax
            col = m + 1
            if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
                tbl = cfg.plm_tables[m + 1]
                @threads for i in 1:nlat
                    g_re = 0.0
                    g_im = 0.0
                    @tturbo warn_check_args=false for l in m:lmax
                        c = cfg.Nlm[l + 1, col] * tbl[l + 1, i]
                        a = alm[l + 1, col]
                        g_re += c * real(a)
                        g_im += c * imag(a)
                    end
                    G[i] = complex(g_re, g_im)
                end
            else
                @threads for i in 1:nlat
                    thread_P = Vector{Float64}(undef, lmax + 1)
                    SHTnsKit.Plm_row!(thread_P, cfg.x[i], lmax, m)
                    g_re = 0.0
                    g_im = 0.0
                    @tturbo warn_check_args=false for l in m:lmax
                        c = cfg.Nlm[l + 1, col] * thread_P[l + 1]
                        a = alm[l + 1, col]
                        g_re += c * real(a)
                        g_im += c * imag(a)
                    end
                    G[i] = complex(g_re, g_im)
                end
            end
            @tturbo warn_check_args=false for i in 1:nlat
                Fφ[i, col] = inv_scaleφ * G[i]
            end
        end
    else
        # Standard m-parallel approach with dynamic scheduling
        @threads :dynamic for m in 0:mmax
            col = m + 1
            if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
                tbl = cfg.plm_tables[m + 1]
                for i in 1:nlat
                    g_re = 0.0
                    g_im = 0.0
                    @tturbo warn_check_args=false for l in m:lmax
                        c = cfg.Nlm[l + 1, col] * tbl[l + 1, i]
                        a = alm[l + 1, col]
                        g_re += c * real(a)
                        g_im += c * imag(a)
                    end
                    G[i] = complex(g_re, g_im)
                end
            else
                for i in 1:nlat
                    SHTnsKit.Plm_row!(P, cfg.x[i], lmax, m)
                    g_re = 0.0
                    g_im = 0.0
                    @tturbo warn_check_args=false for l in m:lmax
                        c = cfg.Nlm[l + 1, col] * P[l + 1]
                        a = alm[l + 1, col]
                        g_re += c * real(a)
                        g_im += c * imag(a)
                    end
                    G[i] = complex(g_re, g_im)
                end
            end
            @tturbo warn_check_args=false for i in 1:nlat
                Fφ[i, col] = inv_scaleφ * G[i]
            end
            if real_output && m > 0
                conj_index = nlon - m + 1
                @tturbo warn_check_args=false for i in 1:nlat
                    Fφ[i, conj_index] = conj(Fφ[i, col])
                end
            end
        end
    end

    f = SHTnsKit.ifft_phi(Fφ)
    return real_output ? real.(f) : f
end

"""
    SHTnsKit.turbo_apply_laplacian!(cfg::SHTnsKit.SHTConfig, alm::AbstractMatrix)

In-place multiplication by l(l+1) in spectral space (Laplacian factor) using
LoopVectorization. This mirrors the common spectral Laplacian application.
"""
function SHTnsKit.turbo_apply_laplacian!(cfg::SHTnsKit.SHTConfig, alm::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))
    @threads for m in 0:mmax
        col = m + 1
        @tturbo warn_check_args=false for l in m:lmax
            alm[l + 1, col] *= l * (l + 1)
        end
    end
    return alm
end

"""
    SHTnsKit.turbo_apply_laplacian!(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex})

In-place Laplacian factor application for packed coefficients (SHTns LM order).
"""
function SHTnsKit.turbo_apply_laplacian!(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex})
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    @tturbo warn_check_args=false for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0 + 1]
        Qlm[lm0 + 1] *= l * (l + 1)
    end
    return Qlm
end

"""
    SHTnsKit.benchmark_turbo_vs_simd(cfg; trials=3)

Run a simple timing comparison between `analysis`/`synthesis` and their turbo
variants. Returns a NamedTuple with timings and speedup estimates.
"""
function SHTnsKit.benchmark_turbo_vs_simd(cfg::SHTnsKit.SHTConfig; trials::Integer=3)
    nlat, nlon = cfg.nlat, cfg.nlon
    f = randn(nlat, nlon)
    alm = randn(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)

    # Warmup
    A1 = SHTnsKit.analysis(cfg, f)
    S1 = SHTnsKit.synthesis(cfg, A1)
    A2 = SHTnsKit.analysis_turbo(cfg, f)
    S2 = SHTnsKit.synthesis_turbo(cfg, A2)
    GC.gc()

    function timeit(fn)
        tmin = Inf
        for _ in 1:trials
            GC.gc()
            t = @elapsed fn()
            tmin = min(tmin, t)
        end
        return tmin
    end

    t_analysis_baseline = timeit(() -> SHTnsKit.analysis(cfg, f))
    t_analysis_turbo    = timeit(() -> SHTnsKit.analysis_turbo(cfg, f))
    t_synth_baseline    = timeit(() -> SHTnsKit.synthesis(cfg, alm))
    t_synth_turbo       = timeit(() -> SHTnsKit.synthesis_turbo(cfg, alm))

    return (;
        analysis_baseline=t_analysis_baseline,
        analysis_turbo=t_analysis_turbo,
        synthesis_baseline=t_synth_baseline,
        synthesis_turbo=t_synth_turbo,
        analysis_speedup = t_analysis_baseline / max(t_analysis_turbo, eps()),
        synthesis_speedup = t_synth_baseline / max(t_synth_turbo, eps()),
        speedup = t_analysis_baseline / max(t_analysis_turbo, eps()),
    )
end

end # module SHTnsKitLoopVecExt
