#=
================================================================================
core_transforms.jl - Core Spherical Harmonic Transform Implementations
================================================================================

This file implements the fundamental forward (analysis) and backward (synthesis)
transforms between 2D spatial grids and spherical harmonic coefficients.

TRANSFORM OVERVIEW
------------------
Analysis (spatial → spectral):
    a_lm = ∫∫ f(θ,φ) Y_l^m*(θ,φ) sin(θ) dθ dφ

    Implementation:
    1. FFT along φ: f(θ,φ) → F_m(θ) for each m = 0..mmax
    2. Legendre integration: a_lm = Σ_θ w(θ) * F_m(θ) * P_l^m(cos θ) * N_lm

Synthesis (spectral → spatial):
    f(θ,φ) = Σ_l Σ_m a_lm Y_l^m(θ,φ)

    Implementation:
    1. Legendre summation: F_m(θ) = Σ_l a_lm * P_l^m(cos θ) * N_lm
    2. Inverse FFT along φ: F_m(θ) → f(θ,φ)

KEY FUNCTIONS
-------------
- analysis(cfg, f)       : Forward transform, allocates output
- analysis!(cfg, alm, f) : Forward transform, in-place output
- synthesis(cfg, alm)    : Backward transform, allocates output
- synthesis!(cfg, f, alm): Backward transform, in-place output

IMPLEMENTATION ARCHITECTURE
---------------------------
This file uses a 3-layer architecture:
1. Public API (analysis, synthesis, analysis!, synthesis!)
   - Input validation, FFT, output allocation
2. Orchestrators (_analysis_scalar_mloop!, _synthesis_scalar_mloop!)
   - Threading over m-modes, table vs on-the-fly dispatch
3. Kernels (in kernels.jl)
   - Per-latitude Legendre accumulation, single source of truth

LEGENDRE COMPUTATION MODES
---------------------------
1. Table-based (cfg.use_plm_tables=true):
   - Uses precomputed Legendre polynomial tables
   - Faster but requires more memory

2. On-the-fly (cfg.on_the_fly=true):
   - Computes Legendre polynomials during transform
   - Lower memory, slightly slower

MULTITHREADING
--------------
- Transforms parallelize over m (azimuthal order)
- Each thread has its own Legendre polynomial buffer
- Use `Threads.nthreads()` to check available threads

NUMERICAL PRECISION
-------------------
- Roundtrip error: f ≈ synthesis(cfg, analysis(cfg, f))
- Expected: O(ε) where ε is machine epsilon (~1e-15 for Float64)
- Larger errors may indicate: insufficient nlat, nlon, or numerical instability

DEBUGGING CHECKLIST
-------------------
1. Dimension checks:
   - f must be (nlat, nlon)
   - alm must be (lmax+1, mmax+1)

2. Coefficient interpretation:
   - alm[1,1] = a_{0,0}: mean value (monopole)
   - alm[2,1] = a_{1,0}: north-south gradient
   - alm[2,2] = a_{1,1}: east-west gradient

3. Common issues:
   - Wrong grid size → DimensionMismatch error
   - alm not zeroed → accumulation of old values
   - Missing normalization → wrong magnitudes

EXAMPLE USAGE
-------------
```julia
cfg = create_gauss_config(32, 48)

# Create test field (Y_2^0 pattern)
f = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat
    f[i, :] .= 0.5 * (3*cfg.x[i]^2 - 1)  # P_2^0(x)
end

# Transform to spectral
alm = analysis(cfg, f)
@assert abs(alm[3,1]) > 0.1  # a_{2,0} should be non-zero

# Transform back
f2 = synthesis(cfg, alm)
@assert maximum(abs, f - f2) < 1e-12
```

================================================================================
=#

"""
Core Spherical Harmonic Transforms

Fundamental forward (analysis) and backward (synthesis) transforms between
2D spatial grids and spherical harmonic spectra. Uses Gauss–Legendre quadrature
in latitude and FFT along longitude. Orthonormal spherical harmonics with
Condon–Shortley phase are used internally; normalization conversion is handled
by higher-level helpers when needed.
"""

# ============================================================================
# THREAD UTILIZATION DIAGNOSTICS
# ============================================================================

"""
    check_thread_utilization(cfg::SHTConfig; warn::Bool=true) -> NamedTuple

Check if threading configuration is optimal for this transform configuration.
Returns utilization statistics and optionally warns about suboptimal setups.

SHTnsKit parallelizes transforms over m-modes (azimuthal order) using
`@threads :static`. If `mmax + 1 < nthreads`, some threads will be idle.

# Arguments
- `cfg::SHTConfig`: The transform configuration to check
- `warn::Bool=true`: If true, emit a warning when utilization is below 50%

# Returns
A named tuple with:
- `nthreads`: Number of Julia threads available
- `mmax`: Maximum azimuthal order from config
- `active_threads`: Number of threads that will actually do work
- `utilization`: Fraction of threads utilized (0.0 to 1.0)

# Example
```julia
cfg = create_gauss_config(8, 10)  # Small problem with mmax=8
stats = check_thread_utilization(cfg)
# If running with 16 threads, this warns about only 9 active threads (56%)

# For better utilization, either:
# 1. Use fewer threads: julia -t 8
# 2. Use larger mmax: cfg = create_gauss_config(32, 34)
```

See also: [`configure_threading!`](@ref)
"""
function check_thread_utilization(cfg::SHTConfig; warn::Bool=true)
    nthreads = Threads.nthreads()
    mmax = cfg.mmax

    # With @threads :static for m in 0:mmax, we have (mmax + 1) work items
    work_items = mmax + 1
    active_threads = min(nthreads, work_items)
    utilization = nthreads > 0 ? active_threads / nthreads : 1.0

    if warn && utilization < 0.5 && nthreads > 1
        @warn "Thread underutilization detected" mmax=mmax nthreads=nthreads active_threads=active_threads utilization_pct=round(utilization*100, digits=1) suggestion="Consider using fewer threads (julia -t $(work_items)) or larger mmax"
    end

    return (nthreads=nthreads, mmax=mmax, active_threads=active_threads,
            utilization=utilization)
end

"""
    analysis(cfg::SHTConfig, f::AbstractMatrix) -> Matrix{ComplexF64}

Forward transform on Gauss-Legendre x equiangular grid.
Returns coefficients `alm[l+1, m+1]` with orthonormal normalization.

Parallelizes over m-modes using static scheduling for consistent performance.
"""
function analysis(cfg::SHTConfig, f::AbstractMatrix; fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing, use_rfft::Bool=false)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))
    if use_rfft
        eltype(f) <: Real || throw(ArgumentError("use_rfft=true requires a real-valued input"))
        fft_scratch === nothing || throw(ArgumentError("fft_scratch is not supported with use_rfft=true"))
        Fph = rfft_phi(f)                   # (nlat, nlon÷2+1)
    else
        Fph = fft_scratch === nothing ? fft_phi(_as_complex(f)) : fft_phi!(fft_scratch, f)
    end
    CT = eltype(Fph)
    alm = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    _analysis_scalar_mloop!(alm, cfg, Fph)
    return alm
end

"""
    analysis!(cfg::SHTConfig, alm_out::AbstractMatrix, f::AbstractMatrix)

In-place forward transform. Writes coefficients into `alm_out`.
`alm_out` must be size `(lmax+1, mmax+1)`.
"""
function analysis!(cfg::SHTConfig, alm_out::AbstractMatrix, f::AbstractMatrix; fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing, use_rfft::Bool=false)
    size(alm_out, 1) == cfg.lmax + 1 || throw(DimensionMismatch("alm_out first dim must be lmax+1=$(cfg.lmax+1)"))
    size(alm_out, 2) == cfg.mmax + 1 || throw(DimensionMismatch("alm_out second dim must be mmax+1=$(cfg.mmax+1)"))
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))
    fill!(alm_out, zero(eltype(alm_out)))
    if use_rfft
        eltype(f) <: Real || throw(ArgumentError("use_rfft=true requires a real-valued input"))
        fft_scratch === nothing || throw(ArgumentError("fft_scratch is not supported with use_rfft=true"))
        Fph = rfft_phi(f)
    else
        Fph = fft_scratch === nothing ? fft_phi(_as_complex(f)) : fft_phi!(fft_scratch, f)
    end
    _analysis_scalar_mloop!(alm_out, cfg, Fph)
    return alm_out
end

"""
    synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true) -> Matrix

Inverse transform back to a grid `(nlat, nlon)`. If `real_output=true`,
Hermitian symmetry is enforced before IFFT.
"""
function synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true, fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing, use_rfft::Bool=false)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    if use_rfft
        real_output || throw(ArgumentError("use_rfft=true implies real_output"))
        fft_scratch === nothing || throw(ArgumentError("fft_scratch is not supported with use_rfft=true"))
        mmax ≤ nlon ÷ 2 || throw(ArgumentError("use_rfft=true requires mmax ≤ nlon÷2, got mmax=$mmax, nlon=$nlon"))
        nbins = nlon ÷ 2 + 1
        Fph = Matrix{CT}(undef, nlat, nbins)
        fill!(Fph, zero(CT))
        _synthesis_scalar_mloop!(Fph, cfg, alm; real_output=false, use_rfft=true)
        RT = real(CT)
        out = Matrix{RT}(undef, nlat, nlon)
        irfft_phi!(out, Fph, nlon)
        return out
    end
    Fph = fft_scratch === nothing ? Matrix{CT}(undef, nlat, nlon) : fft_scratch
    size(Fph, 1) == nlat && size(Fph, 2) == nlon || throw(DimensionMismatch("fft_scratch wrong size"))
    fill!(Fph, zero(CT))
    _synthesis_scalar_mloop!(Fph, cfg, alm; real_output=real_output)
    ifft_phi!(Fph, Fph)
    return real_output ? real.(Fph) : Fph
end

"""
    synthesis!(cfg::SHTConfig, f_out::AbstractMatrix, alm::AbstractMatrix; real_output::Bool=true)

In-place inverse transform. Writes the spatial field into `f_out`.
"""
function synthesis!(cfg::SHTConfig, f_out::AbstractMatrix, alm::AbstractMatrix; real_output::Bool=true, fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing, use_rfft::Bool=false)
    size(f_out, 1) == cfg.nlat || throw(DimensionMismatch("f_out first dim must be nlat=$(cfg.nlat)"))
    size(f_out, 2) == cfg.nlon || throw(DimensionMismatch("f_out second dim must be nlon=$(cfg.nlon)"))
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    if use_rfft
        real_output || throw(ArgumentError("use_rfft=true implies real_output"))
        fft_scratch === nothing || throw(ArgumentError("fft_scratch is not supported with use_rfft=true"))
        eltype(f_out) <: Real || throw(ArgumentError("use_rfft=true requires real-valued f_out"))
        mmax ≤ nlon ÷ 2 || throw(ArgumentError("use_rfft=true requires mmax ≤ nlon÷2, got mmax=$mmax, nlon=$nlon"))
        nbins = nlon ÷ 2 + 1
        Fph = Matrix{CT}(undef, nlat, nbins)
        fill!(Fph, zero(eltype(Fph)))
        _synthesis_scalar_mloop!(Fph, cfg, alm; real_output=false, use_rfft=true)
        irfft_phi!(f_out, Fph, nlon)
        return f_out
    end
    Fph = fft_scratch === nothing ? Matrix{CT}(undef, nlat, nlon) : fft_scratch
    size(Fph, 1) == nlat && size(Fph, 2) == nlon || throw(DimensionMismatch("fft_scratch wrong size"))
    fill!(Fph, zero(eltype(Fph)))
    _synthesis_scalar_mloop!(Fph, cfg, alm; real_output=real_output)
    ifft_phi!(Fph, Fph)
    if real_output
        @inbounds for j in 1:nlon, i in 1:nlat
            f_out[i, j] = real(Fph[i, j])
        end
    else
        @inbounds for j in 1:nlon, i in 1:nlat
            f_out[i, j] = Fph[i, j]
        end
    end
    return f_out
end

# ============================================================================
# SCALAR ORCHESTRATORS
# ============================================================================

"""
    _adjoint_analysis(cfg, Alm̄; θ_globals=1:cfg.nlat, φ_window=nothing)

Adjoint operator of `analysis`. Shares the scalar synthesis kernels; only the
per-row scale differs (`φadj * cfg.w[i]` instead of `inv_scale_phi`). Negative-m
columns stay zero — the adjoint places mass only in measured bins.

Parametric over the θ set we integrate against so the distributed adjoint
(rank-local θ slab) can reuse this exact helper with a different `θ_globals`
range and optional φ-window slice. Lives in src so AD extensions don't have
to duplicate it.
"""
function _adjoint_analysis(cfg::SHTConfig, Alm̄::AbstractMatrix;
                           θ_globals::AbstractVector{<:Integer}=1:cfg.nlat,
                           φ_window::Union{Nothing,UnitRange{Int}}=nothing)
    nlon = cfg.nlon
    nlat_local = length(θ_globals)
    Fφ = Matrix{ComplexF64}(undef, nlat_local, nlon)
    fill!(Fφ, zero(eltype(Fφ)))
    lmax, mmax = cfg.lmax, cfg.mmax
    φadj = 2π  # nlon (ifft adjoint) × cphi (2π/nlon) = 2π
    use_tbl = has_fused_scalar_tables(cfg)
    P = use_tbl ? nothing : Vector{Float64}(undef, lmax + 1)
    for m in 0:mmax
        col = m + 1
        if use_tbl
            NP = cfg.NP_tables[m+1]
            @inbounds for (ii, iglob) in pairs(θ_globals)
                Fφ[ii, col] = (φadj * cfg.w[iglob]) *
                    _scalar_synthesis_kernel(cfg, Alm̄, NP, iglob, col, m, lmax)
            end
        else
            @inbounds for (ii, iglob) in pairs(θ_globals)
                Fφ[ii, col] = (φadj * cfg.w[iglob]) *
                    _scalar_synthesis_kernel_otf(cfg, Alm̄, P, iglob, col, m, lmax)
            end
        end
    end
    ifft_phi!(Fφ, Fφ)
    if φ_window === nothing
        return real.(Fφ)
    end
    out = Matrix{Float64}(undef, nlat_local, length(φ_window))
    @inbounds for (jj, jglob) in pairs(φ_window)
        for i in 1:nlat_local
            out[i, jj] = real(Fφ[i, jglob])
        end
    end
    return out
end

"""Scalar analysis orchestrator. Parallelizes Legendre integration over m-modes."""
function _analysis_scalar_mloop!(alm::AbstractMatrix, cfg::SHTConfig, Fph::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    scale_phi = cfg.cphi
    m_order = cached_m_order(cfg)
    if has_fused_scalar_tables(cfg)
        _analysis_scalar_mloop_tbl!(alm, cfg, Fph, m_order, scale_phi)
    else
        _analysis_scalar_mloop_otf!(alm, cfg, Fph, m_order, scale_phi)
    end
    return alm
end

@inline function _analysis_scalar_mloop_tbl!(alm, cfg, Fph, m_order, scale_phi)
    lmax = cfg.lmax
    nlat = cfg.nlat
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        NP = cfg.NP_tables[m+1]
        @inbounds for i in 1:nlat
            _scalar_analysis_kernel!(alm, cfg, Fph, NP, i, col, m, lmax, scale_phi)
        end
    end
    return alm
end

@inline function _analysis_scalar_mloop_otf!(alm, cfg, Fph, m_order, scale_phi)
    lmax = cfg.lmax
    nlat = cfg.nlat
    thread_local_P = _ensure_otf_scratch!(cfg._otf_scratch_P, lmax)
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        P = thread_local_P[Threads.threadid()]
        @inbounds for i in 1:nlat
            _scalar_analysis_kernel_otf!(alm, cfg, Fph, P, i, col, m, lmax, scale_phi)
        end
    end
    return alm
end

"""Scalar synthesis orchestrator. Parallelizes Legendre summation over m-modes."""
function _synthesis_scalar_mloop!(Fph::AbstractMatrix, cfg::SHTConfig, alm::AbstractMatrix;
                                   real_output::Bool=true, use_rfft::Bool=false)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlat, nlon = cfg.nlat, cfg.nlon
    inv_scale_phi = phi_inv_scale(cfg)
    m_order = cached_m_order(cfg)
    if has_fused_scalar_tables(cfg)
        _synthesis_scalar_mloop_tbl!(Fph, cfg, alm, m_order, inv_scale_phi)
    else
        _synthesis_scalar_mloop_otf!(Fph, cfg, alm, m_order, inv_scale_phi)
    end
    # Fill Hermitian conjugate columns for real-output IFFT — only the full
    # complex buffer (size nlon) can hold these. rfft's output half has no
    # slot for the negative-m bins; irfft_phi! reconstructs them implicitly.
    if real_output && !use_rfft
        for m in 1:mmax
            col = m + 1
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                Fph[i, conj_index] = conj(Fph[i, col])
            end
        end
    end
    return Fph
end

@inline function _synthesis_scalar_mloop_tbl!(Fph, cfg, alm, m_order, inv_scale_phi)
    lmax = cfg.lmax
    nlat = cfg.nlat
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        NP = cfg.NP_tables[m+1]
        @inbounds for i in 1:nlat
            Fph[i, col] = inv_scale_phi * _scalar_synthesis_kernel(cfg, alm, NP, i, col, m, lmax)
        end
    end
end

@inline function _synthesis_scalar_mloop_otf!(Fph, cfg, alm, m_order, inv_scale_phi)
    lmax = cfg.lmax
    nlat = cfg.nlat
    thread_local_P = _ensure_otf_scratch!(cfg._otf_scratch_P, lmax)
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        P = thread_local_P[Threads.threadid()]
        @inbounds for i in 1:nlat
            Fph[i, col] = inv_scale_phi * _scalar_synthesis_kernel_otf(cfg, alm, P, i, col, m, lmax)
        end
    end
end
