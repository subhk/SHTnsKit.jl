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

IMPLEMENTATION VARIANTS
-----------------------
1. Fused loops (default, use_fused_loops=true):
   - Combines operations to improve cache utilization
   - Better for larger problems

2. Unfused loops (use_fused_loops=false):
   - Separate loops for each operation
   - May be better for debugging or small problems

3. Table-based (cfg.use_plm_tables=true):
   - Uses precomputed Legendre polynomial tables
   - Faster but requires more memory

4. On-the-fly (cfg.on_the_fly=true):
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

    # Use maxthreadid() to handle all possible thread IDs with static scheduling
    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]

    @threads :static for m in 0:mmax
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

    # Use maxthreadid() to handle all possible thread IDs with static scheduling
    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]

    @threads :static for m in 0:mmax
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
        @threads :static for m in 0:mmax
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
        # Use maxthreadid() to handle all possible thread IDs with static scheduling
        thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]
        @threads :static for m in 0:mmax
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
        @threads :static for m in 0:mmax
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
        # Use maxthreadid() to handle all possible thread IDs with static scheduling
        thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]
        @threads :static for m in 0:mmax
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
        synthesis_fused(cfg, alm; real_output=real_output, fft_scratch=Fφ)
    else
        synthesis_unfused(cfg, alm; real_output=real_output, fft_scratch=Fφ)
    end
    # After conjugate symmetry + IFFT, Fφ contains the result
    # For real_output=true, imaginary part should be ~0 due to Hermitian symmetry
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

    # Use maxthreadid() to handle all possible thread IDs with static scheduling
    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]
    @threads :static for m in 0:mmax
        col = m + 1
        P = thread_local_P[Threads.threadid()]
        if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
            tbl = cfg.plm_tables[m+1]
            @inbounds for i in 1:nlat
                acc = zero(CT)
                for l in m:lmax
                    acc += (cfg.Nlm[l+1, col] * tbl[l+1, i]) * alm[l+1, col]
                end
                Fφ[i, col] = inv_scaleφ * acc
            end
        else
            @inbounds for i in 1:nlat
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
        @threads :static for m in 0:mmax
            col = m + 1
            tbl = cfg.plm_tables[m+1]
            @inbounds for i in 1:nlat, l in m:lmax
                Fφ[i, col] += (cfg.Nlm[l+1, col] * tbl[l+1, i]) * alm[l+1, col]
            end
        end
    else
        # Use maxthreadid() to handle all possible thread IDs with static scheduling
        thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.maxthreadid()]
        @threads :static for m in 0:mmax
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
    @inbounds for j in 1:nlon, i in 1:nlat
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
