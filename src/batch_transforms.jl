#=
================================================================================
batch_transforms.jl - Batch Spherical Harmonic Transforms
================================================================================

This file implements batch transform operations for processing multiple
fields simultaneously, providing significant performance benefits.

WHY BATCH TRANSFORMS?
---------------------
When transforming many fields (e.g., time series, ensemble members, or
multi-component fields), individual transforms have repeated overhead:

Individual transforms (N fields):
- N × Legendre polynomial computations
- N × FFT plan creations
- N × function call overhead

Batch transforms:
- 1 × Legendre polynomial computation (shared)
- 1 × FFT plan (shared or batched)
- 1 × function call

PERFORMANCE GAINS
-----------------
- Legendre polynomials: Computed once, applied to all fields
- Cache efficiency: Contiguous memory access patterns
- Reduced function overhead: Single dispatch for N fields
- Better vectorization: Larger working sets for SIMD

TYPICAL USE CASES
-----------------
- Time-stepping PDEs: Transform velocity/pressure at each timestep
- Ensemble simulations: Transform all ensemble members together
- Multi-component fields: (u, v, w) velocity components
- Sensitivity analysis: Transform many perturbed fields

FUNCTION OVERVIEW
-----------------
Scalar batch transforms:
    analysis_batch(cfg, fields)      : Multiple spatial → spectral
    synthesis_batch(cfg, alm_batch)  : Multiple spectral → spatial

Vector batch transforms:
    analysis_sphtor_batch(cfg, Vt, Vp)            : 2D vector analysis
    synthesis_sphtor_batch(cfg, Slm, Tlm)         : 2D vector synthesis

QST batch transforms:
    analysis_qst_batch(cfg, Vr, Vt, Vp)           : 3D vector analysis
    synthesis_qst_batch(cfg, Qlm, Slm, Tlm)       : 3D vector synthesis

USAGE EXAMPLE
-------------
```julia
cfg = create_gauss_config(64, 128)

# Process 10 fields at once
nfields = 10
fields = rand(cfg.nlat, cfg.nlon, nfields)

# Batch analysis - much faster than looping
alm_batch = analysis_batch(cfg, fields)

# Modify all spectra (e.g., filter)
alm_batch[50:end, :, :] .= 0  # Low-pass filter

# Batch synthesis
fields_filtered = synthesis_batch(cfg, alm_batch)
```

DATA LAYOUT
-----------
Spatial arrays:  (nlat, nlon, nfields)
Spectral arrays: (lmax+1, mmax+1, nfields)

The third dimension indexes the different fields.

THREADING
---------
Batch operations use @threads for parallelization across m-modes,
with thread-local Legendre polynomial buffers to avoid race conditions.

================================================================================
=#

"""
Batch Spherical Harmonic Transforms

Efficient processing of multiple fields simultaneously. Batch transforms reduce
overhead by sharing precomputed Legendre polynomials and FFT plans across
multiple fields, improving cache utilization and reducing function call overhead.

This mirrors the `shtns_set_many` functionality from the SHTns C library.
"""

@inline _batch_fft_fallback_error(e) = e isa MethodError || e isa ArgumentError || e isa InexactError

# Batch FFT helpers build one plan from the first field and reuse it for every
# sibling slice. Fallback delegates to scalar FFT wrappers so AD element types
# keep the same behavior as the non-batch API.
function _batch_fft_phi!(Fφ_batch::AbstractArray{<:Complex,3}, fields::AbstractArray{<:Real,3})
    nfields = size(fields, 3)
    nfields == 0 && return Fφ_batch

    try
        plan = plan_fft!(view(Fφ_batch, :, :, 1), 2; flags=FFTW.ESTIMATE)
        @inbounds for k in 1:nfields
            Fk = view(Fφ_batch, :, :, k)
            Xk = view(fields, :, :, k)
            @. Fk = complex(Xk)
            mul!(Fk, plan, Fk)
        end
        _FFT_BACKEND[] = _FFT_BACKEND_FFTW
    catch e
        _batch_fft_fallback_error(e) || rethrow(e)
        @inbounds for k in 1:nfields
            fft_phi!(view(Fφ_batch, :, :, k), view(fields, :, :, k))
        end
    end
    return Fφ_batch
end

function _batch_rfft_phi!(Fφ_batch::AbstractArray{<:Complex,3}, fields::AbstractArray{<:Real,3})
    nfields = size(fields, 3)
    nfields == 0 && return Fφ_batch

    try
        plan = plan_rfft(view(fields, :, :, 1), 2; flags=FFTW.ESTIMATE)
        @inbounds for k in 1:nfields
            mul!(view(Fφ_batch, :, :, k), plan, view(fields, :, :, k))
        end
        _FFT_BACKEND[] = _FFT_BACKEND_FFTW
    catch e
        _batch_fft_fallback_error(e) || rethrow(e)
        @inbounds for k in 1:nfields
            view(Fφ_batch, :, :, k) .= rfft_phi(view(fields, :, :, k))
        end
    end
    return Fφ_batch
end

function _batch_ifft_phi!(Fφ_batch::AbstractArray{<:Complex,3})
    nfields = size(Fφ_batch, 3)
    nfields == 0 && return Fφ_batch

    try
        plan = plan_ifft!(view(Fφ_batch, :, :, 1), 2; flags=FFTW.ESTIMATE)
        @inbounds for k in 1:nfields
            Fk = view(Fφ_batch, :, :, k)
            mul!(Fk, plan, Fk)
        end
        _FFT_BACKEND[] = _FFT_BACKEND_FFTW
    catch e
        _batch_fft_fallback_error(e) || rethrow(e)
        @inbounds for k in 1:nfields
            Fk = view(Fφ_batch, :, :, k)
            ifft_phi!(Fk, Fk)
        end
    end
    return Fφ_batch
end

function _batch_ifft_phi_to_output!(f_out::AbstractArray, Fφ_batch::AbstractArray{<:Complex,3}, real_output::Bool)
    _batch_ifft_phi!(Fφ_batch)
    nlat, nlon, nfields = size(Fφ_batch)
    if real_output
        @inbounds for k in 1:nfields, j in 1:nlon, i in 1:nlat
            f_out[i, j, k] = real(Fφ_batch[i, j, k])
        end
    else
        @inbounds for k in 1:nfields, j in 1:nlon, i in 1:nlat
            f_out[i, j, k] = Fφ_batch[i, j, k]
        end
    end
    return f_out
end

function _batch_irfft_phi!(f_out::AbstractArray{<:Real,3}, Fφ_batch::AbstractArray{<:Complex,3}, nlon::Int)
    nfields = size(Fφ_batch, 3)
    nfields == 0 && return f_out

    try
        plan = plan_irfft(view(Fφ_batch, :, :, 1), nlon, 2; flags=FFTW.ESTIMATE)
        @inbounds for k in 1:nfields
            mul!(view(f_out, :, :, k), plan, view(Fφ_batch, :, :, k))
        end
        _FFT_BACKEND[] = _FFT_BACKEND_FFTW
    catch e
        _batch_fft_fallback_error(e) || rethrow(e)
        @inbounds for k in 1:nfields
            irfft_phi!(view(f_out, :, :, k), view(Fφ_batch, :, :, k), nlon)
        end
    end
    return f_out
end

"""
    set_batch_size!(cfg::SHTConfig, howmany::Int; spec_dist::Int=0)

Configure batch processing for multiple fields. After calling this function,
batch transform functions (`analysis_batch`, `synthesis_batch`, etc.) will
process `howmany` fields simultaneously.

# Arguments
- `cfg`: SHTConfig to modify
- `howmany`: Number of fields to process in each batch (≥1)
- `spec_dist`: Distance between spectral coefficient arrays in memory.
               If 0, arrays are assumed contiguous. Otherwise, this is the
               stride between the start of successive spectral arrays.

# Returns
- The configured batch size (always ≥1)

# Example
```julia
cfg = create_gauss_config(32, 34)
set_batch_size!(cfg, 4)  # Process 4 fields at once

# Now use batch transforms
fields = rand(cfg.nlat, cfg.nlon, 4)
alms = analysis_batch(cfg, fields)
```
"""
function set_batch_size!(cfg::SHTConfig, howmany::Int; spec_dist::Int=0)
    howmany ≥ 1 || throw(ArgumentError("howmany must be ≥ 1"))
    spec_dist ≥ 0 || throw(ArgumentError("spec_dist must be ≥ 0"))
    cfg.howmany = howmany
    cfg.spec_dist = spec_dist
    return howmany
end

"""
    get_batch_size(cfg::SHTConfig) -> Int

Return the current batch size configuration.
"""
get_batch_size(cfg::SHTConfig) = cfg.howmany

"""
    reset_batch_size!(cfg::SHTConfig)

Reset batch size to 1 (single field processing).
"""
function reset_batch_size!(cfg::SHTConfig)
    cfg.howmany = 1
    cfg.spec_dist = 0
    return 1
end

# ============================================================================
# BATCH SCALAR TRANSFORMS
# ============================================================================

"""
    analysis_batch(cfg::SHTConfig, fields::AbstractArray{<:Real,3}) -> Array{ComplexF64,3}

Batch forward transform for multiple scalar fields.

# Arguments
- `cfg`: SHTConfig with batch size configured
- `fields`: 3D array of shape `(nlat, nlon, nfields)` containing spatial data

# Returns
- 3D array of shape `(lmax+1, mmax+1, nfields)` with spectral coefficients

# Performance
Batch processing is more efficient than processing fields individually because:
1. Legendre polynomials are computed once and reused across all fields
2. FFT plans are shared
3. Better cache utilization from contiguous memory access patterns
"""
function analysis_batch(cfg::SHTConfig, fields::AbstractArray{<:Real,3}; use_rfft::Bool=false)
    nlat, nlon, nfields = size(fields)
    nlat == cfg.nlat || throw(DimensionMismatch("first dim must be nlat=$(cfg.nlat)"))
    nlon == cfg.nlon || throw(DimensionMismatch("second dim must be nlon=$(cfg.nlon)"))

    lmax, mmax = cfg.lmax, cfg.mmax
    alm_batch = Array{ComplexF64,3}(undef, lmax + 1, mmax + 1, nfields)
    fill!(alm_batch, zero(ComplexF64))  # Initialize to zero for += accumulation in on-the-fly path

    # Allocate FFT scratch buffer — half width when using rfft.
    nbins = use_rfft ? (nlon ÷ 2 + 1) : nlon
    Fφ_batch = Array{ComplexF64,3}(undef, nlat, nbins, nfields)

    # Perform FFT on all fields
    if use_rfft
        _batch_rfft_phi!(Fφ_batch, fields)
    else
        _batch_fft_phi!(Fφ_batch, fields)
    end

    scaleφ = cfg.cphi
    w = cfg.w
    Nlm = cfg.Nlm

    if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
        # Use precomputed tables - most efficient path
        for m in 0:mmax
            col = m + 1
            tbl = cfg.plm_tables[m+1]
            for k in 1:nfields
                @inbounds for l in m:lmax
                    acc = zero(ComplexF64)
                    for i in 1:nlat
                        acc += (w[i] * tbl[l+1, i]) * Fφ_batch[i, col, k]
                    end
                    alm_batch[l+1, col, k] = acc * Nlm[l+1, col] * scaleφ
                end
            end
        end
    else
        # Compute Legendre polynomials on the fly
        # Use maxthreadid() to handle all possible thread IDs
        thread_local_P = _ensure_otf_scratch!(cfg._otf_scratch_P, lmax)
        for m in 0:mmax
            col = m + 1
            P = thread_local_P[Threads.threadid()]

            # For each latitude, compute P_l^m once and apply to all fields
            for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)
                wi = w[i]

                @inbounds for k in 1:nfields
                    Fi = Fφ_batch[i, col, k]
                    for l in m:lmax
                        alm_batch[l+1, col, k] += (wi * P[l+1]) * Fi
                    end
                end
            end

            # Apply normalization and scaling
            @inbounds for k in 1:nfields
                for l in m:lmax
                    alm_batch[l+1, col, k] *= Nlm[l+1, col] * scaleφ
                end
            end
        end
    end

    return alm_batch
end

"""
    analysis_batch!(cfg::SHTConfig, alm_out::AbstractArray{<:Complex,3},
                    fields::AbstractArray{<:Real,3})

In-place batch forward transform.
"""
function analysis_batch!(cfg::SHTConfig, alm_out::AbstractArray{<:Complex,3},
                         fields::AbstractArray{<:Real,3};
                         fft_batch::Union{Nothing,AbstractArray{<:Complex,3}}=nothing,
                         use_rfft::Bool=false)
    nlat, nlon, nfields = size(fields)
    nlat == cfg.nlat || throw(DimensionMismatch("first dim must be nlat=$(cfg.nlat)"))
    nlon == cfg.nlon || throw(DimensionMismatch("second dim must be nlon=$(cfg.nlon)"))

    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm_out, 1) == lmax + 1 || throw(DimensionMismatch("alm first dim must be lmax+1=$(lmax+1)"))
    size(alm_out, 2) == mmax + 1 || throw(DimensionMismatch("alm second dim must be mmax+1=$(mmax+1)"))
    size(alm_out, 3) == nfields || throw(DimensionMismatch("alm third dim must match nfields=$nfields"))

    fill!(alm_out, zero(eltype(alm_out)))

    # Reuse caller-provided scratch if given, else allocate.
    nbins = use_rfft ? (nlon ÷ 2 + 1) : nlon
    if fft_batch === nothing
        Fφ_batch = Array{ComplexF64,3}(undef, nlat, nbins, nfields)
    else
        size(fft_batch) == (nlat, nbins, nfields) || throw(DimensionMismatch("fft_batch size must be (nlat, $(nbins), nfields)"))
        Fφ_batch = fft_batch
    end

    # Perform FFT on all fields
    if use_rfft
        _batch_rfft_phi!(Fφ_batch, fields)
    else
        _batch_fft_phi!(Fφ_batch, fields)
    end

    scaleφ = cfg.cphi
    w = cfg.w
    Nlm = cfg.Nlm

    if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
        for m in 0:mmax
            col = m + 1
            tbl = cfg.plm_tables[m+1]
            for k in 1:nfields
                @inbounds for l in m:lmax
                    acc = zero(ComplexF64)
                    for i in 1:nlat
                        acc += (w[i] * tbl[l+1, i]) * Fφ_batch[i, col, k]
                    end
                    alm_out[l+1, col, k] = acc * Nlm[l+1, col] * scaleφ
                end
            end
        end
    else
        # Use maxthreadid() to handle all possible thread IDs
        thread_local_P = _ensure_otf_scratch!(cfg._otf_scratch_P, lmax)
        for m in 0:mmax
            col = m + 1
            P = thread_local_P[Threads.threadid()]

            for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)
                wi = w[i]

                @inbounds for k in 1:nfields
                    Fi = Fφ_batch[i, col, k]
                    for l in m:lmax
                        alm_out[l+1, col, k] += (wi * P[l+1]) * Fi
                    end
                end
            end

            @inbounds for k in 1:nfields
                for l in m:lmax
                    alm_out[l+1, col, k] *= Nlm[l+1, col] * scaleφ
                end
            end
        end
    end

    return alm_out
end

"""
    synthesis_batch(cfg::SHTConfig, alm_batch::AbstractArray{<:Complex,3};
                    real_output::Bool=true) -> Array

Batch inverse transform for multiple spectral coefficient sets.

# Arguments
- `cfg`: SHTConfig with batch size configured
- `alm_batch`: 3D array of shape `(lmax+1, mmax+1, nfields)` with spectral coefficients
- `real_output`: If true, return real spatial fields; otherwise complex

# Returns
- 3D array of shape `(nlat, nlon, nfields)` with spatial data
"""
Base.@constprop :aggressive function synthesis_batch(cfg::SHTConfig, alm_batch::AbstractArray{<:Complex,3};
                         real_output::Bool=true, use_rfft::Bool=false)
    # Preserve the ergonomic keyword API while giving literal keyword callers
    # a concrete return type through the Val-dispatched implementation.
    return _synthesis_batch(cfg, alm_batch, Val(real_output), Val(use_rfft))
end

"""
    synthesis_batch_cplx(cfg::SHTConfig, alm_batch::AbstractArray{<:Complex,3}) -> Array{ComplexF64,3}

Complex-output batch inverse transform. Equivalent to
`synthesis_batch(cfg, alm_batch; real_output=false)` with a concrete return
type for inference-sensitive callers.
"""
function synthesis_batch_cplx(cfg::SHTConfig, alm_batch::AbstractArray{<:Complex,3})
    return _synthesis_batch(cfg, alm_batch, Val(false), Val(false))
end

function _synthesis_batch(cfg::SHTConfig, alm_batch::AbstractArray{<:Complex,3},
                          ::Val{real_output}, ::Val{use_rfft}) where {real_output, use_rfft}
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm_batch, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm_batch, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))

    nfields = size(alm_batch, 3)
    nlat, nlon = cfg.nlat, cfg.nlon

    if use_rfft
        # Batch rfft synthesis stores only nonnegative m columns; irfft handles
        # the conjugate half per field.
        real_output === true || throw(ArgumentError("use_rfft=true implies real_output"))
        mmax ≤ nlon ÷ 2 || throw(ArgumentError("use_rfft=true requires mmax ≤ nlon÷2"))
    end

    # Allocate FFT scratch buffer
    nbins = use_rfft ? (nlon ÷ 2 + 1) : nlon
    Fφ_batch = Array{ComplexF64,3}(undef, nlat, nbins, nfields)
    fill!(Fφ_batch, zero(ComplexF64))
    inv_scaleφ = phi_inv_scale(cfg)
    Nlm = cfg.Nlm

    if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
        for m in 0:mmax
            col = m + 1
            tbl = cfg.plm_tables[m+1]
            @inbounds for k in 1:nfields
                for i in 1:nlat
                    acc = zero(ComplexF64)
                    for l in m:lmax
                        acc += (Nlm[l+1, col] * tbl[l+1, i]) * alm_batch[l+1, col, k]
                    end
                    Fφ_batch[i, col, k] = inv_scaleφ * acc
                end
            end
        end
    else
        # Use maxthreadid() to handle all possible thread IDs
        thread_local_P = _ensure_otf_scratch!(cfg._otf_scratch_P, lmax)
        for m in 0:mmax
            col = m + 1
            P = thread_local_P[Threads.threadid()]

            for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)

                @inbounds for k in 1:nfields
                    acc = zero(ComplexF64)
                    for l in m:lmax
                        acc += (Nlm[l+1, col] * P[l+1]) * alm_batch[l+1, col, k]
                    end
                    Fφ_batch[i, col, k] = inv_scaleφ * acc
                end
            end
        end
    end

    # Enforce Hermitian symmetry for real output (complex path only; rfft buffer is half-spectrum)
    if real_output && !use_rfft
        @inbounds for k in 1:nfields
            for m in 1:mmax
                col = m + 1
                conj_index = nlon - m + 1
                for i in 1:nlat
                    Fφ_batch[i, conj_index, k] = conj(Fφ_batch[i, col, k])
                end
            end
        end
    end

    if use_rfft
        f_batch = Array{Float64,3}(undef, nlat, nlon, nfields)
        return _batch_irfft_phi!(f_batch, Fφ_batch, nlon)
    elseif real_output
        f_batch = Array{Float64,3}(undef, nlat, nlon, nfields)
        return _batch_ifft_phi_to_output!(f_batch, Fφ_batch, true)
    else
        return _batch_ifft_phi!(Fφ_batch)
    end
end

"""
    synthesis_batch!(cfg::SHTConfig, f_out::AbstractArray,
                     alm_batch::AbstractArray{<:Complex,3}; real_output::Bool=true)

In-place batch inverse transform.
"""
function synthesis_batch!(cfg::SHTConfig, f_out::AbstractArray,
                          alm_batch::AbstractArray{<:Complex,3};
                          real_output::Bool=true,
                          fft_batch::Union{Nothing,AbstractArray{<:Complex,3}}=nothing,
                          use_rfft::Bool=false)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm_batch, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm_batch, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))

    nfields = size(alm_batch, 3)
    nlat, nlon = cfg.nlat, cfg.nlon

    size(f_out, 1) == nlat || throw(DimensionMismatch("f_out first dim must be nlat=$nlat"))
    size(f_out, 2) == nlon || throw(DimensionMismatch("f_out second dim must be nlon=$nlon"))
    size(f_out, 3) == nfields || throw(DimensionMismatch("f_out third dim must be nfields=$nfields"))

    if use_rfft
        real_output || throw(ArgumentError("use_rfft=true implies real_output"))
        eltype(f_out) <: Real || throw(ArgumentError("use_rfft=true requires real-valued f_out"))
        mmax ≤ nlon ÷ 2 || throw(ArgumentError("use_rfft=true requires mmax ≤ nlon÷2"))
    end

    # Reuse caller-provided scratch if given, else allocate.
    nbins = use_rfft ? (nlon ÷ 2 + 1) : nlon
    if fft_batch === nothing
        Fφ_batch = Array{ComplexF64,3}(undef, nlat, nbins, nfields)
    else
        size(fft_batch) == (nlat, nbins, nfields) || throw(DimensionMismatch("fft_batch size must be (nlat, $(nbins), nfields)"))
        Fφ_batch = fft_batch
    end
    fill!(Fφ_batch, zero(eltype(Fφ_batch)))
    inv_scaleφ = phi_inv_scale(cfg)
    Nlm = cfg.Nlm

    if cfg.use_plm_tables && length(cfg.plm_tables) == mmax + 1
        for m in 0:mmax
            col = m + 1
            tbl = cfg.plm_tables[m+1]
            @inbounds for k in 1:nfields
                for i in 1:nlat
                    acc = zero(ComplexF64)
                    for l in m:lmax
                        acc += (Nlm[l+1, col] * tbl[l+1, i]) * alm_batch[l+1, col, k]
                    end
                    Fφ_batch[i, col, k] = inv_scaleφ * acc
                end
            end
        end
    else
        # Use maxthreadid() to handle all possible thread IDs with static scheduling
        thread_local_P = _ensure_otf_scratch!(cfg._otf_scratch_P, lmax)
        for m in 0:mmax
            col = m + 1
            P = thread_local_P[Threads.threadid()]

            for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)

                @inbounds for k in 1:nfields
                    acc = zero(ComplexF64)
                    for l in m:lmax
                        acc += (Nlm[l+1, col] * P[l+1]) * alm_batch[l+1, col, k]
                    end
                    Fφ_batch[i, col, k] = inv_scaleφ * acc
                end
            end
        end
    end

    # Enforce Hermitian symmetry for real output (complex path only; rfft buffer is half-spectrum)
    if real_output && !use_rfft
        @inbounds for k in 1:nfields
            for m in 1:mmax
                col = m + 1
                conj_index = nlon - m + 1
                for i in 1:nlat
                    Fφ_batch[i, conj_index, k] = conj(Fφ_batch[i, col, k])
                end
            end
        end
    end

    if use_rfft
        return _batch_irfft_phi!(f_out, Fφ_batch, nlon)
    end

    # Perform inverse FFT and copy to output
    return _batch_ifft_phi_to_output!(f_out, Fφ_batch, real_output)
end

# ============================================================================
# BATCH VECTOR TRANSFORMS
# ============================================================================

"""
    analysis_sphtor_batch(cfg::SHTConfig, Vt_batch::AbstractArray{<:Real,3},
                          Vp_batch::AbstractArray{<:Real,3})

Batch spheroidal-toroidal analysis for multiple vector fields.

# Arguments
- `cfg`: SHTConfig
- `Vt_batch`: 3D array `(nlat, nlon, nfields)` of theta components
- `Vp_batch`: 3D array `(nlat, nlon, nfields)` of phi components

# Returns
- `(Slm_batch, Tlm_batch)`: Tuple of 3D arrays `(lmax+1, mmax+1, nfields)`
  containing spheroidal and toroidal coefficients
"""
function analysis_sphtor_batch(cfg::SHTConfig, Vt_batch::AbstractArray{<:Real,3},
                               Vp_batch::AbstractArray{<:Real,3})
    nlat, nlon, nfields = size(Vt_batch)
    nlat == cfg.nlat || throw(DimensionMismatch("first dim must be nlat=$(cfg.nlat)"))
    nlon == cfg.nlon || throw(DimensionMismatch("second dim must be nlon=$(cfg.nlon)"))
    size(Vp_batch) == size(Vt_batch) || throw(DimensionMismatch("Vt and Vp must have same shape"))

    lmax, mmax = cfg.lmax, cfg.mmax
    Slm_batch = zeros(ComplexF64, lmax + 1, mmax + 1, nfields)
    Tlm_batch = zeros(ComplexF64, lmax + 1, mmax + 1, nfields)
    plan = SHTPlan(cfg)

    for k in 1:nfields
        analysis_sphtor!(plan, view(Slm_batch, :, :, k), view(Tlm_batch, :, :, k),
                         view(Vt_batch, :, :, k), view(Vp_batch, :, :, k))
    end

    return Slm_batch, Tlm_batch
end

"""
    synthesis_sphtor_batch(cfg::SHTConfig, Slm_batch::AbstractArray{<:Complex,3},
                           Tlm_batch::AbstractArray{<:Complex,3}; real_output::Bool=true)

Batch spheroidal-toroidal synthesis for multiple vector fields.

# Returns
- `(Vt_batch, Vp_batch)`: Tuple of 3D arrays `(nlat, nlon, nfields)`
  containing theta and phi components
"""
Base.@constprop :aggressive function synthesis_sphtor_batch(cfg::SHTConfig, Slm_batch::AbstractArray{<:Complex,3},
                                Tlm_batch::AbstractArray{<:Complex,3}; real_output::Bool=true)
    # Val dispatch keeps the tuple element types concrete for real vs complex
    # vector batch synthesis.
    return _synthesis_sphtor_batch(cfg, Slm_batch, Tlm_batch, Val(real_output))
end

"""Complex-output batch spheroidal/toroidal synthesis with a concrete return type."""
function synthesis_sphtor_batch_cplx(cfg::SHTConfig, Slm_batch::AbstractArray{<:Complex,3},
                                     Tlm_batch::AbstractArray{<:Complex,3})
    return _synthesis_sphtor_batch(cfg, Slm_batch, Tlm_batch, Val(false))
end

function _synthesis_sphtor_batch(cfg::SHTConfig, Slm_batch::AbstractArray{<:Complex,3},
                                 Tlm_batch::AbstractArray{<:Complex,3},
                                 ::Val{real_output}) where {real_output}
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Slm_batch, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1"))
    size(Slm_batch, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1"))
    size(Tlm_batch) == size(Slm_batch) || throw(DimensionMismatch("Slm and Tlm must have same shape"))

    nfields = size(Slm_batch, 3)
    nlat, nlon = cfg.nlat, cfg.nlon

    if real_output
        Vt_batch = Array{Float64,3}(undef, nlat, nlon, nfields)
        Vp_batch = Array{Float64,3}(undef, nlat, nlon, nfields)
    else
        Vt_batch = Array{ComplexF64,3}(undef, nlat, nlon, nfields)
        Vp_batch = Array{ComplexF64,3}(undef, nlat, nlon, nfields)
    end
    plan = SHTPlan(cfg)

    for k in 1:nfields
        synthesis_sphtor!(plan, view(Vt_batch, :, :, k), view(Vp_batch, :, :, k),
                          view(Slm_batch, :, :, k), view(Tlm_batch, :, :, k);
                          real_output=real_output)
    end

    return Vt_batch, Vp_batch
end

# ============================================================================
# BATCH QST (3D VECTOR) TRANSFORMS
# ============================================================================

"""
    analysis_qst_batch(cfg::SHTConfig, Vr_batch::AbstractArray{<:Real,3},
                       Vt_batch::AbstractArray{<:Real,3}, Vp_batch::AbstractArray{<:Real,3})

Batch QST analysis for multiple 3D vector fields.

# Returns
- `(Qlm_batch, Slm_batch, Tlm_batch)`: Tuple of 3D arrays with Q, S, T coefficients
"""
function analysis_qst_batch(cfg::SHTConfig, Vr_batch::AbstractArray{<:Real,3},
                            Vt_batch::AbstractArray{<:Real,3}, Vp_batch::AbstractArray{<:Real,3})
    nlat, nlon, nfields = size(Vr_batch)
    nlat == cfg.nlat || throw(DimensionMismatch("first dim must be nlat=$(cfg.nlat)"))
    nlon == cfg.nlon || throw(DimensionMismatch("second dim must be nlon=$(cfg.nlon)"))
    size(Vt_batch) == size(Vr_batch) || throw(DimensionMismatch("Vr and Vt must have same shape"))
    size(Vp_batch) == size(Vr_batch) || throw(DimensionMismatch("Vr and Vp must have same shape"))

    lmax, mmax = cfg.lmax, cfg.mmax
    Qlm_batch = zeros(ComplexF64, lmax + 1, mmax + 1, nfields)
    Slm_batch = zeros(ComplexF64, lmax + 1, mmax + 1, nfields)
    Tlm_batch = zeros(ComplexF64, lmax + 1, mmax + 1, nfields)
    plan = SHTPlan(cfg)

    for k in 1:nfields
        analysis!(plan, view(Qlm_batch, :, :, k), view(Vr_batch, :, :, k))
        analysis_sphtor!(plan, view(Slm_batch, :, :, k), view(Tlm_batch, :, :, k),
                         view(Vt_batch, :, :, k), view(Vp_batch, :, :, k))
    end

    return Qlm_batch, Slm_batch, Tlm_batch
end

"""
    synthesis_qst_batch(cfg::SHTConfig, Qlm_batch::AbstractArray{<:Complex,3},
                        Slm_batch::AbstractArray{<:Complex,3}, Tlm_batch::AbstractArray{<:Complex,3};
                        real_output::Bool=true)

Batch QST synthesis for multiple 3D vector fields.

# Returns
- `(Vr_batch, Vt_batch, Vp_batch)`: Tuple of 3D arrays with spatial components
"""
Base.@constprop :aggressive function synthesis_qst_batch(cfg::SHTConfig, Qlm_batch::AbstractArray{<:Complex,3},
                             Slm_batch::AbstractArray{<:Complex,3}, Tlm_batch::AbstractArray{<:Complex,3};
                             real_output::Bool=true)
    # QST batch synthesis composes scalar Q with sphtor S/T; the Val barrier
    # keeps all three returned component arrays type-stable.
    return _synthesis_qst_batch(cfg, Qlm_batch, Slm_batch, Tlm_batch, Val(real_output))
end

"""Complex-output batch QST synthesis with a concrete return type."""
function synthesis_qst_batch_cplx(cfg::SHTConfig, Qlm_batch::AbstractArray{<:Complex,3},
                                  Slm_batch::AbstractArray{<:Complex,3},
                                  Tlm_batch::AbstractArray{<:Complex,3})
    return _synthesis_qst_batch(cfg, Qlm_batch, Slm_batch, Tlm_batch, Val(false))
end

function _synthesis_qst_batch(cfg::SHTConfig, Qlm_batch::AbstractArray{<:Complex,3},
                              Slm_batch::AbstractArray{<:Complex,3}, Tlm_batch::AbstractArray{<:Complex,3},
                              ::Val{real_output}) where {real_output}
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Qlm_batch, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1"))
    size(Qlm_batch, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1"))
    size(Slm_batch) == size(Qlm_batch) || throw(DimensionMismatch("Qlm and Slm must have same shape"))
    size(Tlm_batch) == size(Qlm_batch) || throw(DimensionMismatch("Qlm and Tlm must have same shape"))

    nfields = size(Qlm_batch, 3)
    nlat, nlon = cfg.nlat, cfg.nlon

    if real_output
        Vr_batch = Array{Float64,3}(undef, nlat, nlon, nfields)
        Vt_batch = Array{Float64,3}(undef, nlat, nlon, nfields)
        Vp_batch = Array{Float64,3}(undef, nlat, nlon, nfields)
    else
        Vr_batch = Array{ComplexF64,3}(undef, nlat, nlon, nfields)
        Vt_batch = Array{ComplexF64,3}(undef, nlat, nlon, nfields)
        Vp_batch = Array{ComplexF64,3}(undef, nlat, nlon, nfields)
    end
    plan = SHTPlan(cfg)

    for k in 1:nfields
        synthesis!(plan, view(Vr_batch, :, :, k), view(Qlm_batch, :, :, k);
                   real_output=real_output)
        synthesis_sphtor!(plan, view(Vt_batch, :, :, k), view(Vp_batch, :, :, k),
                          view(Slm_batch, :, :, k), view(Tlm_batch, :, :, k);
                          real_output=real_output)
    end

    return Vr_batch, Vt_batch, Vp_batch
end
