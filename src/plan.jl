"""
Optimized Transform Planning for Spherical Harmonic Operations

This module implements a planning system for spherical harmonic transforms that
pre-allocates working arrays and FFT plans to minimize runtime overhead. The
planning approach is inspired by FFTW's philosophy: spend time upfront to
optimize repeated operations.

Benefits of Planning:
- Eliminates repeated memory allocations during transforms
- Pre-optimizes FFTW plans for maximum performance
- Improves cache locality by reusing buffers
- Reduces garbage collection pressure in performance-critical loops

The SHTPlan stores all necessary working arrays and can handle both complex
FFTs and real-optimized FFTs (RFFT) depending on the use case.
"""

abstract type AbstractSHTPlan end

struct SHTPlan <: AbstractSHTPlan
    cfg::SHTConfig                # Configuration parameters
    P::Vector{Float64}           # Working array for Legendre polynomials P_l^m(x)
    dPdx::Vector{Float64}        # Working array for derivatives dP_l^m/dx  
    G::Vector{ComplexF64}        # Temporary array for latitudinal profiles
    Fθk::Matrix{ComplexF64}      # Fourier coefficient matrix [latitude × longitude]
    fft_plan::Any               # Pre-optimized forward FFT plan (or nothing for RFFT)
    ifft_plan::Any              # Pre-optimized inverse FFT plan (or nothing for RFFT)  
    use_rfft::Bool              # Flag: true = use real FFT optimization, false = complex FFT
end

"""
    SHTPlan(cfg::SHTConfig; use_rfft=false)

Create an optimized transform plan with pre-allocated buffers and FFT plans.

This constructor performs the "planning" phase: it allocates all working memory
and optimizes FFTW plans for the specific grid configuration. The resulting
plan can then be reused for many transforms without additional allocations.

Parameters:
- cfg: SHTConfig defining the grid and spectral resolution
- use_rfft: Enable real-FFT optimization for real-valued output fields

Real FFT optimization (use_rfft=true):
- Allocates smaller Fourier buffer (N/2+1 instead of N complex numbers)
- Skips complex FFTW planning (uses RFFT functions directly)
- Reduces memory usage and improves performance for real-valued synthesis

Complex FFT mode (use_rfft=false):
- Full-spectrum Fourier buffer for maximum flexibility
- Pre-optimizes both forward and inverse FFT plans
- Required for complex-valued fields or analysis operations
"""
function SHTPlan(cfg::SHTConfig; use_rfft::Bool=false)
    if is_gpu_config(cfg)
        return gpu_create_plan(cfg; use_rfft=use_rfft)
    end
    # Allocate working arrays for Legendre polynomial computation
    P = Vector{Float64}(undef, cfg.lmax + 1)     # P_l^m(cos θ) values
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)  # dP_l^m/d(cos θ) derivatives
    G = Vector{ComplexF64}(undef, cfg.nlat)      # Temporary latitudinal profiles
    
    if use_rfft
        # Real FFT optimization path
        nlon_half = fld(cfg.nlon, 2) + 1  # Only need positive frequencies + Nyquist
        Fθk = Matrix{ComplexF64}(undef, cfg.nlat, nlon_half)
        fill!(Fθk, 0)  # Initialize to zero
        
        # No FFT planning needed (will use FFTW.rfft/irfft functions directly)
        return SHTPlan(cfg, P, dPdx, G, Fθk, nothing, nothing, true)
        
    else
        # Full complex FFT path 
        Fθk = Matrix{ComplexF64}(undef, cfg.nlat, cfg.nlon)
        fill!(Fθk, 0)  # Initialize to zero
        
        # Pre-optimize FFTW plans for this specific array layout
        # Planning may take time but subsequent transforms will be faster
        fft_plan = FFTW.plan_fft!(Fθk, 2)   # Forward FFT along longitude (dim 2)
        ifft_plan = FFTW.plan_ifft!(Fθk, 2) # Inverse FFT along longitude (dim 2)
        
        return SHTPlan(cfg, P, dPdx, G, Fθk, fft_plan, ifft_plan, false)
    end
end

gpu_create_plan(::SHTConfig; use_rfft::Bool=false) = error("GPU plan support requires SHTnsKitCUDAExt to be loaded.")

destroy_plan!(::AbstractSHTPlan) = nothing

function destroy_plan!(plan::SHTPlan)
    plan.fft_plan !== nothing && FFTW.destroy_plan(plan.fft_plan)
    plan.ifft_plan !== nothing && FFTW.destroy_plan(plan.ifft_plan)
    return nothing
end

"""
    spat_to_SHsphtor!(plan::SHTPlan, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix)

In-place vector analysis. Accumulates Slm/Tlm into preallocated outputs.
Uses a two-pass strategy over φ FFTs to avoid extra buffers.
"""
function spat_to_SHsphtor!(plan::SHTPlan, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix)
    cfg = plan.cfg
    nlat, nlon = cfg.nlat, cfg.nlon
    size(Vt,1)==nlat && size(Vt,2)==nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1)==nlat && size(Vp,2)==nlon || throw(DimensionMismatch("Vp dims"))
    size(Slm_out,1)==cfg.lmax+1 && size(Slm_out,2)==cfg.mmax+1 || throw(DimensionMismatch("Slm_out dims"))
    size(Tlm_out,1)==cfg.lmax+1 && size(Tlm_out,2)==cfg.mmax+1 || throw(DimensionMismatch("Tlm_out dims"))
    lmax, mmax = cfg.lmax, cfg.mmax
    scaleφ = cfg.cphi
    fill!(Slm_out, 0); fill!(Tlm_out, 0)
    # First pass: FFT(Vt) -> partial contributions using Fθ only
    @inbounds for i in 1:nlat, j in 1:nlon
        plan.Fθk[i,j] = Vt[i,j]
    end
    # Robert form: divide by sinθ before analysis
    if cfg.robert_form
        @inbounds for i in 1:nlat
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
            if sθ > 0
                plan.Fθk[i, :] ./= sθ
            end
        end
    end
    plan.fft_plan * plan.Fθk
    for m in 0:mmax
        col = m + 1
        for i in 1:nlat
            Plm_and_dPdx_row!(plan.P, plan.dPdx, cfg.x[i], lmax, m)
            Fθ_i = plan.Fθk[i, col]
            wi = cfg.w[i]
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2)); inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            @inbounds for l in max(1,m):lmax
                N = cfg.Nlm[l+1, col]
                dθY = -sθ * N * plan.dPdx[l+1]
                Y = N * plan.P[l+1]
                coeff = wi * scaleφ / (l*(l+1))
                Tlm_out[l+1, col] += coeff * ((0 + 1im) * m * inv_sθ * Y * Fθ_i)
                Slm_out[l+1, col] += coeff * (Fθ_i * dθY)
            end
        end
    end
    # Second pass: FFT(Vp) -> add remaining contributions using Fφ
    @inbounds for i in 1:nlat, j in 1:nlon
        plan.Fθk[i,j] = Vp[i,j]
    end
    if cfg.robert_form
        @inbounds for i in 1:nlat
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
            if sθ > 0
                plan.Fθk[i, :] ./= sθ
            end
        end
    end
    plan.fft_plan * plan.Fθk
    for m in 0:mmax
        col = m + 1
        for i in 1:nlat
            Plm_and_dPdx_row!(plan.P, plan.dPdx, cfg.x[i], lmax, m)
            Fφ_i = plan.Fθk[i, col]
            wi = cfg.w[i]
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2)); inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            @inbounds for l in max(1,m):lmax
                N = cfg.Nlm[l+1, col]
                dθY = -sθ * N * plan.dPdx[l+1]
                Y = N * plan.P[l+1]
                coeff = wi * scaleφ / (l*(l+1))
                Slm_out[l+1, col] += -coeff * ((0 + 1im) * m * inv_sθ * Y * Fφ_i)
                Tlm_out[l+1, col] += coeff * (Fφ_i * (+sθ * N * plan.dPdx[l+1]))
            end
        end
    end
    # Convert to cfg normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        tmpS = similar(Slm_out); tmpT = similar(Tlm_out)
        convert_alm_norm!(tmpS, Slm_out, cfg; to_internal=false)
        convert_alm_norm!(tmpT, Tlm_out, cfg; to_internal=false)
        copyto!(Slm_out, tmpS); copyto!(Tlm_out, tmpT)
    end
    return Slm_out, Tlm_out
end

"""
    SHsphtor_to_spat!(plan::SHTPlan, Vt_out::AbstractMatrix, Vp_out::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output=true)

In-place vector synthesis. Streams m→k without forming (θ×m) intermediates; inverse FFT Vt then Vp.
"""
function SHsphtor_to_spat!(plan::SHTPlan, Vt_out::AbstractMatrix, Vp_out::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    cfg = plan.cfg
    nlat, nlon = cfg.nlat, cfg.nlon
    size(Vt_out,1)==nlat && size(Vt_out,2)==nlon || throw(DimensionMismatch("Vt_out dims"))
    size(Vp_out,1)==nlat && size(Vp_out,2)==nlon || throw(DimensionMismatch("Vp_out dims"))
    size(Slm,1)==cfg.lmax+1 && size(Slm,2)==cfg.mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1)==cfg.lmax+1 && size(Tlm,2)==cfg.mmax+1 || throw(DimensionMismatch("Tlm dims"))
    lmax, mmax = cfg.lmax, cfg.mmax
    inv_scaleφ = phi_inv_scale(cfg.nlon)
    # Convert to internal normalization if needed
    Slm_int = Slm; Tlm_int = Tlm
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        tmpS = similar(Slm); tmpT = similar(Tlm)
        convert_alm_norm!(tmpS, Slm, cfg; to_internal=true)
        convert_alm_norm!(tmpT, Tlm, cfg; to_internal=true)
        Slm_int = tmpS; Tlm_int = tmpT
    end
    # Synthesize Vt: stream m→k then inverse FFT
    fill!(plan.Fθk, 0)
    for m in 0:mmax
        col = m + 1
        for i in 1:nlat
            Plm_and_dPdx_row!(plan.P, plan.dPdx, cfg.x[i], lmax, m)
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2)); inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            g = 0.0 + 0.0im
            @inbounds for l in max(1,m):lmax
                N = cfg.Nlm[l+1, col]
                dθY = -sθ * N * plan.dPdx[l+1]
                Y = N * plan.P[l+1]
                g += dθY * Slm_int[l+1, col] + (0 + 1im) * m * inv_sθ * Y * Tlm_int[l+1, col]
            end
            plan.G[i] = g
        end
        @inbounds for i in 1:nlat
            plan.Fθk[i, col] = inv_scaleφ * plan.G[i]
        end
        if real_output && m > 0
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                plan.Fθk[i, conj_index] = conj(plan.Fθk[i, col])
            end
        end
    end
    if plan.use_rfft && real_output
        Vt_tmp = FFTW.irfft(plan.Fθk, nlon, 2)
        @inbounds for i in 1:nlat, j in 1:nlon
            plan.Fθk[i,j] = Vt_tmp[i,j]
        end
    else
        plan.ifft_plan * plan.Fθk
    end
    if cfg.robert_form
        @inbounds for i in 1:nlat
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
            for j in 1:nlon
                plan.Fθk[i,j] *= sθ
            end
        end
    end
    if real_output
        @inbounds for i in 1:nlat, j in 1:nlon
            Vt_out[i,j] = real(plan.Fθk[i,j])
        end
    else
        @inbounds for i in 1:nlat, j in 1:nlon
            Vt_out[i,j] = plan.Fθk[i,j]
        end
    end
    # Synthesize Vp similarly
    fill!(plan.Fθk, 0)
    for m in 0:mmax
        col = m + 1
        for i in 1:nlat
            Plm_and_dPdx_row!(plan.P, plan.dPdx, cfg.x[i], lmax, m)
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2)); inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            g = 0.0 + 0.0im
            @inbounds for l in max(1,m):lmax
                N = cfg.Nlm[l+1, col]
                dθY = -sθ * N * plan.dPdx[l+1]
                Y = N * plan.P[l+1]
                g += (0 + 1im) * m * inv_sθ * Y * Slm_int[l+1, col] + (sθ * N * plan.dPdx[l+1]) * Tlm_int[l+1, col]
            end
            plan.G[i] = g
        end
        @inbounds for i in 1:nlat
            plan.Fθk[i, col] = inv_scaleφ * plan.G[i]
        end
        if real_output && m > 0
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                plan.Fθk[i, conj_index] = conj(plan.Fθk[i, col])
            end
        end
    end
    if plan.use_rfft && real_output
        Vt_tmp = FFTW.irfft(plan.Fθk, nlon, 2)
        @inbounds for i in 1:nlat, j in 1:nlon
            plan.Fθk[i,j] = Vt_tmp[i,j]
        end
    else
        plan.ifft_plan * plan.Fθk
    end
    if cfg.robert_form
        @inbounds for i in 1:nlat
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
            for j in 1:nlon
                plan.Fθk[i,j] *= sθ
            end
        end
    end
    if real_output
        @inbounds for i in 1:nlat, j in 1:nlon
            Vp_out[i,j] = real(plan.Fθk[i,j])
        end
    else
        @inbounds for i in 1:nlat, j in 1:nlon
            Vp_out[i,j] = plan.Fθk[i,j]
        end
    end
    return Vt_out, Vp_out
end

"""
    analysis!(plan::SHTPlan, alm_out::AbstractMatrix, f::AbstractMatrix)

In-place forward scalar SHT writing coefficients into `alm_out`.
"""
function analysis!(plan::SHTPlan, alm_out::AbstractMatrix, f::AbstractMatrix)
    cfg = plan.cfg
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f,1)==nlat || throw(DimensionMismatch("f first dim must be nlat"))
    size(f,2)==nlon || throw(DimensionMismatch("f second dim must be nlon"))
    size(alm_out,1)==cfg.lmax+1 || throw(DimensionMismatch("alm rows must be lmax+1"))
    size(alm_out,2)==cfg.mmax+1 || throw(DimensionMismatch("alm cols must be mmax+1"))
    # Copy f into complex buffer and FFT along φ in-place
    @inbounds for i in 1:nlat, j in 1:nlon
        plan.Fθk[i,j] = f[i,j]
    end
    plan.fft_plan * plan.Fθk  # execute planned FFT in-place
    # Compute alm
    fill!(alm_out, 0)
    lmax, mmax = cfg.lmax, cfg.mmax
    scaleφ = cfg.cphi
    for m in 0:mmax
        col = m + 1
        for i in 1:nlat
            Plm_row!(plan.P, cfg.x[i], lmax, m)
            Fi = plan.Fθk[i, col]
            wi = cfg.w[i]
            @inbounds for l in m:lmax
                alm_out[l+1, col] += (wi * plan.P[l+1]) * Fi
            end
        end
        @inbounds for l in m:lmax
            alm_out[l+1, col] *= cfg.Nlm[l+1, col] * scaleφ
        end
    end
    # Convert to cfg normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        tmp = similar(alm_out)
        convert_alm_norm!(tmp, alm_out, cfg; to_internal=false)
        copyto!(alm_out, tmp)
    end
    return alm_out
end

"""
    synthesis!(plan::SHTPlan, f_out::AbstractMatrix, alm::AbstractMatrix; real_output=true)

In-place inverse scalar SHT writing spatial field into `f_out`.
Streams m→k directly without building a (θ×m) intermediate.
"""
function synthesis!(plan::SHTPlan, f_out::AbstractMatrix, alm::AbstractMatrix; real_output::Bool=true)
    cfg = plan.cfg
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f_out,1)==nlat || throw(DimensionMismatch("f_out first dim must be nlat"))
    size(f_out,2)==nlon || throw(DimensionMismatch("f_out second dim must be nlon"))
    size(alm,1)==cfg.lmax+1 || throw(DimensionMismatch("alm rows must be lmax+1"))
    size(alm,2)==cfg.mmax+1 || throw(DimensionMismatch("alm cols must be mmax+1"))
    # Convert alm to internal normalization if needed
    alm_int = alm
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        tmp = similar(alm)
        convert_alm_norm!(tmp, alm, cfg; to_internal=true)
        alm_int = tmp
    end
    # Zero Fourier buffer
    fill!(plan.Fθk, 0)
    lmax, mmax = cfg.lmax, cfg.mmax
    inv_scaleφ = phi_inv_scale(nlon)
    # Stream over m, fill k-bins directly
    for m in 0:mmax
        col = m + 1
        # Build Gm(θ)
        for i in 1:nlat
            Plm_row!(plan.P, cfg.x[i], lmax, m)
            g = 0.0 + 0.0im
            @inbounds for l in m:lmax
                g += (cfg.Nlm[l+1, col] * plan.P[l+1]) * alm_int[l+1, col]
            end
            plan.G[i] = g
        end
        # Place positive m Fourier modes
        @inbounds for i in 1:nlat
            plan.Fθk[i, col] = inv_scaleφ * plan.G[i]
        end
        # Hermitian conjugate for negative m to ensure real output
        if real_output && m > 0
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                plan.Fθk[i, conj_index] = conj(plan.Fθk[i, col])
            end
        end
    end
    # Inverse FFT along φ in-place
    plan.ifft_plan * plan.Fθk
    # Write result
    if real_output
        @inbounds for i in 1:nlat, j in 1:nlon
            f_out[i,j] = real(plan.Fθk[i,j])
        end
    else
        @inbounds for i in 1:nlat, j in 1:nlon
            f_out[i,j] = plan.Fθk[i,j]
        end
    end
    return f_out
end
