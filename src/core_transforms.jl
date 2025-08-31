"""
Core Spherical Harmonic Transforms

This module implements the fundamental forward (analysis) and backward (synthesis)
spherical harmonic transforms for 2D spatial grids. These are the primary
transform functions that convert between spatial and spectral representations.

Key features:
- Supports both fused and unfused loop implementations for performance tuning
- Handles both real and complex input/output fields
- Uses Gauss-Legendre quadrature for accurate integration
- Optimized with precomputed Legendre polynomial tables when available
"""

"""
    analysis(cfg::SHTConfig, f::AbstractMatrix) -> Matrix{ComplexF64}

Forward spherical harmonic transform on Gauss–Legendre × equiangular grid.
Input grid `f` must be sized `(cfg.nlat, cfg.nlon)` and may be real or complex.
Returns coefficients `alm` of size `(cfg.lmax+1, cfg.mmax+1)` with indices `(l+1, m+1)`.
Normalization uses orthonormal spherical harmonics with Condon–Shortley phase.
"""
function analysis(cfg::SHTConfig, f::AbstractMatrix; use_fused_loops::Bool=true)
    if use_fused_loops
        return analysis_fused(cfg, f)
    else
        return analysis_unfused(cfg, f)
    end
end

function analysis_unfused(cfg::SHTConfig, f::AbstractMatrix)
    # Validate input dimensions match the configured grid
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))
    
    # Convert input to complex and perform FFT along longitude (φ) direction
    fC = complex.(f)
    Fφ = fft_phi(fC)  # Now Fφ[lat, m] contains Fourier modes

    # Allocate output array for spherical harmonic coefficients
    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(Fφ)
    alm = Matrix{CT}(undef, lmax + 1, mmax + 1)  # alm[l+1, m+1] for (l,m) indexing
    fill!(alm, 0.0 + 0.0im)

    scaleφ = cfg.cphi  # Longitude step size: 2π / nlon
    
    # Thread-local storage for Legendre polynomial buffers to avoid allocations
    # Each thread gets its own buffer to prevent race conditions
    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
    
    # Process each azimuthal mode m in parallel
    @threads for m in 0:mmax
        col = m + 1  # Julia 1-based indexing
        
        # Get thread-local buffer for this thread
        P = thread_local_P[Threads.threadid()]
        
        # Integrate over colatitude θ using Gauss-Legendre quadrature
        if cfg.use_plm_tables && length(cfg.plm_tables) == mmax+1
            # Fast path: use precomputed Legendre polynomial tables
            tbl = cfg.plm_tables[m+1]  # P_l^m(x_i) values
            @inbounds for i in 1:nlat
                Fφ_weighted = Fφ[i, col] * cfg.wlat[i]  # Include quadrature weight
                for l in m:lmax
                    alm[l+1, col] += Fφ_weighted * tbl[i, l+1]
                end
            end
        else
            # Standard path: compute Legendre polynomials on-the-fly
            @inbounds for i in 1:nlat
                x = cfg.x[i]  # cos(θ_i)
                Plm_row!(P, x, lmax, m)  # Compute P_l^m(x) for all l
                
                Fφ_weighted = Fφ[i, col] * cfg.wlat[i]
                for l in m:lmax
                    alm[l+1, col] += Fφ_weighted * P[l+1]
                end
            end
        end
        
        # Apply longitude scaling factor
        @inbounds for l in m:lmax
            alm[l+1, col] *= scaleφ
        end
    end
    
    return alm
end

function analysis_fused(cfg::SHTConfig, f::AbstractMatrix)
    # Validate input dimensions
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))
    
    # Convert to complex and perform longitude FFT
    fC = complex.(f)
    Fφ = fft_phi(fC)
    
    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(Fφ)
    alm = Matrix{CT}(undef, lmax + 1, mmax + 1)
    fill!(alm, 0.0 + 0.0im)
    
    scaleφ = cfg.cphi
    
    # Fused version: combine Legendre computation and integration in single loop
    if cfg.use_plm_tables && length(cfg.plm_tables) == mmax+1
        # Use precomputed tables for maximum performance
        @threads for m in 0:mmax
            col = m + 1
            tbl = cfg.plm_tables[m+1]
            @inbounds for l in m:lmax, i in 1:nlat
                alm[l+1, col] += Fφ[i, col] * cfg.wlat[i] * tbl[i, l+1]
            end
            @inbounds for l in m:lmax
                alm[l+1, col] *= scaleφ
            end
        end
    else
        # On-the-fly computation with fused loops
        thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
        
        @threads for m in 0:mmax
            col = m + 1
            P = thread_local_P[Threads.threadid()]
            
            for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)
                Fφ_weighted = Fφ[i, col] * cfg.wlat[i]
                
                @inbounds for l in m:lmax
                    alm[l+1, col] += Fφ_weighted * P[l+1]
                end
            end
            
            @inbounds for l in m:lmax
                alm[l+1, col] *= scaleφ
            end
        end
    end
    
    return alm
end

"""
    synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true) -> Matrix

Backward spherical harmonic transform from spectral coefficients to spatial grid.
Input `alm` must be sized `(cfg.lmax+1, cfg.mmax+1)`.
Returns spatial field of size `(cfg.nlat, cfg.nlon)`.
"""
function synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true, use_fused_loops::Bool=true)
    if use_fused_loops
        return synthesis_fused(cfg, alm; real_output=real_output)
    else
        return synthesis_unfused(cfg, alm; real_output=real_output)
    end
end

function synthesis_unfused(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true)
    # Validate input dimensions
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))
    
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    
    # Allocate intermediate array for Fourier modes F(θ,m)
    Fφ = Matrix{CT}(undef, nlat, mmax + 1)
    fill!(Fφ, zero(CT))
    
    # Thread-local Legendre polynomial buffers
    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
    
    # Process each azimuthal mode m in parallel
    @threads for m in 0:mmax
        col = m + 1
        P = thread_local_P[Threads.threadid()]
        
        if cfg.use_plm_tables && length(cfg.plm_tables) == mmax+1
            # Fast path with precomputed tables
            tbl = cfg.plm_tables[m+1]
            for i in 1:nlat
                Fφ[i, col] = zero(CT)
                for l in m:lmax
                    Fφ[i, col] += alm[l+1, col] * tbl[i, l+1]
                end
            end
        else
            # Standard path with on-the-fly computation
            for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)
                
                Fφ[i, col] = zero(CT)
                for l in m:lmax
                    Fφ[i, col] += alm[l+1, col] * P[l+1]
                end
            end
        end
    end
    
    # Inverse FFT in longitude direction to get spatial field
    f_complex = ifft_phi(Fφ)
    
    # Return appropriate output type
    if real_output
        return real.(f_complex)
    else
        return f_complex
    end
end

function synthesis_fused(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true)
    # Validate input dimensions
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))
    
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    Fφ = Matrix{CT}(undef, nlat, mmax + 1)
    fill!(Fφ, zero(CT))
    
    # Fused implementation for better performance
    if cfg.use_plm_tables && length(cfg.plm_tables) == mmax+1
        @threads for m in 0:mmax
            col = m + 1
            tbl = cfg.plm_tables[m+1]
            @inbounds for i in 1:nlat, l in m:lmax
                Fφ[i, col] += alm[l+1, col] * tbl[i, l+1]
            end
        end
    else
        thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
        
        @threads for m in 0:mmax
            col = m + 1
            P = thread_local_P[Threads.threadid()]
            
            for i in 1:nlat
                x = cfg.x[i]
                Plm_row!(P, x, lmax, m)
                
                sum_val = zero(CT)
                @inbounds for l in m:lmax
                    sum_val += alm[l+1, col] * P[l+1]
                end
                Fφ[i, col] = sum_val
            end
        end
    end
    
    # Inverse longitude FFT
    f_complex = ifft_phi(Fφ)
    
    return real_output ? real.(f_complex) : f_complex
end