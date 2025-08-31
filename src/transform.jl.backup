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
            for i in 1:nlat
                Fi = Fφ[i, col]       # Fourier coefficient for this (lat, m)
                wi = cfg.w[i]         # Gauss-Legendre weight
                # CANNOT use @simd ivdep - accumulating into same alm[l+1, col] across iterations
                @inbounds @simd for l in m:lmax  # Only l ≥ m contribute for order m
                    alm[l+1, col] += (wi * tbl[l+1, i]) * Fi
                end
            end
        else
            # Fallback: compute Legendre polynomials on-the-fly with thread-local buffer
            for i in 1:nlat
                Plm_row!(P, cfg.x[i], lmax, m)  # Compute P_l^m(cos(θ_i)) for all l
                Fi = Fφ[i, col]
                wi = cfg.w[i]
                # CANNOT use @simd ivdep - accumulating into same alm[l+1, col] across iterations
                @inbounds @simd for l in m:lmax
                    alm[l+1, col] += (wi * P[l+1]) * Fi
                end
            end
        end
        
        # Apply spherical harmonic normalization and longitude scaling with SIMD
        @inbounds @simd ivdep for l in m:lmax
            alm[l+1, col] *= cfg.Nlm[l+1, col] * scaleφ
        end
    end
    # Convert to user's requested normalization/phase convention if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        alm2 = similar(alm)
        convert_alm_norm!(alm2, alm, cfg; to_internal=false)  # Convert from internal to user format
        return alm2
    else
        return alm  # Already in the desired orthonormal format
    end
end

"""
    analysis_fused(cfg::SHTConfig, f::AbstractMatrix) -> Matrix{ComplexF64}

Cache-optimized fused analysis that combines FFT, Legendre integration, and normalization
in a single pass through memory for better cache locality and reduced memory bandwidth.
"""
function analysis_fused(cfg::SHTConfig, f::AbstractMatrix)
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
    thread_local_P_fused = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
    
    # FUSED LOOP: Process each azimuthal mode m with combined integration and normalization
    @threads for m in 0:mmax
        col = m + 1  # Julia 1-based indexing
        
        # Get thread-local buffer for this thread
        P = thread_local_P_fused[Threads.threadid()]
        
        # Integrate over colatitude θ using Gauss-Legendre quadrature
        if cfg.use_plm_tables && length(cfg.plm_tables) == mmax+1
            # Fast path: use precomputed Legendre polynomial tables with fused normalization
            tbl = cfg.plm_tables[m+1]  # P_l^m(x_i) values
            for i in 1:nlat
                Fi = Fφ[i, col]       # Fourier coefficient for this (lat, m)
                wi = cfg.w[i]         # Gauss-Legendre weight
                # CANNOT use @simd ivdep - accumulating into same alm[l+1, col] across iterations  
                @inbounds @simd for l in m:lmax  # Only l ≥ m contribute for order m
                    # FUSION: Combine integration with normalization in single operation
                    weight_norm = wi * cfg.Nlm[l+1, col] * scaleφ
                    alm[l+1, col] += (weight_norm * tbl[l+1, i]) * Fi
                end
            end
        else
            # Fallback: compute Legendre polynomials on-the-fly with fused normalization and thread-local buffer
            for i in 1:nlat
                Plm_row!(P, cfg.x[i], lmax, m)  # Compute P_l^m(cos(θ_i)) for all l
                Fi = Fφ[i, col]
                wi = cfg.w[i]
                # CANNOT use @simd ivdep - accumulating into same alm[l+1, col] across iterations
                @inbounds @simd for l in m:lmax
                    # FUSION: Combine integration with normalization in single operation
                    weight_norm = wi * cfg.Nlm[l+1, col] * scaleφ
                    alm[l+1, col] += (weight_norm * P[l+1]) * Fi
                end
            end
        end
        
        # Note: Normalization is now fused into the integration loop above
    end
    
    # Convert to user's requested normalization/phase convention if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        alm2 = similar(alm)
        convert_alm_norm!(alm2, alm, cfg; to_internal=false)  # Convert from internal to user format
        return alm2
    else
        return alm  # Already in the desired orthonormal format
    end
end

"""
    synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true) -> Matrix

Inverse spherical harmonic transform.
Input `alm` sized `(cfg.lmax+1, cfg.mmax+1)` with indices `(l+1, m+1)`.
Returns a grid `f` of shape `(cfg.nlat, cfg.nlon)`. If `real_output=true`,
enforces Hermitian symmetry to produce real-valued output.
"""
function synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true, use_fused_loops::Bool=true)
    if use_fused_loops
        return synthesis_fused(cfg, alm; real_output)
    else
        return synthesis_unfused(cfg, alm; real_output)
    end
end

function synthesis_unfused(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true)
    # Validate input coefficient array dimensions
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))

    # Allocate output array for spatial grid
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    Fφ = Matrix{CT}(undef, nlat, nlon)  # Fourier modes: Fφ[lat, m]
    fill!(Fφ, 0.0 + 0.0im)

    # Scale continuous Fourier coefficients to DFT bins for ifft.
    # ifft includes 1/nlon, so we multiply by nlon here to match f(φ) = Σ g_m e^{imφ}.
    inv_scaleφ = phi_inv_scale(nlon)

    # Convert incoming coefficients to internal normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        alm_int = similar(alm)
        convert_alm_norm!(alm_int, alm, cfg; to_internal=true)  # Convert to internal format
        alm = alm_int
    end
    
    # Thread-local storage for synthesis computation to avoid allocations
    # Each thread gets its own buffers to prevent race conditions
    thread_local_P_synth = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
    thread_local_G_synth = [Vector{CT}(undef, nlat) for _ in 1:Threads.nthreads()]
    
    # Build azimuthal Fourier spectrum from spherical harmonic coefficients
    @threads for m in 0:mmax
        col = m + 1  # Julia 1-based indexing
        
        # Get thread-local buffers for this thread
        P = thread_local_P_synth[Threads.threadid()]
        G = thread_local_G_synth[Threads.threadid()]
        
        # Compute latitudinal profile: G(θ_i) = Σ_l [N_lm * P_l^m(x_i) * a_lm]
        if cfg.use_plm_tables && length(cfg.plm_tables) == mmax+1
            # Fast path: use precomputed Legendre polynomial tables
            tbl = cfg.plm_tables[m+1]  # P_l^m(x_i) values
            for i in 1:nlat
                g = 0.0 + 0.0im
                # This is a REDUCTION - multiple l accumulate into same g variable
                @inbounds @simd for l in m:lmax  # Sum over degrees l ≥ m
                    g += (cfg.Nlm[l+1, col] * tbl[l+1, i]) * alm[l+1, col]
                end
                G[i] = g
            end
        else
            # Fallback: compute Legendre polynomials on-the-fly with thread-local buffer
            for i in 1:nlat
                Plm_row!(P, cfg.x[i], lmax, m)  # Compute P_l^m(cos(θ_i)) for all l
                g = 0.0 + 0.0im
                # This is a REDUCTION - multiple l accumulate into same g variable
                @inbounds @simd for l in m:lmax
                    g += (cfg.Nlm[l+1, col] * P[l+1]) * alm[l+1, col]
                end
                G[i] = g
            end
        end
        
        # Store positive m Fourier modes in the frequency domain array
        @inbounds for i in 1:nlat
            Fφ[i, col] = inv_scaleφ * G[i]
        end
        
        # For real output, enforce Hermitian symmetry: F(-m) = F*(m)
        if real_output && m > 0
            conj_index = nlon - m + 1  # Index for negative frequency -m
            @inbounds for i in 1:nlat
                Fφ[i, conj_index] = conj(Fφ[i, col])
            end
        end
    end

    # Inverse FFT along longitude (φ) to get spatial field
    f = ifft_phi(Fφ)
    return real_output ? real.(f) : f  # Return real part if real output requested
end

"""
    synthesis_fused(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true) -> Matrix

Cache-optimized fused synthesis that combines spherical harmonic reconstruction, 
Hermitian symmetry application, and IFFT preparation in optimized loops for
better cache locality and reduced memory passes.
"""
function synthesis_fused(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true)
    # Validate input coefficient array dimensions
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))

    # Allocate output array for spatial grid
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    Fφ = Matrix{CT}(undef, nlat, nlon)  # Fourier modes: Fφ[lat, m]
    fill!(Fφ, 0.0 + 0.0im)

    # Scale continuous Fourier coefficients to DFT bins for ifft.
    inv_scaleφ = phi_inv_scale(nlon)

    # Convert incoming coefficients to internal normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        alm_int = similar(alm)
        convert_alm_norm!(alm_int, alm, cfg; to_internal=true)  # Convert to internal format
        alm = alm_int
    end
    
    # Thread-local storage for fused synthesis computation to avoid allocations
    # Each thread gets its own buffers to prevent race conditions
    thread_local_P_fused_synth = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
    thread_local_G_fused_synth = [Vector{CT}(undef, nlat) for _ in 1:Threads.nthreads()]
    
    # FUSED LOOP: Build azimuthal Fourier spectrum with integrated Hermitian symmetry
    @threads for m in 0:mmax
        col = m + 1  # Julia 1-based indexing
        
        # Get thread-local buffers for this thread
        P = thread_local_P_fused_synth[Threads.threadid()]
        G = thread_local_G_fused_synth[Threads.threadid()]
        
        # Compute latitudinal profile: G(θ_i) = Σ_l [N_lm * P_l^m(x_i) * a_lm]
        if cfg.use_plm_tables && length(cfg.plm_tables) == mmax+1
            # Fast path: use precomputed Legendre polynomial tables
            tbl = cfg.plm_tables[m+1]  # P_l^m(x_i) values
            for i in 1:nlat
                g = 0.0 + 0.0im
                # This is a REDUCTION - multiple l accumulate into same g variable
                @inbounds @simd for l in m:lmax  # Sum over degrees l ≥ m
                    g += (cfg.Nlm[l+1, col] * tbl[l+1, i]) * alm[l+1, col]
                end
                G[i] = g
            end
        else
            # Fallback: compute Legendre polynomials on-the-fly with thread-local buffer
            for i in 1:nlat
                Plm_row!(P, cfg.x[i], lmax, m)  # Compute P_l^m(cos(θ_i)) for all l
                g = 0.0 + 0.0im
                # This is a REDUCTION - multiple l accumulate into same g variable
                @inbounds @simd for l in m:lmax
                    g += (cfg.Nlm[l+1, col] * P[l+1]) * alm[l+1, col]
                end
                G[i] = g
            end
        end
        
        # Store positive m modes - SAFE to vectorize, each i writes to different Fφ[i,col]
        @inbounds @simd ivdep for i in 1:nlat
            scaled_g = inv_scaleφ * G[i]
            Fφ[i, col] = scaled_g
        end
        
        # Apply Hermitian symmetry separately to avoid potential write conflicts
        if real_output && m > 0
            conj_index = nlon - m + 1  # Index for negative frequency -m
            @inbounds @simd ivdep for i in 1:nlat
                Fφ[i, conj_index] = conj(inv_scaleφ * G[i])
            end
        end
    end

    # Inverse FFT along longitude (φ) to get spatial field
    f = ifft_phi(Fφ)
    return real_output ? real.(f) : f  # Return real part if real output requested
end

"""
    spat_to_SH(cfg::SHTConfig, Vr::AbstractVector{<:Real}) -> Vector{ComplexF64}

SHTns-compatible scalar analysis. `Vr` is a flat vector of length `cfg.nspat = nlat*nlon`.
Returns packed coefficients `Qlm` of length `cfg.nlm` with SHTns `LM` ordering.
"""
function spat_to_SH(cfg::SHTConfig, Vr::AbstractVector{<:Real})
    # Validate input size matches the spatial grid
    length(Vr) == cfg.nspat || throw(DimensionMismatch("Vr must have length $(cfg.nspat)"))
    
    # Reshape flat vector to 2D grid: Vr[lat*nlon + lon] -> f[lat, lon]
    f = reshape(Vr, cfg.nlat, cfg.nlon)
    
    # Perform forward spherical harmonic transform
    alm_mat = analysis(cfg, f)  # Get coefficients in (l+1, m+1) matrix format
    
    # Convert to SHTns-compatible packed format using LM indexing
    Qlm = Vector{eltype(alm_mat)}(undef, cfg.nlm)  # Packed coefficient array
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue  # Skip m values not on the resolution grid
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1  # Convert (l,m) to flat index
            Qlm[lm] = alm_mat[l+1, m+1]  # Copy coefficient to packed array
        end
    end
    return Qlm
end

"""
    SH_to_spat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}) -> Vector{Float64}

SHTns-compatible scalar synthesis to a real spatial field. Input is packed `Qlm`.
Returns a flat `Vector{Float64}` length `nlat*nlon`.
"""
function SH_to_spat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex})
    # Validate input packed coefficient array size
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    
    # Convert from packed SHTns format to (l+1, m+1) matrix format
    alm_mat = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)  # Coefficient matrix
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue  # Skip m values not on the resolution grid
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1  # Convert (l,m) to flat index
            alm_mat[l+1, m+1] = Qlm[lm]  # Unpack coefficient from flat array
        end
    end
    
    # Perform inverse spherical harmonic transform
    f = synthesis(cfg, alm_mat; real_output=true)  # Get 2D spatial grid
    
    # Return as flat vector compatible with SHTns convention
    return vec(f)  # Flatten f[lat, lon] -> Vr[lat*nlon + lon]
end

 

"""
    spat_to_SH_l(cfg::SHTConfig, Vr, ltr::Int)

Truncated scalar analysis up to degree `ltr`.
"""
function spat_to_SH_l(cfg::SHTConfig, Vr::AbstractVector{<:Real}, ltr::Int)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    Qlm = spat_to_SH(cfg, Vr)
    # zero out l > ltr
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in (ltr+1):cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            Qlm[lm] = 0.0 + 0.0im
        end
    end
    return Qlm
end

"""
    SH_to_spat_l(cfg::SHTConfig, Qlm, ltr::Int)

Truncated scalar synthesis using only degrees `l ≤ ltr`.
"""
function SH_to_spat_l(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, ltr::Int)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    alm_mat = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in m:min(ltr, cfg.lmax)
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            alm_mat[l+1, m+1] = Qlm[lm]
        end
    end
    f = synthesis(cfg, alm_mat; real_output=true)
    return vec(f)
end

"""
    spat_to_SH_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, ltr::Int)

Legendre-only transform at fixed `m = im*mres` truncated at `ltr`.
`Vr_m` is the Fourier mode along φ for each latitude (length `nlat`).
Returns spectrum `Ql` of length `ltr+1-m` for degrees `l=m..ltr`.
"""
function spat_to_SH_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    length(Vr_m) == nlat || throw(DimensionMismatch("Vr_m must have length nlat"))
    m = im * cfg.mres
    (0 ≤ m ≤ cfg.mmax) || throw(ArgumentError("invalid m from im"))
    (m ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("require m ≤ ltr ≤ lmax"))
    P = Vector{Float64}(undef, cfg.lmax + 1)
    Ql = Vector{ComplexF64}(undef, ltr - m + 1)
    fill!(Ql, 0.0 + 0.0im)
    for i in 1:nlat
        Plm_row!(P, cfg.x[i], cfg.lmax, m)
        wi = cfg.w[i]
        Fi = Vr_m[i]
        @inbounds for l in m:ltr
            Ql[l - m + 1] += wi * P[l+1] * Fi * cfg.Nlm[l+1, m+1] * cfg.cphi
        end
    end
    return Ql
end

"""
    SH_to_spat_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, ltr::Int) -> Vector{ComplexF64}

Legendre-only synthesis at fixed `m = im*mres` truncated at `ltr`.
Returns the Fourier mode across latitudes (length `nlat`).
"""
function SH_to_spat_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, ltr::Int)
    m = im * cfg.mres
    (m ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("require m ≤ ltr ≤ lmax"))
    length(Ql) == ltr - m + 1 || throw(DimensionMismatch("Ql length must be ltr-m+1"))
    nlat = cfg.nlat
    P = Vector{Float64}(undef, cfg.lmax + 1)
    Vr_m = Vector{ComplexF64}(undef, nlat)
    for i in 1:nlat
        Plm_row!(P, cfg.x[i], cfg.lmax, m)
        g = 0.0 + 0.0im
        @inbounds for l in m:ltr
            g += cfg.Nlm[l+1, m+1] * P[l+1] * Ql[l - m + 1]
        end
        Vr_m[i] = (cfg.nlon / (2π)) * g
    end
    return Vr_m
end

"""
    SH_to_point(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real, phi::Real) -> Float64

Evaluate a real field represented by packed `Qlm` at a single point using orthonormal harmonics.
"""
function SH_to_point(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    x = float(cost)
    lmax = cfg.lmax; mmax = cfg.mmax
    P = Vector{Float64}(undef, lmax + 1)
    CT = eltype(Qlm)
    acc = zero(CT)
    # m = 0 term
    Plm_row!(P, x, lmax, 0)
    g0 = zero(CT)
    @inbounds for l in 0:lmax
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        a = Qlm[lm]
        if cfg.norm !== :orthonormal || cfg.cs_phase == false
            k = norm_scale_from_orthonormal(l, 0, cfg.norm)
            α = cs_phase_factor(0, true, cfg.cs_phase)
            a *= (k * α)
        end
        g0 += cfg.Nlm[l+1, 1] * P[l+1] * a
    end
    acc += g0
    # m > 0 with Hermitian symmetry for real field
    for m in 1:mmax
        (m % cfg.mres == 0) || continue
        Plm_row!(P, x, lmax, m)
        gm = zero(CT)
        col = m + 1
        @inbounds for l in m:lmax
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            a = Qlm[lm]
            if cfg.norm !== :orthonormal || cfg.cs_phase == false
                k = norm_scale_from_orthonormal(l, m, cfg.norm)
                α = cs_phase_factor(m, true, cfg.cs_phase)
                a *= (k * α)
            end
            gm += cfg.Nlm[l+1, col] * P[l+1] * a
        end
        acc += 2 * real(gm * cis(m * phi))
    end
    return real(acc)
end
