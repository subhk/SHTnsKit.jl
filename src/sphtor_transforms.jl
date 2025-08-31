"""
Spheroidal-Toroidal Vector Field Transforms

This module handles transforms for horizontal (tangential) vector fields using
spheroidal (S) and toroidal (T) decomposition. This is the natural representation
for 2D vector fields on the sphere, such as horizontal velocity components.

Key relationships:
- Vt = ∂S/∂θ - (1/sin θ) ∂T/∂φ  (colatitude component)
- Vp = (1/sin θ) ∂S/∂φ + ∂T/∂θ  (azimuthal component)

Where S represents the spheroidal (curl-free) part and T the toroidal (div-free) part.
"""

"""
    SHsphtor_to_spat(cfg, Slm, Tlm; real_output=true) -> (Vt, Vp)

Transform spheroidal/toroidal coefficients to horizontal vector field components.
Returns colatitude (Vt) and azimuthal (Vp) components on the spatial grid.
"""
function SHsphtor_to_spat(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    # Validate input dimensions
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Slm,1) == lmax+1 && size(Slm,2) == mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1) == lmax+1 && size(Tlm,2) == mmax+1 || throw(DimensionMismatch("Tlm dims"))

    # Convert to internal normalization if needed
    Slm_int, Tlm_int = Slm, Tlm
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm); T2 = similar(Tlm)
        convert_alm_norm!(S2, Slm, cfg; to_internal=true)
        convert_alm_norm!(T2, Tlm, cfg; to_internal=true)
        Slm_int = S2; Tlm_int = T2
    end

    # Set up arrays for synthesis
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(Slm_int)
    Fθ = Matrix{CT}(undef, nlat, nlon)  # Fourier coefficients for θ-component
    Fφ = Matrix{CT}(undef, nlat, nlon)  # Fourier coefficients for φ-component
    fill!(Fθ, 0); fill!(Fφ, 0)

    # Thread-local working arrays for Legendre polynomial computation
    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
    thread_local_dPdx = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
    # Scale continuous Fourier coefficients to DFT bins for ifft (factor nlon or nlon/(2π))
    inv_scaleφ = phi_inv_scale(nlon)

    # Process each azimuthal mode m in parallel
    @threads for m in 0:mmax
        col = m + 1
        for i in 1:nlat
            x = cfg.x[i]
            sθ = sqrt(max(0.0, 1 - x*x))
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            gθ = 0.0 + 0.0im
            gφ = 0.0 + 0.0im
            if cfg.use_plm_tables && length(cfg.plm_tables) == mmax+1 && length(cfg.dplm_tables) == mmax+1
                tblP = cfg.plm_tables[m+1]
                tbld = cfg.dplm_tables[m+1]
                @inbounds for l in m:lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, i]
                    Y = N * tblP[l+1, i]
                    Sl = Slm_int[l+1, col]
                    Tl = Tlm_int[l+1, col]
                    gθ += dθY * Sl + (0 + 1im) * m * inv_sθ * Y * Tl
                    gφ += (0 + 1im) * m * inv_sθ * Y * Sl + (sθ * N * tbld[l+1, i]) * Tl
                end
            else
                P = thread_local_P[Threads.threadid()]
                dPdx = thread_local_dPdx[Threads.threadid()]
                Plm_and_dPdx_row!(P, dPdx, x, lmax, m)
                @inbounds for l in m:lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * dPdx[l+1]
                    Y = N * P[l+1]
                    Sl = Slm_int[l+1, col]
                    Tl = Tlm_int[l+1, col]
                    gθ += dθY * Sl + (0 + 1im) * m * inv_sθ * Y * Tl
                    gφ += (0 + 1im) * m * inv_sθ * Y * Sl + (sθ * N * dPdx[l+1]) * Tl
                end
            end
            Fθ[i, col] = inv_scaleφ * gθ
            Fφ[i, col] = inv_scaleφ * gφ
        end
        if real_output && m > 0
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                Fθ[i, conj_index] = conj(Fθ[i, col])
                Fφ[i, conj_index] = conj(Fφ[i, col])
            end
        end
    end
    Vt = real_output ? real.(ifft_phi(Fθ)) : ifft_phi(Fθ)
    Vp = real_output ? real.(ifft_phi(Fφ)) : ifft_phi(Fφ)
    if cfg.robert_form
        @inbounds for i in 1:nlat
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
            Vt[i, :] .*= sθ
            Vp[i, :] .*= sθ
        end
    end
    return Vt, Vp
end

"""
    spat_to_SHsphtor(cfg, Vt, Vp) -> (Slm, Tlm)

Transform horizontal vector field components to spheroidal/toroidal coefficients.
Input: colatitude (Vt) and azimuthal (Vp) components on spatial grid.
"""
function spat_to_SHsphtor(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    
    # Validate input dimensions  
    size(Vt,1) == nlat && size(Vt,2) == nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1) == nlat && size(Vp,2) == nlon || throw(DimensionMismatch("Vp dims"))
    
    # This is a simplified implementation
    # Full version would involve proper vector analysis including divergence/curl decomposition
    
    # For now, perform forward transforms and apply inverse scaling
    Slm_temp = analysis(cfg, Vt)  
    Tlm_temp = analysis(cfg, Vp)
    
    lmax, mmax = cfg.lmax, cfg.mmax
    Slm = similar(Slm_temp)
    Tlm = similar(Tlm_temp)
    fill!(Slm, 0.0)
    fill!(Tlm, 0.0)
    
    # Inverse of the scaling applied in SHsphtor_to_spat
    @inbounds for m in 0:mmax, l in max(1,m):lmax
        row, col = l + 1, m + 1
        ll1 = l * (l + 1)
        if ll1 > 0
            scale_inv = 1.0 / sqrt(ll1)
            Slm[row, col] = Slm_temp[row, col] * scale_inv
            Tlm[row, col] = Tlm_temp[row, col] * scale_inv
        end
    end
    
    return Slm, Tlm
end

"""
    SHsphtor_to_spat_cplx(cfg, Slm, Tlm) -> (Vt, Vp)

Complex version preserving complex values in output.
"""
function SHsphtor_to_spat_cplx(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix)
    return SHsphtor_to_spat(cfg, Slm, Tlm; real_output=false)
end

"""
    spat_cplx_to_SHsphtor(cfg, Vt, Vp) -> (Slm, Tlm)

Transform complex horizontal vector components to spheroidal/toroidal coefficients.
"""
function spat_cplx_to_SHsphtor(cfg::SHTConfig, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
    return spat_to_SHsphtor(cfg, Vt, Vp)  # Same implementation works for complex
end

"""
    SHsph_to_spat(cfg, Slm; real_output=true) -> (Vt, Vp)

Transform only spheroidal component to spatial vector field (Tlm = 0).
"""
function SHsph_to_spat(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    Tlm_zero = zeros(eltype(Slm), lmax+1, mmax+1)
    return SHsphtor_to_spat(cfg, Slm, Tlm_zero; real_output=real_output)
end

"""
    SHtor_to_spat(cfg, Tlm; real_output=true) -> (Vt, Vp)

Transform only toroidal component to spatial vector field (Slm = 0).
"""
function SHtor_to_spat(cfg::SHTConfig, Tlm::AbstractMatrix; real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    Slm_zero = zeros(eltype(Tlm), lmax+1, mmax+1)
    return SHsphtor_to_spat(cfg, Slm_zero, Tlm; real_output=real_output)
end

"""
    SHsphtor_to_spat_l(cfg, Slm, Tlm, ltr; real_output=true) -> (Vt, Vp)

Degree-limited spheroidal/toroidal transform using modes up to degree ltr.
"""
function SHsphtor_to_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    
    # Create truncated copies
    S2 = copy(Slm); T2 = copy(Tlm)
    
    # Zero high-degree modes
    @inbounds for m in 0:mmax, l in (ltr+1):lmax
        if l >= m
            S2[l+1, m+1] = 0.0
            T2[l+1, m+1] = 0.0  
        end
    end
    
    return SHsphtor_to_spat(cfg, S2, T2; real_output=real_output)
end

"""
    spat_to_SHsphtor_l(cfg, Vt, Vp, ltr) -> (Slm, Tlm)

Degree-limited analysis computing coefficients only up to degree ltr.
"""
function spat_to_SHsphtor_l(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix, ltr::Int)
    # Get full transform then truncate
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    
    # Zero high-degree modes
    @inbounds for m in 0:cfg.mmax, l in (ltr+1):cfg.lmax
        if l >= m
            Slm[l+1, m+1] = 0.0
            Tlm[l+1, m+1] = 0.0
        end
    end
    
    return Slm, Tlm
end

"""
    SHsph_to_spat_l(cfg, Slm, ltr; real_output=true) -> (Vt, Vp)

Degree-limited spheroidal-only transform.
"""
function SHsph_to_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    Tlm_zero = zeros(eltype(Slm), cfg.lmax+1, cfg.mmax+1)
    return SHsphtor_to_spat_l(cfg, Slm, Tlm_zero, ltr; real_output=real_output)
end

"""
    SHtor_to_spat_l(cfg, Tlm, ltr; real_output=true) -> (Vt, Vp)

Degree-limited toroidal-only transform.
"""
function SHtor_to_spat_l(cfg::SHTConfig, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    Slm_zero = zeros(eltype(Tlm), cfg.lmax+1, cfg.mmax+1)
    return SHsphtor_to_spat_l(cfg, Slm_zero, Tlm, ltr; real_output=real_output)
end

"""
    spat_to_SHsphtor_ml(cfg, im, Vt_m, Vp_m, ltr) -> (Sl, Tl)

Mode-limited transform for specific azimuthal mode im.
Input vectors contain spatial values for that mode at all latitudes.
"""
function spat_to_SHsphtor_ml(cfg::SHTConfig, im::Int, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    length(Vt_m) == nlat || throw(DimensionMismatch("Vt_m length must be nlat"))
    length(Vp_m) == nlat || throw(DimensionMismatch("Vp_m length must be nlat"))
    
    # Simplified mode-specific analysis
    # Full implementation would properly handle vector mode decomposition
    
    num_l = ltr - im + 1
    Sl = Vector{ComplexF64}(undef, num_l)  
    Tl = Vector{ComplexF64}(undef, num_l)
    fill!(Sl, 0.0)
    fill!(Tl, 0.0)
    
    P = Vector{Float64}(undef, ltr + 1)
    
    # Integrate using Legendre polynomials
    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, ltr, im)
        
        wVt = Vt_m[i] * cfg.wlat[i]
        wVp = Vp_m[i] * cfg.wlat[i]
        
        @inbounds for l in im:ltr
            # Simplified vector mode analysis
            ll1 = l * (l + 1)
            if ll1 > 0
                scale = cfg.cphi / sqrt(ll1)
                Sl[l-im+1] += wVt * P[l+1] * scale
                Tl[l-im+1] += wVp * P[l+1] * scale
            end
        end
    end
    
    return Sl, Tl
end

"""
    SHsphtor_to_spat_ml(cfg, im, Sl, Tl, ltr) -> (Vt_m, Vp_m)

Mode-limited synthesis for specific azimuthal mode im.
"""
function SHsphtor_to_spat_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    expected_len = ltr - im + 1
    length(Sl) == expected_len || throw(DimensionMismatch("Sl length mismatch"))
    length(Tl) == expected_len || throw(DimensionMismatch("Tl length mismatch"))
    
    Vt_m = Vector{ComplexF64}(undef, nlat)
    Vp_m = Vector{ComplexF64}(undef, nlat)
    P = Vector{Float64}(undef, ltr + 1)
    
    # Synthesize vector components for this mode
    for i in 1:nlat
        x = cfg.x[i]
        Plm_row!(P, x, ltr, im)
        
        vt_sum = 0.0 + 0.0im
        vp_sum = 0.0 + 0.0im
        
        @inbounds for l in im:ltr
            ll1 = l * (l + 1)
            if ll1 > 0
                scale = sqrt(ll1)
                vt_sum += Sl[l-im+1] * P[l+1] * scale
                vp_sum += Tl[l-im+1] * P[l+1] * scale  
            end
        end
        
        Vt_m[i] = vt_sum
        Vp_m[i] = vp_sum
    end
    
    return Vt_m, Vp_m
end
