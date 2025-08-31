"""
Vector Spherical Harmonic Transforms using Spheroidal/Toroidal Decomposition

This module implements vector spherical harmonic transforms using the Helmholtz
decomposition into spheroidal and toroidal components. Any vector field on the
sphere can be uniquely decomposed as:

V(θ,φ) = ∇S(θ,φ) + r̂ × ∇T(θ,φ)

where S and T are scalar potentials called the spheroidal and toroidal scalars.

Mathematical Framework:
- S and T are expanded in spherical harmonics: S = Σ S_lm Y_l^m, T = Σ T_lm Y_l^m  
- The velocity components are:
  * V_θ = ∂S/∂θ + (im/sin θ) T_lm Y_l^m
  * V_φ = (im/sin θ) S_lm Y_l^m - (1/sin θ) ∂T/∂θ

Physical Interpretation:
- Spheroidal component (∇S): potential flow, divergent part
- Toroidal component (r̂ × ∇T): rotational flow, includes vorticity

Implementation uses FFT in longitude and Gauss-Legendre quadrature in latitude,
following the same efficient approach as scalar transforms.
"""

"""
    SHsphtor_to_spat(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
        -> Vt::Matrix{Float64}, Vp::Matrix{Float64}

Synthesize vector field components from spheroidal and toroidal spectral coefficients.

This function performs the inverse vector spherical harmonic transform, converting
from spectral space (S_lm, T_lm) to physical space (V_θ, V_φ) components.

The synthesis computes:
- V_θ = Σ_lm [∂Y_l^m/∂θ S_lm + (im/sin θ) Y_l^m T_lm]  
- V_φ = Σ_lm [(im/sin θ) Y_l^m S_lm - (1/sin θ) ∂Y_l^m/∂θ T_lm]

Returns the θ (colatitude) and φ (longitude) components of the vector field.
"""
function SHsphtor_to_spat(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    # Validate input dimensions
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Slm,1) == lmax+1 && size(Slm,2) == mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1) == lmax+1 && size(Tlm,2) == mmax+1 || throw(DimensionMismatch("Tlm dims"))
    
    # Convert to internal normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm); T2 = similar(Tlm)
        convert_alm_norm!(S2, Slm, cfg; to_internal=true)
        convert_alm_norm!(T2, Tlm, cfg; to_internal=true)
        Slm = S2; Tlm = T2
    end
    # Set up arrays for synthesis
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(Slm)
    Fθ = Matrix{CT}(undef, nlat, nlon)  # Fourier coefficients for θ-component
    Fφ = Matrix{CT}(undef, nlat, nlon)  # Fourier coefficients for φ-component
    fill!(Fθ, 0.0 + 0.0im); fill!(Fφ, 0.0 + 0.0im)

    # Thread-local working arrays for Legendre polynomial computation
    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
    thread_local_dPdx = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
    # Scale continuous Fourier coefficients to DFT bins for ifft (factor nlon)
    inv_scaleφ = phi_inv_scale(nlon)

    # Process each azimuthal mode m in parallel with dynamic scheduling
    @threads :dynamic for m in 0:mmax
        col = m + 1  # 1-based indexing for Julia arrays
        
        # Compute vector components at each latitude
        for i in 1:nlat
            x = cfg.x[i]                                    # x = cos(θ_i)
            sθ = sqrt(max(0.0, 1 - x*x))                   # sin(θ_i), guarded for poles
            gθ = 0.0 + 0.0im                               # Accumulator for θ-component  
            gφ = 0.0 + 0.0im                               # Accumulator for φ-component
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ             # 1/sin(θ), handling poles
            Ict = one(CT) * (0 + 1im)                      # Imaginary unit for azimuthal derivatives
            
            # Choose computation path: precomputed tables or on-the-fly calculation
            if cfg.use_plm_tables && length(cfg.plm_tables) == mmax+1 && length(cfg.dplm_tables) == mmax+1
                # Fast path: use precomputed Legendre polynomial tables
                tblP = cfg.plm_tables[m+1]; tbld = cfg.dplm_tables[m+1]
                # Sum contributions from all l-degrees for this (i,m) pair
                @inbounds for l in m:lmax
                    N = cfg.Nlm[l+1, col]                           # Normalization factor
                    dθY = -sθ * N * tbld[l+1, i]                   # ∂Y_l^m/∂θ = -sin(θ) N d P_l^m/dx  
                    Y = N * tblP[l+1, i]                           # Y_l^m = N P_l^m(cos θ)
                    
                    # Get spectral coefficients for this (l,m) mode
                    Sl = Slm[l+1, col]                             # Spheroidal coefficient
                    Tl = Tlm[l+1, col]                             # Toroidal coefficient
                    
                    # Accumulate vector components using decomposition formulas
                    gθ += dθY * Sl + Ict * m * inv_sθ * Y * Tl      # V_θ = ∂S/∂θ + (im/sin θ) T
                    gφ += Ict * m * inv_sθ * Y * Sl + (sθ * N * tbld[l+1, i]) * Tl  # V_φ = (im/sin θ) S - ∂T/∂θ
                end
            else
                # Fallback path: compute Legendre polynomials on-the-fly (thread-local buffers)
                P = thread_local_P[Threads.threadid()]
                dPdx = thread_local_dPdx[Threads.threadid()]
                Plm_and_dPdx_row!(P, dPdx, x, lmax, m)
                
                @inbounds for l in m:lmax
                    N = cfg.Nlm[l+1, col]                          # Normalization factor
                    dθY = -sθ * N * dPdx[l+1]                      # ∂Y_l^m/∂θ = -sin(θ) N d P_l^m/dx
                    Y = N * P[l+1]                                 # Y_l^m = N P_l^m(cos θ)
                    
                    # Get spectral coefficients
                    Sl = Slm[l+1, col]
                    Tl = Tlm[l+1, col]
                    
                    # Accumulate vector components
                    gθ += dθY * Sl + Ict * m * inv_sθ * Y * Tl      # V_θ component
                    gφ += Ict * m * inv_sθ * Y * Sl + (sθ * N * dPdx[l+1]) * Tl  # V_φ component
                end
            end
            
            # Store Fourier coefficients with inverse FFT scaling
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
    SHqst_to_spat(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix;
                   real_output::Bool=true) -> Vr, Vt, Vp

3D synthesis: combine scalar radial (Qlm) with vector tangential (Slm,Tlm).
"""
function SHqst_to_spat(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    Vr = synthesis(cfg, Qlm; real_output=real_output)
    Vt, Vp = SHsphtor_to_spat(cfg, Slm, Tlm; real_output=real_output)
    return Vr, Vt, Vp
end

"""
    spat_to_SHqst(cfg::SHTConfig, Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix)
        -> Qlm, Slm, Tlm

3D analysis: project radial onto scalar Y_lm and tangential onto spheroidal/toroidal.
"""
function spat_to_SHqst(cfg::SHTConfig, Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix)
    size(Vr,1) == cfg.nlat && size(Vr,2) == cfg.nlon || throw(DimensionMismatch("Vr dims"))
    size(Vt,1) == cfg.nlat && size(Vt,2) == cfg.nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1) == cfg.nlat && size(Vp,2) == cfg.nlon || throw(DimensionMismatch("Vp dims"))
    Qlm = analysis(cfg, Vr)
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    return Qlm, Slm, Tlm
end

"""
    SHqst_to_spat_cplx(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix)
        -> Vr::Matrix{ComplexF64}, Vt::Matrix{ComplexF64}, Vp::Matrix{ComplexF64}

Complex 3D synthesis.
"""
function SHqst_to_spat_cplx(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix)
    Vr = synthesis(cfg, Qlm; real_output=false)
    Vt, Vp = SHsphtor_to_spat(cfg, Slm, Tlm; real_output=false)
    return Vr, Vt, Vp
end

"""
    spat_cplx_to_SHqst(cfg::SHTConfig, Vr::AbstractMatrix{<:Complex}, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
        -> Qlm, Slm, Tlm

Complex 3D analysis.
"""
function spat_cplx_to_SHqst(cfg::SHTConfig, Vr::AbstractMatrix{<:Complex}, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
    size(Vr,1) == cfg.nlat && size(Vr,2) == cfg.nlon || throw(DimensionMismatch("Vr dims"))
    size(Vt,1) == cfg.nlat && size(Vt,2) == cfg.nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1) == cfg.nlat && size(Vp,2) == cfg.nlon || throw(DimensionMismatch("Vp dims"))
    Qlm = analysis(cfg, Vr)
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    return Qlm, Slm, Tlm
end

"""
    spat_to_SHsphtor(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
        -> Slm::Matrix{ComplexF64}, Tlm::Matrix{ComplexF64}

Analyze vector field components (θ, φ) to spheroidal/toroidal spectra.
"""
function spat_to_SHsphtor(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(Vt,1) == nlat && size(Vt,2) == nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1) == nlat && size(Vp,2) == nlon || throw(DimensionMismatch("Vp dims"))
    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(Vt) <: Complex ? eltype(Vt) : Complex{eltype(Vt)}
    Slm = zeros(CT, lmax+1, mmax+1)
    Tlm = zeros(CT, lmax+1, mmax+1)

    Fθ = fft_phi(complex.(Vt))
    Fφ = fft_phi(complex.(Vp))
    # Thread-local working arrays for Legendre polynomial computation
    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
    thread_local_dPdx = [Vector{Float64}(undef, lmax + 1) for _ in 1:Threads.nthreads()]
    scaleφ = cfg.cphi

    @threads :dynamic for m in 0:mmax
        col = m + 1
        for i in 1:nlat
            x = cfg.x[i]
            sθ = sqrt(max(0.0, 1 - x*x))
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            Fθ_i = Fθ[i, col]
            Fφ_i = Fφ[i, col]
            if cfg.robert_form
                if sθ > 0
                    Fθ_i /= sθ
                    Fφ_i /= sθ
                end
            end
            wi = cfg.w[i]
            Ict = one(CT) * (0 + 1im)
            if cfg.use_plm_tables && length(cfg.plm_tables) == mmax+1 && length(cfg.dplm_tables) == mmax+1
                tblP = cfg.plm_tables[m+1]; tbld = cfg.dplm_tables[m+1]
                @inbounds for l in max(1,m):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, i]
                    Y = N * tblP[l+1, i]
                    coeff = wi * scaleφ / (l*(l+1))
                    Slm[l+1, col] += coeff * (Fθ_i * dθY - Ict * m * inv_sθ * Y * Fφ_i)
                    Tlm[l+1, col] += coeff * (Ict * m * inv_sθ * Y * Fθ_i + Fφ_i * (+sθ * N * tbld[l+1, i]))
                end
            else
                # Compute Legendre polynomials on-the-fly (thread-local buffers)
                P = thread_local_P[Threads.threadid()]
                dPdx = thread_local_dPdx[Threads.threadid()]
                Plm_and_dPdx_row!(P, dPdx, x, lmax, m)
                @inbounds for l in max(1,m):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * dPdx[l+1]
                    Y = N * P[l+1]
                    coeff = wi * scaleφ / (l*(l+1))
                    Slm[l+1, col] += coeff * (Fθ_i * dθY - Ict * m * inv_sθ * Y * Fφ_i)
                    Tlm[l+1, col] += coeff * (Ict * m * inv_sθ * Y * Fθ_i + Fφ_i * (+sθ * N * dPdx[l+1]))
                end
            end
        end
    end
    # Convert to cfg normalization if needed
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm); T2 = similar(Tlm)
        convert_alm_norm!(S2, Slm, cfg; to_internal=false)
        convert_alm_norm!(T2, Tlm, cfg; to_internal=false)
        return S2, T2
    end
    return Slm, Tlm
end

"""
    SHsphtor_to_spat_cplx(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix)
        -> Vt::Matrix{ComplexF64}, Vp::Matrix{ComplexF64}

Complex synthesis wrapper.
"""
function SHsphtor_to_spat_cplx(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix)
    return SHsphtor_to_spat(cfg, Slm, Tlm; real_output=false)
end

"""
    spat_cplx_to_SHsphtor(cfg::SHTConfig, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
        -> Slm, Tlm

Complex analysis wrapper.
"""
function spat_cplx_to_SHsphtor(cfg::SHTConfig, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
    return spat_to_SHsphtor(cfg, Vt, Vp)
end

"""
    SHsph_to_spat(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true)
        -> Vt, Vp

Synthesize only the spheroidal part (gradient of S).
"""
function SHsph_to_spat(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true)
    Z = zeros(eltype(Slm), size(Slm))
    return SHsphtor_to_spat(cfg, Slm, Z; real_output)
end

"""
    SHtor_to_spat(cfg::SHTConfig, Tlm::AbstractMatrix; real_output::Bool=true)
        -> Vt, Vp

Synthesize only the toroidal part.
"""
function SHtor_to_spat(cfg::SHTConfig, Tlm::AbstractMatrix; real_output::Bool=true)
    Z = zeros(eltype(Tlm), size(Tlm))
    return SHsphtor_to_spat(cfg, Z, Tlm; real_output)
end

"""
    SH_to_grad_spat(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true) -> Vt, Vp

Alias to `SHsph_to_spat`, for compatibility with SHTns macro.
"""
SH_to_grad_spat(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true) = SHsph_to_spat(cfg, Slm; real_output)


"""
    SHsphtor_to_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
        -> Vt, Vp

Truncated vector synthesis using only degrees l ≤ ltr.
"""
function SHsphtor_to_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    # Copy and zero out > ltr
    S2 = copy(Slm); T2 = copy(Tlm)
    @inbounds for m in 0:cfg.mmax
        for l in (ltr+1):cfg.lmax
            S2[l+1, m+1] = 0
            T2[l+1, m+1] = 0
        end
    end
    return SHsphtor_to_spat(cfg, S2, T2; real_output)
end

"""
    spat_to_SHsphtor_l(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix, ltr::Int)
        -> Slm, Tlm

Truncated vector analysis; zeroes Slm/Tlm for l > ltr.
"""
function spat_to_SHsphtor_l(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix, ltr::Int)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    @inbounds for m in 0:cfg.mmax
        for l in (ltr+1):cfg.lmax
            Slm[l+1, m+1] = 0
            Tlm[l+1, m+1] = 0
        end
    end
    return Slm, Tlm
end

"""
    SHsph_to_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, ltr::Int; real_output::Bool=true)
"""
function SHsph_to_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    Z = zeros(ComplexF64, size(Slm))
    return SHsphtor_to_spat_l(cfg, Slm, Z, ltr; real_output)
end

"""
    SHtor_to_spat_l(cfg::SHTConfig, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
"""
function SHtor_to_spat_l(cfg::SHTConfig, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    Z = zeros(ComplexF64, size(Tlm))
    return SHsphtor_to_spat_l(cfg, Z, Tlm, ltr; real_output)
end

"""
    spat_to_SHsphtor_ml(cfg::SHTConfig, im::Int, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
        -> Sl::Vector{ComplexF64}, Tl::Vector{ComplexF64}

Per-m vector analysis (no FFT) truncated at ltr.
"""
function spat_to_SHsphtor_ml(cfg::SHTConfig, im::Int, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    length(Vt_m) == nlat && length(Vp_m) == nlat || throw(DimensionMismatch("per-m inputs must have length nlat"))
    m = im * cfg.mres
    (0 ≤ m ≤ cfg.mmax) || throw(ArgumentError("invalid m from im"))
    lstart = max(1, m)
    (lstart ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("require max(1,m) ≤ ltr ≤ lmax"))
    P = Vector{Float64}(undef, cfg.lmax + 1)
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)
    CT = eltype(Vt_m)
    Sl = zeros(CT, ltr - lstart + 1)
    Tl = zeros(CT, ltr - lstart + 1)
    for i in 1:nlat
        x = cfg.x[i]
        sθ = sqrt(max(0.0, 1 - x*x))
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
        wi = cfg.w[i]
        Fθ_i = Vt_m[i]
        Fφ_i = Vp_m[i]
        if cfg.use_plm_tables && length(cfg.plm_tables) > im && length(cfg.dplm_tables) > im
            tblP = cfg.plm_tables[im+1]; tbld = cfg.dplm_tables[im+1]
            @inbounds for l in lstart:ltr
                N = cfg.Nlm[l+1, m+1]
                dθY = -sθ * N * tbld[l+1, i]
                Y = N * tblP[l+1, i]
                coeff = wi / (l*(l+1))
                Ict = one(CT) * (0 + 1im)
                Sl[l - lstart + 1] += coeff * (Fθ_i * dθY - Ict * m * inv_sθ * Y * Fφ_i)
                Tl[l - lstart + 1] += coeff * (Ict * m * inv_sθ * Y * Fθ_i + Fφ_i * (+sθ * N * tbld[l+1, i]))
            end
        else
            Plm_and_dPdx_row!(P, dPdx, x, cfg.lmax, m)
            @inbounds for l in lstart:ltr
                N = cfg.Nlm[l+1, m+1]
                dθY = -sθ * N * dPdx[l+1]
                Y = N * P[l+1]
                coeff = wi / (l*(l+1))
                Ict = one(CT) * (0 + 1im)
                Sl[l - lstart + 1] += coeff * (Fθ_i * dθY - Ict * m * inv_sθ * Y * Fφ_i)
                Tl[l - lstart + 1] += coeff * (Ict * m * inv_sθ * Y * Fθ_i + Fφ_i * (+sθ * N * dPdx[l+1]))
            end
        end
    end
    return Sl, Tl
end

"""
    SHsphtor_to_spat_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
        -> Vt_m::Vector{ComplexF64}, Vp_m::Vector{ComplexF64}

Per-m vector synthesis (no FFT) truncated at ltr.
"""
function SHsphtor_to_spat_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
    m = im * cfg.mres
    lstart = max(1, m)
    length(Sl) == ltr - lstart + 1 || throw(DimensionMismatch("Sl length must be ltr-max(1,m)+1"))
    length(Tl) == ltr - lstart + 1 || throw(DimensionMismatch("Tl length must be ltr-max(1,m)+1"))
    P = Vector{Float64}(undef, cfg.lmax + 1)
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)
    CT = promote_type(eltype(Sl), eltype(Tl))
    Vt_m = Vector{CT}(undef, cfg.nlat)
    Vp_m = Vector{CT}(undef, cfg.nlat)
    for i in 1:cfg.nlat
        x = cfg.x[i]
        sθ = sqrt(max(0.0, 1 - x*x))
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
        gθ = 0.0 + 0.0im
        gφ = 0.0 + 0.0im
        if cfg.use_plm_tables && length(cfg.plm_tables) > im && length(cfg.dplm_tables) > im
            tblP = cfg.plm_tables[im+1]; tbld = cfg.dplm_tables[im+1]
            @inbounds for l in lstart:ltr
                N = cfg.Nlm[l+1, m+1]
                dθY = -sθ * N * tbld[l+1, i]
                Y = N * tblP[l+1, i]
                Slv = Sl[l - lstart + 1]
                Tlv = Tl[l - lstart + 1]
                Ict = one(CT) * (0 + 1im)
                gθ += dθY * Slv + Ict * m * inv_sθ * Y * Tlv
                gφ += Ict * m * inv_sθ * Y * Slv + (sθ * N * tbld[l+1, i]) * Tlv
            end
        else
            Plm_and_dPdx_row!(P, dPdx, x, cfg.lmax, m)
            @inbounds for l in lstart:ltr
                N = cfg.Nlm[l+1, m+1]
                dθY = -sθ * N * dPdx[l+1]
                Y = N * P[l+1]
                Slv = Sl[l - lstart + 1]
                Tlv = Tl[l - lstart + 1]
                Ict = one(CT) * (0 + 1im)
                gθ += dθY * Slv + Ict * m * inv_sθ * Y * Tlv
                gφ += Ict * m * inv_sθ * Y * Slv + (sθ * N * dPdx[l+1]) * Tlv
            end
        end
        Vt_m[i] = gθ
        Vp_m[i] = gφ
    end
    return Vt_m, Vp_m
end

"""
    spat_to_SHqst_l(cfg::SHTConfig, Vr, Vt, Vp, ltr::Int)
        -> Qlm, Slm, Tlm
"""
function spat_to_SHqst_l(cfg::SHTConfig, Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix, ltr::Int)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    Qlm = analysis(cfg, Vr)
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    @inbounds for m in 0:cfg.mmax
        for l in (ltr+1):cfg.lmax
            Qlm[l+1, m+1] = 0
            Slm[l+1, m+1] = 0
            Tlm[l+1, m+1] = 0
        end
    end
    return Qlm, Slm, Tlm
end

"""
    SHqst_to_spat_l(cfg::SHTConfig, Qlm, Slm, Tlm, ltr::Int; real_output::Bool=true)
        -> Vr, Vt, Vp
"""
function SHqst_to_spat_l(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    Q2 = copy(Qlm); S2 = copy(Slm); T2 = copy(Tlm)
    @inbounds for m in 0:cfg.mmax
        for l in (ltr+1):cfg.lmax
            Q2[l+1, m+1] = 0
            S2[l+1, m+1] = 0
            T2[l+1, m+1] = 0
        end
    end
    Vr = synthesis(cfg, Q2; real_output)
    Vt, Vp = SHsphtor_to_spat(cfg, S2, T2; real_output)
    return Vr, Vt, Vp
end

"""
    spat_to_SHqst_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
        -> Ql, Sl, Tl
"""
function spat_to_SHqst_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
    Ql = spat_to_SH_ml(cfg, im, Vr_m, ltr)
    Sl, Tl = spat_to_SHsphtor_ml(cfg, im, Vt_m, Vp_m, ltr)
    return Ql, Sl, Tl
end

"""
    SHqst_to_spat_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
        -> Vr_m, Vt_m, Vp_m
"""
function SHqst_to_spat_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
    Vr_m = SH_to_spat_ml(cfg, im, Ql, ltr)
    Vt_m, Vp_m = SHsphtor_to_spat_ml(cfg, im, Sl, Tl, ltr)
    return Vr_m, Vt_m, Vp_m
end
