#=
================================================================================
sphtor_transforms.jl - Spheroidal-Toroidal Vector Field Transforms
================================================================================

This file implements spherical harmonic transforms for 2D horizontal (tangential)
vector fields on the sphere using the spheroidal-toroidal (S/T) decomposition.

PHYSICAL MOTIVATION
-------------------
Any smooth horizontal vector field V on the sphere can be uniquely decomposed:
    V = V_S + V_T  (spheroidal + toroidal)

where:
- Spheroidal (S): curl-free, associated with DIVERGENT flow (sources/sinks)
- Toroidal (T):   div-free, associated with ROTATIONAL flow (vortices)

This is analogous to the Helmholtz decomposition in 3D, but specialized for
tangent vectors on a 2-sphere.

MATHEMATICAL FORMULATION
------------------------
Given spheroidal potential S(θ,φ) and toroidal potential T(θ,φ), the vector
components are:

    V_θ = ∂S/∂θ - (1/sin θ) ∂T/∂φ    (colatitude/meridional component)
    V_φ = (1/sin θ) ∂S/∂φ + ∂T/∂θ    (azimuthal/zonal component)

In spectral space with Y_l^m(θ,φ) = N_lm P_l^m(cos θ) exp(imφ):

    S(θ,φ) = Σ_{l,m} S_lm Y_l^m(θ,φ)
    T(θ,φ) = Σ_{l,m} T_lm Y_l^m(θ,φ)

The scalar invariants (divergence and vorticity) are directly related:

    δ_lm = -l(l+1) S_lm    (divergence spectrum)
    ζ_lm = -l(l+1) T_lm    (vorticity spectrum)

This makes S/T decomposition extremely useful in fluid dynamics where
divergence and vorticity are fundamental quantities.

APPLICATIONS
------------
- Atmospheric/oceanic flows: wind velocity, ocean currents
- Geophysics: surface plate motions, magnetic field horizontal components
- Astrophysics: stellar surface flows, solar wind

IMPLEMENTATION STRUCTURE
------------------------
Main transforms:
    SHsphtor_to_spat(cfg, Slm, Tlm)   : Synthesis (spectral → spatial)
    spat_to_SHsphtor(cfg, Vt, Vp)     : Analysis (spatial → spectral)

Helper functions:
    SHsph_to_spat(cfg, Slm)           : Spheroidal-only synthesis (T=0)
    SHtor_to_spat(cfg, Tlm)           : Toroidal-only synthesis (S=0)

Spectral operators:
    divergence_from_spheroidal(cfg, Slm)   : δ_lm = -l(l+1) S_lm
    vorticity_from_toroidal(cfg, Tlm)      : ζ_lm = -l(l+1) T_lm
    spheroidal_from_divergence(cfg, δlm)   : Invert for S_lm
    toroidal_from_vorticity(cfg, ζlm)      : Invert for T_lm

Degree-limited variants (suffix _l):
    SHsphtor_to_spat_l, spat_to_SHsphtor_l, etc.

Mode-limited variants (suffix _ml):
    For single azimuthal mode m processing

DEBUGGING TIPS
--------------
1. Test with pure spheroidal field (should have zero vorticity):
   ```julia
   cfg = create_gauss_config(32, 64)
   Slm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
   Slm[3, 1] = 1.0  # l=2, m=0 mode
   Tlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
   Vt, Vp = SHsphtor_to_spat(cfg, Slm, Tlm)
   S_back, T_back = spat_to_SHsphtor(cfg, Vt, Vp)
   @assert norm(T_back) < 1e-10 "Pure spheroidal should have no toroidal"
   ```

2. Verify divergence/vorticity relationships:
   ```julia
   δ = divergence_from_spheroidal(cfg, Slm)
   @assert δ[3,1] ≈ -6.0 * Slm[3,1]  # -l(l+1) = -2*3 = -6
   ```

3. Check pole behavior - sin θ → 0 at poles:
   - V_φ term (1/sinθ)∂S/∂φ can be singular
   - Implementation guards against this via inv_sθ = sθ == 0 ? 0.0 : 1/sθ

4. Robert form scaling:
   - If cfg.robert_form = true, outputs are scaled by sin(θ)
   - This regularizes pole singularities for numerical stability

PERFORMANCE NOTES
-----------------
- Uses thread-local Legendre polynomial arrays to enable @threads parallelism
- Can use precomputed tables (cfg.plm_tables, cfg.dplm_tables) if available
- Requires BOTH P_l^m AND dP_l^m/dx for vector transforms (unlike scalar)

================================================================================
=#

"""
Spheroidal-Toroidal Vector Field Transforms

This module handles transforms for horizontal (tangential) vector fields using
spheroidal (S) and toroidal (T) decomposition. This is the natural representation
for 2D vector fields on the sphere, such as horizontal velocity components.

Key relationships:
- Vt = ∂S/∂θ - (1/sin θ) ∂T/∂φ  (colatitude component)
- Vp = (1/sin θ) ∂S/∂φ + ∂T/∂θ  (azimuthal component)

Where S represents the spheroidal (curl-free) part and T the toroidal (div-free) part.
The corresponding scalar invariants are:
- Divergence: δ = ∇·V = -∑_{l,m} l(l+1) S_lm Y_l^m
- Vorticity:  ζ = (∇×V)·r̂ = -∑_{l,m} l(l+1) T_lm Y_l^m
so spheroidal/toroidal spectra are directly tied to divergence and vorticity.
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
    # Use maxthreadid() to handle all possible thread IDs with static scheduling
    nthreads = Threads.maxthreadid()
    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:nthreads]
    thread_local_dPdx = [Vector{Float64}(undef, lmax + 1) for _ in 1:nthreads]
    
    # Scale continuous Fourier coefficients to DFT bins for ifft (factor nlon or nlon/(2π))
    inv_scaleφ = phi_inv_scale(cfg)

    # Process each azimuthal mode m in parallel
    # Use :static scheduling for consistent load distribution
    @threads :static for m in 0:mmax
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
                    # Vθ = ∂S/∂θ - (im/sinθ) * T
                    gθ += dθY * Sl - (0 + 1im) * m * inv_sθ * Y * Tl
                    # Vφ = (im/sinθ) * S + ∂T/∂θ
                    gφ += (0 + 1im) * m * inv_sθ * Y * Sl + dθY * Tl
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
                    # Vθ = ∂S/∂θ - (im/sinθ) * T
                    gθ += dθY * Sl - (0 + 1im) * m * inv_sθ * Y * Tl
                    # Vφ = (im/sinθ) * S + ∂T/∂θ
                    gφ += (0 + 1im) * m * inv_sθ * Y * Sl + dθY * Tl
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
The returned coefficients satisfy δ_lm = −l(l+1) S_lm (divergence) and
ζ_lm = −l(l+1) T_lm (vorticity) when expressed in the internal normalization.
"""
function spat_to_SHsphtor(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    if is_gpu_config(cfg)
        return gpu_spat_to_SHsphtor(cfg, Vt, Vp)
    end
    return spat_to_SHsphtor_cpu(cfg, Vt, Vp)
end

function spat_to_SHsphtor_cpu(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(Vt,1) == nlat && size(Vt,2) == nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1) == nlat && size(Vp,2) == nlon || throw(DimensionMismatch("Vp dims"))

    Ftθm = fft_phi(complex.(Vt))
    Fpθm = fft_phi(complex.(Vp))

    lmax, mmax = cfg.lmax, cfg.mmax
    Slm_int = zeros(ComplexF64, lmax + 1, mmax + 1)
    Tlm_int = zeros(ComplexF64, lmax + 1, mmax + 1)
    scaleφ = cfg.cphi

    use_tbl = cfg.use_plm_tables &&
              length(cfg.plm_tables) == mmax + 1 &&
              length(cfg.dplm_tables) == mmax + 1 &&
              !isempty(cfg.plm_tables)

    # Use maxthreadid() to handle all possible thread IDs with static scheduling
    nthreads = Threads.maxthreadid()
    thread_local_P = [Vector{Float64}(undef, lmax + 1) for _ in 1:nthreads]
    thread_local_dPdx = [Vector{Float64}(undef, lmax + 1) for _ in 1:nthreads]
    thread_local_Sacc = [Vector{ComplexF64}(undef, lmax + 1) for _ in 1:nthreads]
    thread_local_Tacc = [Vector{ComplexF64}(undef, lmax + 1) for _ in 1:nthreads]

    @threads :static for m in 0:mmax
        col = m + 1
        tid = Threads.threadid()
        
        Sacc = thread_local_Sacc[tid]; fill!(Sacc, 0.0 + 0.0im)
        Tacc = thread_local_Tacc[tid]; fill!(Tacc, 0.0 + 0.0im)

        for i in 1:nlat
            x = cfg.x[i]
            sθ = sqrt(max(0.0, 1 - x*x))
            inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
            wi = cfg.w[i]
            Fθ = Ftθm[i, col]
            Fφ = Fpθm[i, col]

            if cfg.robert_form && sθ > 0
                Fθ /= sθ
                Fφ /= sθ
            end

            if use_tbl
                tblP = cfg.plm_tables[col]
                tbld = cfg.dplm_tables[col]
                @inbounds for l in max(1, m):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * tbld[l+1, i]
                    Y = N * tblP[l+1, i]
                    coeff = wi * scaleφ / (l * (l + 1))
                    term = (0 + 1im) * m * inv_sθ * Y
                    # Adjoint of synthesis: Vθ = dθY*S - term*T, Vφ = term*S + dθY*T
                    Sacc[l+1] += coeff * (Fθ * dθY + conj(term) * Fφ)
                    Tacc[l+1] += coeff * (-conj(term) * Fθ + dθY * Fφ)
                end
            else
                P = thread_local_P[tid]
                dPdx = thread_local_dPdx[tid]
                Plm_and_dPdx_row!(P, dPdx, x, lmax, m)
                @inbounds for l in max(1, m):lmax
                    N = cfg.Nlm[l+1, col]
                    dθY = -sθ * N * dPdx[l+1]
                    Y = N * P[l+1]
                    coeff = wi * scaleφ / (l * (l + 1))
                    term = (0 + 1im) * m * inv_sθ * Y
                    # Adjoint of synthesis: Vθ = dθY*S - term*T, Vφ = term*S + dθY*T
                    Sacc[l+1] += coeff * (Fθ * dθY + conj(term) * Fφ)
                    Tacc[l+1] += coeff * (-conj(term) * Fθ + dθY * Fφ)
                end
            end
        end

        for l in max(1, m):lmax
            Slm_int[l+1, col] = Sacc[l+1]
            Tlm_int[l+1, col] = Tacc[l+1]
        end
    end

    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        S2 = similar(Slm_int)
        T2 = similar(Tlm_int)
        convert_alm_norm!(S2, Slm_int, cfg; to_internal=false)
        convert_alm_norm!(T2, Tlm_int, cfg; to_internal=false)
        return S2, T2
    else
        return Slm_int, Tlm_int
    end
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
    divergence_from_spheroidal(cfg, Slm) -> Matrix

Return divergence spectral coefficients δ_lm corresponding to spheroidal modes
via δ_lm = −l(l+1) S_lm (`δ_{00} = 0`).
"""
function divergence_from_spheroidal(cfg::SHTConfig, Slm::AbstractMatrix)
    δ = similar(Slm)
    return divergence_from_spheroidal!(cfg, δ, Slm)
end

function divergence_from_spheroidal!(cfg::SHTConfig, δ::AbstractMatrix, Slm::AbstractMatrix)
    size(δ) == size(Slm) || throw(DimensionMismatch("divergence output dims"))
    fill!(δ, zero(eltype(δ)))
    lmax, mmax = cfg.lmax, cfg.mmax
    @inbounds for m in 0:mmax, l in max(1, m):lmax
        row = l + 1; col = m + 1
        δ[row, col] = -(l * (l + 1)) * Slm[row, col]
    end
    return δ
end

"""
    spheroidal_from_divergence(cfg, δlm) -> Matrix

Invert δ_lm = −l(l+1) S_lm for l ≥ 1 to recover spheroidal coefficients.
"""
function spheroidal_from_divergence(cfg::SHTConfig, δlm::AbstractMatrix)
    Slm = similar(δlm)
    return spheroidal_from_divergence!(cfg, Slm, δlm)
end

function spheroidal_from_divergence!(cfg::SHTConfig, Slm::AbstractMatrix, δlm::AbstractMatrix)
    size(Slm) == size(δlm) || throw(DimensionMismatch("spheroidal output dims"))
    fill!(Slm, zero(eltype(Slm)))
    lmax, mmax = cfg.lmax, cfg.mmax
    @inbounds for m in 0:mmax, l in max(1, m):lmax
        row = l + 1; col = m + 1
        ll1 = l * (l + 1)
        Slm[row, col] = ll1 == 0 ? zero(eltype(Slm)) : -(δlm[row, col] / ll1)
    end
    return Slm
end

"""
    vorticity_from_toroidal(cfg, Tlm) -> Matrix

Return vorticity spectral coefficients ζ_lm = −l(l+1) T_lm.
"""
function vorticity_from_toroidal(cfg::SHTConfig, Tlm::AbstractMatrix)
    ζ = similar(Tlm)
    return vorticity_from_toroidal!(cfg, ζ, Tlm)
end

function vorticity_from_toroidal!(cfg::SHTConfig, ζ::AbstractMatrix, Tlm::AbstractMatrix)
    size(ζ) == size(Tlm) || throw(DimensionMismatch("vorticity output dims"))
    fill!(ζ, zero(eltype(ζ)))
    lmax, mmax = cfg.lmax, cfg.mmax
    @inbounds for m in 0:mmax, l in max(1, m):lmax
        row = l + 1; col = m + 1
        ζ[row, col] = -(l * (l + 1)) * Tlm[row, col]
    end
    return ζ
end

"""
    toroidal_from_vorticity(cfg, ζlm) -> Matrix

Recover toroidal stream function coefficients from vorticity spectra.
"""
function toroidal_from_vorticity(cfg::SHTConfig, ζlm::AbstractMatrix)
    Tlm = similar(ζlm)
    return toroidal_from_vorticity!(cfg, Tlm, ζlm)
end

function toroidal_from_vorticity!(cfg::SHTConfig, Tlm::AbstractMatrix, ζlm::AbstractMatrix)
    size(Tlm) == size(ζlm) || throw(DimensionMismatch("toroidal output dims"))
    fill!(Tlm, zero(eltype(Tlm)))
    lmax, mmax = cfg.lmax, cfg.mmax
    @inbounds for m in 0:mmax, l in max(1, m):lmax
        row = l + 1; col = m + 1
        ll1 = l * (l + 1)
        Tlm[row, col] = ll1 == 0 ? zero(eltype(Tlm)) : -(ζlm[row, col] / ll1)
    end
    return Tlm
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
    SHsph_to_spat_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, ltr::Int)

Mode-limited spheroidal-only synthesis wrapper.
"""
function SHsph_to_spat_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, ltr::Int)
    Tl_zero = zeros(eltype(Sl), length(Sl))
    return SHsphtor_to_spat_ml(cfg, im, Sl, Tl_zero, ltr)
end

"""
    SHtor_to_spat_ml(cfg::SHTConfig, im::Int, Tl::AbstractVector{<:Complex}, ltr::Int)

Mode-limited toroidal-only synthesis wrapper.
"""
function SHtor_to_spat_ml(cfg::SHTConfig, im::Int, Tl::AbstractVector{<:Complex}, ltr::Int)
    Sl_zero = zeros(eltype(Tl), length(Tl))
    return SHsphtor_to_spat_ml(cfg, im, Sl_zero, Tl, ltr)
end

"""
    spat_to_SHsphtor_ml(cfg, im, Vt_m, Vp_m, ltr) -> (Sl, Tl)

Mode-limited transform for specific azimuthal mode im.
Input vectors contain Fourier coefficients for mode m at all latitudes.
Implements the proper spheroidal-toroidal decomposition using the adjoint of
the synthesis formulas:
    Vθ = ∂S/∂θ - (im/sinθ) * T
    Vφ = (im/sinθ) * S + ∂T/∂θ
"""
function spat_to_SHsphtor_ml(cfg::SHTConfig, im::Int, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    length(Vt_m) == nlat || throw(DimensionMismatch("Vt_m length must be nlat"))
    length(Vp_m) == nlat || throw(DimensionMismatch("Vp_m length must be nlat"))

    num_l = ltr - im + 1
    Sl = Vector{ComplexF64}(undef, num_l)
    Tl = Vector{ComplexF64}(undef, num_l)
    fill!(Sl, 0.0 + 0.0im)
    fill!(Tl, 0.0 + 0.0im)

    P = Vector{Float64}(undef, ltr + 1)
    dPdx = Vector{Float64}(undef, ltr + 1)
    scaleφ = cfg.cphi

    # Integrate using Legendre polynomials and derivatives
    for i in 1:nlat
        x = cfg.x[i]
        sθ = sqrt(max(0.0, 1 - x*x))
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ
        wi = cfg.w[i]
        Fθ = Vt_m[i]
        Fφ = Vp_m[i]

        # Handle Robert form scaling if needed
        if cfg.robert_form && sθ > 0
            Fθ = Fθ / sθ
            Fφ = Fφ / sθ
        end

        Plm_and_dPdx_row!(P, dPdx, x, ltr, im)

        @inbounds for l in max(1, im):ltr
            N = cfg.Nlm[l+1, im+1]
            dθY = -sθ * N * dPdx[l+1]
            Y = N * P[l+1]
            ll1 = l * (l + 1)
            coeff = wi * scaleφ / ll1
            term = (0 + 1im) * im * inv_sθ * Y

            # Adjoint of synthesis: Vθ = dθY*S - term*T, Vφ = term*S + dθY*T
            Sl[l-im+1] += coeff * (Fθ * dθY + conj(term) * Fφ)
            Tl[l-im+1] += coeff * (-conj(term) * Fθ + dθY * Fφ)
        end
    end

    return Sl, Tl
end

"""
    SHsphtor_to_spat_ml(cfg, im, Sl, Tl, ltr) -> (Vt_m, Vp_m)

Mode-limited synthesis for specific azimuthal mode im.
Implements the proper spheroidal-toroidal synthesis formulas:
    Vθ = ∂S/∂θ - (im/sinθ) * T
    Vφ = (im/sinθ) * S + ∂T/∂θ
"""
function SHsphtor_to_spat_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    expected_len = ltr - im + 1
    length(Sl) == expected_len || throw(DimensionMismatch("Sl length mismatch"))
    length(Tl) == expected_len || throw(DimensionMismatch("Tl length mismatch"))

    Vt_m = Vector{ComplexF64}(undef, nlat)
    Vp_m = Vector{ComplexF64}(undef, nlat)
    P = Vector{Float64}(undef, ltr + 1)
    dPdx = Vector{Float64}(undef, ltr + 1)

    # Synthesize vector components for this mode
    for i in 1:nlat
        x = cfg.x[i]
        sθ = sqrt(max(0.0, 1 - x*x))
        inv_sθ = sθ == 0 ? 0.0 : 1.0 / sθ

        Plm_and_dPdx_row!(P, dPdx, x, ltr, im)

        gθ = 0.0 + 0.0im
        gφ = 0.0 + 0.0im

        @inbounds for l in im:ltr
            N = cfg.Nlm[l+1, im+1]
            dθY = -sθ * N * dPdx[l+1]
            Y = N * P[l+1]
            S_coef = Sl[l-im+1]
            T_coef = Tl[l-im+1]

            # Vθ = ∂S/∂θ - (im/sinθ) * T
            gθ += dθY * S_coef - (0 + 1im) * im * inv_sθ * Y * T_coef
            # Vφ = (im/sinθ) * S + ∂T/∂θ
            gφ += (0 + 1im) * im * inv_sθ * Y * S_coef + dθY * T_coef
        end

        # Apply Robert form scaling if needed
        if cfg.robert_form
            gθ *= sθ
            gφ *= sθ
        end

        Vt_m[i] = gθ
        Vp_m[i] = gφ
    end

    return Vt_m, Vp_m
end

"""
    SH_to_grad_spat(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true)

Gradient synthesis alias for compatibility with SHTns.
"""
function SH_to_grad_spat(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true)
    return SHsph_to_spat(cfg, Slm; real_output=real_output)
end

"""
    SH_to_grad_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, ltr::Int; real_output::Bool=true)

Degree-limited gradient synthesis alias.
"""
function SH_to_grad_spat_l(cfg::SHTConfig, Slm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    return SHsph_to_spat_l(cfg, Slm, ltr; real_output=real_output)
end

"""
    SH_to_grad_spat_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, ltr::Int)

Mode-limited gradient synthesis alias.
"""
function SH_to_grad_spat_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, ltr::Int)
    return SHsph_to_spat_ml(cfg, im, Sl, ltr)
end
