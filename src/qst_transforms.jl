#=
================================================================================
qst_transforms.jl - QST (Radial-Spheroidal-Toroidal) Vector Transforms
================================================================================

This file implements spherical harmonic transforms for full 3-component vector
fields on the sphere using the QST decomposition.

WHAT IS QST DECOMPOSITION?
--------------------------
QST extends the 2D spheroidal-toroidal decomposition to 3D by adding a radial
(Q) component:

    V(r,θ,φ) = V_r r̂ + V_θ θ̂ + V_φ φ̂

The three components are:
    Q (Radial):    Scalar field expanded in Y_l^m, gives V_r
    S (Spheroidal): Horizontal divergent flow (curl-free tangent)
    T (Toroidal):   Horizontal rotational flow (div-free tangent)

The horizontal components (V_θ, V_φ) come from S and T exactly as in
the sphtor_transforms.jl file:
    V_θ = ∂S/∂θ - (1/sin θ) ∂T/∂φ
    V_φ = (1/sin θ) ∂S/∂φ + ∂T/∂θ

PHYSICAL INTERPRETATION
-----------------------
In spherical geometry:
    Q_lm : radial flow strength at degree l, order m
    S_lm : horizontal divergent flow (linked to mass convergence/divergence)
    T_lm : horizontal rotational flow (linked to vorticity)

For a divergence-free 3D vector field (like incompressible flow):
    ∇·V = 0  ⟹  Q and S are related by continuity

IMPLEMENTATION
--------------
QST transforms are implemented by combining scalar and sphtor transforms:
    - Q component: standard scalar SH analysis/synthesis
    - S,T components: spheroidal-toroidal vector transforms

Main functions:
    SHqst_to_spat(cfg, Qlm, Slm, Tlm)    : Spectral → Spatial (Vr, Vθ, Vφ)
    spat_to_SHqst(cfg, Vr, Vt, Vp)       : Spatial → Spectral (Q, S, T)

Variants:
    *_cplx     : Complex-valued output
    *_l        : Degree-limited (truncate at degree ltr)
    *_ml       : Single azimuthal mode (for mode-by-mode processing)

USAGE EXAMPLE
-------------
```julia
cfg = create_gauss_config(32, 64)

# Create spectral coefficients
Qlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Slm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
Tlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)

Qlm[3, 1] = 1.0  # Radial l=2, m=0
Slm[4, 2] = 0.5  # Spheroidal l=3, m=1

# Synthesize to spatial
Vr, Vt, Vp = SHqst_to_spat(cfg, Qlm, Slm, Tlm)

# Analyze back to spectral
Q2, S2, T2 = spat_to_SHqst(cfg, Vr, Vt, Vp)
@assert Q2 ≈ Qlm
@assert S2 ≈ Slm
```

APPLICATIONS
------------
- Geodynamics: mantle convection patterns
- Astrophysics: stellar internal flows
- Geomagnetic field modeling (poloidal-toroidal decomposition)
- Any 3D vector field in spherical coordinates

================================================================================
=#

"""
QST Vector Field Transforms

This module handles transforms for 3-component vector fields using the QST decomposition:
- Q: radial (spheroidal) component
- S: tangential spheroidal component
- T: tangential toroidal component

This representation is natural for 3D vector fields on the sphere, such as
velocity fields in spherical coordinates (Vr, Vt, Vp).
"""

"""
    SHqst_to_spat(cfg, Qlm, Slm, Tlm; real_output=true) -> (Vr, Vt, Vp)

Transform QST spectral coefficients to 3D spatial vector field components.
Returns radial (Vr), colatitude (Vt), and azimuthal (Vp) components.
"""
function SHqst_to_spat(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlat, nlon = cfg.nlat, cfg.nlon
    
    # Validate input dimensions
    validate_qst_dimensions(Qlm, Slm, Tlm, cfg)
    
    # Get the spatial components
    Vr = synthesis(cfg, Qlm; real_output=real_output)
    Vt, Vp = SHsphtor_to_spat(cfg, Slm, Tlm; real_output=real_output)
    
    return Vr, Vt, Vp
end

"""
    spat_to_SHqst(cfg, Vr, Vt, Vp) -> (Qlm, Slm, Tlm)

Transform 3D spatial vector field to QST spectral coefficients.
Input: radial (Vr), colatitude (Vt), and azimuthal (Vp) components.
"""
function spat_to_SHqst(cfg::SHTConfig, Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    
    # Validate input dimensions
    validate_vector_spatial_dimensions(Vr, Vt, Vp, cfg)
    
    # Transform each component
    Qlm = analysis(cfg, Vr)
    Slm, Tlm = spat_to_SHsphtor(cfg, Vt, Vp)
    
    return Qlm, Slm, Tlm
end

"""
    SHqst_to_spat_cplx(cfg, Qlm, Slm, Tlm) -> (Vr, Vt, Vp)

Complex version of QST to spatial transform, preserving complex values.
"""
function SHqst_to_spat_cplx(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    
    # Transform to spatial fields keeping complex values
    Vr = synthesis(cfg, Qlm; real_output=false)
    Vt, Vp = SHsphtor_to_spat_cplx(cfg, Slm, Tlm)
    
    return Vr, Vt, Vp
end

"""
    spat_cplx_to_SHqst(cfg, Vr, Vt, Vp) -> (Qlm, Slm, Tlm)

Transform complex spatial vector field to QST coefficients.
"""
function spat_cplx_to_SHqst(cfg::SHTConfig, Vr::AbstractMatrix{<:Complex}, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
    nlat, nlon = cfg.nlat, cfg.nlon
    
    # Validate input dimensions
    validate_vector_spatial_dimensions(Vr, Vt, Vp, cfg)
    
    # Transform each component
    Qlm = analysis(cfg, Vr)
    Slm, Tlm = spat_cplx_to_SHsphtor(cfg, Vt, Vp)
    
    return Qlm, Slm, Tlm
end

"""
    spat_to_SHqst_l(cfg, Vr, Vt, Vp, ltr) -> (Qlm, Slm, Tlm)

Degree-limited version of spat_to_SHqst, computing coefficients only up to degree ltr.
"""
function spat_to_SHqst_l(cfg::SHTConfig, Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix, ltr::Int)
    # Get full transforms first
    Qlm, Slm, Tlm = spat_to_SHqst(cfg, Vr, Vt, Vp)
    
    # Create copies and zero out high-degree modes
    Q2, S2, T2 = copy_spectral_triple(Qlm, Slm, Tlm)
    zero_high_degree_modes!((Q2, S2, T2), cfg, ltr)
    
    return Q2, S2, T2
end

"""
    SHqst_to_spat_l(cfg, Qlm, Slm, Tlm, ltr; real_output=true) -> (Vr, Vt, Vp)

Degree-limited version of SHqst_to_spat, using coefficients only up to degree ltr.
"""
function SHqst_to_spat_l(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    
    # Create truncated coefficient arrays
    Q2 = copy(Qlm); S2 = copy(Slm); T2 = copy(Tlm)
    
    # Zero high-degree modes
    for m in 0:mmax, l in (ltr+1):lmax
        if l >= m
            Q2[l+1, m+1] = 0.0
            S2[l+1, m+1] = 0.0
            T2[l+1, m+1] = 0.0
        end
    end
    
    return SHqst_to_spat(cfg, Q2, S2, T2; real_output=real_output)
end

"""
    spat_to_SHqst_ml(cfg, im, Vr_m, Vt_m, Vp_m, ltr) -> (Ql, Sl, Tl)

Mode-limited transform for specific azimuthal mode im.
"""
function spat_to_SHqst_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
    # Transform each component for this specific mode
    Ql = spat_to_SH_ml(cfg, im, Vr_m, ltr)
    Sl, Tl = spat_to_SHsphtor_ml(cfg, im, Vt_m, Vp_m, ltr)
    
    return Ql, Sl, Tl
end

"""
    SHqst_to_spat_ml(cfg, im, Ql, Sl, Tl, ltr) -> (Vr_m, Vt_m, Vp_m)

Mode-limited synthesis for specific azimuthal mode im.
"""
function SHqst_to_spat_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
    # Synthesize each component for this specific mode
    Vr_m = SH_to_spat_ml(cfg, im, Ql, ltr)
    Vt_m, Vp_m = SHsphtor_to_spat_ml(cfg, im, Sl, Tl, ltr)
    
    return Vr_m, Vt_m, Vp_m
end