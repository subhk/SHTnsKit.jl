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
    synthesis_qst(cfg, Qlm, Slm, Tlm)    : Spectral → Spatial (Vr, Vθ, Vφ)
    analysis_qst(cfg, Vr, Vt, Vp)        : Spatial → Spectral (Q, S, T)

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
Vr, Vt, Vp = synthesis_qst(cfg, Qlm, Slm, Tlm)

# Analyze back to spectral
Q2, S2, T2 = analysis_qst(cfg, Vr, Vt, Vp)
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
    synthesis_qst(cfg, Qlm, Slm, Tlm; real_output=true) -> (Vr, Vt, Vp)

Transform QST spectral coefficients to 3D spatial vector field components.
Returns radial (Vr), colatitude (Vt), and azimuthal (Vp) components.
"""
Base.@constprop :aggressive function synthesis_qst(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    # QST synthesis is scalar Q plus horizontal S/T synthesis. Use a Val
    # barrier so all returned component arrays have concrete element types.
    return _synthesis_qst(cfg, Qlm, Slm, Tlm, Val(real_output))
end

function _synthesis_qst(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix,
                        ::Val{real_output}) where {real_output}
    validate_qst_dimensions(Qlm, Slm, Tlm, cfg)
    # Reuse the scalar and sphtor implementations for rfft choices and
    # complex-output behavior — but NOT for normalization, which they disagree
    # on: `_synthesis_sphtor` converts cfg→internal (src/sphtor_transforms.jl)
    # while `_synthesis` is orthonormal-only by contract. A QST triple must be
    # on ONE convention, and every other QST API (the distributed pair, and the
    # S/T half here) uses the config's, so Q is converted to match rather than
    # silently interpreted in a different basis from its own S and T.
    # Broadcast, NOT `convert_alm_norm!`: this body has no rrule, so Zygote
    # traces straight through it, and the in-place `setindex!` inside
    # `convert_alm_norm!` raises "Mutating arrays is not supported". `M` is the
    # same cached scale matrix that function consumes; entries with l < m are 1.
    # Inverse of `analysis_qst`: the triple is orthonormal, `_synthesis` wants
    # exactly that, and `_synthesis_sphtor` wants cfg-form S/T.
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        M = _ensure_norm_scale_matrix!(cfg)
        Slm = Slm ./ M
        Tlm = Tlm ./ M
    end
    Vr = _synthesis(cfg, Qlm, Val(real_output), nothing, Val(false))
    Vt, Vp = _synthesis_sphtor(cfg, Slm, Tlm, Val(real_output), Val(false))

    return Vr, Vt, Vp
end

"""
    analysis_qst(cfg, Vr, Vt, Vp) -> (Qlm, Slm, Tlm)

Transform 3D spatial vector field to QST spectral coefficients.
Input: radial (Vr), colatitude (Vt), and azimuthal (Vp) components.
"""
function analysis_qst(cfg::SHTConfig, Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon

    # Validate input dimensions
    validate_vector_spatial_dimensions(Vr, Vt, Vp, cfg)

    # Transform each component. `analysis_sphtor` returns S/T in the config's
    # convention while `analysis` is orthonormal-only, so Q is converted to match
    # — otherwise this single call hands back a triple on TWO different
    # normalizations, which then disagrees with `dist_analysis_qst` (uniformly
    # cfg-form) and mis-evaluates if Q is passed to a converting consumer such
    # as `SH_to_lat`. Inverse of the conversion in `_synthesis_qst`.
    # The QST triple is ORTHONORMAL throughout, matching serial `analysis`, the
    # energy diagnostics and the distributed backend. `analysis` already is;
    # `analysis_sphtor` returns cfg-form, so S/T are converted back. Broadcasts,
    # not `convert_alm_norm!`: this body has no rrule and is Zygote-traced, and an
    # in-place write raises "Mutating arrays is not supported".
    Qlm = analysis(cfg, Vr)
    Slm, Tlm = analysis_sphtor(cfg, Vt, Vp)
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        M = _ensure_norm_scale_matrix!(cfg)
        Slm = Slm .* M
        Tlm = Tlm .* M
    end

    return Qlm, Slm, Tlm
end

"""
    synthesis_qst_cplx(cfg, Qlm, Slm, Tlm) -> (Vr, Vt, Vp)

Complex version of QST to spatial transform, preserving complex values.
"""
function synthesis_qst_cplx(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix)
    validate_qst_dimensions(Qlm, Slm, Tlm, cfg)
    # `synthesis_sphtor_cplx` delegates to `_synthesis_sphtor`, which converts
    # cfg→internal, while `synthesis_cplx` is orthonormal-only — so Q needs the
    # same conversion the real-output `_synthesis_qst` applies. (Both `_cplx`
    # sphtor wrappers are thin delegators to the CONVERTING implementations;
    # reading the wrapper bodies alone suggests otherwise.)
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        M = _ensure_norm_scale_matrix!(cfg)
        Slm = Slm ./ M
        Tlm = Tlm ./ M
    end
    Vr = synthesis_cplx(cfg, Qlm)
    Vt, Vp = synthesis_sphtor_cplx(cfg, Slm, Tlm)

    return Vr, Vt, Vp
end

"""
    analysis_qst_cplx(cfg, Vr, Vt, Vp) -> (Qlm, Slm, Tlm)

Transform complex spatial vector field to QST coefficients.
"""
function analysis_qst_cplx(cfg::SHTConfig, Vr::AbstractMatrix{<:Complex}, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
    nlat, nlon = cfg.nlat, cfg.nlon

    # Validate input dimensions
    validate_vector_spatial_dimensions(Vr, Vt, Vp, cfg)

    # Transform each component. `analysis_sphtor_cplx` delegates to
    # `analysis_sphtor`, which converts internal→cfg, so Q must match or this
    # returns a triple on two normalizations (see `analysis_qst`).
    Qlm = analysis(cfg, Vr)
    Slm, Tlm = analysis_sphtor_cplx(cfg, Vt, Vp)
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        M = _ensure_norm_scale_matrix!(cfg)
        Slm = Slm .* M
        Tlm = Tlm .* M
    end

    return Qlm, Slm, Tlm
end

"""
    analysis_qst_l(cfg, Vr, Vt, Vp, ltr) -> (Qlm, Slm, Tlm)

Degree-limited version of analysis_qst, computing coefficients only up to degree ltr.
"""
function analysis_qst_l(cfg::SHTConfig, Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix, ltr::Int)
    # Get full transforms first
    Qlm, Slm, Tlm = analysis_qst(cfg, Vr, Vt, Vp)

    # Create copies and zero out high-degree modes
    Q2, S2, T2 = copy_spectral_triple(Qlm, Slm, Tlm)
    zero_high_degree_modes!((Q2, S2, T2), cfg, ltr)

    return Q2, S2, T2
end

"""
    synthesis_qst_l(cfg, Qlm, Slm, Tlm, ltr; real_output=true) -> (Vr, Vt, Vp)

Degree-limited version of synthesis_qst, using coefficients only up to degree ltr.
"""
Base.@constprop :aggressive function synthesis_qst_l(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    return _synthesis_qst_l(cfg, Qlm, Slm, Tlm, ltr, Val(real_output))
end

function synthesis_qst_l_cplx(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int)
    # Dedicated helper mirrors `synthesis_qst_l(...; real_output=false)` while
    # keeping the output tuple concrete for inference-sensitive code.
    return _synthesis_qst_l(cfg, Qlm, Slm, Tlm, ltr, Val(false))
end

function _synthesis_qst_l(cfg::SHTConfig, Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix,
                          ltr::Int, ::Val{real_output}) where {real_output}
    validate_qst_dimensions(Qlm, Slm, Tlm, cfg)
    # Same convention split as `_synthesis_qst`: `_synthesis_sphtor_l` converts
    # cfg→internal, `_synthesis_l` does not, and `analysis_qst_l` (which routes
    # through `analysis_qst`) returns Q in cfg's convention — so convert Q here
    # too, or the degree-limited round trip breaks for any non-default norm.
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        M = _ensure_norm_scale_matrix!(cfg)
        Slm = Slm ./ M
        Tlm = Tlm ./ M
    end
    Vr = _synthesis_l(cfg, Qlm, ltr, Val(real_output))
    Vt, Vp = _synthesis_sphtor_l(cfg, Slm, Tlm, ltr, Val(real_output))
    return Vr, Vt, Vp
end

"""
    analysis_qst_ml(cfg, im, Vr_m, Vt_m, Vp_m, ltr) -> (Ql, Sl, Tl)

Mode-limited transform for specific azimuthal mode im.
"""
function analysis_qst_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
    # Transform each component for this specific mode
    Ql = analysis_packed_ml(cfg, im, Vr_m, ltr)
    Sl, Tl = analysis_sphtor_ml(cfg, im, Vt_m, Vp_m, ltr)
    # `analysis_sphtor_ml` converts internal→cfg but `analysis_packed_ml` does
    # not, so without this the returned triple sits on two normalizations — the
    # same defect fixed in `analysis_qst`. Mirrors that function's scaling.
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        # `analysis_sphtor_ml`/`synthesis_sphtor_ml` scale only l >= max(1,im)
        # (l=0 has no vector part), so the l=0 slot must stay unscaled here too.
        sc = [l < max(1, im) ? 1.0 :
              norm_scale_from_orthonormal(l, im, cfg.norm) * cs_phase_factor(im, true, cfg.cs_phase)
              for l in im:ltr]
        Sl = Sl .* sc          # analysis_sphtor_ml returns cfg-form; make it orthonormal
        Tl = Tl .* sc
    end

    return Ql, Sl, Tl
end

"""
    synthesis_qst_ml(cfg, im, Ql, Sl, Tl, ltr) -> (Vr_m, Vt_m, Vp_m)

Mode-limited synthesis for specific azimuthal mode im.
"""
function synthesis_qst_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
    # Synthesize each component for this specific mode
    # Inverse of the conversion in `analysis_qst_ml`: Q arrives in cfg's
    # convention (matching S/T) but `synthesis_packed_ml` expects internal.
    if cfg.norm !== :orthonormal || cfg.cs_phase == false
        # `analysis_sphtor_ml`/`synthesis_sphtor_ml` scale only l >= max(1,im)
        # (l=0 has no vector part), so the l=0 slot must stay unscaled here too.
        sc = [l < max(1, im) ? 1.0 :
              norm_scale_from_orthonormal(l, im, cfg.norm) * cs_phase_factor(im, true, cfg.cs_phase)
              for l in im:ltr]
        Sl = Sl ./ sc          # synthesis_sphtor_ml wants cfg-form S/T
        Tl = Tl ./ sc
    end
    Vr_m = synthesis_packed_ml(cfg, im, Ql, ltr)
    Vt_m, Vp_m = synthesis_sphtor_ml(cfg, im, Sl, Tl, ltr)

    return Vr_m, Vt_m, Vp_m
end
