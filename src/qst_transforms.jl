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