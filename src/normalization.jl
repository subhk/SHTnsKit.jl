"""
Spherical Harmonic Normalization and Phase Conversion Utilities

This module handles conversions between different normalization conventions and
phase definitions used in spherical harmonic analysis. Different fields use
different conventions, so we provide conversion utilities to maintain compatibility.

Internal Convention:
- SHTnsKit internally uses orthonormal spherical harmonics with Condon-Shortley phase
- This ensures numerical stability and follows physics conventions

External Conventions Supported:
- :orthonormal - Standard physics normalization: ∫ Y_l^m (Y_{l'}^{m'})* dΩ = δ_{ll'} δ_{mm'}
- :fourpi - Geodesy convention: Y_l^m scaled by sqrt(4π)  
- :schmidt - Semi-normalized: common in geomagnetism and geodesy

Phase Conventions:
- Condon-Shortley phase: includes (-1)^m factor (standard in physics)
- No CS phase: omits the (-1)^m factor (used in some mathematics texts)
"""

"""
    norm_scale_from_orthonormal(l::Int, m::Int, to::Symbol) -> Float64

Calculate the scaling factor to convert from orthonormal to target normalization.

Returns k such that Y_target = k * Y_orthonormal, allowing conversion between
different spherical harmonic normalization conventions while preserving the
mathematical relationships.

The scale factor depends on the target convention:
- :orthonormal → k = 1 (no scaling)
- :fourpi → k = sqrt(4π) (geodesy convention)
- :schmidt → k = sqrt(4π/(2l+1)) (semi-normalized, common in geomagnetics)
"""
function norm_scale_from_orthonormal(l::Int, m::Int, to::Symbol)
    if to === :orthonormal
        # No conversion needed
        return 1.0
        
    elseif to === :fourpi
        # Geodesy convention: multiply by sqrt(4π) 
        # This removes the 1/sqrt(4π) factor from orthonormal normalization
        return sqrt(4π)
        
    elseif to === :schmidt
        # Schmidt semi-normalized spherical harmonics
        # Used extensively in geomagnetic field modeling (e.g., IGRF, WMM)
        return sqrt(4π / (2l + 1))
        
    else
        throw(ArgumentError("Unsupported normalization: $to"))
    end
end

"""
    cs_phase_factor(m::Int, cs_from::Bool, cs_to::Bool) -> Float64

Calculate the phase factor for converting between Condon-Shortley conventions.

The Condon-Shortley phase is a (-1)^m factor included in some spherical harmonic
definitions. This function returns the scaling factor α such that:
Y_to = α * Y_from when switching phase conventions.

The conversion rule is:
- If cs_from = cs_to: α = 1 (no change needed)
- If switching: α = (-1)^m (the CS phase factor itself)

Note: This applies to the basis functions. For coefficients, the transformation
may need to be inverted depending on the context.
"""
function cs_phase_factor(m::Int, cs_from::Bool, cs_to::Bool)
    if cs_from == cs_to
        # No phase conversion needed
        return 1.0
    else
        # Apply Condon-Shortley phase toggle: (-1)^m
        # This handles switching between CS and non-CS conventions
        return (-1.0)^m
    end
end

"""
    convert_alm_norm!(dest, src, cfg; to_internal::Bool=false)

Convert coefficient matrix `src` between cfg's normalization/phase and internal
orthonormal+CS. If `to_internal=true`, maps from cfg to internal. Otherwise maps
from internal to cfg. Writes into `dest` which must match `src` size.
"""
function convert_alm_norm!(dest::AbstractMatrix, src::AbstractMatrix, cfg; to_internal::Bool=false)
    size(dest) == size(src) || throw(DimensionMismatch("dest/src dims mismatch"))
    lmax, mmax = cfg.lmax, cfg.mmax
    # From orthonormal+CS to cfg: alm_cfg = alm_int / k_norm / α_cs
    # To internal: alm_int = alm_cfg * k_norm * α_cs
    for m in 0:mmax
        α = cs_phase_factor(m, true, cfg.cs_phase)  # Y_cfg = α * Y_int if toggling CS
        for l in m:lmax
            k = norm_scale_from_orthonormal(l, m, cfg.norm)
            s = to_internal ? (k * α) : (1.0 / (k * α))
            dest[l+1, m+1] = s * src[l+1, m+1]
        end
    end
    return dest
end

