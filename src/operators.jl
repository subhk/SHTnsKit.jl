#=
================================================================================
operators.jl - Spherical Differential Operators in Spectral Space
================================================================================

This file implements differential operators that act directly on spherical
harmonic coefficients WITHOUT going to physical space. This enables efficient
spectral methods for solving PDEs on the sphere.

WHY SPECTRAL OPERATORS?
-----------------------
Many differential equations on the sphere can be solved more efficiently in
spectral space. Instead of computing derivatives by finite differences, we
apply recurrence relations that couple neighboring degrees.

Key insight: Multiplication by cos(θ) or differentiation ∂/∂θ ONLY couples
modes with adjacent l values (l-1 and l+1) at fixed m. This makes the
operators sparse (tridiagonal in l).

OPERATORS IMPLEMENTED
---------------------
1. cos(θ) multiplication:
   cos(θ) Y_l^m = a_l^m Y_{l-1}^m + b_l^m Y_{l+1}^m

   where:
   a_l^m = sqrt[(l² - m²) / ((2l-1)(2l+1))]
   b_l^m = sqrt[((l+1)² - m²) / ((2l+1)(2l+3))]

2. sin(θ) ∂/∂θ derivative:
   sin(θ) ∂/∂θ Y_l^m = l b_l^m Y_{l+1}^m - (l+1) a_l^m Y_{l-1}^m

   The sin(θ) factor removes the coordinate singularity at poles.

MATRIX STORAGE FORMAT
---------------------
Operators are stored as vectors of coupling coefficients in packed format:
    mx[2*lm + 1] = c_minus  (coupling to l-1)
    mx[2*lm + 2] = c_plus   (coupling to l+1)

where lm is the packed index for (l,m) in SHTns LM order (m≥0 only).

APPLICATIONS
------------
- Solving Laplace/Poisson equations spectrally
- Computing Coriolis terms in shallow water equations
- Latitude-weighted integration
- Building spectral Jacobians for implicit solvers

USAGE EXAMPLE
-------------
```julia
cfg = create_gauss_config(32, 64)
mx = zeros(2 * cfg.nlm)  # Allocate coefficient array

# Fill with cos(θ) operator coefficients
mul_ct_matrix(cfg, mx)

# Apply operator to spectral field
Qlm = pack_alm(cfg, alm)  # Convert to packed format
Rlm = similar(Qlm)
SH_mul_mx(cfg, mx, Qlm, Rlm)  # R = cos(θ) * Q in spectral space
```

DEBUGGING
---------
```julia
# Verify operator application matches spatial multiplication
f = synthesis(cfg, alm)
f_times_costh = f .* cos.(acos.(cfg.x'))  # Spatial multiplication
alm_check = analysis(cfg, f_times_costh)

# Should match spectral operator result
@assert alm_check ≈ unpack_alm(cfg, Rlm)
```

================================================================================
=#

"""
Spherical Differential Operators in Spectral Space

This module implements differential operators that act on spherical harmonic coefficients.
The operators are represented as sparse matrices that couple neighboring degrees l at fixed
azimuthal order m, taking advantage of the recurrence relations for spherical harmonics.

The key operators implemented are:
- cos(θ) multiplication: couples Y_l^m to Y_{l±1}^m
- sin(θ) ∂/∂θ derivative: couples Y_l^m to Y_{l±1}^m with different coefficients

SHTns-compatible functions provided:
- `mul_ct_matrix(cfg, mx)`: fill `mx` (length 2*nlm) with coefficients for cos(θ) operator
- `st_dt_matrix(cfg, mx)`: fill `mx` (length 2*nlm) with coefficients for sin(θ) ∂/∂θ operator
- `SH_mul_mx(cfg, mx, Qlm, Rlm)`: apply a tridiagonal operator that couples (l,m) to l±1 at fixed m

Matrix storage format:
For each packed index `lm` (SHTns LM order, m≥0), we store two coupling coefficients:
- `mx[2*lm+1] = c_minus` for coupling to Y_{l-1}^m (lower degree neighbor)
- `mx[2*lm+2] = c_plus` for coupling to Y_{l+1}^m (higher degree neighbor)
Out-of-range neighbors (l<m or l>lmax) are automatically ignored.
"""

"""
    mul_ct_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real})

Fill `mx` with coupling coefficients for cosθ operator: cosθ Y_l^m = a_l^m Y_{l-1}^m + b_l^m Y_{l+1}^m.
"""
function mul_ct_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real})
    # Validate coefficient matrix size (2 coefficients per (l,m) mode)
    length(mx) == 2*cfg.nlm || throw(DimensionMismatch("mx length must be 2*nlm=$(2*cfg.nlm)"))
    
    # Fill coupling coefficients for each (l,m) mode in packed order
    @inbounds for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0+1]  # Degree for this packed index
        m = cfg.mi[lm0+1]  # Order for this packed index
        
        # Coupling coefficient to Y_{l-1}^m (downward in degree)
        # From recurrence: cos(θ) Y_l^m = a_l^m Y_{l-1}^m + b_l^m Y_{l+1}^m
        a = (l == 0) ? 0.0 : sqrt(max(0.0, (l^2 - m^2) / ((2l - 1) * (2l + 1))))
        
        # Coupling coefficient to Y_{l+1}^m (upward in degree)  
        b = (l == cfg.lmax) ? 0.0 : sqrt(max(0.0, ((l + 1)^2 - m^2) / ((2l + 1) * (2l + 3))))
        
        # Store in packed format: [c_minus, c_plus] for each (l,m)
        mx[2*lm0 + 1] = a  # Coefficient for Y_{l-1}^m
        mx[2*lm0 + 2] = b  # Coefficient for Y_{l+1}^m
    end
    return mx
end

"""
    st_dt_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real})

Fill `mx` with coupling coefficients for sinθ ∂_θ operator:
sinθ ∂_θ Y_l^m = l b_l^m Y_{l+1}^m - (l+1) a_l^m Y_{l-1}^m.
"""
function st_dt_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real})
    # Validate coefficient matrix size (2 coefficients per (l,m) mode)
    length(mx) == 2*cfg.nlm || throw(DimensionMismatch("mx length must be 2*nlm=$(2*cfg.nlm)"))
    
    # Fill coupling coefficients for the sin(θ) ∂/∂θ operator
    @inbounds for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0+1]  # Degree for this packed index
        m = cfg.mi[lm0+1]  # Order for this packed index
        
        # Base coupling coefficients (same as cos(θ) operator)
        a = (l == 0) ? 0.0 : sqrt(max(0.0, (l^2 - m^2) / ((2l - 1) * (2l + 1))))
        b = (l == cfg.lmax) ? 0.0 : sqrt(max(0.0, ((l + 1)^2 - m^2) / ((2l + 1) * (2l + 3))))
        
        # Apply derivative operator weights
        # sin(θ) ∂/∂θ Y_l^m = l b_l^m Y_{l+1}^m - (l+1) a_l^m Y_{l-1}^m
        c_minus = -(l + 1) * a  # Coefficient for Y_{l-1}^m (negative contribution)
        c_plus  =  l * b        # Coefficient for Y_{l+1}^m (positive contribution)
        
        # Store in packed format: [c_minus, c_plus] for each (l,m)
        mx[2*lm0 + 1] = c_minus  # Coefficient for Y_{l-1}^m  
        mx[2*lm0 + 2] = c_plus   # Coefficient for Y_{l+1}^m
    end
    return mx
end

"""
    SH_mul_mx(cfg::SHTConfig, mx::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Apply a nearest-neighbor-in-l operator represented by `mx` to `Qlm` and write to `Rlm`.
Both `Qlm` and `Rlm` are length `cfg.nlm` packed vectors (m≥0, SHTns LM order).
"""
function SH_mul_mx(cfg::SHTConfig, mx::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    # Validate input array dimensions
    length(mx) == 2*cfg.nlm || throw(DimensionMismatch("mx length must be 2*nlm=$(2*cfg.nlm)"))
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    length(Rlm) == cfg.nlm || throw(DimensionMismatch("Rlm length must be nlm=$(cfg.nlm)"))
    
    lmax = cfg.lmax; mres = cfg.mres
    
    # Apply the tridiagonal operator to each (l,m) mode
    @inbounds for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]  # Get (l,m) for this packed index
        
        # Extract coupling coefficients for this (l,m) mode
        c_minus = mx[2*lm0 + 1]  # Coefficient for coupling to Y_{l-1}^m
        c_plus  = mx[2*lm0 + 2]  # Coefficient for coupling to Y_{l+1}^m
        
        acc = 0.0 + 0.0im  # Accumulator for the result
        
        # Couple to lower degree neighbor Y_{l-1}^m
        if l > m && l > 0  # Check bounds: l-1 ≥ m and l-1 ≥ 0
            lm_prev = LM_index(lmax, mres, l-1, m)  # Get packed index for (l-1,m)
            acc += c_minus * Qlm[lm_prev + 1]       # Add contribution from lower neighbor
        end
        
        # Couple to higher degree neighbor Y_{l+1}^m  
        if l < lmax  # Check bounds: l+1 ≤ lmax
            lm_next = LM_index(lmax, mres, l+1, m)  # Get packed index for (l+1,m)
            acc += c_plus * Qlm[lm_next + 1]        # Add contribution from upper neighbor
        end
        
        Rlm[lm0 + 1] = acc  # Store result for this (l,m) mode
    end
    return Rlm
end

