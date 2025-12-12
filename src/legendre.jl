#=
================================================================================
legendre.jl - Associated Legendre Polynomial Computation
================================================================================

This file implements the core Legendre polynomial computations needed for
spherical harmonic transforms. These polynomials are the latitude (θ) basis
functions in the expansion of functions on the sphere.

MATHEMATICAL BACKGROUND
-----------------------
Associated Legendre polynomials P_l^m(x) are solutions to the Legendre
differential equation:

    (1-x²) d²P/dx² - 2x dP/dx + [l(l+1) - m²/(1-x²)] P = 0

where x = cos(θ), l is the degree, and m is the order (|m| ≤ l).

For spherical harmonics:
    Y_l^m(θ,φ) = N_lm * P_l^m(cos θ) * exp(imφ)

where N_lm is a normalization factor.

IMPLEMENTATION NOTES
--------------------
1. Recurrence Relations:
   - P_l^m values are computed using three-term recurrence
   - This is numerically stable and efficient: O(lmax) per m value
   - The recurrence CANNOT be vectorized due to data dependencies

2. Condon-Shortley Phase:
   - We include the (-1)^m factor in P_m^m (physics convention)
   - This affects the sign of odd-m polynomials

3. Numerical Stability:
   - Normalization factors use log-space arithmetic to avoid overflow
   - (1-x²) is clamped to avoid sqrt of negative due to roundoff

4. Indexing Convention:
   - P[l+1] stores P_l^m (1-based Julia indexing)
   - Valid for l = m, m+1, ..., lmax

PERFORMANCE CONSIDERATIONS
--------------------------
- Plm_row!: Zero allocations, modifies P in place
- Cannot use SIMD for main recurrence (iteration dependency)
- Derivative computation CAN use SIMD (no iteration dependency)

DEBUGGING
---------
```julia
# Test P_0^0(x) = 1 for all x
P = zeros(10)
Plm_row!(P, 0.5, 9, 0)
@assert P[1] ≈ 1.0

# Test P_1^0(x) = x
@assert P[2] ≈ 0.5

# Test P_1^1(x) = -sqrt(1-x²) with CS phase
Plm_row!(P, 0.5, 9, 1)
@assert P[2] ≈ -sqrt(1 - 0.5^2)  # ≈ -0.866
```

================================================================================
=#

"""
    Plm_row!(P::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}

Compute associated Legendre polynomials P_l^m(x) for all degrees l = 0..lmax at fixed order m.

This function implements the stable three-term recurrence relations for associated 
Legendre polynomials. The algorithm follows the Ferrers definition with the 
Condon-Shortley phase factor (-1)^m included, which is standard in physics.

The input x = cos(θ) where θ is the colatitude angle.
Results are stored as P[l+1] = P_l^m(x) for l = 0..lmax (1-based indexing).
"""
function Plm_row!(P::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}
    # Initialize output array to zero
    @inbounds fill!(P, zero(T))
    
    # Validate order parameter
    m < 0 && throw(ArgumentError("m must be ≥ 0"))
    
    # Early return if no valid polynomials exist
    lmax >= m || return P

    # ===== SPECIAL CASE: m = 0 (ordinary Legendre polynomials) =====
    if m == 0
        # Base cases for ordinary Legendre polynomials
        P[1] = one(T)                 # P_0^0(x) = 1
        if lmax >= 1
            P[2] = x                  # P_1^0(x) = x
        end
        
        # Three-term recurrence for P_l^0(x) - CANNOT vectorize due to dependencies!
        # P[l+1] depends on P[l] and P[l-1], so each iteration depends on previous ones
        for l in 2:lmax
            # Bonnet's recurrence: (l+1)P_{l+1} = (2l+1)x P_l - l P_{l-1}
            # Rearranged: P_l^0(x) = ((2l-1)x P_{l-1}^0 - (l-1) P_{l-2}^0)/l
            P[l+1] = ((2l - 1) * x * P[l] - (l - 1) * P[l-1]) / l
        end
        return P
    end

    # ===== GENERAL CASE: m > 0 (associated Legendre polynomials) =====
    
    # Start with P_m^m(x) using explicit formula
    # P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^{m/2}
    pmm = one(T)
    sx2 = max(zero(T), 1 - x*x)  # (1-x²), guarded against roundoff for |x|≈1
    fact = one(T)                  # Tracks (2k-1) in double factorial
    
    # CANNOT vectorize: pmm depends on previous iteration, fact is updated each iteration
    for k in 1:m
        pmm *= -fact * sqrt(sx2)   # Build up (-1)^m (2m-1)!! (1-x²)^{m/2}
        fact += 2                  # Next odd number: 1, 3, 5, ...
    end
    P[m+1] = pmm

    # If lmax = m, we're done
    if lmax == m
        return P
    end

    # Compute P_{m+1}^m(x) using explicit formula
    # P_{m+1}^m(x) = x (2m+1) P_m^m(x)
    P[m+2] = x * (2m + 1) * pmm

    # Three-term recurrence for remaining degrees l ≥ m+2 - CANNOT vectorize!
    # P[l+1] depends on P[l] and P[l-1], so each iteration depends on previous ones
    for l in (m+2):lmax
        # Recurrence relation for associated Legendre polynomials:
        # P_l^m(x) = ((2l-1)x P_{l-1}^m - (l+m-1) P_{l-2}^m)/(l-m)
        P[l+1] = ((2l - 1) * x * P[l] - (l + m - 1) * P[l-1]) / (l - m)
    end
    
    return P
end

"""
    Plm_and_dPdx_row!(P, dPdx, x, lmax, m)

Simultaneously compute associated Legendre polynomials P_l^m(x) and their derivatives.

This function efficiently computes both P_l^m(x) and dP_l^m/dx for all degrees
l = m..lmax at fixed order m. The derivatives are computed with respect to the
argument x = cos(θ), not with respect to the angle θ itself.

The derivative calculation uses the standard recurrence relation:
dP_l^m/dx = [l*x*P_l^m - (l+m)*P_{l-1}^m] / (x²-1)

This is essential for computing gradients and differential operators on the sphere.
"""
function Plm_and_dPdx_row!(P::AbstractVector{T}, dPdx::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}
    # First compute the Legendre polynomials
    Plm_row!(P, x, lmax, m)
    
    @inbounds begin
        # Initialize derivative array
        fill!(dPdx, zero(T))
        
        # Early return if no valid polynomials
        if lmax < m
            return P, dPdx
        end
        
        # Precompute common factor (x² - 1) for derivative formula
        x2m1 = x*x - one(T)
        
        # Handle l = m case (base case for derivatives)
        l = m
        dPdx[l+1] = (l == 0) ? zero(T) : (m * x * P[l+1]) / x2m1
        
        # Compute derivatives for l ≥ m+1 using recurrence relation - SAFE to vectorize!
        # Each dPdx[l+1] depends only on already-computed P[l+1] and P[l], no iteration dependencies
        @simd ivdep for l in (m+1):lmax
            # Standard derivative recurrence: 
            # dP_l^m/dx = [l*x*P_l^m - (l+m)*P_{l-1}^m] / (x²-1)
            dPdx[l+1] = (l * x * P[l+1] - (l + m) * P[l]) / x2m1
        end
    end
    
    return P, dPdx
end

"""
    Nlm_table(lmax::Int, mmax::Int)

Precompute normalization factors for orthonormal spherical harmonics.

The normalization ensures that the spherical harmonics form an orthonormal basis:
∫ Y_l^m(θ,φ) [Y_{l'}^{m'}(θ,φ)]* dΩ = δ_{ll'} δ_{mm'}

The normalization factor is:
N_{l,m} = sqrt[(2l+1)/(4π) * (l-m)!/(l+m)!]

This function computes all factors for 0≤m≤mmax, m≤l≤lmax using stable
logarithmic arithmetic to avoid factorial overflow.

Returns matrix N[l+1,m+1] with 1-based indexing.
"""
function Nlm_table(lmax::Int, mmax::Int)
    # Allocate normalization table
    N = Matrix{Float64}(undef, lmax + 1, mmax + 1)
    
    for m in 0:mmax
        for l in 0:lmax
            if l < m
                # No spherical harmonic exists for l < m
                N[l+1, m+1] = 0.0
            else
                # Compute normalization factor in log space for numerical stability
                # log(N_{l,m}) = 0.5 * [log(2l+1) - log(4π) + log(Γ(l-m+1)) - log(Γ(l+m+1))]
                # Using Γ(n) = (n-1)! for integer n
                lr = 0.5 * (log(2l + 1.0) - log(4π)) + 0.5 * (loggamma(l - m + 1) - loggamma(l + m + 1))
                
                # Convert back from log space
                N[l+1, m+1] = exp(lr)
            end
        end
    end
    
    return N
end
