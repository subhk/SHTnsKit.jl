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

# =============================================================================
# NUMERICAL CONSTANTS
# =============================================================================

"""
Multiplier for machine epsilon to detect near-pole conditions.
When sin(θ) < POLE_TOLERANCE_FACTOR * eps(T), we're effectively at a pole
and need special handling to avoid division by zero or numerical instability.
"""
const POLE_TOLERANCE_FACTOR = 100

"""
Convergence criterion for Newton-Raphson iteration in Gauss-Legendre quadrature.
Root finding stops when |z - z_prev| < NEWTON_CONVERGENCE_TOL.
"""
const NEWTON_CONVERGENCE_TOL = 1e-15

# =============================================================================
# LEGENDRE POLYNOMIAL COMPUTATION
# =============================================================================

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

At poles (x = ±1), special handling is used to avoid division by zero:
- For m = 0: dP_l^0/dx|_{x=±1} = (±1)^{l+1} * l(l+1)/2 (known analytical result)
- For m > 0: derivatives are set to 0 (P_l^m(±1) = 0 for m > 0, and the vector
  transform code handles the 0/0 limit via Plm_and_dPdtheta_row!)

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

        # Handle poles (x = ±1) where x² - 1 = 0 causes division by zero
        if abs(x2m1) < POLE_TOLERANCE_FACTOR * eps(T)
            if m == 0
                # For m = 0: dP_l^0/dx|_{x=±1} = (±1)^{l+1} * l(l+1)/2
                # This is the well-known analytical result for Legendre polynomial derivatives at endpoints
                for l in 1:lmax
                    sign_factor = x > 0 ? T((-1)^(l+1)) : one(T)
                    dPdx[l+1] = sign_factor * T(l * (l + 1)) / 2
                end
            else
                # For m > 0: P_l^m(±1) = 0 (because of sin^m factor)
                # The derivative exists but requires L'Hôpital's rule
                # For vector transforms, use Plm_and_dPdtheta_row! which handles this properly
                # Here we set to 0 as a safe fallback
                fill!(dPdx, zero(T))
            end
            return P, dPdx
        end

        # Standard case: not at a pole
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
    Plm_and_dPdtheta_row!(P, dPdtheta, x, lmax, m)

Compute associated Legendre polynomials P_l^m(x) and their θ-derivatives dP_l^m/dθ.

This function computes dP/dθ directly (not dP/dx), which is singularity-free at poles.
The relationship is: dP/dθ = -sin(θ) * dP/dx = -√(1-x²) * dP/dx

For m > 0, this uses the recurrence relation in a form that avoids the pole singularity:
    dP_l^m/dθ = l*cos(θ)*P_l^m/sin(θ) - (l+m)*P_{l-1}^m/sin(θ)
              = l*x*P_l^m/√(1-x²) - (l+m)*P_{l-1}^m/√(1-x²)

At poles (θ=0 or π, x=±1):
- For m = 0: dP_l^0/dθ = 0 (by symmetry)
- For m = 1: dP_l^1/dθ has a finite nonzero limit
- For m > 1: dP_l^m/dθ = 0 (P_l^m vanishes faster than linearly)

This is the preferred function for vector spherical harmonic transforms.
"""
function Plm_and_dPdtheta_row!(P::AbstractVector{T}, dPdtheta::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}
    # First compute the Legendre polynomials
    Plm_row!(P, x, lmax, m)

    @inbounds begin
        # Initialize derivative array
        fill!(dPdtheta, zero(T))

        # Early return if no valid polynomials
        if lmax < m
            return P, dPdtheta
        end

        # Compute sin(θ) = √(1-x²)
        sinth = sqrt(max(zero(T), one(T) - x*x))

        # Handle poles (x = ±1) where sin(θ) = 0
        if sinth < POLE_TOLERANCE_FACTOR * eps(T)
            # At poles, dP/dθ has special values:
            # - For m = 0: dP_l^0/dθ = 0 (symmetry)
            # - For m = 1: dP_l^1/dθ|_{θ=0} = -√(l(l+1)/2) (with normalization)
            #              dP_l^1/dθ|_{θ=π} = (-1)^l * √(l(l+1)/2)
            # - For m > 1: dP_l^m/dθ = 0 (P_l^m ~ sin^m, so derivative ~ sin^{m-1} → 0)

            if m == 0
                # dP_l^0/dθ = 0 at poles
                fill!(dPdtheta, zero(T))
            elseif m == 1
                # For m = 1, use the analytical limit
                # P_l^1 = -sin(θ) * dP_l^0/dx (Condon-Shortley phase)
                # dP_l^1/dθ = -cos(θ)*dP_l^0/dx + sin²(θ)*d²P_l^0/dx²
                # At north pole (θ=0, x=1): dP_l^1/dθ = -1*P'_l(1) = -l(l+1)/2
                # At south pole (θ=π, x=-1): dP_l^1/dθ = +1*P'_l(-1) = (-1)^{l+1}*l(l+1)/2
                for l in 1:lmax
                    if x > 0  # North pole (θ = 0)
                        # dP_l^1/dθ|_{θ=0} = -l(l+1)/2 (always negative, no sign alternation)
                        dPdtheta[l+1] = -T(l * (l + 1)) / 2
                    else  # South pole (θ = π)
                        # dP_l^1/dθ|_{θ=π} = (-1)^{l+1} * l(l+1)/2
                        dPdtheta[l+1] = T((-1)^(l+1)) * T(l * (l + 1)) / 2
                    end
                end
            else
                # For m > 1, dP_l^m/dθ = 0 at poles
                fill!(dPdtheta, zero(T))
            end
            return P, dPdtheta
        end

        # Standard case: not at a pole
        # Use: dP/dθ = -sin(θ) * dP/dx = -sin(θ) * [l*x*P - (l+m)*P_{l-1}] / (x²-1)
        #            = -sin(θ) * [l*x*P - (l+m)*P_{l-1}] / (-sin²θ)
        #            = [l*x*P - (l+m)*P_{l-1}] / sin(θ)
        inv_sinth = one(T) / sinth

        # Handle l = m case
        l = m
        if l == 0
            dPdtheta[l+1] = zero(T)  # dP_0^0/dθ = 0
        else
            # dP_m^m/dθ = m*x*P_m^m/sin(θ) (since P_{m-1}^m = 0)
            dPdtheta[l+1] = m * x * P[l+1] * inv_sinth
        end

        # Compute derivatives for l ≥ m+1
        @simd ivdep for l in (m+1):lmax
            # dP_l^m/dθ = [l*x*P_l^m - (l+m)*P_{l-1}^m] / sin(θ)
            dPdtheta[l+1] = (l * x * P[l+1] - (l + m) * P[l]) * inv_sinth
        end
    end

    return P, dPdtheta
end

"""
    Plm_over_sinth_row!(P, P_over_sinth, x, lmax, m)

Compute P_l^m(x) and P_l^m(x)/sin(θ) with proper pole handling.

For m > 0, P_l^m(x)/sin(θ) has a finite limit at poles because P_l^m ~ sin^m(θ).
This function computes that limit correctly, avoiding the 0/0 indeterminate form.

At poles (x = ±1):
- For m = 0: Not applicable (m must be > 0 for this to be meaningful in vector transforms)
- For m = 1: P_l^1/sin(θ) → finite limit related to dP_l^0/dx
- For m > 1: P_l^m/sin(θ) → 0 (numerator has higher order zero)

This is essential for vector spherical harmonic transforms which require (im/sinθ)*Y terms.
"""
function Plm_over_sinth_row!(P::AbstractVector{T}, P_over_sinth::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}
    # First compute the Legendre polynomials
    Plm_row!(P, x, lmax, m)

    @inbounds begin
        # Initialize output array
        fill!(P_over_sinth, zero(T))

        # Early return if no valid polynomials
        if lmax < m
            return P, P_over_sinth
        end

        # Compute sin(θ) = √(1-x²)
        sinth = sqrt(max(zero(T), one(T) - x*x))

        # Handle poles (x = ±1) where sin(θ) = 0
        if sinth < POLE_TOLERANCE_FACTOR * eps(T)
            if m == 0
                # P_l^0/sin(θ) is genuinely singular for m=0, but this case
                # shouldn't be used in vector transforms (m=0 terms don't have 1/sinθ factor)
                fill!(P_over_sinth, zero(T))
            elseif m == 1
                # For m = 1: P_l^1 = -sin(θ) * dP_l^0/dx (Condon-Shortley)
                # So P_l^1/sin(θ) = -dP_l^0/dx
                # At x = 1 (north pole): dP_l^0/dx = l(l+1)/2, so P_l^1/sinθ = -l(l+1)/2
                # At x = -1 (south pole): dP_l^0/dx = (-1)^{l+1}*l(l+1)/2, so P_l^1/sinθ = (-1)^l*l(l+1)/2
                for l in 1:lmax
                    if x > 0  # North pole
                        P_over_sinth[l+1] = -T(l * (l + 1)) / 2
                    else  # South pole
                        P_over_sinth[l+1] = T((-1)^l) * T(l * (l + 1)) / 2
                    end
                end
            else
                # For m > 1: P_l^m ~ sin^m(θ), so P_l^m/sin(θ) ~ sin^{m-1}(θ) → 0
                fill!(P_over_sinth, zero(T))
            end
            return P, P_over_sinth
        end

        # Standard case: not at a pole
        inv_sinth = one(T) / sinth
        for l in m:lmax
            P_over_sinth[l+1] = P[l+1] * inv_sinth
        end
    end

    return P, P_over_sinth
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

#=
================================================================================
Gauss-Legendre Quadrature Nodes and Weights
================================================================================

This section computes the nodes (abscissas) and weights for Gauss-Legendre
numerical integration on the interval [-1, 1].

WHY GAUSS-LEGENDRE FOR SPHERICAL HARMONICS?
-------------------------------------------
Spherical harmonic transforms require integrating functions against Legendre
polynomials over the latitude (θ) direction. The standard substitution
x = cos(θ) transforms this to an integral over [-1, 1]:

    ∫₀^π f(θ) P_l(cos θ) sin θ dθ = ∫₋₁¹ f(arccos x) P_l(x) dx

Gauss-Legendre quadrature with n points is EXACT for polynomials up to
degree 2n-1. Since P_l(x) has degree l, using n = lmax+1 nodes ensures
exact integration of the Legendre polynomial component.

ALGORITHM
---------
1. Initial guess for roots using Abramowitz & Stegun formula (10.18.10):
   z₀ = cos(π(k - 0.25)/(n + 0.5))

2. Newton-Raphson iteration to refine root of P_n(z):
   z_{new} = z - P_n(z)/P'_n(z)

3. Weights computed from derivative at converged root:
   w_k = 2 / ((1 - z_k²) [P'_n(z_k)]²)

PROPERTIES
----------
- Nodes are symmetric: x_k = -x_{n-k+1}
- Weights are symmetric: w_k = w_{n-k+1}
- Weights sum to 2: Σ w_k = ∫₋₁¹ 1 dx = 2
- Nodes are roots of P_n(x)

DEBUGGING
---------
```julia
x, w = gausslegendre(16)
@assert length(x) == length(w) == 16
@assert abs(sum(w) - 2.0) < 1e-14     # Weights sum to 2
@assert all(x[i] ≈ -x[17-i] for i in 1:8)  # Symmetry
@assert all(w[i] ≈ w[17-i] for i in 1:8)   # Weight symmetry
```

================================================================================
=#

"""
    gausslegendre(n::Int)

Compute Gauss–Legendre nodes and weights for integrating functions on [-1, 1].
Returns `(x::Vector{Float64}, w::Vector{Float64})` with `length == n` where
`∫_{-1}^1 f(x) dx ≈ sum(w .* f.(x))`.
"""
function gausslegendre(n::Int)
    n > 0 || throw(ArgumentError("n must be positive"))
    x = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    # Number of roots computed (symmetry)
    m = (n + 1) >>> 1
    for k in 1:m
        # Initial guess (Abramowitz & Stegun 10.18.10)
        z = cos(pi * (k - 0.25) / (n + 0.5))
        z1 = 0.0
        # Newton iterations
        for _ in 1:50
            pnm1 = 1.0
            pn = z
            # Compute Legendre P_n(z) using recurrence
            for l in 2:n
                pnp1 = ((2l - 1) * z * pn - (l - 1) * pnm1) / l
                pnm1, pn = pn, pnp1
            end
            # Derivative using stable relation
            pd = n * (z * pn - pnm1) / (z^2 - 1.0)
            z1 = z
            z -= pn / pd
            if abs(z - z1) < NEWTON_CONVERGENCE_TOL
                break
            end
        end
        # Compute P_n and derivative at converged root for weights
        pnm1 = 1.0
        pn = z
        for l in 2:n
            pnp1 = ((2l - 1) * z * pn - (l - 1) * pnm1) / l
            pnm1, pn = pn, pnp1
        end
        pd = n * (z * pn - pnm1) / (z^2 - 1.0)

        x[k] = -z
        x[n - k + 1] = z
        wk = 2.0 / ((1.0 - z^2) * pd^2)
        w[k] = wk
        w[n - k + 1] = wk
    end
    return x, w
end

"""
    thetaphi_from_nodes(nlat::Int, nlon::Int)

Return `θ` and `φ` arrays where `θ ∈ [0, π]` (Gauss–Legendre nodes mapped) and
`φ ∈ [0, 2π)` equally spaced longitudes suitable for FFT-based azimuthal transforms.

Note: The returned arrays follow the gausslegendre ordering (south-to-north, x from -1 to +1).
For north-to-south ordering compatible with SHTns conventions, the caller should reverse
the arrays after calling this function (as done in api_compat.jl for shtns_set_grid).
"""
function thetaphi_from_nodes(nlat::Int, nlon::Int)
    x, w = gausslegendre(nlat)
    θ = acos.(x)
    φ = (2π / nlon) .* (0:(nlon-1))
    return θ, φ, x, w
end
