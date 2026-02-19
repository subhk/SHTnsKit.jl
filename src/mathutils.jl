"""
Lightweight mathematical utilities for SHTnsKit.

This module provides efficient, cached implementations of mathematical functions
needed for spherical harmonic computations. By keeping these functions internal,
we avoid dependencies on heavy mathematical packages for basic operations.

The primary use cases are:
- Stable computation of normalization constants for spherical harmonics  
- Wigner d-matrix calculations for rotations
- General combinatorial calculations involving factorials

All functions use cached computation for efficiency in repeated evaluations.
"""

# ===== FACTORIAL CACHE IMPLEMENTATION =====
# Cache stores log(k!) values for k = 0, 1, 2, ... to avoid repeated computation
# Index: cache[k+1] = log(k!), so cache[1] = log(0!) = 0
const _logfac_cache = Ref(Vector{Float64}([0.0]))
const _logfac_lock = ReentrantLock()

"""
    _ensure_logfac!(n::Int)

Internal function to extend the factorial cache up to n! if needed.
Uses the recurrence log(k!) = log((k-1)!) + log(k) for numerical stability.
Thread-safe: uses a lock to protect concurrent cache extension.
"""
function _ensure_logfac!(n::Int)
    # Validate input
    n >= 0 || throw(DomainError(n, "logfactorial expects n ≥ 0"))

    # Fast path: no lock needed if cache is already large enough
    cache = _logfac_cache[]
    length(cache) - 1 >= n && return nothing

    # Slow path: extend cache under lock
    lock(_logfac_lock) do
        cache = _logfac_cache[]
        kmax = length(cache) - 1  # Current maximum cached factorial
        if n > kmax
            # Extend cache incrementally using stable recurrence relation
            # log(k!) = log((k-1)!) + log(k) avoids overflow issues
            for k in (kmax + 1):n
                push!(cache, cache[end] + log(k))
            end
        end
    end

    return nothing
end

"""
    logfactorial(n::Integer) -> Float64

Compute log(n!) using cached values for efficiency and numerical stability.

This function is critical for spherical harmonic normalization calculations,
where factorial ratios appear frequently. By working in log-space and caching
results, we avoid both overflow issues and redundant computations.

The implementation uses exact summation: log(n!) = Σ(k=1 to n) log(k)
which is more accurate than Stirling's approximation for moderate n.
"""
function logfactorial(n::Integer)
    # Convert to Int for consistency
    ni = Int(n)
    
    # Validate input domain  
    ni >= 0 || throw(DomainError(n, "logfactorial expects n ≥ 0"))
    
    # Ensure cache contains the needed value
    _ensure_logfac!(ni)
    
    # Return cached result (1-based indexing: cache[n+1] = log(n!))
    return _logfac_cache[][ni + 1]
end

"""
    loggamma(n::Integer) -> Float64

Compute log(Γ(n)) for positive integers using the gamma-factorial identity.

For positive integers, the gamma function satisfies Γ(n) = (n-1)!, so we can
reuse our cached factorial implementation. This is needed for various spherical
harmonic calculations involving beta functions and normalization constants.
"""
function loggamma(n::Integer)
    # Convert to Int for consistency
    ni = Int(n)
    
    # Validate input (gamma function requires positive arguments for integers)
    ni >= 1 || throw(DomainError(n, "loggamma expects n ≥ 1 for Integer inputs"))
    
    # Use identity: Γ(n) = (n-1)! for positive integers
    return logfactorial(ni - 1)
end

"""
    loggamma(x::Real)

Fallback for non-integer real arguments.

This implementation deliberately throws an error to prevent accidental use
of an inadequate approximation. For general real-valued log-gamma function,
users should add SpecialFunctions.jl as a dependency.

This design keeps SHTnsKit lightweight while ensuring numerical accuracy
for the specific integer cases we need.
"""
function loggamma(x::Real)
    # Handle integer-valued reals by delegation  
    isinteger(x) && return loggamma(Int(round(x)))
    
    # Reject non-integer reals with helpful error message
    throw(ArgumentError("loggamma(::Real) is only defined for integers here; add SpecialFunctions for general inputs"))
end

"""
    driscoll_healy_weights(n::Int) -> Vector{Float64}

Compute Driscoll-Healy quadrature weights for n latitude points.

The Driscoll-Healy quadrature provides an exact integration rule for spherical
harmonics up to degree n/2-1 when using n equally-spaced latitude samples.
The weights approximate: ∫₀^π f(θ) sin(θ) dθ ≈ Σⱼ w[j] f(θ[j])

Formula (normalized for ∫₀^π sin(θ)dθ = 2):
    w[j] = (4/n) * sin(πj/n) * Σ(l=0 to n/2-1) [sin((2l+1)πj/n) / (2l+1)]

where j = 0, 1, ..., n-1 indexes the latitude points θ[j] = πj/n.

Reference:
    Driscoll, J.R. and D.M. Healy, "Computing Fourier transforms and
    convolutions on the 2-sphere", Adv. Appl. Math., 15, 202-250, 1994.

Note:
    - n must be even for the DH quadrature to be exact
    - The weights sum to 2 (matching ∫₀^π sin(θ)dθ = 2)
    - The first weight (north pole, j=0) is always zero
    - The last weight is also near zero
    - Grid θ[j] = πj/n includes north pole but NOT south pole
"""
function driscoll_healy_weights(n::Int; apply_4pi_normalization::Bool=false)
    # Validate input
    n >= 2 || throw(ArgumentError("n must be ≥ 2"))
    iseven(n) || throw(ArgumentError("n must be even for Driscoll-Healy quadrature"))

    # Allocate output array
    w = zeros(Float64, n)

    # Normalization factor from DHaj formula in SHTOOLS
    # The raw DHaj formula: w[j] = (√8/n) * sin(πj/n) * Σ[sin((2l+1)πj/n) / (2l+1)]
    # For proper integration of ∫₀^π f(θ) sin(θ) dθ ≈ Σ w[j] f(θ[j])
    # we need weights to sum to 2 (since ∫₀^π sin(θ) dθ = 2)
    # The raw formula gives sum ≈ √2, so we multiply by √2 to get sum = 2
    norm_factor = sqrt(8.0) / n * sqrt(2.0)  # = 4/n

    # Compute weights for each latitude point
    for j in 0:(n-1)
        # Compute the inner sum: Σ(l=0 to n/2-1) [sin((2l+1)πj/n) / (2l+1)]
        sum1 = 0.0
        for l in 0:(n÷2-1)
            sum1 += sin((2*l + 1) * π * j / n) / (2*l + 1)
        end

        # Apply the full formula
        w[j+1] = norm_factor * sin(π * j / n) * sum1
    end

    # SHTOOLS applies an additional √(4π) normalization for spherical harmonic conventions
    # This may be needed depending on the spherical harmonic normalization used
    if apply_4pi_normalization
        w .*= sqrt(4.0 * π)
    end

    return w
end

