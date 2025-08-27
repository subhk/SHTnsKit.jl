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

"""
    _ensure_logfac!(n::Int)

Internal function to extend the factorial cache up to n! if needed.
Uses the recurrence log(k!) = log((k-1)!) + log(k) for numerical stability.
"""
function _ensure_logfac!(n::Int)
    # Validate input
    n >= 0 || throw(DomainError(n, "logfactorial expects n ≥ 0"))
    
    # Check if cache needs extension
    cache = _logfac_cache[]
    kmax = length(cache) - 1  # Current maximum cached factorial
    
    if n > kmax
        # Extend cache incrementally using stable recurrence relation
        # log(k!) = log((k-1)!) + log(k) avoids overflow issues
        for k in (kmax + 1):n
            push!(cache, cache[end] + log(k))
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

