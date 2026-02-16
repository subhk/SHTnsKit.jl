#=
================================================================================
layout.jl - SHTns-Compatible Mode Counting and Index Mapping
================================================================================

This file implements the packed coefficient storage layout used by SHTns.
Understanding this layout is essential for efficient memory access and
compatibility with other SHTns-based tools.

THE (l,m) INDEX PROBLEM
-----------------------
Spherical harmonic coefficients are naturally indexed by two integers:
- l: degree (0 ≤ l ≤ lmax)
- m: order (-l ≤ m ≤ l)

Storing these in a 2D array wastes memory since half the entries are
invalid (|m| > l). For lmax=100, only ~50% of a 101×201 array is used.

THE PACKED LAYOUT SOLUTION
--------------------------
SHTns uses a 1D packed layout that stores only valid (l,m) pairs.
For real-valued fields, Hermitian symmetry (a_{l,-m} = (-1)^m conj(a_{l,m}))
means we only need m ≥ 0 modes.

Layout scheme (for mres=1):
    Index 0: (0,0)
    Index 1: (1,0)
    ...
    Index lmax: (lmax,0)     <- End of m=0 block
    Index lmax+1: (1,1)      <- Start of m=1 block
    Index lmax+2: (2,1)
    ...

Modes are grouped by m, with l running from m to lmax within each block.

THE mres PARAMETER
------------------
mres controls which m-values are included:
- mres=1: all m (standard)
- mres=2: only even m (equatorial symmetry)
- mres=3: only m divisible by 3 (3-fold symmetry)

This is useful for flows with discrete rotational symmetry.

CORE FUNCTIONS
--------------
    nlm_calc(lmax, mmax, mres)     : Count total packed coefficients
    LM_index(lmax, mres, l, m)     : Convert (l,m) → packed index
    LiM_index(lmax, mres, l, im)   : Same but with reduced m-index
    build_li_mi(lmax, mmax, mres)  : Build reverse lookup tables

MEMORY EFFICIENCY
-----------------
For lmax=100, mmax=100, mres=1:
    2D array: 101 × 101 = 10,201 elements (many invalid)
    Packed:   nlm_calc = 5,151 elements (all valid)
    Savings:  ~50% memory reduction

COMPATIBILITY
-------------
These functions match the SHTns C library layout exactly, enabling:
- Direct data exchange with SHTns
- Same algorithms as reference implementation
- Validated test cases from SHTns

================================================================================
=#

"""
SHTns-compatible layout utilities: mode counting and index mapping.

These functions handle the conversion between 2D (l,m) spherical harmonic indices
and 1D packed array indices used by SHTns. This is crucial for efficient memory
layout and compatibility with the underlying C library.

All functions assume non-negative orders `m` that are multiples of `mres`.
"""

"""
    nlm_calc(lmax::Integer, mmax::Integer, mres::Integer) -> Int

Calculate the total number of packed spectral coefficients for a real scalar field.
Real fields only need m ≥ 0 modes due to Hermitian symmetry: Y_l^{-m} = (-1)^m (Y_l^m)*

The packing scheme groups modes by m-order, then by l-degree within each m block.
This layout is optimized for vectorized operations and cache efficiency.
"""
function nlm_calc(lmax::Integer, mmax::Integer, mres::Integer)
    # Validate input parameters for mathematical consistency
    lmax ≥ 0 && mmax ≥ 0 && mres ≥ 1 || throw(ArgumentError("invalid sizes"))
    
    # Early return if mmax exceeds lmax (no valid modes exist)
    mmax ≤ lmax || return 0
    
    # Count total number of (l,m) pairs where l ≥ m
    s = 0
    for m in 0:mres:mmax  # Step by mres to handle resolution constraints
        s += (lmax - m + 1)  # For fixed m, l runs from m to lmax
    end
    
    return s
end

"""
    nlm_cplx_calc(lmax::Integer, mmax::Integer, mres::Integer) -> Int

Calculate the total number of spectral coefficients for a complex field.
Complex fields require both positive and negative m modes, unlike real fields
which use Hermitian symmetry to store only m ≥ 0 modes.

This is used for complex-valued spherical harmonic expansions where no
symmetry assumptions can be made about the coefficients.
"""
function nlm_cplx_calc(lmax::Integer, mmax::Integer, mres::Integer)
    # Validate input parameters
    lmax ≥ 0 && mmax ≥ 0 && mres ≥ 1 || throw(ArgumentError("invalid sizes"))
    mres == 1 || throw(ArgumentError("nlm_cplx_calc only defined for mres == 1"))
    
    # Early return if no valid modes exist
    mmax ≤ lmax || return 0
    
    # Count modes for both positive and negative m values
    s = 0
    for m in -mmax:mres:mmax  # Include negative m values for complex fields
        s += (lmax - abs(m) + 1)  # For each |m|, l runs from |m| to lmax
    end
    
    return s
end

"""
    LM_index(lmax::Int, mres::Int, l::Int, m::Int) -> Int

Convert 2D spherical harmonic indices (l,m) to a 1D packed array index.
This implements the SHTns packing convention for efficient memory layout.

The packing scheme groups coefficients by m-order blocks, with each block
containing all valid l-degrees for that m. This enables vectorized operations
on modes with the same azimuthal symmetry.

Matches SHTns C macro: LM(shtns, l, m)
"""
function LM_index(lmax::Int, mres::Int, l::Int, m::Int)
    # Validate that m is non-negative (real field assumption)
    m ≥ 0 || throw(ArgumentError("m must be ≥ 0 for packed layout"))
    
    # Check that m is on the resolution grid
    (m % mres == 0) || throw(ArgumentError("m must be a multiple of mres"))
    
    # Ensure valid spherical harmonic indices
    (l ≥ m && l ≤ lmax) || throw(ArgumentError("require m ≤ l ≤ lmax"))
    
    # Convert m to reduced index
    im = m ÷ mres
    
    # Calculate base offset for this m-block using SHTns formula
    # This accounts for all previous m-blocks in the packed layout
    base = (im * (2*lmax + 2 - (im + 1)*mres)) >>> 1
    
    # Add l-offset within this m-block
    return base + l
end

"""
    LiM_index(lmax::Int, mres::Int, l::Int, im::Int) -> Int

Convenience function to compute packed index using reduced m-index im.
This is useful when iterating over m-modes using im = 0, 1, 2, ... 
instead of m = 0, mres, 2*mres, ...

Equivalent to LM_index(lmax, mres, l, im*mres).
Matches SHTns macro: LiM(shtns, l, im)
"""
function LiM_index(lmax::Int, mres::Int, l::Int, im::Int)
    # Validate reduced m-index
    (im ≥ 0) || throw(ArgumentError("im must be ≥ 0"))
    
    # Convert to actual m value
    m = im * mres
    
    # Check spherical harmonic constraint
    (l ≥ m && l ≤ lmax) || throw(ArgumentError("require im*mres ≤ l ≤ lmax"))
    
    # Use same calculation as LM_index
    base = (im * (2*lmax + 2 - (im + 1)*mres)) >>> 1
    return base + l
end

"""
    build_li_mi(lmax::Int, mmax::Int, mres::Int)

Build lookup arrays mapping packed indices to spherical harmonic (l,m) values.
These arrays enable efficient iteration over spherical harmonic modes when
working with packed coefficient arrays.

Returns (li, mi) where li[k] and mi[k] give the degree and order for 
the k-th packed coefficient.
"""
function build_li_mi(lmax::Int, mmax::Int, mres::Int)
    # Get total number of packed coefficients
    n = nlm_calc(lmax, mmax, mres)
    
    # Allocate lookup arrays
    li = Vector{Int}(undef, n)  # Degree (l) for each packed index
    mi = Vector{Int}(undef, n)  # Order (m) for each packed index
    
    # Fill arrays in packed order (by m-blocks, then by l within each block)
    k = 1
    for m in 0:mres:mmax          # Loop over m-orders on resolution grid
        for l in m:lmax           # For each m, loop over valid l-degrees
            li[k] = l             # Store degree
            mi[k] = m             # Store order
            k += 1                # Move to next packed index
        end
    end
    
    return li, mi
end

"""
    im_from_lm(lm::Int, lmax::Int, mres::Int) -> Int

Inverse operation: determine the reduced m-index (im = m/mres) from a packed index.
This function searches through the m-blocks to find which block contains
the given packed index.

Useful for algorithms that need to determine the azimuthal symmetry
of a coefficient from its packed storage location.
"""
function im_from_lm(lm::Int, lmax::Int, mres::Int)
    # Validate packed index
    lm ≥ 0 || throw(ArgumentError("lm must be ≥ 0"))
    
    # Search through m-blocks to find the one containing this packed index
    im = 0      # Current reduced m-index being tested
    base = 0    # Base offset for current m-block
    
    while true
        # Size of current m-block (number of l-modes for this m)
        block = lmax - im*mres + 1
        
        # Check if lm falls within current m-block
        if lm < base + block
            return im    # Found the correct m-block
        end
        
        # Move to next m-block
        base += block
        im += 1
    end
end

