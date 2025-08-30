"""
FFT Utilities with Automatic Differentiation Support

This module provides FFT operations along the longitude dimension with automatic
fallback to pure Julia DFT when FFTW cannot handle certain element types.

The primary use case is enabling automatic differentiation through SHTnsKit
transforms. FFTW cannot handle ForwardDiff.Dual numbers or other AD types,
so we provide a pure Julia DFT fallback that works with any numeric type.

Performance Notes:
- FFTW path: O(N log N) - used for normal Float64/ComplexF64 operations
- DFT fallback: O(N²) - slower but works with arbitrary element types
- Fallback is essential for gradient computations using ForwardDiff, Zygote, etc.
"""

# Precompute 2π for efficiency in DFT calculations
const _TWO_PI = 2π

# Track which backend was used most recently for φ-FFTs: :fftw or :dft
const _FFT_BACKEND = Ref{Symbol}(:unknown)

fft_phi_backend() = _FFT_BACKEND[]

"""
    _dft_phi(A::AbstractMatrix, dir::Int)

Pure Julia discrete Fourier transform implementation along longitude (phi direction).

This function implements the standard DFT formula manually, without relying on
FFTW. While slower than optimized FFT libraries, it works with any numeric type
including automatic differentiation types like ForwardDiff.Dual.

Parameters:
- A: Input matrix [latitude × longitude] 
- dir: Direction flag (+1 for inverse DFT, -1 for forward DFT)

The DFT formula implemented is:
Y[k] = Σⱼ A[j] * exp(dir * 2πi * k * j / N)
"""
function _dft_phi(A::AbstractMatrix, dir::Int)
    nlat, nlon = size(A)
    Y = similar(complex.(A))  # Ensure output is complex-valued
    
    # Compute DFT for each latitude band independently with SIMD optimization
    @inbounds for i in 1:nlat
        # For each output frequency k
        for k in 0:(nlon-1)
            s = zero(eltype(Y))  # Accumulator for this frequency
            
            # Sum over all input points j - USE REDUCTION, NOT ivdep!
            # Multiple j iterations accumulate into same 's', so this is a reduction operation
            @simd for j in 0:(nlon-1)
                # DFT kernel: exp(dir * 2πi * k * j / N)
                s += A[i, j+1] * cis(dir * _TWO_PI * k * j / nlon)
            end
            
            Y[i, k+1] = s  # Store result (converting to 1-based indexing)
        end
    end
    
    return Y
end

"""
    fft_phi(A::AbstractMatrix)

Forward FFT along the longitude dimension with automatic differentiation support.

This function first attempts to use FFTW's optimized FFT. If that fails (e.g.,
due to unsupported element types in AD), it automatically falls back to the
pure Julia DFT implementation.

The longitude dimension corresponds to the azimuthal angle φ in spherical
coordinates, hence the function name.
"""
function fft_phi(A::AbstractMatrix)
    try
        # Primary path: use optimized FFTW along dimension 2 (longitude)
        local Y = fft(A, 2)
        _FFT_BACKEND[] = :fftw
        return Y
    catch
        # Check if we need DFT fallback for special number types (like ForwardDiff.Dual)
        T = eltype(A)
        needs_dft_fallback = !(T <: Union{Float32, Float64, ComplexF32, ComplexF64}) ||
                           contains(string(T), "ForwardDiff.Dual")
        
        if get(ENV, "SHTNSKIT_FORCE_FFTW", "0") == "1" && !needs_dft_fallback
            error("FFTW unavailable but SHTNSKIT_FORCE_FFTW=1; refusing DFT fallback")
        end
        # Fallback path: use pure Julia DFT for AD compatibility or when FFTW fails
        local Y = _dft_phi(A, -1)  # Forward transform uses -1 direction
        _FFT_BACKEND[] = :dft
        return Y
    end
end

"""
    ifft_phi(A::AbstractMatrix)

Inverse FFT along the longitude dimension with automatic differentiation support.

Like fft_phi, this function attempts FFTW first and falls back to pure Julia
DFT if needed. The inverse transform includes the proper normalization factor.

The scaling by 1/N is required to make the forward and inverse transforms
true inverses of each other.
"""
function ifft_phi(A::AbstractMatrix)
    nlon = size(A,2)  # Number of longitude points for normalization
    
    try
        # Primary path: use optimized FFTW inverse FFT
        local y = ifft(A, 2)
        _FFT_BACKEND[] = :fftw
        return y
    catch
        # Check if we need DFT fallback for special number types (like ForwardDiff.Dual)
        T = eltype(A)
        needs_dft_fallback = !(T <: Union{Float32, Float64, ComplexF32, ComplexF64}) ||
                           contains(string(T), "ForwardDiff.Dual")
        
        if get(ENV, "SHTNSKIT_FORCE_FFTW", "0") == "1" && !needs_dft_fallback
            error("FFTW unavailable but SHTNSKIT_FORCE_FFTW=1; refusing DFT fallback")
        end
        # Fallback path: use pure Julia inverse DFT with proper scaling  
        local y = (1/nlon) * _dft_phi(A, +1)  # Inverse transform uses +1 direction
        _FFT_BACKEND[] = :dft
        return y
    end
end
