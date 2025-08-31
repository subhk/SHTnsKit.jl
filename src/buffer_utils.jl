"""
Buffer Allocation Utilities

This module provides common buffer allocation and manipulation patterns
used throughout the SHTnsKit codebase to eliminate code duplication
and provide consistent, optimized buffer management.
"""

"""
    allocate_spectral_pair(template1, template2) -> (buf1, buf2)

Allocate a pair of spectral coefficient buffers based on template arrays.
Commonly used for (Slm, Tlm) or (Qlm, Slm) pairs.
"""
function allocate_spectral_pair(template1::AbstractMatrix, template2::AbstractMatrix)
    return similar(template1), similar(template2)
end

"""
    allocate_spectral_triple(template1, template2, template3) -> (buf1, buf2, buf3)

Allocate three spectral coefficient buffers, commonly used for (Qlm, Slm, Tlm) triples.
"""
function allocate_spectral_triple(template1::AbstractMatrix, template2::AbstractMatrix, template3::AbstractMatrix)
    return similar(template1), similar(template2), similar(template3)
end

"""
    copy_spectral_pair(src1, src2) -> (copy1, copy2)

Create working copies of a pair of spectral coefficient arrays.
"""
function copy_spectral_pair(src1::AbstractMatrix, src2::AbstractMatrix)
    return copy(src1), copy(src2)
end

"""
    copy_spectral_triple(src1, src2, src3) -> (copy1, copy2, copy3)

Create working copies of three spectral coefficient arrays.
"""
function copy_spectral_triple(src1::AbstractMatrix, src2::AbstractMatrix, src3::AbstractMatrix)
    return copy(src1), copy(src2), copy(src3)
end

"""
    allocate_spatial_pair(template1, template2) -> (buf1, buf2)

Allocate a pair of spatial field buffers, commonly used for (Vt, Vp) components.
"""
function allocate_spatial_pair(template1::AbstractMatrix, template2::AbstractMatrix)
    return similar(template1), similar(template2)
end

"""
    allocate_spatial_triple(template1, template2, template3) -> (buf1, buf2, buf3)

Allocate three spatial field buffers for (Vr, Vt, Vp) components.
"""
function allocate_spatial_triple(template1::AbstractMatrix, template2::AbstractMatrix, template3::AbstractMatrix)
    return similar(template1), similar(template2), similar(template3)
end

"""
    zero_high_degree_modes!(arrays::Tuple, cfg::SHTConfig, ltr::Int)

Zero out high-degree modes (l > ltr) in multiple coefficient arrays simultaneously.
This is a common operation for degree-limited transforms.
"""
function zero_high_degree_modes!(arrays::Tuple, cfg::SHTConfig, ltr::Int)
    lmax, mmax = cfg.lmax, cfg.mmax
    
    for array in arrays
        @inbounds for m in 0:mmax, l in (ltr+1):lmax
            if l >= m  # Only valid (l,m) combinations
                array[l+1, m+1] = 0.0
            end
        end
    end
    
    return arrays
end

"""
    zero_high_degree_modes!(array::AbstractMatrix, cfg::SHTConfig, ltr::Int)

Zero out high-degree modes in a single coefficient array.
"""
function zero_high_degree_modes!(array::AbstractMatrix, cfg::SHTConfig, ltr::Int)
    return zero_high_degree_modes!((array,), cfg, ltr)[1]
end

"""
    create_zero_coefficients(cfg::SHTConfig, element_type::Type=ComplexF64) -> Matrix

Create a zero-filled coefficient matrix with standard (lmax+1, mmax+1) dimensions.
"""
function create_zero_coefficients(cfg::SHTConfig, element_type::Type=ComplexF64)
    lmax, mmax = cfg.lmax, cfg.mmax
    array = Matrix{element_type}(undef, lmax+1, mmax+1)
    fill!(array, zero(element_type))
    return array
end

"""
    validate_spectral_dimensions(array::AbstractMatrix, cfg::SHTConfig, name::String="array")

Validate that a spectral coefficient array has the correct dimensions.
Throws DimensionMismatch with descriptive message if invalid.
"""
function validate_spectral_dimensions(array::AbstractMatrix, cfg::SHTConfig, name::String="array")
    lmax, mmax = cfg.lmax, cfg.mmax
    expected_rows, expected_cols = lmax+1, mmax+1
    actual_rows, actual_cols = size(array)
    
    if actual_rows != expected_rows
        throw(DimensionMismatch("$name first dim must be lmax+1=$expected_rows, got $actual_rows"))
    end
    if actual_cols != expected_cols  
        throw(DimensionMismatch("$name second dim must be mmax+1=$expected_cols, got $actual_cols"))
    end
    
    return array
end

"""
    validate_spatial_dimensions(array::AbstractMatrix, cfg::SHTConfig, name::String="array")

Validate that a spatial field array has the correct grid dimensions.
"""
function validate_spatial_dimensions(array::AbstractMatrix, cfg::SHTConfig, name::String="array")
    nlat, nlon = cfg.nlat, cfg.nlon
    actual_rows, actual_cols = size(array)
    
    if actual_rows != nlat
        throw(DimensionMismatch("$name first dim must be nlat=$nlat, got $actual_rows"))
    end
    if actual_cols != nlon
        throw(DimensionMismatch("$name second dim must be nlon=$nlon, got $actual_cols"))
    end
    
    return array
end

"""
    validate_spectral_pair_dimensions(array1, array2, cfg::SHTConfig, names::Tuple{String,String}=("array1","array2"))

Validate dimensions for a pair of spectral arrays (commonly Slm, Tlm).
"""
function validate_spectral_pair_dimensions(array1::AbstractMatrix, array2::AbstractMatrix, cfg::SHTConfig, names::Tuple{String,String}=("array1","array2"))
    validate_spectral_dimensions(array1, cfg, names[1])
    validate_spectral_dimensions(array2, cfg, names[2])
    return array1, array2
end

"""
    validate_spatial_pair_dimensions(array1, array2, cfg::SHTConfig, names::Tuple{String,String}=("array1","array2"))

Validate dimensions for a pair of spatial arrays (commonly Vt, Vp).
"""
function validate_spatial_pair_dimensions(array1::AbstractMatrix, array2::AbstractMatrix, cfg::SHTConfig, names::Tuple{String,String}=("array1","array2"))
    validate_spatial_dimensions(array1, cfg, names[1])
    validate_spatial_dimensions(array2, cfg, names[2])
    return array1, array2
end

"""
    validate_qst_dimensions(Qlm, Slm, Tlm, cfg::SHTConfig)

Validate dimensions for QST coefficient triple with descriptive error messages.
"""
function validate_qst_dimensions(Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix, cfg::SHTConfig)
    validate_spectral_dimensions(Qlm, cfg, "Qlm")
    validate_spectral_dimensions(Slm, cfg, "Slm") 
    validate_spectral_dimensions(Tlm, cfg, "Tlm")
    return Qlm, Slm, Tlm
end

"""
    validate_vector_spatial_dimensions(Vr, Vt, Vp, cfg::SHTConfig)

Validate dimensions for spatial vector field triple (Vr, Vt, Vp).
"""
function validate_vector_spatial_dimensions(Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix, cfg::SHTConfig)
    validate_spatial_dimensions(Vr, cfg, "Vr")
    validate_spatial_dimensions(Vt, cfg, "Vt")
    validate_spatial_dimensions(Vp, cfg, "Vp")
    return Vr, Vt, Vp
end

"""
    thread_local_legendre_buffers(lmax::Int, nthreads::Int=Threads.nthreads()) -> Vector{Vector{Float64}}

Create thread-local Legendre polynomial buffers to avoid allocations and race conditions.
Returns vector of buffers, one per thread.
"""
function thread_local_legendre_buffers(lmax::Int, nthreads::Int=Threads.nthreads())
    return [Vector{Float64}(undef, lmax + 1) for _ in 1:nthreads]
end