module GPUCommon

using KernelAbstractions
using SHTnsKit

export laplacian_kernel!, legendre_table_kernel!, scalar_analysis_kernel!,
       scalar_synthesis_kernel!, coefficient_conversion_kernel!,
       scalar_config_signature, scalar_host_tables

"""
Fingerprint every host-owned configuration value consumed by a scalar GPU
transform. `SHTConfig` is mutable for compatibility, so object identity alone
is not a valid cache key: changing pole order, quadrature, or a convention must
select fresh device tables.
"""
function scalar_config_signature(cfg::SHTnsKit.SHTConfig)
    grid_hash = hash(Tuple(cfg.x), hash(Tuple(cfg.w)))
    return hash((
        objectid(cfg), cfg.lmax, cfg.mmax, cfg.mres, cfg.nlat, cfg.nlon,
        cfg.grid_type, cfg.cphi, cfg.south_pole_first,
        cfg.norm, cfg.real_norm, cfg.cs_phase, grid_hash,
    ))
end

"""Build the small typed host setup vectors copied once into a vendor cache."""
function scalar_host_tables(cfg::SHTnsKit.SHTConfig, ::Type{T}) where {T<:AbstractFloat}
    scales = Matrix{T}(undef, cfg.lmax + 1, cfg.mmax + 1)
    fill!(scales, one(T))
    for m in 0:cfg.mmax, l in m:cfg.lmax
        scales[l + 1, m + 1] = T(SHTnsKit.coefficient_scale_to_canonical(cfg, l, m))
    end
    return T.(cfg.x), T.(cfg.w), scales
end

"""Build orthonormal, Condon--Shortley associated Legendre values on device."""
@kernel function legendre_table_kernel!(Plm, x, lmax, mmax)
    i, m_idx = @index(Global, NTuple)
    if i <= length(x) && m_idx <= mmax + 1
        m = m_idx - 1
        xi = x[i]
        T = typeof(xi)
        sint = sqrt(max(zero(T), one(T) - xi * xi))
        pmm = inv(sqrt(T(4) * T(pi)))
        @inbounds for k in 1:m
            tk = T(k)
            pmm = -sqrt((T(2) * tk + one(T)) / (T(2) * tk)) * sint * pmm
        end
        Plm[i, m + 1, m_idx] = pmm

        if m < lmax
            pm1m = sqrt(T(2m + 3)) * xi * pmm
            Plm[i, m + 2, m_idx] = pm1m
            previous2 = pmm
            previous1 = pm1m
            @inbounds for l in (m + 2):lmax
                tl = T(l)
                tm = T(m)
                a = sqrt(((T(2) * tl - one(T)) * (T(2) * tl + one(T))) /
                         ((tl - tm) * (tl + tm)))
                b = sqrt(((T(2) * tl + one(T)) * (tl - one(T) - tm) *
                          (tl - one(T) + tm)) /
                         ((T(2) * tl - T(3)) * (tl - tm) * (tl + tm)))
                value = a * xi * previous1 - b * previous2
                Plm[i, l + 1, m_idx] = value
                previous2 = previous1
                previous1 = value
            end
        end
    end
end

"""Latitude integration after the vendor FFT; output is canonical."""
@kernel function scalar_analysis_kernel!(canonical, fourier, Plm, weights,
                                          cphi, lmax, mmax, mres)
    l_idx, m_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1
        l = l_idx - 1
        m = m_idx - 1
        if l >= m && m % mres == 0
            value = zero(eltype(canonical))
            @inbounds for i in 1:length(weights)
                value += weights[i] * Plm[i, l_idx, m_idx] * fourier[i, m_idx]
            end
            canonical[l_idx, m_idx] = cphi * value
        else
            canonical[l_idx, m_idx] = zero(eltype(canonical))
        end
    end
end

"""Convert dense coefficients entirely on device at the public boundary."""
@kernel function coefficient_conversion_kernel!(output, input, scales,
                                                  lmax, mmax, to_internal)
    l_idx, m_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1
        l = l_idx - 1
        m = m_idx - 1
        if l >= m
            scale = scales[l_idx, m_idx]
            output[l_idx, m_idx] = to_internal ?
                scale * input[l_idx, m_idx] : input[l_idx, m_idx] / scale
        else
            output[l_idx, m_idx] = zero(eltype(output))
        end
    end
end

"""Legendre synthesis into vendor-IFFT bins from canonical coefficients."""
@kernel function scalar_synthesis_kernel!(fourier, canonical, Plm, inv_scale,
                                           nlon, lmax, mmax, mres, real_output)
    i, m_idx = @index(Global, NTuple)
    if i <= size(fourier, 1) && m_idx <= mmax + 1
        m = m_idx - 1
        if m % mres == 0
            value = zero(eltype(fourier))
            @inbounds for l in m:lmax
                value += Plm[i, l + 1, m_idx] * canonical[l + 1, m_idx]
            end
            bin = inv_scale * value
            fourier[i, m_idx] = bin
            if real_output && m > 0
                negative_idx = nlon - m + 1
                if negative_idx != m_idx
                    fourier[i, negative_idx] = conj(bin)
                end
            end
        end
    end
end

# Device-neutral kernels live here. Vendor extensions own array placement,
# FFT libraries, synchronization, device selection, and runtime inspection.
@kernel function laplacian_kernel!(output, input, lmax, mmax)
    l, m = @index(Global, NTuple)
    if l <= lmax + 1 && m <= mmax + 1
        l_val = l - 1
        m_val = m - 1
        if l_val >= m_val
            output[l, m] = -l_val * (l_val + 1) * input[l, m]
        end
    end
end

end # module GPUCommon
