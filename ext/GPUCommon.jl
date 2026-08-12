module GPUCommon

using KernelAbstractions
using SHTnsKit

export laplacian_kernel!, legendre_table_kernel!, scalar_analysis_kernel!,
       scalar_synthesis_kernel!, coefficient_conversion_kernel!,
       coefficient_batch_conversion_kernel!,
       real_pack_kernel!, real_unpack_kernel!,
       mode_analysis_kernel!, mode_synthesis_kernel!,
       scalar_batch_analysis_kernel!, scalar_batch_synthesis_kernel!,
       complex_packed_analysis_kernel!, complex_packed_synthesis_kernel!,
       vector_derivative_table_kernel!, vector_analysis_kernel!,
       vector_synthesis_kernel!, vector_diagonal_kernel!,
       vector_mode_analysis_kernel!, vector_mode_synthesis_kernel!,
       vector_batch_analysis_kernel!, vector_batch_synthesis_kernel!,
       vector_host_tables,
       scalar_config_signature, vector_config_signature, scalar_host_tables,
       ScalarTableCache, scalar_cache_lookup, scalar_cache_insert!,
       scalar_cache_publish!,
       scalar_cache_clear!, scalar_cache_size,
       ScalarWorkspaceCache, scalar_workspace_use!,
       scalar_workspace_clear!, scalar_workspace_size

"""One cached table set plus its mutable-configuration signature and LRU tick."""
struct ScalarTableCacheEntry
    signature::UInt
    tick::UInt64
    value::Any
end

"""Batched form of coefficient conversion without broadcast temporaries."""
@kernel function coefficient_batch_conversion_kernel!(output, input, scales,
                                                        lmax, mmax, to_internal)
    l_idx, m_idx, batch_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1 &&
       batch_idx <= size(output, 3)
        l = l_idx - 1
        m = m_idx - 1
        if l >= m
            scale = scales[l_idx, m_idx]
            output[l_idx, m_idx, batch_idx] = to_internal ?
                scale * input[l_idx, m_idx, batch_idx] :
                input[l_idx, m_idx, batch_idx] / scale
        else
            output[l_idx, m_idx, batch_idx] = zero(eltype(output))
        end
    end
end

"""
Thread-safe, per-device bounded cache for immutable scalar transform tables.

The dictionary key deliberately uses configuration identity rather than its
mutable signature. A convention/grid mutation therefore replaces the stale
entry instead of accumulating another device allocation. Values are built
outside this cache's lock by the vendor extension.
"""
mutable struct ScalarTableCache
    entries::Dict{Tuple{Any,UInt,DataType},ScalarTableCacheEntry}
    tick::UInt64
    max_per_device::Int
    lock::ReentrantLock
end

function ScalarTableCache(max_per_device::Integer=8)
    max_per_device > 0 || throw(ArgumentError("max_per_device must be positive"))
    return ScalarTableCache(
        Dict{Tuple{Any,UInt,DataType},ScalarTableCacheEntry}(),
        0, Int(max_per_device), ReentrantLock(),
    )
end

function scalar_cache_lookup(cache::ScalarTableCache, device, identity::UInt,
                             precision::DataType, signature::UInt)
    key = (device, identity, precision)
    return lock(cache.lock) do
        entry = get(cache.entries, key, nothing)
        entry === nothing && return nothing
        if entry.signature != signature
            delete!(cache.entries, key)
            return nothing
        end
        cache.tick += 1
        cache.entries[key] = ScalarTableCacheEntry(signature, cache.tick, entry.value)
        return entry.value
    end
end

function scalar_cache_insert!(cache::ScalarTableCache, device, identity::UInt,
                              precision::DataType, signature::UInt, value)
    key = (device, identity, precision)
    return lock(cache.lock) do
        existing = get(cache.entries, key, nothing)
        if existing !== nothing && existing.signature == signature
            cache.tick += 1
            cache.entries[key] = ScalarTableCacheEntry(
                signature, cache.tick, existing.value,
            )
            return existing.value
        end

        # Replacement for the same config identity does not consume capacity.
        existing === nothing || delete!(cache.entries, key)
        device_keys = Tuple{Any,UInt,DataType}[
            candidate for candidate in keys(cache.entries) if candidate[1] == device
        ]
        if length(device_keys) >= cache.max_per_device
            oldest = argmin(candidate -> cache.entries[candidate].tick, device_keys)
            delete!(cache.entries, oldest)
        end
        cache.tick += 1
        cache.entries[key] = ScalarTableCacheEntry(signature, cache.tick, value)
        return value
    end
end

"""
    scalar_cache_publish!(complete, cache, device, identity, precision, signature, value)

Wait for an asynchronous immutable-table build to complete before making the
value visible to cache readers. `complete` deliberately runs before
`scalar_cache_insert!`, outside the cache lock. The insertion retains the
cache's double-checked behavior when concurrent builders race for one key.
"""
function scalar_cache_publish!(complete, cache::ScalarTableCache, device,
                               identity::UInt, precision::DataType,
                               signature::UInt, value)
    complete()
    return scalar_cache_insert!(
        cache, device, identity, precision, signature, value,
    )
end

function scalar_cache_clear!(cache::ScalarTableCache; device=nothing)
    lock(cache.lock) do
        if device === nothing
            empty!(cache.entries)
        else
            for key in collect(keys(cache.entries))
                key[1] == device && delete!(cache.entries, key)
            end
        end
    end
    return nothing
end

function scalar_cache_size(cache::ScalarTableCache; device=nothing)
    return lock(cache.lock) do
        device === nothing && return length(cache.entries)
        return count(key -> key[1] == device, keys(cache.entries))
    end
end

"""One weakly-owned, independently locked GPU transform workspace."""
mutable struct ScalarWorkspaceCacheEntry
    owner::WeakRef
    signature::UInt
    tick::UInt64
    value::Any
    lock::ReentrantLock
end

"""
Bounded device workspace cache used by in-place scalar and batch transforms.

Keys contain only `objectid(owner)`, never the owner itself. The accompanying
`WeakRef` prevents a cached vendor buffer or FFT plan from keeping an SHTPlan
or SHTConfig alive. Each entry has its own lock, so unrelated plans may execute
concurrently while reuse of the same mutable FFT workspace is serialized.
"""
mutable struct ScalarWorkspaceCache
    entries::Dict{Tuple{Any,UInt,DataType,Symbol,Tuple},ScalarWorkspaceCacheEntry}
    tick::UInt64
    max_per_device::Int
    lock::ReentrantLock
end

function ScalarWorkspaceCache(max_per_device::Integer=8)
    max_per_device > 0 || throw(ArgumentError("max_per_device must be positive"))
    return ScalarWorkspaceCache(
        Dict{Tuple{Any,UInt,DataType,Symbol,Tuple},ScalarWorkspaceCacheEntry}(),
        0, Int(max_per_device), ReentrantLock(),
    )
end

function _workspace_entry!(builder, cache::ScalarWorkspaceCache, device,
                           owner, precision::DataType, kind::Symbol,
                           shape::Tuple, signature::UInt)
    key = (device, objectid(owner), precision, kind, shape)
    return lock(cache.lock) do
        # Drop dead weak owners eagerly; this also protects against object-id
        # reuse selecting buffers belonging to a reclaimed plan/config.
        for candidate in collect(keys(cache.entries))
            cache.entries[candidate].owner.value === nothing &&
                delete!(cache.entries, candidate)
        end
        entry = get(cache.entries, key, nothing)
        if entry !== nothing && entry.owner.value === owner &&
           entry.signature == signature
            cache.tick += 1
            entry.tick = cache.tick
            return entry
        end
        entry === nothing || delete!(cache.entries, key)
        device_keys = [
            candidate for candidate in keys(cache.entries)
            if candidate[1] == device
        ]
        if length(device_keys) >= cache.max_per_device
            oldest = argmin(candidate -> cache.entries[candidate].tick, device_keys)
            delete!(cache.entries, oldest)
        end
        value = builder()
        cache.tick += 1
        built = ScalarWorkspaceCacheEntry(
            WeakRef(owner), signature, cache.tick, value, ReentrantLock(),
        )
        cache.entries[key] = built
        return built
    end
end

"""Run `f(workspace)` while holding the selected workspace's per-entry lock."""
function scalar_workspace_use!(f, builder, cache::ScalarWorkspaceCache,
                               device, owner, precision::DataType,
                               kind::Symbol, shape::Tuple, signature::UInt)
    entry = _workspace_entry!(
        builder, cache, device, owner, precision, kind, shape, signature,
    )
    return lock(entry.lock) do
        f(entry.value)
    end
end

function scalar_workspace_clear!(cache::ScalarWorkspaceCache; device=nothing)
    lock(cache.lock) do
        if device === nothing
            empty!(cache.entries)
        else
            for key in collect(keys(cache.entries))
                key[1] == device && delete!(cache.entries, key)
            end
        end
    end
    return nothing
end

function scalar_workspace_size(cache::ScalarWorkspaceCache; device=nothing)
    return lock(cache.lock) do
        device === nothing && return length(cache.entries)
        return count(key -> key[1] == device, keys(cache.entries))
    end
end

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

"""
Fingerprint every mutable input consumed only by vector derivative tables.

Scalar transforms do not depend on `Nlm`; keeping this separate prevents an
`Nlm` mutation from rebuilding scalar tables while still invalidating the
pole-sensitive vector cache entry for the same configuration identity.
"""
function vector_config_signature(cfg::SHTnsKit.SHTConfig)
    return hash(Tuple(cfg.Nlm), scalar_config_signature(cfg))
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

"""Build the typed host setup copied into a vendor's vector-table cache."""
function vector_host_tables(cfg::SHTnsKit.SHTConfig,
                            ::Type{T}) where {T<:AbstractFloat}
    x, weights, scales = scalar_host_tables(cfg, T)
    return x, weights, scales, T.(cfg.Nlm)
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

"""
Build orthonormal P, dP/dtheta, and P/sin(theta) tables in backend precision.
The exact-pole branch evaluates the finite m=1 limits analytically; no kernel
ever forms a singular quotient and masks it afterwards.
"""
@kernel function vector_derivative_table_kernel!(Plm, dtheta, over_sin,
                                                  x, Nlm, lmax, mmax)
    i, m_idx = @index(Global, NTuple)
    if i <= length(x) && m_idx <= mmax + 1
        m = m_idx - 1
        xi = x[i]
        T = typeof(xi)
        s = sqrt(max(zero(T), one(T) - xi * xi))
        pmm = inv(sqrt(T(4) * T(pi)))
        @inbounds for k in 1:m
            tk = T(k)
            pmm = -sqrt((T(2) * tk + one(T)) / (T(2) * tk)) * s * pmm
        end
        Plm[i, m + 1, m_idx] = pmm
        if m < lmax
            pm1m = sqrt(T(2m + 3)) * xi * pmm
            Plm[i, m + 2, m_idx] = pm1m
            previous2 = pmm
            previous1 = pm1m
            @inbounds for l in (m + 2):lmax
                tl = T(l); tm = T(m)
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

        @inbounds for l in 0:lmax
            if l < m
                Plm[i, l + 1, m_idx] = zero(T)
                dtheta[i, l + 1, m_idx] = zero(T)
                over_sin[i, l + 1, m_idx] = zero(T)
            elseif iszero(s)
                if m == 1
                    half_ll1 = T(l * (l + 1)) / T(2)
                    N = Nlm[l + 1, m_idx]
                    north = xi > zero(T)
                    dsign = north ? -one(T) : (isodd(l) ? one(T) : -one(T))
                    psign = north ? -one(T) : (iseven(l) ? one(T) : -one(T))
                    dtheta[i, l + 1, m_idx] = dsign * N * half_ll1
                    over_sin[i, l + 1, m_idx] = psign * N * half_ll1
                else
                    dtheta[i, l + 1, m_idx] = zero(T)
                    over_sin[i, l + 1, m_idx] = zero(T)
                end
            else
                current = Plm[i, l + 1, m_idx]
                previous = l == m ? zero(T) : Plm[i, l, m_idx]
                beta = l == m ? zero(T) :
                    sqrt(T((2l + 1) * (l * l - m * m)) / T(2l - 1))
                dtheta[i, l + 1, m_idx] =
                    (T(l) * xi * current - beta * previous) / s
                over_sin[i, l + 1, m_idx] = current / s
            end
        end
    end
end

"""Latitude contraction for the two tangential Fourier components."""
@kernel function vector_analysis_kernel!(Sout, Tout, Ftheta, Fphi,
                                          dtheta, over_sin, weights, scales,
                                          x, cphi, lcap, mmax, mres,
                                          robert_form)
    l_idx, m_idx = @index(Global, NTuple)
    if l_idx <= lcap + 1 && m_idx <= mmax + 1
        l = l_idx - 1
        m = m_idx - 1
        if l <= lcap && l >= max(1, m) && m % mres == 0
            Svalue = zero(eltype(Sout))
            Tvalue = zero(eltype(Tout))
            @inbounds for i in 1:length(weights)
                s = sqrt(max(zero(eltype(x)), one(eltype(x)) - x[i] * x[i]))
                Ft = Ftheta[i, m_idx]
                Fp = Fphi[i, m_idx]
                if robert_form && !iszero(s)
                    Ft /= s
                    Fp /= s
                end
                d = dtheta[i, l_idx, m_idx]
                term = complex(zero(d), typeof(d)(m) * over_sin[i, l_idx, m_idx])
                factor = weights[i] * cphi / typeof(d)(l * (l + 1))
                Svalue += factor * (Ft * d + conj(term) * Fp)
                Tvalue += factor * (-conj(term) * Ft + d * Fp)
            end
            scale = scales[l_idx, m_idx]
            Sout[l_idx, m_idx] = Svalue / scale
            Tout[l_idx, m_idx] = Tvalue / scale
        else
            Sout[l_idx, m_idx] = zero(eltype(Sout))
            Tout[l_idx, m_idx] = zero(eltype(Tout))
        end
    end
end

"""Vector Legendre synthesis into vendor-IFFT Fourier bins."""
@kernel function vector_synthesis_kernel!(Ftheta, Fphi, Sin, Tin,
                                           dtheta, over_sin, scales, x,
                                           inv_scale, nlon, lmax, mmax,
                                           mres, real_output, robert_form)
    i, m_idx = @index(Global, NTuple)
    if i <= size(Ftheta, 1) && m_idx <= mmax + 1
        m = m_idx - 1
        if m % mres == 0
            gt = zero(eltype(Ftheta))
            gp = zero(eltype(Fphi))
            @inbounds for l in max(1, m):lmax
                d = dtheta[i, l + 1, m_idx]
                p = over_sin[i, l + 1, m_idx]
                scale = scales[l + 1, m_idx]
                S = scale * Sin[l + 1, m_idx]
                Tv = scale * Tin[l + 1, m_idx]
                term = complex(zero(d), typeof(d)(m) * p)
                gt += d * S - term * Tv
                gp += term * S + d * Tv
            end
            if robert_form
                s = sqrt(max(zero(eltype(x)), one(eltype(x)) - x[i] * x[i]))
                gt *= s
                gp *= s
            end
            bt = inv_scale * gt
            bp = inv_scale * gp
            Ftheta[i, m_idx] = bt
            Fphi[i, m_idx] = bp
            if real_output && m > 0
                negative_idx = nlon - m + 1
                if negative_idx != m_idx
                    Ftheta[i, negative_idx] = conj(bt)
                    Fphi[i, negative_idx] = conj(bp)
                end
            end
        end
    end
end

"""Analyze one stored vector order without expanding to a dense spectrum."""
@kernel function vector_mode_analysis_kernel!(Sout, Tout, Ftheta, Fphi,
                                               dtheta, over_sin, weights,
                                               scales, x, cphi, physical_m,
                                               lcap, robert_form)
    q_idx = @index(Global)
    l = physical_m + q_idx - 1
    if l <= lcap
        if l < max(1, physical_m)
            # Vector spherical harmonics have no l=0 coefficient.  Keep its
            # logical axisymmetric storage slot exact without evaluating the
            # undefined 1/(l(l+1)) analysis factor.
            Sout[q_idx] = zero(eltype(Sout))
            Tout[q_idx] = zero(eltype(Tout))
        else
            Svalue = zero(eltype(Sout))
            Tvalue = zero(eltype(Tout))
            @inbounds for i in 1:length(weights)
                s = sqrt(max(zero(eltype(x)), one(eltype(x)) - x[i] * x[i]))
                Ft = Ftheta[i]
                Fp = Fphi[i]
                if robert_form && !iszero(s)
                    Ft /= s
                    Fp /= s
                end
                d = dtheta[i, l + 1, physical_m + 1]
                term = complex(zero(d), typeof(d)(physical_m) *
                                over_sin[i, l + 1, physical_m + 1])
                factor = weights[i] * cphi / typeof(d)(l * (l + 1))
                Svalue += factor * (Ft * d + conj(term) * Fp)
                Tvalue += factor * (-conj(term) * Ft + d * Fp)
            end
            scale = scales[l + 1, physical_m + 1]
            Sout[q_idx] = Svalue / scale
            Tout[q_idx] = Tvalue / scale
        end
    end
end

"""Synthesize one stored vector order directly into latitude vectors."""
@kernel function vector_mode_synthesis_kernel!(Vtheta, Vphi, Sin, Tin,
                                                dtheta, over_sin, scales, x,
                                                inv_scale, physical_m, lcap,
                                                robert_form)
    i = @index(Global)
    if i <= length(Vtheta)
        gt = zero(eltype(Vtheta))
        gp = zero(eltype(Vphi))
        @inbounds for l in max(1, physical_m):lcap
            scale = scales[l + 1, physical_m + 1]
            S = scale * Sin[l - physical_m + 1]
            Tvalue = scale * Tin[l - physical_m + 1]
            d = dtheta[i, l + 1, physical_m + 1]
            term = complex(zero(d), typeof(d)(physical_m) *
                            over_sin[i, l + 1, physical_m + 1])
            gt += d * S - term * Tvalue
            gp += term * S + d * Tvalue
        end
        if robert_form
            s = sqrt(max(zero(eltype(x)), one(eltype(x)) - x[i] * x[i]))
            gt *= s
            gp *= s
        end
        Vtheta[i] = inv_scale * gt
        Vphi[i] = inv_scale * gp
    end
end

"""Latitude contraction for vector fields in a trailing batch dimension."""
@kernel function vector_batch_analysis_kernel!(Sout, Tout, Ftheta, Fphi,
                                                dtheta, over_sin, weights,
                                                scales, x, cphi, lmax, mmax,
                                                mres, robert_form)
    l_idx, m_idx, batch_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1 &&
       batch_idx <= size(Sout, 3)
        l = l_idx - 1
        m = m_idx - 1
        if l >= max(1, m) && m % mres == 0
            Svalue = zero(eltype(Sout))
            Tvalue = zero(eltype(Tout))
            @inbounds for i in 1:length(weights)
                s = sqrt(max(zero(eltype(x)), one(eltype(x)) - x[i] * x[i]))
                Ft = Ftheta[i, m_idx, batch_idx]
                Fp = Fphi[i, m_idx, batch_idx]
                if robert_form && !iszero(s)
                    Ft /= s
                    Fp /= s
                end
                d = dtheta[i, l_idx, m_idx]
                term = complex(zero(d), typeof(d)(m) * over_sin[i, l_idx, m_idx])
                factor = weights[i] * cphi / typeof(d)(l * (l + 1))
                Svalue += factor * (Ft * d + conj(term) * Fp)
                Tvalue += factor * (-conj(term) * Ft + d * Fp)
            end
            scale = scales[l_idx, m_idx]
            Sout[l_idx, m_idx, batch_idx] = Svalue / scale
            Tout[l_idx, m_idx, batch_idx] = Tvalue / scale
        else
            Sout[l_idx, m_idx, batch_idx] = zero(eltype(Sout))
            Tout[l_idx, m_idx, batch_idx] = zero(eltype(Tout))
        end
    end
end

"""Vector synthesis with independent fields in the trailing batch dimension."""
@kernel function vector_batch_synthesis_kernel!(Ftheta, Fphi, Sin, Tin,
                                                 dtheta, over_sin, scales, x,
                                                 inv_scale, nlon, lmax, mmax,
                                                 mres, real_output, robert_form)
    i, m_idx, batch_idx = @index(Global, NTuple)
    if i <= size(Ftheta, 1) && m_idx <= mmax + 1 &&
       batch_idx <= size(Ftheta, 3)
        m = m_idx - 1
        if m % mres == 0
            gt = zero(eltype(Ftheta))
            gp = zero(eltype(Fphi))
            @inbounds for l in max(1, m):lmax
                scale = scales[l + 1, m_idx]
                S = scale * Sin[l + 1, m_idx, batch_idx]
                Tvalue = scale * Tin[l + 1, m_idx, batch_idx]
                d = dtheta[i, l + 1, m_idx]
                term = complex(zero(d), typeof(d)(m) * over_sin[i, l + 1, m_idx])
                gt += d * S - term * Tvalue
                gp += term * S + d * Tvalue
            end
            if robert_form
                s = sqrt(max(zero(eltype(x)), one(eltype(x)) - x[i] * x[i]))
                gt *= s
                gp *= s
            end
            bt = inv_scale * gt
            bp = inv_scale * gp
            Ftheta[i, m_idx, batch_idx] = bt
            Fphi[i, m_idx, batch_idx] = bp
            if real_output && m > 0
                negative_idx = nlon - m + 1
                if negative_idx != m_idx
                    Ftheta[i, negative_idx, batch_idx] = conj(bt)
                    Fphi[i, negative_idx, batch_idx] = conj(bp)
                end
            end
        end
    end
end

"""Apply or invert the tangential `-l(l+1)` spectral multiplier."""
@kernel function vector_diagonal_kernel!(output, input, lmax, mmax, mres,
                                         inverse)
    l_idx, m_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1
        l = l_idx - 1
        m = m_idx - 1
        if l >= max(1, m) && m % mres == 0
            ll1 = l * (l + 1)
            output[l_idx, m_idx] = inverse ?
                -(input[l_idx, m_idx] / ll1) :
                -ll1 * input[l_idx, m_idx]
        else
            output[l_idx, m_idx] = zero(eltype(output))
        end
    end
end

"""Latitude integration after the vendor FFT; output is canonical."""
@kernel function scalar_analysis_kernel!(canonical, fourier, Plm, weights,
                                          cphi, lmax, mmax, mres, lcap)
    l_idx, m_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1
        l = l_idx - 1
        m = m_idx - 1
        if l <= lcap && l >= m && m % mres == 0
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

"""Pack a dense non-negative-order spectrum, optionally truncating in degree."""
@kernel function real_pack_kernel!(packed, dense, lmax, mmax, mres, lcap)
    l_idx, im_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && im_idx <= mmax ÷ mres + 1
        l = l_idx - 1
        im = im_idx - 1
        m = im * mres
        if m <= mmax && l >= m
            base = (im * (2lmax + 2 - (im + 1) * mres)) >>> 1
            packed[base + l + 1] = l <= lcap ? dense[l_idx, m + 1] : zero(eltype(packed))
        end
    end
end

"""Expand SHTns LM storage to a dense matrix with an explicit degree cap."""
@kernel function real_unpack_kernel!(dense, packed, lmax, mmax, mres, lcap)
    l_idx, m_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1
        l = l_idx - 1
        m = m_idx - 1
        if l <= lcap && l >= m && m % mres == 0
            im = m ÷ mres
            base = (im * (2lmax + 2 - (im + 1) * mres)) >>> 1
            dense[l_idx, m_idx] = packed[base + l + 1]
        else
            dense[l_idx, m_idx] = zero(eltype(dense))
        end
    end
end

"""Analyze one physical Fourier order over an explicit degree interval."""
@kernel function mode_analysis_kernel!(output, mode, Plm, weights, scale,
                                       physical_m, lcap)
    q_idx = @index(Global)
    l = physical_m + q_idx - 1
    if l <= lcap
        value = zero(eltype(output))
        @inbounds for i in 1:length(weights)
            value += weights[i] * Plm[i, l + 1, physical_m + 1] * mode[i]
        end
        output[q_idx] = scale * value
    end
end

"""Synthesize one physical Fourier order over an explicit degree interval."""
@kernel function mode_synthesis_kernel!(mode, coefficients, Plm, scale,
                                        physical_m, lcap)
    i = @index(Global)
    if i <= size(Plm, 1)
        value = zero(eltype(mode))
        @inbounds for l in physical_m:lcap
            value += Plm[i, l + 1, physical_m + 1] *
                     coefficients[l - physical_m + 1]
        end
        mode[i] = scale * value
    end
end

"""Latitude integration for independent scalar fields in the trailing axis."""
@kernel function scalar_batch_analysis_kernel!(canonical, fourier, Plm, weights,
                                                cphi, lmax, mmax, mres)
    l_idx, m_idx, batch_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1 &&
       batch_idx <= size(canonical, 3)
        l = l_idx - 1
        m = m_idx - 1
        if l >= m && m % mres == 0
            value = zero(eltype(canonical))
            @inbounds for i in 1:length(weights)
                value += weights[i] * Plm[i, l_idx, m_idx] *
                         fourier[i, m_idx, batch_idx]
            end
            canonical[l_idx, m_idx, batch_idx] = cphi * value
        else
            canonical[l_idx, m_idx, batch_idx] = zero(eltype(canonical))
        end
    end
end

"""Legendre synthesis for independent scalar fields in the trailing axis."""
@kernel function scalar_batch_synthesis_kernel!(fourier, canonical, Plm, inv_scale,
                                                 nlon, lmax, mmax, mres,
                                                 real_output)
    i, m_idx, batch_idx = @index(Global, NTuple)
    if i <= size(fourier, 1) && m_idx <= mmax + 1 &&
       batch_idx <= size(fourier, 3)
        m = m_idx - 1
        if m % mres == 0
            value = zero(eltype(fourier))
            @inbounds for l in m:lmax
                value += Plm[i, l + 1, m_idx] * canonical[l + 1, m_idx, batch_idx]
            end
            bin = inv_scale * value
            fourier[i, m_idx, batch_idx] = bin
            if real_output && m > 0
                negative_idx = nlon - m + 1
                if negative_idx != m_idx
                    fourier[i, negative_idx, batch_idx] = conj(bin)
                end
            end
        end
    end
end

@inline function _lm_cplx_device_index(l, m, mmax)
    return l <= mmax ? l * (l + 1) + m : mmax * (2l - mmax) + l + m
end

"""Analyze both Fourier signs directly into SHTns LM_cplx storage."""
@kernel function complex_packed_analysis_kernel!(packed, fourier, Plm, weights,
                                                 scales, cphi, nlon, lcap,
                                                 mmax, mcap)
    l_idx, signed_idx = @index(Global, NTuple)
    m = signed_idx - mcap - 1
    am = abs(m)
    l = l_idx - 1
    if l <= lcap && am <= mcap && l >= am
        column = m >= 0 ? m + 1 : nlon + m + 1
        value = zero(eltype(packed))
        @inbounds for i in 1:length(weights)
            value += weights[i] * Plm[i, l_idx, am + 1] * fourier[i, column]
        end
        packed[_lm_cplx_device_index(l, m, mmax) + 1] =
            cphi * value / scales[l_idx, am + 1]
    end
end

"""Synthesize both Fourier signs directly from SHTns LM_cplx storage."""
@kernel function complex_packed_synthesis_kernel!(fourier, packed, Plm, scales,
                                                  inv_scale, nlon, lcap,
                                                  mmax, mcap)
    i, signed_idx = @index(Global, NTuple)
    m = signed_idx - mcap - 1
    am = abs(m)
    if i <= size(fourier, 1) && am <= mcap
        value = zero(eltype(fourier))
        @inbounds for l in am:lcap
            coefficient = packed[_lm_cplx_device_index(l, m, mmax) + 1] *
                          scales[l + 1, am + 1]
            value += Plm[i, l + 1, am + 1] * coefficient
        end
        column = m >= 0 ? m + 1 : nlon + m + 1
        fourier[i, column] = inv_scale * value
    end
end

@inline function _real_packed_device_index(l, m, lmax, mres)
    im = m ÷ mres
    base = (im * (2lmax + 2 - (im + 1) * mres)) >>> 1
    return base + l
end

"""Evaluate a dense non-negative-m real spectrum at one or more longitudes."""
@kernel function local_scalar_kernel!(output, coefficients, Plm, scales,
                                      phi0, phi_step, lmax, mmax, mres,
                                      lcap, mcap)
    j = @index(Global)
    if j <= length(output)
        phi = phi0 + (j - 1) * phi_step
        value = zero(eltype(output))
        @inbounds for m in 0:mcap
            if m % mres == 0
                radial = zero(eltype(coefficients))
                for l in m:lcap
                    radial += Plm[1, l + 1, m + 1] *
                              scales[l + 1, m + 1] * coefficients[l + 1, m + 1]
                end
                wave = radial * cis(m * phi)
                value += m == 0 ? real(wave) : 2real(wave)
            end
        end
        output[j] = value
    end
end

"""Evaluate SHTns LM_cplx storage at one or more longitudes."""
@kernel function local_complex_kernel!(output, coefficients, Plm, scales,
                                       phi0, phi_step, lmax, mmax, lcap)
    j = @index(Global)
    if j <= length(output)
        phi = phi0 + (j - 1) * phi_step
        value = zero(eltype(output))
        @inbounds for m in -mmax:mmax
            am = abs(m)
            radial = zero(eltype(output))
            for l in am:lcap
                index = _lm_cplx_device_index(l, m, mmax) + 1
                radial += Plm[1, l + 1, am + 1] *
                          scales[l + 1, am + 1] * coefficients[index]
            end
            value += radial * cis(m * phi)
        end
        output[j] = value
    end
end

"""
Evaluate packed real Q/S/T spectra at one or more longitudes. Boolean component
flags let the scalar-gradient path reuse this kernel without allocating zero
spectra on the device.
"""
@kernel function local_qst_kernel!(Vr, Vt, Vp, Q, S, Tlm,
                                   Plm, dtheta, over_sin, scales,
                                   phi0, phi_step, lmax, mmax, mres,
                                   lcap, mcap, has_q, has_s, has_t,
                                   robert_form, sinth)
    j = @index(Global)
    if j <= length(Vr)
        phi = phi0 + (j - 1) * phi_step
        vr = zero(eltype(Vr))
        vt = zero(eltype(Vt))
        vp = zero(eltype(Vp))
        imagunit = complex(zero(eltype(Vr)), one(eltype(Vr)))
        @inbounds for m in 0:mcap
            if m % mres == 0
                qmode = zero(eltype(Q))
                smode_t = zero(eltype(S))
                smode_p = zero(eltype(S))
                tmode_t = zero(eltype(Tlm))
                tmode_p = zero(eltype(Tlm))
                for l in m:lcap
                    index = _real_packed_device_index(l, m, lmax, mres) + 1
                    scale = scales[l + 1, m + 1]
                    has_q && (qmode += Plm[1, l + 1, m + 1] * scale * Q[index])
                    if has_s
                        coefficient = scale * S[index]
                        smode_t += dtheta[1, l + 1, m + 1] * coefficient
                        smode_p += imagunit * m * over_sin[1, l + 1, m + 1] * coefficient
                    end
                    if has_t
                        coefficient = scale * Tlm[index]
                        tmode_t -= imagunit * m * over_sin[1, l + 1, m + 1] * coefficient
                        tmode_p += dtheta[1, l + 1, m + 1] * coefficient
                    end
                end
                phase = cis(m * phi)
                if m == 0
                    vr += real(qmode)
                    vt += real(smode_t + tmode_t)
                    vp += real(smode_p + tmode_p)
                else
                    vr += 2real(qmode * phase)
                    vt += 2real((smode_t + tmode_t) * phase)
                    vp += 2real((smode_p + tmode_p) * phase)
                end
            end
        end
        Vr[j] = vr
        Vt[j] = robert_form ? sinth * vt : vt
        Vp[j] = robert_form ? sinth * vp : vp
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
