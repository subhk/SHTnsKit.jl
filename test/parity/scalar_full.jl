using Test
using SHTnsKit

"""Test-only adapter for the scalar full-grid parity suite."""
abstract type ScalarParityAdapter end

struct CPUScalarAdapter <: ScalarParityAdapter end

place(::CPUScalarAdapter, ::SHTConfig, value, ::Symbol) = value
collect_result(::CPUScalarAdapter, value, ::SHTConfig) = Array(value)
analysis_call(::CPUScalarAdapter, cfg, field) = analysis(CPU(), cfg, field)
synthesis_call(::CPUScalarAdapter, cfg, coefficients, _prototype; real_output) =
    synthesis(CPU(), cfg, coefficients; real_output)
synthesis_cplx_call(::CPUScalarAdapter, cfg, coefficients, _prototype) =
    synthesis_cplx(cfg, coefficients)
assert_resident(::CPUScalarAdapter, value) = @test on_device(value) isa CPU

const _SCALAR_GRID_KINDS = (:gauss, :gauss_fly, :regular, :regular_poles)

function _scalar_config(kind::Symbol, lmax::Int, nlat::Int;
                        mres::Int=1, norm::Symbol=:orthonormal,
                        real_norm::Bool=false, cs_phase::Bool=true,
                        south_pole_first::Bool=false)
    nlon = 2lmax + 2
    cfg = if kind === :gauss
        create_gauss_config(lmax, nlat; nlon, mres, norm, real_norm, cs_phase)
    elseif kind === :gauss_fly
        create_gauss_fly_config(lmax, nlat; nlon, mres, norm, real_norm, cs_phase)
    elseif kind === :regular
        create_regular_config(lmax, nlat; nlon, mres, norm, real_norm, cs_phase,
                              include_poles=false, precompute_plm=true)
    elseif kind === :regular_poles
        create_regular_config(lmax, nlat; nlon, mres, norm, real_norm, cs_phase,
                              include_poles=true, precompute_plm=true)
    else
        error("unknown scalar parity grid kind: $kind")
    end
    south_pole_first && set_south_pole_first!(cfg)
    return cfg
end

_scalar_tol(::Type{Float32}) = (atol=2f-4, rtol=2f-4)
_scalar_tol(::Type{Float64}) = (atol=3e-11, rtol=3e-11)

function _canonical_coefficients(cfg, coefficients)
    canonical = similar(coefficients)
    SHTnsKit.convert_alm_norm!(canonical, coefficients, cfg; to_internal=true)
    return canonical
end

"""Independent direct sum for the non-negative-m dense scalar representation."""
function _direct_scalar_sum(cfg, coefficients::AbstractMatrix{Complex{T}};
                            real_output::Bool) where {T<:AbstractFloat}
    canonical = _canonical_coefficients(cfg, coefficients)
    result = Matrix{Complex{T}}(undef, cfg.nlat, cfg.nlon)
    P = Vector{Float64}(undef, cfg.lmax + 1)
    for j in 1:cfg.nlon, i in 1:cfg.nlat
        value = zero(Complex{T})
        for m in 0:cfg.mmax
            SHTnsKit.Plm_norm_row!(P, cfg.x[i], cfg.lmax, m)
            radial = zero(Complex{T})
            for l in m:cfg.lmax
                radial += T(P[l + 1]) * canonical[l + 1, m + 1]
            end
            wave = radial * cis(T(m) * T(cfg.φ[j]))
            if real_output
                value += m == 0 ? Complex{T}(real(wave), 0) : Complex{T}(2real(wave), 0)
            else
                value += wave
            end
        end
        result[i, j] = value
    end
    return real_output ? real.(result) : result
end

function _test_scalar_case(adapter::ScalarParityAdapter, cfg, ::Type{T}) where {T<:AbstractFloat}
    tol = _scalar_tol(T)
    CT = Complex{T}
    coefficients = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    coefficients[1, 1] = CT(T(0.35), 0)
    coefficients[3, 1] = CT(T(-0.2), 0)
    coefficients[3, 3] = CT(T(0.17), T(-0.11))
    if cfg.mres == 1
        coefficients[2, 2] = CT(T(-0.13), T(0.09))
    end

    reference_real = _direct_scalar_sum(cfg, coefficients; real_output=true)
    prototype = place(adapter, cfg, reference_real, :spatial)
    device_coefficients = place(adapter, cfg, coefficients, :spectral)

    reconstructed = synthesis_call(
        adapter, cfg, device_coefficients, prototype; real_output=true)
    assert_resident(adapter, reconstructed)
    reconstructed_host = collect_result(adapter, reconstructed, cfg)
    @test eltype(reconstructed_host) === T
    @test reconstructed_host ≈ reference_real atol=tol.atol rtol=tol.rtol

    analyzed = analysis_call(adapter, cfg, prototype)
    assert_resident(adapter, analyzed)
    analyzed_host = collect_result(adapter, analyzed, cfg)
    @test eltype(analyzed_host) === CT
    @test analyzed_host ≈ coefficients atol=tol.atol rtol=tol.rtol

    # Complex-output synthesis is the positive-m half represented by the dense
    # matrix API. It must not accidentally install Hermitian negative-m bins.
    complex_coefficients = copy(coefficients)
    complex_coefficients[1, 1] = CT(T(0.35), T(0.21))
    reference_complex = _direct_scalar_sum(cfg, complex_coefficients; real_output=false)
    device_complex_coefficients = place(adapter, cfg, complex_coefficients, :spectral)
    reconstructed_complex = synthesis_call(
        adapter, cfg, device_complex_coefficients, prototype; real_output=false)
    assert_resident(adapter, reconstructed_complex)
    reconstructed_complex_host = collect_result(adapter, reconstructed_complex, cfg)
    @test eltype(reconstructed_complex_host) === CT
    @test reconstructed_complex_host ≈ reference_complex atol=tol.atol rtol=tol.rtol

    reconstructed_explicit = synthesis_cplx_call(
        adapter, cfg, device_complex_coefficients, prototype,
    )
    assert_resident(adapter, reconstructed_explicit)
    reconstructed_explicit_host = collect_result(adapter, reconstructed_explicit, cfg)
    @test eltype(reconstructed_explicit_host) === CT
    @test reconstructed_explicit_host ≈ reference_complex atol=tol.atol rtol=tol.rtol
    @test reconstructed_explicit_host ≈ reconstructed_complex_host atol=tol.atol rtol=tol.rtol

    complex_input = CT.(reference_real, T(0.3) .* reference_real)
    complex_prototype = place(adapter, cfg, complex_input, :spatial)
    complex_analysis = analysis_call(adapter, cfg, complex_prototype)
    assert_resident(adapter, complex_analysis)
    @test collect_result(adapter, complex_analysis, cfg) ≈
          analysis(CPU(), cfg, complex_input) atol=tol.atol rtol=tol.rtol

    return nothing
end

function _test_noncontiguous_scalar_input(adapter::ScalarParityAdapter)
    cfg = _scalar_config(:gauss, 3, 7)
    host = Matrix{Float64}(undef, cfg.nlat, cfg.nlon)
    for j in axes(host, 2), i in axes(host, 1)
        host[i, j] = 0.2 + cfg.x[i] - 0.4sqrt(1 - cfg.x[i]^2) * cos(cfg.φ[j])
    end
    longitude_first = permutedims(host, (2, 1))
    noncontiguous = PermutedDimsArray(longitude_first, (2, 1))
    @test parent(noncontiguous) === longitude_first
    placed = place(adapter, cfg, noncontiguous, :spatial)
    assert_resident(adapter, placed)
    got = analysis_call(adapter, cfg, placed)
    assert_resident(adapter, got)
    @test collect_result(adapter, got, cfg) ≈ analysis(CPU(), cfg, host) atol=3e-11 rtol=3e-11
    return nothing
end

function _test_mres_filters_unstored_orders(adapter::ScalarParityAdapter)
    cfg = _scalar_config(:gauss, 3, 8; mres=2)
    unsupported = Matrix{Float64}(undef, cfg.nlat, cfg.nlon)
    for j in 1:cfg.nlon, i in 1:cfg.nlat
        unsupported[i, j] = sqrt(1 - cfg.x[i]^2) * cos(cfg.φ[j])
    end
    placed = place(adapter, cfg, unsupported, :spatial)
    coefficients = analysis_call(adapter, cfg, placed)
    host = collect_result(adapter, coefficients, cfg)
    @test all(iszero, host[:, 2])

    unsupported_coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    unsupported_coefficients[2, 2] = 1 - 0.5im
    device_coefficients = place(adapter, cfg, unsupported_coefficients, :spectral)
    synthesized = synthesis_call(
        adapter, cfg, device_coefficients, placed; real_output=false,
    )
    @test all(iszero, collect_result(adapter, synthesized, cfg))
    return nothing
end

function test_mres_scalar_adjoints()
    cfg = _scalar_config(:gauss, 3, 8; mres=2)
    spatial_cotangent = Matrix{Float64}(undef, cfg.nlat, cfg.nlon)
    for j in 1:cfg.nlon, i in 1:cfg.nlat
        spatial_cotangent[i, j] = sqrt(1 - cfg.x[i]^2) * cos(cfg.φ[j])
    end
    coefficient_cotangent = SHTnsKit._adjoint_synthesis(cfg, spatial_cotangent)
    @test all(iszero, coefficient_cotangent[:, 2])

    unsupported_coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    unsupported_coefficients[2, 2] = 1 - 0.5im
    field_cotangent = SHTnsKit._adjoint_analysis(cfg, unsupported_coefficients)
    @test all(iszero, field_cotangent)
    return nothing
end

"""
Run scalar full-grid parity through a backend adapter. The default sweep covers
all grid, precision, m-resolution, convention, phase, and pole-order cells.
Runners may pass smaller axes for expensive hardware/collective smoke checks.
"""
function run_scalar_full_parity(adapter::ScalarParityAdapter;
                                grid_kinds=_SCALAR_GRID_KINDS,
                                precisions=(Float32, Float64),
                                mres_values=(1, 2),
                                norms=(:orthonormal, :fourpi, :schmidt),
                                real_norm_values=(false, true),
                                cs_phase_values=(false, true),
                                pole_orders=(false, true))
    @testset "scalar full-grid parity $(nameof(typeof(adapter)))" begin
        for kind in grid_kinds, T in precisions, mres in mres_values,
            norm in norms, real_norm in real_norm_values,
            cs_phase in cs_phase_values, south_pole_first in pole_orders
            cfg = _scalar_config(
                kind, 3, 8;
                mres, norm, real_norm, cs_phase, south_pole_first,
            )
            _test_scalar_case(adapter, cfg, T)
        end
        _test_noncontiguous_scalar_input(adapter)
        _test_mres_filters_unstored_orders(adapter)
    end
    return nothing
end

"""Compile and numerically check the vendor-neutral kernels on a KA CPU backend."""
function run_shared_scalar_kernel_reference(common, backend)
    cfg = _scalar_config(
        :gauss, 3, 8;
        mres=2, norm=:schmidt, real_norm=true, cs_phase=false,
    )
    T = Float32
    CT = ComplexF32
    signature = common.scalar_config_signature(cfg)
    cfg.w[1] += eps(Float64)
    @test common.scalar_config_signature(cfg) != signature
    cfg.w[1] -= eps(Float64)
    signature = common.scalar_config_signature(cfg)
    cfg.norm = :fourpi
    @test common.scalar_config_signature(cfg) != signature
    cfg.norm = :schmidt

    x, weights, scales = common.scalar_host_tables(cfg, T)
    Plm = zeros(T, cfg.nlat, cfg.lmax + 1, cfg.mmax + 1)
    event = common.legendre_table_kernel!(backend)(
        Plm, x, cfg.lmax, cfg.mmax;
        ndrange=(cfg.nlat, cfg.mmax + 1),
    )
    event === nothing || wait(event)
    reference_row = zeros(Float64, cfg.lmax + 1)
    for m in 0:cfg.mmax, i in 1:cfg.nlat
        SHTnsKit.Plm_norm_row!(reference_row, cfg.x[i], cfg.lmax, m)
        @test Plm[i, (m + 1):(cfg.lmax + 1), m + 1] ≈
              T.(reference_row[(m + 1):(cfg.lmax + 1)]) atol=3f-6 rtol=3f-6
    end

    fourier = zeros(CT, cfg.nlat, cfg.nlon)
    for i in 1:cfg.nlat
        fourier[i, 1] = CT(T(0.2cfg.x[i]), T(0.1))
        fourier[i, 3] = CT(T(-0.3 + cfg.x[i]), T(0.15cfg.x[i]))
    end
    canonical = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    event = common.scalar_analysis_kernel!(backend)(
        canonical, fourier, Plm, weights, T(cfg.cphi), cfg.lmax, cfg.mmax,
        cfg.mres; ndrange=size(canonical),
    )
    event === nothing || wait(event)
    @test all(iszero, canonical[:, 2])

    configured = similar(canonical)
    event = common.coefficient_conversion_kernel!(backend)(
        configured, canonical, scales, cfg.lmax, cfg.mmax, false;
        ndrange=size(canonical),
    )
    event === nothing || wait(event)
    canonical_roundtrip = similar(canonical)
    event = common.coefficient_conversion_kernel!(backend)(
        canonical_roundtrip, configured, scales, cfg.lmax, cfg.mmax, true;
        ndrange=size(canonical),
    )
    event === nothing || wait(event)
    @test canonical_roundtrip ≈ canonical atol=8f-7 rtol=8f-7

    synthesized_bins = zeros(CT, cfg.nlat, cfg.nlon)
    event = common.scalar_synthesis_kernel!(backend)(
        synthesized_bins, canonical_roundtrip, Plm,
        T(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon, cfg.lmax, cfg.mmax,
        cfg.mres, true; ndrange=(cfg.nlat, cfg.mmax + 1),
    )
    event === nothing || wait(event)
    @test synthesized_bins[:, 2] == zeros(CT, cfg.nlat)
    @test synthesized_bins[:, cfg.nlon - 1] ≈ conj.(synthesized_bins[:, 3])
    return nothing
end
