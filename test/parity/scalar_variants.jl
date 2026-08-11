using Test
using Random
using SHTnsKit

struct VariantNonCPUArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
end
Base.size(array::VariantNonCPUArray) = size(array.parent)
Base.getindex(array::VariantNonCPUArray, indices...) = getindex(array.parent, indices...)
SHTnsKit.on_device(::VariantNonCPUArray) = SHTnsKit.GPU()

function _variant_coefficients(cfg, ::Type{T}) where {T<:AbstractFloat}
    coefficients = zeros(Complex{T}, cfg.lmax + 1, cfg.mmax + 1)
    for m in 0:cfg.mmax
        m % cfg.mres == 0 || continue
        for l in m:cfg.lmax
            coefficients[l + 1, m + 1] = Complex{T}(
                T(0.03 * (l + 1) - 0.02m), T(m == 0 ? 0 : 0.01 * (l + m + 1)),
            )
        end
    end
    return coefficients
end

"""Exercise the vendor-neutral variant kernels on a KA CPU backend."""
function run_shared_scalar_variant_kernel_reference(common, backend)
    for name in (
        :real_pack_kernel!, :real_unpack_kernel!,
        :mode_analysis_kernel!, :mode_synthesis_kernel!,
        :scalar_batch_analysis_kernel!, :scalar_batch_synthesis_kernel!,
        :complex_packed_analysis_kernel!, :complex_packed_synthesis_kernel!,
    )
        @test isdefined(common, name)
    end

    cfg = create_gauss_config(5, 8; nlon=14, mres=2)
    dense = _variant_coefficients(cfg, Float32)
    packed = zeros(ComplexF32, cfg.nlm)
    event = common.real_pack_kernel!(backend)(
        packed, dense, cfg.lmax, cfg.mmax, cfg.mres, 3;
        ndrange=(cfg.lmax + 1, cfg.mmax ÷ cfg.mres + 1),
    )
    event === nothing || wait(event)
    for im in 0:(cfg.mmax ÷ cfg.mres), l in (im * cfg.mres):cfg.lmax
        idx = LM_index(cfg.lmax, cfg.mres, l, im * cfg.mres) + 1
        @test packed[idx] == (l <= 3 ? dense[l + 1, im * cfg.mres + 1] : 0)
    end

    unpacked = fill(ComplexF32(9, -4), size(dense))
    event = common.real_unpack_kernel!(backend)(
        unpacked, packed, cfg.lmax, cfg.mmax, cfg.mres, 3;
        ndrange=size(unpacked),
    )
    event === nothing || wait(event)
    for m in 0:cfg.mmax, l in 0:cfg.lmax
        expected = l >= m && m % cfg.mres == 0 && l <= 3 ? dense[l + 1, m + 1] : 0
        @test unpacked[l + 1, m + 1] == expected
    end

    x, weights, scales = common.scalar_host_tables(cfg, Float32)
    Plm = zeros(Float32, cfg.nlat, cfg.lmax + 1, cfg.mmax + 1)
    event = common.legendre_table_kernel!(backend)(
        Plm, x, cfg.lmax, cfg.mmax;
        ndrange=(cfg.nlat, cfg.mmax + 1),
    )
    event === nothing || wait(event)

    physical_m = 2
    mode_coefficients = ComplexF32[0.12 - 0.04im, -0.08 + 0.03im,
                                   0.05 + 0.02im, -0.01 + 0.06im]
    mode = zeros(ComplexF32, cfg.nlat)
    event = common.mode_synthesis_kernel!(backend)(
        mode, mode_coefficients, Plm, Float32(SHTnsKit.phi_inv_scale(cfg)),
        physical_m, cfg.lmax; ndrange=cfg.nlat,
    )
    event === nothing || wait(event)
    mode_back = zeros(ComplexF32, length(mode_coefficients))
    event = common.mode_analysis_kernel!(backend)(
        mode_back, mode, Plm, weights, Float32(cfg.cphi),
        physical_m, cfg.lmax; ndrange=length(mode_back),
    )
    event === nothing || wait(event)
    @test mode_back ≈ mode_coefficients atol=2f-5 rtol=2f-5

    canonical_batch = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1, 2)
    canonical_batch[:, :, 1] .= dense
    canonical_batch[:, :, 2] .= 2f0 .* dense
    batch_bins = zeros(ComplexF32, cfg.nlat, cfg.nlon, 2)
    event = common.scalar_batch_synthesis_kernel!(backend)(
        batch_bins, canonical_batch, Plm, Float32(SHTnsKit.phi_inv_scale(cfg)),
        cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres, true;
        ndrange=(cfg.nlat, cfg.mmax + 1, 2),
    )
    event === nothing || wait(event)
    batch_back = zeros(ComplexF32, size(canonical_batch))
    event = common.scalar_batch_analysis_kernel!(backend)(
        batch_back, batch_bins, Plm, weights, Float32(cfg.cphi),
        cfg.lmax, cfg.mmax, cfg.mres; ndrange=size(batch_back),
    )
    event === nothing || wait(event)
    @test batch_back ≈ canonical_batch atol=2f-5 rtol=2f-5

    complex_cfg = create_gauss_config(3, 6; nlon=10)
    cx, cw, cscales = common.scalar_host_tables(complex_cfg, Float32)
    cPlm = zeros(Float32, complex_cfg.nlat, complex_cfg.lmax + 1,
                 complex_cfg.mmax + 1)
    event = common.legendre_table_kernel!(backend)(
        cPlm, cx, complex_cfg.lmax, complex_cfg.mmax;
        ndrange=(complex_cfg.nlat, complex_cfg.mmax + 1),
    )
    event === nothing || wait(event)
    complex_coefficients = zeros(ComplexF32, nlm_cplx_calc(3, 3, 1))
    complex_coefficients[LM_cplx_index(3, 3, 2, -1) + 1] = 0.2f0 - 0.1f0im
    complex_coefficients[LM_cplx_index(3, 3, 3, 2) + 1] = -0.08f0 + 0.04f0im
    complex_bins = zeros(ComplexF32, complex_cfg.nlat, complex_cfg.nlon)
    event = common.complex_packed_synthesis_kernel!(backend)(
        complex_bins, complex_coefficients, cPlm, cscales,
        Float32(SHTnsKit.phi_inv_scale(complex_cfg)), complex_cfg.nlon,
        complex_cfg.lmax, complex_cfg.mmax;
        ndrange=(complex_cfg.nlat, 2complex_cfg.mmax + 1),
    )
    event === nothing || wait(event)
    complex_back = similar(complex_coefficients)
    event = common.complex_packed_analysis_kernel!(backend)(
        complex_back, complex_bins, cPlm, cw, cscales, Float32(complex_cfg.cphi),
        complex_cfg.nlon, complex_cfg.lmax, complex_cfg.mmax;
        ndrange=(complex_cfg.lmax + 1, 2complex_cfg.mmax + 1),
    )
    event === nothing || wait(event)
    @test complex_back ≈ complex_coefficients atol=2f-5 rtol=2f-5
    return nothing
end

@testset "scalar variant parity" begin
    @testset "fixed-order im maps through mres" begin
        cfg = create_gauss_config(6, 9; nlon=16, mres=2)
        im = 2
        m = im * cfg.mres
        ltr = cfg.lmax
        coefficients = ComplexF64[0.21 - 0.13im, -0.08 + 0.17im, 0.05 + 0.02im]

        mode = synthesis_packed_ml(cfg, im, coefficients, ltr)
        reference = synthesis_packed_ml(
            create_gauss_config(6, 9; nlon=16, mres=1),
            m, coefficients, ltr,
        )

        @test mode ≈ reference atol=2e-12 rtol=2e-12
        @test analysis_packed_ml(cfg, im, mode, ltr) ≈ coefficients atol=2e-11 rtol=2e-11
    end

    @testset "variant precision is preserved" begin
        cfg = create_gauss_config(4, 7; nlon=12)
        axis_coefficients = ComplexF32[0.2, -0.1, 0.05, 0.03, -0.02]
        axis_field = synthesis_axisym(cfg, axis_coefficients)
        @test eltype(axis_field) === Float32
        @test eltype(analysis_axisym(cfg, axis_field)) === ComplexF32

        mode_coefficients = ComplexF32[0.17 - 0.08im, -0.04 + 0.11im, 0.03]
        mode_field = synthesis_packed_ml(cfg, 2, mode_coefficients, 4)
        @test eltype(mode_field) === ComplexF32
        @test eltype(analysis_packed_ml(cfg, 2, mode_field, 4)) === ComplexF32
    end

    @testset "truncation and stored-order validation is consistent" begin
        cfg = create_gauss_config(5, 8; nlon=14, mres=2)
        field = zeros(Float64, cfg.nspat)
        packed = zeros(ComplexF64, cfg.nlm)
        axis = zeros(Float64, cfg.nlat)

        for ltr in (-1, cfg.lmax + 1)
            @test_throws ArgumentError analysis_packed_l(cfg, field, ltr)
            @test_throws ArgumentError synthesis_packed_l(cfg, packed, ltr)
            @test_throws ArgumentError analysis_axisym_l(cfg, axis, ltr)
            @test_throws ArgumentError synthesis_axisym_l(
                cfg, zeros(ComplexF64, cfg.lmax + 1), ltr,
            )
        end

        invalid_im = cfg.mmax ÷ cfg.mres + 1
        @test_throws ArgumentError analysis_packed_ml(
            cfg, invalid_im, zeros(ComplexF64, cfg.nlat), cfg.lmax,
        )
        @test_throws ArgumentError synthesis_packed_ml(
            cfg, invalid_im, ComplexF64[], cfg.lmax,
        )
        @test_throws ArgumentError analysis_packed_ml(
            cfg, 2, zeros(ComplexF64, cfg.nlat), 3,
        )
        @test_throws ArgumentError synthesis_packed_ml(
            cfg, 2, ComplexF64[], 3,
        )
    end

    @testset "packed and truncated storage matches dense transforms" begin
        for T in (Float32, Float64), norm in (:orthonormal, :schmidt)
            cfg = create_gauss_config(
                5, 8; nlon=14, mres=2, norm,
                real_norm=norm === :schmidt, cs_phase=norm !== :schmidt,
            )
            dense = _variant_coefficients(cfg, T)
            packed = SHTnsKit.pack_lm(cfg, dense)
            @test SHTnsKit.unpack_lm(cfg, packed) == dense
            flat = synthesis_packed(cfg, packed)
            @test eltype(flat) === T
            @test reshape(flat, cfg.nlat, cfg.nlon) ≈ synthesis(cfg, dense)
            @test analysis_packed(cfg, flat) ≈ packed atol=10eps(T) rtol=2e-4

            ltr = 3
            truncated_dense = copy(dense)
            for m in 0:cfg.mmax, l in max(m, ltr + 1):cfg.lmax
                truncated_dense[l + 1, m + 1] = 0
            end
            truncated_packed = SHTnsKit.pack_lm(cfg, truncated_dense)
            high_noise = copy(packed)
            for m in 0:cfg.mmax
                m % cfg.mres == 0 || continue
                for l in max(m, ltr + 1):cfg.lmax
                    high_noise[LM_index(cfg.lmax, cfg.mres, l, m) + 1] =
                        Complex{T}(T(100 + l), T(-70 - m))
                end
            end
            low_field = synthesis_packed_l(cfg, high_noise, ltr)
            @test low_field ≈ synthesis_packed(cfg, truncated_packed) atol=10eps(T) rtol=2e-5
            analyzed = analysis_packed_l(cfg, low_field, ltr)
            @test analyzed ≈ truncated_packed atol=30eps(T) rtol=2e-4
            for m in 0:cfg.mmax
                m % cfg.mres == 0 || continue
                for l in max(m, ltr + 1):cfg.lmax
                    @test iszero(analyzed[LM_index(cfg.lmax, cfg.mres, l, m) + 1])
                end
            end
        end
    end

    @testset "typed CPU variant entry points preserve inferred APIs" begin
        cfg = create_gauss_config(3, 6; nlon=10)
        dense = _variant_coefficients(cfg, Float64)
        packed = SHTnsKit.pack_lm(cfg, dense)
        field = synthesis_packed(cfg, packed)
        @test analysis_packed(SHTnsKit.CPU(), cfg, field) ≈ analysis_packed(cfg, field)
        @test synthesis_packed(SHTnsKit.CPU(), cfg, packed) ≈ synthesis_packed(cfg, packed)
        @test analysis_packed_l(SHTnsKit.CPU(), cfg, field, 2) ≈ analysis_packed_l(cfg, field, 2)
        @test synthesis_packed_l(SHTnsKit.CPU(), cfg, packed, 2) ≈ synthesis_packed_l(cfg, packed, 2)
        axis = synthesis_axisym(cfg, dense[:, 1])
        @test analysis_axisym(SHTnsKit.CPU(), cfg, axis) ≈ analysis_axisym(cfg, axis)
        @test synthesis_axisym(SHTnsKit.CPU(), cfg, dense[:, 1]) ≈ synthesis_axisym(cfg, dense[:, 1])
        @test analysis_packed_ml(SHTnsKit.CPU(), cfg, 2, synthesis_packed_ml(cfg, 2, dense[3:4, 3], 3), 3) ≈ dense[3:4, 3]
        @test synthesis_packed_ml(SHTnsKit.CPU(), cfg, 2, dense[3:4, 3], 3) ≈
              synthesis_packed_ml(cfg, 2, dense[3:4, 3], 3)

        batch_field = reshape(synthesis(cfg, dense), cfg.nlat, cfg.nlon, 1)
        batch_coefficients = reshape(dense, cfg.lmax + 1, cfg.mmax + 1, 1)
        @test analysis_batch(SHTnsKit.CPU(), cfg, batch_field) ≈ analysis_batch(cfg, batch_field)
        @test synthesis_batch(SHTnsKit.CPU(), cfg, batch_coefficients) ≈
              synthesis_batch(cfg, batch_coefficients)
        @test synthesis_batch_cplx(SHTnsKit.CPU(), cfg, batch_coefficients) ≈
              synthesis_batch_cplx(cfg, batch_coefficients)

        cpacked = zeros(ComplexF64, nlm_cplx_calc(cfg.lmax, cfg.mmax, 1))
        cpacked[LM_cplx_index(cfg.lmax, cfg.mmax, 2, -1) + 1] = 0.2 - 0.1im
        cfield = synthesis_packed_cplx(cfg, cpacked)
        @test synthesis_packed_cplx(SHTnsKit.CPU(), cfg, cpacked) ≈ cfield
        @test analysis_packed_cplx(SHTnsKit.CPU(), cfg, cfield) ≈ cpacked atol=2e-11 rtol=2e-11

        @test_throws ArgumentError analysis_packed(
            SHTnsKit.CPU(), cfg, VariantNonCPUArray(field),
        )
        @test_throws ArgumentError synthesis_packed(
            SHTnsKit.CPU(), cfg, VariantNonCPUArray(packed),
        )
        @test_throws ArgumentError analysis_packed_cplx(
            SHTnsKit.CPU(), cfg, VariantNonCPUArray(cfield),
        )
        @test_throws ArgumentError analysis_batch(
            SHTnsKit.CPU(), cfg, VariantNonCPUArray(batch_field),
        )
    end

    @testset "all stored fixed orders use physical m" begin
        for T in (Float32, Float64)
            cfg = create_gauss_config(
                6, 9; nlon=16, mres=2, norm=:fourpi,
                real_norm=true, cs_phase=false,
            )
            physical_cfg = create_gauss_config(
                6, 9; nlon=16, mres=1, norm=:fourpi,
                real_norm=true, cs_phase=false,
            )
            tol = T === Float32 ? 2f-5 : 2e-11
            for im in 0:(cfg.mmax ÷ cfg.mres)
                m = im * cfg.mres
                coefficients = Complex{T}[
                    Complex{T}(T(0.04 * (l + 1)), T(m == 0 ? 0 : -0.03 * (l + 1)))
                    for l in m:cfg.lmax
                ]
                got = synthesis_packed_ml(cfg, im, coefficients, cfg.lmax)
                ref = synthesis_packed_ml(physical_cfg, m, coefficients, cfg.lmax)
                @test got ≈ ref atol=tol rtol=tol
                @test analysis_packed_ml(cfg, im, got, cfg.lmax) ≈
                      coefficients atol=tol rtol=tol
            end
        end
    end

    @testset "axisymmetric and complex packed paths" begin
        cfg = create_gauss_config(
            4, 7; nlon=12, norm=:schmidt, real_norm=true, cs_phase=false,
        )
        axis_coefficients = ComplexF32[0.2, -0.1, 0.05, 0.03, -0.02]
        axis_field = synthesis_axisym(cfg, axis_coefficients)
        @test synthesis_axisym_l(cfg, axis_coefficients, 2) ≈
              synthesis_axisym(cfg, [axis_coefficients[1:3]; zeros(ComplexF32, 2)])
        @test analysis_axisym(cfg, axis_field) ≈ axis_coefficients atol=2f-5 rtol=2f-5

        cpacked = zeros(ComplexF32, nlm_cplx_calc(cfg.lmax, cfg.mmax, 1))
        for l in 0:cfg.lmax, m in -min(l, cfg.mmax):min(l, cfg.mmax)
            cpacked[LM_cplx_index(cfg.lmax, cfg.mmax, l, m) + 1] =
                ComplexF32(0.02f0 * (l + 1), 0.01f0 * (m - l))
        end
        complex_field = synthesis_packed_cplx(cfg, cpacked)
        @test eltype(complex_field) === ComplexF32
        @test eltype(analysis_packed_cplx(cfg, complex_field)) === ComplexF32
        @test analysis_packed_cplx(cfg, complex_field) ≈ cpacked atol=3f-5 rtol=3f-5
    end

    @testset "plan and scalar batches preserve boundary semantics" begin
        for T in (Float32, Float64), nfields in (1, 2, 5)
            cfg = create_gauss_config(
                4, 7; nlon=12, norm=:schmidt, real_norm=true, cs_phase=false,
            )
            coefficients = _variant_coefficients(cfg, T)
            reference = synthesis(cfg, coefficients)
            fields = repeat(reshape(reference, cfg.nlat, cfg.nlon, 1), 1, 1, nfields)
            for k in 1:nfields
                fields[:, :, k] .*= T(k)
            end
            batch = analysis_batch(cfg, fields)
            @test eltype(batch) === Complex{T}
            @test size(batch) == (cfg.lmax + 1, cfg.mmax + 1, nfields)
            for k in 1:nfields
                @test batch[:, :, k] ≈ T(k) .* coefficients atol=30eps(T) rtol=2e-4
            end
            reconstructed = synthesis_batch(cfg, batch)
            @test eltype(reconstructed) === T
            @test reconstructed ≈ fields atol=30eps(T) rtol=2e-4

            analysis_out = similar(batch)
            analysis_scratch = zeros(Complex{T}, cfg.nlat, cfg.nlon, nfields)
            analysis_batch!(cfg, analysis_out, fields; fft_batch=analysis_scratch)
            @test analysis_out ≈ batch atol=30eps(T) rtol=2e-4
            synthesis_out = similar(fields)
            synthesis_scratch = similar(analysis_scratch)
            synthesis_batch!(cfg, synthesis_out, batch; fft_batch=synthesis_scratch)
            @test synthesis_out ≈ fields atol=30eps(T) rtol=2e-4
        end

        cfg = create_gauss_config(
            4, 7; nlon=12, norm=:schmidt, real_norm=true, cs_phase=false,
        )
        coefficients = _variant_coefficients(cfg, Float64)
        reference = synthesis(cfg, coefficients)
        plan = SHTPlan(cfg)
        planned = similar(reference)
        synthesis!(plan, planned, coefficients)
        @test planned ≈ reference atol=2e-12 rtol=2e-12
        recovered = similar(coefficients)
        analysis!(plan, recovered, planned)
        @test recovered ≈ coefficients atol=2e-11 rtol=2e-11

        # Legal shared-parent views must be read completely before overlapping
        # output is mutated.
        storage = zeros(ComplexF64, cfg.nlat, cfg.nlon + cfg.mmax + 1)
        field_view = @view storage[:, 1:cfg.nlon]
        field_view .= reference
        coefficient_view = @view storage[1:(cfg.lmax + 1), 1:(cfg.mmax + 1)]
        analysis!(plan, coefficient_view, field_view)
        @test coefficient_view ≈ coefficients atol=2e-11 rtol=2e-11

        # Caller-provided batch scratch must not overlap an input or output it
        # services: reject before the first mutation instead of silently
        # corrupting coefficients/Fourier bins. Direct output/input overlap is
        # safe when the independent scratch owns the complete intermediate.
        batch_fields = reshape(reference, cfg.nlat, cfg.nlon, 1)
        batch_coefficients = reshape(coefficients, cfg.lmax + 1, cfg.mmax + 1, 1)
        shared_batch = zeros(ComplexF64, cfg.nlat, cfg.nlon, 1)
        shared_coefficients = @view shared_batch[
            1:(cfg.lmax + 1), 1:(cfg.mmax + 1), :,
        ]
        shared_coefficients .= batch_coefficients
        @test_throws ArgumentError analysis_batch!(
            cfg, shared_coefficients, batch_fields; fft_batch=shared_batch,
        )
        @test_throws ArgumentError synthesis_batch!(
            cfg, zeros(size(batch_fields)), shared_coefficients;
            fft_batch=shared_batch,
        )

        shared_coefficients .= batch_coefficients
        independent_scratch = similar(shared_batch)
        expected_complex = synthesis_batch_cplx(cfg, batch_coefficients)
        synthesis_batch!(
            cfg, shared_batch, shared_coefficients;
            real_output=false, fft_batch=independent_scratch,
        )
        @test shared_batch ≈ expected_complex atol=2e-12 rtol=2e-12
    end

    @testset "plan and batch honor the stored-order stride" begin
        cfg = create_gauss_config(5, 8; nlon=14, mres=2)
        rng = MersenneTwister(0x6a17)
        field = randn(rng, cfg.nlat, cfg.nlon)
        reference_coefficients = analysis(cfg, field)
        batch_coefficients = analysis_batch(cfg, reshape(field, cfg.nlat, cfg.nlon, 1))
        @test batch_coefficients[:, :, 1] ≈ reference_coefficients atol=2e-11 rtol=2e-11

        plan = SHTPlan(cfg)
        planned_coefficients = similar(reference_coefficients)
        analysis!(plan, planned_coefficients, field)
        @test planned_coefficients ≈ reference_coefficients atol=2e-11 rtol=2e-11

        invalid_order_coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        invalid_order_coefficients[4, 2] = 0.7 - 0.2im
        reference_field = synthesis(cfg, invalid_order_coefficients)
        @test iszero(maximum(abs, reference_field))
        @test synthesis_batch(
            cfg, reshape(invalid_order_coefficients, cfg.lmax + 1, cfg.mmax + 1, 1),
        )[:, :, 1] ≈ reference_field atol=2e-12 rtol=2e-12
        planned_field = similar(reference_field)
        synthesis!(plan, planned_field, invalid_order_coefficients)
        @test planned_field ≈ reference_field atol=2e-12 rtol=2e-12
    end
end
