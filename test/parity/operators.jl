using Test
using Random
using SHTnsKit

function _operator_cfg(::Type{T}=Float64; mres=1, norm=:orthonormal,
                       real_norm=false, cs_phase=true) where {T<:AbstractFloat}
    return create_gauss_config(
        6, 9; nlon=16, mres, norm, real_norm, cs_phase,
    )
end

@inline function _ct_base(::Type{T}, l, m, direction) where {T}
    if direction === :down
        l == 0 && return zero(T)
        return sqrt(max(zero(T), T(l*l - m*m) / T((2l - 1) * (2l + 1))))
    end
    return sqrt(max(zero(T), T((l + 1)^2 - m*m) / T((2l + 1) * (2l + 3))))
end

function _operator_oracle(cfg, mx, input)
    CT = promote_type(eltype(input), complex(eltype(mx)))
    output = zeros(CT, cfg.nlm)
    for k in eachindex(input)
        l = cfg.li[k]; m = cfg.mi[k]
        if l > m
            below = LM_index(cfg.lmax, cfg.mres, l - 1, m) + 1
            output[k] += mx[2below] * input[below]
        end
        if l < cfg.lmax
            above = LM_index(cfg.lmax, cfg.mres, l + 1, m) + 1
            output[k] += mx[2above - 1] * input[above]
        end
    end
    return output
end

function test_cpu_operator_parity()
    @testset "CPU spectral operator parity" begin
        for T in (Float32, Float64), mres in (1, 2), convention in (
            (norm=:orthonormal, real_norm=false, cs_phase=true),
            (norm=:fourpi, real_norm=true, cs_phase=false),
            (norm=:schmidt, real_norm=true, cs_phase=false),
        )
            cfg = _operator_cfg(T; mres, convention...)
            tol = T === Float32 ? T(8e-6) : T(2e-14)
            ct = fill(T(NaN), 2cfg.nlm)
            dt = similar(ct)
            @test mul_ct_matrix(CPU(), cfg, ct) === ct
            @test st_dt_matrix(CPU(), cfg, dt) === dt
            @test eltype(ct) === T

            for k in 1:cfg.nlm
                l = cfg.li[k]; m = cfg.mi[k]
                source_scale = T(SHTnsKit.coefficient_scale_to_canonical(cfg, l, m))
                down = l == m ? zero(T) : _ct_base(T, l, m, :down) * source_scale /
                    T(SHTnsKit.coefficient_scale_to_canonical(cfg, l - 1, m))
                up = l == cfg.lmax ? zero(T) : _ct_base(T, l, m, :up) * source_scale /
                    T(SHTnsKit.coefficient_scale_to_canonical(cfg, l + 1, m))
                @test ct[2k - 1] ≈ down atol=tol rtol=tol
                @test ct[2k] ≈ up atol=tol rtol=tol
                @test dt[2k - 1] ≈ -(l + 1) * down atol=tol rtol=tol
                @test dt[2k] ≈ l * up atol=tol rtol=tol
            end

            rng = MersenneTwister(0x511 + mres)
            input = randn(rng, Complex{T}, cfg.nlm)
            expected = _operator_oracle(cfg, ct, input)
            output = fill(Complex{T}(91, -17), cfg.nlm)
            @test SH_mul_mx(CPU(), cfg, ct, input, output) === output
            @test output ≈ expected atol=tol rtol=tol
            inferred = similar(output)
            @test SH_mul_mx(cfg, ct, input, inferred) === inferred
            @test inferred == output

            # Every output is validated before mutation, and all overlapping
            # input/output views obey the documented no-alias contract.
            sentinel = fill(Complex{T}(13, -7), cfg.nlm)
            @test_throws DimensionMismatch SH_mul_mx(
                CPU(), cfg, ct[1:end-1], input, sentinel,
            )
            @test all(==(Complex{T}(13, -7)), sentinel)
            storage = randn(rng, Complex{T}, cfg.nlm + 1)
            source = @view storage[1:cfg.nlm]
            destination = @view storage[2:(cfg.nlm + 1)]
            before = copy(storage)
            @test_throws ArgumentError SH_mul_mx(
                CPU(), cfg, ct, source, destination,
            )
            @test storage == before
        end

        # Dense diagonal operators are convention invariant, validate the
        # configured shape, filter invalid storage, and support true in-place
        # operation (including overlapping views) without reading zeros that
        # the destination fill wrote first.
        cfg = _operator_cfg(Float32; mres=2, norm=:schmidt,
                            real_norm=true, cs_phase=false)
        rng = MersenneTwister(0xD1A6)
        input = randn(rng, ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
        reference = divergence_from_spheroidal(cfg, input)
        @test divergence_from_spheroidal(SHTnsKit.CPU(), cfg, input) == reference
        @test vorticity_from_toroidal(SHTnsKit.CPU(), cfg, input) == reference
        @test spheroidal_from_divergence(
            SHTnsKit.CPU(), cfg, reference,
        ) == spheroidal_from_divergence(cfg, reference)
        @test toroidal_from_vorticity(
            SHTnsKit.CPU(), cfg, reference,
        ) == toroidal_from_vorticity(cfg, reference)
        inplace = copy(input)
        @test divergence_from_spheroidal!(cfg, inplace, inplace) === inplace
        @test inplace == reference
        @test all(iszero, reference[:, 2])
        @test iszero(reference[1, 1])
        recovered = copy(reference)
        spheroidal_from_divergence!(cfg, recovered, recovered)
        expected = copy(input)
        for m in 0:cfg.mmax, l in 0:cfg.lmax
            if m % cfg.mres != 0 || l < max(1, m)
                expected[l + 1, m + 1] = 0
            end
        end
        @test recovered ≈ expected
        wrong = fill(ComplexF32(9, -4), cfg.lmax + 2, cfg.mmax + 1)
        before = copy(wrong)
        @test_throws DimensionMismatch divergence_from_spheroidal!(cfg, wrong, wrong)
        @test wrong == before

        # The dense distributed compatibility bindings share the exact same
        # invalid-storage and no-alias contracts as their packed/Pencil paths.
        dense_lap = copy(input)
        @test SHTnsKit.dist_apply_laplacian!(cfg, dense_lap) === dense_lap
        @test dense_lap == reference
        dense_output = fill(ComplexF32(91, -17), size(input))
        packed_mx = zeros(Float32, 2cfg.nlm)
        mul_ct_matrix(cfg, packed_mx)
        @test SHTnsKit.dist_SH_mul_mx!(
            cfg, packed_mx, input, dense_output,
        ) === dense_output
        packed_input = SHTnsKit.pack_lm(cfg, input)
        packed_output = similar(packed_input)
        SH_mul_mx(cfg, packed_mx, packed_input, packed_output)
        @test dense_output == SHTnsKit.unpack_lm(cfg, packed_output)
        alias_before = copy(input)
        @test_throws ArgumentError SHTnsKit.dist_SH_mul_mx!(
            cfg, packed_mx, alias_before, alias_before,
        )
        @test alias_before == input

        grad = synthesis_grad(cfg, input)
        @test grad == synthesis_sph(cfg, input)

        zero_cfg = create_gauss_config(0, 1; nlon=1)
        zero_mx = fill(Float32(NaN), 2zero_cfg.nlm)
        mul_ct_matrix(SHTnsKit.CPU(), zero_cfg, zero_mx)
        @test all(iszero, zero_mx)
        zero_input = ComplexF32[3 + 2im]
        zero_output = similar(zero_input)
        SH_mul_mx(SHTnsKit.CPU(), zero_cfg, zero_mx, zero_input, zero_output)
        @test all(iszero, zero_output)
    end
end

function run_shared_operator_kernel_reference(common, backend)
    @testset "shared operator kernels" begin
      for T in (Float32, Float64)
        cfg = _operator_cfg(T; mres=2, norm=:schmidt,
                            real_norm=true, cs_phase=false)
        tol = T === Float32 ? T(8e-6) : T(2e-14)
        down_ratios = ones(T, cfg.nlm)
        up_ratios = ones(T, cfg.nlm)
        for k in 1:cfg.nlm
            l = cfg.li[k]; m = cfg.mi[k]
            source = SHTnsKit.coefficient_scale_to_canonical(cfg, l, m)
            l > m && (down_ratios[k] = source /
                SHTnsKit.coefficient_scale_to_canonical(cfg, l - 1, m))
            l < cfg.lmax && (up_ratios[k] = source /
                SHTnsKit.coefficient_scale_to_canonical(cfg, l + 1, m))
        end
        li = Int32.(cfg.li)
        mi = Int32.(cfg.mi)
        ct = fill(T(NaN), 2cfg.nlm)
        event = common.operator_matrix_kernel!(backend)(
            ct, li, mi, down_ratios, up_ratios, cfg.lmax, false;
            ndrange=cfg.nlm,
        )
        event === nothing || wait(event)
        expected_ct = zeros(T, 2cfg.nlm)
        mul_ct_matrix(SHTnsKit.CPU(), cfg, expected_ct)
        @test ct ≈ expected_ct atol=tol rtol=tol

        lower = zeros(Int32, cfg.nlm)
        upper = zeros(Int32, cfg.nlm)
        for k in 1:cfg.nlm
            l = cfg.li[k]; m = cfg.mi[k]
            l > m && (lower[k] = LM_index(cfg.lmax, cfg.mres, l - 1, m) + 1)
            l < cfg.lmax && (upper[k] = LM_index(cfg.lmax, cfg.mres, l + 1, m) + 1)
        end
        CT = Complex{T}
        input = CT[CT(T(k / 11), T(-k / 17)) for k in 1:cfg.nlm]
        output = fill(CT(91, -17), cfg.nlm)
        event = common.packed_operator_kernel!(backend)(
            output, input, ct, lower, upper; ndrange=cfg.nlm,
        )
        event === nothing || wait(event)
        @test output ≈ _operator_oracle(cfg, ct, input) atol=tol rtol=tol

        dense = reshape(
            CT.(1:((cfg.lmax + 1) * (cfg.mmax + 1))),
            cfg.lmax + 1, cfg.mmax + 1,
        )
        lap = fill(CT(91, -17), size(dense))
        event = common.laplacian_kernel!(backend)(
            lap, dense, cfg.lmax, cfg.mmax, cfg.mres; ndrange=size(dense),
        )
        event === nothing || wait(event)
        @test lap == divergence_from_spheroidal(cfg, dense)
      end
    end
end

abstract type GPUOperatorAdapter end
operator_place(::GPUOperatorAdapter, value) = error("operator_place not implemented")
operator_strided_place(::GPUOperatorAdapter, value) = error("operator_strided_place not implemented")
operator_overlapping_matrix_place(::GPUOperatorAdapter, value) =
    error("operator_overlapping_matrix_place not implemented")
operator_collect(::GPUOperatorAdapter, value) = error("operator_collect not implemented")
operator_resident(::GPUOperatorAdapter, value) = error("operator_resident not implemented")

function run_gpu_operator_parity(adapter::GPUOperatorAdapter)
    @testset "device-resident operator parity" begin
      for T in (Float32, Float64)
        cfg = _operator_cfg(T; mres=2, norm=:schmidt,
                            real_norm=true, cs_phase=false)
        tol = T === Float32 ? T(8e-6) : T(2e-14)
        CT = Complex{T}
        host_ct = zeros(T, 2cfg.nlm)
        mul_ct_matrix(SHTnsKit.CPU(), cfg, host_ct)
        device_ct = operator_strided_place(adapter, fill(T(NaN), 2cfg.nlm))
        @test mul_ct_matrix(SHTnsKit.GPU(), cfg, device_ct) === device_ct
        operator_resident(adapter, device_ct)
        @test eltype(device_ct) === T
        @test operator_collect(adapter, device_ct) ≈ host_ct atol=tol rtol=tol
        fill!(device_ct, T(NaN))
        @test mul_ct_matrix(cfg, device_ct) === device_ct

        input = CT[CT(T(k / 11), T(-k / 17)) for k in 1:cfg.nlm]
        device_input = operator_strided_place(adapter, input)
        device_output = operator_strided_place(
            adapter, fill(CT(91, -17), cfg.nlm),
        )
        @test SH_mul_mx(
            SHTnsKit.GPU(), cfg, device_ct, device_input, device_output,
        ) === device_output
        operator_resident(adapter, device_output)
        @test operator_collect(adapter, device_output) ≈
              _operator_oracle(cfg, host_ct, input) atol=tol rtol=tol
        @test_throws ArgumentError SH_mul_mx(
            SHTnsKit.GPU(), cfg, device_ct, device_input, device_input,
        )

        dense = reshape(
            CT.(1:((cfg.lmax + 1) * (cfg.mmax + 1))),
            cfg.lmax + 1, cfg.mmax + 1,
        )
        device_dense = operator_place(adapter, dense)
        @test gpu_apply_laplacian!(cfg, device_dense) === device_dense
        operator_resident(adapter, device_dense)
        @test operator_collect(adapter, device_dense) ==
              divergence_from_spheroidal(cfg, dense)
        placed_dense = operator_place(adapter, dense)
        typed_div = divergence_from_spheroidal(
            SHTnsKit.GPU(), cfg, placed_dense,
        )
        operator_resident(adapter, typed_div)
        @test eltype(typed_div) === CT
        @test operator_collect(adapter, typed_div) ==
              divergence_from_spheroidal(cfg, dense)
        typed_back = spheroidal_from_divergence(
            SHTnsKit.GPU(), cfg, typed_div,
        )
        @test operator_collect(adapter, typed_back) ==
              spheroidal_from_divergence(
                  cfg, divergence_from_spheroidal(cfg, dense),
              )

        source_view, destination_view =
            operator_overlapping_matrix_place(adapter, dense)
        @test divergence_from_spheroidal!(
            SHTnsKit.GPU(), cfg, destination_view, source_view,
        ) === destination_view
        @test operator_collect(adapter, destination_view) ==
              divergence_from_spheroidal(cfg, dense)
      end
    end
end
