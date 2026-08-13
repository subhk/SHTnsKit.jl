# SHTnsKit.jl - Rotation AD-adjoint gradient tests
#
# Validates the reverse-mode adjoints (ChainRules rrules in SHTnsKitAdvancedADExt
# and the Zygote @adjoints in SHTnsKitZygoteExt) for the packed real-field
# rotations against finite differences.
#
# Regression guard: the Qlm-gradients were previously WRONG. SH_Zrotate/SH_Yrotate
# conjugated / mis-weighted the cotangent, and SH_Xrotate90 used non-inverse ZYZ
# angles. The correct standard-inner-product adjoint of a packed rotation R is
#   Q̄ = W · R⁻¹ · (W⁻¹ ȳ),   W = diag(wm),  wm = 2 for m>0, 1 for m=0
# because the m>0 packed modes carry double weight in the physical field inner
# product while Zygote/ChainRules use the unweighted packed inner product.
# The angle (dα) gradients were already correct; they are checked here too.

using Test
using Random
using SHTnsKit
using ChainRulesCore

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

const _HAS_ZYGOTE_ROT = try
    @eval using Zygote
    true
catch
    false
end

@testset "Complex-packed analysis rrule respects configured convention" begin
    lmax = 4
    cfg = create_gauss_config(lmax, 7; nlon=11, norm=:schmidt,
                              real_norm=true, cs_phase=false)
    rng = MersenneTwister(8491)
    z = randn(rng, ComplexF64, cfg.nlat, cfg.nlon)
    h = randn(rng, ComplexF64, size(z))
    cotangent = randn(rng, ComplexF64, nlm_cplx_calc(lmax, lmax, 1))

    _, pullback = ChainRulesCore.rrule(analysis_packed_cplx, cfg, z)
    _, _, zbar = pullback(cotangent)
    loss(zv) = real(sum(conj(cotangent) .* analysis_packed_cplx(cfg, zv)))
    epsilon = 1e-6
    fd = (loss(z .+ epsilon .* h) - loss(z .- epsilon .* h)) / (2epsilon)
    ad = real(sum(conj(zbar) .* h))
    @test isapprox(ad, fd; rtol=2e-6, atol=2e-8)
end

@testset "Setter rotation rrules use effective Euler angles" begin
    lmax = 4
    cfg = create_gauss_config(lmax, 7; nlon=11)
    rng = MersenneTwister(0xAD37)
    real_input = randn(rng, ComplexF64, cfg.nlm)
    real_cotangent = randn(rng, ComplexF64, cfg.nlm)
    for index in eachindex(real_input)
        cfg.mi[index] == 0 && (real_input[index] = real(real_input[index]))
    end
    complex_input = randn(rng, ComplexF64, nlm_cplx_calc(lmax, lmax, 1))
    complex_cotangent = randn(rng, ComplexF64, length(complex_input))
    angles = (0.23, 0.41, -0.17)
    epsilon = 1e-6

    function setter_rotation(convention, values)
        rotation = SHTRotation(lmax, lmax)
        if convention === :ZYZ
            shtns_rotation_set_angles_ZYZ(rotation, values...)
        else
            shtns_rotation_set_angles_ZXZ(rotation, values...)
        end
        return rotation
    end

    function equivalent_direct(convention, values)
        alpha, beta, gamma = values
        effective = convention === :ZYZ ?
            (gamma, beta, alpha) :
            (gamma - pi / 2, beta, alpha + pi / 2)
        return SHTRotation(lmax, lmax; α=effective[1], β=effective[2], γ=effective[3])
    end

    function primal_and_pullback(apply, rotation, input, cotangent)
        output = similar(input)
        primal, pullback = ChainRulesCore.rrule(apply, rotation, input, output)
        _, rotation_bar, input_bar, _ = pullback(cotangent)
        return copy(primal), rotation_bar, input_bar
    end

    for (name, apply, input, cotangent) in (
        ("real", shtns_rotation_apply_real, real_input, real_cotangent),
        ("complex", shtns_rotation_apply_cplx, complex_input, complex_cotangent),
    ), convention in (:ZYZ, :ZXZ)
        @testset "$name $convention" begin
            setter = setter_rotation(convention, angles)
            direct = equivalent_direct(convention, angles)
            setter_y, setter_rbar, setter_input_bar =
                primal_and_pullback(apply, setter, input, cotangent)
            direct_y, direct_rbar, direct_input_bar =
                primal_and_pullback(apply, direct, input, cotangent)

            @test setter_y ≈ direct_y atol=2e-12 rtol=2e-12
            @test setter_input_bar ≈ direct_input_bar atol=2e-12 rtol=2e-12
            @test [setter_rbar.α, setter_rbar.β, setter_rbar.γ] ≈
                  [direct_rbar.γ, direct_rbar.β, direct_rbar.α] atol=2e-12 rtol=2e-12

            loss(values) = begin
                rotation = setter_rotation(convention, values)
                output = apply(rotation, input, similar(input))
                real(sum(conj(cotangent) .* output))
            end
            finite_difference(index) = begin
                plus = ntuple(i -> angles[i] + (i == index ? epsilon : 0.0), 3)
                minus = ntuple(i -> angles[i] - (i == index ? epsilon : 0.0), 3)
                (loss(plus) - loss(minus)) / (2epsilon)
            end
            stored_tangent = (setter_rbar.α, setter_rbar.β, setter_rbar.γ)
            for index in 1:3
                @test stored_tangent[index] ≈ finite_difference(index) rtol=2e-5 atol=2e-7
            end
        end
    end

    @testset "Direct and angle-axis rotations retain canonical rrules" begin
        angle_axis = SHTRotation(lmax, lmax)
        shtns_rotation_set_angle_axis(angle_axis, 0.37, 0.2, -0.4, 0.7)
        direct = SHTRotation(lmax, lmax; α=angles[1], β=angles[2], γ=angles[3])
        real_direction = randn(rng, ComplexF64, cfg.nlm)
        for index in eachindex(real_direction)
            cfg.mi[index] == 0 && (real_direction[index] = real(real_direction[index]))
        end
        complex_direction = randn(rng, ComplexF64, length(complex_input))

        for (name, apply, input, cotangent, direction) in (
            ("real", shtns_rotation_apply_real, real_input, real_cotangent,
             real_direction),
            ("complex", shtns_rotation_apply_cplx, complex_input,
             complex_cotangent, complex_direction),
        ), (origin, rotation) in (("direct", direct), ("angle-axis", angle_axis))
            @testset "$name $origin" begin
                _, rotation_bar, input_bar =
                    primal_and_pullback(apply, rotation, input, cotangent)
                coefficient_loss(candidate) = real(sum(
                    conj(cotangent) .* apply(rotation, candidate, similar(candidate)),
                ))
                coefficient_fd = (
                    coefficient_loss(input .+ epsilon .* direction) -
                    coefficient_loss(input .- epsilon .* direction)
                ) / (2epsilon)
                coefficient_ad = real(sum(conj(input_bar) .* direction))
                @test coefficient_ad ≈ coefficient_fd rtol=2e-5 atol=2e-7

                stored = (rotation.α, rotation.β, rotation.γ)
                stored_tangent = (rotation_bar.α, rotation_bar.β, rotation_bar.γ)
                angle_loss(values) = begin
                    candidate = SHTRotation(
                        lmax, lmax; α=values[1], β=values[2], γ=values[3],
                    )
                    real(sum(conj(cotangent) .* apply(
                        candidate, input, similar(input),
                    )))
                end
                for index in 1:3
                    plus = ntuple(i -> stored[i] + (i == index ? epsilon : 0.0), 3)
                    minus = ntuple(i -> stored[i] - (i == index ? epsilon : 0.0), 3)
                    fd = (angle_loss(plus) - angle_loss(minus)) / (2epsilon)
                    @test stored_tangent[index] ≈ fd rtol=2e-5 atol=2e-7
                end
            end
        end
    end
end

if _HAS_ZYGOTE_ROT
@testset "Rotation AD adjoints vs finite differences" begin
    # Real-field-compatible packed vector: m=0 entries must be real.
    function _rfvec(rng, cfg)
        v = randn(rng, ComplexF64, cfg.nlm)
        @inbounds for k in 1:cfg.nlm
            cfg.mi[k] == 0 && (v[k] = real(v[k]))
        end
        return v
    end

    for lmax in (5, 8)
        nlat = lmax + 2
        nlon = 2 * lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(4242 + lmax)

        Q = _rfvec(rng, cfg)
        C = _rfvec(rng, cfg)   # fixed target -> phase-sensitive linear loss
        h = _rfvec(rng, cfg)   # FD direction (real-field-compatible)
        alpha = 0.7
        ϵ = 1e-6

        # ---- gradient w.r.t. Qlm ----
        # loss(Q) = real(Σ conj(C) · rot(Q)); Zygote convention:
        # L(Q+ϵh) ≈ L(Q) + ϵ Re(Σ conj(g) · h)
        function check_dQ(name, rot)
            loss(q) = real(sum(conj(C) .* rot(q)))
            g = Zygote.gradient(loss, Q)[1]
            @test g !== nothing
            dL_ad = real(sum(conj(g) .* h))
            dL_fd = (loss(Q .+ ϵ .* h) - loss(Q .- ϵ .* h)) / (2ϵ)
            VERBOSE && @info "rotation dQ" name dL_ad dL_fd
            @test isapprox(dL_ad, dL_fd; rtol=1e-4, atol=1e-8)
        end

        check_dQ("SH_Zrotate",   q -> SH_Zrotate(cfg, q, alpha, similar(q)))
        check_dQ("SH_Yrotate",   q -> SH_Yrotate(cfg, q, alpha, similar(q)))
        check_dQ("SH_Yrotate90", q -> SH_Yrotate90(cfg, q, similar(q)))
        check_dQ("SH_Xrotate90", q -> SH_Xrotate90(cfg, q, similar(q)))

        # ---- gradient w.r.t. rotation angle α ----
        function check_dα(name, rot)
            lossα(a) = real(sum(conj(C) .* rot(Q, a)))
            gα = Zygote.gradient(lossα, alpha)[1]
            @test gα !== nothing
            fd = (lossα(alpha + ϵ) - lossα(alpha - ϵ)) / (2ϵ)
            VERBOSE && @info "rotation dα" name gα fd
            @test isapprox(gα, fd; rtol=1e-4, atol=1e-8)
        end

        check_dα("SH_Zrotate α", (q, a) -> SH_Zrotate(cfg, q, a, similar(q)))
        check_dα("SH_Yrotate α", (q, a) -> SH_Yrotate(cfg, q, a, similar(q)))
    end
end
else
    @info "Skipping rotation-adjoint FD check (Zygote not available in this test context)"
end
