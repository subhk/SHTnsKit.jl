using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

# These expectations are the SHTns convention itself, not values obtained from
# the conversion implementation under test.  SHTns stores only m >= 0 for real
# fields, so sin(theta)cos(phi) has canonical a[1,1] = -sqrt(2pi/3) when the
# Condon--Shortley phase is enabled.
function _shtns_coefficient_scale_to_canonical(norm::Symbol, real_norm::Bool,
                                                cs_phase::Bool, l::Int, m::Int,
                                                ::Type{T}=Float64) where {T<:AbstractFloat}
    norm_scale = norm === :orthonormal ? one(T) :
                 norm === :fourpi ? sqrt(T(4pi)) :
                 norm === :schmidt ? sqrt(T(4pi / (2l + 1))) :
                 error("unsupported test normalization")
    # SHTns REAL_NORM stores m>0 coefficients sqrt(2) larger.  Equivalently,
    # the multiplier from configured coefficients to canonical coefficients is
    # 1/sqrt(2).  This is pinned by shlm_e1 and mpos_renorm in SHTns 3.7.
    real_scale = real_norm && m > 0 ? inv(sqrt(T(2))) : one(T)
    phase_scale = cs_phase || iseven(m) ? one(T) : -one(T)
    return norm_scale * real_scale * phase_scale
end

function _external_coeff(canonical, norm, real_norm, cs_phase, l, m)
    T = typeof(real(canonical))
    return canonical / _shtns_coefficient_scale_to_canonical(norm, real_norm,
                                                              cs_phase, l, m, T)
end

function _single_mode(::Type{T}, lmax, l, m, value) where {T<:AbstractFloat}
    a = zeros(Complex{T}, lmax + 1, lmax + 1)
    a[l + 1, m + 1] = value
    return a
end

@testset "SHTns normalization, real normalization, and phase parity" begin
    for T in (Float32, Float64)
        atol = T === Float32 ? 2f-5 : 2e-12
        rtol = T === Float32 ? 2f-5 : 2e-12
        lmax = 3
        nlat = 6
        nlon = 9

        theta_field(cfg) = T.(reshape(cfg.x, :, 1)) .* ones(T, 1, cfg.nlon)
        function equatorial_field(cfg)
            f = Matrix{T}(undef, cfg.nlat, cfg.nlon)
            for j in 1:cfg.nlon, i in 1:cfg.nlat
                phi = T(2pi * (j - 1) / cfg.nlon)
                f[i, j] = sqrt(max(zero(T), one(T) - T(cfg.x[i])^2)) * cos(phi)
            end
            return f
        end

        for norm in (:orthonormal, :fourpi, :schmidt),
            real_norm in (false, true), cs_phase in (false, true)
            cfg = create_gauss_config(lmax, nlat; nlon, norm, real_norm, cs_phase)

            fields = (
                (ones(T, cfg.nlat, cfg.nlon), 0, 0, sqrt(T(4pi))),
                (theta_field(cfg), 1, 0, sqrt(T(4pi / 3))),
                (equatorial_field(cfg), 1, 1, -sqrt(T(2pi / 3))),
            )

            for (field, l, m, canonical) in fields
                expected = _external_coeff(canonical, norm, real_norm, cs_phase, l, m)
                alm = analysis(cfg, field)
                @test alm[l + 1, m + 1] ≈ expected rtol=rtol atol=atol

                requested = _single_mode(T, lmax, l, m, expected)
                @test synthesis(cfg, requested) ≈ field rtol=rtol atol=atol
            end
        end
    end
end

@testset "Convention conversions cover dense, packed, and batch storage" begin
    lmax = 4
    cfg = create_gauss_config(lmax, 7; nlon=11, norm=:schmidt,
                              real_norm=true, cs_phase=false)
    canonical = zeros(ComplexF32, lmax + 1, lmax + 1)
    canonical[1, 1] = 2f0
    canonical[2, 1] = -3f0
    canonical[2, 2] = 1f0 + 2f0im

    external = similar(canonical)
    SHTnsKit.convert_alm_norm!(external, canonical, cfg; to_internal=false)
    @test eltype(external) === ComplexF32
    @test external[2, 2] ≈ canonical[2, 2] /
        _shtns_coefficient_scale_to_canonical(:schmidt, true, false, 1, 1, Float32)

    dense_back = similar(canonical)
    SHTnsKit.convert_alm_norm!(dense_back, external, cfg; to_internal=true)
    @test dense_back ≈ canonical

    packed = SHTnsKit.pack_lm(cfg, canonical)
    packed_external = similar(packed)
    SHTnsKit.convert_alm_norm!(packed_external, packed, cfg; to_internal=false)
    packed_back = similar(packed)
    SHTnsKit.convert_alm_norm!(packed_back, packed_external, cfg; to_internal=true)
    @test packed_back ≈ packed

    batch = cat(canonical, 2canonical; dims=3)
    batch_external = similar(batch)
    SHTnsKit.convert_alm_norm!(batch_external, batch, cfg; to_internal=false)
    batch_back = similar(batch)
    SHTnsKit.convert_alm_norm!(batch_back, batch_external, cfg; to_internal=true)
    @test batch_back ≈ batch

    @test_throws DimensionMismatch SHTnsKit.convert_alm_norm!(similar(packed, length(packed) - 1), packed, cfg)
    @test_throws DimensionMismatch SHTnsKit.convert_alm_norm!(similar(batch, lmax, lmax + 1, 2), batch, cfg)
end


@testset "Vector, mode, and complex-packed boundaries use configured coefficients" begin
    lmax = 4
    canonical_cfg = create_gauss_config(lmax, 7; nlon=11)
    norm, real_norm, cs_phase = :fourpi, true, false
    cfg = create_gauss_config(lmax, 7; nlon=11, norm, real_norm, cs_phase)

    canonical_S = zeros(ComplexF64, lmax + 1, lmax + 1)
    canonical_T = zeros(ComplexF64, lmax + 1, lmax + 1)
    canonical_S[3, 2] = 0.3 - 0.2im
    canonical_T[4, 3] = -0.1 + 0.4im
    external_S = zeros(ComplexF64, size(canonical_S))
    external_T = zeros(ComplexF64, size(canonical_T))
    for m in 0:lmax, l in m:lmax
        scale = _shtns_coefficient_scale_to_canonical(norm, real_norm, cs_phase,
                                                       l, m, Float64)
        external_S[l + 1, m + 1] = canonical_S[l + 1, m + 1] / scale
        external_T[l + 1, m + 1] = canonical_T[l + 1, m + 1] / scale
    end

    Vt_ref, Vp_ref = synthesis_sphtor(canonical_cfg, canonical_S, canonical_T)
    Vt, Vp = synthesis_sphtor(cfg, external_S, external_T)
    @test Vt ≈ Vt_ref rtol=2e-12 atol=2e-12
    @test Vp ≈ Vp_ref rtol=2e-12 atol=2e-12
    Sback, Tback = analysis_sphtor(cfg, Vt_ref, Vp_ref)
    @test Sback ≈ external_S rtol=2e-10 atol=2e-11
    @test Tback ≈ external_T rtol=2e-10 atol=2e-11

    plan = SHTPlan(cfg)
    Vtp, Vpp = zeros(cfg.nlat, cfg.nlon), zeros(cfg.nlat, cfg.nlon)
    synthesis_sphtor!(plan, Vtp, Vpp, external_S, external_T)
    @test Vtp ≈ Vt_ref rtol=2e-12 atol=2e-12
    @test Vpp ≈ Vp_ref rtol=2e-12 atol=2e-12

    # The QST wrapper must compose already-converting scalar and S/T boundaries
    # without applying a second conversion to any component.
    canonical_Q = zeros(ComplexF64, size(canonical_S)); canonical_Q[2, 1] = 0.7
    external_Q = zeros(ComplexF64, size(canonical_Q))
    for m in 0:lmax, l in m:lmax
        scale = _shtns_coefficient_scale_to_canonical(norm, real_norm, cs_phase,
                                                       l, m, Float64)
        external_Q[l + 1, m + 1] = canonical_Q[l + 1, m + 1] / scale
    end
    qref = synthesis_qst(canonical_cfg, canonical_Q, canonical_S, canonical_T)
    qgot = synthesis_qst(cfg, external_Q, external_S, external_T)
    @test all(isapprox.(qgot, qref; rtol=2e-12, atol=2e-12))

    # m=0 axisymmetric and m=2 fixed-mode paths have their own kernels.
    ql_can = ComplexF64[0.2, -0.4, 0.7, 0.1, -0.3]
    ql_ext = [ql_can[l + 1] /
              _shtns_coefficient_scale_to_canonical(norm, real_norm, cs_phase,
                                                      l, 0, Float64) for l in 0:lmax]
    axis_ref = synthesis_axisym(canonical_cfg, ql_can)
    @test synthesis_axisym(cfg, ql_ext) ≈ axis_ref rtol=2e-12 atol=2e-12
    @test analysis_axisym(cfg, axis_ref) ≈ ql_ext rtol=2e-11 atol=2e-12

    m, ltr = 2, lmax
    ml_can = ComplexF64[0.2 + 0.1im, -0.3 + 0.4im, 0.5 - 0.2im]
    ml_ext = [ml_can[l - m + 1] /
              _shtns_coefficient_scale_to_canonical(norm, real_norm, cs_phase,
                                                      l, m, Float64) for l in m:ltr]
    ml_ref = synthesis_packed_ml(canonical_cfg, m, ml_can, ltr)
    @test synthesis_packed_ml(cfg, m, ml_ext, ltr) ≈ ml_ref rtol=2e-12 atol=2e-12
    @test analysis_packed_ml(cfg, m, ml_ref, ltr) ≈ ml_ext rtol=2e-11 atol=2e-12

    # Complex packed storage carries both signs of m explicitly.
    ncomplex = nlm_cplx_calc(lmax, lmax, 1)
    ccan = zeros(ComplexF64, ncomplex)
    ccan[LM_cplx_index(lmax, lmax, 2, 1) + 1] = 0.4 - 0.3im
    ccan[LM_cplx_index(lmax, lmax, 3, -2) + 1] = -0.2 + 0.5im
    cext = similar(ccan)
    for l in 0:lmax, m in -l:l
        scale = _shtns_coefficient_scale_to_canonical(norm, real_norm, cs_phase,
                                                       l, abs(m), Float64)
        idx = LM_cplx_index(lmax, lmax, l, m) + 1
        cext[idx] = ccan[idx] / scale
    end
    zref = synthesis_packed_cplx(canonical_cfg, ccan)
    @test synthesis_packed_cplx(cfg, cext) ≈ zref rtol=2e-12 atol=2e-12
    @test analysis_packed_cplx(cfg, zref) ≈ cext rtol=2e-10 atol=2e-11
end

@testset "Convention semantics cross packed, batch, QST, and planned boundaries once" begin
    lmax = 4
    canonical_cfg = create_gauss_config(lmax, 7; nlon=11)
    norm = :schmidt
    real_norm = true
    cs_phase = false
    cfg = create_gauss_config(lmax, 7; nlon=11, norm, real_norm, cs_phase)

    canonical = zeros(ComplexF64, lmax + 1, lmax + 1)
    canonical[1, 1] = sqrt(4pi)
    canonical[2, 2] = -sqrt(2pi / 3)
    external = zeros(ComplexF64, size(canonical))
    for m in 0:lmax, l in m:lmax
        external[l + 1, m + 1] = _external_coeff(canonical[l + 1, m + 1],
                                                 norm, real_norm, cs_phase, l, m)
    end
    reference = synthesis(canonical_cfg, canonical)

    @test reshape(synthesis_packed(cfg, SHTnsKit.pack_lm(cfg, external)), cfg.nlat, cfg.nlon) ≈ reference
    @test synthesis_batch(cfg, reshape(external, lmax + 1, lmax + 1, 1))[:, :, 1] ≈ reference

    plan = SHTPlan(cfg)
    planned = zeros(cfg.nlat, cfg.nlon)
    synthesis!(plan, planned, external)
    @test planned ≈ reference
    planned_back = zeros(ComplexF64, lmax + 1, lmax + 1)
    analysis!(plan, planned_back, reference)
    @test planned_back ≈ external rtol=2e-11 atol=2e-12

    zero_st = zeros(ComplexF64, size(external))
    Vr, Vt, Vp = synthesis_qst(cfg, external, zero_st, zero_st)
    @test Vr ≈ reference
    @test iszero(Vt)
    @test iszero(Vp)
    Qback, Sback, Tback = analysis_qst(cfg, Vr, Vt, Vp)
    @test Qback ≈ external rtol=2e-10 atol=2e-11
    @test maximum(abs, Sback) < 2e-11
    @test maximum(abs, Tback) < 2e-11
end

@testset "Physical energy is invariant under configured conventions" begin
    lmax = 4
    canonical_cfg = create_gauss_config(lmax, 7; nlon=11)
    canonical = zeros(ComplexF64, lmax + 1, lmax + 1)
    canonical[1, 1] = 1.25
    canonical[3, 2] = -0.75 + 0.5im
    expected = energy_scalar(canonical_cfg, canonical)

    for norm in (:orthonormal, :fourpi, :schmidt),
        real_norm in (false, true), cs_phase in (false, true)
        cfg = create_gauss_config(lmax, 7; nlon=11, norm, real_norm, cs_phase)
        external = zeros(ComplexF64, size(canonical))
        for m in 0:lmax, l in m:lmax
            external[l + 1, m + 1] = _external_coeff(canonical[l + 1, m + 1],
                                                     norm, real_norm, cs_phase, l, m)
        end
        @test energy_scalar(cfg, external) ≈ expected rtol=2e-14 atol=2e-14
        @test energy_scalar_packed(cfg, SHTnsKit.pack_lm(cfg, external)) ≈ expected rtol=2e-14 atol=2e-14
    end
end


@testset "Energy and enstrophy spectra are convention invariant" begin
    lmax = 4
    canonical_cfg = create_gauss_config(lmax, 7; nlon=11)
    scalar_can = zeros(ComplexF64, lmax + 1, lmax + 1)
    scalar_can[1, 1] = 0.6
    scalar_can[3, 2] = -0.2 + 0.5im
    S_can = zeros(ComplexF64, size(scalar_can)); S_can[4, 2] = 0.3 - 0.1im
    T_can = zeros(ComplexF64, size(scalar_can)); T_can[3, 3] = -0.4 + 0.2im

    scalar_l_ref = energy_scalar_l_spectrum(canonical_cfg, scalar_can)
    scalar_m_ref = energy_scalar_m_spectrum(canonical_cfg, scalar_can)
    scalar_lm_ref = energy_scalar_lm(canonical_cfg, scalar_can)
    vector_l_ref = energy_vector_l_spectrum(canonical_cfg, S_can, T_can)
    vector_m_ref = energy_vector_m_spectrum(canonical_cfg, S_can, T_can)
    vector_lm_ref = energy_vector_lm(canonical_cfg, S_can, T_can)
    enstrophy_ref = enstrophy(canonical_cfg, T_can)
    enstrophy_l_ref = enstrophy_l_spectrum(canonical_cfg, T_can)
    enstrophy_m_ref = enstrophy_m_spectrum(canonical_cfg, T_can)
    enstrophy_lm_ref = enstrophy_lm(canonical_cfg, T_can)

    norm, real_norm, cs_phase = :schmidt, true, false
    cfg = create_gauss_config(lmax, 7; nlon=11, norm, real_norm, cs_phase)
    scalar_ext = zeros(ComplexF64, size(scalar_can))
    S_ext = zeros(ComplexF64, size(S_can)); T_ext = zeros(ComplexF64, size(T_can))
    for m in 0:lmax, l in m:lmax
        scale = _shtns_coefficient_scale_to_canonical(norm, real_norm, cs_phase,
                                                       l, m, Float64)
        scalar_ext[l + 1, m + 1] = scalar_can[l + 1, m + 1] / scale
        S_ext[l + 1, m + 1] = S_can[l + 1, m + 1] / scale
        T_ext[l + 1, m + 1] = T_can[l + 1, m + 1] / scale
    end

    @test energy_scalar_l_spectrum(cfg, scalar_ext) ≈ scalar_l_ref
    @test energy_scalar_m_spectrum(cfg, scalar_ext) ≈ scalar_m_ref
    @test energy_scalar_lm(cfg, scalar_ext) ≈ scalar_lm_ref
    @test energy_vector_l_spectrum(cfg, S_ext, T_ext) ≈ vector_l_ref
    @test energy_vector_m_spectrum(cfg, S_ext, T_ext) ≈ vector_m_ref
    @test energy_vector_lm(cfg, S_ext, T_ext) ≈ vector_lm_ref
    @test enstrophy(cfg, T_ext) ≈ enstrophy_ref
    @test enstrophy_l_spectrum(cfg, T_ext) ≈ enstrophy_l_ref
    @test enstrophy_m_spectrum(cfg, T_ext) ≈ enstrophy_m_ref
    @test enstrophy_lm(cfg, T_ext) ≈ enstrophy_lm_ref

    @test sum(energy_scalar_l_spectrum(cfg, scalar_ext)) ≈ energy_scalar(cfg, scalar_ext)
    @test sum(energy_vector_l_spectrum(cfg, S_ext, T_ext)) ≈ energy_vector(cfg, S_ext, T_ext)
    @test sum(enstrophy_l_spectrum(cfg, T_ext)) ≈ enstrophy(cfg, T_ext)
end


@testset "Adjoint helpers follow public convention boundaries" begin
    lmax = 3
    canonical_cfg = create_gauss_config(lmax, 6; nlon=9)
    norm, real_norm, cs_phase = :schmidt, true, false
    cfg = create_gauss_config(lmax, 6; nlon=9, norm, real_norm, cs_phase)
    grid_bar = reshape(collect(1.0:(cfg.nlat * cfg.nlon)), cfg.nlat, cfg.nlon) ./ 17

    canonical_synth_bar = SHTnsKit._adjoint_synthesis(canonical_cfg, grid_bar)
    configured_synth_bar = SHTnsKit._adjoint_synthesis(cfg, grid_bar)
    expected_synth_bar = zeros(ComplexF64, size(canonical_synth_bar))
    for m in 0:lmax, l in m:lmax
        scale = _shtns_coefficient_scale_to_canonical(norm, real_norm, cs_phase,
                                                       l, m, Float64)
        expected_synth_bar[l + 1, m + 1] = scale * canonical_synth_bar[l + 1, m + 1]
    end
    @test configured_synth_bar ≈ expected_synth_bar rtol=2e-12 atol=2e-12

    external_bar = zeros(ComplexF64, lmax + 1, lmax + 1)
    external_bar[2, 1] = 0.2
    external_bar[3, 2] = -0.3 + 0.4im
    canonical_bar = zeros(ComplexF64, size(external_bar))
    for m in 0:lmax, l in m:lmax
        scale = _shtns_coefficient_scale_to_canonical(norm, real_norm, cs_phase,
                                                       l, m, Float64)
        canonical_bar[l + 1, m + 1] = external_bar[l + 1, m + 1] / scale
    end
    @test SHTnsKit._adjoint_analysis(cfg, external_bar) ≈
          SHTnsKit._adjoint_analysis(canonical_cfg, canonical_bar) rtol=2e-12 atol=2e-12

    canonical_Sbar, canonical_Tbar =
        SHTnsKit._adjoint_synthesis_sphtor(canonical_cfg, grid_bar, reverse(grid_bar; dims=2))
    configured_Sbar, configured_Tbar =
        SHTnsKit._adjoint_synthesis_sphtor(cfg, grid_bar, reverse(grid_bar; dims=2))
    expected_Sbar = zeros(ComplexF64, size(canonical_Sbar))
    expected_Tbar = zeros(ComplexF64, size(canonical_Tbar))
    for m in 0:lmax, l in m:lmax
        scale = _shtns_coefficient_scale_to_canonical(norm, real_norm, cs_phase,
                                                       l, m, Float64)
        expected_Sbar[l + 1, m + 1] = scale * canonical_Sbar[l + 1, m + 1]
        expected_Tbar[l + 1, m + 1] = scale * canonical_Tbar[l + 1, m + 1]
    end
    @test configured_Sbar ≈ expected_Sbar rtol=2e-12 atol=2e-12
    @test configured_Tbar ≈ expected_Tbar rtol=2e-12 atol=2e-12

    got_vtbar, got_vpbar = SHTnsKit._adjoint_analysis_sphtor(cfg, external_bar, external_bar)
    ref_vtbar, ref_vpbar = SHTnsKit._adjoint_analysis_sphtor(canonical_cfg, canonical_bar, canonical_bar)
    @test got_vtbar ≈ ref_vtbar rtol=2e-12 atol=2e-12
    @test got_vpbar ≈ ref_vpbar rtol=2e-12 atol=2e-12
end
