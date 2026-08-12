using Test
using SHTnsKit
using FFTW

isdefined(@__MODULE__, :_vector_config) || include("sphtor_full.jl")

"""Independent fixed-order S/T sum using the public stored-order convention."""
function _direct_sphtor_mode(cfg::SHTConfig, im::Int, Sl, Tl, ltr::Int)
    m = im * cfg.mres
    CT = promote_type(eltype(Sl), eltype(Tl))
    length(Sl) == ltr - m + 1 || throw(DimensionMismatch("Sl oracle length"))
    length(Tl) == length(Sl) || throw(DimensionMismatch("Tl oracle length"))
    Vt = zeros(CT, cfg.nlat)
    Vp = zeros(CT, cfg.nlat)
    P = zeros(Float64, ltr + 1)
    dP = similar(P)
    Pover = similar(P)
    scratch = zeros(Float64, ltr + 2)
    for i in 1:cfg.nlat
        SHTnsKit.Plm_norm_dPdtheta_over_sinth_row!(
            P, dP, Pover, cfg.x[i], ltr, m, scratch,
        )
        for l in max(1, m):ltr
            scale = SHTnsKit.coefficient_scale_to_canonical(cfg, l, m)
            S = scale * Sl[l - m + 1]
            T = scale * Tl[l - m + 1]
            term = im * 0 # keep the stored index visibly distinct from physical m
            coupling = complex(0.0, m * Pover[l + 1])
            Vt[i] += dP[l + 1] * S - coupling * T
            Vp[i] += coupling * S + dP[l + 1] * T
        end
        if cfg.robert_form
            sin_theta = sqrt(max(0.0, 1 - cfg.x[i]^2))
            Vt[i] *= sin_theta
            Vp[i] *= sin_theta
        end
    end
    scale_phi = SHTnsKit.phi_inv_scale(cfg)
    return scale_phi .* Vt, scale_phi .* Vp
end

function _dense_from_mode(cfg::SHTConfig, im::Int, values, ltr::Int)
    m = im * cfg.mres
    dense = zeros(eltype(values), cfg.lmax + 1, cfg.mmax + 1)
    for l in m:ltr
        dense[l + 1, m + 1] = values[l - m + 1]
    end
    return dense
end

function _variant_cfg(::Type{T}; mres=1, robert_form=false,
                      norm=:orthonormal, real_norm=false,
                      cs_phase=true, grid=:gauss) where {T}
    lmax = 6
    nlat = 10
    kwargs = (; nlon=18, mmax=6, mres, robert_form,
              norm, real_norm, cs_phase)
    grid === :gauss && return create_gauss_config(lmax, nlat; kwargs...)
    grid === :regular && return create_regular_config(lmax, nlat; kwargs...)
    return create_regular_config(lmax, nlat; kwargs..., include_poles=true)
end

function test_cpu_vector_variant_reds()
    @testset "stored-order vector/QST `_ml` semantics" begin
        for T in (Float32, Float64), im in (0, 1, 2)
            cfg = _variant_cfg(T; mres=2, norm=:schmidt,
                               real_norm=true, cs_phase=false)
            m = im * cfg.mres
            ltr = cfg.lmax
            CT = Complex{T}
            Sl = [CT(T(0.04l), T(-0.01l)) for l in m:ltr]
            Tl = [CT(T(-0.025l), T(0.015l)) for l in m:ltr]
            got = synthesis_sphtor_ml(cfg, im, Sl, Tl, ltr)
            expected = _direct_sphtor_mode(cfg, im, Sl, Tl, ltr)
            tol = T === Float32 ? 2f-5 : 2e-12
            @test got[1] ≈ expected[1] atol=tol rtol=tol
            @test got[2] ≈ expected[2] atol=tol rtol=tol
            @test synthesis_grad_ml(cfg, im, Sl, ltr) ==
                  synthesis_sph_ml(cfg, im, Sl, ltr)

            denseS = _dense_from_mode(cfg, im, Sl, ltr)
            denseT = _dense_from_mode(cfg, im, Tl, ltr)
            full = synthesis_sphtor_l(cfg, denseS, denseT, ltr;
                                      real_output=false)
            fourier = map(full) do component
                fft(component, 2)[:, m + 1]
            end
            @test got[1] ≈ fourier[1] atol=tol rtol=tol
            @test got[2] ≈ fourier[2] atol=tol rtol=tol

            Ql = reverse(Sl)
            qst = synthesis_qst_ml(cfg, im, Ql, Sl, Tl, ltr)
            @test qst[1] ≈ synthesis_packed_ml(cfg, im, Ql, ltr) atol=tol rtol=tol
            @test qst[2] ≈ got[1] atol=tol rtol=tol
            @test qst[3] ≈ got[2] atol=tol rtol=tol
        end
    end

    @testset "variant validation happens before work" begin
        cfg = _variant_cfg(Float64; mres=2)
        field = zeros(ComplexF64, cfg.nlat)
        coefficients = zeros(ComplexF64, cfg.lmax + 1)
        for bad_ltr in (-1, cfg.lmax + 1, big(typemax(Int)) + 1)
            @test_throws ArgumentError synthesis_sphtor_l(
                cfg, zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1),
                zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1), bad_ltr,
            )
            @test_throws ArgumentError analysis_sphtor_ml(
                cfg, 0, field, field, bad_ltr,
            )
        end
        @test_throws ArgumentError synthesis_sphtor_ml(cfg, -1, coefficients,
                                                       coefficients, cfg.lmax)
        @test_throws ArgumentError synthesis_sphtor_ml(cfg, 4, coefficients,
                                                       coefficients, cfg.lmax)
    end

    @testset "degree truncation, gradient, and unsupported-order noise" begin
        for T in (Float32, Float64), ltr in (0, 3, 6)
            cfg = _variant_cfg(T; mres=2, robert_form=true,
                               grid=:regular_poles)
            CT = Complex{T}
            S = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
            Tlm = zero(S)
            S[2, 1] = CT(0.2, 0)
            S[5, 5] = CT(0.07, -0.03)
            Tlm[4, 3] = CT(-0.04, 0.02)
            # Invalid stored order and degree-above-cap noise must be ignored.
            S[4, 2] = CT(9, -7)
            if ltr < cfg.lmax
                Tlm[ltr + 2, 1] = CT(-8, 6)
            end
            Sref = copy(S); Tref = copy(Tlm)
            for m in 0:cfg.mmax, l in 0:cfg.lmax
                if l > ltr || l < m || m % cfg.mres != 0
                    Sref[l + 1, m + 1] = 0
                    Tref[l + 1, m + 1] = 0
                end
            end
            got = synthesis_sphtor_l(cfg, S, Tlm, ltr)
            ref = synthesis_sphtor(cfg, Sref, Tref)
            tol = T === Float32 ? 4f-5 : 4e-12
            @test got[1] ≈ ref[1] atol=tol rtol=tol
            @test got[2] ≈ ref[2] atol=tol rtol=tol
            @test synthesis_grad_l(cfg, S, ltr) == synthesis_sph_l(cfg, S, ltr)

            Q = copy(S)
            qgot = synthesis_qst_l(cfg, Q, S, Tlm, ltr)
            Qref = copy(Q)
            for m in 0:cfg.mmax, l in 0:cfg.lmax
                if l > ltr || l < m || m % cfg.mres != 0
                    Qref[l + 1, m + 1] = 0
                end
            end
            qref = (synthesis(cfg, Qref), ref...)
            for k in 1:3
                @test qgot[k] ≈ qref[k] atol=tol rtol=tol
            end
        end
    end

    @testset "vector/QST batches and typed CPU routing" begin
        cfg = _variant_cfg(Float32)
        ltr = 4
        Sone = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
        Tone = zero(Sone); Qone = zero(Sone)
        Sone[3, 2] = 0.02f0 - 0.01f0im
        Tone[4, 3] = -0.015f0 + 0.005f0im
        Qone[2, 1] = 0.03f0
        @test synthesis_sphtor_l_cplx(CPU(), cfg, Sone, Tone, ltr) ==
              synthesis_sphtor_l_cplx(cfg, Sone, Tone, ltr)
        @test synthesis_sph_l(CPU(), cfg, Sone, ltr) == synthesis_sph_l(cfg, Sone, ltr)
        @test synthesis_sph_l_cplx(CPU(), cfg, Sone, ltr) ==
              synthesis_sph_l_cplx(cfg, Sone, ltr)
        @test synthesis_tor_l(CPU(), cfg, Tone, ltr) == synthesis_tor_l(cfg, Tone, ltr)
        @test synthesis_tor_l_cplx(CPU(), cfg, Tone, ltr) ==
              synthesis_tor_l_cplx(cfg, Tone, ltr)
        @test synthesis_grad(CPU(), cfg, Sone) == synthesis_sph(cfg, Sone)
        @test synthesis_grad_l(CPU(), cfg, Sone, ltr) == synthesis_sph_l(cfg, Sone, ltr)
        mode = ComplexF32[0.02 - 0.01im for _ in 1:ltr]
        @test synthesis_sph_ml(CPU(), cfg, 1, mode, ltr) ==
              synthesis_sph_ml(cfg, 1, mode, ltr)
        @test synthesis_tor_ml(CPU(), cfg, 1, mode, ltr) ==
              synthesis_tor_ml(cfg, 1, mode, ltr)
        @test synthesis_grad_ml(CPU(), cfg, 1, mode, ltr) ==
              synthesis_sph_ml(cfg, 1, mode, ltr)
        @test synthesis_qst_l_cplx(CPU(), cfg, Qone, Sone, Tone, ltr) ==
              synthesis_qst_l_cplx(cfg, Qone, Sone, Tone, ltr)
        for nfields in (1, 2, 5)
            S = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1, nfields)
            Tlm = zero(S); Q = zero(S)
            for k in 1:nfields
                Q[2, 1, k] = 0.03f0k
                S[3, 2, k] = ComplexF32(0.02f0k, -0.01f0)
                Tlm[4, 3, k] = ComplexF32(-0.015f0k, 0.005f0)
            end
            vector = synthesis_sphtor_batch(CPU(), cfg, S, Tlm)
            qst = synthesis_qst_batch(CPU(), cfg, Q, S, Tlm)
            @test size(vector[1]) == (cfg.nlat, cfg.nlon, nfields)
            @test size(qst[1]) == (cfg.nlat, cfg.nlon, nfields)
            for k in 1:nfields
                @test vector[1][:, :, k] ≈ synthesis_sphtor(
                    cfg, S[:, :, k], Tlm[:, :, k],
                )[1] atol=3f-5 rtol=3f-5
                expected = synthesis_qst(cfg, Q[:, :, k], S[:, :, k], Tlm[:, :, k])
                @test qst[1][:, :, k] ≈ expected[1] atol=3f-5 rtol=3f-5
            end
            @test analysis_sphtor_batch(CPU(), cfg, vector...) isa Tuple
            @test analysis_qst_batch(CPU(), cfg, qst...) isa Tuple
            @test synthesis_sphtor_batch_cplx(CPU(), cfg, S, Tlm) ==
                  synthesis_sphtor_batch_cplx(cfg, S, Tlm)
            @test synthesis_qst_batch_cplx(CPU(), cfg, Q, S, Tlm) ==
                  synthesis_qst_batch_cplx(cfg, Q, S, Tlm)
        end
        empty_spatial = zeros(Float32, cfg.nlat, cfg.nlon, 0)
        empty_spectral = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1, 0)
        @test_throws ArgumentError analysis_sphtor_batch(CPU(), cfg,
                                                         empty_spatial, empty_spatial)
        @test_throws ArgumentError synthesis_sphtor_batch(CPU(), cfg,
                                                          empty_spectral, empty_spectral)
        @test_throws ArgumentError analysis_qst_batch(CPU(), cfg,
                                                      empty_spatial, empty_spatial,
                                                      empty_spatial)
        @test_throws ArgumentError synthesis_qst_batch(CPU(), cfg,
                                                       empty_spectral, empty_spectral,
                                                       empty_spectral)
    end
    return nothing
end

"""Always-run extension ownership and shared-kernel contract (no GPU needed)."""
function test_gpu_vector_variant_contract(extension, matrix_real, matrix_complex,
                                          vector_complex, array3_real,
                                          array3_complex)
    @testset "GPU vector/QST variant ownership" begin
        common = extension.GPUCommon
        for name in (:vector_mode_analysis_kernel!, :vector_mode_synthesis_kernel!,
                     :vector_batch_analysis_kernel!, :vector_batch_synthesis_kernel!)
            @test isdefined(common, name)
        end
        signatures = (
            (:analysis_sphtor_l, Tuple{SHTConfig,matrix_real,matrix_real,Int}),
            (:synthesis_sphtor_l, Tuple{SHTConfig,matrix_complex,matrix_complex,Int}),
            (:analysis_sphtor_ml, Tuple{SHTConfig,Int,vector_complex,vector_complex,Int}),
            (:synthesis_sphtor_ml, Tuple{SHTConfig,Int,vector_complex,vector_complex,Int}),
            (:synthesis_grad_l, Tuple{SHTConfig,matrix_complex,Int}),
            (:synthesis_grad_ml, Tuple{SHTConfig,Int,vector_complex,Int}),
            (:analysis_qst_l, Tuple{SHTConfig,matrix_real,matrix_real,matrix_real,Int}),
            (:synthesis_qst_l, Tuple{SHTConfig,matrix_complex,matrix_complex,matrix_complex,Int}),
            (:analysis_qst_ml, Tuple{SHTConfig,Int,vector_complex,vector_complex,vector_complex,Int}),
            (:synthesis_qst_ml, Tuple{SHTConfig,Int,vector_complex,vector_complex,vector_complex,Int}),
            (:analysis_sphtor_batch, Tuple{SHTConfig,array3_real,array3_real}),
            (:synthesis_sphtor_batch, Tuple{SHTConfig,array3_complex,array3_complex}),
            (:analysis_qst_batch, Tuple{SHTConfig,array3_real,array3_real,array3_real}),
            (:synthesis_qst_batch, Tuple{SHTConfig,array3_complex,array3_complex,array3_complex}),
        )
        for (name, signature) in signatures
            @test hasmethod(getproperty(SHTnsKit, name), signature)
            hasmethod(getproperty(SHTnsKit, name), signature) &&
                @test which(getproperty(SHTnsKit, name), signature).module === extension
        end
        typed_signatures = (
            (:synthesis_sphtor_l_cplx,
             Tuple{SHTnsKit.GPU,SHTConfig,matrix_complex,matrix_complex,Int}),
            (:synthesis_sph_l, Tuple{SHTnsKit.GPU,SHTConfig,matrix_complex,Int}),
            (:synthesis_sph_l_cplx, Tuple{SHTnsKit.GPU,SHTConfig,matrix_complex,Int}),
            (:synthesis_tor_l, Tuple{SHTnsKit.GPU,SHTConfig,matrix_complex,Int}),
            (:synthesis_tor_l_cplx, Tuple{SHTnsKit.GPU,SHTConfig,matrix_complex,Int}),
            (:synthesis_sph_ml, Tuple{SHTnsKit.GPU,SHTConfig,Int,vector_complex,Int}),
            (:synthesis_tor_ml, Tuple{SHTnsKit.GPU,SHTConfig,Int,vector_complex,Int}),
            (:synthesis_grad, Tuple{SHTnsKit.GPU,SHTConfig,matrix_complex}),
            (:synthesis_grad_l, Tuple{SHTnsKit.GPU,SHTConfig,matrix_complex,Int}),
            (:synthesis_grad_ml, Tuple{SHTnsKit.GPU,SHTConfig,Int,vector_complex,Int}),
            (:synthesis_qst_l_cplx,
             Tuple{SHTnsKit.GPU,SHTConfig,matrix_complex,matrix_complex,matrix_complex,Int}),
            (:synthesis_sphtor_batch_cplx,
             Tuple{SHTnsKit.GPU,SHTConfig,array3_complex,array3_complex}),
            (:synthesis_qst_batch_cplx,
             Tuple{SHTnsKit.GPU,SHTConfig,array3_complex,array3_complex,array3_complex}),
        )
        for (name, signature) in typed_signatures
            @test hasmethod(getproperty(SHTnsKit, name), signature)
        end
    end
    return nothing
end
