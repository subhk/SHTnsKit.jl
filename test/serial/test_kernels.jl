# SHTnsKit.jl - Low-level kernel tests
# Exercises src/kernels.jl directly: scalar/sphtor kernels, pole helpers,
# table vs on-the-fly equivalence at the kernel layer.

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Kernels (legendre accumulation)" begin
    @testset "Pole helpers: m=0 → 0" begin
        for l in 0:5
            @test SHTnsKit._dPdtheta_at_pole(l, 0, 1.0, 1.0) == 0.0
            @test SHTnsKit._dPdtheta_at_pole(l, 0, -1.0, 1.0) == 0.0
            @test SHTnsKit._P_over_sinth_at_pole(l, 0, 1.0, 1.0) == 0.0
            @test SHTnsKit._P_over_sinth_at_pole(l, 0, -1.0, 1.0) == 0.0
        end
    end

    @testset "Pole helpers: m≥2 → 0" begin
        for l in 2:6, m in 2:l
            @test SHTnsKit._dPdtheta_at_pole(l, m, 1.0, 0.7) == 0.0
            @test SHTnsKit._P_over_sinth_at_pole(l, m, -1.0, 0.7) == 0.0
        end
    end

    @testset "Pole helpers: m=1 analytical limit" begin
        # m=1: dP/dθ|_north = -N * l(l+1)/2
        for l in 1:5
            N = 1.5
            d_north = SHTnsKit._dPdtheta_at_pole(l, 1, 1.0, N)
            @test d_north ≈ -N * l * (l + 1) / 2
            d_south = SHTnsKit._dPdtheta_at_pole(l, 1, -1.0, N)
            @test d_south ≈ N * (-1.0)^(l+1) * l * (l + 1) / 2

            Y_north = SHTnsKit._P_over_sinth_at_pole(l, 1, 1.0, N)
            @test Y_north ≈ -N * l * (l + 1) / 2
            Y_south = SHTnsKit._P_over_sinth_at_pole(l, 1, -1.0, N)
            @test Y_south ≈ N * (-1.0)^l * l * (l + 1) / 2
        end
    end

    @testset "Scalar synthesis kernel: table vs on-the-fly" begin
        lmax = 6
        nlat, nlon = lmax + 2, 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        prepare_plm_tables!(cfg)
        @test cfg.use_plm_tables

        alm = randn(ComplexF64, lmax + 1, lmax + 1)
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        P = Vector{Float64}(undef, lmax + 1)
        for m in 0:cfg.mmax, i in 1:cfg.nlat
            col = m + 1
            tbl = cfg.plm_tables[col]
            val_tbl = SHTnsKit._scalar_synthesis_kernel(cfg, alm, tbl, i, col, m, lmax)
            val_otf = SHTnsKit._scalar_synthesis_kernel_otf(cfg, alm, P, i, col, m, lmax)
            @test isapprox(val_tbl, val_otf; rtol=1e-12, atol=1e-14)
        end
    end

    @testset "Scalar analysis kernel: table vs on-the-fly" begin
        lmax = 5
        nlat, nlon = lmax + 2, 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        prepare_plm_tables!(cfg)

        # Fake Fourier-space field Fph[nlat, mmax+1]
        Fph = randn(ComplexF64, nlat, lmax + 1)
        scale_phi = cfg.cphi

        alm_tbl = zeros(ComplexF64, lmax + 1, lmax + 1)
        alm_otf = zeros(ComplexF64, lmax + 1, lmax + 1)
        P = Vector{Float64}(undef, lmax + 1)

        for m in 0:cfg.mmax
            col = m + 1
            tbl = cfg.plm_tables[col]
            for i in 1:nlat
                SHTnsKit._scalar_analysis_kernel!(alm_tbl, cfg, Fph, tbl, i, col, m, lmax, scale_phi)
                SHTnsKit._scalar_analysis_kernel_otf!(alm_otf, cfg, Fph, P, i, col, m, lmax, scale_phi)
            end
        end

        @test isapprox(alm_tbl, alm_otf; rtol=1e-12, atol=1e-14)
    end

    @testset "Sphtor synthesis kernel: table vs on-the-fly (off-pole)" begin
        lmax = 5
        nlat, nlon = lmax + 4, 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        prepare_plm_tables!(cfg)

        Slm = randn(ComplexF64, lmax + 1, lmax + 1)
        Tlm = randn(ComplexF64, lmax + 1, lmax + 1)
        for m in 0:lmax, l in 0:(m-1)
            Slm[l+1, m+1] = 0; Tlm[l+1, m+1] = 0
        end

        P = Vector{Float64}(undef, lmax + 1)
        dPdtheta = Vector{Float64}(undef, lmax + 1)
        P_over_sinth = Vector{Float64}(undef, lmax + 1)

        # Gauss nodes are strictly inside (−1, 1), so no pole case is hit here
        for m in 0:cfg.mmax, i in 1:cfg.nlat
            col = m + 1
            tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]

            gt_tbl, gp_tbl = SHTnsKit._sphtor_synthesis_kernel(
                cfg, Slm, Tlm, tblP, tbld, i, col, m, lmax)
            gt_otf, gp_otf = SHTnsKit._sphtor_synthesis_kernel_otf(
                cfg, Slm, Tlm, P, dPdtheta, P_over_sinth, i, col, m, lmax)

            @test isapprox(gt_tbl, gt_otf; rtol=1e-11, atol=1e-13)
            @test isapprox(gp_tbl, gp_otf; rtol=1e-11, atol=1e-13)
        end
    end

    @testset "Sphtor analysis kernel: table vs on-the-fly" begin
        lmax = 4
        nlat, nlon = lmax + 4, 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        prepare_plm_tables!(cfg)

        Ftheta = randn(ComplexF64, nlat)
        Fphi   = randn(ComplexF64, nlat)
        scale_phi = cfg.cphi

        P = Vector{Float64}(undef, lmax + 1)
        dPdtheta = Vector{Float64}(undef, lmax + 1)
        P_over_sinth = Vector{Float64}(undef, lmax + 1)

        for m in 0:cfg.mmax
            col = m + 1
            tblP = cfg.plm_tables[col]; tbld = cfg.dplm_tables[col]

            Sacc_tbl = zeros(ComplexF64, lmax + 1)
            Tacc_tbl = zeros(ComplexF64, lmax + 1)
            Sacc_otf = zeros(ComplexF64, lmax + 1)
            Tacc_otf = zeros(ComplexF64, lmax + 1)

            for i in 1:nlat
                wi = cfg.w[i]
                SHTnsKit._sphtor_analysis_kernel!(
                    Sacc_tbl, Tacc_tbl, cfg, Ftheta[i], Fphi[i], wi,
                    tblP, tbld, i, col, m, lmax, scale_phi)
                SHTnsKit._sphtor_analysis_kernel_otf!(
                    Sacc_otf, Tacc_otf, cfg, Ftheta[i], Fphi[i], wi,
                    P, dPdtheta, P_over_sinth, i, col, m, lmax, scale_phi)
            end

            @test isapprox(Sacc_tbl, Sacc_otf; rtol=1e-11, atol=1e-13)
            @test isapprox(Tacc_tbl, Tacc_otf; rtol=1e-11, atol=1e-13)
        end
    end

    @testset "Scalar synthesis kernel: l<m early return" begin
        lmax = 3
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        prepare_plm_tables!(cfg)
        alm = zeros(ComplexF64, lmax + 1, lmax + 1)
        # m=2, but entries below l=2 must be ignored
        m = 2; col = m + 1
        tbl = cfg.plm_tables[col]
        val = SHTnsKit._scalar_synthesis_kernel(cfg, alm, tbl, 1, col, m, lmax)
        @test val == zero(ComplexF64)
    end
end
