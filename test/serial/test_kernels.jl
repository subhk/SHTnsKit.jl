# SHTnsKit.jl - Low-level kernel tests
# Focuses on pole helpers and high-level agreement between table-based and
# on-the-fly code paths. Direct kernel-level numerical equivalence is left
# to the public API tests, since the kernels are composed with additional
# wrapping in the orchestrator loops.

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

    @testset "Table vs on-the-fly: scalar synthesis matches via public API" begin
        # Two configs: one with PLM tables, one without. Result of synthesis
        # must agree to machine precision for the same alm input.
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1

        cfg_tbl = create_gauss_config(lmax, nlat; nlon=nlon)
        prepare_plm_tables!(cfg_tbl)
        cfg_otf = create_gauss_config(lmax, nlat; nlon=nlon)

        alm = randn(ComplexF64, lmax + 1, lmax + 1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m - 1)
            alm[l + 1, m + 1] = 0
        end

        f_tbl = synthesis(cfg_tbl, alm; real_output=true)
        f_otf = synthesis(cfg_otf, alm; real_output=true)
        @test isapprox(f_tbl, f_otf; rtol=1e-11, atol=1e-13)

        # Same for analysis
        alm_back_tbl = analysis(cfg_tbl, f_tbl)
        alm_back_otf = analysis(cfg_otf, f_otf)
        @test isapprox(alm_back_tbl, alm_back_otf; rtol=1e-11, atol=1e-13)
    end

    @testset "Table vs on-the-fly: sphtor agrees via public API" begin
        lmax = 6
        cfg_tbl = create_gauss_config(lmax, lmax + 4; nlon=2*lmax + 1)
        prepare_plm_tables!(cfg_tbl)
        cfg_otf = create_gauss_config(lmax, lmax + 4; nlon=2*lmax + 1)

        Slm = randn(ComplexF64, lmax + 1, lmax + 1)
        Tlm = randn(ComplexF64, lmax + 1, lmax + 1)
        Slm[:, 1] .= real.(Slm[:, 1]); Tlm[:, 1] .= real.(Tlm[:, 1])
        Slm[1, 1] = 0; Tlm[1, 1] = 0
        for m in 0:lmax, l in 0:(m - 1)
            Slm[l + 1, m + 1] = 0; Tlm[l + 1, m + 1] = 0
        end

        Vt_tbl, Vp_tbl = synthesis_sphtor(cfg_tbl, Slm, Tlm; real_output=true)
        Vt_otf, Vp_otf = synthesis_sphtor(cfg_otf, Slm, Tlm; real_output=true)
        @test isapprox(Vt_tbl, Vt_otf; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_tbl, Vp_otf; rtol=1e-10, atol=1e-12)

        S_tbl, T_tbl = analysis_sphtor(cfg_tbl, Vt_tbl, Vp_tbl)
        S_otf, T_otf = analysis_sphtor(cfg_otf, Vt_otf, Vp_otf)
        @test isapprox(S_tbl, S_otf; rtol=1e-10, atol=1e-12)
        @test isapprox(T_tbl, T_otf; rtol=1e-10, atol=1e-12)
    end
end
