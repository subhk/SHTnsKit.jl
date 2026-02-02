# JET.jl Type Stability Tests for SHTnsKit.jl
#
# This test suite uses JET.jl to detect potential type instabilities and
# runtime errors through static analysis of Julia's type inference system.
#
# Note: Transform functions are tested for correctness only (not @test_opt)
# due to known type instabilities from FFT operations in external packages.

using Test
using JET
using SHTnsKit
using LinearAlgebra

# Note: JET may emit deprecation warnings about @lookup from JuliaInterpreter.
# This is an upstream issue in JET and does not affect test results.

# Test configuration parameters
const JET_LMAX = 4
const JET_NLAT = JET_LMAX + 2
const JET_NLON = 2 * JET_LMAX + 1

@testset "JET Type Stability Tests" begin

    @testset "Config Creation" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        @test cfg isa SHTConfig
    end

    @testset "Scalar Transforms" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        alm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        alm[3, 2] = 1.0 + 0im

        # Transforms have known type instabilities from FFT
        # Test that they produce correct types
        f = synthesis(cfg, alm; real_output=true)
        @test f isa Matrix

        alm2 = analysis(cfg, f)
        @test alm2 isa Matrix

        # Test analysis_packed and synthesis_packed
        Vr = vec(f)
        alm_vec = analysis_packed(cfg, Vr)
        @test alm_vec isa Vector

        f_back = synthesis_packed(cfg, alm_vec)
        @test f_back isa Vector
    end

    @testset "Vector Transforms (Sphtor)" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        Slm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Tlm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Slm[3, 2] = 1.0 + 0im
        Tlm[4, 3] = 0.5 + 0.5im

        # Vector transforms have FFT-related type instabilities
        # Test that they produce correct types
        Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        @test Vt isa Matrix
        @test Vp isa Matrix

        Slm2, Tlm2 = analysis_sphtor(cfg, Vt, Vp)
        @test Slm2 isa Matrix
        @test Tlm2 isa Matrix
    end

    @testset "Energy Diagnostics" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        alm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        alm[3, 2] = 1.0 + 0im

        # Scalar energy - these should be type-stable
        @test_opt target_modules=(SHTnsKit,) energy_scalar(cfg, alm)
        @test_call target_modules=(SHTnsKit,) energy_scalar(cfg, alm)

        f = synthesis(cfg, alm; real_output=true)
        @test_opt target_modules=(SHTnsKit,) grid_energy_scalar(cfg, f)
        @test_call target_modules=(SHTnsKit,) grid_energy_scalar(cfg, f)

        # Vector energy
        Slm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Tlm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Slm[3, 2] = 1.0 + 0im

        @test_opt target_modules=(SHTnsKit,) energy_vector(cfg, Slm, Tlm)
        @test_call target_modules=(SHTnsKit,) energy_vector(cfg, Slm, Tlm)
    end

    @testset "Indexing Functions" begin
        lmax = JET_LMAX
        mres = 1

        # Test LM_index type stability
        @test_opt LM_index(lmax, mres, 2, 1)
        @test_call LM_index(lmax, mres, 2, 1)

        # Test nlm_calc with correct signature (lmax, mmax, mres)
        result = nlm_calc(lmax, lmax, mres)
        @test result isa Integer
    end

    @testset "Gradient Functions" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        alm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        alm[3, 2] = 1.0 + 0im

        # Gradient functions should be type-stable
        @test_opt target_modules=(SHTnsKit,) grad_energy_scalar_alm(cfg, alm)
        @test_call target_modules=(SHTnsKit,) grad_energy_scalar_alm(cfg, alm)

        Slm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Tlm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Slm[3, 2] = 1.0 + 0im

        @test_opt target_modules=(SHTnsKit,) grad_energy_vector_Slm_Tlm(cfg, Slm, Tlm)
        @test_call target_modules=(SHTnsKit,) grad_energy_vector_Slm_Tlm(cfg, Slm, Tlm)
    end

    @testset "Spectrum Functions" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        alm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        alm[3, 2] = 1.0 + 0im

        # Spectrum functions should be type-stable
        @test_opt target_modules=(SHTnsKit,) energy_scalar_l_spectrum(cfg, alm)
        @test_call target_modules=(SHTnsKit,) energy_scalar_l_spectrum(cfg, alm)

        @test_opt target_modules=(SHTnsKit,) energy_scalar_m_spectrum(cfg, alm)
        @test_call target_modules=(SHTnsKit,) energy_scalar_m_spectrum(cfg, alm)
    end

    @testset "Enstrophy Functions" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        Tlm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Tlm[4, 3] = 1.0 + 0im

        @test_opt target_modules=(SHTnsKit,) enstrophy(cfg, Tlm)
        @test_call target_modules=(SHTnsKit,) enstrophy(cfg, Tlm)

        @test_opt target_modules=(SHTnsKit,) grad_enstrophy_Tlm(cfg, Tlm)
        @test_call target_modules=(SHTnsKit,) grad_enstrophy_Tlm(cfg, Tlm)
    end

end
