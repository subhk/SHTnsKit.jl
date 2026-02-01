# JET.jl Type Stability Tests for SHTnsKit.jl
#
# This test suite uses JET.jl to detect potential type instabilities and
# runtime errors through static analysis of Julia's type inference system.

using Test
using JET
using SHTnsKit
using LinearAlgebra

# Test configuration parameters
const JET_LMAX = 4
const JET_NLAT = JET_LMAX + 2
const JET_NLON = 2 * JET_LMAX + 1

@testset "JET Type Stability Tests" begin

    @testset "Config Creation" begin
        # Test that config creation is type-stable
        @test_opt create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        @test_call create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
    end

    @testset "Scalar Transforms" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        alm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        alm[3, 2] = 1.0 + 0im

        # Test synthesis (spectral -> spatial)
        @test_opt synthesis(cfg, alm; real_output=true)
        @test_call synthesis(cfg, alm; real_output=true)

        f = synthesis(cfg, alm; real_output=true)

        # Test analysis (spatial -> spectral)
        @test_opt analysis(cfg, f)
        @test_call analysis(cfg, f)

        # Test spat_to_SH and SH_to_spat
        Vr = vec(f)
        @test_opt spat_to_SH(cfg, Vr)
        @test_call spat_to_SH(cfg, Vr)

        alm_vec = spat_to_SH(cfg, Vr)
        @test_opt SH_to_spat(cfg, alm_vec)
        @test_call SH_to_spat(cfg, alm_vec)
    end

    @testset "Vector Transforms (Sphtor)" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        Slm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Tlm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Slm[3, 2] = 1.0 + 0im
        Tlm[4, 3] = 0.5 + 0.5im

        # Test SHsphtor_to_spat (spectral -> spatial for vectors)
        @test_opt SHsphtor_to_spat(cfg, Slm, Tlm; real_output=true)
        @test_call SHsphtor_to_spat(cfg, Slm, Tlm; real_output=true)

        Vt, Vp = SHsphtor_to_spat(cfg, Slm, Tlm; real_output=true)

        # Test spat_to_SHsphtor (spatial -> spectral for vectors)
        @test_opt spat_to_SHsphtor(cfg, Vt, Vp)
        @test_call spat_to_SHsphtor(cfg, Vt, Vp)
    end

    @testset "Energy Diagnostics" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        alm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        alm[3, 2] = 1.0 + 0im
        f = synthesis(cfg, alm; real_output=true)

        # Scalar energy
        @test_opt energy_scalar(cfg, alm)
        @test_call energy_scalar(cfg, alm)

        @test_opt grid_energy_scalar(cfg, f)
        @test_call grid_energy_scalar(cfg, f)

        # Vector energy
        Slm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Tlm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Slm[3, 2] = 1.0 + 0im
        Vt, Vp = SHsphtor_to_spat(cfg, Slm, Tlm; real_output=true)

        @test_opt energy_vector(cfg, Slm, Tlm)
        @test_call energy_vector(cfg, Slm, Tlm)

        @test_opt grid_energy_vector(cfg, Vt, Vp)
        @test_call grid_energy_vector(cfg, Vt, Vp)
    end

    @testset "Indexing Functions" begin
        lmax = JET_LMAX
        mres = 1

        # Test LM_index type stability
        @test_opt LM_index(lmax, mres, 2, 1)
        @test_call LM_index(lmax, mres, 2, 1)

        # Test nlm_calc
        @test_opt nlm_calc(lmax, 1)
        @test_call nlm_calc(lmax, 1)
    end

    @testset "Gradient Functions" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        alm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        alm[3, 2] = 1.0 + 0im

        # Test gradient of scalar energy
        @test_opt grad_energy_scalar_alm(cfg, alm)
        @test_call grad_energy_scalar_alm(cfg, alm)

        # Test gradient of vector energy
        Slm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Tlm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Slm[3, 2] = 1.0 + 0im

        @test_opt grad_energy_vector_Slm_Tlm(cfg, Slm, Tlm)
        @test_call grad_energy_vector_Slm_Tlm(cfg, Slm, Tlm)
    end

    @testset "Rotations" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        alm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        alm[3, 2] = 1.0 + 0im

        # Test Z rotation
        @test_opt SH_Zrotate(cfg, alm, 0.5)
        @test_call SH_Zrotate(cfg, alm, 0.5)

        # Test Y rotation
        @test_opt SH_Yrotate(cfg, alm, 0.5)
        @test_call SH_Yrotate(cfg, alm, 0.5)
    end

    @testset "Operators" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)

        # Test operator matrix creation
        @test_opt mul_ct_matrix(cfg)
        @test_call mul_ct_matrix(cfg)

        @test_opt st_dt_matrix(cfg)
        @test_call st_dt_matrix(cfg)
    end

    @testset "Spectrum Functions" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        alm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        alm[3, 2] = 1.0 + 0im

        # Test l-spectrum
        @test_opt energy_scalar_l_spectrum(cfg, alm)
        @test_call energy_scalar_l_spectrum(cfg, alm)

        # Test m-spectrum
        @test_opt energy_scalar_m_spectrum(cfg, alm)
        @test_call energy_scalar_m_spectrum(cfg, alm)
    end

    @testset "Enstrophy Functions" begin
        cfg = create_gauss_config(JET_LMAX, JET_NLAT; nlon=JET_NLON)
        Tlm = zeros(ComplexF64, JET_LMAX + 1, JET_LMAX + 1)
        Tlm[4, 3] = 1.0 + 0im

        @test_opt enstrophy(cfg, Tlm)
        @test_call enstrophy(cfg, Tlm)

        @test_opt grad_enstrophy_Tlm(cfg, Tlm)
        @test_call grad_enstrophy_Tlm(cfg, Tlm)
    end

end
